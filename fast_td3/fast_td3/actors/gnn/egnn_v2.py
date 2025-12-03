from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egnn import E_GCL, env_with_object


class EGNN_V2(nn.Module):
    """
    EGNN v2 with improved object handling for mixed-type environments.
    
    For environments with objects (like h1-balance_simple-v0):
    - Uses E_GCL layers for message passing within the joint graph
    - Uses MLP to process object features
    - Broadcasts object features and concatenates with joint features
    - Uses global aggregation MLP to produce final actions
    
    For environments without objects:
    - Uses E_GCL layers for message passing within the joint graph
    - Uses root context as global information
    """
    def __init__(
        self,
        in_joint_nf,
        in_object_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf,
        device,
        batch_size,
        act_fn,
        n_layers,
        robot,
        env_name,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        coords_agg="mean"
    ):
        """
        :param in_joint_nf: Number of features for joint nodes (velocity + position)
        :param in_object_nf: Number of features for object nodes
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        """

        super(EGNN_V2, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf
        self.batch_size = batch_size
        self.has_mixed_node_types = env_name in env_with_object
        self.env_name = env_name
        self.robot = robot
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        self.num_edges = self.graph_builder.robot.num_edges
        self._joint_edges_cache = {}
        
        # Joint graph layers (message passing within joints)
        self.joint_layers = nn.ModuleList(
            [
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                    coords_agg=coords_agg,
                )
                for _ in range(n_layers)
            ]
        )
        
        # Input embedding for joints
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_joint_nf, self.hidden_nf), act_fn
        )
        
        # Object MLP for processing object features (uses LazyLinear for flexibility)
        self.object_mlp = nn.Sequential(nn.LazyLinear(self.hidden_nf), act_fn)
        
        # Global aggregation MLP: combines joint and object features
        # Input: [hidden_nf * 2] (joint features + object features)
        # Output: out_node_nf (action per joint)
        self.global_aggregation = nn.Sequential(
            nn.Linear(self.hidden_nf * 2, self.hidden_nf * 4),
            act_fn,
            nn.Linear(self.hidden_nf * 4, self.hidden_nf * 2),
            act_fn,
            nn.Linear(self.hidden_nf * 2, self.out_node_nf),
        )
        
        self.to(self.device)

    def forward(self, obs: torch.Tensor, xanchor: torch.Tensor) -> torch.Tensor:
        current_batch_size = obs.shape[0]
        joint_edges = self.get_cached_joint_edges(current_batch_size)
        
        if self.has_mixed_node_types:
            return self.process_mixed_types(obs, xanchor, joint_edges, current_batch_size)
        else:
            return self.process_single_type(obs, xanchor, joint_edges, current_batch_size)

    def process_mixed_types(
        self,
        obs: torch.Tensor,
        xanchor: torch.Tensor,
        edges: torch.Tensor,
        current_batch_size: int,
    ) -> torch.Tensor:
        """Process environments with objects using separate clusters for joints and objects."""
        h_joint, h_object, x_joint, _ = (
            self.graph_builder.generate_input_for_mixed_type(obs, xanchor)
        )

        # Embed and process joint features through EGNN layers
        h_joints = self.joint_embedding_in(h_joint)
        for layer in self.joint_layers:
            h_joints, x_joint, _ = layer(h=h_joints, edge_index=edges, coord=x_joint)
        
        # Reshape to [batch, num_joints, hidden]
        h_joints_batched = (
            h_joints.view(current_batch_size, self.num_joints, self.hidden_nf) * 2
        )

        # Process object features with MLP
        h_object_processed = self.object_mlp(h_object)
        
        # Broadcast object features to match joint dimensions [batch, num_joints, hidden]
        h_object_broadcasted = h_object_processed.unsqueeze(1).expand(
            -1, self.num_joints, -1
        )

        # Concatenate joint and object features
        h_concat = torch.cat([h_joints_batched, h_object_broadcasted], dim=-1)
        
        # Global aggregation to produce actions
        h_global = self.global_aggregation(h_concat)

        actions = torch.tanh(h_global)
        return actions.view(current_batch_size, self.num_joints)

    def process_single_type(
        self,
        obs: torch.Tensor,
        xanchor: torch.Tensor,
        edges: torch.Tensor,
        current_batch_size: int,
    ) -> torch.Tensor:
        """Process environments without objects using joints and root context."""
        h_joints, x_joint, h_root, _ = self.graph_builder.generate_input(obs, xanchor)

        # Embed and process joint features through EGNN layers
        h_joints = self.joint_embedding_in(h_joints)
        for layer in self.joint_layers:
            h_joints, x_joint, _ = layer(h=h_joints, edge_index=edges, coord=x_joint)
        
        # Reshape to [batch, num_joints, hidden]
        h_joints_batched = (
            h_joints.view(current_batch_size, self.num_joints, self.hidden_nf) * 2
        )

        # Process root features with MLP
        h_root = self.object_mlp(h_root)
        
        # Broadcast root features to match joint dimensions [batch, num_joints, hidden]
        h_root_broadcasted = h_root.unsqueeze(1).expand(-1, self.num_joints, -1)

        # Concatenate joint and root features
        h_concat = torch.cat([h_joints_batched, h_root_broadcasted], dim=-1)
        
        # Global aggregation to produce actions
        h_global = self.global_aggregation(h_concat)

        actions = torch.tanh(h_global)
        return actions.view(current_batch_size, self.num_joints)

    def generate_joint_edges(self, batch_size: int, device: torch.device):
        src, dst = zip(*self.graph_builder.robot.joint_connections)

        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

        return torch.stack([src_batch, dst_batch])

    def get_cached_joint_edges(self, current_batch_size: int):
        """Get cached edge indices for the joint graph."""
        if current_batch_size in self._joint_edges_cache:
            return self._joint_edges_cache[current_batch_size]
        
        edges = self.generate_joint_edges(current_batch_size, self.device)
        self._joint_edges_cache[current_batch_size] = edges
        return edges
