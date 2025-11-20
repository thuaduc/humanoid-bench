from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egnn import E_GCL, env_with_object


class EGNN_V2(nn.Module):
    """
    EGNN v2 with cross-graph message passing from objects to joints.
    
    Uses E_GCL for both:
    1. Message passing within the joint graph
    2. Unidirectional message passing from objects to joints
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
        :param batch_size: Batch size
        :param act_fn: Non-linearity
        :param n_layers: Number of layers for message passing
        :param robot: Robot type ('h1' or 'g1')
        :param env_name: Environment name
        :param residual: Use residual connections
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinate messages
        :param tanh: Sets a tanh activation at the output of phi_x
        :param coords_agg: Coordinate aggregation method ('sum' or 'mean')
        """
        super(EGNN_V2, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf
        self.batch_size = batch_size
        self.has_mixed_node_types = env_name in env_with_object
        self.robot = robot
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        self.num_edges = self.graph_builder.robot.num_edges
        self._joint_edges_cache = {}
        self._cross_edges_cache = {}
        
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
        
        # Cross-graph layer: object → joint (unidirectional)
        self.cross_layer = E_GCL(
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
        
        # Input embeddings
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_joint_nf, self.hidden_nf), 
            act_fn
        )
        
        self.object_embedding = nn.Sequential(
            nn.Linear(in_object_nf, self.hidden_nf), 
            act_fn
        )
        
        # Output projection
        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )
        
        self.to(self.device)
    
    def forward(self, obs: torch.Tensor, xanchor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Message passing within joint graph (L rounds)
        2. Cross-graph message passing from objects to joints (1 layer)
        3. Output actions
        """
        current_batch_size = obs.shape[0]
        
        # Get input features and coordinates using existing generate_input
        h_joints, x_joints, h_objects, x_objects = self.graph_builder.generate_input(obs, xanchor)
        
        # Embed joint and object features
        h_joints = self.joint_embedding_in(h_joints)  # [batch*num_joints, hidden]
        h_objects = self.object_embedding(h_objects)  # [batch, hidden]
        
        # Get joint graph edges
        joint_edges = self.get_cached_joint_edges(current_batch_size)
        
        # Step 1: Message passing within joint graph (L rounds)
        for layer in self.joint_layers:
            h_joints, x_joints, _ = layer(h=h_joints, edge_index=joint_edges, coord=x_joints)
        
        # Step 2: Cross-graph message passing from objects to joints
        # Combine joints and objects into a single graph for E_GCL
        # Need to expand objects to [batch*1, hidden] and [batch*1, 3]
        h_combined = torch.cat([h_joints, h_objects], dim=0)
        x_combined = torch.cat([x_joints, x_objects], dim=0)
        
        # Get cross-graph edges (object → joint, unidirectional)
        cross_edges = self.get_cached_cross_edges(current_batch_size)
        
        # Apply cross-graph E_GCL layer
        h_combined, x_combined, _ = self.cross_layer(h=h_combined, edge_index=cross_edges, coord=x_combined)
        
        # Extract updated joint features and coordinates
        h_joints = h_combined[:current_batch_size * self.num_joints]
        x_joints = x_combined[:current_batch_size * self.num_joints]
        
        # Step 3: Output projection
        actions = self.joint_embedding_out(h_joints)
        
        return actions.view(current_batch_size, self.num_joints)
    
    def generate_joint_edges(self, batch_size: int):
        """Generate edge indices for the joint graph across batches."""
        src, dst = zip(*self.graph_builder.robot.joint_connections)
        
        src = torch.tensor(src, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst, dtype=torch.long, device=self.device)
        
        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=self.device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        
        return torch.stack([src_batch, dst_batch])
    
    def generate_cross_edges(self, batch_size: int):
        """
        Generate unidirectional edge indices from objects to joints.
        
        For each batch:
        - Object index: batch*num_joints + batch_idx
        - Joint indices: batch_idx*num_joints to (batch_idx+1)*num_joints - 1
        
        Creates edges: object → all joints in the same batch
        """
        num_joints = self.num_joints
        
        # For each batch, create edges from the object to all joints
        src_list = []
        dst_list = []
        
        for batch_idx in range(batch_size):
            object_idx = batch_size * num_joints + batch_idx
            joint_start = batch_idx * num_joints
            joint_end = (batch_idx + 1) * num_joints
            
            # Create edges from object to all joints in this batch
            for joint_idx in range(joint_start, joint_end):
                src_list.append(object_idx)  # Source: object
                dst_list.append(joint_idx)    # Destination: joint
        
        src = torch.tensor(src_list, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst_list, dtype=torch.long, device=self.device)
        
        return torch.stack([src, dst])
    
    def get_cached_joint_edges(self, current_batch_size: int):
        """Get cached edge indices for the joint graph."""
        if current_batch_size in self._joint_edges_cache:
            return self._joint_edges_cache[current_batch_size]
        
        edges = self.generate_joint_edges(current_batch_size)
        self._joint_edges_cache[current_batch_size] = edges
        return edges
    
    def get_cached_cross_edges(self, current_batch_size: int):
        """Get cached edge indices for cross-graph connections."""
        if current_batch_size in self._cross_edges_cache:
            return self._cross_edges_cache[current_batch_size]
        
        edges = self.generate_cross_edges(current_batch_size)
        self._cross_edges_cache[current_batch_size] = edges
        return edges
