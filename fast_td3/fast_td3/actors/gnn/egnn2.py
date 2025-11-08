from torch import nn
import torch
from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egcl import E_GCL
from fast_td3.actors.gnn.env_config import env_with_object


class EGNN(nn.Module):
    def __init__(
        self,
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
        coords_agg="mean",
    ):
        """
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

        super(EGNN, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf
        self.batch_size = batch_size
        self.has_mixed_node_types = env_name in env_with_object
        self.robot = robot
        self.env_name = env_name
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = len(self.graph_builder.robot.JOINT)
        self.num_edges = len(self.graph_builder.robot.joint_connections)

        # EGNN layers for local joint-to-joint processing
        self.layers = nn.ModuleList(
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
                    edge_coords_nf=1,
                )
                for _ in range(n_layers)
            ]
        )

        self.global_layer = E_GCL(
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
            edge_coords_nf=1,
        )

        # Embedding layers for initial node features
        self.joint_embedding_in = nn.Sequential(nn.LazyLinear(self.hidden_nf), act_fn)
        self.root_embedding_in = nn.Sequential(nn.LazyLinear(self.hidden_nf), act_fn)
        self.embedding_out = nn.Sequential(nn.LazyLinear(out_node_nf), nn.Tanh())

        self._edges_cache = {}
        self._root_edges_cache = {}
        
        self.to(self.device)

    def forward(self, obs: torch.Tensor, xpos: torch.Tensor) -> torch.Tensor:
        current_batch_size = obs.shape[0]
        edges = self.get_cached_edges(current_batch_size)
        root_edges = self.get_cached_root_edges(current_batch_size)

        if self.has_mixed_node_types:
            return self.process_mixed_types(obs, xpos, edges, current_batch_size)
        else:
            return self.process_single_type(
                obs, xpos, edges, root_edges, current_batch_size
            )

    def process_mixed_types(
        self,
        obs: torch.Tensor,
        xpos: torch.Tensor,
        edges: torch.Tensor,
        current_batch_size: int,
    ) -> torch.Tensor:
        """Process environments with objects using separate clusters for joints and objects."""
        raise NotImplementedError("Mixed type processing is currently disabled.")

        # h_joint, h_object, x_joint, _ = (
        #     self.graph_builder.generate_input_for_mixed_type(obs, xpos)
        # )

        # h_joints = self.joint_embedding_in(h_joint)
        # for layer in self.layers:
        #     h_joints, x_joint, _ = layer(h=h_joints, edge_index=edges, coord=x_joint)
        # h_joints_batched = (
        #     h_joints.view(current_batch_size, self.num_joints, self.hidden_nf) * 5
        # )

        # h_object_processed = self.object_mlp(h_object)
        # h_object_broadcasted = h_object_processed.unsqueeze(1).expand(
        #     -1, self.num_joints, -1
        # )

        # h_concat = torch.cat([h_joints_batched, h_object_broadcasted], dim=-1)
        # h_global = self.global_aggregation(h_concat)

        # actions = torch.tanh(h_global)
        # return actions.view(current_batch_size, self.num_joints)

    def process_single_type(
        self,
        obs: torch.Tensor,
        xpos: torch.Tensor,
        edges: torch.Tensor,
        root_edges: torch.Tensor,
        current_batch_size: int,
    ) -> torch.Tensor:
        """Process environments without objects using joints and root context."""
        h_joints, x_joint, h_root, x_root = self.graph_builder.generate_input2(
            obs, xpos
        )

        h_joints = self.joint_embedding_in(h_joints)
        h_root = self.root_embedding_in(h_root)

        for layer in self.layers:
            h_joints, x_joint, _ = layer(h=h_joints, edge_index=edges, coord=x_joint)

        # Combine joints and root nodes into a single graph
        h_combined = torch.cat([h_joints, h_root], dim=0)
        x_combined = torch.cat([x_joint, x_root], dim=0)
        h_combined, _, _ = self.global_layer(
            h=h_combined, edge_index=root_edges, coord=x_combined
        )

        actions = self.embedding_out(h_combined)[:current_batch_size*self.num_joints].view(current_batch_size, self.num_joints)

        return actions

    def generate_index(self, batch_size: int, device="cuda"):
        """
        Generate joint-to-joint edge indices for given batch size.
        Since we now process objects separately with MLP, we only need joint edges for EGNN.
        """
        # Always use joint-to-joint connections only
        src, dst = zip(*self.graph_builder.robot.joint_connections)

        # Convert to tensors
        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

        return torch.stack([src_batch, dst_batch])

    def generate_root_index(self, batch_size: int, device="cuda"):
        src = [i for i in range(0, self.num_joints)]

        # Convert to tensors
        src = torch.tensor(src, dtype=torch.long, device=device)

        # Create batch offsets for src (joints) and dst (root nodes)
        joint_offsets = torch.arange(batch_size, device=device) * self.num_joints
        root_offsets = torch.arange(batch_size, device=device) + (
            batch_size * self.num_joints
        )

        # Expand src with joint offsets: [0-18, 19-37, ...]
        src_batch = (src.unsqueeze(0) + joint_offsets.unsqueeze(1)).flatten()

        # Expand dst with root indices: [38]*19, [39]*19, ...
        dst_batch = root_offsets.unsqueeze(1).expand(-1, self.num_joints).flatten()

        return torch.stack([src_batch, dst_batch])

    def get_cached_edges(self, current_batch_size: int):
        """
        Optimized method to get edge indices with dynamic caching.
        Automatically caches new batch sizes as they're encountered.
        """
        # Check if already cached
        if current_batch_size in self._edges_cache:
            return self._edges_cache[current_batch_size]

        # Generate, cache, and return
        edges = self.generate_index(current_batch_size, self.device)
        self._edges_cache[current_batch_size] = edges
        return edges

    def get_cached_root_edges(self, current_batch_size: int):
        """
        Optimized method to get root edge indices with dynamic caching.
        Automatically caches new batch sizes as they're encountered.
        """
        # Check if already cached
        if current_batch_size in self._root_edges_cache:
            return self._root_edges_cache[current_batch_size]

        # Generate, cache, and return
        root_edges = self.generate_root_index(current_batch_size, self.device)
        self._root_edges_cache[current_batch_size] = root_edges
        return root_edges
