from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder


# Environment classification for object inclusion
env_with_object = [
    "h1-push-v0",  # medium
    "h1-basketball-v0",  # very hard
    "h1-package-v0",  # medium
    "h1-sit_hard-v0",  # hard
    "h1-balance_simple-v0",  # hard
]


env_without_object = [
    "h1-walk-v0",
    "h1-reach-v0",
    "h1-hurdle-v0",
    "h1-crawl-v0",
    "h1-maze-v0",
    "h1-highbar_simple-v0",
    "h1-stand-v0",
    "h1-run-v0",
    "h1-sit_simple-v0",
    "h1-stairs-v0",
    "h1-slide-v0",
    "h1-pole-v0",
]


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer

    Mathematical operations:
    1. Compute squared distance: d_{ij}^2 = ||x_i - x_j||^2 (rotation/translation invariant)
    2. Edge message: m_{ij} = φ_e(h_i, h_j, d_{ij}^2, a_{ij})
    3. Coordinate update: x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i - x_j) * φ_x(m_{ij})
    4. Feature update: h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij})
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

    def coord2radial(self, edge_index, coord):
        """
        Step 1: Compute squared distance d_{ij}^2 = ||x_i - x_j||^2
        This is rotation and translation equivariant.
        Also computes coordinate differences (x_i - x_j) for equivariant updates.
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = coord_diff.pow(2).sum(dim=1, keepdim=True)
        return radial

    def edge_model(self, source, target, radial, edge_attr):
        """
        Step 2: Compute edge message m_{ij} = φ_e(h_i, h_j, d_{ij}^2, a_{ij}).
        Combines source node features, target node features, radial distance, and edge attributes.
        """
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out
    

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        Step 4: Feature update h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij}).
        Aggregates edge messages and updates node features.
        """
        row, col = edge_index
        
        if torch.isnan(edge_attr).any():
            print(f"[DEBUG] NaN in node_model edge_attr input: has_nan={torch.isnan(edge_attr).sum()}")
        
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        
        if torch.isnan(agg).any():
            print(f"[DEBUG] NaN after unsorted_segment_sum: has_nan={torch.isnan(agg).sum()}")
        
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        if torch.isnan(agg).any():
            print(f"[DEBUG] NaN after cat in node_model: has_nan={torch.isnan(agg).sum()}")
        
        out = self.node_mlp(agg)
        
        # Add numerical stability: clamp output
        out = torch.clamp(out, min=-1e6, max=1e6)
        
        if torch.isnan(out).any():
            print(f"[DEBUG] NaN after node_mlp: has_nan={torch.isnan(out).sum()}", flush=True)
            print(f"  out stats: min={out.min()}, max={out.max()}", flush=True)
        
        if self.residual:
            out = x + out
            if torch.isnan(out).any():
                print(f"[DEBUG] NaN after residual connection: has_nan={torch.isnan(out).sum()}", flush=True)
        
        return out

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        h = self.node_model(h, edge_index, edge_feat, node_attr)
        return h


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
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
        :param in_node_nf: Number of features for 'h' at the input
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
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        self.num_edges = self.graph_builder.robot.num_edges
        self._edges_cache = {}

        self.layer = E_GCL(
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
        
        # self.layer = torch.compile(self.layer, dynamic=True)

        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_node_nf, self.hidden_nf), act_fn
        )
        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )
        
        self.to(self.device)

    def forward(self, obs: torch.Tensor, xanchor: torch.Tensor) -> torch.Tensor:
        current_batch_size = obs.shape[0]
        edges = self.get_cached_edges(current_batch_size)
        h_joints, x_joint, _, _ = self.graph_builder.generate_input(obs, xanchor)

        h_joints = self.joint_embedding_in(h_joints)
        
        # Apply single layer (fix: was referencing self.layers which doesn't exist)
        h_joints = self.layer(h=h_joints, edge_index=edges, coord=x_joint)
        
        actions = self.joint_embedding_out(h_joints)

        return actions.view(current_batch_size, self.num_joints)

    def generate_index(self, batch_size: int, device="cuda"):
        src, dst = zip(*self.graph_builder.robot.joint_connections)

        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

        return torch.stack([src_batch, dst_batch])

    def get_cached_edges(self, current_batch_size: int):
        if current_batch_size in self._edges_cache:
            return self._edges_cache[current_batch_size]

        # Generate, cache, and return
        edges = self.generate_index(current_batch_size, self.device)
        self._edges_cache[current_batch_size] = edges
        return edges


class EGNN_V3(nn.Module):
    def __init__(
        self,
        in_joint_nf,
        object_feature_dim,
        out_node_nf,
        hidden_nf,
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
        coord_norm=False,
    ):
        """
        :param in_joint_nf: Number of features for joint nodes (velocity + position)
        :param object_feature_dim: Total dimension of object features (from get_object_feature_dim())
        :param hidden_nf: Number of hidden features
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

        super(EGNN_V3, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.batch_size = batch_size
        self.env_name = env_name
        self.robot = robot
        self.object_feature_dim = object_feature_dim
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        
        self.register_buffer("joint_edges", self.generate_joint_edges(self.batch_size))
        self.joint_out_dim = 2
        
        self.joint_object_dim = self.joint_out_dim * self.num_joints + self.object_feature_dim

        # Joint graph layers (message passing within joints)
        self.layer = E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=0,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                    coords_agg=coords_agg,
                )
        # self.layer = torch.compile(self.layer, dynamic=True)
        # Combined MLP for joint + object features
        self.joint_object_mlp = nn.Sequential(
            nn.Linear(self.joint_object_dim, self.hidden_nf * 4),
            act_fn,
            nn.Linear(self.hidden_nf * 4, self.hidden_nf * 2),
            act_fn,
            nn.Linear(self.hidden_nf * 2, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.num_joints),
            nn.Tanh(),
        )

        # Input embeddings
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_joint_nf, self.hidden_nf), act_fn
        )
        
        # Add LayerNorm for stability
        self.layer_norm = nn.LayerNorm(self.hidden_nf)

        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, self.joint_out_dim), act_fn
        )

        self.to(self.device)

    def forward(self, obs: torch.Tensor, xanchor: torch.Tensor) -> torch.Tensor:
        h_joints, x_joints, h_objects = structure_input(obs, xanchor, self.env_name)
        current_batch_size = h_objects.shape[0]
        joint_edges = self.joint_edges[:, : self.num_joints * current_batch_size * (self.num_joints - 1)]

        h_joints = self.joint_embedding_in(h_joints) 

        for layer in self.joint_layers:
            h_joints = layer(h=h_joints, edge_index=joint_edges, coord=x_joints)

        h_joints = self.joint_embedding_out(h_joints)
        
        h_joints_flat = h_joints.reshape(current_batch_size, -1)
        h_combined = torch.cat([h_joints_flat, h_objects], dim=-1)
        
        actions = self.joint_object_mlp(h_combined)
        
        return actions

    def generate_joint_edges(self, batch_size: int):
        n_nodes = self.num_joints
        idx = torch.arange(n_nodes, device=self.device)

        # Fully-connected directed graph without self-loops
        row = idx.repeat_interleave(n_nodes)
        col = idx.repeat(n_nodes)
        mask = row != col
        row, col = row[mask], col[mask]

        if batch_size == 1:
            return torch.stack([row, col], dim=0)

        # Batch offsets
        offsets = torch.arange(batch_size, device=self.device) * n_nodes
        row = row.unsqueeze(0) + offsets.unsqueeze(1)
        col = col.unsqueeze(0) + offsets.unsqueeze(1)

        return torch.stack(
            [row.reshape(-1), col.reshape(-1)],
            dim=0
        )


@torch.jit.script
def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    JIT-compiled optimized unsorted segment sum using scatter_add.
    """
    result = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids_expanded, data)
    return result


@torch.jit.script
def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    JIT-compiled optimized unsorted segment mean with efficient counting.
    """
    result = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    
    # Sum values
    result.scatter_add_(0, segment_ids_expanded, data)
    
    # Count occurrences
    count = torch.zeros(num_segments, data.size(1), dtype=data.dtype, device=data.device)
    ones = torch.ones_like(data)
    count.scatter_add_(0, segment_ids_expanded, ones)
    
    # Use torch.where to handle division by zero
    return torch.where(count > 0, result / count, result)

@torch.jit.script
def structure_input(obs: torch.Tensor, xanchor: torch.Tensor, env_name: str):
    xanchor = xanchor / 10
    
    if env_name in (
        "h1-walk-v0",
        "h1-reach-v0",
        "h1-hurdle-v0",
        "h1-crawl-v0",
        "h1-maze-v0",
        "h1-highbar_simple-v0",
        "h1-stand-v0",
        "h1-run-v0",
        "h1-sit_simple-v0",
        "h1-stairs-v0",
        "h1-slide-v0",
        "h1-pole-v0",
    ):
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 32:51].reshape(-1, 1)  # [batch*19, 1]
        h = torch.cat([joint_vel, joint_pos], dim=1)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]
        idx = torch.cat(
            [
                torch.arange(0, 7),
                torch.arange(26, 32),
            ]
        )
        h_object = obs[:, idx]
        return h, x, h_object
    elif env_name in ("h1-balance_simple-v0", "h1-sit_hard-v0"):
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 39:58].reshape(-1, 1)  # [batch*19, 1]
        h = torch.cat([joint_vel, joint_pos], dim=1)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]
        idx = torch.cat(
            [
                torch.arange(0, 7),
                torch.arange(26, 39),
                torch.arange(58, 64),
            ]
        )
        h_object = obs[:, idx]
        return h, x, h_object
    elif env_name == "h1-reach-v0":
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 32:51].reshape(-1, 1)  # [batch*19, 1]
        h = torch.cat([joint_vel, joint_pos], dim=1)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]
        idx = torch.cat(
            [
                torch.arange(0, 7),
                torch.arange(26, 32),
                torch.arange(51, 57),
            ]
        )
        h_object = obs[:, idx]
        return h, x, h_object
    elif env_name == "h1-push-v0":
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 32:51].reshape(-1, 1)  # [batch*19, 1]
        h = torch.cat([joint_vel, joint_pos], dim=1)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]
        idx = torch.cat(
            [
                torch.arange(0, 7),
                torch.arange(26, 32),
                torch.arange(51, 63),
            ]
        )
        h_object = obs[:, idx]
        return h, x, h_object
    elif env_name == "h1-door-v0":
        joint_pos = obs[:, 7:26].reshape(-1, 1)  # [batch*19, 1]
        joint_vel = obs[:, 34:53].reshape(-1, 1)  # [batch*19, 1]
        h = torch.cat([joint_vel, joint_pos], dim=1)
        x = (xanchor[:, 1:] - xanchor[:, [0]]).reshape(-1, 3)  # [batch*19, 3]
        idx = torch.cat(
            [
                torch.arange(0, 7),
                torch.arange(26, 34),
                torch.arange(53, 55),
            ]
        )
        h_object = obs[:, idx]
        return h, x, h_object