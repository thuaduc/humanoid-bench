from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egcl import E_GCL


class VirtualNodeModule(nn.Module):
    """
    Virtual Node Module that aggregates node features and broadcasts back.
    Does NOT touch coordinates, maintaining equivariance.
    Includes LayerNorm, residual dropout, and learnable scaling.
    """

    def __init__(self, hidden_nf, act_fn, dropout=0.05, init_scale=0.1):
        """
        Args:
            hidden_nf: Hidden feature dimension
            act_fn: Activation function
            dropout: Dropout probability applied on residual connection
            init_scale: Initial value for learnable scaling
        """
        super(VirtualNodeModule, self).__init__()
        self.hidden_nf = hidden_nf

        # MLP for processing aggregated features
        self.virtual_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        # Learnable scaling parameter
        self.alpha = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_nf)
        # Residual dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h, batch_size, num_joints):
        """
        Args:
            h: Node features [batch_size * num_joints, hidden_nf]
            batch_size: Number of graphs in the batch
            num_joints: Number of nodes per graph

        Returns:
            h_updated: Updated node features with virtual node information
        """
        hidden_nf = h.shape[-1]

        # Reshape to separate batches: [batch_size, num_joints, hidden_nf]
        h_reshaped = h.view(batch_size, num_joints, hidden_nf)

        # Aggregate: mean pooling across nodes within each graph
        h_virtual = h_reshaped.mean(dim=1)  # [batch_size, hidden_nf]

        # Process through MLP
        h_virtual = self.virtual_mlp(h_virtual)  # [batch_size, hidden_nf]

        # Expand and repeat for each node: [batch_size, num_joints, hidden_nf]
        h_broadcast = h_virtual.unsqueeze(1).expand(batch_size, num_joints, hidden_nf)

        # Flatten back to match input shape: [batch_size * num_joints, hidden_nf]
        h_broadcast = h_broadcast.reshape(batch_size * num_joints, hidden_nf)

        # Apply LayerNorm, residual dropout, and learnable scaling
        h_updated = h + self.alpha * self.dropout(self.norm(h_broadcast))

        return h_updated


class EGNN_VN(nn.Module):
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
        coords_agg="mean",
        virtual_node_gate=True,
    ):
        """
        EGNN with Virtual Node integration.

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij)
        :param virtual_node_gate: Whether to use gating in virtual node module
        """

        super(EGNN_VN, self).__init__()
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.out_node_nf = out_node_nf
        self.batch_size = batch_size
        self.robot = robot
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        self.num_edges = self.graph_builder.robot.num_edges
        self._edges_cache = {}

        # EGNN layers
        self.layers = nn.ModuleList()
        self.virtual_node_modules = nn.ModuleList()

        for _ in range(n_layers):
            # Standard EGNN layer
            self.layers.append(
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
            )

            # Virtual node module after each EGNN layer
            self.virtual_node_modules.append(
                VirtualNodeModule(
                    hidden_nf=self.hidden_nf,
                    act_fn=act_fn,
                )
            )

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

        # Initial embedding
        h_joints = self.joint_embedding_in(h_joints)

        # Process through EGNN layers with virtual node updates
        for egnn_layer, virtual_node_module in zip(
            self.layers, self.virtual_node_modules
        ):
            # 1. Standard EGNN update (equivariant operations on h and x)
            h_joints, x_joint, _ = egnn_layer(
                h=h_joints, edge_index=edges, coord=x_joint
            )

            # 2. Virtual node update (only touches h, not x - preserves equivariance)
            h_joints = virtual_node_module(
                h_joints, current_batch_size, self.num_joints
            )

        # Final output projection
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
