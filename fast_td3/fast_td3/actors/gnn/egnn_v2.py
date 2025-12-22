from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egnn import (
    E_GCL,
    env_with_object,
    unsorted_segment_mean,
    unsorted_segment_sum,
)
from humanoid_bench.envs.custom_env import unflatten_obs


class E_GCL_2(nn.Module):
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
        attention=False,
    ):
        super(E_GCL_2, self).__init__()
        input_edge = input_nf * 2
        self.attention = attention
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

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

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
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return out
        
    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index

        radial = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        h = self.node_model(h, edge_index, edge_feat, node_attr)

        return h


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
        coords_agg="mean",
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
        self.cross_layer = E_GCL_2(
            self.hidden_nf,
            self.hidden_nf,
            self.hidden_nf * 2,
            edges_in_d=in_edge_nf,
            act_fn=act_fn,
            attention=attention,
        )

        # Input embeddings
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_joint_nf, self.hidden_nf), act_fn
        )

        self.object_embedding = nn.Sequential(
            nn.Linear(in_object_nf, self.hidden_nf), act_fn
        )

        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = unflatten_obs(obs, self.env_name)

        # shapes / edges
        current_batch_size = obs["joint_velocities"].shape[0]
        num_objects = obs["object_positions"].shape[1]

        joint_edges = self.get_cached_joint_edges(current_batch_size)
        cross_edges = self.get_cached_cross_edges(current_batch_size, num_objects)

        # joint inputs
        h_joints = torch.stack(
            [
                obs["joint_velocities"].reshape(-1),
                obs["joint_positions"].reshape(-1),
            ],
            dim=1,
        )
        h_objects = torch.cat(
            [
                obs["object_quaternions"],
                obs["object_velocities"],
            ],
            dim=-1,
        )
        x_joints = obs["joint_x"].reshape(-1, 3)
        h_objects = h_objects.reshape(current_batch_size * num_objects, -1)
        x_objects = obs["object_positions"].reshape(current_batch_size * num_objects, 3)

        h_joints = self.joint_embedding_in(h_joints)
        h_objects = self.object_embedding(h_objects)

        f_fact = h_joints  # SAVE PRE-LOCAL VERSION

        h_cross = torch.cat([f_fact, h_objects], dim=0)
        x_cross = torch.cat([x_joints, x_objects], dim=0)
        h_cross = self.cross_layer(
            h=h_cross,
            edge_index=cross_edges,
            coord=x_cross,
        )

        h_joints = f_fact
        for layer in self.joint_layers:
            h_joints, x_joints, _ = layer(
                h=h_joints,
                edge_index=joint_edges,
                coord=x_joints,
            )

        h_final = h_joints + h_cross[: f_fact.shape[0]]
        actions = self.joint_embedding_out(h_final)

        return actions.reshape(current_batch_size, self.num_joints)

    def generate_joint_edges(self, batch_size: int):
        src, dst = zip(*self.graph_builder.robot.joint_connections)

        src = torch.tensor(src, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst, dtype=torch.long, device=self.device)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=self.device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

        return torch.stack([src_batch, dst_batch])
    
    def generate_cross_edges(self, batch_size: int, num_objs: int):
        num_joints = self.num_joints
        object_start_idx = batch_size * num_joints
        
        batch_ids = torch.arange(batch_size, device=self.device)
        joint_ids_local = torch.arange(num_joints, device=self.device)
        obj_ids_local = torch.arange(num_objs, device=self.device)
        
        # Joints: [B, J, O]
        joints_expanded = (batch_ids[:, None] * num_joints + joint_ids_local[None, :]).unsqueeze(2).expand(-1, -1, num_objs)
        
        # Objects: [B, J, O]
        objs_expanded = (object_start_idx + batch_ids[:, None] * num_objs + obj_ids_local[None, :]).unsqueeze(1).expand(-1, num_joints, -1)
        
        # Flatten to get edge lists
        src_obj_to_joint = objs_expanded.flatten()
        dst_obj_to_joint = joints_expanded.flatten()
        
        # Bidirectional: Add Joint -> Object
        src_joint_to_obj = dst_obj_to_joint
        dst_joint_to_obj = src_obj_to_joint
        
        src = torch.cat([src_obj_to_joint, src_joint_to_obj])
        dst = torch.cat([dst_obj_to_joint, dst_joint_to_obj])

        return torch.stack([src, dst])
    
    def get_cached_joint_edges(self, current_batch_size: int):
        """Get cached edge indices for the joint graph."""
        if current_batch_size in self._joint_edges_cache:
            return self._joint_edges_cache[current_batch_size]
        
        edges = self.generate_joint_edges(current_batch_size)
        self._joint_edges_cache[current_batch_size] = edges
        return edges
    
    def get_cached_cross_edges(self, current_batch_size: int, num_objs: int = None):
        """Get cached edge indices for cross-graph connections."""
        if current_batch_size in self._cross_edges_cache:
            return self._cross_edges_cache[current_batch_size]
        
        edges = self.generate_cross_edges(current_batch_size, num_objs=num_objs)
        self._cross_edges_cache[current_batch_size] = edges
        return edges
