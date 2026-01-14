from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egcl import E_GCL, env_with_object
from fast_td3.fast_td3_utils import EmpiricalNormalization
from humanoid_bench.envs.custom_env import unflatten_obs


class EGNN_V3(nn.Module):
    """
    EGNN v3 with cross-graph message passing from objects to joints.

    Uses E_GCL for both:
    1. Message passing within the joint graph
    2. Unidirectional message passing from objects to joints
    """

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
        self._joint_edges_cache = {}
        
        self.joint_out_dim = 8
        
        self.joint_object_dim = self.joint_out_dim * self.num_joints + self.object_feature_dim

        # Joint graph layers (message passing within joints)
        self.joint_layers = nn.ModuleList(
            [
                E_GCL(
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
                    coord_norm=coord_norm,
                )
                for _ in range(n_layers)
            ]
        )
        # Combined MLP for joint + object features
        self.joint_object_mlp = nn.Sequential(
            nn.Linear(self.joint_object_dim, self.hidden_nf * 5),
            act_fn,
            nn.Linear(self.hidden_nf * 5, self.hidden_nf  * 2),
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

        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, self.joint_out_dim), act_fn
        )

        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = unflatten_obs(obs, self.env_name)
        current_batch_size = obs["joint_velocities"].shape[0]
        joint_edges = self.get_cached_joint_edges(current_batch_size)

        # joint inputs
        h_joints = torch.stack([obs["joint_velocities"].reshape(-1), obs["joint_positions"].reshape(-1),], dim=1)
        h_joints = self.joint_embedding_in(h_joints)
        x_joints = obs["joint_x"].reshape(-1, 3)
        
        h_objects = torch.cat(
            [obs["object_x"], obs["object_quaternions"], obs["object_velocities"]],dim=-1,
        ).reshape(current_batch_size, -1) 
        
        if "object_others" in obs:
            h_objects = torch.cat([h_objects, obs["object_others"]], dim=-1)

        # Message passing within joints
        for layer in self.joint_layers:
            h_joints, x_joints, _ = layer(h=h_joints, edge_index=joint_edges, coord=x_joints)

        h_joints = self.joint_embedding_out(h_joints)
        
        h_joints_flat = h_joints.reshape(current_batch_size, -1)
        h_combined = torch.cat([h_joints_flat, h_objects], dim=-1)
        
        actions = self.joint_object_mlp(h_combined)
        
        return actions

    def generate_joint_edges(self, batch_size: int):
        src, dst = zip(*self.graph_builder.robot.joint_connections)

        src = torch.tensor(src, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst, dtype=torch.long, device=self.device)

        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=self.device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

        return torch.stack([src_batch, dst_batch])
    
    def get_cached_joint_edges(self, current_batch_size: int):
        """Get cached edge indices for the joint graph."""
        if current_batch_size in self._joint_edges_cache:
            return self._joint_edges_cache[current_batch_size]
        
        edges = self.generate_joint_edges(current_batch_size)
        self._joint_edges_cache[current_batch_size] = edges
        return edges
