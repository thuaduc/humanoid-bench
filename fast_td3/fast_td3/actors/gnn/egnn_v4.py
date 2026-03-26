from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egcl import E_GCL, env_with_object
from humanoid_bench.envs.custom_env import unflatten_obs
from fast_td3.actors.so3.so3 import SO3_Embedding, SO3_Rotation


class ObjectEquivariantEncoder(nn.Module):
    def __init__(self, num_objects, scalar_dim, hidden_dim, output_dim, lmax=1):
        super().__init__()
        self.num_objects = num_objects
        self.lmax = lmax
        self.output_dim = output_dim
        self.so3_rotation = SO3_Rotation(lmax)
        
        # Input: 2 vectors (linear vel, angular vel) -> L=1 features.
        # Plus magnitudes -> L=0 features.
        # Total coefficients: 1 (L=0) + 3 (L=1) = 4 per vector.
        # We have 2 vectors (channels).
        # Flattened dim = num_channels * ( (lmax+1)^2 ) = 2 * 4 = 8.
        
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, quaternions, scalars):
        # quaternions: (batch, num_objects, 4) (w, x, y, z)
        # scalars: (batch, num_objects, 6) (vx, vy, vz, wx, wy, wz)
        
        batch_size = quaternions.shape[0]
        device = quaternions.device
        
        # Flatten batch and objects
        quats_flat = quaternions.reshape(-1, 4)
        scalars_flat = scalars.reshape(-1, 6)
        
        # 1. Convert quaternions to rotation matrices
        rot_mats = self.quaternion_to_matrix(quats_flat) # (N, 3, 3)
        
        # 2. Prepare velocities
        lin_vel_global = scalars_flat[:, :3]
        ang_vel_local = scalars_flat[:, 3:]
        
        # Convert local angular velocity to global
        # w_global = R * w_local
        ang_vel_global = torch.bmm(rot_mats, ang_vel_local.unsqueeze(-1)).squeeze(-1)
        
        # 3. Construct SO3 Embedding
        lin_mag = torch.norm(lin_vel_global, dim=1, keepdim=True)
        ang_mag = torch.norm(ang_vel_global, dim=1, keepdim=True)
        
        # Create embedding tensor: (N, 4, 2)
        # Channel 0: Linear
        # Channel 1: Angular
        embedding_tensor = torch.zeros(batch_size * self.num_objects, 4, 2, device=device)
        
        # L=0
        embedding_tensor[:, 0, 0] = lin_mag.squeeze(-1)
        embedding_tensor[:, 0, 1] = ang_mag.squeeze(-1)
        
        # L=1 components.
        # We follow the standard SO(3)/e3nn convention where, for l=1, the
        # spherical harmonic ordering is (m = -1, 0, 1) which corresponds to
        # Cartesian directions (Y, Z, X), in that order.
        #
        # Therefore, given a vector with Cartesian components (x, y, z), we map
        # it to SH indices 1, 2, 3 as (y, z, x) respectively:
        #   Index 1: m = -1 -> Y component
        #   Index 2: m =  0 -> Z component
        #   Index 3: m =  1 -> X component
        
        # Linear (channel 0)
        embedding_tensor[:, 1, 0] = lin_vel_global[:, 1] # y -> m=-1
        embedding_tensor[:, 2, 0] = lin_vel_global[:, 2] # z -> m=0
        embedding_tensor[:, 3, 0] = lin_vel_global[:, 0] # x -> m=1
        
        # Angular (channel 1)
        embedding_tensor[:, 1, 1] = ang_vel_global[:, 1] # y -> m=-1
        embedding_tensor[:, 2, 1] = ang_vel_global[:, 2] # z -> m=0
        embedding_tensor[:, 3, 1] = ang_vel_global[:, 0] # x -> m=1
        
        embedding = SO3_Embedding(
            length=batch_size * self.num_objects,
            lmax_list=[1],
            num_channels=2,
            device=device,
            dtype=torch.float32
        )
        embedding.set_embedding(embedding_tensor)
        
        # 4. Rotate by inverse of object rotation
        self.so3_rotation.set_wigner(rot_mats)
        rotated_embedding = self.so3_rotation.rotate_inv(embedding.embedding, 1, 1)
        
        # 5. Extract features (invariants)
        # Flatten coefficients and channels
        features = rotated_embedding.reshape(batch_size * self.num_objects, -1)
        
        # 6. MLP
        out = self.mlp(features)
        
        return out.reshape(batch_size, self.num_objects, self.output_dim)

    def quaternion_to_matrix(self, q):
        # q: (N, 4) -> (w, x, y, z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w
        
        row0 = torch.stack([1 - 2*yy - 2*zz, 2*xy - 2*zw, 2*xz + 2*yw], dim=1)
        row1 = torch.stack([2*xy + 2*zw, 1 - 2*xx - 2*zz, 2*yz - 2*xw], dim=1)
        row2 = torch.stack([2*xz - 2*yw, 2*yz + 2*xw, 1 - 2*xx - 2*yy], dim=1)
        
        return torch.stack([row0, row1, row2], dim=1)


class EGNN_V4(nn.Module):
    """
    EGNN v4 with integrated ObjectEquivariantEncoder.

    Combines joint graph message passing with learned geometric object representations
    using equivariant features (scalars + vectors).
    """

    def __init__(
        self,
        in_joint_nf,
        out_node_nf,
        hidden_nf,
        device,
        batch_size,
        act_fn,
        n_layers,
        robot,
        env_name,
        encoder_hidden_dim=32,
        encoder_output_dim=10,
        residual=True,
        attention=False,
        normalize=False,
        tanh=False,
        coords_agg="mean",
        coord_norm=False,
    ):
        """
        :param in_joint_nf: Number of features for joint nodes (velocity + position)
        :param hidden_nf: Number of hidden features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layers for the EGNN
        :param robot: Robot instance
        :param env_name: Environment name
        :param encoder_hidden_dim: Hidden dimension for geometric encoder
        :param encoder_output_dim: Output dimension for geometric encoder
        :param residual: Use residual connections
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij)
        :param coords_agg: Coordinate aggregation method
        :param coord_norm: Coordinate normalization
        """

        super(EGNN_V4, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.batch_size = batch_size
        self.env_name = env_name
        self.robot = robot
        self.graph_builder = GraphBuilder(env_name, batch_size, device, robot)
        self.num_joints = self.graph_builder.robot.num_joints
        self.num_objects = 2 if self.env_name in env_with_object else 1
        self._joint_edges_cache = {}
        
        self.joint_out_dim = 8
                
        self.object_encoder = ObjectEquivariantEncoder(
            num_objects=self.num_objects,
            scalar_dim=6,
            hidden_dim=32,
            output_dim=20,
        )
        
        self.joint_object_dim = self.joint_out_dim * self.num_joints + 26

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

        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, self.joint_out_dim), act_fn
        )

        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = unflatten_obs(obs, self.env_name)
        current_batch_size = obs["joint_velocities"].shape[0]
        joint_edges = self.get_cached_joint_edges(current_batch_size)

        # Joint inputs
        h_joints = torch.stack(
            [obs["joint_velocities"].reshape(-1), obs["joint_positions"].reshape(-1)],
            dim=1,
        )
        h_joints = self.joint_embedding_in(h_joints)
        x_joints = obs["joint_x"].reshape(-1, 3)
        
        # Encode object features using geometric encoder
        h_objects = self.object_encoder(
            quaternions=obs["object_quaternions"],  # (batch, num_objects, 4)
            scalars=obs["object_velocities"],  # (batch, num_objects, 6)
        )
        h_objects = torch.cat(
            [h_objects, obs["object_x"].reshape(current_batch_size, -1)],dim=-1,
        ).reshape(current_batch_size, -1)

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
