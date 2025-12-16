from torch import nn
import torch

from fast_td3.robots.graph_builder import GraphBuilder
from fast_td3.actors.gnn.egnn import E_GCL, env_with_object
from humanoid_bench.envs.custom_env import unflatten_obs


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
            nn.Linear(in_joint_nf, self.hidden_nf), act_fn
        )
        
        self.object_embedding = nn.Sequential(
            nn.Linear(in_object_nf, self.hidden_nf), 
            act_fn
        )
        
        self.joint_embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, out_node_nf),
            nn.Tanh(),
        )
        
        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = unflatten_obs(obs, self.env_name)    
        
        current_batch_size = obs["joint_velocities"].shape[0]
        num_objects = obs["object_positions"].shape[1]  # Get num_objects from shape
        
        joint_edges = self.generate_joint_edges(current_batch_size)
        cross_edges = self.get_cached_cross_edges(current_batch_size, num_objects)
        
        # Joint features: stack velocity and position, then flatten for all joints in batch
        h_joints = torch.stack([obs["joint_velocities"].reshape(-1), obs["joint_positions"].reshape(-1)], dim=1)
        x_joints = obs["joint_x"].reshape(-1, 3)
        
        # Object features: stack linear and angular velocities
        # Shape: (batch_size, num_objects, 3) -> (batch_size, num_objects, 6) -> (batch_size*num_objects, 6)
        h_objects = torch.cat([
            obs["object_quaternions"], # (batch_size, num_objects, 4)
            obs["object_velocities"]  # (batch_size, num_objects, 3)
        ], dim=-1)  # (batch_size, num_objects, 6)
        h_objects = h_objects.reshape(current_batch_size * num_objects, -1)  # (batch_size*num_objects, 6)
        
        # Object positions: flatten to (batch_size*num_objects, 3)
        x_objects = obs["object_positions"].reshape(current_batch_size * num_objects, 3)
       
        h_joints = self.joint_embedding_in(h_joints)  # [batch*num_joints, hidden]
        h_objects = self.object_embedding(h_objects)  # [batch*num_objects, hidden]
            
        h_combined = torch.cat([h_joints, h_objects], dim=0)
        x_combined = torch.cat([x_joints, x_objects], dim=0)
        h_combined, x_combined, _ = self.cross_layer(h=h_combined, edge_index=cross_edges, coord=x_combined)
        
        h_joints = h_combined[:current_batch_size * self.num_joints]
        x_joints = x_combined[:current_batch_size * self.num_joints]
        for layer in self.joint_layers:
            h_joints, x_joints, _ = layer(h=h_joints, edge_index=joint_edges, coord=x_joints)

        actions = self.joint_embedding_out(h_joints)

        return actions.view(current_batch_size, self.num_joints)

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
