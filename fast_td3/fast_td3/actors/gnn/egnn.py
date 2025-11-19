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

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

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

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

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

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        Step 3: Coordinate update x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i - x_j) * φ_x(m_{ij}).
        Updates coordinates using direction vectors (x_i - x_j) weighted by scalar φ_x(m_{ij}).
        This ensures rotation equivariance: if x -> Rx, then x^{l+1} -> Rx^{l+1}.
        """
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception(f"Wrong coords_agg parameter: {self.coords_agg}")
        coord = coord + agg
        return coord

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
        if self.residual:
            out = x + out
        return out, agg

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index

        radial, coord_diff = self.coord2radial(edge_index, coord)
        
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


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
                )
                for _ in range(n_layers)
            ]
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

        h_joints = self.joint_embedding_in(h_joints)
        for layer in self.layers:
            h_joints, x_joint, _ = layer(h=h_joints, edge_index=edges, coord=x_joint)

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


class EGNN_V2(nn.Module):
    """
    EGNN v2 with separate joint and object graphs and cross-graph aggregation.
    
    Implements the architecture from Section 3.3:
    1. Disjoint graphs for joints and objects
    2. L rounds of E(n)-Equivariant Graph Convolution on each subgraph
    3. Cross-graph aggregation from object to joint nodes
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
        self._edges_cache = {}
        
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
        
        # Object graph layers (if needed for object-object interactions)
        # For now, we keep objects static as they don't interact with each other
        # But we embed them into hidden space
        self.object_embedding = nn.Sequential(
            nn.Linear(in_object_nf, self.hidden_nf), 
            act_fn
        )
        
        # Cross-graph aggregation: object → joint
        # MLPs for computing cross-graph messages
        self.cross_edge_mlp = nn.Sequential(
            nn.Linear(self.hidden_nf * 2 + 1, self.hidden_nf),  # h_joint, h_object, dist
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
        )
        
        self.cross_node_mlp = nn.Sequential(
            nn.Linear(self.hidden_nf * 2, self.hidden_nf),  # h_joint, aggregated_messages
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )
        
        # Coordinate update from cross-graph
        layer = nn.Linear(self.hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(self.hidden_nf, self.hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if tanh:
            coord_mlp.append(nn.Tanh())
        self.cross_coord_mlp = nn.Sequential(*coord_mlp)
        
        # Input embeddings
        self.joint_embedding_in = nn.Sequential(
            nn.Linear(in_joint_nf, self.hidden_nf), 
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
        Forward pass implementing Algorithm from Section 3.3:
        1. Local processing in joint and object subgraphs
        2. Cross-graph aggregation from objects to joints
        3. Output actions
        """
        current_batch_size = obs.shape[0]
        
        # Get input features and coordinates
        h_joints, x_joints, h_objects, x_objects = self.graph_builder.generate_input_v2(obs, xanchor)
        
        # Embed joint and object features
        h_joints = self.joint_embedding_in(h_joints)  # [batch*num_joints, hidden]
        h_objects = self.object_embedding(h_objects)  # [batch, hidden]
        
        # Get joint graph edges
        joint_edges = self.get_cached_edges(current_batch_size)
        
        # Step 1: Local processing within joint subgraph (L rounds)
        for layer in self.joint_layers:
            h_joints, x_joints, _ = layer(h=h_joints, edge_index=joint_edges, coord=x_joints)
        
        # Step 2: Cross-graph aggregation from objects to joints
        # Each joint receives messages from all objects in its batch
        h_joints_pooled, x_joints_pooled = self.cross_graph_aggregation(
            h_joints, x_joints, h_objects, x_objects, current_batch_size
        )
        
        # Update joint features and coordinates with pooled information
        h_joints = h_joints + h_joints_pooled
        x_joints = x_joints + x_joints_pooled
        
        # Step 3: Output projection
        actions = self.joint_embedding_out(h_joints)
        
        return actions.view(current_batch_size, self.num_joints)
    
    def cross_graph_aggregation(
        self, 
        h_joints: torch.Tensor,  # [batch*num_joints, hidden]
        x_joints: torch.Tensor,  # [batch*num_joints, 3]
        h_objects: torch.Tensor,  # [batch, hidden]
        x_objects: torch.Tensor,  # [batch, 3]
        batch_size: int
    ):
        """
        Implement cross-graph aggregation: object → joint.
        Each joint acts as a virtual node receiving messages from all object nodes.
        
        Equations (8) and (9):
        h_joint_pooled_i = h_i + φ_cross(h_i, Σ_{j∈V_object} m_obj→joint_ij)
        x_joint_pooled_i = x_i + Σ_{j∈V_object} (x_i - x_j) φ_x(m_obj→joint_ij)
        """
        # Reshape joints to [batch, num_joints, ...]
        h_joints_batch = h_joints.view(batch_size, self.num_joints, -1)  # [batch, num_joints, hidden]
        x_joints_batch = x_joints.view(batch_size, self.num_joints, 3)   # [batch, num_joints, 3]
        
        # Expand objects for broadcasting: [batch, 1, hidden/3]
        h_objects_exp = h_objects.unsqueeze(1)  # [batch, 1, hidden]
        x_objects_exp = x_objects.unsqueeze(1)  # [batch, 1, 3]
        
        # Compute coordinate differences: x_joint - x_object
        coord_diff = x_joints_batch - x_objects_exp  # [batch, num_joints, 3]
        
        # Compute squared distances
        radial = coord_diff.pow(2).sum(dim=2, keepdim=True)  # [batch, num_joints, 1]
        
        # Expand joint features for broadcasting: [batch, num_joints, hidden]
        # Object features already expanded: [batch, 1, hidden]
        # Need to broadcast: [batch, num_joints, hidden]
        h_objects_broadcast = h_objects_exp.expand(-1, self.num_joints, -1)  # [batch, num_joints, hidden]
        
        # Compute edge messages: m_obj→joint_ij = φ_e(h_joint_i, h_object_j, ||x_i - x_j||²)
        edge_input = torch.cat([
            h_joints_batch,      # [batch, num_joints, hidden]
            h_objects_broadcast, # [batch, num_joints, hidden]
            radial               # [batch, num_joints, 1]
        ], dim=2)  # [batch, num_joints, 2*hidden + 1]
        
        # Flatten batch dimension for MLP
        edge_feat = self.cross_edge_mlp(
            edge_input.view(-1, 2 * self.hidden_nf + 1)
        ).view(batch_size, self.num_joints, self.hidden_nf)  # [batch, num_joints, hidden]
        
        # Aggregate messages from all objects (in this case, just one object per batch)
        # If there were multiple objects, we'd sum over the object dimension
        agg_messages = edge_feat  # [batch, num_joints, hidden]
        
        # Update node features: h_i + φ_h(h_i, Σ_j m_ij)
        node_input = torch.cat([
            h_joints_batch,  # [batch, num_joints, hidden]
            agg_messages     # [batch, num_joints, hidden]
        ], dim=2)  # [batch, num_joints, 2*hidden]
        
        h_pooled = self.cross_node_mlp(
            node_input.view(-1, 2 * self.hidden_nf)
        ).view(batch_size, self.num_joints, self.hidden_nf)  # [batch, num_joints, hidden]
        
        # Update coordinates: Σ_j (x_i - x_j) * φ_x(m_ij)
        coord_weights = self.cross_coord_mlp(
            edge_feat.view(-1, self.hidden_nf)
        ).view(batch_size, self.num_joints, 1)  # [batch, num_joints, 1]
        
        x_pooled = coord_diff * coord_weights  # [batch, num_joints, 3]
        
        # Reshape back to [batch*num_joints, ...]
        h_pooled = h_pooled.view(-1, self.hidden_nf)  # [batch*num_joints, hidden]
        x_pooled = x_pooled.view(-1, 3)               # [batch*num_joints, 3]
        
        return h_pooled, x_pooled
    
    def generate_index(self, batch_size: int, device="cuda"):
        """Generate edge indices for the joint graph across batches."""
        src, dst = zip(*self.graph_builder.robot.joint_connections)
        
        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)
        
        # Create batch offsets and expand edges
        offsets = torch.arange(batch_size, device=device) * self.num_joints
        src_batch = (src.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        dst_batch = (dst.unsqueeze(0) + offsets.unsqueeze(1)).flatten()
        
        return torch.stack([src_batch, dst_batch])
    
    def get_cached_edges(self, current_batch_size: int):
        """Get cached edge indices for the given batch size."""
        if current_batch_size in self._edges_cache:
            return self._edges_cache[current_batch_size]
        
        # Generate, cache, and return
        edges = self.generate_index(current_batch_size, self.device)
        self._edges_cache[current_batch_size] = edges
        return edges


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