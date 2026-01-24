from torch import nn
import torch


# Coordinate normalization following SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95
class CoordNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


# Environment classification for object inclusion
env_with_object = [
    "h1-push-v0",  # medium
    "h1-basketball-v0",  # very hard
    "h1-package-v0",  # medium
    "h1-sit_hard-v0",  # hard
    "h1-balance_simple-v0",  # hard
    "h1-push-v1",  # medium
    "h1-basketball-v1",  # very hard
    "h1-package-v1",  # medium
    "h1-sit_hard-v1",  # hard
    "h1-balance_simple-v1",  # hard
    "h1-push-v1",  # hard
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
        coord_norm=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.coord_norm_enabled = coord_norm
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
            act_fn,
            nn.LayerNorm(output_nf),
        )

        # layer = nn.Linear(hidden_nf, 1, bias=False)
        # torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        # coord_mlp = []
        # coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        # coord_mlp.append(act_fn)
        # coord_mlp.append(layer)

        # if self.tanh:
        #     coord_mlp.append(nn.Tanh())
        # self.coord_mlp = nn.Sequential(*coord_mlp)

        # if coord_norm:
        #     self.coord_norm = CoordNorm(eps=1e-8, scale_init=1.0)

        # if self.attention:
        #     self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def coord2radial(self, edge_index, coord):
        """
        Step 1: Compute squared distance d_{ij}^2 = ||x_i - x_j||^2
        This is rotation and translation equivariant.
        Also computes coordinate differences (x_i - x_j) for equivariant updates.
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = coord_diff.pow(2).sum(dim=1, keepdim=True)

        # if self.normalize:
        #     norm = torch.sqrt(radial).detach() + self.epsilon
        #     coord_diff = coord_diff / norm

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
        
        if self.coord_norm_enabled:
            coord = self.coord_norm(coord)
        
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

        radial = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        # coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class E_GCL_DualModel(nn.Module):
    """
    E(n) Equivariant Convolutional Layer with dual models for different edge types.
    
    Supports two types of edges with separate node_model and edge_model:
    - Type 0: Joint-Joint interactions
    - Type 1: Object-Joint interactions
    
    Mathematical operations:
    1. Compute squared distance: d_{ij}^2 = ||x_i - x_j||^2 (rotation/translation invariant)
    2. Edge message: m_{ij} = φ_e^{type}(h_i, h_j, d_{ij}^2, a_{ij})  [type-specific]
    3. Coordinate update: x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i - x_j) * φ_x(m_{ij})
    4. Feature update: h_i^{l+1} = φ_h^{type}(h_i, Σ_{j∈N(i)} m_{ij})  [type-specific]
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
        coord_norm=False,
    ):
        super(E_GCL_DualModel, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.coord_norm_enabled = coord_norm
        self.epsilon = 1e-8
        edge_coords_nf = 1

        # Joint-Joint edge model
        self.edge_mlp_joint = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        # Object-Joint edge model
        self.edge_mlp_cross = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        # Joint-Joint node model
        self.node_mlp_joint = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
            act_fn,
            nn.LayerNorm(output_nf)
        )

        # Object-Joint node model
        self.node_mlp_cross = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
            act_fn,
            nn.LayerNorm(output_nf)
        )
        
        # Coordinate update model (shared across edge types)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if coord_norm:
            self.coord_norm = CoordNorm(eps=1e-8, scale_init=1.0)

        if self.attention:
            self.att_mlp_joint = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())
            self.att_mlp_cross = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

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

    def edge_model(self, source, target, radial, edge_attr, edge_type):
        """
        Step 2: Compute edge message m_{ij} = φ_e^{type}(h_i, h_j, d_{ij}^2, a_{ij}).
        Combines source node features, target node features, radial distance, and edge attributes.
        Uses different MLPs based on edge type.
        
        Args:
            source: Source node features
            target: Target node features
            radial: Squared distance
            edge_attr: Edge attributes
            edge_type: 0 for joint-joint, 1 for object-joint
        """
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        
        # Select appropriate edge MLP based on edge type
        if edge_type == 0:  # Joint-Joint
            out = self.edge_mlp_joint(out)
            if self.attention:
                att_val = self.att_mlp_joint(out)
                out = out * att_val
        else:  # Object-Joint (type 1)
            out = self.edge_mlp_cross(out)
            if self.attention:
                att_val = self.att_mlp_cross(out)
                out = out * att_val
        
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        Step 3: Coordinate update x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i - x_j) * φ_x(m_{ij}).
        Updates coordinates using direction vectors (x_i - x_j) weighted by scalar φ_x(m_{ij}).
        This ensures rotation equivariance: if x -> Rx, then x^{l+1} -> Rx^{l+1}.
        Coordinate update model is shared across edge types.
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
        
        if self.coord_norm_enabled:
            coord = self.coord_norm(coord)
        
        return coord

    def node_model(self, x, edge_index, edge_attr, edge_type, node_attr=None):
        """
        Step 4: Feature update h_i^{l+1} = φ_h^{type}(h_i, Σ_{j∈N(i)} m_{ij}).
        Aggregates edge messages and updates node features.
        Uses different MLPs based on edge type.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge messages
            edge_type: 0 for joint-joint, 1 for object-joint
            node_attr: Optional additional node attributes
        """
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        # Select appropriate node MLP based on edge type
        if edge_type == 0:  # Joint-Joint
            out = self.node_mlp_joint(agg)
        else:  # Object-Joint (type 1)
            out = self.node_mlp_cross(agg)
        
        if self.residual:
            out = x + out
        
        return out, agg

    def forward(
        self,
        h,
        edge_index_joint,
        edge_index_cross,
        coord,
        edge_attr_joint=None,
        edge_attr_cross=None,
        node_attr=None,
    ):
        """
        Forward pass with dual edge types.
        
        Args:
            h: Node features
            edge_index_joint: Joint-joint edge indices [2, num_joint_edges]
            edge_index_cross: Object-joint edge indices [2, num_cross_edges]
            coord: Node coordinates
            edge_attr_joint: Joint-joint edge attributes (optional)
            edge_attr_cross: Object-joint edge attributes (optional)
            node_attr: Additional node attributes (optional)
            
        Returns:
            h: Updated node features
            coord: Updated coordinates
            edge_attr: Updated edge attributes (for consistency)
        """
        # Process joint-joint edges
        if edge_index_joint.shape[1] > 0:
            row, col = edge_index_joint
            radial_joint, coord_diff_joint = self.coord2radial(edge_index_joint, coord)
            edge_feat_joint = self.edge_model(
                h[row], h[col], radial_joint, edge_attr_joint, edge_type=0
            )
            coord = self.coord_model(coord, edge_index_joint, coord_diff_joint, edge_feat_joint)
            h, _ = self.node_model(h, edge_index_joint, edge_feat_joint, edge_type=0, node_attr=node_attr)

        # Process object-joint edges
        if edge_index_cross.shape[1] > 0:
            row, col = edge_index_cross
            radial_cross, coord_diff_cross = self.coord2radial(edge_index_cross, coord)
            edge_feat_cross = self.edge_model(
                h[row], h[col], radial_cross, edge_attr_cross, edge_type=1
            )
            coord = self.coord_model(coord, edge_index_cross, coord_diff_cross, edge_feat_cross)
            h, _ = self.node_model(h, edge_index_cross, edge_feat_cross, edge_type=1, node_attr=node_attr)

        return h, coord, None



@torch.jit.script
def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """
    JIT-compiled optimized unsorted segment sum using scatter_add.
    """
    result = torch.zeros(
        num_segments, data.size(1), dtype=data.dtype, device=data.device
    )
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids_expanded, data)
    return result


@torch.jit.script
def unsorted_segment_mean(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """
    JIT-compiled optimized unsorted segment mean with efficient counting.
    """
    result = torch.zeros(
        num_segments, data.size(1), dtype=data.dtype, device=data.device
    )
    segment_ids_expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))

    # Sum values
    result.scatter_add_(0, segment_ids_expanded, data)

    # Count occurrences
    count = torch.zeros(
        num_segments, data.size(1), dtype=data.dtype, device=data.device
    )
    ones = torch.ones_like(data)
    count.scatter_add_(0, segment_ids_expanded, ones)

    # Use torch.where to handle division by zero
    return torch.where(count > 0, result / count, result)
