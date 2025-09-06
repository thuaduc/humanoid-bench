import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import glorot_uniform, uniform
from jax.tree_util import tree_map

from fast_td3.skeleton_builder import build_edge_index_and_attr
from fast_td3.robots.h1 import h1_fk
from fast_td3.robots.h1_jax import h1_jax_fk

import torch


def unsorted_segment_sum(data, segment_ids, num_segments):
    return jax.ops.segment_sum(data, segment_ids, num_segments)


def unsorted_segment_mean(data, segment_ids, num_segments):
    seg_sum = jax.ops.segment_sum(data, segment_ids, num_segments)
    seg_count = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    seg_count = jnp.maximum(seg_count, 1)  # Avoid 0 division
    return seg_sum / seg_count


def xavier_init(gain):
    def init(key, shape, dtype):
        bound = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    return init


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer

    Mathematical operations:
    1. Compute squared distance: d_{ij}^2 = ||x_i - x_j||^2 (rotation/translation invariant)
    2. Edge message: m_{ij} = φ_e(h_i, h_j, d_{ij}^2, a_{ij})
    3. Coordinate update: x_i^{l+1} = x_i^l + Σ_{j∈N(i)} (x_i - x_j) * φ_x(m_{ij})
    4. Feature update: h_i^{l+1} = φ_h(h_i, Σ_{j∈N(i)} m_{ij})
    """

    input_nf: int
    output_nf: int
    hidden_nf: int
    edges_in_d: int = 0
    act_fn: callable = nn.silu
    residual: bool = True
    attention: bool = False
    normalize: bool = False
    coords_agg: str = "mean"
    tanh: bool = False
    epsilon: float = 1e-8

    def setup(self):
        input_edge = self.input_nf * 2
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
            ]
        )

        self.node_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(
                    self.output_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
            ]
        )

        coord_mlp_layers = [
            nn.Dense(
                self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
            ),
            self.act_fn,
            nn.Dense(1, use_bias=False, kernel_init=xavier_init(gain=0.001)),
        ]

        if self.tanh:
            coord_mlp_layers.append(nn.tanh)

        self.coord_mlp = nn.Sequential(coord_mlp_layers)

        if self.attention:
            self.att_mlp = nn.Sequential(
                [
                    nn.Dense(1, kernel_init=glorot_uniform(), bias_init=uniform()),
                    nn.sigmoid,
                ]
            )

    def coord2radial(self, edge_index, coord):
        """
        Step 1: Compute squared distance d_{ij}^2 = ||x_i - x_j||^2
        This is rotation and translation equivariant.
        Also computes coordinate differences (x_i - x_j) for equivariant updates.
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = jnp.sum(coord_diff**2, axis=1, keepdims=True)

        if self.normalize:
            norm = jnp.sqrt(radial) + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr):
        """
        Step 2: Compute edge message m_{ij} = φ_e(h_i, h_j, d_{ij}^2, a_{ij}).
        Combines source node features, target node features, radial distance, and edge attributes.
        """
        if edge_attr is None:
            out = jnp.concatenate([source, target, radial], axis=1)
        else:
            out = jnp.concatenate([source, target, radial, edge_attr], axis=1)
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
            agg = unsorted_segment_sum(trans, row, num_segments=coord.shape[0])
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])
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
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([x, agg, node_attr], axis=1)
        else:
            agg = jnp.concatenate([x, agg], axis=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def __call__(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index

        radial, coord_diff = self.coord2radial(edge_index, coord)
        
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class E_GCL_OLD(nn.Module):
    """
    E(n) Equivariant Message Passing Layer
    """

    hidden_nf: int
    act_fn: callable
    velocity: bool = False

    def edge_model(self, edge_index, h, coord, edge_attr):
        row, col = edge_index
        source, target = h[row], h[col]
        radial = self.coord2radial(edge_index, coord)

        edge_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_nf),
                self.act_fn,
                nn.Dense(self.hidden_nf),
                self.act_fn,
            ]
        )

        out = jnp.concatenate([source, target, radial, edge_attr], axis=1)

        return edge_mlp(out)

    def node_model(self, edge_index, edge_attr, x):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])

        node_mlp = nn.Sequential(
            [nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(self.hidden_nf)]
        )

        agg = jnp.concatenate([x, agg], axis=1)
        out = node_mlp(agg)

        # TODO do we need to add x to out? to update it
        return out, agg

    def coord_model(self, edge_index, edge_feat, coord):
        row, col = edge_index
        coord_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_nf),
                self.act_fn,
                nn.Dense(1, kernel_init=xavier_init(gain=0.001)),
            ]
        )

        coord_out = coord_mlp(edge_feat)
        trans = (coord[row] - coord[col]) * coord_out

        agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])

        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        senders, receivers = edge_index
        coord_i, coord_j = coord[senders], coord[receivers]
        distance = jnp.sum((coord_i - coord_j) ** 2, axis=1, keepdims=True)
        return distance

    def coord_vel_model(self, coord, h, vel):
        coord_mlp_vel = nn.Sequential(
            [nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(1)]
        )

        coord += coord_mlp_vel(h) * vel
        return coord

    @nn.compact
    def __call__(self, h, edge_index, coord, vel=None, edge_attr=None):
        m_ij = self.edge_model(edge_index, h, coord, edge_attr)
        h, agg = self.node_model(edge_index, m_ij, h)
        coord = self.coord_model(edge_index, m_ij, coord)
        if self.velocity:
            coord = self.coord_vel_model(coord, h, vel)
        return h, coord, m_ij


class E_GCL_OG(nn.Module):
    """
    E(n) Equivariant Message Passing Layer - Reproduction of initial
    """

    input_nf: int
    output_nf: int
    hidden_nf: int
    edges_in_d: int = 0
    nodes_attr_dim: int = 0
    act_fn: callable = nn.relu
    recurrent: bool = True
    coords_weight: float = 1.0
    attention: bool = False

    def setup(self):
        input_edge = self.input_nf * 2
        self.edge_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
            ]
        )

        self.node_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(
                    self.output_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
            ]
        )

        self.coord_mlp = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_nf, kernel_init=glorot_uniform(), bias_init=uniform()
                ),
                self.act_fn,
                nn.Dense(1, use_bias=False),
            ]
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                [
                    nn.Dense(1, kernel_init=glorot_uniform(), bias_init=uniform()),
                    nn.sigmoid,
                ]
            )

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = jnp.concatenate([source, target, radial], axis=1)
        else:
            edge_attr = edge_attr.reshape(-1, edge_attr.shape[-1])
            out = jnp.concatenate(
                [source, target, radial.reshape(-1, 1), edge_attr], axis=1
            )
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = jax.ops.segment_sum(edge_attr, row, num_segments=x.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([x, agg, node_attr], axis=1)
        else:
            agg = jnp.concatenate([x, agg], axis=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        trans = jnp.clip(trans, -100, 100)
        agg = jax.ops.segment_sum(trans, row, coord.shape[0])
        coord += agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = jnp.sum(coord_diff**2, axis=1, keepdims=True)
        # if self.norm_diff:
        #     norm = jnp.sqrt(radial) + 1
        #     coord_diff = coord_diff / norm
        return radial, coord_diff

    def coord_vel_model(self, coord, h, vel):
        coord_mlp_vel = nn.Sequential(
            [nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(1)]
        )
        coord += coord_mlp_vel(h) * vel
        return coord

    def __call__(
        self,
        h,
        edge_index,
        coord,
        node_mask,
        edge_mask,
        edge_attr=None,
        node_attr=None,
        n_nodes=None,
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network
    """
    in_node_nf: int
    hidden_nf: int
    out_node_nf: int
    in_edge_nf: int
    batch_size: int
    act_fn: callable = nn.silu
    n_layers: int = 4
    robot: str = "h1"
    init_scale: float = 1.0
    residual: bool = True
    attention: bool = False
    normalize: bool = False
    tanh: bool = False
    coords_agg: str = "mean"
    device: str = None

    def setup(self):
        self.embedding_in = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn
        ])
        self.embedding_out = nn.Sequential([
            nn.Dense(self.out_node_nf),
            nn.tanh,
        ])

    def build_batched_egnn_input(self, obs: jnp.ndarray | torch.Tensor):
        """Build EGNN input from observations"""
        if isinstance(obs, torch.Tensor):
            obs_jnparray = jnp.asarray(obs.cpu().numpy())
        else:
            obs_jnparray = obs
        
        # Get forward kinematics positions using pure JAX method
        x = h1_jax_fk._fk_joint_positions_jax(obs_jnparray[:, :26]).reshape(-1, 3)
        x = x[:, 1:]  # Remove first dimension
        x = x - x[:, 0:1]  # Relative to first joint

        # Build node features based on robot type
        if self.robot == "h1":
            h = jnp.concatenate([
                obs_jnparray[:, 32:].reshape(-1, 1), 
                obs_jnparray[:, 7:26].reshape(-1, 1)
            ], axis=1)  # (B*N, 2)
        elif self.robot == "g1":
            h = jnp.concatenate([
                obs_jnparray[:, 59:].reshape(-1, 1), 
                obs_jnparray[:, 7:44].reshape(-1, 1)
            ], axis=1)  # (B*N, 2)
        else:
            raise ValueError(f"Unsupported robot type: {self.robot}")

        return h, x

    def get_edge_index(self, batch_size, robot_info):
        """Get edge indices for the current batch size using robot structure"""
        if self.robot == "h1":
            from fast_td3.robots.h1 import H1
            robot = H1
        elif self.robot == "g1":
            from fast_td3.robots.g1 import G1
            robot = G1
        else:
            raise ValueError(f"Unsupported robot type: {self.robot}")
        
        edge_list = robot.edge_list
        src, dst = zip(*edge_list)  # Unpack edge list into two tuples
        src = jnp.array(src, dtype=jnp.int32)
        dst = jnp.array(dst, dtype=jnp.int32)

        # Create batch offsets and expand edges
        offsets = jnp.arange(batch_size) * robot.num_nodes
        src_batch = (src[None, :] + offsets[:, None]).flatten()
        dst_batch = (dst[None, :] + offsets[:, None]).flatten()

        edge_index = (src_batch, dst_batch)
        return edge_index, robot.num_nodes, robot.num_edges

    @nn.compact
    def __call__(self, obs):
        h, x = self.build_batched_egnn_input(obs)
        
        # Determine dimensions
        batch_size = obs.shape[0]
        
        # Get edge indices and robot info
        edge_index, num_nodes, num_edges = self.get_edge_index(batch_size, None)
        
        # Embedding
        h = self.embedding_in(h)

        # Apply EGCL layers
        for i in range(self.n_layers):
            layer = E_GCL(
                input_nf=self.hidden_nf,
                output_nf=self.hidden_nf,
                hidden_nf=self.hidden_nf,
                edges_in_d=self.in_edge_nf,
                act_fn=self.act_fn,
                residual=self.residual,
                attention=self.attention,
                normalize=self.normalize,
                tanh=self.tanh,
                coords_agg=self.coords_agg,
                name=f'egcl_layer_{i}'
            )
            h, x, _ = layer(h, edge_index, x)

        # Output embedding
        h = self.embedding_out(h)

        # Reshape to (batch_size, num_nodes)
        h = h.reshape(batch_size, num_nodes)

        return h


class EGNN_equiv(nn.Module):
    hidden_nf: int
    out_node_nf: int
    act_fn: callable = nn.relu
    n_layers: int = 4
    velocity: bool = False

    @nn.compact
    def __call__(self, h, x, edges, vel=None, edge_attr=None):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            h, x, _ = E_GCL(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                act_fn=self.act_fn,
                attention=True,
            )(
                h,
                edges,
                x,
                jnp.ones_like(x),
                jnp.ones((edges[0].shape[0], 1)),
                edge_attr=edge_attr,
            )
        h = nn.Dense(self.out_node_nf)(h)
        return h, x


class EGNN_QM9(nn.Module):
    hidden_nf: int
    out_node_nf: int
    act_fn: callable = nn.relu  # default activation function
    n_layers: int = 4
    attention: bool = False

    @nn.compact
    def __call__(self, h, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            h, x, _ = E_GCL_OG(
                self.hidden_nf,
                self.hidden_nf,
                self.hidden_nf,
                act_fn=self.act_fn,
                attention=self.attention,
            )(h, edges, x, node_mask, edge_mask, edge_attr=None)
        h = h * node_mask  # Ensure node_mask is broadcasted correctly
        h = h.reshape(-1, n_nodes, self.hidden_nf)
        h = jnp.sum(h, axis=1)
        h = nn.Dense(self.out_node_nf)(h)
        h = jnp.squeeze(h, axis=-1)  # Squeeze the last dimension like in original repo
        return h, x


def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges[..., None] / charge_scale) ** jnp.arange(charge_power + 1)
    charge_tensor = charge_tensor.reshape(*charges.shape, -1)
    atom_scalars = (one_hot[..., None] * charge_tensor[..., None, :]).reshape(
        charges.shape[0], -1
    )
    return atom_scalars


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = jnp.ones(len(edges[0]) * batch_size, dtype=jnp.float32).reshape(-1, 1)
    edges = [
        jnp.array(edges[0]).astype(jnp.int32),
        jnp.array(edges[1]).astype(jnp.int32),
    ]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [jnp.concatenate(rows), jnp.concatenate(cols)]
    return edges, edge_attr