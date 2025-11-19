# EGNN v2 Actor Implementation

This document describes the implementation of EGNN v2 with cross-graph aggregation for humanoid control.

## Overview

EGNN v2 implements a two-graph architecture that processes robot joints and environment objects separately before aggregating information across graphs. This follows the architecture described in Section 3.3 of the research paper.

## Architecture

### 1. Disjoint Graph Construction

The architecture maintains two separate graphs:

- **Joint Graph**: Represents robot joints with their connections
- **Object Graph**: Represents objects/root information in the environment

### 2. Message Passing (Equations 5-7)

Both graphs undergo L rounds (typically 3-5) of E(n)-Equivariant Graph Convolution:

```
mij = φe(h_i^l, h_j^l, ||x_i^l - x_j^l||^2)                    (Eq. 5)
x_i^(l+1) = x_i^l + Σ_{j∈N(i)} (x_i^l - x_j^l) φx(mij)        (Eq. 6)
h_i^(l+1) = h_i^l + φh(h_i^l, Σ_{j∈N(i)} mij)                 (Eq. 7)
```

### 3. Cross-Graph Aggregation (Equations 8-9)

After local processing, object information is pooled to joint nodes:

```
h_joint_pooled_i = h_i^L + φ_cross(h_i^L, Σ_{j∈V_object} m_obj→joint_ij)    (Eq. 8)
x_joint_pooled_i = x_i^L + Σ_{j∈V_object} (x_i^L - x_j^L) φx(m_obj→joint_ij)  (Eq. 9)
```

where:
```
m_obj→joint_ij = φe(h_i^L, h_j^L, ||x_i^L - x_j^L||^2)
```

## Implementation Details

### Node Indexing

Node indices follow a batch-aware pattern to ensure proper separation:

- **Batch 0 joints**: indices 0-18 (for H1 robot with 19 joints)
- **Batch 1 joints**: indices 19-37
- **Batch 0 object**: conceptually index 38 (handled separately in implementation)
- **Batch 1 object**: conceptually index 39 (handled separately in implementation)

In the actual implementation, object nodes are not given explicit global indices. Instead, they're maintained as a separate tensor `[batch, features]` and broadcasting is used during cross-graph aggregation.

### Key Components

#### 1. GraphBuilder.generate_input_v2()

Located in: `fast_td3/fast_td3/robots/graph_builder.py`

Processes observations and anchor points to create separate feature tensors:

```python
h_joints, x_joints, h_objects, x_objects = graph_builder.generate_input_v2(obs, xanchor)
```

**Returns:**
- `h_joints`: Joint node features [batch*num_joints, 2] (velocity, position)
- `x_joints`: Joint node coordinates [batch*num_joints, 3]
- `h_objects`: Object node features [batch, 6] (root height, orientation, velocities)
- `x_objects`: Object node coordinates [batch, 3]

#### 2. EGNN_V2 Class

Located in: `fast_td3/fast_td3/actors/gnn/egnn.py`

Main model implementing the dual-graph architecture:

**Components:**
- `joint_layers`: ModuleList of E_GCL layers for joint graph processing
- `object_embedding`: MLP for embedding object features
- `cross_edge_mlp`: MLP for computing cross-graph edge messages
- `cross_node_mlp`: MLP for updating joint features with aggregated object information
- `cross_coord_mlp`: MLP for computing coordinate update weights

**Key Methods:**
- `forward()`: Main forward pass implementing the full algorithm
- `cross_graph_aggregation()`: Implements Equations 8-9 for object→joint pooling
- `generate_index()`: Creates edge indices for the joint graph
- `get_cached_edges()`: Caches edge indices for efficiency

#### 3. ActorEGNN_V2 Class

Located in: `fast_td3/fast_td3/actors/actor_egnn.py`

Actor wrapper around EGNN_V2 with exploration noise:

```python
actor = ActorEGNN_V2(
    num_envs=128,
    hidden_dim=256,
    batch_size=32,
    device=device,
    n_layers=3,
    act_fn="silu",
    env_name="h1-stand-v0",
    robot="h1",
)
```

## Usage

### Training with EGNN v2

From the command line:

```bash
cd fast_td3
python train.py \
    --actor egnn_v2 \
    --env_name h1-stand-v0 \
    --num_envs 16 \
    --batch_size 8192 \
    --total_timesteps 100000
```

With custom model parameters:

```bash
python train.py \
    --actor egnn_v2 \
    --env_name h1-push-v0 \
    --model_kwargs model_configs/egnn_v2_config.json
```

Example `model_configs/egnn_v2_config.json`:
```json
{
    "hidden_dim": 256,
    "n_layers": 4,
    "act_fn": "silu",
    "attention": false,
    "coords_agg": "mean",
    "normalize": false,
    "tanh": false,
    "std_min": 0.001,
    "std_max": 0.4
}
```

### Using the Alternative Train Script

For the general-purpose training script that supports multiple environment types:

```bash
cd fast_td3/fast_td3
python train.py \
    --env_name h1-stand-v0 \
    --actor_type egnn_v2 \
    --actor_n_layers 3 \
    --actor_act_fn silu \
    --robot h1
```

## Differences from EGNN v1

| Feature | EGNN v1 | EGNN v2 |
|---------|---------|---------|
| Graph Structure | Single joint graph | Dual graphs (joints + objects) |
| Object Handling | Objects as global context | Objects as separate graph nodes |
| Message Passing | Joint graph only | Joint graph + cross-graph aggregation |
| Equivariance | E(3) for joint coords | E(3) for both joint and object coords |
| Parameters | `generate_input()` | `generate_input_v2()` |

## E(3)-Equivariance Properties

The architecture preserves E(3)-equivariance (rotation and translation) through:

1. **Coordinate Differences**: Using `x_i - x_j` instead of absolute positions
2. **Squared Distances**: Computing `||x_i - x_j||^2` which is rotation-invariant
3. **Directional Updates**: Updating coordinates along direction vectors: `(x_i - x_j) * scalar`

## Performance Considerations

- **Edge Caching**: Edge indices are cached per batch size for efficiency
- **Batch Processing**: All operations are batch-aware to minimize overhead
- **Torch Compile**: The `generate_input_v2` method uses `@torch.compile(dynamic=True)` for JIT optimization
- **Cross-Graph Efficiency**: Broadcasting is used to avoid explicit object node index management

## Testing

Run the test suite:

```bash
cd fast_td3
python -m unittest test.test_egnn_v2 -v
```

Or run individual test classes:

```bash
python test/test_egnn_v2.py TestEGNN_V2
```

## Future Extensions

Potential improvements and extensions:

1. **Multi-Object Support**: Extend to handle multiple objects per scene
2. **Object-Object Interactions**: Add message passing within the object graph
3. **Attention Mechanisms**: Add cross-attention between joint and object nodes
4. **Hierarchical Graphs**: Create hierarchical joint representations (e.g., limbs)
5. **Temporal Information**: Incorporate temporal message passing across time steps

## References

- Section 3.3: Message Passing and Global Integration (from the research paper)
- E(n) Equivariant Graph Neural Networks paper
- Original EGNN implementation in `fast_td3/fast_td3/actors/gnn/egnn.py`
