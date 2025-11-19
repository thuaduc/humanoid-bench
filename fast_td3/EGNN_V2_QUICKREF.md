# EGNN v2 Quick Reference

## Training Commands

### Basic Usage

```bash
# Train EGNN v2 on H1-stand environment
cd fast_td3
python train.py --actor egnn_v2 --env_name h1-stand-v0

# Train with more environments for faster training
python train.py --actor egnn_v2 --env_name h1-walk-v0 --num_envs 32

# Train on an environment with objects (uses cross-graph aggregation)
python train.py --actor egnn_v2 --env_name h1-push-v0 --num_envs 16
```

### Using Alternative Training Script

```bash
# More flexible training script with additional options
cd fast_td3/fast_td3
python train.py \
    --env_name h1-stand-v0 \
    --actor_type egnn_v2 \
    --actor_n_layers 3 \
    --actor_act_fn silu \
    --robot h1
```

## Model Configuration

### Default Parameters

- `hidden_dim`: 256
- `n_layers`: 3
- `act_fn`: "silu"
- `attention`: False
- `coords_agg`: "mean"
- `normalize`: False
- `tanh`: False
- `std_min`: 0.001
- `std_max`: 0.4

### Custom Configuration

Create a JSON file (e.g., `model_configs/egnn_v2_large.json`):

```json
{
    "hidden_dim": 512,
    "n_layers": 5,
    "act_fn": "silu",
    "attention": true,
    "coords_agg": "mean",
    "std_min": 0.01,
    "std_max": 0.5
}
```

Then use it:

```bash
python train.py --actor egnn_v2 --env_name h1-push-v0 --model_kwargs model_configs/egnn_v2_large.json
```

## Actor Types Comparison

| Feature | MLP | EGNN | EGNN v2 |
|---------|-----|------|---------|
| Input | Flat obs | Obs + xanchor | Obs + xanchor |
| Structure | Feedforward | Single graph | Dual graph |
| Object handling | Concatenated | Global features | Separate graph |
| Message passing | None | Joint graph | Joint + cross-graph |
| Equivariance | No | E(3) | E(3) |
| Parameters | Lowest | Medium | Higher |

## Common Environments

### Without Objects
- `h1-stand-v0`
- `h1-walk-v0`
- `h1-reach-v0`
- `h1-run-v0`
- `h1-sit_simple-v0`

### With Objects (benefits most from EGNN v2)
- `h1-push-v0`
- `h1-basketball-v0`
- `h1-package-v0`
- `h1-balance_simple-v0`

## Troubleshooting

### Out of Memory

Reduce batch size or number of environments:
```bash
python train.py --actor egnn_v2 --env_name h1-stand-v0 --num_envs 8 --batch_size 4096
```

### Slow Training

Increase batch size and number of environments (if GPU memory allows):
```bash
python train.py --actor egnn_v2 --env_name h1-stand-v0 --num_envs 32 --batch_size 16384
```

### Unstable Training

Reduce learning rate and increase std_min:
```json
{
    "std_min": 0.05,
    "std_max": 0.4
}
```

## File Locations

- Model implementation: `fast_td3/fast_td3/actors/gnn/egnn.py` (EGNN_V2 class)
- Actor wrapper: `fast_td3/fast_td3/actors/actor_egnn.py` (ActorEGNN_V2 class)
- Graph builder: `fast_td3/fast_td3/robots/graph_builder.py` (generate_input_v2 method)
- Training script: `fast_td3/train.py`
- Tests: `fast_td3/test/test_egnn_v2.py`

## Architecture Details

See [EGNN_V2_README.md](EGNN_V2_README.md) for full documentation.
