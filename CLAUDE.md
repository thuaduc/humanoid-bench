# HumanoidBench — Claude Code Guide

## Project Overview

A simulated humanoid robot benchmark with **27 whole-body locomotion and manipulation tasks** built on MuJoCo. The primary training algorithm is **FastTD3** (custom implementation) located in `fast_td3/`.

## Research Goal

Current FastTD3 actor networks (`fast_td3/fast_td3/actors/actor.py`) use **MLPs** that process robot observations as flat, unstructured vectors — ignoring the robot's physical topology. This causes:

- Implicit learning of spatial relationships (inductive bias mismatch)
- High sample complexity
- Poor generalization to unseen robot configurations

**The core research goal is to incorporate the robot's physical topology into the policy architecture.** Two approaches are being explored:

| Approach | Description |
|----------|-------------|
| **Transformer** | Attention over robot body parts/joints |
| **EGNN** | Equivariant GNN respecting physical graph structure |

> ⚠️ **Keep this goal in mind for every task.** All changes should serve the aim of topology-aware policies.

## Scope

We work **exclusively** on custom environments:

- `humanoid_bench/envs/custom_env_h1hand.py`
- `humanoid_bench/envs/custom_env.py`

Do not modify standard HumanoidBench envs or unrelated benchmark tasks.

## Key Files

| Path | Purpose |
|------|---------|
| `fast_td3/fast_td3/actors/actor.py` | MLP actor — primary target for architecture changes |
| `humanoid_bench/envs/custom_env.py` | Custom env (no hand) |
| `humanoid_bench/envs/custom_env_h1hand.py` | Custom env (with hand) |

## Repository Layout
```
humanoid_bench/      # Core benchmark: environments, assets, wrappers
fast_td3/            # Primary training code (FastTD3 + actor architectures)
  fast_td3/
    actors/          # Actor network architectures (MLP, GNN, EGNN, Transformer)
    environments/    # Env wrappers for HumanoidBench
    model_config/    # JSON hyperparameter configs per architecture
    fast_td3.py      # Core TD3 algorithm
    train.py         # Training entry point (also at fast_td3/train.py)
    hyperparams.py   # Argument parsing & defaults
data/                # Pre-trained low-level skill policy weights
dreamerv3/           # DreamerV3 training code
jaxrl_m/             # SAC training code (JAX)
ppo/                 # PPO training code (stable-baselines3)
tdmpc2/              # TD-MPC2 training code
logs/                # Local training logs
```

## Running Training (FastTD3)
```bash
cd fast_td3
python train.py --env_name h1hand-walk-v0 --actor_type transformer
```
Key flags defined in `fast_td3/hyperparams.py`. Model architecture configs live in `fast_td3/model_config/`.

## Actor Architectures
| File | Type |
|------|------|
| `actor.py` | MLP baseline |
| `actor_egnn.py` / `actor_egnn_v3.py` / `actor_egnn_v5.py` | Equivariant GNN variants |
| `actor_transformer.py` | Transformer v1 |
| `actor_transformer_v2.py` | Transformer v2 |

## Evaluation Criterion

**A better actor reaches the target `eval_avg_return` with the least wall-clock runtime.**

- Primary metric: `eval_avg_return` logged by W&B
- Secondary metric: wall-clock time to reach the target (from W&B `_runtime`)
- Do **not** compare raw returns across runs with different step budgets — normalize by runtime

### Target Returns per Task

| Task                | Target  | Task                | Target  | Task                | Target  |
|---------------------|---------|---------------------|---------|---------------------|---------|
| walk                | 700.0   | stand               | 800.0   | run                 | 700.0   |
| reach               | 12000.0 | hurdle              | 700.0   | crawl               | 700.0   |
| maze                | 1200.0  | sit_simple          | 750.0   | sit_hard            | 750.0   |
| balance_simple      | 800.0   | balance_hard        | 800.0   | stair               | 700.0   |
| slide               | 700.0   | pole                | 700.0   | push                | 700.0   |
| cabinet             | 2500.0  | highbar             | 750.0   | door                | 600.0   |
| truck               | 3000.0  | cube                | 370.0   | bookshelf_simple    | 2000.0  |
| bookshelf_hard      | 2000.0  | basketball          | 1200.0  | window              | 650.0   |
| spoon               | 650.0   | kitchen             | 4.0     | package             | 1500.0  |
| powerlift           | 800.0   | room                | 400.0   | insert_small        | 350.0   |
| insert_normal       | 350.0   |                     |         |                     |         |

## W&B Integration
- Entity: `thuaduc24042001-technical-university-of-munich`
- Active project: **Benchmark New** (created 2026-03-12)
- Other benchmark projects: `Benchmark final`, `Benchmark`, `HB - benchmark`
- Runs log via `wandb` in `fast_td3/train.py`
