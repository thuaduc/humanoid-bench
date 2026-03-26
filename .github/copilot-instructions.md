# FastTD3 HumanoidBench AI Coding Instructions

## Project Overview
**HumanoidBench** is a simulated humanoid robot benchmark for whole-body locomotion and manipulation tasks using MuJoCo physics simulation, with 27 tasks across 6 robot variants.

**Main Goal**: Optimize EGNN (Equivariant Graph Neural Network) policy for custom environment tasks (v1 variants with alternative locomotion implementations) using **FastTD3** algorithm.

## Architecture

### Core Components (HumanoidBench)
- **`humanoid_bench/`** — Core benchmark package
  - `env.py` — `HumanoidEnv` (Gymnasium wrapper around MuJoCo) and task registration system
  - `robots.py` — Robot classes (H1, H1Hand, G1) with sensor access methods
  - `tasks.py` — Base `Task` class defining reward/termination logic
  - `wrappers.py` — Wrapper classes for hierarchical control (reach + manipulation)
  - `envs/` — 27 task implementations (Walk, Reach, Push, Cube, Kitchen, etc.)

### FastTD3 Components
- **`fast_td3/`** — High-performance TD3 algorithm optimized for humanoid control
  - `train.py` — Main training loop
  - `fast_td3.py` — Actor/Critic network definitions
  - `fast_td3_utils.py` — `EmpiricalNormalization`, `SimpleReplayBuffer`
  - `hyperparams.py` — Per-task hyperparameters (learning rates, buffer size, action clip)

### Data Flow (FastTD3 Training)
1. User registers env via `gymnasium.make(f"{robot}-{task}-v0")` or `gymnasium.make(f"{robot}-{task}-v1")` (custom_env)
2. `HumanoidBenchEnv` wrapper in `fast_td3/environments/` creates vectorized parallel environments
3. FastTD3 actor samples actions, critic evaluates Q-values
4. Experience collected into `SimpleReplayBuffer`
5. Models updated via PyTorch with optional AMP (mixed precision) and torch.compile optimization

## Key Patterns

### Task System (Plugin Architecture)
Tasks are registered in `_TASKS_ORIGINAL` dict and inherit from `Task` base class:
```python
class Task:
    def get_obs(self): return np.concatenate((qpos, qvel))
    def get_reward(self): return reward, info_dict
    def get_terminated(self): return done, info_dict
    def normalize_action(self): # [-1, 1] → action bounds
```
**Adding new tasks**: Create `humanoid_bench/envs/my_task.py` → add class → register in `env.py` dicts.

### Robot Hierarchy
`H1` base class provides sensor abstractions; `H1Hand`/`H1SimpleHand` extend with hand DOFs (76/52 vs 26 base).
Access via `env.robot.joint_angles()`, `env.robot.left_hand_position()` — these query MuJoCo's named data accessors.

### Environment Registration
`__init__.py` loops robot/task combinations and registers `gym.make()` compatible IDs:
- Pattern: `{robot}-{task}-v{version}` (e.g., `h1hand-reach-v0`)
- All register via `HumanoidEnv` entry point with task/robot kwargs
- v0 = original tasks, v1 = custom_env variants (alternative locomotion implementations)

## Developer Workflows

**Environment Setup**: Always use the `fasttd3_hb` conda environment for development and testing:
```bash
conda activate fasttd3_hb
```

### Running Environments
```bash
python -m train --actor egnn_v2 --env_name h1-balance_simple-v1 --total_timesteps 1_000_001 --model_kwargs model_config/egnn.json --num_envs 64
```

### Key Files for Experimentation
- `fast_td3/hyperparams.py` — Task hyperparameters (learning rates, replay buffer, action clip)
- `fast_td3/fast_td3_utils.py` — `EmpiricalNormalization`, `SimpleReplayBuffer`
- `humanoid_bench/env.py` lines 350+ — Environment registration section
