"""
Profiling script for EGNN_V3 actor
This script profiles the forward pass and explore methods of the EGNN_V3 actor
"""
import os
import sys
import json
import torch
import argparse
import numpy as np
from datetime import datetime

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from fast_td3.train_utils import create_actor
from fast_td3.environments.humanoid_bench_env import HumanoidBenchEnv


def profile_egnn_v3(
    env_name: str = "h1-balance_simple-v1",
    num_envs: int = 64,
    batch_size: int = 32768,
    model_config_path: str = "model_config/egnn.json",
    num_iterations: int = 1,
    warmup_iterations: int = 100,
    profile_output_dir: str = "./logs/profiler",
    profile_forward: bool = True,
    profile_explore: bool = True,
    use_cuda: bool = True,
):
    """
    Profile EGNN_V3 actor forward and explore passes
    
    Args:
        env_name: Environment name
        num_envs: Number of parallel environments
        batch_size: Batch size for actor
        model_config_path: Path to model configuration JSON
        num_iterations: Number of iterations to profile
        warmup_iterations: Number of warmup iterations before profiling
        profile_output_dir: Directory to save profiler traces
        profile_forward: Whether to profile forward pass
        profile_explore: Whether to profile explore method
        use_cuda: Whether to use CUDA if available
    """
    
    # Setup device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(profile_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model config
    with open(model_config_path, "r") as f:
        model_kwargs = json.load(f)
    
    print(f"Model config: {model_kwargs}")
    
    # Create environment to get observation space
    env = HumanoidBenchEnv(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
    )
    
    n_act = env.num_actions
    n_obs = env.num_obs if type(env.num_obs) == int else env.num_obs[0]
    
    # Create actor
    print(f"Creating EGNN_V3 actor for {env_name}...")
    actor = create_actor(
        actor_type="egnn_v3",
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        batch_size=batch_size,
        device=device,
        model_kwargs=model_kwargs,
        env_name=env_name,
        init_scale=0.01,
    )
    actor = actor.to(device)
    actor.eval()
    
    print(f"Actor created with {sum(p.numel() for p in actor.parameters())} parameters")
    
    # Generate random observation
    obs = torch.randn(num_envs, n_obs, device=device)
    dones = torch.zeros(num_envs, device=device, dtype=torch.bool)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Device: {device}")
    
    # Warmup
    print(f"\nWarming up for {warmup_iterations} iterations...")
    with torch.no_grad():
        for i in range(warmup_iterations):
            if profile_forward:
                _ = actor(obs)
            if profile_explore:
                _ = actor.explore(obs, dones=dones, deterministic=False)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    print("Warmup complete.")
    
    # Profile forward pass
    if profile_forward:
        print(f"\n{'='*60}")
        print(f"Profiling forward pass for {num_iterations} iterations...")
        print(f"{'='*60}")
        
        forward_output_path = os.path.join(
            profile_output_dir, 
            f"egnn_v3_forward_{timestamp}"
        )
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(forward_output_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                for i in range(num_iterations):
                    _ = actor(obs)
                    prof.step()
        
        print(f"\nForward pass profiling complete. Traces saved to: {forward_output_path}")
        print("\nKey averages (grouped by operator):")
        print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))
        
        # Save detailed report
        with open(os.path.join(profile_output_dir, f"forward_report_{timestamp}.txt"), "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=50))
    
    # Profile explore method
    if profile_explore:
        print(f"\n{'='*60}")
        print(f"Profiling explore method for {num_iterations} iterations...")
        print(f"{'='*60}")
        
        explore_output_path = os.path.join(
            profile_output_dir,
            f"egnn_v3_explore_{timestamp}"
        )
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(explore_output_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                for i in range(num_iterations):
                    # Randomly mark some environments as done
                    dones = torch.rand(num_envs, device=device) < 0.1
                    _ = actor.explore(obs, dones=dones, deterministic=False)
                    prof.step()
        
        print(f"\nExplore method profiling complete. Traces saved to: {explore_output_path}")
        print("\nKey averages (grouped by operator):")
        print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))
        
        # Save detailed report
        with open(os.path.join(profile_output_dir, f"explore_report_{timestamp}.txt"), "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=50))
    
    print(f"\n{'='*60}")
    print("Profiling complete!")
    print(f"{'='*60}")
    print(f"View results with: tensorboard --logdir={profile_output_dir}")
    print(f"Text reports saved in: {profile_output_dir}")
    

def main():
    parser = argparse.ArgumentParser(description="Profile EGNN_V3 actor")
    parser.add_argument(
        "--env_name",
        type=str,
        default="h1-balance_simple-v1",
        help="Environment name",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32768,
        help="Batch size for actor",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config/egnn.json",
        help="Path to model configuration JSON",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of iterations to profile",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=100,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/profiler",
        help="Directory to save profiler traces",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Only profile forward pass",
    )
    parser.add_argument(
        "--explore_only",
        action="store_true",
        help="Only profile explore method",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Profile on CPU only",
    )
    
    args = parser.parse_args()
    
    # Determine what to profile
    profile_forward = not args.explore_only
    profile_explore = not args.forward_only
    
    profile_egnn_v3(
        env_name=args.env_name,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        model_config_path=args.model_config,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations,
        profile_output_dir=args.output_dir,
        profile_forward=profile_forward,
        profile_explore=profile_explore,
        use_cuda=not args.cpu_only,
    )


if __name__ == "__main__":
    main()
