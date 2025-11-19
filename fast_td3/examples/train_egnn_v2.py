"""
Example script demonstrating how to train with EGNN v2 actor.

This script shows how to use the new EGNN v2 architecture for training
a humanoid robot control policy.
"""

# Example 1: Train with EGNN v2 on H1 robot
# ------------------------------------------
# Basic training command:
"""
cd fast_td3
python train.py \
    --actor egnn_v2 \
    --env_name h1-stand-v0 \
    --num_envs 16 \
    --batch_size 8192 \
    --total_timesteps 100000 \
    --wandb
"""

# Example 2: Train with custom model configuration
# -------------------------------------------------
# First create a JSON config file: model_configs/egnn_v2_custom.json
"""
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
"""

# Then run:
"""
cd fast_td3
python train.py \
    --actor egnn_v2 \
    --env_name h1-push-v0 \
    --model_kwargs model_configs/egnn_v2_custom.json \
    --num_envs 32 \
    --batch_size 16384 \
    --total_timesteps 200000
"""

# Example 3: Using the general training script
# ---------------------------------------------
"""
cd fast_td3/fast_td3
python train.py \
    --env_name h1-stand-v0 \
    --actor_type egnn_v2 \
    --actor_n_layers 3 \
    --actor_act_fn silu \
    --robot h1 \
    --num_envs 128 \
    --batch_size 32768
"""

# Example 4: Programmatic usage
# ------------------------------
if __name__ == "__main__":
    """
    Example of using EGNN v2 programmatically.
    Note: This requires PyTorch and other dependencies to be installed.
    """
    try:
        import torch
        from fast_td3.actors import ActorEGNN_V2
        
        # Create the actor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        actor = ActorEGNN_V2(
            num_envs=128,
            hidden_dim=256,
            batch_size=32,
            device=device,
            n_layers=3,
            act_fn="silu",
            env_name="h1-stand-v0",
            robot="h1",
            std_min=0.001,
            std_max=0.4,
        )
        
        print(f"Created ActorEGNN_V2 with {sum(p.numel() for p in actor.parameters())} parameters")
        
        # Example forward pass
        batch_size = 2
        obs = torch.randn(batch_size, 51, device=device)
        xanchor = torch.randn(batch_size, 20, 3, device=device)
        
        # Get deterministic actions
        actions = actor(obs, xanchor)
        print(f"Actions shape: {actions.shape}")  # Should be [batch_size, 19]
        
        # Get stochastic actions with exploration
        actions_explore = actor.explore(obs, xanchor, deterministic=False)
        print(f"Exploration actions shape: {actions_explore.shape}")
        
        print("✅ ActorEGNN_V2 is working correctly!")
        
    except ImportError as e:
        print(f"⚠️  Dependencies not installed: {e}")
        print("To run this example, install the required packages:")
        print("  cd fast_td3 && pip install -r requirements/requirements.txt")
