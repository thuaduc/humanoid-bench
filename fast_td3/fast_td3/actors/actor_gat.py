import torch
import torch.nn as nn

from fast_td3.actors.gnn.gat import GAT
from humanoid_bench.envs.custom_env import get_env_class


class ActorGAT(nn.Module):
    """
    Actor wrapper for GAT with self-attention and concat + MLP architecture.
    
    Architecture:
    - Joint embedding + GAT layers (self-attention)
    - Embed down joints to lower dimension
    - Concatenate flattened joints with object features
    - Large MLP action head
    """
    
    def __init__(
        self,
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        batch_size: int,
        device: torch.device,
        n_layers: int,
        act_fn: str,
        env_name: str,
        robot: str = "h1",
        std_min: float = 0.05,
        std_max: float = 0.8,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        joint_out_dim: int = 2,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            hidden_dim: Hidden dimension for GAT layers
            batch_size: Batch size for training
            device: Device to place the model on
            n_layers: Number of GAT layers
            act_fn: Activation function ('relu', 'silu', 'leaky_relu')
            env_name: Environment name (e.g., 'h1-push-v1')
            robot: Robot type ('h1', 'h1hand', 'g1')
            std_min: Minimum exploration noise scale
            std_max: Maximum exploration noise scale
            num_heads: Number of attention heads for GAT layers
            dropout: Dropout probability for attention
            residual: Whether to use residual connections
            joint_out_dim: Embedding dimension before concatenation
        """
        super().__init__()
        
        # Parse activation function
        match act_fn:
            case "leaky_relu":
                act_fn = nn.LeakyReLU()
            case "silu":
                act_fn = nn.SiLU()
            case "relu":
                act_fn = nn.ReLU()
            case _:
                raise ValueError(f"Unknown activation function: {act_fn}")

        # Determine object feature dimensions
        env_class = get_env_class(env_name)
        object_feature_dim = env_class.get_object_feature_dim()

        # GAT network
        self.gat = GAT(
            n_act=n_act,
            in_joint_nf=5,  # [pos, vel, x, y, z]
            hidden_nf=hidden_dim,
            object_feature_dim=object_feature_dim,
            n_layers=n_layers,
            num_heads=num_heads,
            dropout=dropout,
            act_fn=act_fn,
            device=device,
            batch_size=batch_size,
            env_name=env_name,
            robot=robot,
            residual=residual,
            joint_out_dim=joint_out_dim,
        )

        # Initialize exploration noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAT.
        
        Args:
            obs: Flat observation tensor (batch_size, obs_dim)
            
        Returns:
            Actions: (batch_size, num_joints)
        """
        return self.gat(obs)

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Exploration policy with adaptive noise.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            dones: Done flags for resampling noise (batch_size,)
            deterministic: If True, return actions without noise
            
        Returns:
            Actions with exploration noise: (batch_size, num_joints)
        """
        # Resample noise for environments that are done
        if dones is not None:
            # Generate new noise scales for done environments
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        # Get deterministic actions
        act = self(obs)
        
        if deterministic:
            return act

        # Add exploration noise
        noise = torch.randn_like(act) * self.noise_scales
        return act + noise
