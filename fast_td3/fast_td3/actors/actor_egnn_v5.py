import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_td3.actors.gnn.egnn_v5 import EGNN_V5
from humanoid_bench.envs.custom_env import get_env_class


class ActorEGNN_V5(nn.Module):
    """
    Actor wrapper for EGNN_V5 with cross-attention.
    
    Improvements over ActorEGNN_V3:
    - Uses EGNN_V5 with cross-attention for object-joint integration
    - Maintains full hidden_nf throughout (no information bottleneck)
    - More parameter efficient with better performance
    """
    
    def __init__(
        self,
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
        attention: bool = False,
        coords_agg: str = "mean",
        normalize: bool = False,
        tanh: bool = False,
        residual: bool = True,
        coord_norm: bool = False,
        num_attn_heads: int = 1,
        attn_dropout: float = 0.0,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            hidden_dim: Hidden dimension for EGNN layers
            batch_size: Batch size for training
            device: Device to place the model on
            n_layers: Number of E_GCL message passing layers
            act_fn: Activation function ('relu', 'silu', 'leaky_relu')
            env_name: Environment name (e.g., 'h1-push-v1')
            robot: Robot type ('h1', 'h1hand', 'g1')
            std_min: Minimum exploration noise scale
            std_max: Maximum exploration noise scale
            attention: Whether to use attention in E_GCL
            coords_agg: Coordinate aggregation method ('mean', 'sum')
            normalize: Whether to normalize coordinates in E_GCL
            tanh: Whether to use tanh in coordinate updates
            residual: Whether to use residual connections
            coord_norm: Whether to apply coordinate normalization
            num_attn_heads: Number of attention heads for cross-attention (default: 4)
            attn_dropout: Dropout probability for cross-attention (default: 0.0)
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
        in_object_nf = object_feature_dim

        # EGNN v5 with cross-attention
        self.egnn = EGNN_V5(
            hidden_nf=hidden_dim,
            in_joint_nf=2,  # velocity + position
            in_object_nf=in_object_nf,  # object features per object
            batch_size=batch_size,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            robot=robot,
            attention=attention,
            coords_agg=coords_agg,
            normalize=normalize,
            tanh=tanh,
            env_name=env_name,
            residual=residual,
            coord_norm=coord_norm,
            num_attn_heads=num_attn_heads,
            attn_dropout=attn_dropout,
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
        Forward pass through EGNN_V5.
        
        Args:
            obs: Flat observation tensor (batch_size, obs_dim)
            
        Returns:
            Actions: (batch_size, num_joints)
        """
        return self.egnn(obs)

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
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        # Get deterministic actions
        act = self(obs)
        
        if deterministic:
            return act

        # Add exploration noise
        noise = torch.randn_like(act) * self.noise_scales
        return torch.clamp(act + noise, -1.0, 1.0)
