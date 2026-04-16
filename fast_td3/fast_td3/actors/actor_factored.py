"""
Factored Actor with Structural Kinematic Priors

This actor implements three zero-overhead structural priors:
1. Observation-space structure: Groups observations by kinematic chain
2. Weight sharing via bilateral symmetry: Left/right limbs use mirrored weights
3. Factored policy with shared encoders: Each limb type has a shared encoder

The architecture is:
- Root encoder (pelvis + global state)
- Shared leg encoder (processes left/right legs with weight sharing)
- Shared arm encoder (processes left/right arms with weight sharing)  
- Shared hand encoder (processes left/right hands with weight sharing)
- Global MLP trunk that combines all encoded features → actions
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from humanoid_bench.envs.custom_env import get_object_feature_dim


class LimbEncoder(nn.Module):
    """Small MLP encoder for a single limb type (leg, arm, hand)"""
    def __init__(self, input_dim: int, hidden_dim: int, device: torch.device):
        super().__init__()
        # Single layer for minimal overhead
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorFactored(nn.Module):
    """
    Factored actor with kinematic structure priors.
    
    Uses weight sharing across bilateral symmetric limbs and factored encoding
    to inject structural inductive bias with minimal overhead.
    """
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = None,
        robot: str = "h1hand",  # h1hand or h1
        env_name: str = "h1hand-walk-v0",
    ):
        super().__init__()
        self.n_act = n_act
        self.robot = robot
        self.device = device
        self.env_name = env_name
        
        # Get object feature dimension from environment (dynamic)
        self.in_object_nf = get_object_feature_dim(env_name)
        
        # Determine number of joints based on robot type
        self.num_joints = 69 if "h1hand" in env_name else 19
        
        # Define encoders for each limb type (sized to match ~350K total parameters)
        root_dim = self.in_object_nf
        leg_dim = 5 * 5  # 5 joints per leg × 5 features each
        arm_dim = 5 * 5  # 5 joints per arm × 5 features each
        hand_dim = 24 * 5  # 24 joints per Shadow Hand × 5 features each
        torso_dim = 5

        root_hidden = 128
        leg_hidden = 128
        arm_hidden = 128
        hand_hidden = 64
        torso_hidden = 32
        
        self.root_encoder = LimbEncoder(root_dim, root_hidden, device)
        self.leg_encoder = LimbEncoder(leg_dim, leg_hidden, device)
        self.arm_encoder = LimbEncoder(arm_dim, arm_hidden, device)
        self.hand_encoder = LimbEncoder(hand_dim, hand_hidden, device)
        self.torso_encoder = LimbEncoder(torso_dim, torso_hidden, device)  # 1 joint × 5 features
        
        concat_dim = root_hidden + torso_hidden + leg_hidden * 2 + arm_hidden * 2 + hand_hidden * 2
        trunk_hidden = concat_dim // 2
        
        self.action_head = nn.Sequential(
            nn.Linear(concat_dim, trunk_hidden, device=device),
            nn.ReLU(),
            nn.Linear(trunk_hidden, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.action_head[2].weight, 0.0, init_scale)
        nn.init.constant_(self.action_head[2].bias, 0.0)
        
        # Noise parameters
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with factored encoding and weight sharing.
        
        Args:
            obs: [batch_size, obs_dim] observation tensor
        
        Returns:
            [batch_size, n_act] action tensor
        """
        batch_size = obs.shape[0]
        
        # Extract root features - use dynamic dimension
        root_encoded = self.root_encoder(obs[:, :self.in_object_nf])
        
        # Reshape joint features once: [batch, num_joints, 5]
        joint_features = obs[:, self.in_object_nf:].view(batch_size, self.num_joints, 5)
        
        # Extract and encode torso (single joint at index 10)
        torso_encoded = self.torso_encoder(joint_features[:, 10:11, :].reshape(batch_size, -1))
        
        # Legs: Extract and batch process (with mirroring for right leg)
        left_leg = joint_features[:, 0:5, :].reshape(batch_size, -1)  # [batch, 25]
        right_leg = joint_features[:, 5:10, :].clone().reshape(batch_size, 5, 5)  # [batch, 5, 5]
        right_leg[:, :, 3] = -right_leg[:, :, 3]  # Mirror y-component
        right_leg = right_leg.reshape(batch_size, -1)  # [batch, 25]
        
        legs_batch = torch.stack([left_leg, right_leg], dim=1)  # [batch, 2, 25]
        legs_encoded = self.leg_encoder(legs_batch.reshape(batch_size * 2, -1)).view(batch_size, 2, -1)
        
        # Arms: Extract and batch process (with mirroring for right arm)
        # Left arm: observation 11-15 (qpos 18-22: shoulder_pitch to wrist_yaw, 5 joints)
        # Right arm: observation 40-44 (qpos 47-51: shoulder_pitch to wrist_yaw, 5 joints)
        left_arm = joint_features[:, 11:16, :].reshape(batch_size, -1)  # [batch, 25]
        right_arm = joint_features[:, 40:45, :].clone().reshape(batch_size, 5, 5)  # [batch, 5, 5]
        right_arm[:, :, 3] = -right_arm[:, :, 3]  # Mirror y-component
        right_arm = right_arm.reshape(batch_size, -1)  # [batch, 25]
        
        arms_batch = torch.stack([left_arm, right_arm], dim=1)  # [batch, 2, 25]
        arms_encoded = self.arm_encoder(arms_batch.reshape(batch_size * 2, -1)).view(batch_size, 2, -1)
        
        # Shadow Hands: Extract and batch process (with mirroring for right hand)
        # Left hand: observation 16-39 (qpos 23-46, 24 joints)
        # Right hand: observation 45-68 (qpos 52-75, 24 joints)
        left_hand = joint_features[:, 16:40, :].reshape(batch_size, -1)  # [batch, 120] (24 joints * 5)
        right_hand = joint_features[:, 45:69, :].clone().reshape(batch_size, 24, 5)  # [batch, 24, 5]
        right_hand[:, :, 3] = -right_hand[:, :, 3]  # Mirror y-component
        right_hand = right_hand.reshape(batch_size, -1)  # [batch, 120]
        
        hands_batch = torch.stack([left_hand, right_hand], dim=1)  # [batch, 2, 120]
        hands_encoded = self.hand_encoder(hands_batch.reshape(batch_size * 2, -1)).view(batch_size, 2, -1)
        
        # Concatenate: [root, torso, left_leg, right_leg, left_arm, right_arm, left_hand, right_hand]
        x = torch.cat([
            root_encoded,
            torso_encoded,
            legs_encoded[:, 0, :],
            legs_encoded[:, 1, :],
            arms_encoded[:, 0, :],
            arms_encoded[:, 1, :],
            hands_encoded[:, 0, :],
            hands_encoded[:, 1, :],
        ], dim=-1)
        
        return self.action_head(x)
    
    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        """Exploration with noise injection"""
        if dones is not None:
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )
        
        act = self(obs)
        if deterministic:
            return act
        
        noise = torch.randn_like(act) * self.noise_scales
        return act + noise
