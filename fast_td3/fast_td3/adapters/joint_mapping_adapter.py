"""Joint mapping adapter for cross-embodiment transfer."""

import torch
from typing import Tuple


class JointMappingAdapter:
    """Simple adapter: extract H1 joints from G1, zero-pad missing joints."""
    
    # H1 -> G1 joint index mapping
    H1_TO_G1 = {
        0: 2,   # left_hip_yaw -> left_hip_yaw
        1: 1,   # left_hip_roll -> left_hip_roll
        2: 0,   # left_hip_pitch -> left_hip_pitch
        3: 3,   # left_knee -> left_knee
        4: 4,   # left_ankle -> left_ankle_pitch
        5: 8,   # right_hip_yaw -> right_hip_yaw
        6: 7,   # right_hip_roll -> right_hip_roll
        7: 6,   # right_hip_pitch -> right_hip_pitch
        8: 9,   # right_knee -> right_knee
        9: 10,  # right_ankle -> right_ankle_pitch
        10: 12, # torso -> torso
        11: 13, # left_shoulder_pitch -> left_shoulder_pitch
        12: 14, # left_shoulder_roll -> left_shoulder_roll
        13: 15, # left_shoulder_yaw -> left_shoulder_yaw
        14: 16, # left_elbow -> left_elbow_pitch
        15: 25, # right_shoulder_pitch -> right_shoulder_pitch
        16: 26, # right_shoulder_roll -> right_shoulder_roll
        17: 27, # right_shoulder_yaw -> right_shoulder_yaw
        18: 28, # right_elbow -> right_elbow_pitch
    }
    
    def __init__(self, source_robot: str, target_robot: str, device: torch.device, trainable_unmapped: bool = False):
        self.source_robot = source_robot.lower()
        self.target_robot = target_robot.lower()
        self.device = device
        self.mapping = self.H1_TO_G1
    
    def adapt_act(self, h1_actions: torch.Tensor) -> torch.Tensor:
        """Map H1 actions to G1 actions. Zero-pad unmapped G1 joints."""
        batch_size = h1_actions.shape[0]
        g1_actions = torch.zeros(batch_size, 37, device=self.device, dtype=h1_actions.dtype)
        for h1_idx, g1_idx in self.mapping.items():
            g1_actions[:, g1_idx] = h1_actions[:, h1_idx]
        return g1_actions
    
    def adapt_obs(self, g1_obs: torch.Tensor, g1_xanchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract H1 joints from G1 observations."""
        batch_size = g1_obs.shape[0]
        device = g1_obs.device
        dtype = g1_obs.dtype
        
        # H1 obs: [root(7), joint_pos(19), root(6), joint_vel(19)]
        # G1 obs: [root(7), joint_pos(37), root(6), joint_vel(37)]
        h1_obs = torch.zeros(batch_size, 51, device=device, dtype=dtype)
        
        # Copy entire root position and orientation (indices 0-6)
        h1_obs[:, :7] = g1_obs[:, :7]
        
        # Extract H1 joints from G1 (indices 7-25 for positions)
        for h1_idx, g1_idx in self.mapping.items():
            h1_obs[:, 7 + h1_idx] = g1_obs[:, 7 + g1_idx]  # positions
        
        # Copy root velocity unchanged (indices 26-31)
        h1_obs[:, 26:32] = g1_obs[:, 44:50]
        
        # Extract H1 joints velocities from G1 (indices 32-50)
        for h1_idx, g1_idx in self.mapping.items():
            h1_obs[:, 32 + h1_idx] = g1_obs[:, 50 + g1_idx]  # velocities
        
        # Extract H1 joints from G1 xanchor
        h1_xanchor = torch.zeros(batch_size, 20, 3, device=device, dtype=dtype)
        h1_xanchor[:, 0] = g1_xanchor[:, 0]  # Root
        for h1_idx, g1_idx in self.mapping.items():
            h1_xanchor[:, h1_idx + 1] = g1_xanchor[:, g1_idx + 1]
        
        return h1_obs, h1_xanchor
    
    def print_mapping_info(self):
        """Print mapping info."""
        print(f"H1 -> G1 Joint Mapping (19 -> 37)")
        print(f"Mapped joints: {len(self.mapping)}/19")

