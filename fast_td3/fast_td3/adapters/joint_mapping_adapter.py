"""Joint mapping adapter for cross-embodiment transfer.

This module provides a mapping layer that translates actions from one robot
morphology to another by matching joints by name and zero-padding for missing joints.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class JointMappingAdapter(nn.Module):
    """Adapter for mapping actions between different robot morphologies.
    
    This adapter maps actions from a source robot (e.g., H1 with 19 joints) to
    a target robot (e.g., G1 with 37 joints) by:
    1. Matching joints by semantic name (e.g., left_hip_pitch, right_knee)
    2. Zero-padding for joints present in target but not in source
    3. Optionally making extra joints trainable for fine-tuning
    
    Example:
        >>> # Map H1 (19 joints) to G1 (37 joints)
        >>> adapter = JointMappingAdapter(
        ...     source_robot="h1",
        ...     target_robot="g1",
        ...     device="cuda"
        ... )
        >>> h1_actions = torch.randn(16, 19)  # batch of 16
        >>> g1_actions = adapter(h1_actions)  # (16, 37)
    """
    
    def __init__(
        self,
        source_robot: str,
        target_robot: str,
        device: torch.device,
        trainable_unmapped: bool = False,
        init_scale: float = 0.01
    ):
        """Initialize the joint mapping adapter.
        
        Args:
            source_robot: Source robot type ("h1" or "g1")
            target_robot: Target robot type ("h1" or "g1")
            device: Device to place tensors on
            trainable_unmapped: If True, unmapped joints get trainable parameters
                               If False, unmapped joints are always zero
            init_scale: Initialization scale for trainable parameters
        """
        super().__init__()
        self.source_robot = source_robot.lower()
        self.target_robot = target_robot.lower()
        self.device = device
        self.trainable_unmapped = trainable_unmapped
        
        # Import robot classes
        from fast_td3.robots.h1 import H1
        from fast_td3.robots.g1 import G1
        
        # Get robot instances
        if self.source_robot == "h1":
            self.source = H1()
        elif self.source_robot == "g1":
            self.source = G1()
        else:
            raise ValueError(f"Unknown source robot: {source_robot}")
            
        if self.target_robot == "h1":
            self.target = H1()
        elif self.target_robot == "g1":
            self.target = G1()
        else:
            raise ValueError(f"Unknown target robot: {target_robot}")
        
        # Build joint name mappings
        self.source_joint_names = {joint.name: idx for idx, joint in enumerate(self.source.JOINT)}
        self.target_joint_names = {joint.name: idx for idx, joint in enumerate(self.target.JOINT)}
        
        # Create mapping from source indices to target indices
        self.mapping, self.unmapped_target = self._create_mapping()
        
        # Create mapping tensor for efficient batch processing
        # Shape: (target_joints,) with values indicating source index or -1 for unmapped
        self.register_buffer(
            "mapping_indices",
            torch.tensor([self.mapping.get(i, -1) for i in range(len(self.target.JOINT))],
                        dtype=torch.long, device=device)
        )
        
        # For unmapped joints, either create trainable parameters or use zeros
        if self.trainable_unmapped and len(self.unmapped_target) > 0:
            # Initialize with small random values
            unmapped_init = torch.randn(len(self.unmapped_target), device=device) * init_scale
            self.unmapped_actions = nn.Parameter(unmapped_init)
        else:
            self.register_buffer(
                "unmapped_actions",
                torch.zeros(len(self.unmapped_target), device=device)
            )
    
    def _create_mapping(self) -> Tuple[Dict[int, int], List[int]]:
        """Create mapping from target joint indices to source joint indices.
        
        Returns:
            mapping: Dict mapping target index -> source index for matched joints
            unmapped_target: List of target indices with no source match
        """
        mapping = {}
        unmapped_target = []
        
        # Try to match joints by name
        for target_idx, target_joint in enumerate(self.target.JOINT):
            target_name = target_joint.name
            
            # Try direct match first
            if target_name in self.source_joint_names:
                source_idx = self.source_joint_names[target_name]
                mapping[target_idx] = source_idx
            else:
                # Try fuzzy matching for slight name differences
                # e.g., "left_ankle" in H1 -> "left_ankle_pitch" in G1
                matched = False
                for source_name, source_idx in self.source_joint_names.items():
                    # Check if one name is a prefix/suffix of the other
                    if (source_name in target_name or target_name in source_name):
                        # Additional check: ensure they're semantically similar
                        # (same limb/side)
                        if self._are_joints_similar(source_name, target_name):
                            mapping[target_idx] = source_idx
                            matched = True
                            break
                
                if not matched:
                    unmapped_target.append(target_idx)
        
        return mapping, unmapped_target
    
    def _are_joints_similar(self, name1: str, name2: str) -> bool:
        """Check if two joint names are semantically similar.
        
        Args:
            name1: First joint name
            name2: Second joint name
            
        Returns:
            True if joints are similar (same side, same limb type)
        """
        # Check for same side (left/right)
        if "left" in name1 and "right" in name2:
            return False
        if "right" in name1 and "left" in name2:
            return False
        
        # Check for same limb type
        limb_types = ["hip", "knee", "ankle", "shoulder", "elbow", "torso"]
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        for limb in limb_types:
            if limb in name1_lower and limb in name2_lower:
                return True
        
        return False
    
    def forward(self, source_actions: torch.Tensor) -> torch.Tensor:
        """Map actions from source robot to target robot.
        
        Args:
            source_actions: Tensor of shape (batch, source_joints)
            
        Returns:
            target_actions: Tensor of shape (batch, target_joints)
        """
        batch_size = source_actions.shape[0]
        target_actions = torch.zeros(
            batch_size, len(self.target.JOINT),
            device=self.device, dtype=source_actions.dtype
        )
        
        # Map matched joints using advanced indexing
        # For each target joint, get the corresponding source joint
        for target_idx, source_idx in self.mapping.items():
            target_actions[:, target_idx] = source_actions[:, source_idx]
        
        # Fill in unmapped joints
        if len(self.unmapped_target) > 0:
            for i, target_idx in enumerate(self.unmapped_target):
                target_actions[:, target_idx] = self.unmapped_actions[i]
        
        return target_actions
    
    def get_mapping_info(self) -> Dict[str, any]:
        """Get information about the joint mapping.
        
        Returns:
            Dictionary with mapping statistics and details
        """
        mapped_pairs = []
        for target_idx, source_idx in self.mapping.items():
            target_name = self.target.JOINT(target_idx).name
            source_name = self.source.JOINT(source_idx).name
            mapped_pairs.append((source_name, target_name))
        
        unmapped_names = [self.target.JOINT(idx).name for idx in self.unmapped_target]
        
        return {
            "source_robot": self.source_robot,
            "target_robot": self.target_robot,
            "source_joints": len(self.source.JOINT),
            "target_joints": len(self.target.JOINT),
            "mapped_joints": len(self.mapping),
            "unmapped_target_joints": len(self.unmapped_target),
            "mapped_pairs": mapped_pairs,
            "unmapped_target_names": unmapped_names,
            "trainable_unmapped": self.trainable_unmapped
        }
    
    def print_mapping_info(self):
        """Print detailed information about the joint mapping."""
        info = self.get_mapping_info()
        
        print(f"\n{'='*70}")
        print(f"Joint Mapping: {info['source_robot'].upper()} → {info['target_robot'].upper()}")
        print(f"{'='*70}")
        print(f"Source joints: {info['source_joints']}")
        print(f"Target joints: {info['target_joints']}")
        print(f"Mapped joints: {info['mapped_joints']}")
        print(f"Unmapped target joints: {info['unmapped_target_joints']}")
        print(f"Trainable unmapped: {info['trainable_unmapped']}")
        
        print(f"\n{'Mapped Joint Pairs:'}")
        print(f"{'-'*70}")
        for source_name, target_name in info['mapped_pairs']:
            arrow = "→" if source_name == target_name else "≈>"
            print(f"  {source_name:30} {arrow} {target_name}")
        
        if info['unmapped_target_names']:
            print(f"\n{'Unmapped Target Joints (will be zero or trainable):'}")
            print(f"{'-'*70}")
            for name in info['unmapped_target_names']:
                print(f"  {name}")
        
        print(f"{'='*70}\n")
