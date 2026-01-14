"""
Dict observation tasks for humanoid_bench.

These tasks return observations as a flat vector (like the original tasks),
but provide an unflatten_obs service to convert to dict format when needed
for normalization and actor processing.

The observation dict (when unflattened) includes:
- object_features: Combined object state [position(3) + quaternion(4) + velocities(6)] = 13 per object
- joint_positions: Joint angles
- joint_velocities: Joint velocities
- joint_x: Joint anchor coordinates (3D positions)
- object_others: Task-specific additional features (Reach, Push)
"""

import numba
import torch
import numpy as np
from gymnasium.spaces import Box, Dict

from humanoid_bench.envs.basic_locomotion_envs import (
    Walk as WalkV0,
    Stand as StandV0,
    Run as RunV0,
    Crawl as CrawlV0,
    ClimbingUpwards as ClimbingUpwardsV0,
    Stair as StairV0,
    Slide as SlideV0,
    Hurdle as HurdleV0,
    Sit as SitV0,
    SitHard as SitHardV0,
)
from humanoid_bench.envs.balance import BalanceSimple as BalanceSimpleV0
from humanoid_bench.envs.reach import Reach as ReachV0
from humanoid_bench.envs.push import Push as PushV0


class CustomObservation:
    """Mixin class that provides flat observations with unflatten service for locomotion tasks."""

    base_task_name = None
    num_objects = 1
    num_joints = 19

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(108,),
            dtype=np.float64,
        )

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = xanchor / 8 # normalize positions


        # Extract pelvis state (free joint)
        pelvis_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        pelvis_quaternion = qpos[3:7]  # w, x, y, z

        # Extract pelvis velocity
        pelvis_velocities = qvel[:6]  # wx, wy, wz

        # Extract joint state (excluding free joint)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]
        joint_x = xanchor[1:, :] - xanchor[0, :]  # relative to pelvis

        # Concatenate into flat vector
        return np.concatenate(
            [
                pelvis_position,
                pelvis_quaternion,
                pelvis_velocities,
                joint_positions,
                joint_velocities,
                joint_x.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes(num_objects=1, num_joints=19):
        """
        Get observation shape dictionary.
        
        Args:
            num_objects: Number of objects (default: 1)
            num_joints: Number of joints (default: 19)
            
        Returns:
            Dictionary mapping observation keys to their shapes
        """
        return {
            "object_features": (num_objects, 13),
            "joint_positions": (num_joints,),
            "joint_velocities": (num_joints,),
            "joint_x": (num_joints, 3),
        }

    @classmethod
    def unflatten_obs(cls, flat_obs, num_objects=None, num_joints=None):
        """
        Convert flat observation to dictionary format.
        
        Args:
            flat_obs: Flat observation tensor of shape (batch_size, 104)
            num_objects: Number of objects (default: class num_objects property)
            num_joints: Number of joints (default: class num_joints property)
            
        Returns:
            Dictionary with observation components
        """
        if num_objects is None:
            num_objects = cls.num_objects if hasattr(cls, 'num_objects') else 1
        if num_joints is None:
            num_joints = cls.num_joints if hasattr(cls, 'num_joints') else 19
            
        # object_features = [x(3), quaternion(4), velocities(6)] = 13 per object
        object_features = flat_obs[:, 0:13].reshape(-1, num_objects, 13)
            
        return {
            "object_features": object_features,
            "joint_positions": flat_obs[:, 13:32],
            "joint_velocities": flat_obs[:, 32:51],
            "joint_x": flat_obs[:, 51:].reshape(-1, num_joints, 3),
        }

    @staticmethod
    def flatten_obs(obs_dict):
        """
        Convert dictionary observation to flat format.
        
        Args:
            obs_dict: Dictionary with observation components
            
        Returns:
            Flat observation tensor of shape (batch_size, 104)
        """
        return torch.cat(
            [
                obs_dict["object_features"].reshape(
                    obs_dict["object_features"].shape[0], -1
                ),
                obs_dict["joint_positions"],
                obs_dict["joint_velocities"],
                obs_dict["joint_x"].reshape(obs_dict["joint_x"].shape[0], -1),
            ],
            axis=-1,
        )


class Stand(CustomObservation, StandV0):
    base_task_name = "stand"


class Walk(CustomObservation, WalkV0):
    base_task_name = "walk"


class Run(CustomObservation, RunV0):
    base_task_name = "run"


class Crawl(CustomObservation, CrawlV0):
    base_task_name = "crawl"


class ClimbingUpwards(CustomObservation, ClimbingUpwardsV0):
    base_task_name = "climbing_upwards"


class Stair(CustomObservation, StairV0):
    base_task_name = "stair"


class Slide(CustomObservation, SlideV0):
    base_task_name = "slide"


class Hurdle(CustomObservation, HurdleV0):
    base_task_name = "hurdle"


class Sit(CustomObservation, SitV0):
    base_task_name = "sit_simple"


class BalanceSimple(CustomObservation, BalanceSimpleV0):
    base_task_name = "balance_simple"
    num_objects = 2
    num_joints = 19
    
    # Hard-coded observation dimensions
    # Total: 121 dims = pelvis(13) + board(13) + joints(38) + joint_x(57)
    PELVIS_DIM = 13  # pos(3) + quat(4) + vel(6)
    BOARD_DIM = 13   # pos(3) + quat(4) + vel(6)
    JOINT_POS_DIM = 19
    JOINT_VEL_DIM = 19
    JOINT_X_DIM = 57  # 19 * 3
    
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(121,),
            dtype=np.float64,
        )
    
    @staticmethod
    def get_object_feature_dim():
        """Return total object feature dimension: 2 objects * 13 = 26"""
        return 26

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions

        # Concatenate with hard-coded order: pelvis first, board second
        return np.concatenate(
            [
                qpos[0:7],                # [0:7] pelvis position and quaternion
                qvel[0:6],                # [7:13] pelvis velocities
                xanchor[20, :],           # [13:16] board position
                qpos[29:33],              # [16:20] board quaternion
                qvel[25:31],              # [20:26] board velocities
                qpos[7:26],               # [26:45] joint positions
                qvel[6:25],               # [45:64] joint velocities
                xanchor[1:20, :].flatten(),  # [64:121] joint_x
            ]
        )

    @staticmethod
    def get_obs_shapes(num_objects=2, num_joints=19):
        """
        Get observation shape dictionary for BalanceSimple.
        
        Args:
            num_objects: Number of objects (default: 2 - pelvis and board)
            num_joints: Number of joints (default: 19)
            
        Returns:
            Dictionary mapping observation keys to their shapes
        """
        return {
            "object_features": (num_objects, 13),
            "joint_positions": (num_joints,),
            "joint_velocities": (num_joints,),
            "joint_x": (num_joints, 3),
        }

    @classmethod
    def unflatten_obs(cls, flat_obs, num_objects=None, num_joints=None):
        """
        Convert flat observation to dictionary format for BalanceSimple.
        
        Hard-coded flat observation structure (121 dims):
        [0:3]    pelvis_position
        [3:7]    pelvis_quaternion
        [7:13]   pelvis_velocities
        [13:16]  board_position
        [16:20]  board_quaternion
        [20:26]  board_velocities
        [26:45]  joint_positions (19)
        [45:64]  joint_velocities (19)
        [64:121] joint_x (57 = 19*3)
        
        Args:
            flat_obs: Flat observation tensor of shape (batch_size, 121)
            num_objects: Ignored - hard-coded to 2
            num_joints: Ignored - hard-coded to 19
            
        Returns:
            Dictionary with observation components structured for egnn_v3
        """
        # Hard-coded slicing for CUDA graph compatibility
        pelvis_features = flat_obs[:, 0:13]  # pos(3) + quat(4) + vel(6)
        board_features = flat_obs[:, 13:26]  # pos(3) + quat(4) + vel(6)
        
        # Stack objects: pelvis first, then board
        object_features = torch.stack([pelvis_features, board_features], dim=1)  # (batch, 2, 13)
        
        return {
            "object_features": object_features,
            "joint_positions": flat_obs[:, 26:45],
            "joint_velocities": flat_obs[:, 45:64],
            "joint_x": flat_obs[:, 64:121].reshape(-1, 19, 3),
        }

    @staticmethod
    def flatten_obs(obs_dict):
        """
        Convert dictionary observation to flat format for BalanceSimple.
        
        Args:
            obs_dict: Dictionary with observation components
            
        Returns:
            Flat observation tensor of shape (batch_size, 121)
        """
        return torch.cat(
            [
                obs_dict["object_features"].reshape(
                    obs_dict["object_features"].shape[0], -1
                ),
                obs_dict["joint_positions"],
                obs_dict["joint_velocities"],
                obs_dict["joint_x"].reshape(obs_dict["joint_x"].shape[0], -1),
            ],
            axis=-1,
        )


class SitHard(CustomObservation, SitHardV0):
    base_task_name = "sit_hard"
    num_objects = 2
    
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(121,),
            dtype=np.float64,
        )

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions

        return np.concatenate(
            [
                qpos[0:7],                # pelvis position and quaternion
                qpos[26:30],              # board position and quaternion
                qvel[0:6],                # pelvis velocities
                qvel[25:31],              # board velocities
                qpos[7:26],               # joint positions
                qvel[6:25],               # joint velocities
                xanchor[1:20, :].flatten(),  # joint_x
            ]
        )

    @staticmethod
    def get_obs_shapes(num_objects=2, num_joints=19):
        """
        Get observation shape dictionary for SitHard.
        
        Args:
            num_objects: Number of objects (default: 2 - pelvis and board)
            num_joints: Number of joints (default: 19)
            
        Returns:
            Dictionary mapping observation keys to their shapes
        """
        return {
            "object_features": (num_objects, 13),
            "joint_positions": (num_joints,),
            "joint_velocities": (num_joints,),
            "joint_x": (num_joints, 3),
        }

    @classmethod
    def unflatten_obs(cls, flat_obs, num_objects=None, num_joints=None):
        """
        Convert flat observation to dictionary format for SitHard.
        
        The flat observation structure:
        [0:3] pelvis_position
        [3:6] board_position
        [6:10] pelvis_quaternion
        [10:14] board_quaternion
        [14:20] pelvis_velocities
        [20:26] board_velocities
        [26:45] joint_positions
        [45:64] joint_velocities
        [64:121] joint_x (flattened)
        
        Args:
            flat_obs: Flat observation tensor of shape (batch_size, 121)
            num_objects: Number of objects (default: 2)
            num_joints: Number of joints (default: 19)
            
        Returns:
            Dictionary with observation components
        """
        if num_objects is None:
            num_objects = cls.num_objects if hasattr(cls, 'num_objects') else 2
        if num_joints is None:
            num_joints = cls.num_joints if hasattr(cls, 'num_joints') else 19
        
        # object_features = [x(3), quaternion(4), velocities(6)] = 13 per object
        object_features = flat_obs[:, 0:26].reshape(-1, num_objects, 13)
            
        return {
            "object_features": object_features,
            "joint_positions": flat_obs[:, 26:45],
            "joint_velocities": flat_obs[:, 45:64],
            "joint_x": flat_obs[:, 64:].reshape(-1, num_joints, 3),
        }

    @staticmethod
    def flatten_obs(obs_dict):
        """
        Convert dictionary observation to flat format for SitHard.
        
        Args:
            obs_dict: Dictionary with observation components
            
        Returns:
            Flat observation tensor of shape (batch_size, 121)
        """
        return torch.cat(
            [
                obs_dict["object_features"].reshape(
                    obs_dict["object_features"].shape[0], -1
                ),
                obs_dict["joint_positions"],
                obs_dict["joint_velocities"],
                obs_dict["joint_x"].reshape(obs_dict["joint_x"].shape[0], -1),
            ],
            axis=-1,
        )



class Reach(CustomObservation, ReachV0):
    base_task_name = "reach"
    num_objects = 1
    num_joints = 19
    
    # Hard-coded observation dimensions
    # Total: 114 dims = pelvis(13) + joints(38) + joint_x(57) + object_others(6)
    PELVIS_DIM = 13  # pos(3) + quat(4) + vel(6)
    JOINT_POS_DIM = 19
    JOINT_VEL_DIM = 19
    JOINT_X_DIM = 57  # 19 * 3
    OBJECT_OTHERS_DIM = 6  # left_hand(3) + target(3)
    
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(114,),
            dtype=np.float64,
        )
    
    @staticmethod
    def get_object_feature_dim():
        """Return total object feature dimension: 1 pelvis * 13 + 6 others = 19"""
        return 19

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions
        
        return np.concatenate(
            [
                qpos[0:7],                # pelvis position and quaternion
                qvel[0:6],                # pelvis velocities
                self.robot.left_hand_position(),  # object_others: left_hand
                self.goal,                # object_others: target
                qpos[7:26],               # joint positions
                qvel[6:25],               # joint velocities
                xanchor[1:20, :].flatten(),  # joint_x
            ]
        )

    @staticmethod
    def get_obs_shapes(num_objects=1, num_joints=19):
        """
        Get observation shape dictionary for Reach.
        
        Args:
            num_objects: Number of objects (default: 1)
            num_joints: Number of joints (default: 19)
            
        Returns:
            Dictionary mapping observation keys to their shapes
        """
        return {
            "object_features": (num_objects, 13),
            "object_others": (6,),
            "joint_positions": (num_joints,),
            "joint_velocities": (num_joints,),
            "joint_x": (num_joints, 3),
        }

    @classmethod
    def unflatten_obs(cls, flat_obs, num_objects=None, num_joints=None):
        """
        Convert flat observation to dictionary format for Reach.
        
        Hard-coded flat observation structure (114 dims):
        [0:3]    pelvis_position
        [3:7]    pelvis_quaternion
        [7:13]   pelvis_velocities
        [13:19]  object_others (left_hand + target)
        [19:38]  joint_positions (19)
        [38:57]  joint_velocities (19)
        [57:114] joint_x (57 = 19*3)
        
        Args:
            flat_obs: Flat observation tensor of shape (batch_size, 114)
            num_objects: Ignored - hard-coded to 1
            num_joints: Ignored - hard-coded to 19
            
        Returns:
            Dictionary with observation components structured for egnn_v3
        """
        # Hard-coded slicing for CUDA graph compatibility
        pelvis_features = flat_obs[:, 0:13]  # pos(3) + quat(4) + vel(6)
        
        # Object features: pelvis only (reshaped to match expected format)
        object_features = pelvis_features.unsqueeze(1)  # (batch, 1, 13)
        
        return {
            "object_features": object_features,
            "object_others": flat_obs[:, 13:19],
            "joint_positions": flat_obs[:, 19:38],
            "joint_velocities": flat_obs[:, 38:57],
            "joint_x": flat_obs[:, 57:114].reshape(-1, 19, 3),
        }

    @staticmethod
    def flatten_obs(obs_dict):
        """
        Convert dictionary observation to flat format for Reach.
        
        Args:
            obs_dict: Dictionary with observation components
            
        Returns:
            Flat observation tensor of shape (batch_size, 114)
        """
        return torch.cat(
            [
                obs_dict["object_features"].reshape(
                    obs_dict["object_features"].shape[0], -1
                ),
                obs_dict["object_others"],
                obs_dict["joint_positions"],
                obs_dict["joint_velocities"],
                obs_dict["joint_x"].reshape(obs_dict["joint_x"].shape[0], -1),
            ],
            axis=-1,
        )


class Push(CustomObservation, PushV0):
    base_task_name = "push"
    num_objects = 2
    num_joints = 19
    
    # Hard-coded observation dimensions
    # Total: 133 dims = pelvis(13) + box(13) + joints(38) + joint_x(57) + object_others(12)
    PELVIS_DIM = 13  # pos(3) + quat(4) + vel(6)
    BOX_DIM = 13     # pos(3) + quat(4) + vel(6)
    JOINT_POS_DIM = 19
    JOINT_VEL_DIM = 19
    JOINT_X_DIM = 57  # 19 * 3
    OBJECT_OTHERS_DIM = 12  # left_hand(3) + target(3) + box(3) + box_vel(3)
    
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(133,),
            dtype=np.float64,
        )
    
    @staticmethod
    def get_object_feature_dim():
        """Return total object feature dimension: 2 objects * 13 + 12 others = 38"""
        return 38

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions
        
        box_position = qpos[-7:-4]
        box_quaternion = qpos[-4:]
        dofadr = self._env.named.model.body_dofadr["object"]
        box_linear_vel = qvel[dofadr : dofadr + 3]
        
        # Concatenate with hard-coded order: pelvis first, box second
        return np.concatenate(
            [
                qpos[0:7],                # [0:7] pelvis position and quaternion
                qvel[0:6],                # [7:13] pelvis velocities
                box_position,             # [13:16] box position
                box_quaternion,           # [16:20] box quaternion
                box_linear_vel,           # [20:23] box linear velocity
                np.zeros(3),              # [23:26] box angular velocity
                self.robot.left_hand_position(),  # [26:29] object_others: left_hand
                self.goal,                # [29:32] object_others: target
                box_position,             # [32:35] object_others: box
                box_linear_vel,           # [35:38] object_others: box_vel
                qpos[7:26],               # [38:57] joint positions
                qvel[6:25],               # [57:76] joint velocities
                xanchor[1:20, :].flatten(),  # [76:133] joint_x
            ]
        )

    @staticmethod
    def get_obs_shapes(num_objects=2, num_joints=19):
        """
        Get observation shape dictionary for Push.
        
        Args:
            num_objects: Number of objects (default: 2)
            num_joints: Number of joints (default: 19)
            
        Returns:
            Dictionary mapping observation keys to their shapes
        """
        return {
            "object_features": (num_objects, 13),
            "object_others": (12,),
            "joint_positions": (num_joints,),
            "joint_velocities": (num_joints,),
            "joint_x": (num_joints, 3),
        }

    @classmethod
    def unflatten_obs(cls, flat_obs, num_objects=None, num_joints=None):
        """
        Convert flat observation to dictionary format for Push.
        
        Hard-coded flat observation structure (133 dims):
        [0:3]    pelvis_position
        [3:7]    pelvis_quaternion
        [7:13]   pelvis_velocities
        [13:16]  box_position
        [16:20]  box_quaternion
        [20:26]  box_velocities
        [26:38]  object_others (left_hand + target + box + box_vel)
        [38:57]  joint_positions (19)
        [57:76]  joint_velocities (19)
        [76:133] joint_x (57 = 19*3)
        
        Args:
            flat_obs: Flat observation tensor of shape (batch_size, 133)
            num_objects: Ignored - hard-coded to 2
            num_joints: Ignored - hard-coded to 19
            
        Returns:
            Dictionary with observation components structured for egnn_v3
        """
        # Hard-coded slicing for CUDA graph compatibility
        pelvis_features = flat_obs[:, 0:13]  # pos(3) + quat(4) + vel(6)
        box_features = flat_obs[:, 13:26]  # pos(3) + quat(4) + vel(6)
        
        # Stack objects: pelvis first, then box
        object_features = torch.stack([pelvis_features, box_features], dim=1)  # (batch, 2, 13)
        
        return {
            "object_features": object_features,
            "object_others": flat_obs[:, 26:38],
            "joint_positions": flat_obs[:, 38:57],
            "joint_velocities": flat_obs[:, 57:76],
            "joint_x": flat_obs[:, 76:133].reshape(-1, 19, 3),
        }

    @staticmethod
    def flatten_obs(obs_dict):
        """
        Convert dictionary observation to flat format for Push.
        
        Args:
            obs_dict: Dictionary with observation components
            
        Returns:
            Flat observation tensor of shape (batch_size, 133)
        """
        return torch.cat(
            [
                obs_dict["object_features"].reshape(
                    obs_dict["object_features"].shape[0], -1
                ),
                obs_dict["object_others"],
                obs_dict["joint_positions"],
                obs_dict["joint_velocities"],
                obs_dict["joint_x"].reshape(obs_dict["joint_x"].shape[0], -1),
            ],
            axis=-1,
        )


# Map environment names to their corresponding environment classes
ENV_CLASS_MAP = {
    "h1-stand-v1": Stand,
    "h1-walk-v1": Walk,
    "h1-run-v1": Run,
    "h1-crawl-v1": Crawl,
    "h1-climbing_upwards-v1": ClimbingUpwards,
    "h1-stair-v1": Stair,
    "h1-slide-v1": Slide,
    "h1-hurdle-v1": Hurdle,
    "h1-sit_simple-v1": Sit,
    "h1-sit_hard-v1": SitHard,
    "h1-balance_simple-v1": BalanceSimple,
    "h1-reach-v1": Reach,
    "h1-push-v1": Push,
}


def get_env_class(env_name: str):
    """Get the environment class for a given environment name."""
    if env_name not in ENV_CLASS_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Supported environments: {list(ENV_CLASS_MAP.keys())}")
    return ENV_CLASS_MAP[env_name]


def unflatten_obs(flat_obs, env_name: str):
    """
    Convert flat observation to dictionary format based on environment.
    
    Delegates to the environment class's classmethod.
    
    Args:
        flat_obs: Flat observation tensor of shape (batch_size, 108)
        env_name: Environment name to determine observation structure
        
    Returns:
        Dictionary with observation components
    """
    env_class = get_env_class(env_name)
    return env_class.unflatten_obs(flat_obs)


def flatten_obs(obs_dict, env_name: str):
    """
    Convert dictionary observation to flat format based on environment.
    
    Delegates to the environment class's static method.
    
    Args:
        obs_dict: Dictionary with observation components
        env_name: Environment name to determine observation structure
        
    Returns:
        Flat observation tensor of shape (batch_size, 108)
    """
    env_class = get_env_class(env_name)
    return env_class.flatten_obs(obs_dict)


def get_obs_shapes(env_name: str):
    """
    Get observation shape dictionary for a given environment.
    
    Delegates to the environment class's static method.
    
    Args:
        env_name: Environment name
        
    Returns:
        Dictionary mapping observation keys to their shapes
    """
    env_class = get_env_class(env_name)
    return env_class.get_obs_shapes()


def quat_to_rot6d(q):
    """
    q: (..., 4) array, unit quaternion in (w, x, y, z) order
    returns: (..., 6) 6D rotation representation
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    r1x = 1 - 2*(y*y + z*z)
    r1y = 2*(x*y + w*z)
    r1z = 2*(x*z - w*y)

    r2x = 2*(x*y - w*z)
    r2y = 1 - 2*(x*x + z*z)
    r2z = 2*(y*z + w*x)

    rot6d = np.stack(
        [r1x, r1y, r1z, r2x, r2y, r2z],
        axis=-1
    )

    return rot6d
