"""
Dict observation tasks for humanoid_bench.

These tasks return observations as a flat vector (like the original tasks),
but provide an unflatten_obs service to convert to dict format when needed
for normalization and actor processing.

The observation dict (when unflattened) includes:
- object_positions: Root position (x, y, z)
- object_quaternions: Root quaternion (w, x, y, z)
- object_velocities: Root linear and angular velocities (vx, vy, vz, wx, wy, wz)
- joint_positions: Joint angles
- joint_velocities: Joint velocities
- joint_x: Joint anchor coordinates (3D positions)
"""

import numba
import torch
import numpy as np
from gymnasium.spaces import Box, Dict

from humanoid_bench.envs.basic_locomotion_envs import (
    Walk,
    Stand,
    Run,
    Crawl,
    ClimbingUpwards,
    Stair,
    Slide,
    Hurdle,
    Sit,
    SitHard,
)
from humanoid_bench.envs.balance import BalanceSimple


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

        # Extract pelvis state (free joint)
        pelvis_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        pelvis_quaternion = qpos[3:7]  # w, x, y, z

        # Extract pelvis velocity
        pelvis_linear_velocity = qvel[:3]  # vx, vy, vz
        pelvis_angular_velocity = qvel[3:6]  # wx, wy, wz

        # Extract joint state (excluding free joint)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]
        joint_x = xanchor[1:, :] - xanchor[0, :]  # relative to pelvis

        # Concatenate into flat vector
        return np.concatenate(
            [
                pelvis_position,
                pelvis_quaternion,
                pelvis_linear_velocity,
                pelvis_angular_velocity,
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
            "object_positions": (num_objects, 3),
            "object_quaternions": (num_objects, 4),
            "object_velocities": (num_objects, 2),
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
            
        return {
            "object_positions": flat_obs[:, 0:3].reshape(-1, num_objects, 3),
            "object_quaternions": flat_obs[:, 3:7].reshape(-1, num_objects, 4),
            "object_velocities": flat_obs[:, 7:9].reshape(-1, num_objects, 2),
            "joint_positions": flat_obs[:, 9:28],
            "joint_velocities": flat_obs[:, 28:47],
            "joint_x": flat_obs[:, 47:].reshape(-1, num_joints, 3),
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
                obs_dict["object_positions"].reshape(
                    obs_dict["object_positions"].shape[0], -1
                ),
                obs_dict["object_quaternions"].reshape(
                    obs_dict["object_quaternions"].shape[0], -1
                ),
                obs_dict["object_velocities"].reshape(
                    obs_dict["object_velocities"].shape[0], -1
                ),
                obs_dict["joint_positions"],
                obs_dict["joint_velocities"],
                obs_dict["joint_x"].reshape(obs_dict["joint_x"].shape[0], -1),
            ],
            axis=-1,
        )


class Stand(CustomObservation, Stand):
    base_task_name = "stand"


class Walk(CustomObservation, Walk):
    base_task_name = "walk"


class Run(CustomObservation, Run):
    base_task_name = "run"


class Crawl(CustomObservation, Crawl):
    base_task_name = "crawl"


class ClimbingUpwards(CustomObservation, ClimbingUpwards):
    base_task_name = "climbing_upwards"


class Stair(CustomObservation, Stair):
    base_task_name = "stair"


class Slide(CustomObservation, Slide):
    base_task_name = "slide"


class Hurdle(CustomObservation, Hurdle):
    base_task_name = "hurdle"


class Sit(CustomObservation, Sit):
    base_task_name = "sit"


class BalanceSimple(CustomObservation, BalanceSimple):
    base_task_name = "balance_simple"
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
        
        xanchor = xanchor[:, :] - xanchor[0, :]

        # Get pelvis state using named access
        pelvis_position = xanchor[0, :]  # x, y, z
        board_position = xanchor[20, :]  # x, y, z
        
        pelvis_quaternion = qpos[3:7]
        board_quaternion = qpos[29:33]

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        object_velocities = np.concatenate([qvel[0:6], qvel[25:31]])

        # Concatenate into flat vector
        return np.concatenate(
            [
                pelvis_position,
                board_position,
                pelvis_quaternion,
                board_quaternion,
                object_velocities,
                joint_positions,
                joint_velocities,
                joint_x.flatten(),
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
            "object_positions": (num_objects, 3),
            "object_quaternions": (num_objects, 4),
            "object_velocities": (num_objects, 6),
            "joint_positions": (num_joints,),
            "joint_velocities": (num_joints,),
            "joint_x": (num_joints, 3),
        }

    @classmethod
    def unflatten_obs(cls, flat_obs, num_objects=None, num_joints=None):
        """
        Convert flat observation to dictionary format for BalanceSimple.
        
        The flat observation structure:
        The flat observation structure:
        [0:3] pelvis_position
        [3:6] board_position
        [6:10] pelvis_quaternion
        [10:14] board_quaternion
        [14:16] pelvis_velocity_norms (lin, ang)
        [16:18] board_velocity_norms (lin, ang)
        [18:37] joint_positions
        [37:56] joint_velocities
        [56:113] joint_x (flattened)
        
        Args:
            flat_obs: Flat observation tensor of shape (batch_size, 113)
            num_objects: Number of objects (default: 2)
            num_joints: Number of joints (default: 19)
            
        Returns:
            Dictionary with observation components
        """
        if num_objects is None:
            num_objects = cls.num_objects if hasattr(cls, 'num_objects') else 2
        if num_joints is None:
            num_joints = cls.num_joints if hasattr(cls, 'num_joints') else 19
            
        return {
            "object_positions": flat_obs[:, 0:6].reshape(-1, num_objects, 3),
            "object_quaternions": flat_obs[:, 6:14].reshape(-1, num_objects, 4),
            "object_velocities": flat_obs[:, 14:26].reshape(-1, num_objects, 6),
            "joint_positions": flat_obs[:, 26:45],
            "joint_velocities": flat_obs[:, 45:64],
            "joint_x": flat_obs[:, 64:].reshape(-1, num_joints, 3),
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
                obs_dict["object_positions"].reshape(
                    obs_dict["object_positions"].shape[0], -1
                ),
                obs_dict["object_quaternions"].reshape(
                    obs_dict["object_quaternions"].shape[0], -1
                ),
                obs_dict["object_velocities"].reshape(
                    obs_dict["object_velocities"].shape[0], -1
                ),
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


@numba.jit(nopython=True)
def quat_conjugate_multiply(q_inv: np.array, q: np.array) -> np.array:
    """
    Specialized function: multiply conjugate of q_inv with q.
    For unit quaternion, conjugate is just negating imaginary parts.
    Faster than computing conjugate then multiplying separately.
    Expects [w, x, y, z] format.
    """
    # q_inv_conj = [w, -x, -y, -z]
    w1, x1, y1, z1 = q_inv[0], -q_inv[1], -q_inv[2], -q_inv[3]
    w2, x2, y2, z2 = q[0], q[1], q[2], q[3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


@numba.jit(nopython=True)
def extract_yaw_quaternion(q: np.array) -> np.array:
    """
    Extract the yaw component (rotation around global z-axis) from a quaternion.
    Returns a quaternion representing only the yaw rotation.
    
    For q = [w, x, y, z], the yaw angle can be computed from:
    yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    
    The returned quaternion is [cos(yaw/2), 0, 0, sin(yaw/2)].
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


@numba.jit(nopython=True)
def canonicalize_quaternions(q_base: np.array, q_other: np.array) -> tuple:
    """
    Remove global yaw from both quaternions, preserving relative orientation.
    
    This is the proper way to achieve rotation invariance without information loss:
    1. Extract the yaw (z-rotation) component of the base quaternion
    2. Apply the inverse yaw rotation to both quaternions
    
    Result:
    - q_base_canonical has zero yaw but keeps pitch and roll
    - q_other_canonical preserves its relative orientation to q_base
    
    This makes observations invariant to global heading while keeping:
    - The absolute tilt (pitch/roll) of both bodies
    - The relative orientation between them
    
    Args:
        q_base: Base quaternion [w, x, y, z] (e.g., pelvis)
        q_other: Other quaternion [w, x, y, z] (e.g., board)
        
    Returns:
        Tuple of (q_base_canonical, q_other_canonical)
    """
    q_yaw = extract_yaw_quaternion(q_base)
    
    # Apply inverse yaw to both quaternions
    q_base_canonical = quat_conjugate_multiply(q_yaw, q_base)
    q_other_canonical = quat_conjugate_multiply(q_yaw, q_other)
    
    return q_base_canonical, q_other_canonical


def project_gravity(q: np.array) -> np.array:
    """
    Project global gravity (0, 0, -1) into the local frame defined by q.
    This effectively tells the body "where is down" relative to itself.
    Result is [gx, gy, gz, 0] (padded to 4D to fit in quaternion slots).
    
    Math: v_local = q_inv * v_global * q
    Since v_global is just [0, 0, -1], this simplifies to extracting
    the 3rd row of the rotation matrix R(q) and negating it.
    
    R(q) row 3: [2(xz - wy), 2(yz + wx), 1 - 2(x^2 + y^2)]
    Negated:   [2(wy - xz), -2(yz + wx), 2(x^2 + y^2) - 1]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    gx = 2 * (w * y - x * z)
    gy = -2 * (y * z + w * x)
    gz = 2 * (x * x + y * y) - 1
    
    return np.array([gx, gy, gz, 0.0])


