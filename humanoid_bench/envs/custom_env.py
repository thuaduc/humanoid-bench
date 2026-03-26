import torch
import numpy as np
from gymnasium.spaces import Box

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
from humanoid_bench.envs.door import Door as DoorV0


class CustomObservation:
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
        pelvis = qpos[0:7]  # position (3) + quaternion (4)

        # Extract pelvis velocity
        pelvis_velocities = qvel[:6]

        # Extract joint state (excluding free joint)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]
        joint_x = xanchor[1:, :] - xanchor[0, :]  # relative to pelvis

        # Interleave per-node: each row is [pos, vel, x, y, z] for joint i
        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate([pelvis, pelvis_velocities, joint_features.flatten()])

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (13,),
            "joint_features": (19, 5),
        }

    @staticmethod
    def get_object_feature_dim():
        """Return total object feature dimension: 13 * num_objects"""
        return 13

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        # flat_obs: (batch, obs_dim)
        batch_size = flat_obs.shape[0]
        # h_objects: (batch, 1, 13)
        h_objects = flat_obs[:, :13].unsqueeze(1)
        # h_joints: (batch, 19, 5) — zero-copy view: [pos, vel, x, y, z] per node
        h_joints = flat_obs[:, 13:].view(batch_size, 19, 5)
        return h_joints, h_objects


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
    num_joints = 19
    
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
        return 26

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        # Interleave per-node: each row is [pos, vel, x, y, z] for joint i
        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],          # [0:7]   pelvis position and quaternion
                qvel[0:6],          # [7:13]  pelvis velocities
                xanchor[20, :],     # [13:16] board position
                qpos[29:33],        # [16:20] board quaternion
                qvel[25:31],        # [20:26] board velocities
                joint_features.flatten(),  # [26:121] per-node joint features
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (26,),
            "joint_features": (19, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        # h_objects: (batch, 1, 26)
        h_objects = flat_obs[:, :26].unsqueeze(1)
        # h_joints: (batch, 19, 5) — zero-copy view: [pos, vel, x, y, z] per node
        h_joints = flat_obs[:, 26:].view(batch_size, 19, 5)
        return h_joints, h_objects

class SitHard(CustomObservation, SitHardV0):
    base_task_name = "sit_hard"
    num_objects = 2
    
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(118,),
            dtype=np.float64,
        )

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        # Interleave per-node: each row is [pos, vel, x, y, z] for joint i
        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],          # pelvis position and quaternion  (7)
                qpos[26:30],        # board position and quaternion   (4)
                qvel[0:6],          # pelvis velocities               (6)
                qvel[25:31],        # board velocities                (6)
                joint_features.flatten(),  # per-node joint features (95)
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (23,),
            "joint_features": (19, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        # h_objects: (batch, 1, 23)
        h_objects = flat_obs[:, :23].unsqueeze(1)
        # h_joints: (batch, 19, 5) — zero-copy view: [pos, vel, x, y, z] per node
        h_joints = flat_obs[:, 23:].view(batch_size, 19, 5)
        return h_joints, h_objects


class Reach(CustomObservation, ReachV0):
    base_task_name = "reach"
    num_joints = 19
   
    
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
        return 19

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        # Interleave per-node: each row is [pos, vel, x, y, z] for joint i
        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],                        # pelvis position and quaternion (7)
                qvel[0:6],                        # pelvis velocities              (6)
                self.robot.left_hand_position(),  # left_hand                      (3)
                self.goal,                        # target                         (3)
                joint_features.flatten(),         # per-node joint features        (95)
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (19,),
            "joint_features": (19, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        # h_objects: (batch, 1, 19)
        h_objects = flat_obs[:, :19].unsqueeze(1)
        # h_joints: (batch, 19, 5) — zero-copy view: [pos, vel, x, y, z] per node
        h_joints = flat_obs[:, 19:].view(batch_size, 19, 5)
        return h_joints, h_objects


class Push(CustomObservation, PushV0):
    base_task_name = "push"
    num_objects = 2
    num_joints = 19
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(130,),
            dtype=np.float64,
        )
    
    @staticmethod
    def get_object_feature_dim():
        """Return total object feature dimension: 2 objects * 13 + 12 others = 38"""
        return 35

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8  # normalize positions
        
        box_position = qpos[-7:-4]
        box_quaternion = qpos[-4:]
        dofadr = self._env.named.model.body_dofadr["object"]
        box_linear_vel = qvel[dofadr : dofadr + 3]

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        # Interleave per-node: each row is [pos, vel, x, y, z] for joint i
        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],                        # [0:7]   pelvis position and quaternion
                qvel[0:6],                        # [7:13]  pelvis velocities
                box_position,                     # [13:16] box position
                box_quaternion,                   # [16:20] box quaternion
                box_linear_vel,                   # [20:23] box linear velocity
                self.robot.left_hand_position(),  # [23:26] left_hand
                self.goal,                        # [26:29] target
                box_position,                     # [29:32] box (duplicated for policy)
                box_linear_vel,                   # [32:35] box_vel
                joint_features.flatten(),         # [35:130] per-node joint features
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (35,),
            "joint_features": (19, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        # h_objects: (batch, 1, 35)
        h_objects = flat_obs[:, :35].unsqueeze(1)
        # h_joints: (batch, 19, 5) — zero-copy view: [pos, vel, x, y, z] per node
        h_joints = flat_obs[:, 35:].view(batch_size, 19, 5)
        return h_joints, h_objects


class Door(CustomObservation, DoorV0):
    base_task_name = "door"
    num_objects = 1
    
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(112,),
            dtype=np.float64,
        )
    
    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8
        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        # Interleave per-node: each row is [pos, vel, x, y, z] for joint i
        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],                                          # pelvis pos+quat     (7)
                qvel[0:6],                                          # pelvis velocities   (6)
                np.array([qpos[26], qpos[27]]),                     # door hinge positions (2)
                np.array([qvel[25], qvel[26]]),                     # door hinge velocities (2)
                joint_features.flatten(),                           # per-node joint features (95)
            ]
        )
        
    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (17,),
            "joint_features": (19, 5),
        }
        
    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        # h_objects: (batch, 1, 17)
        h_objects = flat_obs[:, :17].unsqueeze(1)
        # h_joints: (batch, 19, 5) — zero-copy view: [pos, vel, x, y, z] per node
        h_joints = flat_obs[:, 17:].view(batch_size, 19, 5)
        return h_joints, h_objects
     

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
    "h1-door-v1": Door,
}


def get_env_class(env_name: str):
    if env_name not in ENV_CLASS_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Supported environments: {list(ENV_CLASS_MAP.keys())}")
    return ENV_CLASS_MAP[env_name]


def unflatten_obs(flat_obs, env_name: str):
    env_class = get_env_class(env_name)
    return env_class.unflatten_obs(flat_obs)


def get_obs_shapes(env_name: str):
    env_class = get_env_class(env_name)
    return env_class.get_obs_shapes()


def get_object_feature_dim(env_name: str) -> int:
    """Get the object feature dimension for a given environment."""
    env_class = get_env_class(env_name)
    return env_class.get_object_feature_dim()
