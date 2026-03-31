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


class CustomObservationH1Hand:
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
        xanchor = xanchor / 8

        pelvis = qpos[0:7]
        pelvis_velocities = qvel[:6]
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]
        joint_x = xanchor[1:, :] - xanchor[0, :]

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
        return 13

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :13].unsqueeze(1)
        h_joints = flat_obs[:, 13:].view(batch_size, 19, 5)
        return h_joints, h_objects


class Stand(CustomObservationH1Hand, StandV0):
    base_task_name = "stand"
class Walk(CustomObservationH1Hand, WalkV0):
    base_task_name = "walk"
class Run(CustomObservationH1Hand, RunV0):
    base_task_name = "run"
class Crawl(CustomObservationH1Hand, CrawlV0):
    base_task_name = "crawl"
class ClimbingUpwards(CustomObservationH1Hand, ClimbingUpwardsV0):
    base_task_name = "climbing_upwards"
class Stair(CustomObservationH1Hand, StairV0):
    base_task_name = "stair"
class Slide(CustomObservationH1Hand, SlideV0):
    base_task_name = "slide"
class Hurdle(CustomObservationH1Hand, HurdleV0):
    base_task_name = "hurdle"
class Sit(CustomObservationH1Hand, SitV0):
    base_task_name = "sit_simple"


class BalanceSimple(CustomObservationH1Hand, BalanceSimpleV0):
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
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],
                qvel[0:6],
                xanchor[20, :],
                qpos[29:33],
                qvel[25:31],
                joint_features.flatten(),
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
        h_objects = flat_obs[:, :26].unsqueeze(1)
        h_joints = flat_obs[:, 26:].view(batch_size, 19, 5)
        return h_joints, h_objects

class SitHard(CustomObservationH1Hand, SitHardV0):
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

    @staticmethod
    def get_object_feature_dim():
        return 23

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],
                qpos[26:30],
                qvel[0:6],
                qvel[25:31],
                joint_features.flatten(),
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
        h_objects = flat_obs[:, :23].unsqueeze(1)
        h_joints = flat_obs[:, 23:].view(batch_size, 19, 5)
        return h_joints, h_objects


class Reach(CustomObservationH1Hand, ReachV0):
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
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],
                qvel[0:6],
                self.robot.left_hand_position(),
                self.goal,
                joint_features.flatten(),
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
        h_objects = flat_obs[:, :19].unsqueeze(1)
        h_joints = flat_obs[:, 19:].view(batch_size, 19, 5)
        return h_joints, h_objects


class Push(CustomObservationH1Hand, PushV0):
    base_task_name = "push"
    num_objects = 2
    num_joints = 19
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(124,),
            dtype=np.float64,
        )
    
    @staticmethod
    def get_object_feature_dim():
        return 29

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8
        
        box_position = qpos[-7:-4]
        box_quaternion = qpos[-4:]
        dofadr = self._env.named.model.body_dofadr["object"]
        box_linear_vel = qvel[dofadr : dofadr + 3]

        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],
                qvel[0:6],
                box_position,
                box_quaternion,
                box_linear_vel,
                self.robot.left_hand_position(),
                self.goal,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (29,),
            "joint_features": (19, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :29].unsqueeze(1)
        h_joints = flat_obs[:, 29:].view(batch_size, 19, 5)
        return h_joints, h_objects


class Door(CustomObservationH1Hand, DoorV0):
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

    @staticmethod
    def get_object_feature_dim():
        return 17
    
    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8
        joint_positions = qpos[7:26]
        joint_velocities = qvel[6:25]
        joint_x = xanchor[1:20, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                qpos[0:7],
                qvel[0:6],
                np.array([qpos[26], qpos[27]]),
                np.array([qvel[25], qvel[26]]),
                joint_features.flatten(),
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
        h_objects = flat_obs[:, :17].unsqueeze(1)
        h_joints = flat_obs[:, 17:].view(batch_size, 19, 5)
        return h_joints, h_objects
     

ENV_CLASS_MAP = {
    "h1hand-stand-v1": Stand,
    "h1hand-walk-v1": Walk,
    "h1hand-run-v1": Run,
    "h1hand-crawl-v1": Crawl,
    "h1hand-climbing_upwards-v1": ClimbingUpwards,
    "h1hand-stair-v1": Stair,
    "h1hand-slide-v1": Slide,
    "h1hand-hurdle-v1": Hurdle,
    "h1hand-sit_simple-v1": Sit,
    "h1hand-sit_hard-v1": SitHard,
    "h1hand-balance_simple-v1": BalanceSimple,
    "h1hand-reach-v1": Reach,
    "h1hand-push-v1": Push,
    "h1hand-door-v1": Door,
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
    env_class = get_env_class(env_name)
    return env_class.get_object_feature_dim()
