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
from humanoid_bench.envs.balance import BalanceSimple as BalanceSimpleV0, BalanceHard as BalanceHardV0
from humanoid_bench.envs.reach import Reach as ReachV0
from humanoid_bench.envs.push import Push as PushV0
from humanoid_bench.envs.door import Door as DoorV0
from humanoid_bench.envs.window import Window as WindowV0
from humanoid_bench.envs.basketball import Basketball as BasketballV0


class CustomObservationH1Hand:
    base_task_name = None
    num_joints = 69  # H1Hand has 76 DOF: 7 for pelvis + 69 joints

    @property
    def observation_space(self):
        # Observation structure:
        # - pelvis (7) + pelvis_velocities (6) = 13 (object features)
        # - For each of 69 joints: position (1) + velocity (1) + xanchor (3) = 5 features
        # Total: 13 + 69 * 5 = 358
        num_joints = self.robot.dof - 7  # Subtract 7 for pelvis (3 pos + 4 quat)
        obs_dim = 13 + num_joints * 5
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
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
            "joint_features": (69, 5),  # 69 joints, 5 features each
        }

    @staticmethod
    def get_object_feature_dim():
        return 13

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :13].unsqueeze(1)
        h_joints = flat_obs[:, 13:].view(batch_size, 69, 5)  # 69 joints, not 19
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
    num_joints = 69
    
    @property
    def observation_space(self):
        # Object features: pelvis (7) + pelvis_vel (6) + balance_object (3) + box (4) + box_vel (6) = 26
        # Joint features: 69 joints × 5 features = 345
        # Total: 26 + 345 = 371
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(371,),
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

        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        balance_object = xanchor[70, :]  # Assuming balance object is at anchor 70
        box_pos = qpos[76:80]  # Box position (4 elements including quaternion)
        box_vel = qvel[75:81]  # Box velocity (6 elements)
        
        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                balance_object,
                box_pos,
                box_vel,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (26,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :26].unsqueeze(1)
        h_joints = flat_obs[:, 26:].view(batch_size, 69, 5)
        return h_joints, h_objects

class SitHard(CustomObservationH1Hand, SitHardV0):
    base_task_name = "sit_hard"
    num_joints = 69
    
    @property
    def observation_space(self):
        # Object features: pelvis (7) + chair (4) + pelvis_vel (6) + chair_vel (6) = 23
        # Joint features: 69 joints × 5 features = 345
        # Total: 23 + 345 = 368
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(368,),
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

        pelvis = qpos[0:7]
        chair_pos = qpos[76:80]  # Chair position
        pelvis_velocities = qvel[0:6]
        chair_vel = qvel[75:81]  # Chair velocity
        
        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                chair_pos,
                pelvis_velocities,
                chair_vel,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (23,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :23].unsqueeze(1)
        h_joints = flat_obs[:, 23:].view(batch_size, 69, 5)
        return h_joints, h_objects


class Reach(CustomObservationH1Hand, ReachV0):
    base_task_name = "reach"
    num_joints = 69
   
    
    @property
    def observation_space(self):
        # Object features: pelvis (7) + pelvis_vel (6) + left_hand_pos (3) + goal (3) = 19
        # Joint features: 69 joints × 5 features = 345
        # Total: 19 + 345 = 364
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(364,),
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

        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        left_hand_pos = self.robot.left_hand_position()
        goal = self.goal
        
        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                left_hand_pos,
                goal,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (19,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :19].unsqueeze(1)
        h_joints = flat_obs[:, 19:].view(batch_size, 69, 5)
        return h_joints, h_objects


class Push(CustomObservationH1Hand, PushV0):
    base_task_name = "push"
    num_joints = 69
    
    @property
    def observation_space(self):
        # Object features: pelvis (7) + pelvis_vel (6) + box_pos (3) + box_quat (4) + box_vel (3) + hand_pos (3) + goal (3) = 29
        # Joint features: 69 joints × 5 features = 345
        # Total: 29 + 345 = 374
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(374,),
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
        
        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        box_position = qpos[-7:-4]
        box_quaternion = qpos[-4:]
        dofadr = self._env.named.model.body_dofadr["object"]
        box_linear_vel = qvel[dofadr : dofadr + 3]
        left_hand_pos = self.robot.left_hand_position()
        goal = self.goal

        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                box_position,
                box_quaternion,
                box_linear_vel,
                left_hand_pos,
                goal,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (29,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :29].unsqueeze(1)
        h_joints = flat_obs[:, 29:].view(batch_size, 69, 5)
        return h_joints, h_objects


class Door(CustomObservationH1Hand, DoorV0):
    base_task_name = "door"
    num_joints = 69
    
    @property
    def observation_space(self):
        # Object features: pelvis (7) + pelvis_vel (6) + door_pos (2) + door_vel (2) = 17
        # Joint features: 69 joints × 5 features = 345
        # Total: 17 + 345 = 362
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(362,),
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
        
        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        door_pos = np.array([qpos[76], qpos[77]])  # Door position
        door_vel = np.array([qvel[75], qvel[76]])  # Door velocity
        
        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                door_pos,
                door_vel,
                joint_features.flatten(),
            ]
        )
        
    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (17,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :17].unsqueeze(1)
        h_joints = flat_obs[:, 17:].view(batch_size, 69, 5)
        return h_joints, h_objects
     

class Window(CustomObservationH1Hand, WindowV0):
    base_task_name = "window"
    num_joints = 69

    @property
    def observation_space(self):
        # Object features: pelvis (7) + pelvis_vel (6) + tool_pos_quat (7) + tool_vel (6) + head_pos (3) = 29
        # Joint features: 69 joints × 5 features = 345
        # Total: 29 + 345 = 374
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(374,),
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

        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        tool_pos_quat = qpos[76:83]   # wiping tool freejoint (3 pos + 4 quat)
        tool_vel = qvel[75:81]        # wiping tool velocity (6 elements)
        head_pos = self._env.named.data.site_xpos["head"].copy()

        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                tool_pos_quat,
                tool_vel,
                head_pos,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (29,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :29].unsqueeze(1)
        h_joints = flat_obs[:, 29:].view(batch_size, 69, 5)
        return h_joints, h_objects


class BalanceHard(CustomObservationH1Hand, BalanceHardV0):
    base_task_name = "balance_hard"
    num_joints = 69

    @property
    def observation_space(self):
        # Object features: pelvis (7) + pelvis_vel (6) + obj1_pos (7) + obj1_vel (6) + obj2_pos (7) + obj2_vel (6) = 39
        # Joint features: 69 joints × 5 features = 345
        # Total: 39 + 345 = 384
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(384,),
            dtype=np.float64,
        )

    @staticmethod
    def get_object_feature_dim():
        return 39

    def get_obs(self) -> np.ndarray:
        qpos = self._env.data.qpos.flat.copy()
        qvel = self._env.data.qvel.flat.copy()
        xanchor = self._env.data.xanchor.copy()
        xanchor = (xanchor[:, :] - xanchor[0, :]) / 8

        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        obj1_pos = qpos[76:83]   # first balance object (3 pos + 4 quat)
        obj1_vel = qvel[75:81]   # first balance object velocity (6 elements)
        obj2_pos = qpos[83:90]   # second balance object (3 pos + 4 quat)
        obj2_vel = qvel[81:87]   # second balance object velocity (6 elements)

        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                obj1_pos,
                obj1_vel,
                obj2_pos,
                obj2_vel,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (39,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :39].unsqueeze(1)
        h_joints = flat_obs[:, 39:].view(batch_size, 69, 5)
        return h_joints, h_objects


class Basketball(CustomObservationH1Hand, BasketballV0):
    base_task_name = "basketball"
    num_joints = 69

    @property
    def observation_space(self):
        # Object features: pelvis (7) + pelvis_vel (6) + ball_pos (3) + ball_quat (4) + ball_vel (6) = 26
        # Joint features: 69 joints × 5 features = 345
        # Total: 26 + 345 = 371
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(371,),
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

        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        ball_pos = qpos[76:79]    # basketball position (3 elements)
        ball_quat = qpos[79:83]   # basketball quaternion (4 elements)
        ball_vel = qvel[75:81]    # basketball velocity (6 elements)

        joint_positions = qpos[7:76]
        joint_velocities = qvel[6:75]
        joint_x = xanchor[1:70, :]

        joint_features = np.column_stack([joint_positions, joint_velocities, joint_x])
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                ball_pos,
                ball_quat,
                ball_vel,
                joint_features.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (26,),
            "joint_features": (69, 5),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        batch_size = flat_obs.shape[0]
        h_objects = flat_obs[:, :26].unsqueeze(1)
        h_joints = flat_obs[:, 26:].view(batch_size, 69, 5)
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
    "h1hand-balance_hard-v1": BalanceHard,
    "h1hand-reach-v1": Reach,
    "h1hand-push-v1": Push,
    "h1hand-door-v1": Door,
    "h1hand-window-v1": Window,
    "h1hand-basketball-v1": Basketball,
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
