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
        pelvis_velocities = qvel[:6]  # wx, wy, wz

        # Extract joint state (excluding free joint)
        joint_positions = qpos[7:]
        joint_velocities = qvel[6:]
        joint_x = xanchor[1:, :] - xanchor[0, :]  # relative to pelvis

        # Concatenate into flat vector (no interleaving for performance)
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                joint_positions,
                joint_velocities,
                joint_x.flatten(),
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (13,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }

    @staticmethod
    def get_object_feature_dim():
        """Return total object feature dimension: 13 * num_objects"""
        return 13

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        # flat_obs: (batch, obs_dim)
        # Extract separate blocks: positions [13:32], velocities [32:51]
        joint_positions = flat_obs[:, 13:32]  # (batch, 19)
        joint_velocities = flat_obs[:, 32:51]  # (batch, 19)
        
        # Stack to (batch, 19, 2) format [position, velocity] then reshape to (batch*19, 2)
        h_joints = torch.stack([joint_positions, joint_velocities], dim=2).reshape(-1, 2)
        
        # Reshape joint coordinates: (batch*19, 3)
        x_joints = flat_obs[:, 51:].reshape(-1, 3)
        
        # h_objects: (batch, 13)
        h_objects = flat_obs[:, 0:13]
        
        return h_joints, x_joints, h_objects


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
    def get_obs_shapes():
        return {
            "object_features": (26,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        # Extract separate blocks: positions [26:45], velocities [45:64]
        joint_positions = flat_obs[:, 26:45]  # (batch, 19)
        joint_velocities = flat_obs[:, 45:64]  # (batch, 19)
        
        # Stack to (batch, 19, 2) format [position, velocity] then reshape to (batch*19, 2)
        h_joints = torch.stack([joint_positions, joint_velocities], dim=2).reshape(-1, 2)
        
        h_objects = flat_obs[:, 0:26] 
        x_joints = flat_obs[:, 64:121].reshape(-1, 3)
       
        return h_joints, x_joints, h_objects

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
    def get_obs_shapes():
        return {
            "object_features": (26,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        # flat_obs: (batch, obs_dim)
        # Extract separate blocks: positions [26:45], velocities [45:64]
        joint_positions = flat_obs[:, 26:45]  # (batch, 19)
        joint_velocities = flat_obs[:, 45:64]  # (batch, 19)
        
        # Stack to (batch, 19, 2) format [position, velocity] then reshape to (batch*19, 2)
        h_joints = torch.stack([joint_positions, joint_velocities], dim=2).reshape(-1, 2)
        
        # Reshape joint coordinates: (batch*19, 3)
        x_joints = flat_obs[:, 64:].reshape(-1, 3)
        
        # h_objects: (batch, 26)
        h_objects = flat_obs[:, 0:26]
        
        return h_joints, x_joints, h_objects


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
    def get_obs_shapes():
        return {
            "object_features": (19,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        # flat_obs: (batch, obs_dim)
        # Extract separate blocks: positions [19:38], velocities [38:57]
        joint_positions = flat_obs[:, 19:38]  # (batch, 19)
        joint_velocities = flat_obs[:, 38:57]  # (batch, 19)
        
        # Stack to (batch, 19, 2) format [position, velocity] then reshape to (batch*19, 2)
        h_joints = torch.stack([joint_positions, joint_velocities], dim=2).reshape(-1, 2)
        
        # Reshape joint coordinates: (batch*19, 3)
        x_joints = flat_obs[:, 57:114].reshape(-1, 3)
        
        # h_objects: (batch, 19)
        h_objects = flat_obs[:, 0:19]
        
        return h_joints, x_joints, h_objects


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

        # Concatenate with hard-coded order: pelvis first, box second
        return np.concatenate(
            [
                qpos[0:7],                # [0:7] pelvis position and quaternion
                qvel[0:6],                # [7:13] pelvis velocities
                box_position,             # [13:16] box position
                box_quaternion,           # [16:20] box quaternion
                box_linear_vel,           # [20:23] box linear velocity
                self.robot.left_hand_position(),  # [23:26] object_others: left_hand
                self.goal,                # [26:29] object_others: target
                box_position,             # [29:32] object_others: box
                box_linear_vel,           # [32:35] object_others: box_vel
                qpos[7:26],               # [35:54] joint positions
                qvel[6:25],               # [54:73] joint velocities
                xanchor[1:20, :].flatten(),  # [73:130] joint_x
            ]
        )

    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (35,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }

    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):
        # flat_obs: (batch, obs_dim)
        # Extract separate blocks: positions [35:54], velocities [54:73]
        joint_positions = flat_obs[:, 35:54]  # (batch, 19)
        joint_velocities = flat_obs[:, 54:73]  # (batch, 19)
        
        # Stack to (batch, 19, 2) format [position, velocity] then reshape to (batch*19, 2)
        h_joints = torch.stack([joint_positions, joint_velocities], dim=2).reshape(-1, 2)
        
        x_joints = flat_obs[:, 73:130].reshape(-1, 3)
        h_objects = flat_obs[:, 0:35]
        
        return h_joints, x_joints, h_objects


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
        
        pelvis = qpos[0:7]
        pelvis_velocities = qvel[0:6]
        
        door_hinge_pos = qpos[26]
        door_hatch_hinge_pos = qpos[27]
        door_hinge_vel = qvel[25]
        door_hatch_hinge_vel = qvel[26]
        
        return np.concatenate(
            [
                pelvis,
                pelvis_velocities,
                np.array([door_hinge_pos, door_hatch_hinge_pos]),
                np.array([door_hinge_vel, door_hatch_hinge_vel]),
                joint_positions,
                joint_velocities,
                joint_x.flatten(),
            ]
        )
        
    @staticmethod
    def get_obs_shapes():
        return {
            "object_features": (17,),
            "joint_positions": (19,),
            "joint_velocities": (19,),
            "joint_x": (19, 3),
        }
        
    @staticmethod
    @torch.jit.script
    def unflatten_obs(flat_obs):     
        # flat_obs: (batch, obs_dim)
        # Extract separate blocks: positions [17:36], velocities [36:55]
        joint_positions = flat_obs[:, 17:36]  # (batch, 19)
        joint_velocities = flat_obs[:, 36:55]  # (batch, 19)
        
        # Stack to (batch, 19, 2) format [position, velocity] then reshape to (batch*19, 2)
        h_joints = torch.stack([joint_positions, joint_velocities], dim=2).reshape(-1, 2)
        
        # Reshape joint coordinates: (batch*19, 3)
        x_joints = flat_obs[:, 55:].reshape(-1, 3)
        
        # h_objects: (batch, 17)
        h_objects = flat_obs[:, 0:17]
        
        return h_joints, x_joints, h_objects
     

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
