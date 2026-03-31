import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import humanoid_bench.dmc_deps.dmc_index as index
import collections
NamedIndexStructs = collections.namedtuple(
    'NamedIndexStructs', ['model', 'data'])

from dm_control.utils import rewards

from humanoid_bench.dmc_deps.dmc_wrapper import MjDataWrapper, MjModelWrapper

from .wrappers import (
    SingleReachWrapper,
    DoubleReachAbsoluteWrapper,
    DoubleReachRelativeWrapper,
    BlockedHandsLocoWrapper,
    ObservationWrapper,
)

from .robots import H1, H1Hand, H1SimpleHand, H1Touch, H1Strong, G1
from .envs.cube import Cube
from .envs.bookshelf import BookshelfSimple, BookshelfHard
from .envs.window import Window
from .envs.spoon import Spoon
from .envs.door import Door
from .envs.basketball import Basketball
from .envs.basic_locomotion_envs import (
    Stand,
    Walk,
    Run,
    Hurdle,
    Crawl,
    Sit,
    SitHard,
    Stair,
    Slide,
)
from .envs.custom_env import (
    Stand as StandV1,
    Walk as WalkV1,
    Run as RunV1,
    Hurdle as HurdleV1,
    Crawl as CrawlV1,
    Sit as SitV1,
    Stair as StairV1,
    Slide as SlideV1,
    BalanceSimple as BalanceSimpleV1,
    SitHard as SitHardV1,
    Reach as ReachV1,
    Push as PushV1,
    Door as DoorV1,
)
from .envs.custom_env_h1hand import (
    Stand as StandH1HandV1,
    Walk as WalkH1HandV1,
    Run as RunH1HandV1,
    Hurdle as HurdleH1HandV1,
    Crawl as CrawlH1HandV1,
    Sit as SitH1HandV1,
    Stair as StairH1HandV1,
    Slide as SlideH1HandV1,
    BalanceSimple as BalanceSimpleH1HandV1,
    SitHard as SitHardH1HandV1,
    Reach as ReachH1HandV1,
    Push as PushH1HandV1,
    Door as DoorH1HandV1,
)
from .envs.reach import Reach
from .envs.pole import Pole
from .envs.push import Push
from .envs.maze import Maze
from .envs.highbar import HighBarSimple, HighBarHard
from .envs.kitchen import Kitchen
from .envs.truck import Truck
from .envs.package import Package
from .envs.cabinet import Cabinet
from .envs.balance import BalanceHard, BalanceSimple
from .envs.room import Room
from .envs.powerlift import Powerlift
from .envs.insert import Insert

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}
DEFAULT_RANDOMNESS = 0.01

ROBOTS = {"h1": H1, "h1hand": H1Hand, "h1simplehand": H1SimpleHand, "h1strong": H1Strong, "h1touch": H1Touch, "g1": G1}

# Original tasks
_TASKS_ORIGINAL = {
    "stand": Stand,
    "walk": Walk,
    "run": Run,
    "kitchen": Kitchen,
    "maze": Maze,
    "hurdle": Hurdle,
    "cube": Cube,
    "bookshelf_simple": BookshelfSimple,
    "bookshelf_hard": BookshelfHard,
    "highbar_simple": HighBarSimple,
    "highbar_hard": HighBarHard,
    "crawl": Crawl,
    "window": Window,
    "spoon": Spoon,
    "door": Door,
    "push": Push,
    "reach": Reach,
    "basketball": Basketball,
    "truck": Truck,
    "package": Package,
    "cabinet": Cabinet,
    "sit_simple": Sit,
    "sit_hard": SitHard,
    "balance_simple": BalanceSimple,
    "balance_hard": BalanceHard,
    "stair": Stair,
    "slide": Slide,
    "pole": Pole,
    "room": Room,
    "insert_normal": Insert,
    "insert_small": Insert,  # This is not an error
    "powerlift": Powerlift,
}

# Custom tasks with better naming convention
TASKS_CUSTOM = {
    "stand-v1": StandV1,
    "walk-v1": WalkV1,
    "run-v1": RunV1,
    "slide-v1": SlideV1,
    "crawl-v1": CrawlV1,
    "sit-v1": SitV1,
    "hurdle-v1": HurdleV1,
    "balance_simple-v1": BalanceSimpleV1,
    "stair-v1": StairV1,
    "sit_simple-v1": SitV1,
    "sit_hard-v1": SitHardV1,
    "reach-v1": ReachV1,
    "push-v1": PushV1,
    "door-v1": DoorV1,
}

# H1Hand custom tasks - separate dict to allow custom registration
TASKS_CUSTOM_H1HAND = {
    "h1hand-stand-v1": StandH1HandV1,
    "h1hand-walk-v1": WalkH1HandV1,
    "h1hand-run-v1": RunH1HandV1,
    "h1hand-slide-v1": SlideH1HandV1,
    "h1hand-crawl-v1": CrawlH1HandV1,
    "h1hand-sit-v1": SitH1HandV1,
    "h1hand-hurdle-v1": HurdleH1HandV1,
    "h1hand-balance_simple-v1": BalanceSimpleH1HandV1,
    "h1hand-stair-v1": StairH1HandV1,
    "h1hand-sit_simple-v1": SitH1HandV1,
    "h1hand-sit_hard-v1": SitHardH1HandV1,
    "h1hand-reach-v1": ReachH1HandV1,
    "h1hand-push-v1": PushH1HandV1,
    "h1hand-door-v1": DoorH1HandV1,
}

# Merged tasks dictionary (used by HumanoidEnv) - includes both h1 and h1hand variants
TASKS = {**_TASKS_ORIGINAL, **TASKS_CUSTOM, **TASKS_CUSTOM_H1HAND}


class HumanoidEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        robot=None,
        control=None,
        task=None,
        render_mode="rgb_array",
        width=256,
        height=256,
        randomness=DEFAULT_RANDOMNESS,
        **kwargs,
    ):
        assert robot and control and task, f"{robot} {control} {task}"
        gym.utils.EzPickle.__init__(self, metadata=self.metadata)

        asset_path = os.path.join(os.path.dirname(__file__), "assets")

        if "model_path" in kwargs:
            model_path = kwargs["model_path"]
        else:
            # Check if the task has a base_task_name for dict observation tasks
            task_class = TASKS.get(task) if isinstance(task, str) else task
            if hasattr(task_class, 'base_task_name') and task_class.base_task_name:
                base_task = task_class.base_task_name
                model_path = f"envs/{robot}_{control}_{base_task}.xml"
            else:
                model_path = f"envs/{robot}_{control}_{task}.xml"
        
        model_path = os.path.join(asset_path, model_path)

        self.robot = ROBOTS[robot](self)
        if isinstance(task, str):
            task_info = TASKS[task](self.robot, None, **kwargs)
        else:
            task_info = task(self.robot, None, **kwargs)

        self.obs_wrapper = kwargs.get("obs_wrapper", None)
        if self.obs_wrapper is not None:
            self.obs_wrapper = kwargs.get("obs_wrapper", "False").lower() == "true"
        else:
            self.obs_wrapper = False

        self.blocked_hands = kwargs.get("blocked_hands", None)
        if self.blocked_hands is not None:
            self.blocked_hands = kwargs.get("blocked_hands", "False").lower() == "true"
        else:
            self.blocked_hands = False

        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=task_info.frame_skip,
            observation_space=task_info.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name,
        )

        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )

        if isinstance(task, str):
            self.task = TASKS[task](self.robot, self, **kwargs)
        else:
            self.task = task(self.robot, self, **kwargs)

        if self.blocked_hands:
            self.task = BlockedHandsLocoWrapper(self.task, **kwargs)

        # Wrap for hierarchical control
        if (
            "policy_type" in kwargs
            and kwargs["policy_type"]
            and kwargs["policy_type"] is not None
            and kwargs["policy_type"] != "flat"
        ):
            if kwargs["policy_type"] == "reach_single":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = SingleReachWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_absolute":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachAbsoluteWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_relative":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachRelativeWrapper(self.task, **kwargs)
            else:
                raise ValueError(f"Unknown policy_type: {kwargs['policy_type']}")
        

        if self.obs_wrapper:
            # Note that observation wrapper is not compatible with hierarchical policy
            self.task = ObservationWrapper(self.task, **kwargs)
            self.observation_space = self.task.observation_space

        # Keyframe
        self.keyframe = (
            self.model.key(kwargs["keyframe"]).id if "keyframe" in kwargs else 0
        )

        self.randomness = randomness
        if isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)):
            self.randomness = 0
            print("No randomness in this env. This is the default behavior for (BookshelfHard, BookshelfSimple, Kitchen, Cube)")
        
        # Set up named indexing.
        data = MjDataWrapper(self.data)
        model = MjModelWrapper(self.model)
        axis_indexers = index.make_axis_indexers(model)
        self.named = NamedIndexStructs(
            model=index.struct_indexer(model, "mjmodel", axis_indexers),
            data=index.struct_indexer(data, "mjdata", axis_indexers),
        )

        assert self.robot.dof + self.task.dof == len(data.qpos), (
            self.robot.dof,
            self.task.dof,
            len(data.qpos),
        )

    def step(self, action):
        # print("world", self.data.xquat[0], self.data.xpos[0])
        # print("pelvis", self.data.xquat[1], self.data.xpos[1])
        # print("torso", self.data.xquat[12], self.data.xpos[12])
        # print("board", self.data.xquat[23], self.data.xpos[23])
        # print()
        
        return self.task.step(action)

    def reset(self, *, seed=None, options=None):
        # Handle options for randomization
        self._random_position = options.get("random_position", False) if options else False
        self._random_orientation = options.get("random_orientation", False) if options else False
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe)
        mujoco.mj_forward(self.model, self.data)

        def euler_to_quat(angles):
            cr, cp, cy = (
                np.cos(angles[0] / 2),
                np.cos(angles[1] / 2),
                np.cos(angles[2] / 2),
            )
            sr, sp, sy = (
                np.sin(angles[0] / 2),
                np.sin(angles[1] / 2),
                np.sin(angles[2] / 2),
            )
            return np.array(
                [
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                ]
            )

        # Add randomness
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        r = self.randomness
        
        if not (self._random_position or self._random_orientation):
            # Default randomness to all qpos
            # init_qpos += self.np_random.uniform(-r, r, size=self.model.nq)
            init_qpos += 0
        else:
            # reset robot to random position
            # if self._random_position:
                # init_qpos[:2] += self.np_random.uniform(-10, 10, size=2)
                
            # rotate robot randomly (orientation only, keep position centered)
            if self._random_orientation:
                # Rotate quaternions
                rotation_angle = np.pi / 2
                init_qpos[3:7] = euler_to_quat(np.array([0.0, 0.0, rotation_angle]))
                init_qpos[29:33] = euler_to_quat(np.array([0.0, 0.0, rotation_angle]))
                
                # Also rotate the xy position around the board center (0,0) to keep feet centered
                xy_pos = init_qpos[:2].copy()
                cos_angle = np.cos(rotation_angle)
                sin_angle = np.sin(rotation_angle)
                init_qpos[0] = xy_pos[0] * cos_angle - xy_pos[1] * sin_angle
                init_qpos[1] = xy_pos[0] * sin_angle + xy_pos[1] * cos_angle
                
        self.set_state(init_qpos, init_qvel)

        # Task-specific reset and return observations
        return self.task.reset_model()

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        return self.task.render()


if __name__ == "__main__":
    register(
        id="temp-v0",
        entry_point="humanoid_bench.env:HumanoidEnv",
        max_episode_steps=1000,
        kwargs={
            "robot": "h1hand",
            "control": "pos",
            "task": "maze_hard",
        },
    )

    env = gym.make("temp-v0", render_mode="human")
    ob, _ = env.reset()
    print(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    print(f"ac_space = {env.action_space.shape}")
    env.render()
    while True:
        action = env.action_space.sample()
        ob, rew, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            env.reset()
    env.close()
