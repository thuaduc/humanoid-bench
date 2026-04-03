"""
Kinematic type definitions for topology-aware transformer embeddings.

A "joint type" groups joints that play the same biomechanical role
(e.g., left_knee and right_knee share the same type ID). This lets the
transformer share embedding weights across symmetric counterparts,
encoding the robot's kinematic topology as an inductive bias rather
than learning it from scratch via positional IDs.

Usage
-----
    from fast_td3.actors.transformer.kinematic_types import build_token_type_ids, NUM_JOINT_TYPES

    type_ids = build_token_type_ids(robot="h1hand", num_joints=69)  # shape (70,)
    embedding = nn.Embedding(NUM_JOINT_TYPES, hidden_nf)

Robot joint counts
------------------
    h1      : 19 (body only)
    h1hand  : 19 (custom env — body joints only, hand excluded from obs)
              69 (full robot: body + 2 × shadow hand)
"""

import torch

# ── Body joint type IDs ───────────────────────────────────────────────────────
HIP_YAW        = 0
HIP_ROLL       = 1
HIP_PITCH      = 2
KNEE           = 3
ANKLE          = 4
TORSO          = 5
SHOULDER_PITCH = 6
SHOULDER_ROLL  = 7
SHOULDER_YAW   = 8
ELBOW          = 9
WRIST_YAW      = 10  # h1hand arm wrist (left_wrist_yaw / right_wrist_yaw)

# ── Shadow hand joint type IDs ────────────────────────────────────────────────
# Each hand: WRJ2, WRJ1 (wrist), then 4 fingers × 4 joints + little × 5 + thumb × 5
WRIST_2          = 11  # lh/rh_WRJ2 — wrist deviation
WRIST_1          = 12  # lh/rh_WRJ1 — wrist flexion
FINGER_ABDUCTION = 13  # *J4 (MCP abduction/adduction) and LFJ5 (little CMC spread)
FINGER_MCP       = 14  # *J3 (MCP flexion)
FINGER_PIP       = 15  # *J2 (PIP flexion)
FINGER_DIP       = 16  # *J1 (DIP flexion)
THUMB_CMC_ABD    = 17  # THJ5 (CMC abduction / palmar arch)
THUMB_CMC_FLEX   = 18  # THJ4 (CMC flexion)
THUMB_MCP_ABD    = 19  # THJ3 (MCP abduction / rotation)
THUMB_MCP_FLEX   = 20  # THJ2 (MCP flexion)
THUMB_IP         = 21  # THJ1 (IP flexion)

# ── Global / task-context token ───────────────────────────────────────────────
GLOBAL_TOKEN    = 22

NUM_JOINT_TYPES = 23  # types 0-22

# ── Per-robot joint type sequences ───────────────────────────────────────────
# Ordered to match MuJoCo's qpos layout (DOFs after the free base joint).

# H1 body: 19 joints
# qpos[7:26] after the free base
_H1_TYPES: list[int] = [
    HIP_YAW, HIP_ROLL, HIP_PITCH, KNEE, ANKLE,          # left leg     (0-4)
    HIP_YAW, HIP_ROLL, HIP_PITCH, KNEE, ANKLE,          # right leg    (5-9)
    TORSO,                                                # torso        (10)
    SHOULDER_PITCH, SHOULDER_ROLL, SHOULDER_YAW, ELBOW,  # left arm     (11-14)
    SHOULDER_PITCH, SHOULDER_ROLL, SHOULDER_YAW, ELBOW,  # right arm    (15-18)
]  # 19 joints

# H1Hand full robot: 69 joints
# Layout verified against mujoco model (nv=87, njnt=72 incl. free + task objects):
#   joints 0-15  : h1 body left side (leg + torso + left arm + left_wrist_yaw)
#   joints 16-39 : left shadow hand  (WRJ2, WRJ1, then 4 fingers + little + thumb)
#   joints 40-44 : h1 body right arm (shoulder × 3 + elbow + right_wrist_yaw)
#   joints 45-68 : right shadow hand (mirror of left)
_H1HAND_TYPES_69: list[int] = [
    # ── Left leg (0-4) ───────────────────────────────────────────────────────
    HIP_YAW, HIP_ROLL, HIP_PITCH, KNEE, ANKLE,
    # ── Right leg (5-9) ──────────────────────────────────────────────────────
    HIP_YAW, HIP_ROLL, HIP_PITCH, KNEE, ANKLE,
    # ── Torso (10) ───────────────────────────────────────────────────────────
    TORSO,
    # ── Left arm (11-15): shoulder × 3, elbow, h1-wrist ─────────────────────
    SHOULDER_PITCH, SHOULDER_ROLL, SHOULDER_YAW, ELBOW, WRIST_YAW,
    # ── Left shadow hand wrist (16-17) ───────────────────────────────────────
    WRIST_2, WRIST_1,
    # ── Left shadow hand fingers (18-39) ─────────────────────────────────────
    # Index finger:  FFJ4(abd), FFJ3(mcp), FFJ2(pip), FFJ1(dip)
    FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,
    # Middle finger: MFJ4(abd), MFJ3(mcp), MFJ2(pip), MFJ1(dip)
    FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,
    # Ring finger:   RFJ4(abd), RFJ3(mcp), RFJ2(pip), RFJ1(dip)
    FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,
    # Little finger: LFJ5(cmc_spread), LFJ4(abd), LFJ3(mcp), LFJ2(pip), LFJ1(dip)
    FINGER_ABDUCTION, FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,
    # Thumb:         THJ5(cmc_abd), THJ4(cmc_flex), THJ3(mcp_abd), THJ2(mcp_flex), THJ1(ip)
    THUMB_CMC_ABD, THUMB_CMC_FLEX, THUMB_MCP_ABD, THUMB_MCP_FLEX, THUMB_IP,
    # ── Right arm (40-44): shoulder × 3, elbow, h1-wrist ─────────────────────
    SHOULDER_PITCH, SHOULDER_ROLL, SHOULDER_YAW, ELBOW, WRIST_YAW,
    # ── Right shadow hand wrist (45-46) ──────────────────────────────────────
    WRIST_2, WRIST_1,
    # ── Right shadow hand fingers (47-68): mirror of left ────────────────────
    FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,   # index
    FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,   # middle
    FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,   # ring
    FINGER_ABDUCTION, FINGER_ABDUCTION, FINGER_MCP, FINGER_PIP, FINGER_DIP,  # little
    THUMB_CMC_ABD, THUMB_CMC_FLEX, THUMB_MCP_ABD, THUMB_MCP_FLEX, THUMB_IP,  # thumb
]  # 69 joints

assert len(_H1_TYPES) == 19
assert len(_H1HAND_TYPES_69) == 69

# Registry: (robot_name, num_joints) → joint type id list
_REGISTRY: dict[tuple[str, int], list[int]] = {
    ("h1",     19): _H1_TYPES,
    # custom_env_h1hand.py slices qpos[7:26], exposing only the 19 body joints
    ("h1hand", 19): _H1_TYPES,
    # full h1hand observation (body + shadow hands)
    ("h1hand", 69): _H1HAND_TYPES_69,
}


def get_joint_type_ids(robot: str, num_joints: int) -> list[int]:
    """Return the joint type id list for the given robot and joint count.

    Raises ValueError for unsupported (robot, num_joints) combinations.
    """
    key = (robot, num_joints)
    if key not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"No kinematic type map for robot='{robot}', num_joints={num_joints}. "
            f"Available: {available}"
        )
    return _REGISTRY[key]


def build_token_type_ids(robot: str, num_joints: int) -> torch.Tensor:
    """Return a long tensor of shape (num_joints + 1,).

    The first num_joints entries are joint type IDs; the last entry is
    GLOBAL_TOKEN (the task-context / object token).
    """
    joint_types = get_joint_type_ids(robot, num_joints)
    return torch.tensor(joint_types + [GLOBAL_TOKEN], dtype=torch.long)
