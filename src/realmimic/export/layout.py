"""Mapping between the training network's input/output layout and SONIC's decoder interface.

Training (supermimic, ``goal_features="sonic"``, ``anchor_obs="pelvis_ori10"``) feeds the policy
  history (930): term-major [joint_pos-q0 | joint_vel | base_ang_vel | projected_gravity | action],
                 10 frames oldest->newest per term, joints in MuJoCo order, actions in training units;
  goal (650):    10 frames x [ref joint_pos (abs) 29 | ref joint_vel 29 | ref pelvis z 1], frame-major,
                 then 10 x rot6d anchor orientation.
SONIC builds ``obs_dict`` from the terms in ``SONIC_OBS_ORDER`` (IsaacLab joint order, term-major,
see docs/sonic-observation-contract.md). Every training input element is one SONIC element
times a per-element gain (non-unit only for last actions on joints whose action scale differs from
SONIC's compile-time constant). The exported ONNX applies this gather + gain in front of the network
and the inverse permutation/scaling behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from realmimic.obs import sonic

HISTORY = 10
FRAMES = 10
NJ = 29

SONIC_OBS_ORDER = (
    "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1",
    "his_gravity_dir_10frame_step1",
    "motion_joint_positions_10frame_step1",
    "motion_joint_velocities_10frame_step1",
    "motion_root_z_position_10frame_step1",
    "motion_anchor_orientation_10frame_step1",
)


def _offsets() -> dict[str, int]:
    out, pos = {}, 0
    for name in SONIC_OBS_ORDER:
        out[name] = pos
        pos += sonic.TERM_DIMS[name]
    return out


@dataclass(frozen=True)
class DecoderLayout:
    """``train_input = sonic_obs[gather] * gain``; ``sonic_action = train_action[action_perm] * action_gain``."""

    gather: np.ndarray  # (1580,) int64: SONIC obs_dict index of each training input element
    gain: np.ndarray  # (1580,) float32
    action_perm: np.ndarray  # (29,) int64: training (MuJoCo) index of each SONIC (IsaacLab) action
    action_gain: np.ndarray  # (29,) float32: train_scale / sonic_scale per SONIC action element

    @property
    def sonic_dim(self) -> int:
        return sonic.decoder_input_dim(SONIC_OBS_ORDER)


def decoder_layout(train_action_scale_mujoco: np.ndarray) -> DecoderLayout:
    """Build the layout for a policy trained with per-joint action scale ``train_action_scale_mujoco``
    (MuJoCo order); SONIC multiplies the network output by ``sonic.ACTION_SCALE`` instead."""
    train_scale = np.asarray(train_action_scale_mujoco, dtype=np.float64)
    sonic_scale = np.asarray(sonic.ACTION_SCALE, dtype=np.float64)
    off = _offsets()
    # IsaacLab position of each MuJoCo joint.
    il_pos = np.empty(NJ, dtype=np.int64)
    il_pos[list(sonic.MUJOCO_TO_ISAACLAB)] = np.arange(NJ)

    gather: list[int] = []
    gain: list[float] = []

    def joint_history(term: str, scale: np.ndarray | None = None) -> None:
        for t in range(HISTORY):  # training: per term, frames oldest->newest, MuJoCo joints
            for j in range(NJ):
                gather.append(off[term] + t * NJ + il_pos[j])
                gain.append(1.0 if scale is None else float(scale[j]))

    def vec_history(term: str, dim: int) -> None:
        for t in range(HISTORY):
            for d in range(dim):
                gather.append(off[term] + t * dim + d)
                gain.append(1.0)

    joint_history("his_body_joint_positions_10frame_step1")
    joint_history("his_body_joint_velocities_10frame_step1")
    vec_history("his_base_angular_velocity_10frame_step1", 3)
    vec_history("his_gravity_dir_10frame_step1", 3)
    # SONIC stores its own (scaled) action units: a_train = a_sonic * sonic_scale / train_scale.
    joint_history("his_last_actions_10frame_step1", sonic_scale / train_scale)

    for k in range(FRAMES):  # goal: frame-major
        for j in range(NJ):
            gather.append(off["motion_joint_positions_10frame_step1"] + k * NJ + il_pos[j])
            gain.append(1.0)
        for j in range(NJ):
            gather.append(off["motion_joint_velocities_10frame_step1"] + k * NJ + il_pos[j])
            gain.append(1.0)
        gather.append(off["motion_root_z_position_10frame_step1"] + k)
        gain.append(1.0)
    for i in range(FRAMES * 6):
        gather.append(off["motion_anchor_orientation_10frame_step1"] + i)
        gain.append(1.0)

    action_perm = np.asarray(sonic.MUJOCO_TO_ISAACLAB, dtype=np.int64)
    action_gain = (train_scale / sonic_scale)[action_perm]
    return DecoderLayout(
        gather=np.asarray(gather, dtype=np.int64),
        gain=np.asarray(gain, dtype=np.float32),
        action_perm=action_perm,
        action_gain=action_gain.astype(np.float32),
    )
