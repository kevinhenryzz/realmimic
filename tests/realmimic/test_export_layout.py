"""Contract: the training network's inputs/outputs equal SONIC's decoder interface after the export
layout (gather + gain in, permutation + gain out). Both sides are computed independently: training
side with the supermimic code paths, SONIC side with realmimic.obs.sonic."""

from __future__ import annotations

import re

import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME

from realmimic.export.layout import FRAMES, HISTORY, SONIC_OBS_ORDER, decoder_layout
from realmimic.obs import sonic
from realmimic.robots import get_robot_spec
from realmimic.robots.g1 import joint_gains
from supermimic.kinematics.rotations import anchor_window_obs, projected_gravity
from supermimic.world_model.heads import flatten_history_term_major

N = 4
NAMES = list(sonic.JOINT_NAMES_MUJOCO)


def _mjlab_default_pose() -> np.ndarray:
    q0 = np.zeros(len(NAMES))
    for i, name in enumerate(NAMES):
        for pattern, value in KNEES_BENT_KEYFRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                q0[i] = value
    return q0


def _rand_quat(*shape: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(*shape, 4, generator=gen, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def test_training_default_pose_is_sonic_default():
    np.testing.assert_allclose(_mjlab_default_pose(), sonic.DEFAULT_ANGLES, atol=0)


def test_decoder_dims():
    _, _, scale = joint_gains(get_robot_spec("g1_brainco"), NAMES)
    layout = decoder_layout(scale)
    assert layout.sonic_dim == 1580
    assert layout.gather.shape == (HISTORY * 93 + FRAMES * 59 + FRAMES * 6,)
    assert len(set(layout.gather.tolist())) == layout.sonic_dim  # a bijection onto obs_dict


def test_inputs_match_sonic_terms():
    gen = torch.Generator().manual_seed(0)
    _, _, scale_np = joint_gains(get_robot_spec("g1_brainco"), NAMES)
    layout = decoder_layout(scale_np)
    q0 = torch.tensor(_mjlab_default_pose())
    train_scale = torch.tensor(scale_np)
    sonic_scale = torch.tensor(sonic.ACTION_SCALE)

    # Robot history (oldest -> newest) and reference window (current -> future).
    q = q0 + 0.3 * torch.randn(N, HISTORY, 29, generator=gen, dtype=torch.float64)
    qd = torch.randn(N, HISTORY, 29, generator=gen, dtype=torch.float64)
    w = torch.randn(N, HISTORY, 3, generator=gen, dtype=torch.float64)
    base_quat = _rand_quat(N, HISTORY, gen=gen)
    a_train = torch.randn(N, HISTORY, 29, generator=gen, dtype=torch.float64)
    q_ref = q0 + 0.3 * torch.randn(N, FRAMES, 29, generator=gen, dtype=torch.float64)
    qd_ref = torch.randn(N, FRAMES, 29, generator=gen, dtype=torch.float64)
    root_pos_ref = torch.randn(N, FRAMES, 3, generator=gen, dtype=torch.float64)
    root_quat_ref = _rand_quat(N, FRAMES, gen=gen)

    # Training side (supermimic code paths).
    obs = torch.cat([q - q0, qd, w, projected_gravity(base_quat), a_train], dim=-1)  # (N, T, 93)
    hist = flatten_history_term_major(obs.transpose(0, 1))
    goal = torch.cat([q_ref, qd_ref, root_pos_ref[..., 2:3]], dim=-1).reshape(N, -1)
    anchor = anchor_window_obs(base_quat[:, -1], root_quat_ref)
    train_in = torch.cat([hist, goal, anchor], dim=-1)

    # SONIC side (realmimic.obs.sonic); its last-action history holds SONIC action units.
    a_sonic = a_train * train_scale / sonic_scale
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand(N, 4)
    terms = {
        "his_base_angular_velocity_10frame_step1": sonic.his_base_angular_velocity(w),
        "his_body_joint_positions_10frame_step1": sonic.his_body_joint_positions(q),
        "his_body_joint_velocities_10frame_step1": sonic.his_body_joint_velocities(qd),
        "his_last_actions_10frame_step1": sonic.his_last_actions(a_sonic),
        "his_gravity_dir_10frame_step1": sonic.his_gravity_dir(base_quat),
        "motion_joint_positions_10frame_step1": sonic.motion_joint_positions(q_ref),
        "motion_joint_velocities_10frame_step1": sonic.motion_joint_velocities(qd_ref),
        "motion_root_z_position_10frame_step1": sonic.motion_root_z_position(root_pos_ref),
        "motion_anchor_orientation_10frame_step1": sonic.motion_anchor_orientation(
            base_quat[:, -1], root_quat_ref, identity, "base"
        ),
    }
    sonic_in = sonic.assemble_decoder_input(terms, SONIC_OBS_ORDER)

    mapped = sonic_in[:, torch.from_numpy(layout.gather)] * torch.from_numpy(layout.gain).double()
    torch.testing.assert_close(mapped, train_in, atol=1e-6, rtol=1e-6)


def test_action_mapping_reproduces_training_targets():
    gen = torch.Generator().manual_seed(1)
    _, _, scale_np = joint_gains(get_robot_spec("g1_brainco"), NAMES)
    layout = decoder_layout(scale_np)
    a_train = torch.randn(N, 29, generator=gen, dtype=torch.float64)
    q0 = torch.tensor(_mjlab_default_pose())
    sonic_action = a_train[:, torch.from_numpy(layout.action_perm)] * torch.from_numpy(
        layout.action_gain
    ).double()
    torch.testing.assert_close(
        sonic.joint_position_targets(sonic_action), q0 + a_train * torch.tensor(scale_np)
    )


def test_hip_pitch_is_the_only_rescaled_joint_for_old_hip_motor():
    _, _, scale = joint_gains(get_robot_spec("g1_brainco"), NAMES)
    layout = decoder_layout(scale)
    rescaled = {sonic.JOINT_NAMES_ISAACLAB[i] for i in np.flatnonzero(np.abs(layout.action_gain - 1) > 1e-6)}
    assert rescaled == {"left_hip_pitch_joint", "right_hip_pitch_joint"}
