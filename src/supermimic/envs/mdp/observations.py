from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand
from supermimic.kinematics.rotations import anchor_window_obs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def anchor_ori_window(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, 6K) reference pelvis orientation per goal frame relative to the robot pelvis (SONIC's
    motion_anchor_orientation_10frame_step1)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    ref = command.library.window_root_quat(command.frame, command.cfg.future_frames)
    return anchor_window_obs(command.robot.data.root_link_quat_w, ref)


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal
