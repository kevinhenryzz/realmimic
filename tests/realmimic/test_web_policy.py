"""The browser export (raw MuJoCo inputs, observations built in the graph) must reproduce the training
path: per-tick 93-d observations in the env's history buffer, the env's goal terms
(``goal_future_frames`` via ``MultiClipMotionCommand.goal``, ``anchor_ori_window``) on a motion
library, then the policy. Inputs are recorded from a CPU MuJoCo closed loop that tracks a synthetic
clip past its end (frozen reference frames)."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
REAL_CKPT = Path(os.environ.get("WEB_POLICY_CHECKPOINT", ROOT / "logs/release/ckpt_latest.pt"))


def _fake_checkpoint(path: Path) -> None:
    from dataclasses import asdict

    from supermimic.config import TrainConfig
    from supermimic.goal_spec import GOAL_FEAT_DIMS, goal_dim
    from supermimic.policy.nets import Policy

    cfg = TrainConfig()
    cfg.env.goal_features, cfg.env.anchor_obs, cfg.env.future_frames = "sonic", "pelvis_ori10", 10
    cfg.policy.goal_features, cfg.policy.goal_frame_dim = "sonic", GOAL_FEAT_DIMS["sonic"]
    cfg.policy.goal_dim = goal_dim(10, "sonic", "pelvis_ori10")
    cfg.policy.input_norm = "running"
    torch.manual_seed(0)
    policy = Policy(cfg.policy)
    for norm in (policy.encoders["motion"].norm, policy.decoder.norm):  # non-trivial normalisers
        norm.mean.normal_(0.0, 0.3)
        norm.var.uniform_(0.2, 3.0)
    torch.save({"policy": policy.state_dict(), "config": asdict(cfg), "iteration": 0}, path)


def _synthetic_clip(path: Path, q0: np.ndarray, frames: int = 40) -> None:
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    t = np.arange(frames) / 50.0
    phase = np.arange(29)
    q = q0 + 0.15 * np.sin(2 * np.pi * 0.8 * t[:, None] + phase)
    qd = 0.15 * 2 * np.pi * 0.8 * np.cos(2 * np.pi * 0.8 * t[:, None] + phase)
    yaw, pitch = 0.4 * t, 0.05 * np.sin(3 * t)
    quat = np.stack([np.cos(yaw / 2) * np.cos(pitch / 2), -np.sin(yaw / 2) * np.sin(pitch / 2), np.cos(yaw / 2) * np.sin(pitch / 2), np.sin(yaw / 2) * np.cos(pitch / 2)], -1)
    nb = len(TRACKED_BODIES)
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((frames, nb, 3)) * 0.1
    pos[:, 0] = np.stack([0.1 * t, 0.0 * t, 0.75 + 0.02 * np.sin(4 * t)], -1)
    bquat = np.repeat(quat[:, None], nb, axis=1)
    lin = np.zeros((frames, nb, 3))
    ang = np.zeros((frames, nb, 3))
    ang[:, 0, 2] = 0.4
    np.savez(path, joint_pos=q, joint_vel=qd, body_names=np.array(TRACKED_BODIES), body_pos_w=pos, body_quat_w=bquat, body_lin_vel_w=lin, body_ang_vel_w=ang)


class _Command:
    goal = None  # MultiClipMotionCommand.goal, bound in _env_goal


def _env_goal(lib, frame: torch.Tensor, robot_quat: torch.Tensor, k: int) -> torch.Tensor:
    """The env's goal group: ``goal_future_frames`` then ``anchor_ori10`` on a stand-in command."""
    from supermimic.envs.mdp import observations as sm_obs
    from supermimic.envs.mdp.commands import MultiClipMotionCommand

    _Command.goal = MultiClipMotionCommand.goal
    cmd = _Command()
    cmd.library, cmd.frame = lib, frame
    cmd.cfg = SimpleNamespace(future_frames=k, goal_features="sonic")
    cmd.robot = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=robot_quat))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: cmd))
    return torch.cat([sm_obs.goal_future_frames(env, "motion"), sm_obs.anchor_ori_window(env, "motion")], dim=-1)


def _check(checkpoint: Path, tmp_path: Path, ticks: int = 60) -> float:
    import onnxruntime as ort

    from realmimic.export.web_policy import build, export_onnx
    from realmimic.export.web_scene import physics_model, training_scene
    from realmimic.robots import get_robot_spec
    from realmimic.robots.g1 import joint_gains
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

    web, c, ck = build(checkpoint)
    onnx_path = tmp_path / "policy.onnx"
    export_onnx(web, c, onnx_path, encoding="fp32")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    policy, H, K = web.policy, web.policy.cfg.history, web.policy.cfg.future_frames
    _, _, scale = joint_gains(get_robot_spec(ck["config"]["env"].get("robot", "g1")), c.joint_names)

    clip = tmp_path / "clip.npz"
    _synthetic_clip(clip, c.default_joint_pos)
    lib = MotionLibrary([clip], TRACKED_BODIES, c.default_joint_pos, device="cpu", freeze_end=True)
    scene, cfg = training_scene()
    model = physics_model(scene, cfg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    act_jnt = model.actuator_trnid[:, 0] - 1
    q0 = torch.as_tensor(c.default_joint_pos, dtype=torch.float32)
    xi = torch.zeros(1, 33)

    def state32():
        return np.concatenate([data.qpos, data.qvel]).astype(np.float32)

    s = state32()
    prev = np.zeros(29, dtype=np.float32)
    obs = deque([analytic_observation(torch.from_numpy(s)[None], torch.from_numpy(prev)[None], xi, q0)] * H, maxlen=H)
    states, actions = deque([s] * H, maxlen=H), deque([prev] * H, maxlen=H)
    worst = 0.0
    for k in range(ticks):
        frame = torch.tensor([min(k, lib.num_frames - 1)])
        idx, frozen = lib._window_index(frame, K)  # as goal_window: the target frame holds at the clip end
        ref = lib.frozen_ref71(lib._ref[idx], frozen).numpy().astype(np.float32)
        with torch.no_grad():
            goal = _env_goal(lib, frame, torch.from_numpy(states[-1][3:7])[None], K)
            a = policy(flatten_history_term_major(torch.stack(list(obs))), goal)[0].numpy()
        x = {"state_hist": np.stack(states)[None], "action_hist": np.stack(actions)[None], "ref_window": ref}
        a_onnx, target = (y[0] for y in sess.run(["action", "joint_target"], x))
        worst = max(worst, float(np.abs(a_onnx - a).max()), float(np.abs(target - (a * scale + c.default_joint_pos)).max()))
        data.ctrl[:] = (a * scale + c.default_joint_pos)[act_jnt]
        for _ in range(4):
            mujoco.mj_step(model, data)
        s, prev = state32(), a.astype(np.float32)
        states.append(s)
        actions.append(prev)
        obs.append(analytic_observation(torch.from_numpy(s)[None], torch.from_numpy(prev)[None], xi, q0))
    return worst


def test_web_export_matches_training_path(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    _fake_checkpoint(ckpt)
    assert _check(ckpt, tmp_path) < 1e-5


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="no trained checkpoint (set WEB_POLICY_CHECKPOINT)")
def test_web_export_matches_training_path_trained(tmp_path):
    assert _check(REAL_CKPT, tmp_path) < 1e-5


def test_weight_encodings(tmp_path):
    """Each encoding's graph computes the policy with its decoded weights (float32 compute)."""
    from realmimic.export.web_policy import dequantize, export_web_policy, quantize

    w = np.random.default_rng(0).standard_normal((8, 100)).astype(np.float32)
    err = {}
    for bits, group in ((8, 0), (4, 0), (4, 32)):
        q, s = quantize(w, bits, group)
        assert np.abs(q).max() <= 2 ** (bits - 1) - 1 and q.shape[1] % (group or 100) == 0
        err[bits, group] = np.abs(dequantize(q, s, 100) - w).mean()
    assert err[8, 0] < 0.01 and err[8, 0] < err[4, 32] < err[4, 0]
    ckpt = tmp_path / "ckpt.pt"
    _fake_checkpoint(ckpt)
    sizes = {}
    for enc in ("fp32", "fp16", "int8", "int4g64"):
        report = export_web_policy(ckpt, tmp_path / enc, encoding=enc, parity_samples=4)
        assert report["max_abs_err"]["action"] < 1e-4
        sizes[enc] = report["bytes"]
    assert sizes["fp32"] > sizes["fp16"] > sizes["int4g64"] > sizes["int8"]
