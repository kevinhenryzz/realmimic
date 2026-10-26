"""SONIC release policy on our env (realmimic.eval.sonic_release_policy): encoder input contract,
runtime frame / heading bookkeeping, SONIC actuator gains, and the torch transcription of the
release ONNX graphs (the last needs the release models)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from realmimic.eval import sonic_obs_check as chk
from realmimic.eval import sonic_release_policy as srp
from realmimic.obs import sonic

RELEASE_DIR = Path(
    os.environ.get(
        "REALMIMIC_SONIC_RELEASE_DIR",
        "~/realmimic-sonic/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release",
    )
).expanduser()
needs_release = pytest.mark.skipif(
    not (RELEASE_DIR / "model_encoder.onnx").exists(),
    reason=f"release models absent: {RELEASE_DIR}",
)

# policy/release/observation_config.yaml @ SONIC_COMMIT, encoder section (file order), with the
# registry dimensions of g1_deploy_onnx_ref.cpp:1711-1815.
RELEASE_ENCODER = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
)
RELEASE_ORDER = tuple(name for name, _ in RELEASE_ENCODER)
G1_REQUIRED = srp.G1_TERMS  # encoder_modes "g1" (mode_id 0) required_observations, same order
RELEASE_DECODER = (
    "token_state",
    "his_base_angular_velocity_10frame_step1",
    "his_body_joint_positions_10frame_step1",
    "his_body_joint_velocities_10frame_step1",
    "his_last_actions_10frame_step1",
    "his_gravity_dir_10frame_step1",
)


def _offsets() -> dict[str, int]:
    out, pos = {}, 0
    for name, dim in RELEASE_ENCODER:
        out[name] = pos
        pos += dim
    return out


def _random_quat(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=torch.float64, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------- encoder input


def test_release_encoder_layout():
    for name, dim in RELEASE_ENCODER:
        assert sonic.TERM_DIMS[name] == dim, name
    assert sum(dim for _, dim in RELEASE_ENCODER) == 1762  # model_encoder.onnx obs_dict [1, 1762]
    # The ONNX g1 branch reads obs_dict[4:584] and [601:661] (sonic_release_policy.G1_*): these are
    # exactly the joint pos + vel windows and the 10-frame anchor orientation of this layout.
    off = _offsets()
    assert srp.G1_COMMAND == slice(
        off["motion_joint_positions_10frame_step5"], off["motion_root_z_position_10frame_step5"]
    )
    assert off["motion_joint_velocities_10frame_step5"] == 294
    assert srp.G1_ANCHOR == slice(
        off["motion_anchor_orientation_10frame_step5"],
        off["motion_joint_positions_lowerbody_10frame_step5"],
    )
    # Decoder: 64-d token + the five 10-frame history terms = model_decoder.onnx obs_dict [1, 994].
    assert 64 + sonic.decoder_input_dim(RELEASE_DECODER[1:]) == 994


@needs_release
def test_release_config_file_matches_layout():
    cfg = srp.ReleaseConfig.load(RELEASE_DIR / "observation_config.yaml")
    assert tuple(cfg.encoder_order) == RELEASE_ORDER
    assert tuple(cfg.g1_required) == G1_REQUIRED
    assert tuple(cfg.decoder_order) == RELEASE_DECODER
    assert cfg.token_dim == 64


def test_encoder_mode_encoding():
    # GatherEncoderMode(buf, offset, 3): the mode, then 3 zeros (cpp:1691-1697, 1713).
    out = sonic.encoder_mode(torch.tensor([0.0, 2.0]))
    assert out.tolist() == [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]


def test_encoder_input_zero_fills_terms_outside_the_mode():
    n = 3
    # Every term, required or not, is provided with a distinct non-zero value.
    terms = {
        name: torch.full((n, dim), float(i + 1)) for i, (name, dim) in enumerate(RELEASE_ENCODER)
    }
    x = sonic.assemble_encoder_input(terms, RELEASE_ORDER, G1_REQUIRED)
    assert x.shape == (n, 1762)
    off = _offsets()
    for i, (name, dim) in enumerate(RELEASE_ENCODER):
        part = x[:, off[name] : off[name] + dim]
        # cpp:2030-2031 zero the buffer, cpp:2093-2116 gather only required_observations.
        assert torch.all(part == (float(i + 1) if name in G1_REQUIRED else 0.0)), name
    del terms["motion_root_z_position"]  # not required: may be absent
    sonic.assemble_encoder_input(terms, RELEASE_ORDER, G1_REQUIRED)
    del terms["motion_anchor_orientation_10frame_step5"]
    with pytest.raises(KeyError):
        sonic.assemble_encoder_input(terms, RELEASE_ORDER, G1_REQUIRED)
    with pytest.raises(KeyError):
        sonic.assemble_encoder_input(terms, RELEASE_ORDER[:3], G1_REQUIRED)
    terms["motion_anchor_orientation_10frame_step5"] = torch.zeros(n, 6)
    with pytest.raises(ValueError):
        sonic.assemble_encoder_input(terms, RELEASE_ORDER, G1_REQUIRED)


def test_encoder_terms_step5_window():
    gen = torch.Generator().manual_seed(0)
    timesteps, n = 30, 2
    clip_q = torch.randn(timesteps, 29, dtype=torch.float64, generator=gen)  # MuJoCo order
    clip_dq = torch.randn(timesteps, 29, dtype=torch.float64, generator=gen)
    clip_quat = _random_quat(timesteps, gen)
    frame = torch.tensor([0, 12])
    # Frame k: current + 5k, clamped to the last frame (cpp:440-447 pattern, cpp:763-770, 830-836).
    idx = sonic.future_frame_indices(frame, torch.full((n,), timesteps - 1), 10, 5)
    assert idx[0].tolist() == [0, 5, 10, 15, 20, 25, 29, 29, 29, 29]
    assert idx[1].tolist() == [12, 17, 22, 27] + [29] * 6
    base = _random_quat(n, gen)
    align = sonic.yaw_to_quat(torch.tensor([0.3, -1.0], dtype=torch.float64))
    t = srp.encoder_terms(base, clip_q[idx], clip_dq[idx], clip_quat[idx], align)
    for name in G1_REQUIRED:
        assert t[name].shape == (n, sonic.TERM_DIMS[name]), name
    assert t["encoder_mode_4"].tolist() == [[0.0] * 4] * n
    # Absolute reference angles (no default offset), IsaacLab order, frame-major.
    jp = t["motion_joint_positions_10frame_step5"].view(n, 10, 29)
    assert torch.equal(jp[1, 3], clip_q[27, list(sonic.MUJOCO_TO_ISAACLAB)])
    jv = t["motion_joint_velocities_10frame_step5"].view(n, 10, 29)
    assert torch.equal(jv[0, 6], clip_dq[29, list(sonic.MUJOCO_TO_ISAACLAB)])
    # Anchor: rot6d(base^-1 * A * ref) of each window frame (cpp:660-687).
    rel = sonic.quat_mul(sonic.quat_conjugate(base[1]), sonic.quat_mul(align[1], clip_quat[22]))
    anchor = t["motion_anchor_orientation_10frame_step5"].view(n, 10, 6)
    torch.testing.assert_close(anchor[1, 2], sonic.rot6d(rel))
    # Paused: every frame is the current one and velocities are 0 (cpp:842-847).
    idx = sonic.future_frame_indices(frame, torch.full((n,), timesteps - 1), 10, 5, playing=False)
    t = srp.encoder_terms(base, clip_q[idx], clip_dq[idx], clip_quat[idx], align, playing=False)
    assert torch.all(t["motion_joint_velocities_10frame_step5"] == 0)
    assert torch.equal(t["motion_joint_positions_10frame_step5"].view(n, 10, 29)[1, 9], jp[1, 0])


# ---------------------------------------------------------------------------- runtime bookkeeping


def test_runtime_frames_and_heading_reinit():
    # paused 2 ticks, play a 4-frame clip, it ends (back to frame 0, paused), paused 2 ticks
    playing = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=float)
    frame, reinit = chk.runtime_frames(playing, timesteps=4)
    assert frame.tolist() == [0, 0, 0, 1, 2, 3, 0, 0]
    assert np.flatnonzero(reinit).tolist() == [0, 6]  # first tick, tick after the clip end
    base = np.zeros((8, 4))
    base[:, 0] = 1.0
    base[6] = sonic.yaw_to_quat(torch.tensor(0.5, dtype=torch.float64)).numpy()  # turned at t=6
    ref = sonic.yaw_to_quat(torch.tensor([0.2, 0.0, 0.0, 0.0], dtype=torch.float64)).numpy()
    a = chk.runtime_alignment(base, ref, frame, reinit)
    yaw = 2 * np.arctan2(a[:, 3], a[:, 0])
    np.testing.assert_allclose(yaw[:6], -0.2, atol=1e-12)  # heading(base_0) - heading(ref_0)
    np.testing.assert_allclose(yaw[6:], 0.5 - 0.2, atol=1e-12)


def test_sonic_gains_articulation():
    from realmimic.robots import g1, get_robot_spec

    for robot in ("g1", "g1_brainco"):
        spec = get_robot_spec(robot)
        art = g1.sonic_gains_articulation(spec)
        assert [a.target_names_expr for a in art.actuators] == [
            (name,) for name in sonic.JOINT_NAMES_MUJOCO
        ]
        np.testing.assert_allclose([a.stiffness for a in art.actuators], sonic.KPS, rtol=1e-12)
        np.testing.assert_allclose([a.damping for a in art.actuators], sonic.KDS, rtol=1e-12)
        # Effort limits and armature stay the spec's: these specs have a 7520-14 hip pitch (88 Nm)
        # trained with kp 40.2, which SONIC drives at its 7520-22 kp 99.1.
        kp, _, _ = g1.joint_gains(spec, list(sonic.JOINT_NAMES_MUJOCO))
        hip = sonic.JOINT_NAMES_MUJOCO.index("left_hip_pitch_joint")
        assert math.isclose(kp[hip], sonic.STIFFNESS["7520_14"], rel_tol=1e-8)
        assert art.actuators[hip].effort_limit == 88.0
        assert math.isclose(art.actuators[hip].armature, sonic.ARMATURE["7520_14"], rel_tol=1e-8)
        assert math.isclose(art.actuators[hip].stiffness, sonic.STIFFNESS["7520_22"])


# ---------------------------------------------------------------------------- networks


@needs_release
def test_nets_match_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    policy = srp.SonicReleasePolicy(RELEASE_DIR, device="cpu")
    enc = ort.InferenceSession(str(RELEASE_DIR / "model_encoder.onnx"))
    dec = ort.InferenceSession(str(RELEASE_DIR / "model_decoder.onnx"))
    gen = torch.Generator().manual_seed(1)
    obs = torch.randn(16, 1762, generator=gen)
    obs[:, :4] = 0.0  # g1 mode; the other branches' inputs stay random and must not leak in
    tokens = policy.nets.encode(obs)
    for i in range(len(obs)):
        ref = enc.run(None, {"obs_dict": obs[i : i + 1].numpy()})[0]
        np.testing.assert_array_equal(tokens[i : i + 1].numpy(), ref)
    assert torch.all((tokens * 16) == torch.round(tokens * 16))  # FSQ levels k/16
    x = torch.cat((tokens, torch.randn(16, 930, generator=gen)), dim=1)
    actions = policy.nets.decode(x)
    for i in range(len(x)):
        ref = dec.run(None, {"obs_dict": x[i : i + 1].numpy()})[0]
        np.testing.assert_allclose(actions[i : i + 1].numpy(), ref, rtol=1e-5, atol=1e-5)
    obs[0, 0] = 1.0  # teleop mode is not transcribed
    with pytest.raises(ValueError):
        policy.nets.encode(obs)
