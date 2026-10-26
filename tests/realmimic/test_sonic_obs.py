"""SONIC observation contract (docs/sonic-observation-contract.md): realmimic.obs.sonic."""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import pytest
import torch

from realmimic.obs import sonic

ROOT = Path(__file__).resolve().parents[2]
D = torch.float64

# IsaacLab joint order as SONIC's policy sees it (left/right/waist interleaved per tree depth).
ISAACLAB_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)  # fmt: skip

# policy_parameters.hpp @ 7f151314, lines 100-104 and 210-240 (verbatim), and the values of its
# per-motor expressions (lines 40-65) evaluated in double precision.
HEADER_ISAACLAB_TO_MUJOCO = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
HEADER_MUJOCO_TO_ISAACLAB = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
HEADER_DEFAULT_ANGLES = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)
S5020, S14, S22, S4010 = 14.25062309787429, 40.179238471373175, 99.09842777666113, 16.77832748089279
D5020, D14, D22, D4010 = 0.907222843292423, 2.5578897650279457, 6.3088018534966395, 1.06814150219
A5020, A14, A22, A4010 = (
    0.43857731392336724,
    0.5475464652142304,
    0.3506614663788243,
    0.07450087032950714,
)
LEG = (S22, S22, S14, S22, 2 * S5020, 2 * S5020)
ARM = (S5020,) * 5 + (S4010,) * 2
HEADER_KPS = LEG + LEG + (S14, 2 * S5020, 2 * S5020) + ARM + ARM
LEG_D = (D22, D22, D14, D22, 2 * D5020, 2 * D5020)
ARM_D = (D5020,) * 5 + (D4010,) * 2
HEADER_KDS = LEG_D + LEG_D + (D14, 2 * D5020, 2 * D5020) + ARM_D + ARM_D
LEG_A = (A22, A22, A14, A22, A5020, A5020)
ARM_A = (A5020,) * 5 + (A4010,) * 2
HEADER_ACTION_SCALE = LEG_A + LEG_A + (A14, A5020, A5020) + ARM_A + ARM_A


def _yaw(a: float) -> torch.Tensor:
    return torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)], dtype=D)


def _pitch(a: float) -> torch.Tensor:
    return torch.tensor([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0], dtype=D)


def _rot6d_rz(a: float) -> list[float]:
    return [math.cos(a), -math.sin(a), math.sin(a), math.cos(a), 0.0, 0.0]


def _random_quat(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=D, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------- constants / orders


def test_constants_match_header_values():
    assert sonic.ISAACLAB_TO_MUJOCO == HEADER_ISAACLAB_TO_MUJOCO
    assert sonic.MUJOCO_TO_ISAACLAB == HEADER_MUJOCO_TO_ISAACLAB
    assert sonic.DEFAULT_ANGLES == HEADER_DEFAULT_ANGLES
    for ours, header in (
        (sonic.KPS, HEADER_KPS),
        (sonic.KDS, HEADER_KDS),
        (sonic.ACTION_SCALE, HEADER_ACTION_SCALE),
    ):
        assert len(ours) == 29
        assert ours == pytest.approx(header, rel=1e-12)


def test_joint_orders_and_round_trip():
    assert sonic.JOINT_NAMES_ISAACLAB == ISAACLAB_NAMES
    assert sorted(sonic.ISAACLAB_TO_MUJOCO) == list(range(29))
    for i in range(29):
        assert sonic.ISAACLAB_TO_MUJOCO[sonic.MUJOCO_TO_ISAACLAB[i]] == i
        assert ISAACLAB_NAMES[sonic.ISAACLAB_TO_MUJOCO[i]] == sonic.JOINT_NAMES_MUJOCO[i]
    x = torch.randn(3, 5, 29, dtype=D)
    assert torch.equal(sonic.isaaclab_to_mujoco(sonic.mujoco_to_isaaclab(x)), x)
    assert torch.equal(sonic.mujoco_to_isaaclab(sonic.isaaclab_to_mujoco(x)), x)
    mj = torch.arange(29.0)
    assert sonic.mujoco_to_isaaclab(mj)[1].item() == sonic.JOINT_NAMES_MUJOCO.index(
        "right_hip_pitch_joint"
    )


def test_joint_names_match_mjlab_g1():
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, KNEES_BENT_KEYFRAME

    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    assert tuple(m.joint(i).name for i in range(1, m.njnt)) == sonic.JOINT_NAMES_MUJOCO
    for name, q0 in zip(sonic.JOINT_NAMES_MUJOCO, sonic.DEFAULT_ANGLES, strict=True):
        pose = KNEES_BENT_KEYFRAME.joint_pos
        expected = next((v for p, v in pose.items() if re.fullmatch(p, name)), 0.0)
        assert q0 == expected, name


def test_joint_position_targets():
    a = torch.zeros(1, 29, dtype=D)
    assert torch.allclose(
        sonic.joint_position_targets(a)[0], torch.tensor(sonic.DEFAULT_ANGLES, dtype=D)
    )
    a[0, sonic.JOINT_NAMES_ISAACLAB.index("right_knee_joint")] = 1.0
    q = sonic.joint_position_targets(a)[0]
    k = sonic.JOINT_NAMES_MUJOCO.index("right_knee_joint")
    assert q[k].item() == pytest.approx(0.669 + sonic.ACTION_SCALE[k])
    assert sonic.MOTOR_TYPES[k] == "7520_22"


# ---------------------------------------------------------------------------- quaternion math


def test_quat_mul_matches_cpp_formula():
    """math_utils.hpp quat_mul_d (the 'optimized' isaacgym formula) is the Hamilton product."""
    gen = torch.Generator().manual_seed(0)
    a, b = _random_quat(64, gen), _random_quat(64, gen)
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    cpp = torch.stack(
        [
            qq - ww + (z1 - y1) * (y2 - z2),
            qq - xx + (x1 + w1) * (x2 + w2),
            qq - yy + (w1 - x1) * (y2 + z2),
            qq - zz + (z1 + y1) * (w2 - x2),
        ],
        dim=-1,
    )
    assert torch.allclose(sonic.quat_mul(a, b), cpp, atol=1e-14)


def test_gravity_and_heading_match_training_helpers():
    from supermimic.kinematics.rotations import heading_yaw, projected_gravity

    q = _random_quat(64, torch.Generator().manual_seed(1))
    g = sonic.his_gravity_dir(q[:, None, :], num_frames=1)
    assert torch.allclose(g, projected_gravity(q), atol=1e-12)
    assert torch.allclose(sonic.calc_heading(q), heading_yaw(q), atol=1e-12)


def test_rot6d_layout():
    """Row-wise flattening of the first two columns: [R00, R01, R10, R11, R20, R21]."""
    q = _yaw(0.3)
    assert sonic.rot6d(q).tolist() == pytest.approx(_rot6d_rz(0.3))
    c, s = math.cos(0.4), math.sin(0.4)
    assert sonic.rot6d(_pitch(0.4)).tolist() == pytest.approx([c, 0.0, 0.0, 1.0, -s, 0.0])
    assert sonic.rot6d(2.0 * _pitch(0.4)).tolist() == pytest.approx([c, 0.0, 0.0, 1.0, -s, 0.0])


# ---------------------------------------------------------------------------- anchor orientation


def _anchor(base, ref_window, init_base, init_ref, mode="base", delta=None):
    align = sonic.heading_alignment(init_base[None], init_ref[None], delta)
    return sonic.motion_anchor_orientation(base[None], ref_window[None], align, mode)[0]


def test_anchor_identity():
    one = _yaw(0.0)
    out = _anchor(one, one.expand(10, 4), one, one)
    assert out.shape == (60,)
    assert out.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0, 0.0, 0.0] * 10)


def test_anchor_yaw_offset_removed_by_heading_reinit():
    """Robot at yaw 0.5, clip starts at yaw -0.7 and turns 0.1 rad per frame: the start offset is
    absorbed by the heading alignment, only the clip's own turning remains."""
    ref = torch.stack([_yaw(-0.7 + 0.1 * k) for k in range(10)])
    out = _anchor(_yaw(0.5), ref, init_base=_yaw(0.5), init_ref=ref[0])
    assert out.tolist() == pytest.approx(
        [v for k in range(10) for v in _rot6d_rz(0.1 * k)], abs=1e-12
    )
    # The operator's delta heading rotates the whole reference.
    out = _anchor(
        _yaw(0.5), ref, init_base=_yaw(0.5), init_ref=ref[0], delta=torch.tensor([0.3], dtype=D)
    )
    assert out.tolist() == pytest.approx(
        [v for k in range(10) for v in _rot6d_rz(0.3 + 0.1 * k)], abs=1e-12
    )
    # A robot that turned by 0.2 since the re-initialisation sees the reference 0.2 further right.
    out = _anchor(_yaw(0.7), ref, init_base=_yaw(0.5), init_ref=ref[0])
    assert out[:6].tolist() == pytest.approx(_rot6d_rz(-0.2), abs=1e-12)


def test_anchor_pitch():
    c, s = math.cos(0.3), math.sin(0.3)
    one = _yaw(0.0)
    # Reference pitched forward, robot level (heading of a pitched pelvis is unchanged).
    out = _anchor(one, _pitch(0.3).expand(10, 4), one, _pitch(0.3))
    assert out[:6].tolist() == pytest.approx([c, 0.0, 0.0, 1.0, -s, 0.0], abs=1e-12)
    # Robot pitched, reference level: full-base mode sees the robot's pitch, heading mode does not.
    out = _anchor(_pitch(0.3), one.expand(10, 4), one, one)
    assert out[:6].tolist() == pytest.approx([c, 0.0, 0.0, 1.0, s, 0.0], abs=1e-12)
    out = _anchor(_pitch(0.3), one.expand(10, 4), one, one, mode="heading")
    assert out[:6].tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], abs=1e-12)


def test_anchor_refheading_uses_first_window_frame():
    ref = torch.stack([_yaw(0.4 + 0.1 * k) for k in range(10)])
    # Robot heading is irrelevant in refheading mode; init frames differ from the window.
    out = _anchor(_yaw(1.3), ref, init_base=_yaw(-0.2), init_ref=_yaw(0.9), mode="refheading")
    assert out.tolist() == pytest.approx(
        [v for k in range(10) for v in _rot6d_rz(0.1 * k)], abs=1e-12
    )


def test_refheading_equals_traj_goal_rotation():
    """The 'traj' goal's per-frame rotation (motion_library.goal_window) is SONIC's refheading
    term, whatever the yaw alignment."""
    from supermimic.kinematics.rotations import quat_inv, quat_mul, rot6d_from_quat, yaw_quat

    gen = torch.Generator().manual_seed(5)
    ref = _random_quat(4 * 10, gen).view(4, 10, 4)
    h = yaw_quat(ref[:, 0])[:, None, :].expand(4, 10, 4)
    traj = rot6d_from_quat(quat_mul(quat_inv(h), ref)).flatten(1)
    align = sonic.heading_alignment(_random_quat(4, gen), _random_quat(4, gen))
    ours = sonic.motion_anchor_orientation(_random_quat(4, gen), ref, align, "refheading")
    assert torch.allclose(ours, traj, atol=1e-12)


def test_anchor_matches_training_anchor_obs_without_heading_offset():
    """With identical world frames (alignment = identity) the SONIC term is our anchor_relative_obs
    ('ori') applied to the pelvis."""
    # Needs src/supermimic/data, which the repo's `data/` ignore rule keeps out of git.
    anchor_relative_obs = pytest.importorskip("supermimic.world_model.heads").anchor_relative_obs

    gen = torch.Generator().manual_seed(2)
    base, ref = _random_quat(32, gen), _random_quat(32, gen)
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=D).expand(32, 4)
    ours = sonic.motion_anchor_orientation(base, ref[:, None], ident)
    zero = torch.zeros(32, 3, dtype=D)
    assert torch.allclose(ours, anchor_relative_obs(zero, base, zero, ref, mode="ori"), atol=1e-12)


# ---------------------------------------------------------------------------- history terms


def test_history_order_and_dims():
    n, t = 2, 12
    tick = torch.arange(t, dtype=D)[None, :, None].expand(n, t, 1)
    jp = tick.expand(n, t, 29) + torch.tensor(sonic.DEFAULT_ANGLES, dtype=D)
    out = sonic.his_body_joint_positions(jp)
    assert out.shape == (n, 290)
    # Oldest of the last 10 ticks first, the current tick last; each frame in IsaacLab order.
    assert out.view(n, 10, 29)[:, :, 0].tolist() == [list(map(float, range(2, 12)))] * n
    assert sonic.his_body_joint_velocities(tick.expand(n, t, 29)).shape == (n, 290)
    assert sonic.his_last_actions(tick.expand(n, t, 29)).shape == (n, 290)
    av = sonic.his_base_angular_velocity(tick.expand(n, t, 3))
    assert av.shape == (n, 30) and av[0, -3:].tolist() == [11.0, 11.0, 11.0]
    q = _yaw(0.0).expand(n, t, 4)
    assert sonic.his_gravity_dir(q).tolist() == [[0.0, 0.0, -1.0] * 10] * n
    assert sonic.his_body_joint_positions(jp, num_frames=4).shape == (n, 116)


def test_history_terms_permute_like_training_obs():
    """Per term the layout equals flatten_history_term_major (oldest->newest), up to joint order."""
    heads = pytest.importorskip("supermimic.world_model.heads")  # see above
    flatten_history_term_major = heads.flatten_history_term_major

    gen = torch.Generator().manual_seed(3)
    hist = torch.randn(10, 4, 93, dtype=D, generator=gen)  # (H, N, 93) oldest->newest
    flat = flatten_history_term_major(hist)
    per = hist.permute(1, 0, 2)  # (N, H, 93)
    ours = sonic.his_body_joint_velocities(per[..., 29:58])
    assert torch.equal(ours, sonic.mujoco_to_isaaclab(flat[:, 290:580].view(4, 10, 29)).flatten(1))
    assert torch.equal(sonic.his_base_angular_velocity(per[..., 58:61]), flat[:, 580:610])


def test_history_zero_padding_at_control_start():
    """Two ticks after `start`: 8 StateLogger padding entries at the old end."""
    n, t = 1, 10
    jp = torch.full((n, t, 29), 0.5, dtype=D)
    valid = torch.tensor([2])
    out = sonic.his_body_joint_positions(jp, valid).view(10, 29)
    assert torch.all(out[:8] == 0.0)  # zero = default pose, not -default
    rel = torch.tensor(sonic.DEFAULT_ANGLES, dtype=D)
    assert torch.allclose(out[8:], sonic.mujoco_to_isaaclab(0.5 - rel).expand(2, 29))
    g = sonic.his_gravity_dir(_yaw(0.2).expand(n, t, 4), valid).view(10, 3)
    assert g[:8].tolist() == [[0.0, 0.0, 1.0]] * 8  # zero quaternion -> +z
    assert torch.allclose(g[8:], torch.tensor([0.0, 0.0, -1.0], dtype=D).expand(2, 3))
    av = sonic.his_base_angular_velocity(torch.ones(n, t, 3, dtype=D), valid).view(10, 3)
    assert av.sum().item() == 6.0
    with pytest.raises(ValueError):
        sonic.his_last_actions(torch.zeros(n, 9, 29, dtype=D))


# ---------------------------------------------------------------------------- reference terms


def test_future_frames_clamp_and_pause():
    frame, last = torch.tensor([5, 98]), torch.tensor([99, 99])
    idx = sonic.future_frame_indices(frame, last)
    assert idx[0].tolist() == list(range(5, 15))
    assert idx[1].tolist() == [98] + [99] * 9
    assert sonic.future_frame_indices(frame, last, step=5)[0].tolist() == list(range(5, 55, 5))
    paused = sonic.future_frame_indices(frame, last, playing=torch.tensor([False, True]))
    assert paused[0].tolist() == [5] * 10 and paused[1].tolist() == idx[1].tolist()


def test_reference_terms():
    n, f = 2, 10
    jp = torch.arange(29.0, dtype=D).expand(n, f, 29)
    out = sonic.motion_joint_positions(jp)
    assert out.shape == (n, 290)
    assert out[0, :29].tolist() == [
        float(j) for j in sonic.MUJOCO_TO_ISAACLAB
    ]  # absolute, no offset
    jv = torch.ones(n, f, 29, dtype=D)
    assert sonic.motion_joint_velocities(jv).sum().item() == 2 * 290
    paused = sonic.motion_joint_velocities(jv, playing=torch.tensor([True, False]))
    assert paused[0].sum().item() == 290 and paused[1].sum().item() == 0
    pos = torch.zeros(n, f, 3, dtype=D)
    pos[..., 2] = torch.linspace(0.7, 0.8, f, dtype=D)
    z = sonic.motion_root_z_position(pos)
    assert z.shape == (n, 10) and z[0].tolist() == pytest.approx(
        torch.linspace(0.7, 0.8, f).tolist()
    )


# ---------------------------------------------------------------------------- decoder input

PLANNED_ORDER = (
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


def test_assemble_decoder_input():
    n = 3
    terms = {
        name: torch.full((n, sonic.TERM_DIMS[name]), float(i))
        for i, name in enumerate(PLANNED_ORDER)
    }
    x = sonic.assemble_decoder_input(terms, PLANNED_ORDER)
    assert (
        x.shape
        == (n, sonic.decoder_input_dim(PLANNED_ORDER))
        == (n, 30 + 290 * 3 + 30 + 290 * 2 + 10 + 60)
    )
    offset = 0
    for i, name in enumerate(PLANNED_ORDER):
        assert torch.all(x[:, offset : offset + sonic.TERM_DIMS[name]] == i)
        offset += sonic.TERM_DIMS[name]
    with pytest.raises(KeyError):
        sonic.assemble_decoder_input(terms, PLANNED_ORDER + ("token_state",))
    with pytest.raises(ValueError):
        sonic.assemble_decoder_input(
            {**terms, "motion_root_z_position_10frame_step1": torch.zeros(n, 9)}, PLANNED_ORDER
        )


def test_term_functions_produce_registry_dims():
    n, t = 2, 10
    gen = torch.Generator().manual_seed(4)
    q = _random_quat(n * t, gen).view(n, t, 4)
    out = {
        "his_base_angular_velocity_10frame_step1": sonic.his_base_angular_velocity(
            torch.zeros(n, t, 3)
        ),
        "his_body_joint_positions_10frame_step1": sonic.his_body_joint_positions(
            torch.zeros(n, t, 29)
        ),
        "his_body_joint_velocities_10frame_step1": sonic.his_body_joint_velocities(
            torch.zeros(n, t, 29)
        ),
        "his_last_actions_10frame_step1": sonic.his_last_actions(torch.zeros(n, t, 29)),
        "his_gravity_dir_10frame_step1": sonic.his_gravity_dir(q),
        "motion_joint_positions_10frame_step1": sonic.motion_joint_positions(torch.zeros(n, t, 29)),
        "motion_joint_velocities_10frame_step1": sonic.motion_joint_velocities(
            torch.zeros(n, t, 29)
        ),
        "motion_root_z_position_10frame_step1": sonic.motion_root_z_position(torch.zeros(n, t, 3)),
        "motion_anchor_orientation_10frame_step1": sonic.motion_anchor_orientation(
            q[:, 0], q, q[:, 0]
        ),
        "motion_anchor_orientation": sonic.motion_anchor_orientation(q[:, 0], q[:, :1], q[:, 0]),
    }
    for name, value in out.items():
        assert value.shape == (n, sonic.TERM_DIMS[name]), name


# ---------------------------------------------------------------------------- constants checker


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_sonic_constants", ROOT / "tools/check_sonic_constants.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Motor expression per MuJoCo joint, transcribed from policy_parameters.hpp lines 109-207.
_MOTOR = (
    ("7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020") * 2
    + ("7520_14", "5020", "5020")
    + (("5020",) * 5 + ("4010",) * 2) * 2
)
_GAIN = ("", "", "", "", "2.0 * ", "2.0 * ") * 2 + ("", "2.0 * ", "2.0 * ") + ("",) * 14


def _header_snippet(kp_hip: str = "STIFFNESS_7520_22") -> str:
    """policy_parameters.hpp, reduced to the declarations the checker reads (same syntax)."""
    joints = ", ".join(map(str, HEADER_ISAACLAB_TO_MUJOCO))
    inv = ", ".join(map(str, HEADER_MUJOCO_TO_ISAACLAB))
    sc = [f"0.25 * EFFORT_LIMIT_{m} / STIFFNESS_{m}, // j{i}" for i, m in enumerate(_MOTOR)]
    kp = [f"{g}STIFFNESS_{m}, // j{i}" for i, (g, m) in enumerate(zip(_GAIN, _MOTOR, strict=True))]
    kd = [f"{g}DAMPING_{m}, // j{i}" for i, (g, m) in enumerate(zip(_GAIN, _MOTOR, strict=True))]
    kp[0] = f"{kp_hip}, // left_hip_pitch_joint (old is 7520_14 new is 7520_22)"
    return f"""
const double ARMATURE_5020 = 0.003609725;
const double ARMATURE_7520_14 = 0.010177520;
const double ARMATURE_7520_22 = 0.025101925;
const double ARMATURE_4010 = 0.00425;
const double NATURAL_FREQ = 10 * 2.0 * 3.1415926535; // 10Hz
const double DAMPING_RATIO = 2;
const double STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ;
const double DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ;
const double DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ;
const double DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ;
const double DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ;
const double EFFORT_LIMIT_5020 = 25.0;    // 5020 motor type
const double EFFORT_LIMIT_7520_14 = 88.0; // 7520_14 motor type
const double EFFORT_LIMIT_7520_22 = 139.0; // 7520_22 motor type
const double EFFORT_LIMIT_4010 = 5.0;     // 4010 motor type
const std::array<int, 5> vr_5point_index = {{28, 29, 0, 18, 19}};
const std::array<int, 29> isaaclab_to_mujoco = {{{joints}}};
const std::array<int, 29> mujoco_to_isaaclab = {{{inv}}};
const std::array<double, 29> g1_action_scale = {{
{chr(10).join(sc)}
}};
const std::array<float, 29> kps = {{
{chr(10).join(kp)}
}};
const std::array<float, 29> kds = {{
{chr(10).join(kd)}
}};
const std::array<double, 29> default_angles = {{
{", ".join(map(str, HEADER_DEFAULT_ANGLES))} // trailing comment
}};
"""


def test_checker_parses_header_and_registry():
    chk = _load_checker()
    arrays = chk.parse_policy_parameters(_header_snippet())
    registry = (
        '{"his_gravity_dir_10frame_step1", 30, [this](std::vector<double>& buf, size_t offset) {}'
    )
    dims = chk.parse_registry_dims(registry)
    assert dims == {"his_gravity_dir_10frame_step1": 30}
    assert chk.compare(arrays, None) == []
    errors = chk.compare(
        chk.parse_policy_parameters(_header_snippet(kp_hip="STIFFNESS_7520_14")), None
    )
    assert len(errors) == 1 and errors[0].startswith("kps[0] (left_hip_pitch_joint)")
    errors = chk.compare(arrays, {**sonic.TERM_DIMS, "gravity_dir": 4})
    assert errors == ["term gravity_dir: SONIC dim 4 != ours 3"]
