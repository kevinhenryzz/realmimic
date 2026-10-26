"""Reference implementation of the SONIC deploy runtime's decoder and encoder input terms.

Mirrors ``gear_sonic_deploy`` (NVlabs/GR00T-WholeBodyControl @ ``SONIC_COMMIT``); every rule here is
cited in ``docs/sonic-observation-contract.md``.

Inputs use our conventions: joint vectors in MuJoCo (= hardware) order, absolute joint angles [rad],
quaternions wxyz, pelvis angular velocity in the pelvis frame. Histories are ``(N, T, D)`` ordered
oldest -> newest with the current control tick last; reference windows are ``(N, F, D)`` ordered
current -> future (see :func:`future_frame_indices`). Outputs are exactly SONIC's per-term layout:
IsaacLab joint order, frames concatenated frame-major inside the term.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor

SONIC_COMMIT = "7f151314d4d1606544bf249d2a7a1cb754c64582"
CONTROL_DT = 0.02  # control_dt_: one history step / one reference frame; training must step at it

# ---------------------------------------------------------------------------- joint orders

JOINT_NAMES_MUJOCO = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)  # fmt: skip

# policy_parameters.hpp names, same semantics as SONIC's Python G1_*_DOF arrays:
#   mujoco[..., i] = isaaclab[..., ISAACLAB_TO_MUJOCO[i]]   (IsaacLab index of MuJoCo joint i)
#   isaaclab[..., i] = mujoco[..., MUJOCO_TO_ISAACLAB[i]]   (MuJoCo index of IsaacLab joint i)
ISAACLAB_TO_MUJOCO = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)  # fmt: skip
MUJOCO_TO_ISAACLAB = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)  # fmt: skip
JOINT_NAMES_ISAACLAB = tuple(JOINT_NAMES_MUJOCO[j] for j in MUJOCO_TO_ISAACLAB)


def mujoco_to_isaaclab(x: Tensor) -> Tensor:
    """Reorder the last dim (29) from MuJoCo to IsaacLab order."""
    return x[..., list(MUJOCO_TO_ISAACLAB)]


def isaaclab_to_mujoco(x: Tensor) -> Tensor:
    """Reorder the last dim (29) from IsaacLab to MuJoCo order."""
    return x[..., list(ISAACLAB_TO_MUJOCO)]


# ---------------------------------------------------------------------------- runtime constants
# Compile-time constants of policy_parameters.hpp, evaluated with the header's formulas and
# operation order. All per-joint tuples below are in MuJoCo (hardware) order, as in the header.

ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFFORT_LIMIT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2
STIFFNESS = {k: a * NATURAL_FREQ * NATURAL_FREQ for k, a in ARMATURE.items()}
DAMPING = {k: 2.0 * DAMPING_RATIO * a * NATURAL_FREQ for k, a in ARMATURE.items()}

# Motor behind each joint as the runtime assumes it (hip pitch: 7520-22, the newer revision).
MOTOR_TYPES = (
    ("7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020") * 2
    + ("7520_14", "5020", "5020")
    + ("5020",) * 5 + ("4010",) * 2
    + ("5020",) * 5 + ("4010",) * 2
)  # fmt: skip
# Ankles and waist roll/pitch are driven by two 5020 motors through a linkage: 2x kp and kd, but
# the action scale uses a single motor's effort / stiffness.
_GAIN_MULT = (1.0, 1.0, 1.0, 1.0, 2.0, 2.0) * 2 + (1.0, 2.0, 2.0) + (1.0,) * 14
# kps / kds are std::array<float> in the header: the runtime uses these rounded to float32.
KPS = tuple(m * STIFFNESS[t] for m, t in zip(_GAIN_MULT, MOTOR_TYPES, strict=True))
KDS = tuple(m * DAMPING[t] for m, t in zip(_GAIN_MULT, MOTOR_TYPES, strict=True))
ACTION_SCALE = tuple(0.25 * EFFORT_LIMIT[t] / STIFFNESS[t] for t in MOTOR_TYPES)
DEFAULT_ANGLES = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)  # fmt: skip


def joint_position_targets(action: Tensor) -> Tensor:
    """Runtime action post-processing: raw network output (N, 29) in IsaacLab order -> motor
    position targets (N, 29) in MuJoCo/hardware order, ``default + action * ACTION_SCALE``.
    kp/kd are ``KPS``/``KDS``, each multiplied by its ``--motor-k[pd]-scale`` factor if given."""
    scale = action.new_tensor(ACTION_SCALE)
    default = action.new_tensor(DEFAULT_ANGLES)
    return default + isaaclab_to_mujoco(action) * scale


# ---------------------------------------------------------------------------- quaternion math
# Transcriptions of math_utils.hpp (wxyz). quat_rotate does not normalise, like quat_rotate_d.


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product a ⊗ b (== quat_mul_d)."""
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(q: Tensor) -> Tensor:
    return q * q.new_tensor((1.0, -1.0, -1.0, -1.0))


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """(2w²-1) v + 2w (q_v × v) + 2 q_v (q_v · v) (== quat_rotate_d; exact rotation for unit q)."""
    w = q[..., :1]
    qv = q[..., 1:]
    return (
        v * (2.0 * w * w - 1.0)
        + torch.cross(qv, v, dim=-1) * w * 2.0
        + qv * (qv * v).sum(-1, keepdim=True) * 2.0
    )


def yaw_to_quat(yaw: Tensor) -> Tensor:
    """Rotation about +z (== euler_z_to_quat_d)."""
    half = 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def calc_heading(q: Tensor) -> Tensor:
    """Yaw of the rotated body x-axis, atan2(y, x) (== calc_heading_d)."""
    x_axis = quat_rotate(q, q.new_tensor((1.0, 0.0, 0.0)).expand(q.shape[:-1] + (3,)))
    return torch.atan2(x_axis[..., 1], x_axis[..., 0])


def heading_quat(q: Tensor) -> Tensor:
    """Yaw-only quaternion of q (== calc_heading_quat_d)."""
    return yaw_to_quat(calc_heading(q))


def rot6d(q: Tensor) -> Tensor:
    """First two columns of R(q), flattened row-wise: [R00, R01, R10, R11, R20, R21] (q normalised
    first, as quat_to_rotation_matrix_d does)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
            2 * (x * z - w * y), 2 * (y * z + w * x),
        ),
        dim=-1,
    )  # fmt: skip


# ---------------------------------------------------------------------------- history terms
# GatherHis*: StateLogger::GetLatest(F, dt, newest_first=false) -> oldest..newest, the newest entry
# logged at the start of the current tick. Missing entries (fewer than F ticks since control
# started) are all-zero entries placed at the old end.


def _last_frames(x: Tensor, num_frames: int, num_valid: Tensor | None) -> Tensor:
    """Last ``num_frames`` entries of an oldest->newest history (N, T, D); entries older than the
    ``num_valid`` (N,) real ticks become zero, like StateLogger's padding entries."""
    if x.shape[-2] < num_frames:
        raise ValueError(f"history has {x.shape[-2]} frames, need {num_frames}")
    x = x[..., -num_frames:, :]
    if num_valid is None:
        return x
    age = torch.arange(num_frames - 1, -1, -1, device=x.device)  # 0 for the current tick
    padded = age[None, :] >= num_valid.to(x.device)[:, None]
    return torch.where(padded[..., None], torch.zeros_like(x), x)


def his_base_angular_velocity(
    base_ang_vel: Tensor, num_valid: Tensor | None = None, num_frames: int = 10
) -> Tensor:
    """Pelvis IMU gyroscope (pelvis frame, rad/s), (N, T, 3) -> (N, 3F)."""
    return _last_frames(base_ang_vel, num_frames, num_valid).flatten(-2)


def his_body_joint_positions(
    joint_pos: Tensor, num_valid: Tensor | None = None, num_frames: int = 10
) -> Tensor:
    """Measured joint angles (N, T, 29), MuJoCo order, absolute -> (N, 29F) IsaacLab order, minus
    ``DEFAULT_ANGLES`` (padding entries are 0, i.e. the default pose)."""
    rel = joint_pos - joint_pos.new_tensor(DEFAULT_ANGLES)
    return mujoco_to_isaaclab(_last_frames(rel, num_frames, num_valid)).flatten(-2)


def his_body_joint_velocities(
    joint_vel: Tensor, num_valid: Tensor | None = None, num_frames: int = 10
) -> Tensor:
    """Measured joint velocities (N, T, 29), MuJoCo order -> (N, 29F) IsaacLab order."""
    return mujoco_to_isaaclab(_last_frames(joint_vel, num_frames, num_valid)).flatten(-2)


def his_last_actions(
    last_actions: Tensor, num_valid: Tensor | None = None, num_frames: int = 10
) -> Tensor:
    """(N, T, 29) -> (N, 29F) IsaacLab order. Entry t must be the raw network output of tick t-1
    (zeros at the first control tick) in SONIC's action units (target = default + a * ACTION_SCALE),
    given in MuJoCo order: ``isaaclab_to_mujoco(onnx_action)``."""
    return mujoco_to_isaaclab(_last_frames(last_actions, num_frames, num_valid)).flatten(-2)


def his_gravity_dir(
    base_quat: Tensor, num_valid: Tensor | None = None, num_frames: int = 10
) -> Tensor:
    """Pelvis IMU orientation (N, T, 4) wxyz -> gravity (0, 0, -1) in the pelvis frame, (N, 3F).
    Padding entries carry the zero quaternion, which this formula maps to (0, 0, +1)."""
    q = _last_frames(base_quat, num_frames, num_valid)
    g = q.new_tensor((0.0, 0.0, -1.0)).expand(q.shape[:-1] + (3,))
    return quat_rotate(quat_conjugate(q), g).flatten(-2)


# ---------------------------------------------------------------------------- reference terms


def future_frame_indices(
    frame: Tensor,
    last_frame: Tensor,
    num_frames: int = 10,
    step: int = 1,
    playing: bool | Tensor = True,
) -> Tensor:
    """Reference frames a term reads at this tick, (N, F): ``frame + k*step`` clamped to the clip's
    last frame; all equal to ``frame`` while playback is paused. ``frame`` is the frame the runtime
    shows this tick (it advances after the policy step)."""
    offs = torch.arange(num_frames, device=frame.device) * step
    idx = torch.minimum(frame[:, None] + offs[None, :], last_frame[:, None])
    playing = torch.as_tensor(playing, device=frame.device)
    if playing.ndim == 1:
        playing = playing[:, None]
    return torch.where(playing, idx, frame[:, None].expand_as(idx))


def motion_joint_positions(ref_joint_pos: Tensor) -> Tensor:
    """Reference joint angles (N, F, 29), MuJoCo order, absolute -> (N, 29F) IsaacLab order. Not
    relative to ``DEFAULT_ANGLES``."""
    return mujoco_to_isaaclab(ref_joint_pos).flatten(-2)


def motion_joint_velocities(ref_joint_vel: Tensor, playing: bool | Tensor = True) -> Tensor:
    """Reference joint velocities (N, F, 29), MuJoCo order -> (N, 29F) IsaacLab order; all zeros
    while playback is paused."""
    out = mujoco_to_isaaclab(ref_joint_vel).flatten(-2)
    playing = torch.as_tensor(playing, device=out.device)
    if playing.ndim == 1:
        playing = playing[:, None]
    return torch.where(playing, out, torch.zeros_like(out))


def motion_root_z_position(ref_root_pos: Tensor) -> Tensor:
    """Reference pelvis (body 0) position (N, F, 3) in the motion's world frame -> z, (N, F)."""
    return ref_root_pos[..., 2]


def heading_alignment(
    init_base_quat: Tensor, init_ref_root_quat: Tensor, delta_heading: Tensor | None = None
) -> Tensor:
    """Yaw rotation applied to every reference root orientation (ComputeApplyDeltaHeading):
    ``Rz(delta) ⊗ heading(init_base) ⊗ heading(init_ref)^-1``, (N, 4).

    ``init_base_quat`` is the pelvis IMU orientation at the tick of the last heading
    re-initialisation (control start, motion switch/restart, clip end); ``init_ref_root_quat`` the
    reference pelvis orientation at the frame shown that tick (refreshed whenever frame 0 is shown).
    ``delta_heading`` is the operator's yaw offset [rad], reset to 0 by re-initialisation."""
    align = quat_mul(heading_quat(init_base_quat), yaw_to_quat(-calc_heading(init_ref_root_quat)))
    if delta_heading is not None:
        align = quat_mul(yaw_to_quat(delta_heading), align)
    return align


AnchorMode = Literal["base", "heading", "refheading"]


def motion_anchor_orientation(
    base_quat: Tensor, ref_root_quat: Tensor, alignment: Tensor, mode: AnchorMode = "base"
) -> Tensor:
    """Reference pelvis orientation relative to the robot pelvis, rot6d per frame, (N, 6F).

    ``base_quat`` (N, 4) is the current pelvis IMU orientation, ``ref_root_quat`` (N, F, 4) the
    reference pelvis orientations of the window, ``alignment`` from :func:`heading_alignment`.
    Per frame: ``rot6d(left^-1 ⊗ alignment ⊗ ref)`` with ``left`` = base_quat (``base``:
    motion_anchor_orientation*), heading(base_quat) (``heading``: *_heading*), or heading of the
    aligned first window frame (``refheading``: *_refheading*)."""
    ref = quat_mul(alignment[:, None, :].expand_as(ref_root_quat), ref_root_quat)
    if mode == "base":
        left = base_quat
    elif mode == "heading":
        left = heading_quat(base_quat)
    elif mode == "refheading":
        left = heading_quat(ref[:, 0])
    else:
        raise ValueError(f"unknown anchor mode {mode!r}")
    rel = quat_mul(quat_conjugate(left)[:, None, :].expand_as(ref), ref)
    return rot6d(rel).flatten(-2)


# ---------------------------------------------------------------------------- decoder input

# Registry dimensions (GetObservationRegistry) of the terms implemented above.
TERM_DIMS: dict[str, int] = {
    "base_angular_velocity": 3,
    "body_joint_positions": 29,
    "body_joint_velocities": 29,
    "last_actions": 29,
    "gravity_dir": 3,
    "his_body_joint_positions_4frame_step1": 116,
    "his_body_joint_velocities_4frame_step1": 116,
    "his_last_actions_4frame_step1": 116,
    "his_base_angular_velocity_4frame_step1": 12,
    "his_gravity_dir_4frame_step1": 12,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_gravity_dir_10frame_step1": 30,
    "motion_joint_positions": 29,
    "motion_joint_velocities": 29,
    "motion_joint_positions_3frame_step1": 87,
    "motion_joint_velocities_3frame_step1": 87,
    "motion_joint_positions_5frame_step5": 145,
    "motion_joint_velocities_5frame_step5": 145,
    "motion_joint_positions_10frame_step1": 290,
    "motion_joint_velocities_10frame_step1": 290,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position": 1,
    "motion_root_z_position_3frame_step1": 3,
    "motion_root_z_position_10frame_step1": 10,
    "motion_root_z_position_10frame_step5": 10,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step1": 60,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_anchor_orientation_heading": 6,
    "motion_anchor_orientation_heading_10frame_step1": 60,
    "motion_anchor_orientation_heading_10frame_step5": 60,
    "motion_anchor_orientation_refheading": 6,
    "motion_anchor_orientation_refheading_10frame_step1": 60,
    "motion_anchor_orientation_refheading_10frame_step5": 60,
    # Encoder-only terms of the release config. The "g1" encoder mode does not require them, so
    # the runtime leaves them zero (see assemble_encoder_input); only their sizes matter here.
    "encoder_mode_4": 4,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}


def decoder_input_dim(order: Sequence[str]) -> int:
    """Total input size of a decoder observing ``order``; must equal the ONNX ``obs_dict`` size."""
    return sum(TERM_DIMS[name] for name in order)


def assemble_decoder_input(
    terms: Mapping[str, Tensor], order: Sequence[str], extra_dims: Mapping[str, int] | None = None
) -> Tensor:
    """Concatenate terms in YAML order into the ``obs_dict`` vector (N, D), as
    InitializeObservationFunctions lays them out (each term at the running offset).
    ``extra_dims``: sizes of terms set by the config rather than the registry (``token_state``)."""
    dims = {**TERM_DIMS, **(extra_dims or {})}
    parts = []
    for name in order:
        if name not in dims:
            raise KeyError(f"unknown SONIC observation term {name!r}")
        if name not in terms:
            raise KeyError(f"missing observation term {name!r}")
        t = terms[name]
        if t.shape[-1] != dims[name]:
            raise ValueError(f"term {name!r} has dim {t.shape[-1]}, SONIC expects {dims[name]}")
        parts.append(t)
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------- encoder input
# With ``token_state`` in the decoder config, GatherTokenState (cpp:1648-1689) fills the encoder's
# own buffer and runs the encoder each tick, before the decoder terms are gathered.


def encoder_mode(mode: Tensor) -> Tensor:
    """``encoder_mode_4`` (GatherEncoderMode, cpp:1691-1697, 1713): the encode mode of the current
    motion (N,), then 3 zeros, (N, 4). Clip and planner motions are set to mode 0 ("g1") whenever
    an encoder is loaded (cpp:2402, 2417-2426)."""
    return torch.cat((mode[:, None], mode.new_zeros(mode.shape[0], 3)), dim=-1)


def assemble_encoder_input(
    terms: Mapping[str, Tensor], order: Sequence[str], required: Sequence[str]
) -> Tensor:
    """The encoder ``obs_dict`` (N, D): every enabled ``encoder_observations`` term in file order at
    its running offset (cpp:1871-1893), but only the active mode's ``required_observations`` are
    gathered; the buffer is zeroed first, so the others stay 0 (GatherEncoderObservations,
    cpp:2030-2031, 2088-2116). ``terms`` must hold the required ones; others are ignored."""
    missing = set(required) - set(order)
    if missing:
        raise KeyError(f"required encoder terms not in the encoder order: {sorted(missing)}")
    ref = terms[required[0]]
    parts = []
    for name in order:
        if name not in TERM_DIMS:
            raise KeyError(f"unknown SONIC observation term {name!r}")
        if name not in required:
            parts.append(ref.new_zeros(ref.shape[0], TERM_DIMS[name]))
            continue
        if name not in terms:
            raise KeyError(f"missing observation term {name!r}")
        t = terms[name]
        if t.shape[-1] != TERM_DIMS[name]:
            raise ValueError(
                f"term {name!r} has dim {t.shape[-1]}, SONIC expects {TERM_DIMS[name]}"
            )
        parts.append(t)
    return torch.cat(parts, dim=-1)
