"""Analytic BeyondMimic tracking reward in torch.

One differentiable implementation shared by the simulator side (tests / logging)
and the imagination side (world-model reward head). Mirrors mjlab's
``tasks/tracking/mdp/rewards.py`` term by term, including the re-anchoring of the
reference bodies to the robot anchor in xy + yaw only
(``MotionCommand.update_relative_body_poses``).

Inputs are world-frame link quantities of the *tracked* bodies (same ordering as
``TRACKED_BODIES``), the anchor being one of them (``anchor_index``).
Rewards are returned already weighted and multiplied by ``dt`` like mjlab's reward
manager (``scale_rewards_by_dt=True``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    # Effort regulariser on the PD torque normalised by each joint's effort limit (sum over joints of
    # w_j (tau / tau_max)^2, w_j = EffortModel.joint_weight). Policy objective only; 0 reproduces the
    # paper's reward.
    w_joint_torque: float = 0.0
    # SONIC's shaping terms (gear_sonic/config/manager_env/rewards/terms/; weights of the release and
    # v1.1 experiment configs): feet_acc = joint_acc_l2 on the ankles (-2.5e-6), anti_shake_ang_vel =
    # mean over wrists of relu(|omega| - 1.5)^2 (-5e-3), and v1.1's energy_consumption sum |tau qd|
    # (-1e-4). Policy objective only.
    w_energy: float = 0.0
    w_ankle_acc: float = 0.0
    w_anti_shake: float = 0.0
    anti_shake_threshold: float = 1.5
    # Imagined-reward-only surrogate for the tracking kernels: (1 - b) exp(-d^2) + b relu(1 - d / d_max)
    # with d = error / std. exp(-d^2) has no gradient once d >> 1; the linear tail keeps a gradient
    # up to d_max. b = 0 is the paper's kernel; the optimum (d = 0, value 1) is unchanged.
    kernel_tail: float = 0.0
    kernel_tail_range: float = 5.0
    # Sustained per-motor load (what heats a motor): an exponential moving average over the imagined
    # rollout, time constant sustained_load_tau [s], of each motor's |tau| / tau_max (parallel-linkage
    # pairs as their two motors, |tau_p +- tau_r| / 2 against half the joint limit), penalised as
    # sum_m w_m relu(avg_m - sustained_load_threshold)^2. Short corrective peaks barely move the average.
    # Policy objective only (computed along the rollout in ImagHeads); 0 = off.
    w_sustained_load: float = 0.0
    sustained_load_threshold: float = 0.1
    sustained_load_tau: float = 0.5
    # Pelvis tilt: Gaussian kernel on the difference between the reference's and the robot's pelvis-frame gravity
    # (the tilt measure of the sim-to-sim deployment tests), std in radians at small angles. 0 = off (the paper's reward).
    w_pelvis_tilt: float = 0.0
    std_pelvis_tilt: float = 0.05
    pelvis_index: int = 0  # index of the pelvis among the tracked bodies
    # Sustained torso lean: relu(EMA_tau(|q_waist_pitch|) - threshold)^2 along the imagined rollout, the waist pitch
    # read from the state (absolute angle, rad); averaged like the sustained load, so brief bends in a squat or a
    # kneel cost little while a held lean costs. Policy objective only; 0 = off.
    w_waist_lean: float = 0.0
    waist_lean_threshold: float = 0.035
    waist_lean_tau: float | None = None  # averaging time constant [s]; None = sustained_load_tau; 0 = instantaneous
    waist_lean_index: int = -1  # index of waist_pitch_joint in the robot's joint order
    # Per-body weights of the body position / orientation errors (len = tracked bodies; None = uniform mean).
    body_pos_weights: tuple[float, ...] | None = None
    body_ori_weights: tuple[float, ...] | None = None
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori + self.w_pelvis_tilt)


def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the reference bodies in a frame that shares the robot anchor's xy position and yaw.

    Exactly mjlab's ``update_relative_body_poses``: z, roll and pitch stay those of the reference.
    Shapes: bodies (N, B, ·), anchors (N, ·).
    """
    nb = ref_body_pos_w.shape[-2]
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # (N,3)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # (N,4)
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])
    return rel_pos, rel_quat


@dataclass
class EffortModel:
    """Per-joint PD constants (robot joint order) for the analytic actuator torque
    tau = kp (q0 + a * scale - q) - kd qd, as the position actuators apply it."""

    default_joint_pos: torch.Tensor  # (29,)
    action_scale: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    effort_limit: torch.Tensor
    joint_weight: torch.Tensor | None = None  # (29,) per-joint weight of the torque term; None = ones
    ankle_mask: torch.Tensor | None = None  # (29,) bool, joints of the ankle-acceleration term
    # Fraction of the effort limit below which the torque term is free: relu(|tau|/tau_max - t)^2.
    effort_threshold: float = 0.0
    # (P, 2) joint indices (pitch, roll) of parallel-linkage pairs driven by two motors (ankles, waist).
    motor_pairs: torch.Tensor | None = None
    # (29,) per-motor thresholds of the sustained-load term; None = TrackingRewardCfg.sustained_load_threshold.
    load_threshold: torch.Tensor | None = None


def joint_weights(joint_names: list[str], spec: str, default: float = 1.0) -> list[float]:
    """Per-joint values from "regex=w,regex=w" (full-match on joint names; unmatched joints ``default``)."""
    rules = [(pattern, float(w)) for pattern, w in (item.split("=") for item in spec.split(",") if item.strip())]
    for pattern, _ in rules:
        if not any(re.fullmatch(pattern, n) for n in joint_names):
            raise ValueError(f"joint weight pattern {pattern!r} matches no joint")
    return [next((w for pattern, w in rules if re.fullmatch(pattern, n)), default) for n in joint_names]


def pd_torque(model: EffortModel, joint_pos: torch.Tensor, joint_vel: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Unclamped PD torque of the position actuators, (N, 29)."""
    target = model.default_joint_pos + action * model.action_scale
    return model.kp * (target - joint_pos) - model.kd * joint_vel


def normalized_torque_sq(model: EffortModel, joint_pos: torch.Tensor, joint_vel: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """sum_j w_j relu(|tau_j| / tau_max_j - t)^2 of the unclamped PD torque, (N,); t =
    ``model.effort_threshold`` (0: the plain (tau / tau_max)^2)."""
    x = pd_torque(model, joint_pos, joint_vel, action).abs() / model.effort_limit
    sq = torch.relu(x - model.effort_threshold) ** 2 if model.effort_threshold > 0 else x**2
    return (sq * model.joint_weight if model.joint_weight is not None else sq).sum(-1)


def motor_load(model: EffortModel, joint_pos: torch.Tensor, joint_vel: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Per-motor load |tau| / tau_max of the unclamped PD torque, (N, 29): a joint's own torque, or for a
    parallel-linkage pair (``model.motor_pairs``) its two motors' |tau_p +- tau_r| / 2 against half the
    joint limit, in the pitch and roll slots."""
    tau = pd_torque(model, joint_pos, joint_vel, action)
    load = tau.abs() / model.effort_limit
    if model.motor_pairs is not None:
        p, r = model.motor_pairs[:, 0], model.motor_pairs[:, 1]
        load = load.clone()
        load[..., p] = (tau[..., p] + tau[..., r]).abs() / model.effort_limit[p]
        load[..., r] = (tau[..., p] - tau[..., r]).abs() / model.effort_limit[r]
    return load


def sustained_load_penalty(model: EffortModel, loads: torch.Tensor, init: torch.Tensor, threshold: float | torch.Tensor, alpha: float) -> torch.Tensor:
    """sum_m w_m relu(avg_m - threshold_m)^2 along a rollout, (T, N): ``loads`` (T, N, 29) per step,
    ``init`` (N, 29) the average before the first step, avg_t = alpha avg_{t-1} + (1 - alpha) load_t."""
    avg, out = init, []
    w = model.joint_weight if model.joint_weight is not None else torch.ones_like(init[0])
    for t in range(loads.shape[0]):
        avg = alpha * avg + (1.0 - alpha) * loads[t]
        out.append((w * torch.relu(avg - threshold) ** 2).sum(-1))
    return torch.stack(out)


def sustained_excess_penalty(values: torch.Tensor, init: torch.Tensor, threshold: float, alpha: float) -> torch.Tensor:
    """relu(avg_t - threshold)^2 along a rollout, (T, N): ``values`` (T, N) per step, ``init`` (N) the average before
    the first step, avg_t = alpha avg_{t-1} + (1 - alpha) value_t."""
    avg, out = init, []
    for t in range(values.shape[0]):
        avg = alpha * avg + (1.0 - alpha) * values[t]
        out.append(torch.relu(avg - threshold) ** 2)
    return torch.stack(out)


def mechanical_power(model: EffortModel, joint_pos: torch.Tensor, joint_vel: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """sum_j |tau_j qd_j| with the torque clamped at the effort limit, as applied (SONIC's
    energy_consumption on applied_torque), (N,)."""
    tau = pd_torque(model, joint_pos, joint_vel, action)
    tau = torch.maximum(torch.minimum(tau, model.effort_limit), -model.effort_limit)
    return (tau * joint_vel).abs().sum(-1)


def _body_mean(err: torch.Tensor, weights: tuple[float, ...] | None) -> torch.Tensor:
    """Mean over the tracked bodies (last dim), weighted if ``weights`` is given (weights need not sum to 1)."""
    if weights is None:
        return err.mean(-1)
    w = torch.as_tensor(weights, dtype=err.dtype, device=err.device)
    return (err * w).sum(-1) / w.sum()


def _sq_tilt_err(q_ref: torch.Tensor, q_robot: torch.Tensor) -> torch.Tensor:
    """Squared difference of the gravity direction seen in each body frame: tilt (roll and pitch) only, blind to
    yaw; about the squared tilt angle for small differences."""
    g = torch.zeros(q_ref.shape[:-1] + (3,), dtype=q_ref.dtype, device=q_ref.device)
    g[..., 2] = -1.0
    return ((quat_apply(quat_inv(q_ref), g) - quat_apply(quat_inv(q_robot), g)) ** 2).sum(-1)


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between quaternions, smooth at zero (== quat_error_magnitude**2)."""
    v = quat_box_minus(q1, q2)
    return (v * v).sum(-1)


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
    joint_vel: torch.Tensor | None = None,
    effort_model: EffortModel | None = None,
    joint_acc: torch.Tensor | None = None,
    anti_shake_indexes: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'. Shaping terms with weight 0 are zeros."""
    ra_pos = robot_body_pos_w[..., anchor_index, :]
    ra_quat = robot_body_quat_w[..., anchor_index, :]
    ma_pos = ref_body_pos_w[..., anchor_index, :]
    ma_quat = ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = _body_mean(((rel_pos - robot_body_pos_w) ** 2).sum(-1), cfg.body_pos_weights)
    e_body_ori = _body_mean(_sq_quat_err(rel_quat, robot_body_quat_w), cfg.body_ori_weights)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)

    lo = soft_limits[..., 0]
    hi = soft_limits[..., 1]
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)
    joint_limit = out.sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt

    def kernel(sq_err: torch.Tensor, std: float) -> torch.Tensor:
        k = torch.exp(-sq_err / std**2)
        if cfg.kernel_tail == 0.0:
            return k
        d = torch.sqrt(sq_err + 1e-12) / std
        return (1.0 - cfg.kernel_tail) * k + cfg.kernel_tail * torch.relu(1.0 - d / cfg.kernel_tail_range)

    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * kernel(e_root_pos, cfg.std_root_pos),
        "motion_global_root_ori": dt * cfg.w_root_ori * kernel(e_root_ori, cfg.std_root_ori),
        "motion_body_pos": dt * cfg.w_body_pos * kernel(e_body_pos, cfg.std_body_pos),
        "motion_body_ori": dt * cfg.w_body_ori * kernel(e_body_ori, cfg.std_body_ori),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * kernel(e_body_lin, cfg.std_body_lin_vel),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * kernel(e_body_ang, cfg.std_body_ang_vel),
        "pelvis_tilt": dt * cfg.w_pelvis_tilt * kernel(_sq_tilt_err(ref_body_quat_w[..., cfg.pelvis_index, :], robot_body_quat_w[..., cfg.pelvis_index, :]), cfg.std_pelvis_tilt)
        if cfg.w_pelvis_tilt != 0.0 else torch.zeros_like(e_root_pos),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    if cfg.w_joint_torque != 0.0:
        if effort_model is None or joint_vel is None:
            raise ValueError("w_joint_torque needs effort_model and joint_vel")
        terms["joint_torque"] = dt * cfg.w_joint_torque * normalized_torque_sq(effort_model, joint_pos, joint_vel, action)
    else:
        terms["joint_torque"] = torch.zeros_like(joint_limit)
    terms["energy"] = torch.zeros_like(joint_limit)
    if cfg.w_energy != 0.0:
        if effort_model is None or joint_vel is None:
            raise ValueError("w_energy needs effort_model and joint_vel")
        terms["energy"] = dt * cfg.w_energy * mechanical_power(effort_model, joint_pos, joint_vel, action)
    terms["ankle_acc"] = torch.zeros_like(joint_limit)
    if cfg.w_ankle_acc != 0.0:
        if joint_acc is None or effort_model is None or effort_model.ankle_mask is None:
            raise ValueError("w_ankle_acc needs joint_acc and effort_model.ankle_mask")
        terms["ankle_acc"] = dt * cfg.w_ankle_acc * (joint_acc[..., effort_model.ankle_mask] ** 2).sum(-1)
    terms["anti_shake"] = torch.zeros_like(joint_limit)
    if cfg.w_anti_shake != 0.0:
        if not anti_shake_indexes:
            raise ValueError("w_anti_shake needs anti_shake_indexes")
        speed = torch.linalg.norm(robot_body_ang_vel_w[..., list(anti_shake_indexes), :], dim=-1)
        terms["anti_shake"] = dt * cfg.w_anti_shake * (torch.relu(speed - cfg.anti_shake_threshold) ** 2).mean(-1)
    total = terms["motion_global_root_pos"]
    for k, v in terms.items():
        if k != "motion_global_root_pos":
            total = total + v
    terms["total"] = total
    return terms
