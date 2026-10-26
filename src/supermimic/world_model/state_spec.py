"""State representation for the world model.

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(29), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)

Root frame F of the network's root quantities (``StateSpecCfg.frame``):
    "body"     F = the root (pelvis) frame itself. Features [root_z(1), gravity in F(3),
               root_lin_vel in F(3), root_ang_vel_b(3), joint_pos(29), joint_vel(29)] (68).
               Yaw-invariant and continuous for every orientation.
    "heading"  F = the yaw frame of the pelvis x-axis (the fork's choice). Features [root_z(1),
               rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in F(3), root_ang_vel_b(3),
               joint_pos(29), joint_vel(29)] (71). The heading is undefined when the pelvis x-axis
               is vertical (mid-flip) and jumps by up to pi there, so features and increments jump
               while the motion is smooth.

Network output ``delta`` (70) and the integration step (dt = 0.02 s), R_F = rotation of F at s:
    dv_F(3), dw_b(3), dqd(29)            velocity increments
    dp_F(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_F dv_F               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_F dp_F       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

STATE_DIM = 71
FEAT_DIMS = {"body": 68, "heading": 71}
DELTA_DIM = 70
NJ = 29


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Root frame of the features and increments: "body" or "heading" (see the module docstring).
    # Checkpoints written before this option existed are "heading".
    frame: str = "body"
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_F, dw_b, dqd, dp_F, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, gravity_F | rot6d, lin_vel_F, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset.
    xi_offset: tuple[float, ...] = field(default_factory=lambda: (0.75,) + (0.0,) * 29 + (0.0,) * 3)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: (0.45,) + (0.01,) * 29 + (0.05,) * 3)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        if cfg.frame not in FEAT_DIMS:
            raise ValueError(f"unknown state frame {cfg.frame!r}")
        self.cfg = cfg
        self.dt = cfg.dt
        self.frame = cfg.frame
        self.feat_dim = FEAT_DIMS[cfg.frame]
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, NJ, 3, 3, NJ)).to(device=device, dtype=dtype)
        orient = 3 if cfg.frame == "body" else 6
        self.in_scale = rep(cfg.in_scale, (1, orient, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device, dtype=dtype)

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

    def root_frame(self, q: torch.Tensor) -> torch.Tensor:
        """Orientation of the frame F of the root features and increments, for unit root quaternions q."""
        return q if self.frame == "body" else heading_quat(q)

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Yaw-invariant, scaled network input features (…, feat_dim)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        fq = self.root_frame(q)
        if self.frame == "body":
            g = torch.tensor([0.0, 0.0, -1.0], dtype=q.dtype, device=q.device).expand(q.shape[:-1] + (3,))
            orient = quat_apply_inverse(q, g)
        else:
            orient = rot6d_from_quat(quat_canonical(quat_mul(quat_apply_inverse_quat(fq), q)))
        feat = torch.cat([p[..., 2:3], orient, quat_apply_inverse(fq, v), w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
        return self.integrate_physical(s, delta * self.out_scale)

    def integrate_physical(self, s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Apply increments ``d`` (…, 70) in physical units [dv_F, dw_b, dqd, dp_F, dth_b, dq] to ``s``."""
        dv_f, dw_b, dqd, dp_f, dth_b, dq = d[..., 0:3], d[..., 3:6], d[..., 6:35], d[..., 35:38], d[..., 38:41], d[..., 41:70]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        fq = self.root_frame(q)
        v_new = v + quat_apply(fq, dv_f)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(fq, dp_f)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    def target_delta(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Inverse of ``integrate_physical``: the increments (…, 70) that map ``s`` exactly to ``s_next``."""
        p, q, qj, v, w, qd = self.split(s)
        p1, q1, qj1, v1, w1, qd1 = self.split(s_next)
        q, q1 = quat_normalize(q), quat_normalize(q1)
        fq = self.root_frame(q)
        dt = self.dt if self.cfg.explicit_integration else 0.0
        dv_f = quat_apply_inverse(fq, v1 - v)
        dp_f = quat_apply_inverse(fq, p1 - p - dt * v1)
        dth_b = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q), q1)) - dt * w1
        return torch.cat([dv_f, w1 - w, qd1 - qd, dp_f, dth_b, qj1 - qj - dt * qd1], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
