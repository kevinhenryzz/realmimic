"""World-model root frame (StateSpecCfg.frame): exact integration round trip, yaw invariance and
equivariance, and continuity through a vertical pelvis x-axis (mid-flip), where the fork's heading
frame jumps by up to pi."""

from __future__ import annotations

import math

import pytest
import torch

from supermimic.kinematics.rotations import quat_apply, quat_from_rotvec, quat_mul
from supermimic.world_model.state_spec import FEAT_DIMS, StateSpec, StateSpecCfg

DT = torch.float64


def random_states(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(n, 3, generator=g, dtype=DT)
    q = torch.randn(n, 4, generator=g, dtype=DT)
    q = q / q.norm(dim=-1, keepdim=True)
    qj = 0.5 * torch.randn(n, 29, generator=g, dtype=DT)
    v = torch.randn(n, 3, generator=g, dtype=DT)
    w = 2.0 * torch.randn(n, 3, generator=g, dtype=DT)
    qd = 3.0 * torch.randn(n, 29, generator=g, dtype=DT)
    return torch.cat([p, q, qj, v, w, qd], dim=-1)


def pitched(beta: torch.Tensor, yaw: float = 0.3) -> torch.Tensor:
    """Root quaternions whose x-axis is ``beta`` rad from vertical (+z), on either side of it."""
    pitch = -(math.pi / 2 - beta)  # rotating about y by -pi/2 points x up
    q_pitch = quat_from_rotvec(torch.stack([torch.zeros_like(beta), pitch, torch.zeros_like(beta)], dim=-1))
    q_yaw = quat_from_rotvec(torch.tensor([0.0, 0.0, yaw], dtype=DT).expand(beta.shape[0], 3))
    return quat_mul(q_yaw, q_pitch)


def yaw_rotate(s: torch.Tensor, psi: float) -> torch.Tensor:
    """The same physical state turned by ``psi`` about the world vertical (root angular velocity is body-frame)."""
    rz = quat_from_rotvec(torch.tensor([0.0, 0.0, psi], dtype=DT).expand(s.shape[0], 3))
    out = s.clone()
    out[:, 0:3] = quat_apply(rz, s[:, 0:3])
    out[:, 3:7] = quat_mul(rz, s[:, 3:7])
    out[:, 36:39] = quat_apply(rz, s[:, 36:39])
    return out


def same_rotation(qa: torch.Tensor, qb: torch.Tensor) -> None:
    torch.testing.assert_close(qa * torch.sign((qa * qb).sum(-1, keepdim=True)), qb, atol=1e-10, rtol=0)


def assert_states_close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=1e-10, rtol=0)
    same_rotation(a[:, 3:7], b[:, 3:7])
    torch.testing.assert_close(a[:, 7:], b[:, 7:], atol=1e-10, rtol=0)


@pytest.mark.parametrize("frame", ["body", "heading"])
def test_target_delta_inverts_integration(frame):
    spec = StateSpec(StateSpecCfg(frame=frame), "cpu", DT)
    s = random_states(256)
    s[:128, 3:7] = pitched(torch.linspace(-0.2, 0.2, 128, dtype=DT))  # around vertical too
    d = 0.1 * torch.randn(256, 70, dtype=DT, generator=torch.Generator().manual_seed(1))
    s1 = spec.integrate_physical(s, d)
    torch.testing.assert_close(spec.target_delta(s, s1), d, atol=1e-9, rtol=0)
    assert_states_close(spec.integrate_physical(s, spec.target_delta(s, s1)), s1)


@pytest.mark.parametrize("frame", ["body", "heading"])
def test_features_are_yaw_invariant_and_integration_yaw_equivariant(frame):
    spec = StateSpec(StateSpecCfg(frame=frame), "cpu", DT)
    s = random_states(256, seed=2)
    feat = spec.canonical(s)
    assert feat.shape == (256, FEAT_DIMS[frame])
    d = 0.1 * torch.randn(256, 70, dtype=DT, generator=torch.Generator().manual_seed(3))
    for psi in (0.7, -2.9):
        rs = yaw_rotate(s, psi)
        torch.testing.assert_close(spec.canonical(rs), feat, atol=1e-9, rtol=0)
        # Up to the free z of the root position (canonical keeps only its height).
        assert_states_close(spec.integrate_physical(rs, d), yaw_rotate(spec.integrate_physical(s, d), psi))


def test_body_frame_is_continuous_through_a_vertical_pelvis():
    """Two orientations 1 degree apart on either side of vertical: the body-frame features and the
    integrated next states differ by O(1 degree); the heading frame's by O(1)."""
    eps = math.radians(0.5)
    base = random_states(64, seed=4)
    a, b = base.clone(), base.clone()
    a[:, 3:7] = pitched(torch.full((64,), eps, dtype=DT))
    b[:, 3:7] = pitched(torch.full((64,), -eps, dtype=DT))
    d = 0.1 * torch.randn(64, 70, dtype=DT, generator=torch.Generator().manual_seed(5))
    body, heading = StateSpec(StateSpecCfg(frame="body"), "cpu", DT), StateSpec(StateSpecCfg(frame="heading"), "cpu", DT)
    feat_gap = lambda spec: (spec.canonical(a) - spec.canonical(b)).abs().amax()  # noqa: E731
    step_gap = lambda spec: (spec.integrate_physical(a, d)[:, 36:39] - spec.integrate_physical(b, d)[:, 36:39]).abs().amax()  # noqa: E731
    # Body frame: velocity features rotate by 1 degree (|v| up to ~3 m/s -> < 0.06).
    assert feat_gap(body) < 0.1 and step_gap(body) < 0.01
    assert feat_gap(heading) > 0.5 and step_gap(heading) > 0.05


def test_body_frame_gravity_feature_is_the_projected_gravity():
    spec = StateSpec(StateSpecCfg(frame="body"), "cpu", DT)
    s = random_states(16, seed=6)
    g_b = spec.canonical(s)[:, 1:4] * spec.in_scale[1:4]
    from supermimic.kinematics.rotations import quat_apply_inverse

    torch.testing.assert_close(g_b, quat_apply_inverse(s[:, 3:7], torch.tensor([0.0, 0.0, -1.0], dtype=DT).expand(16, 3)), atol=1e-12, rtol=0)
