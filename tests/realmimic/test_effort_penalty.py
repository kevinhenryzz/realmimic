"""Analytic PD-torque effort penalty: matches MuJoCo's position actuators and is opt-in."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity.entity import Entity

from realmimic.robots import get_robot_spec
from realmimic.robots.g1 import joint_gains, robot_cfg
from supermimic.envs.mdp.rewards import (
    EffortModel,
    TrackingRewardCfg,
    joint_weights,
    motor_load,
    normalized_torque_sq,
    sustained_load_penalty,
    tracking_reward_terms,
)


def _effort_model(names: list[str], q0: np.ndarray) -> EffortModel:
    kp, kd, scale = joint_gains(get_robot_spec("g1_brainco"), names)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
    return EffortModel(t(q0), t(scale), t(kp), t(kd), t(4.0 * scale * kp))


def test_pd_torque_matches_mujoco_actuators():
    entity = Entity(robot_cfg(get_robot_spec("g1_brainco")))
    model = entity.spec.compile()
    data = mujoco.MjData(model)
    names = [model.joint(model.actuator(i).trnid[0]).name for i in range(model.nu)]
    rng = np.random.default_rng(0)
    q0 = rng.uniform(-0.2, 0.2, model.nu)
    em = _effort_model(names, q0)
    qadr = np.array([model.jnt_qposadr[model.actuator(i).trnid[0]] for i in range(model.nu)])
    dadr = np.array([model.jnt_dofadr[model.actuator(i).trnid[0]] for i in range(model.nu)])
    data.qpos[qadr] = rng.uniform(-0.3, 0.3, model.nu)
    data.qvel[dadr] = rng.uniform(-1.0, 1.0, model.nu)
    action = rng.uniform(-0.5, 0.5, model.nu)
    data.ctrl[:] = q0 + action * em.action_scale.numpy()
    mujoco.mj_forward(model, data)
    ours = em.kp * (em.default_joint_pos + torch.as_tensor(action) * em.action_scale - torch.as_tensor(data.qpos[qadr])) - em.kd * torch.as_tensor(data.qvel[dadr])
    sim = data.actuator_force  # clamped by forcerange; compare where unsaturated
    free = np.abs(ours.numpy()) < model.actuator_forcerange[:, 1] - 1e-6
    np.testing.assert_allclose(sim[free], ours.numpy()[free], rtol=1e-6, atol=1e-6)
    assert free.sum() > 20


def _reward_inputs(n: int, w: float):
    names = [f"j{i}" for i in range(29)]
    em = EffortModel(*(torch.ones(29, dtype=torch.float64) for _ in range(5)))
    b = lambda *s: torch.zeros(n, *s, dtype=torch.float64)
    quat = b(14, 4)
    quat[..., 0] = 1.0
    kwargs = {
        "robot_body_pos_w": b(14, 3), "robot_body_quat_w": quat, "robot_body_lin_vel_w": b(14, 3),
        "robot_body_ang_vel_w": b(14, 3), "ref_body_pos_w": b(14, 3), "ref_body_quat_w": quat.clone(),
        "ref_body_lin_vel_w": b(14, 3), "ref_body_ang_vel_w": b(14, 3), "anchor_index": 7,
        "action": torch.full((n, 29), 0.5, dtype=torch.float64), "prev_action": b(29), "joint_pos": b(29),
        "soft_limits": torch.tensor([[-3.0, 3.0]] * 29, dtype=torch.float64),
        "cfg": TrackingRewardCfg(w_joint_torque=w), "joint_vel": b(29), "effort_model": em,
    }  # fmt: skip
    return kwargs, names


def test_penalty_is_opt_in_and_negative():
    kwargs, _ = _reward_inputs(3, 0.0)
    off = tracking_reward_terms(**kwargs)
    assert torch.all(off["joint_torque"] == 0)
    kwargs, _ = _reward_inputs(3, -0.5)
    on = tracking_reward_terms(**kwargs)
    # tau = 1 * (1 + 0.5 - 0) = 1.5 per joint, normalised by 1 -> 29 * 2.25
    assert on["joint_torque"] == pytest.approx(torch.full((3,), 0.02 * -0.5 * 29 * 2.25, dtype=torch.float64))
    assert torch.allclose(on["total"], off["total"] + on["joint_torque"])


def test_normalized_torque_has_gradient_wrt_action():
    em = EffortModel(*(torch.ones(29) for _ in range(5)))
    a = torch.zeros(1, 29, requires_grad=True)
    normalized_torque_sq(em, torch.zeros(1, 29), torch.zeros(1, 29), a).sum().backward()
    assert torch.all(a.grad > 0)


def test_joint_weights_scale_the_torque_term():
    kwargs, names = _reward_inputs(2, -0.5)
    base = tracking_reward_terms(**kwargs)["joint_torque"]
    weights = joint_weights(names, "j(0|1)=5")
    assert weights[:3] == [5.0, 5.0, 1.0]
    kwargs["effort_model"].joint_weight = torch.tensor(weights, dtype=torch.float64)
    weighted = tracking_reward_terms(**kwargs)["joint_torque"]
    assert weighted == pytest.approx(base * (27 + 2 * 5) / 29)
    with pytest.raises(ValueError, match="matches no joint"):
        joint_weights(names, "waist_.*=2")


def test_sonic_shaping_terms():
    n = 2
    kwargs, _ = _reward_inputs(n, 0.0)
    em = kwargs["effort_model"]
    em.effort_limit = torch.full((29,), 1.2, dtype=torch.float64)  # tau = 1.5 clamps to 1.2
    em.ankle_mask = torch.zeros(29, dtype=torch.bool)
    em.ankle_mask[[4, 5]] = True
    kwargs["joint_vel"] = torch.full((n, 29), -2.0, dtype=torch.float64)  # tau = 1.5 + 2 = 3.5 -> 1.2
    kwargs["joint_acc"] = torch.arange(29, dtype=torch.float64).expand(n, 29)
    kwargs["robot_body_ang_vel_w"][:, 10] = torch.tensor([2.5, 0.0, 0.0])  # 1.0 above the deadzone
    kwargs["robot_body_ang_vel_w"][:, 13] = torch.tensor([0.0, 1.0, 0.0])  # inside it
    kwargs["anti_shake_indexes"] = (10, 13)
    kwargs["cfg"] = TrackingRewardCfg(w_energy=-1e-4, w_ankle_acc=-2.5e-6, w_anti_shake=-5e-3)
    t = tracking_reward_terms(**kwargs)
    assert t["energy"] == pytest.approx(torch.full((n,), 0.02 * -1e-4 * 29 * 1.2 * 2.0, dtype=torch.float64))
    assert t["ankle_acc"] == pytest.approx(torch.full((n,), 0.02 * -2.5e-6 * (16 + 25), dtype=torch.float64))
    assert t["anti_shake"] == pytest.approx(torch.full((n,), 0.02 * -5e-3 * (1.0 + 0.0) / 2, dtype=torch.float64))
    assert torch.all(t["joint_torque"] == 0)
    kwargs["cfg"] = TrackingRewardCfg()
    off = tracking_reward_terms(**kwargs)
    assert all(torch.all(off[k] == 0) for k in ("energy", "ankle_acc", "anti_shake"))
    assert torch.allclose(t["total"], off["total"] + t["energy"] + t["ankle_acc"] + t["anti_shake"])


def test_kernel_tail_keeps_gradient_at_large_errors():
    kwargs, _ = _reward_inputs(1, 0.0)
    far = torch.zeros(1, 14, 3, dtype=torch.float64)
    far[..., 2] = 2.0  # 2 m height error (re-anchoring keeps the reference's z): d = 6.7 std
    kwargs["robot_body_pos_w"] = far.clone().requires_grad_(True)
    paper = tracking_reward_terms(**kwargs)["motion_body_pos"]
    (g_paper,) = torch.autograd.grad(paper.sum(), kwargs["robot_body_pos_w"])
    kwargs["robot_body_pos_w"] = (far * 0.5).requires_grad_(True)  # d = 3.3 std, inside the tail range
    kwargs["cfg"] = TrackingRewardCfg(kernel_tail=0.1)
    tailed = tracking_reward_terms(**kwargs)["motion_body_pos"]
    (g_tail,) = torch.autograd.grad(tailed.sum(), kwargs["robot_body_pos_w"])
    assert g_paper.abs().max() < 1e-12 < g_tail.abs().max()
    kwargs["robot_body_pos_w"] = torch.zeros(1, 14, 3, dtype=torch.float64)
    perfect = tracking_reward_terms(**kwargs)["motion_body_pos"]
    assert perfect == pytest.approx(torch.tensor([0.02], dtype=torch.float64))  # optimum unchanged
    kwargs["cfg"] = TrackingRewardCfg()
    assert tracking_reward_terms(**kwargs)["motion_body_pos"] == pytest.approx(perfect)


def test_effort_threshold_frees_low_torques():
    """relu(|tau|/tau_max - t)^2: zero (and no gradient) below the threshold, the squared excess above."""
    em = EffortModel(*(torch.ones(29) for _ in range(5)), effort_threshold=0.3)
    low = torch.full((1, 29), -0.8, requires_grad=True)  # tau = 1 + a = 0.2 < 0.3
    assert normalized_torque_sq(em, torch.zeros(1, 29), torch.zeros(1, 29), low).item() == 0.0
    normalized_torque_sq(em, torch.zeros(1, 29), torch.zeros(1, 29), low).sum().backward()
    assert torch.all(low.grad == 0)
    high = torch.full((1, 29), -0.2, requires_grad=True)  # tau = 0.8
    v = normalized_torque_sq(em, torch.zeros(1, 29), torch.zeros(1, 29), high)
    assert v.item() == pytest.approx(29 * 0.5**2)
    v.sum().backward()
    assert torch.all(high.grad > 0)
    neg = torch.full((1, 29), -1.8)  # tau = -0.8: same penalty for either sign
    assert normalized_torque_sq(em, torch.zeros(1, 29), torch.zeros(1, 29), neg).item() == pytest.approx(29 * 0.5**2)


def _unit_model(**kw) -> EffortModel:
    """kp 1, kd 0, scale 1, q0 0: tau = action at q = 0; effort limit 1 (10 for joints 0 and 1)."""
    lim = torch.ones(29)
    lim[:2] = 10.0
    return EffortModel(torch.zeros(29), torch.ones(29), torch.ones(29), torch.zeros(29), lim, **kw)


def test_motor_load_splits_parallel_pairs_into_motors():
    em = _unit_model(motor_pairs=torch.tensor([[0, 1]]))
    a = torch.zeros(1, 29)
    a[0, 0], a[0, 1], a[0, 5] = 4.0, 1.0, -0.5  # pair pitch 4 Nm, roll 1 Nm; joint 5 alone
    load = motor_load(em, torch.zeros(1, 29), torch.zeros(1, 29), a)
    # Motors carry |tau_p +- tau_r| / 2 = 2.5 and 1.5 Nm against half the 10 Nm joint limit.
    torch.testing.assert_close(load[0, :2], torch.tensor([0.5, 0.3]))
    assert load[0, 5] == pytest.approx(0.5) and load[0, 2] == 0.0


def test_sustained_load_penalises_holding_not_spikes():
    em = _unit_model()
    T, alpha, thr = 32, float(np.exp(-0.02 / 0.5)), 0.1
    held = torch.full((T, 1, 29), 0.0)
    held[:, :, 3] = 0.3
    spike = torch.zeros(T, 1, 29)
    spike[10:12, :, 3] = 0.3
    zero = torch.zeros(1, 29)
    p_held = sustained_load_penalty(em, held, held[0], thr, alpha)
    p_spike = sustained_load_penalty(em, spike, zero, thr, alpha)
    # Held from the start: the average stays at 0.3, so every step costs (0.3 - 0.1)^2.
    torch.testing.assert_close(p_held[:, 0], torch.full((T,), 0.2**2))
    # A 2-step spike to the same level moves the 0.5 s average to ~0.02 < threshold: free.
    assert p_spike.sum() == 0.0
    # Below the threshold nothing is penalised; the 7520-style weight scales the term.
    low = torch.full((T, 1, 29), 0.05)
    assert sustained_load_penalty(em, low, low[0], thr, alpha).sum() == 0.0
    em.joint_weight = torch.full((29,), 0.25)
    torch.testing.assert_close(sustained_load_penalty(em, held, held[0], thr, alpha), 0.25 * p_held)


def test_sustained_load_per_motor_thresholds():
    em = _unit_model()
    T, alpha = 8, float(np.exp(-0.02 / 0.5))
    held = torch.zeros(T, 1, 29)
    held[:, :, 3] = 0.3
    held[:, :, 4] = 0.3
    thr = torch.full((29,), 0.2)
    thr[3] = 0.1  # e.g. a reference policy's mean load on this motor
    p = sustained_load_penalty(em, held, held[0], thr, alpha)
    torch.testing.assert_close(p[:, 0], torch.full((T,), 0.2**2 + 0.1**2))


def test_joint_weights_default_for_unmatched_joints():
    from supermimic.envs.mdp.rewards import joint_weights

    names = ["waist_pitch_joint", "waist_roll_joint", "left_knee_joint"]
    assert joint_weights(names, r"waist_(pitch|roll)_joint=0.13", 0.2) == [0.13, 0.13, 0.2]
    assert joint_weights(names, "") == [1.0, 1.0, 1.0]


def test_pelvis_tilt_term_and_body_weights():
    import math

    from supermimic.envs.mdp.rewards import TrackingRewardCfg, _body_mean, _sq_tilt_err

    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a = 0.05  # 0.05 rad of pitch
    pitched = torch.tensor([[math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]])
    yawed = torch.tensor([[math.cos(0.5), 0.0, 0.0, math.sin(0.5)]])
    assert float(_sq_tilt_err(upright, upright)) == 0.0
    assert float(_sq_tilt_err(upright, yawed)) < 1e-12  # tilt only, blind to yaw
    assert float(_sq_tilt_err(upright, pitched)) == pytest.approx(a**2, rel=1e-3)
    err = torch.tensor([[1.0, 3.0]])
    assert float(_body_mean(err, None)) == 2.0
    assert float(_body_mean(err, (1.0, 0.0))) == 1.0
    assert TrackingRewardCfg(w_pelvis_tilt=0.5).max_step_reward == pytest.approx(TrackingRewardCfg().max_step_reward + 0.02 * 0.5)


def test_sustained_excess_penalises_a_held_lean_not_a_brief_bend():
    from supermimic.envs.mdp.rewards import sustained_excess_penalty

    T, alpha, thr = 32, float(np.exp(-0.02 / 0.5)), 0.035
    held = torch.full((T, 1), 0.2)
    p_held = sustained_excess_penalty(held, held[0], thr, alpha)
    torch.testing.assert_close(p_held[:, 0], torch.full((T,), (0.2 - thr) ** 2))
    bend = torch.zeros(T, 1)
    bend[10:13] = 0.2  # a 60 ms bend barely moves the 0.5 s average
    assert sustained_excess_penalty(bend, torch.zeros(1), thr, alpha).sum() == 0.0
    assert sustained_excess_penalty(torch.full((T, 1), 0.03), torch.full((1,), 0.03), thr, alpha).sum() == 0.0


def test_instantaneous_lean_penalty_has_no_lag():
    from supermimic.envs.mdp.rewards import sustained_excess_penalty

    lean = torch.zeros(6, 1)
    lean[3:] = 0.2
    p = sustained_excess_penalty(lean, torch.zeros(1), 0.035, 0.0)  # alpha 0 = instantaneous
    assert p[:3].sum() == 0.0
    torch.testing.assert_close(p[3:, 0], torch.full((3,), (0.2 - 0.035) ** 2))
