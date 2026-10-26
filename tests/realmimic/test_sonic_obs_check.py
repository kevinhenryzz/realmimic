"""Policy-input log check (realmimic.eval.sonic_obs_check) against hand-built runtime rows."""

from __future__ import annotations

import numpy as np

from realmimic.eval import sonic_obs_check as chk
from realmimic.obs.sonic import DEFAULT_ANGLES, MUJOCO_TO_ISAACLAB

ORDER = [
    "his_body_joint_positions_10frame_step1",
    "his_gravity_dir_10frame_step1",
    "his_last_actions_10frame_step1",
    "motion_root_z_position_10frame_step1",
    "motion_anchor_orientation_10frame_step1",
]


def _six_digits(x: np.ndarray) -> np.ndarray:
    """What `std::ostream << double` writes by default."""
    return np.array([float(f"{v:.6g}") for v in x.ravel()]).reshape(x.shape)


def _runtime_rows(q, base_quat, action, target) -> np.ndarray:
    """Logged obs rows written out the way the runtime lays them out (independent of sonic.*)."""
    rows = []
    for t in range(len(q)):
        jp, grav, act = [], [], []
        for k in range(t - 9, t + 1):  # oldest -> newest, zero entries before control start
            if k < 0:
                jp.append(np.zeros(29))
                grav.append([0.0, 0.0, 1.0])  # zero quaternion through quat_rotate_d
                act.append(np.zeros(29))
                continue
            jp.append((q[k] - DEFAULT_ANGLES)[list(MUJOCO_TO_ISAACLAB)])
            w, x, y, z = base_quat[k]
            grav.append([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])
            act.append(action[k])
        ref_z = np.full(10, target[t, 2]) + np.arange(10) * 0.01  # later frames: not checked
        anchor = np.ones(60)  # not checked
        rows.append(np.concatenate([np.ravel(jp), np.ravel(grav), np.ravel(act), ref_z, anchor]))
    return _six_digits(np.asarray(rows))


def _states(rng: np.random.Generator, ticks: int):
    q = np.asarray(DEFAULT_ANGLES) + rng.normal(0, 0.1, (ticks, 29))
    axis = rng.normal(size=(ticks, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = rng.uniform(0, 0.3, (ticks, 1))
    base_quat = np.concatenate([np.cos(ang / 2), np.sin(ang / 2) * axis], 1)
    action = rng.normal(0, 1, (ticks, 29))  # IsaacLab order, as StateLogger writes it
    target = np.concatenate([rng.normal(0, 1, (ticks, 7)), q + 0.05], 1)
    return q, base_quat, action, target


def test_history_windows_padding():
    win, valid = chk.history_windows(np.arange(3.0)[:, None], frames=4)
    np.testing.assert_array_equal(win[2, :, 0], [0, 0, 1, 2])
    np.testing.assert_array_equal(valid, [1, 2, 3])


def test_matching_log_passes_and_perturbation_is_found():
    rng = np.random.default_rng(0)
    q, base_quat, action, target = _states(rng, 12)
    logged = _runtime_rows(q, base_quat, action, target)
    expected = chk.expected_terms(q, np.zeros_like(q), base_quat, np.zeros((12, 3)), action, target)
    res = chk.compare(logged, ORDER, expected)
    for name in ORDER[:3]:
        assert res[name]["checked"] == "full"
        assert res[name]["frac_out_of_tol"] == 0.0, (name, res[name])
    assert res["motion_root_z_position_10frame_step1"]["checked"] == "elements 0:1"
    assert res["motion_root_z_position_10frame_step1"]["frac_out_of_tol"] == 0.0
    assert res["motion_anchor_orientation_10frame_step1"] == {"checked": False}

    logged[7, 290 + 3 * 9 + 2] += 1e-3  # gravity z of the newest frame at tick 7
    res = chk.compare(logged, ORDER, expected)
    assert res["his_gravity_dir_10frame_step1"]["first_bad_tick"] == 7
    assert res["his_body_joint_positions_10frame_step1"]["frac_out_of_tol"] == 0.0


def test_observation_order_token_state(tmp_path):
    cfg = tmp_path / "obs.yaml"
    cfg.write_text(
        "observations:\n"
        '  - name: "token_state"\n    enabled: true\n'
        '  - name: "his_gravity_dir_10frame_step1"\n    enabled: false\n'
        '  - name: "his_base_angular_velocity_10frame_step1"\n    enabled: true\n'
        "encoder:\n  dimension: 64\n"
    )
    order, extra = chk.observation_order(cfg)
    assert order == ["token_state", "his_base_angular_velocity_10frame_step1"]
    assert extra == {"token_state": 64}


def test_bundle_policy_check_aligns_next_tick_actions(tmp_path):
    import torch

    from realmimic.eval.sonic_obs_check import bundle_policy_check

    net = torch.nn.Linear(6, 3)
    path = tmp_path / "policy.onnx"
    torch.onnx.export(net, (torch.zeros(1, 6),), str(path), input_names=["obs_dict"], output_names=["action"], dynamo=False)
    inputs = np.random.default_rng(0).normal(size=(5, 6)).astype(np.float32)
    with torch.no_grad():
        produced = net(torch.from_numpy(inputs)).numpy()
    logged = np.vstack([np.zeros((1, 3)), produced])[:5]  # tick t logs the action of tick t - 1
    assert bundle_policy_check(path, inputs, logged)["action"]["ticks_out_of_tol"] == 0
    assert bundle_policy_check(path, inputs, logged + 0.1)["action"]["ticks_out_of_tol"] == 4
