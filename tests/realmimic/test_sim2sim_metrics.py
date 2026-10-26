"""Sim-to-sim metrics (realmimic.eval.sim2sim) on synthetic logs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from realmimic.eval import sim2sim as m
from realmimic.obs.sonic import JOINT_NAMES_MUJOCO


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])[None]


def test_orientation_helpers():
    ident = np.array([[1.0, 0, 0, 0]])
    np.testing.assert_allclose(m.gravity_in_body(ident), [[0, 0, -1]])
    pitch = _quat_axis_angle([0, 1, 0], 0.3)
    np.testing.assert_allclose(m.tilt_angle(pitch), [0.3])
    yawed = _quat_axis_angle([0, 0, 1], 1.2)
    np.testing.assert_allclose(m.yaw(yawed), [1.2])
    np.testing.assert_allclose(m.tilt_angle(yawed), [0.0], atol=1e-7)
    # heading-invariant: same tilt under a different yaw has no error
    a = _quat_axis_angle([1, 0, 0], 0.2)
    w, x, y, z = a[0]
    yw, _, _, yz = yawed[0]  # yawed ⊗ a (pure yaw: x = y = 0)
    b = np.array([[yw * w - yz * z, yw * x - yz * y, yw * y + yz * x, yw * z + yz * w]])
    np.testing.assert_allclose(m.tilt_error(a, b), [0.0], atol=1e-6)
    np.testing.assert_allclose(m.tilt_error(a, ident), [0.2], atol=1e-7)


def test_first_fall():
    z = np.array([0.78, 0.7, 0.5, 0.34, 0.2])
    tilt = np.array([0.0, 0.1, 1.2, 0.5, 0.5])
    assert m.first_fall(z, tilt) == 2  # tilt first
    assert m.first_fall(z, np.zeros(5)) == 3
    assert m.first_fall(np.full(5, 0.7), np.zeros(5)) is None


def test_torque_stats_and_groups():
    dt = 0.005
    tau = np.zeros((200, 2))
    tau[:50, 0] = 30.0  # 50 samples above 0.5 * 50 Nm
    tau[:, 1] = -10.0
    s = m.torque_stats(tau, dt, [50.0, 139.0], ["left_ankle_pitch_joint", "left_knee_joint"])
    a = s["left_ankle_pitch_joint"]
    assert a["mean_abs"] == pytest.approx(7.5)
    assert a["rms"] == pytest.approx(15.0)
    assert a["time_above_s"] == pytest.approx(0.25)
    assert a["frac_above"] == pytest.approx(0.25)
    assert s["left_knee_joint"]["time_above_s"] == 0.0
    g = m.group_torque(s)
    assert g["ankle"]["mean_abs"] == pytest.approx(7.5)
    assert g["knee"]["mean_abs"] == pytest.approx(10.0)
    assert "waist" not in g


def test_phase_index_and_intervals():
    idx = m.phase_index(np.array([0.5, 1.0, 1.5, 3.0]), [1.0, 2.0])
    np.testing.assert_array_equal(idx, [-1, 0, 0, 1])
    st = m.interval_stats(np.array([0.0, 0.02, 0.04, 0.07]))
    assert st["max_ms"] == pytest.approx(30.0)
    assert st["mean_ms"] == pytest.approx(70.0 / 3)


def test_parse_loop_timing():
    text = (
        "Loop timing - LowState age: 1.5ms, Streaming data mean delay: 0ms, IMU age: 1ms, Obs: 40us,"
        " Policy: 300us, Obs 2 Motor Command: 350us, Post processing: 5us, Planner - Gather Input:"
        " 3us, Model: 900us, Convert50Hz: 10us, Total: 1000us | HandCloseRatio: 1\n"
        "Loop timing - LowState age: 2.5ms, Obs: 60us, Policy: 500us, Obs 2 Motor Command: 600us\n"
        "something else\n"
    )
    t = m.parse_loop_timing(text)
    assert t["lowstate_age_ms"] == {"mean": 2.0, "max": 2.5, "n": 2}
    assert t["policy_us"]["mean"] == 400.0
    assert t["planner_total_us"] == {"mean": 1000.0, "max": 1000.0, "n": 1}


def _write_state_csv(path, t, values, prefix):
    header = "index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp," + ",".join(
        f"{prefix}_{i}" for i in range(values.shape[1])
    )
    rows = [
        f"{i},{i * 20.0:.3f},0.000,{tt * 1e3:.3f},0.000000000," + ",".join(f"{v:.9f}" for v in row)
        for i, (tt, row) in enumerate(zip(t, values, strict=True))
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_compute_report_synthetic_run(tmp_path):
    """Robot stands still at the reference for 2 s after release, then falls in phase 'walk'."""
    names = list(JOINT_NAMES_MUJOCO)
    rec_dt, t0 = 0.005, 100.0
    t = t0 + np.arange(800) * rec_dt  # 4 s of sim rows
    z = np.where(t < t0 + 3.0, 0.78, 0.2)
    q0 = np.linspace(-0.3, 0.3, 29)
    tau = np.zeros((len(t), 29))
    tau[:, names.index("left_knee_joint")] = 80.0  # > 0.5 * 139
    quat = np.tile([1.0, 0, 0, 0], (len(t), 1))
    meta = {
        "sim_dt": 0.001, "lowstate_dt": 0.002, "record_dt": rec_dt, "joint_names": names,
        "effort_limit": [139.0] * 29,
    }  # fmt: skip
    rows = {
        "t_mono": t, "sim_time": t - t0, "base_pos": np.stack([0 * t, 0 * t, z], 1),
        "base_quat": quat, "base_lin_vel": np.tile([0.5, 0, 0], (len(t), 1)),
        "base_ang_vel": np.zeros((len(t), 3)), "q": np.tile(q0, (len(t), 1)),
        "dq": np.zeros((len(t), 29)), "tau": tau, "q_cmd": np.tile(q0, (len(t), 1)),
        "kp": np.full((len(t), 29), 40.0), "kd": np.full((len(t), 29), 2.5),
        "foot_contact": np.ones((len(t), 2)), "other_floor_contacts": np.zeros(len(t)),
        "band": t < t0 + 0.5, "lag": np.full(len(t), 1e-4),
    }  # fmt: skip
    np.savez(tmp_path / "sim.npz", meta=json.dumps(meta), **rows)
    logs = tmp_path / "deploy_logs"
    logs.mkdir()
    tick = t0 + np.arange(200) * 0.02
    q_meas = np.tile(q0, (len(tick), 1)) + 0.01
    _write_state_csv(logs / "q.csv", tick, q_meas, "q")
    _write_state_csv(
        logs / "base_quat.csv", tick, np.tile([1.0, 0, 0, 0], (len(tick), 1)), "base_q"
    )
    target = np.concatenate(
        [np.tile([0, 0, 0.75, 1, 0, 0, 0], (len(tick), 1)), np.tile(q0, (len(tick), 1))], 1
    )
    (tmp_path / "target_motion.csv").write_text(
        "".join(",".join(f"{v:g}" for v in row) + ",\n" for row in target)
    )
    (tmp_path / "deploy.log").write_text(
        "Loop timing - LowState age: 1ms, Obs: 10us, Policy: 20us\n"
    )
    session = {
        "robot": "g1", "bundle": "x", "session": "planner", "aborted": True,
        "abort_reason": "SIM fall", "release_t": t0 + 0.5, "end_t": t0 + 3.9,
        "phases": [{"name": "hold", "t": t0}, {"name": "idle", "t": t0 + 0.5},
                   {"name": "walk", "t": t0 + 2.5}],
    }  # fmt: skip
    (tmp_path / "session.json").write_text(json.dumps(session))

    r = m.compute_report(tmp_path)
    assert r["fall"]["phase"] == "walk"
    assert r["fall"]["after_release_s"] == pytest.approx(2.5, abs=rec_dt)
    assert r["scored_s"] == pytest.approx(2.5, abs=rec_dt)
    assert set(r["phases"]) == {"idle", "walk"}  # hold is before the release
    assert r["phases"]["idle"]["forward_speed_mps"] == pytest.approx(0.5)
    knee = r["torque"]["per_joint"]["left_knee_joint"]
    assert knee["mean_abs"] == pytest.approx(80.0)
    assert knee["time_above_s"] == pytest.approx(2.5, abs=2 * rec_dt)
    assert r["torque"]["groups"]["ankle"]["mean_abs"] == 0.0
    assert r["tracking"]["mean_abs_rad"] == pytest.approx(0.01, abs=1e-6)
    assert r["tracking"]["root_z_mean_abs_m"] == pytest.approx(0.03, abs=1e-6)
    assert r["tracking"]["tilt_err_mean_rad"] == pytest.approx(0.0, abs=1e-6)
    assert r["gains_seen"]["kp"]["left_hip_pitch_joint"] == 40.0
    assert r["timing"]["control_tick"]["mean_ms"] == pytest.approx(20.0, abs=1e-3)
    assert r["timing"]["sim_realtime_factor"] == pytest.approx(1.0)
    row = m.summary_row(r)
    assert row["fall"].startswith("walk")
