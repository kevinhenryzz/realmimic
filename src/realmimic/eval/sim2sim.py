"""Metrics of one sim-to-sim session through SONIC's deploy binary (``deploy/sonic/sim2sim.py``).

All inputs live in one run directory; every time stamp is CLOCK_MONOTONIC seconds, shared by the
simulator (``time.monotonic``), the runtime (``std::chrono::steady_clock``) and the session script:

- ``sim.npz`` — ``deploy/sonic/sonic_sim.py`` recording, one row every ``record_dt``;
- ``deploy_logs/*.csv`` — the runtime's StateLogger (``--enable-csv-logs``), one row per CONTROL tick;
- ``target_motion.csv`` — ``--target-motion-logfile``, one row per CONTROL tick: reference pelvis
  position (3), quaternion wxyz (4), joint positions (29, MuJoCo order, absolute);
- ``session.json`` — the scripted operator session (phases, band release, end, fall);
- ``deploy.log`` — runtime stdout (``Loop timing`` lines).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

FALL_HEIGHT = 0.35  # pelvis height [m]
FALL_TILT = 1.0  # pelvis z-axis angle from vertical [rad]
# Joints whose sustained torque is the thermal proxy (ankle overheating is the reported failure).
THERMAL_GROUPS = {
    "ankle": r".*_ankle_(pitch|roll)_joint",
    "knee": r".*_knee_joint",
    "hip_pitch": r".*_hip_pitch_joint",
    "waist": r"waist_(yaw|roll|pitch)_joint",
}
TORQUE_FRACTION = 0.5  # "time above" threshold, fraction of the joint's effort limit
LOOP_TIMING_KEYS = {
    "lowstate_age_ms": r"LowState age: ([\d.eE+-]+)ms",
    "obs_us": r"Obs: (\d+)us",
    "policy_us": r"Policy: (\d+)us",
    "planner_total_us": r"Planner - .*?Total: (\d+)us",
}

# ---------------------------------------------------------------------------- readers


def read_state_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """StateLogger CSV (index, time_ms, time_realtime_ms, time_monotonic_ms, ros_timestamp,
    values...) -> (CLOCK_MONOTONIC seconds (T,), values (T, D))."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return arr[:, 3] / 1000.0, arr[:, 5:]


def read_rows_csv(path: str | Path) -> np.ndarray:
    """Header-less CSV with a trailing comma per row (target-motion / policy-input logs)."""
    rows = [
        [float(v) for v in line.rstrip().rstrip(",").split(",")]
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)


# ---------------------------------------------------------------------------- geometry (wxyz)


def gravity_in_body(quat: np.ndarray) -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the body frame, (N, 3)."""
    w, x, y, z = (quat / np.linalg.norm(quat, axis=-1, keepdims=True)).T
    # third row of R(q), negated: R^T e_z = (R20, R21, R22)
    return -np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1)


def tilt_angle(quat: np.ndarray) -> np.ndarray:
    """Angle between the body z-axis and world up [rad], (N,)."""
    return np.arccos(np.clip(-gravity_in_body(quat)[:, 2], -1.0, 1.0))


def tilt_error(quat_ref: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Heading-invariant orientation error: angle between the two body-frame gravity vectors."""
    dot = (gravity_in_body(quat_ref) * gravity_in_body(quat)).sum(-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def yaw(quat: np.ndarray) -> np.ndarray:
    """Heading of the body x-axis, atan2(R10, R00) [rad]."""
    w, x, y, z = (quat / np.linalg.norm(quat, axis=-1, keepdims=True)).T
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------------------- metrics


def first_fall(
    height: np.ndarray,
    tilt: np.ndarray,
    min_height: float = FALL_HEIGHT,
    max_tilt: float = FALL_TILT,
) -> int | None:
    """Index of the first sample with the pelvis below ``min_height`` or tilted past ``max_tilt``."""
    hit = np.flatnonzero((height < min_height) | (tilt > max_tilt))
    return int(hit[0]) if hit.size else None


def torque_stats(
    tau: np.ndarray,
    dt: float,
    limits: Sequence[float],
    names: Sequence[str],
    fraction: float = TORQUE_FRACTION,
) -> dict[str, dict[str, float]]:
    """Per joint: mean |tau|, RMS (∝ copper loss), p95 |tau| [Nm], and time / fraction of time with
    |tau| above ``fraction`` of the effort limit."""
    if tau.shape[0] == 0:
        return {}
    a = np.abs(tau)
    above = a > fraction * np.asarray(limits)[None, :]
    return {
        name: {
            "mean_abs": float(a[:, j].mean()),
            "rms": float(np.sqrt((tau[:, j] ** 2).mean())),
            "p95_abs": float(np.percentile(a[:, j], 95)),
            "time_above_s": float(above[:, j].sum() * dt),
            "frac_above": float(above[:, j].mean()),
        }
        for j, name in enumerate(names)
    }


def group_torque(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Thermal groups: the worst joint's mean |tau| / RMS / time above threshold."""
    out = {}
    for group, pattern in THERMAL_GROUPS.items():
        joints = [s for n, s in stats.items() if re.fullmatch(pattern, n)]
        if joints:
            out[group] = {
                k: max(s[k] for s in joints)
                for k in ("mean_abs", "rms", "time_above_s", "frac_above")
            }
    return out


def interval_stats(t: np.ndarray) -> dict[str, float]:
    """Spacing of time stamps [ms]."""
    if t.size < 2:
        return {}
    d = np.diff(t) * 1e3
    return {
        "mean_ms": float(d.mean()),
        "std_ms": float(d.std()),
        "p99_ms": float(np.percentile(d, 99)),
        "max_ms": float(d.max()),
    }


def parse_loop_timing(text: str) -> dict[str, dict[str, float]]:
    """Mean / max of the runtime's periodic ``Loop timing`` line fields."""
    out = {}
    lines = [line for line in text.splitlines() if line.startswith("Loop timing")]
    for key, pattern in LOOP_TIMING_KEYS.items():
        vals = [float(m.group(1)) for line in lines if (m := re.search(pattern, line))]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "n": len(vals)}
    return out


def phase_index(t: np.ndarray, starts: Sequence[float]) -> np.ndarray:
    """Index of the phase each time stamp falls in (phases sorted by start; -1 before the first)."""
    return np.searchsorted(np.asarray(starts), t, side="right") - 1


def joint_tracking(q: np.ndarray, q_ref: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """Mean |q - q_ref| [rad]: over all joints, per thermal group, per joint."""
    if q.shape[0] == 0:
        return {}
    err = np.abs(q - q_ref).mean(0)
    groups = {
        g: float(np.mean([e for n, e in zip(names, err) if re.fullmatch(p, n)]))
        for g, p in THERMAL_GROUPS.items()
    }
    return {
        "mean_abs_rad": float(err.mean()),
        "groups": groups,
        "per_joint": dict(zip(names, map(float, err))),
    }


# ---------------------------------------------------------------------------- report


def compute_report(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    session = json.loads((run / "session.json").read_text())
    sim = np.load(run / "sim.npz")
    meta = json.loads(str(sim["meta"]))
    names: list[str] = meta["joint_names"]
    t_sim = sim["t_mono"]
    release, end = session["release_t"], session["end_t"]
    report: dict[str, Any] = {
        "run": {
            **{k: session[k] for k in ("robot", "bundle", "session", "aborted", "abort_reason")},
            "perturb": session.get("perturb", "none"),
        },
        "sim": {k: meta[k] for k in ("sim_dt", "lowstate_dt", "record_dt")},
    }
    if release is None:
        report["error"] = "elastic band never released: the session did not reach control"
        return report

    # Scored window: band release -> session end, cut at the first fall.
    win = (t_sim >= release) & (t_sim <= end)
    height, tilt = sim["base_pos"][:, 2], tilt_angle(sim["base_quat"])
    idx = np.flatnonzero(win)
    fall = first_fall(height[idx], tilt[idx]) if idx.size else None
    starts = [p["t"] for p in session["phases"]]
    phase_names = [p["name"] for p in session["phases"]]
    if fall is not None:
        i = idx[fall]
        end = float(t_sim[i])
        win &= t_sim <= end
        report["fall"] = {
            "after_release_s": end - release,
            "phase": phase_names[phase_index(t_sim[i : i + 1], starts)[0]],
            "pelvis_z": float(height[i]),
            "tilt_rad": float(tilt[i]),
        }
    else:
        report["fall"] = None
    report["scored_s"] = end - release

    # Torque (thermal proxy) from the simulator: applied, clipped motor torque.
    stats = torque_stats(sim["tau"][win], meta["record_dt"], meta["effort_limit"], names)
    report["torque"] = {
        "threshold_fraction_of_limit": TORQUE_FRACTION,
        "groups": group_torque(stats),
        "per_joint": {
            n: s
            for n, s in stats.items()
            if any(re.fullmatch(p, n) for p in THERMAL_GROUPS.values())
        },
    }
    report["gains_seen"] = {
        "kp": dict(zip(names, map(float, np.median(sim["kp"][win], 0)))),
        "kd": dict(zip(names, map(float, np.median(sim["kd"][win], 0)))),
    } if win.any() else {}  # fmt: skip

    # Tracking: runtime ticks (measured q, IMU) against the reference it logged at the same tick.
    t_tick, q = read_state_csv(run / "deploy_logs" / "q.csv")
    _, base_quat = read_state_csv(run / "deploy_logs" / "base_quat.csv")
    target = read_rows_csv(run / "target_motion.csv")
    n = min(len(t_tick), len(target), len(base_quat))
    t_tick, q, base_quat, target = t_tick[:n], q[:n], base_quat[:n], target[:n]
    tick_win = (t_tick >= release) & (t_tick < end)  # a fall ends tracking before that sample
    z_sim = np.interp(t_tick, t_sim, height)
    report["tracking"] = joint_tracking(q[tick_win], target[tick_win, 7:36], names)
    if tick_win.any():
        report["tracking"]["root_z_mean_abs_m"] = float(
            np.abs(z_sim[tick_win] - target[tick_win, 2]).mean()
        )
        report["tracking"]["tilt_err_mean_rad"] = float(
            tilt_error(target[tick_win, 3:7], base_quat[tick_win]).mean()
        )

    # Per phase (after release).
    ph_sim, ph_tick = phase_index(t_sim, starts), phase_index(t_tick, starts)
    yaw_sim = np.unwrap(yaw(sim["base_quat"]))
    phases = {}
    for k, name in enumerate(phase_names):
        s, tk = win & (ph_sim == k), tick_win & (ph_tick == k)
        if not s.any():
            continue
        entry: dict[str, Any] = {
            "duration_s": float(t_sim[s][-1] - t_sim[s][0]),
            "pelvis_z_min": float(height[s].min()),
            "pelvis_z_mean": float(height[s].mean()),
            "tilt_max_rad": float(tilt[s].max()),
            "forward_speed_mps": float(sim["base_lin_vel"][s, 0].mean()),
            "yaw_change_rad": float(yaw_sim[s][-1] - yaw_sim[s][0]),
            "torque_mean_abs": {
                g: v["mean_abs"]
                for g, v in group_torque(
                    torque_stats(sim["tau"][s], meta["record_dt"], meta["effort_limit"], names)
                ).items()
            },
        }
        if tk.any():
            entry["joint_err_mean_abs_rad"] = float(np.abs(q[tk] - target[tk, 7:36]).mean())
            entry["ref_root_z_mean"] = float(target[tk, 2].mean())
            entry["ref_tilt_max_rad"] = float(tilt_angle(target[tk, 3:7]).max())
            entry["tilt_err_mean_rad"] = float(tilt_error(target[tk, 3:7], base_quat[tk]).mean())
        phases[name] = entry
    report["phases"] = phases

    lag = sim["lag"][win]
    span = t_sim[win]
    report["timing"] = {
        "control_tick": interval_stats(t_tick[tick_win]),
        "runtime_loop": parse_loop_timing((run / "deploy.log").read_text(errors="replace")),
        "sim_lag_ms": {"p99": float(np.percentile(lag, 99) * 1e3), "max": float(lag.max() * 1e3)}
        if lag.size else {},
        "sim_realtime_factor": float(
            (sim["sim_time"][win][-1] - sim["sim_time"][win][0]) / (span[-1] - span[0])
        ) if span.size > 1 else None,
    }  # fmt: skip
    return report


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    """One table row: falls, thermal-group torque, tracking, timing."""
    run, torque, tr = (
        report["run"],
        report.get("torque", {}).get("groups", {}),
        report.get("tracking", {}),
    )
    fall = report.get("fall")
    return {
        "robot": run["robot"],
        "bundle": run["bundle"],
        "session": run["session"],
        "perturb": run.get("perturb", "none"),
        "fall": f"{fall['phase']} +{fall['after_release_s']:.1f}s" if fall else "none",
        "scored_s": round(report.get("scored_s", 0.0), 1),
        **{f"{g}_mean|tau|": round(torque[g]["mean_abs"], 2) for g in torque},
        **{f"{g}_t>{TORQUE_FRACTION:.0%}": round(torque[g]["time_above_s"], 1) for g in torque},
        "joint_err_rad": round(tr.get("mean_abs_rad", float("nan")), 3),
        "root_z_err_m": round(tr.get("root_z_mean_abs_m", float("nan")), 3),
        "tick_p99_ms": round(
            report.get("timing", {}).get("control_tick", {}).get("p99_ms", float("nan")), 2
        ),
    }
