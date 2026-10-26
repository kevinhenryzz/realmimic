"""Offline SONIC-planner rollouts: runtime-loop fidelity, determinism and clip sanity.

The first group checks the stitching/resampling/context code against literal Python ports of the
C++ loops (SONIC ``gear_sonic_deploy`` @ 7f151314) and the replan cadence with a fake planner; it
needs no model. The second group runs ``planner_sonic.onnx`` (path from ``$REALMIMIC_PLANNER_ONNX``
or ``third_party/gear_sonic/planner_sonic.onnx``) and is skipped when it is absent.
"""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path

import numpy as np
import pytest

from realmimic.data import planner_rollouts as pr

ROOT = Path(__file__).resolve().parents[2]
PLANNER = Path(os.environ.get("REALMIMIC_PLANNER_ONNX", ROOT / pr.DEFAULT_PLANNER_PATH))
LAFAN_DIR = Path(os.environ.get("REALMIMIC_LAFAN_NPZ_DIR", ROOT / "data/npz/lafan1"))

# Continuity tolerances for the exported 50 Hz clips inside stitch windows. A "jump" would be a
# single-frame step far above what the motion itself does. Maxima over the 200 x 20 s pilot set
# (data/npz/planner_v0, seed 0), stitch windows vs elsewhere: joint step 0.27 vs 0.24 rad/frame,
# joint second difference 0.26 vs 0.21, root step 0.050 vs 0.043 m/frame, root rotation 0.127 vs
# 0.082 rad/frame. The rotation excess is real runtime behaviour: cross-fading a non-turning old
# plan into a turning new one (facing step) overshoots the yaw rate at the end of the 8-frame
# blend. The bounds sit ~1.5x above those maxima.
MAX_JOINT_STEP = 0.4  # rad per 20 ms frame (20 rad/s)
MAX_JOINT_ACC = 0.4  # rad per frame^2 (|q[t+1] - 2 q[t] + q[t-1]|)
MAX_ROOT_STEP = 0.075  # m per frame (3.75 m/s; walk is capped at 1.5 m/s)
MAX_ROOT_ROT_STEP = 0.2  # rad per frame (10 rad/s)
# Unclamped planner joints leave the g1.xml ranges by up to 0.26 rad (ankle roll, pilot set).
MAX_RAW_LIMIT_EXCESS = 0.35


# ----------------------------------------------------------------------------- literal C++ ports


def _cpp_resample(q30: np.ndarray) -> np.ndarray:
    """ResampleGeneratedSequence50Hz (localmotion_kplanner.hpp:449-498), joints in MuJoCo order."""
    n = q30.shape[0]
    out = np.empty((math.floor(float(n) / 30.0 * 50), 36))
    for f in range(out.shape[0]):
        f30 = f / 50.0 * 30
        f0 = math.floor(f30)
        f1 = min(f0 + 1, n - 1)
        w0 = 1.0 - (f30 - f0)
        w1 = 1.0 - w0
        out[f, :3] = w0 * q30[f0, :3] + w1 * q30[f1, :3]
        out[f, 3:7] = _cpp_slerp(q30[f0, 3:7], q30[f1, 3:7], f30 - f0)
        out[f, 7:] = w0 * q30[f0, 7:] + w1 * q30[f1, 7:]
    return out


def _cpp_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """quat_slerp_d (math_utils.hpp:343-378)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta = math.acos(abs(dot))
    s = math.sin(theta)
    return math.sin((1 - t) * theta) / s * q0 + math.sin(t * theta) / s * q1


def _cpp_context(m: np.ndarray, gen_frame: int) -> np.ndarray:
    """UpdateContextFromMotion (localmotion_kplanner.hpp:637-677) on a (T, 36) buffer."""
    T = m.shape[0]
    ctx = np.empty((4, 36))
    for n in range(4):
        f50 = (gen_frame / 50.0 + n / 30.0) * 50.0
        f0 = min(math.floor(f50), T - 1)
        f1 = min(f0 + 1, T - 1)
        w0 = 1.0 - (f50 - f0)
        w1 = 1.0 - w0
        ctx[n, 3:7] = _cpp_slerp(m[f0, 3:7], m[f1, 3:7], f50 - f0)
        ctx[n, :3] = w0 * m[f0, :3] + w1 * m[f1, :3]
        ctx[n, 7:] = w0 * m[f0, 7:] + w1 * m[f1, 7:]
    return ctx


def _cpp_blend(old: np.ndarray, cf: int, new: np.ndarray, fgen: int) -> np.ndarray:
    """In-place cross-fade of CurrentFrameAdvancement (g1_deploy_onnx_ref.cpp:3235-3293)."""
    T_old = old.shape[0]
    length = fgen - cf + new.shape[0]
    buf = np.zeros((max(length, T_old) + 8, 36))
    buf[:T_old] = old
    start = max(0, fgen - cf)
    for f in range(length):
        f_old = min(max(f + cf, 0), T_old - 1)
        f_new = min(max(f + cf - fgen, 0), new.shape[0] - 1)
        w_new = min(max((f - start) / 8, 0.0), 1.0)
        w_old = 1.0 - w_new
        q0, q1 = buf[f_old, 3:7].copy(), new[f_new, 3:7]
        buf[f, 7:] = w_old * buf[f_old, 7:] + w_new * new[f_new, 7:]
        buf[f, :3] = w_old * buf[f_old, :3] + w_new * new[f_new, :3]
        buf[f, 3:7] = _cpp_slerp(q0, q1, w_new)
    return buf[:length]


def _random_motion(rng: np.random.Generator, n: int) -> np.ndarray:
    """Smooth random (n, 36) qpos with unit quaternions."""
    t = np.linspace(0, 1, n)[:, None]
    q = np.zeros((n, 36))
    q[:, :3] = rng.normal(size=3) + t * rng.normal(size=3)
    axis_angle = rng.normal(size=3) * 0.3 + t * rng.normal(size=3)
    angle = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    q[:, 3:4] = np.cos(angle / 2)
    q[:, 4:7] = np.sin(angle / 2) * axis_angle / angle
    q[:, 7:] = rng.normal(size=29) * 0.5 + np.sin(6 * t + rng.normal(size=29))
    return q


def _motion(q: np.ndarray) -> pr.Motion:
    return pr.Motion(q[:, :3].copy(), q[:, 3:7].copy(), q[:, 7:].copy())


@pytest.mark.parametrize("n", [24, 36, 40, 44, 64])
def test_resample_matches_runtime(n):
    q30 = _random_motion(np.random.default_rng(n), n)
    np.testing.assert_allclose(pr.resample_plan_50hz(q30).qpos(), _cpp_resample(q30), atol=1e-9)


@pytest.mark.parametrize("gen_frame", [0, 2, 7, 60, 70, 80])
def test_context_matches_runtime(gen_frame):
    m = _random_motion(np.random.default_rng(gen_frame), 73)
    got = pr.sample_context(_motion(m), gen_frame)
    np.testing.assert_allclose(got, _cpp_context(m, gen_frame).astype(np.float32), atol=1e-6)


@pytest.mark.parametrize(
    ("t_old", "cf", "fgen", "t_new"),
    [
        (73, 0, 2, 66),
        (73, 5, 7, 73),
        (73, 6, 7, 60),
        (66, 9, 7, 73),
        (12, 11, 13, 60),
        (5, 4, 6, 73),
    ],
)
def test_blend_matches_runtime(t_old, cf, fgen, t_new):
    rng = np.random.default_rng(t_old + cf + fgen)
    old, new = _random_motion(rng, t_old), _random_motion(rng, t_new)
    got = pr.blend_plan(_motion(old), cf, _motion(new), fgen)
    assert got is not None
    np.testing.assert_allclose(got.qpos(), _cpp_blend(old, cf, new, fgen), atol=1e-9)


# ----------------------------------------------------------------------------- fake planner loop


class _FakePlanner:
    """Returns 44 frames that continue the context pose and log every call."""

    def __init__(self):
        self.config = pr.RuntimeConfig()
        self.num_calls, self.total_seconds = 0, 0.0
        self.states: list[pr.MovementState] = []

    def __call__(self, context, state, seed):
        self.num_calls += 1
        self.states.append(state)
        out = np.repeat(np.asarray(context[:1], np.float64), 44, axis=0)
        out[:, 0] += np.arange(44) / 30.0 * max(state.speed, 0.0)
        return out.astype(np.float32)


def test_replan_cadence_and_triggers():
    walk = pr.CommandScript(
        [pr.Phase(0.0, 1.0, "idle"), pr.Phase(1.0, 6.0, "walk", speed=1.0, move_offset=0.0),
         pr.Phase(6.0, 9.0, "squat", height=0.5), pr.Phase(9.0, 10.0, "squat", height=0.6),
         pr.Phase(10.0, 12.0, "idle", yaw_rate=0.5)]
    )  # fmt: skip
    fake = _FakePlanner()
    ro = pr.rollout(fake, walk, 12.0, planner_phase=0, pickup_delay=0)
    by_reason: dict[str, list[int]] = {}
    for s in ro.stitches:
        by_reason.setdefault(s.reason, []).append(s.frame)
    # Mode changes at 1 s, 6 s, 10 s are seen by the next planner tick (after control tick 50,
    # 300, 500) and picked up one tick later.
    assert by_reason["mode"] == [51, 301, 501]
    # Walking with a constant command re-plans only on the float32 timer: +0.1 per planner tick,
    # firing at >= 1.0 (the 10th tick) and reset only when it fires, never by other re-plans. It
    # also fires (ignored) during the static idle, so it runs on a fixed 1 s grid from the start.
    assert by_reason["timer"] == [96, 146, 196, 246, 296]
    # Squat is static: no timer re-plans, only the height change.
    assert by_reason["height"] == [451]
    # Turning in place (idle facing change) re-plans at every planner tick (5 frames).
    assert np.all(np.diff(by_reason["facing"]) == 5)
    # Initial plan + one per stitch + the plan requested at the last tick (frame 600), which the
    # control thread would only pick up after the clip ends.
    assert ro.planner_calls == 1 + len(ro.stitches) + 1
    # The plan is blended in from gen_frame = current_frame + 2, which is 2 frames after the
    # pickup frame with no pickup delay.
    assert all(s.blend_start == s.frame + 2 for s in ro.stitches)


def test_pickup_delay_shifts_blend_start():
    script = pr.CommandScript([pr.Phase(0.0, 3.0, "walk", speed=1.0, move_offset=0.0)])
    ro = pr.rollout(_FakePlanner(), script, 3.0, planner_phase=3, pickup_delay=1)
    assert [s.frame for s in ro.stitches][:1] == [5]  # tick after control tick 3, +1 delay
    assert all(s.blend_start == s.frame + 1 for s in ro.stitches)


def test_command_script_roundtrip_and_ranges():
    rng = np.random.default_rng(0)
    for _ in range(50):
        script = pr.sample_command_script(rng, 20.0)
        assert pr.CommandScript.from_json(script.to_json()).to_json() == script.to_json()
        assert script.phases[0].mode == "idle" and script.phases[0].t0 == 0.0
        assert script.phases[-1].t1 >= 20.0
        for p in script.phases:
            spec = pr.MODES[p.mode]
            assert p.mode == "idle" or p.mode in pr.V0_MODES
            if spec.static:
                assert p.speed == -1.0 and p.move_offset is None
            if spec.standing:
                assert p.height == -1.0
            if spec.height_range:
                assert spec.height_range[0] <= p.height <= spec.height_range[1]
            if spec.speed_range and p.speed != -1.0:
                assert spec.speed_range[0] <= p.speed <= spec.speed_range[1]
        # Every non-idle segment is followed by idle.
        modes = [p.mode for p in script.phases]
        for a, b in itertools.pairwise(modes):
            if a != b and a != "idle":
                assert b == "idle"


# ----------------------------------------------------------------------------- real planner

needs_planner = pytest.mark.skipif(not PLANNER.exists(), reason=f"planner model absent: {PLANNER}")


@pytest.fixture(scope="module")
def planner():
    pytest.importorskip("onnxruntime")
    return pr.PlannerSession(PLANNER, num_threads=4)


@pytest.fixture(scope="module")
def g1():
    from supermimic.data.motion_convert import load_g1_model

    return load_g1_model()


@pytest.fixture(scope="module")
def clips(planner, g1):
    return [pr.generate_clip(planner, s, 14.0, f"t{s}", model=g1) for s in (1, 2, 3, 4)]


@needs_planner
def test_seed_reproducibility(planner, g1, clips):
    again = pr.generate_clip(planner, 1, 14.0, "t1", model=g1)
    for k, v in clips[0].arrays.items():
        np.testing.assert_array_equal(again.arrays[k], v, err_msg=k)
    assert again.script.to_json() == clips[0].script.to_json()
    assert not np.array_equal(clips[1].arrays["joint_pos"], clips[0].arrays["joint_pos"])


@needs_planner
def test_npz_matches_converter_and_lafan(clips, g1, tmp_path):
    from supermimic.data.motion_convert import convert_csv_to_npz

    res = clips[0]
    path = tmp_path / "clip.npz"
    pr.save_npz(path, res.arrays)
    ours = np.load(path)
    # Same stream through the CSV converter itself (LAFAN1 layout: quat xyzw).
    s = res.rollout.stream
    jp = np.clip(s.joints, g1.jnt_range[1:, 0], g1.jnt_range[1:, 1])
    csv = tmp_path / "clip.csv"
    np.savetxt(csv, np.concatenate([s.pos, s.quat[:, [1, 2, 3, 0]], jp], 1), "%.17g", ",")
    convert_csv_to_npz(csv, tmp_path / "ref.npz", input_fps=50.0, model=g1)
    ref = np.load(tmp_path / "ref.npz")
    assert ours.files == ref.files
    for k in ref.files:
        assert ours[k].dtype == ref[k].dtype or k == "source", k
        assert ours[k].shape == ref[k].shape, k
        if ours[k].dtype.kind == "f":
            np.testing.assert_allclose(ours[k], ref[k], atol=2e-5, err_msg=k)
        elif k != "source":
            np.testing.assert_array_equal(ours[k], ref[k], err_msg=k)
    lafan = sorted(LAFAN_DIR.glob("*.npz"))
    if not lafan:
        pytest.skip(f"no LAFAN1 npz in {LAFAN_DIR}")
    la = np.load(lafan[0])
    assert ours.files == la.files
    for k in la.files:
        assert ours[k].dtype.kind == la[k].dtype.kind, k
        assert ours[k].shape[1:] == la[k].shape[1:], k
        if ours[k].dtype.kind == "f":
            assert ours[k].dtype == la[k].dtype, k
    np.testing.assert_array_equal(ours["joint_names"], la["joint_names"])
    np.testing.assert_array_equal(ours["body_names"], la["body_names"])
    np.testing.assert_array_equal(ours["fps"], la["fps"])


@needs_planner
def test_finite_and_joint_limits(clips, g1):
    lo, hi = g1.jnt_range[1:, 0], g1.jnt_range[1:, 1]
    for res in clips:
        for k, v in res.arrays.items():
            if v.dtype.kind == "f":
                assert np.isfinite(v).all(), k
        jp = res.arrays["joint_pos"]
        assert (jp >= lo - 1e-6).all() and (jp <= hi + 1e-6).all()
        # Unclamped planner output: bounded excursions (mostly ankles), see the module doc.
        assert res.raw_limit_excess.max() < MAX_RAW_LIMIT_EXCESS
        q = res.arrays["body_quat_w"]
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)


@needs_planner
@pytest.mark.parametrize(
    ("mode", "height", "z_range"),
    [("idle", -1.0, (0.74, 0.82)), ("squat", 0.5, (0.42, 0.58)), ("squat", 0.7, (0.62, 0.78)),
     ("kneelTwoLeg", 0.3, (0.25, 0.45)), ("kneelOneLeg", 0.3, (0.25, 0.5))],
)  # fmt: skip
def test_root_height_per_mode(planner, g1, mode, height, z_range):
    script = pr.CommandScript(
        [pr.Phase(0.0, 1.0, "idle"), pr.Phase(1.0, 6.0, mode, height=height),
         pr.Phase(6.0, 10.0, "idle")]
    )  # fmt: skip
    arrays = pr.motion_to_npz_arrays(pr.rollout(planner, script, 10.0).stream, g1, "t")
    z = arrays["body_pos_w"][:, 0, 2]
    assert z_range[0] <= z[250:300].min() and z[250:300].max() <= z_range[1]  # 5-6 s: settled
    assert 0.74 <= z[-1] <= 0.82  # back to standing
    assert z.min() > 0.15 and z.max() < 0.85


@needs_planner
def test_walk_moves_at_commanded_speed(planner, g1):
    script = pr.CommandScript(
        [pr.Phase(0.0, 1.0, "idle"),
         pr.Phase(1.0, 8.0, "walk", speed=1.2, facing=math.pi / 2, move_offset=0.0)]
    )  # fmt: skip
    arrays = pr.motion_to_npz_arrays(pr.rollout(planner, script, 8.0).stream, g1, "t")
    root = arrays["body_pos_w"][:, 0].astype(np.float64)
    v = (root[375] - root[175]) / 4.0
    assert 0.8 < v[1] < 1.4 and abs(v[0]) < 0.25  # walks along +y (facing pi/2)
    assert (root[150:, 2] > 0.65).all() and (root[150:, 2] < 0.82).all()


@needs_planner
def test_stitch_continuity(clips):
    for res in clips:
        assert len(res.rollout.stitches) > 0
        c = pr.stitch_continuity(res.arrays, res.rollout.stitches)
        assert c["joint_step_stitch"].max() < MAX_JOINT_STEP
        assert c["joint_acc_stitch"].max() < MAX_JOINT_ACC
        assert c["root_step_stitch"].max() < MAX_ROOT_STEP
        assert c["root_rot_step_stitch"].max() < MAX_ROOT_ROT_STEP
