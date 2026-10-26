"""Offline rollouts of SONIC's kinematic planner (``planner_sonic.onnx``) as 50 Hz motion clips.

At deploy time SONIC's C++ runtime, not a mocap file, produces the reference our tracker follows: a
10 Hz planner thread re-plans from the reference that is currently playing, and the 50 Hz control
thread cross-fades every new plan into it. This module replays that loop offline, driven by
randomized operator command scripts, and writes the *realized* reference (the frame the control
thread plays at each 50 Hz tick) in exactly the npz format of
:func:`supermimic.data.motion_convert.convert_csv_to_npz`.

Runtime semantics reproduced here (SONIC ``gear_sonic_deploy`` at commit 7f151314; paths are
relative to ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/``):

* **Planner model I/O** (``docs/source/references/planner_onnx.md``): 11 inputs, output
  ``mujoco_qpos [1, 64, 36]`` at 30 Hz (root xyz, root quat **wxyz**, 29 joints in MuJoCo/hardware
  order) of which the first ``num_pred_frames`` (int32 ``[1]`` in this export) are valid. The
  deployed file lives at ``planner/target_vel/V2/`` (``download_from_hf.py:51``), so the runtime
  treats it as version 2 (27 modes) (``src/g1_deploy_onnx_ref.cpp:2431-2449``).
* **Advanced inputs** are fixed by the active TensorRT backend
  (``include/localmotion_kplanner_tensorrt.hpp:90-102``): ``has_specific_target = 0``, zero
  waypoint positions/headings, ``allowed_pred_num_tokens = [0,0,0,1,1,1,0,0,0,0,0]`` (9-11 tokens,
  i.e. 36-44 frames at 30 Hz). (The deprecated ONNX backend allows 6-11 tokens instead,
  ``include/localmotion_kplanner_onnx.hpp:155-163``.) ``random_seed`` is
  ``PlannerConfig::initial_random_seed = 1234`` (``include/localmotion_kplanner.hpp:217``) and is
  never changed afterwards: ``Planner()`` calls ``UpdatePlanning`` without a seed
  (``src/g1_deploy_onnx_ref.cpp:3769-3777``) and the backend only overwrites it for seeds != -1
  (``localmotion_kplanner_tensorrt.hpp:409-412``).
* **Initialisation** (``include/localmotion_kplanner.hpp:332-393, 591-624``): 4 identical context
  frames at the origin, identity orientation (the robot's IMU quaternion is ignored),
  ``z = 0.788740`` and the robot's *measured* joint positions (``src/g1_deploy_onnx_ref.cpp:
  3609-3615``); inputs IDLE, ``target_vel = height = -1``, no movement, facing +x. The first
  plan is copied verbatim into the playing reference (``src/g1_deploy_onnx_ref.cpp:3199-3222``).
* **Context for re-plans** (``include/localmotion_kplanner.hpp:557, 628-678``): taken from the
  *playing reference*, not the robot state: ``gen_frame = current_frame + 2``
  (``motion_look_ahead_steps``, line 215) and 4 samples at 30 Hz spacing from
  ``gen_frame / 50`` s, linearly interpolated between 50 Hz frames (slerp for the root
  quaternion), indices clamped to the last frame.
* **30 Hz -> 50 Hz** (``include/localmotion_kplanner.hpp:395-519``): ``floor(N / 30 * 50)`` frames,
  50 Hz frame ``k`` samples 30 Hz frame ``k * 30 / 50`` (lerp, slerp for the quaternion), the
  upper index clamped to ``N - 1`` (so the tail holds the last frame); a plan with more than 64
  frames or any NaN is an error that disables the planner.
* **Re-plan decision** (``src/g1_deploy_onnx_ref.cpp:3655-3736``), every 0.1 s
  (``planner_dt_``, line 2187), comparing the latest operator ``MovementState`` with the one used
  for the last plan (initialised to IDLE / 0 / +x / -1 / -1, line 230): always re-plan when mode,
  facing or height changed; for non-static modes also when speed or movement direction changed,
  or when a float32 counter (incremented by 0.1 each tick, reset whenever it reaches the mode's
  interval: run 0.1 s, handCrawling 0.2 s, otherwise 1.0 s, lines 231-235, 3694-3725) fires and
  the speed is non-zero. Static modes (``include/localmotion_kplanner.hpp:108-115``) never
  re-plan on the timer.
* **Stitching** (``src/g1_deploy_onnx_ref.cpp:3226-3295, 3325``): when the control thread picks
  up a new plan, the old reference is rebased so ``current_frame`` becomes frame 0, the new plan
  is placed at ``gen_frame - current_frame`` and cross-faded in over 8 frames starting there
  (lerp for joints and root position, slerp for the root quaternion; old frames clamp to the old
  last frame), the blend is done in place, and ``current_frame`` is reset to 0. Afterwards the
  control thread advances one frame per tick and holds the last frame when the reference runs out
  (lines 3336-3339).

Operator commands follow the gamepad / keyboard handlers (``include/input_interface/gamepad.hpp:
300-404, 684-708``, ``keyboard_handler.hpp:260-296, 556-683``): facing and movement directions
are world-frame unit vectors ``(cos a, sin a, 0)`` in the planner's frame (which starts at yaw 0),
movement = facing angle + a stick offset, speeds slowWalk 0.2-0.8, walk 0.8-1.5, run 1.5-3.0 m/s
(keyboard walk uses the mode default, -1), heights of squat 0.4-0.8 m and kneel 0.2-0.4 m (the
documented ranges in ``planner_onnx.md``; the gamepad clamps all three to 0.1-0.8), standing
modes send ``height = -1`` and static modes ``target_vel = -1`` with zero movement.

Assumptions (verify against a real session log when one is available):

* **Open loop.** We have no robot, so the context comes from the planner's own playing reference.
  That *is* what the runtime does for re-plans (see above); the only robot-state inputs are the
  joint positions at initialisation (we use SONIC's ``default_angles``,
  ``include/policy_parameters.hpp:210-240``, i.e. a robot that settled into the standing pose)
  and the IDLE re-adaptation at the end of a held reference (``src/g1_deploy_onnx_ref.cpp:
  3340-3401``), which blends the lower-body reference 2 %/tick toward the *measured* joints when
  the tracking error exceeds 0.10 rad. With good tracking it is a no-op (it blends toward the
  stored target), so it is skipped. Clips therefore match the runtime only while tracking is good.
* **Thread timing.** Control ticks are exactly 0.02 s and planner ticks exactly every 5th control
  tick, with a per-clip random phase; a new plan is picked up ``pickup_delay`` control ticks after
  the one following the planner tick. The TensorRT latency on the robot was not measured; with
  < 20 ms latency and a random thread phase the delay is 0 or 1 tick, sampled per clip.
* **Realized reference.** We store the frame the control thread plays at each tick. The policy
  additionally previews future frames of the *current* plan, which a later re-plan may replace;
  clips built from the realized stream cannot represent those discarded previews.
* **Velocities** are recomputed from the realized stream with ``motion_convert``'s conventions
  (``np.gradient``, central SO(3) differences, MuJoCo FK), not the runtime's blended forward
  differences, so planner clips and LAFAN1 clips are consistent.
* **Determinism.** A clip is a pure function of its seed for a fixed onnxruntime thread count;
  CPU kernels partition reductions by thread count, so other counts differ by ~1e-4 rad.

Known planner artefacts (measured on the 200 x 20 s pilot set, ``g1.xml`` collision geometry):
stance soles sit ~8 mm below the floor and kneeling shins/knees ~4.5 cm below it; the
initialisation context (``z = 0.788740`` with ``default_angles``) floats the soles 3 cm for the
first frames; swing feet clear only ~4 cm and touch down while still moving at 2-3 m/s for
~4 frames per step; each cross-fade absorbs a 0.14 rad median (0.42 rad p99) joint mismatch
between old and new plan, doubling the p99 joint second difference inside stitch windows and
overshooting the yaw rate after facing steps.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    NUM_JOINTS,
    forward_kinematics,
    load_g1_model,
    quat_wxyz_to_xyzw,
    resample_motion,
)

DEFAULT_PLANNER_PATH = Path("third_party/gear_sonic/planner_sonic.onnx")

QPOS_DIM = 7 + NUM_JOINTS  # 36
PLANNER_FPS = 30.0
CONTROL_FPS = 50.0
MAX_PLAN_FRAMES_30HZ = 64  # localmotion_kplanner.hpp:400

# SONIC default_angles (policy_parameters.hpp:210-240), MuJoCo/hardware order == G1_JOINT_NAMES.
SONIC_DEFAULT_JOINT_POS: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)  # fmt: skip


# ----------------------------------------------------------------------------- modes


@dataclass(frozen=True)
class ModeSpec:
    """One planner mode and how an operator commands it."""

    name: str
    index: int
    static: bool  # is_static_motion_mode (localmotion_kplanner.hpp:108-115)
    standing: bool  # is_standing_motion_mode (localmotion_kplanner.hpp:117-139): height = -1
    speed_range: tuple[float, float] | None = None  # None: mode default (-1)
    height_range: tuple[float, float] | None = None


# Indices/names from planner_onnx.md and the LocomotionMode enum (localmotion_kplanner.hpp:78-106).
# Speed ranges: gamepad.hpp:353-367 (slowWalk, walk, run), keyboard_handler.hpp:277-290 (crawling,
# boxing); heights: planner_onnx.md (squat ~0.4-0.8, kneel 0.2-0.4), keyboard clamp 0.2-0.8 for
# the remaining ground modes (keyboard_handler.hpp:263-265).
MODES: dict[str, ModeSpec] = {
    m.name: m
    for m in (
        ModeSpec("idle", 0, static=True, standing=True),
        ModeSpec("slowWalk", 1, static=False, standing=True, speed_range=(0.2, 0.8)),
        ModeSpec("walk", 2, static=False, standing=True, speed_range=(0.8, 1.5)),
        ModeSpec("run", 3, static=False, standing=True, speed_range=(1.5, 3.0)),
        ModeSpec("squat", 4, static=True, standing=False, height_range=(0.4, 0.8)),
        ModeSpec("kneelTwoLeg", 5, static=True, standing=False, height_range=(0.2, 0.4)),
        ModeSpec("kneelOneLeg", 6, static=True, standing=False, height_range=(0.2, 0.4)),
        ModeSpec("lyingFacedown", 7, static=True, standing=False, height_range=(0.2, 0.8)),
        ModeSpec("handCrawling", 8, False, False, speed_range=(0.4, 1.0), height_range=(0.2, 0.8)),
        ModeSpec("idleBoxing", 9, static=True, standing=True),
        ModeSpec("walkBoxing", 10, static=False, standing=True, speed_range=(0.7, 1.5)),
        ModeSpec("leftJab", 11, static=False, standing=True, speed_range=(0.7, 1.5)),
        ModeSpec("rightJab", 12, static=False, standing=True, speed_range=(0.7, 1.5)),
        ModeSpec("randomPunches", 13, static=False, standing=True, speed_range=(0.7, 1.5)),
        ModeSpec("elbowCrawling", 14, False, False, speed_range=(0.7, 1), height_range=(0.2, 0.8)),
        ModeSpec("leftHook", 15, static=False, standing=True, speed_range=(0.7, 1.5)),
        ModeSpec("rightHook", 16, static=False, standing=True, speed_range=(0.7, 1.5)),
        *(
            ModeSpec(name, idx, static=False, standing=True)
            for idx, name in enumerate(
                ("happy", "stealth", "injured", "careful", "objectCarrying", "crouch",
                 "happyDance", "zombie", "point", "scared"),
                start=17,
            )
        ),
    )
}  # fmt: skip
V0_MODES: tuple[str, ...] = ("slowWalk", "walk", "squat", "kneelTwoLeg", "kneelOneLeg")
"""Non-idle modes sampled by default (v0: no run, lying, crawling, boxing or styled walks)."""

RUN_INDEX, CRAWL_INDEX = MODES["run"].index, MODES["handCrawling"].index
PUNCH_INDICES = frozenset(MODES[n].index for n in ("leftJab", "rightJab", "randomPunches",
                                                   "leftHook", "rightHook"))  # fmt: skip
STATIC_INDICES = frozenset(m.index for m in MODES.values() if m.static)
MODE_NAMES = {m.index: m.name for m in MODES.values()}


# ----------------------------------------------------------------------------- runtime config


@dataclass(frozen=True)
class RuntimeConfig:
    """Constants of the SONIC runtime loop (see the module docstring for sources)."""

    control_dt: float = 0.02  # g1_deploy_onnx_ref.cpp:2186
    planner_dt: float = 0.1  # g1_deploy_onnx_ref.cpp:2187
    planner_ticks: int = 5  # control ticks per planner tick
    look_ahead_steps: int = 2  # localmotion_kplanner.hpp:215
    blend_frames: int = 8  # g1_deploy_onnx_ref.cpp:3248
    default_height: float = 0.788740  # localmotion_kplanner.hpp:216
    random_seed: int = 1234  # localmotion_kplanner.hpp:217
    allowed_pred_num_tokens: tuple[int, ...] = (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)
    replan_interval: float = 1.0  # g1_deploy_onnx_ref.cpp:231-234
    replan_interval_running: float = 0.1
    replan_interval_crawling: float = 0.2
    replan_interval_boxing: float = 1.0
    init_joint_pos: tuple[float, ...] = SONIC_DEFAULT_JOINT_POS


@dataclass(frozen=True)
class MovementState:
    """Operator command as read by the planner thread (localmotion_kplanner.hpp:60-73)."""

    mode: int
    movement_direction: tuple[float, float, float]
    facing_direction: tuple[float, float, float]
    speed: float
    height: float


class PlannerError(RuntimeError):
    """The planner returned an invalid plan (the runtime disables the planner in this case)."""


# ----------------------------------------------------------------------------- planner session


class PlannerLike(Protocol):
    """What :func:`rollout` needs from a planner (lets tests substitute a fake)."""

    config: RuntimeConfig
    num_calls: int
    total_seconds: float

    def __call__(self, context: np.ndarray, state: MovementState, seed: int) -> np.ndarray: ...


class PlannerSession:
    """``planner_sonic.onnx`` on onnxruntime's CPU execution provider, with the runtime defaults."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_PLANNER_PATH,
        num_threads: int = 4,
        config: RuntimeConfig | None = None,
    ):
        import onnxruntime as ort

        self.config = config or RuntimeConfig()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        expected = {
            "context_mujoco_qpos", "target_vel", "mode", "movement_direction", "facing_direction",
            "height", "random_seed", "has_specific_target", "specific_target_positions",
            "specific_target_headings", "allowed_pred_num_tokens",
        }  # fmt: skip
        if self.input_names != expected:
            raise ValueError(f"unexpected planner inputs: {sorted(self.input_names)}")
        self.num_calls = 0
        self.total_seconds = 0.0

    def __call__(self, context: np.ndarray, state: MovementState, seed: int) -> np.ndarray:
        """Run one plan. ``context`` is (4, 36); returns the valid (N, 36) float32 30 Hz frames."""
        cfg = self.config
        feeds = {
            "context_mujoco_qpos": np.asarray(context, np.float32).reshape(1, 4, QPOS_DIM),
            "target_vel": np.array([state.speed], np.float32),
            # TensorRT backend clamps out-of-range modes to IDLE (tensorrt.hpp:384-388).
            "mode": np.array([state.mode if 0 <= state.mode < 27 else 0], np.int64),
            "movement_direction": np.array([state.movement_direction], np.float32),
            "facing_direction": np.array([state.facing_direction], np.float32),
            "height": np.array([state.height], np.float32),
            "random_seed": np.array([seed], np.int64),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.array([cfg.allowed_pred_num_tokens], np.int64),
        }
        t0 = time.perf_counter()
        outputs = self.session.run(None, feeds)
        self.total_seconds += time.perf_counter() - t0
        self.num_calls += 1
        qpos = np.asarray(outputs[0])
        n = int(np.asarray(outputs[1]).reshape(-1)[0])
        if not 2 <= n <= MAX_PLAN_FRAMES_30HZ:
            raise PlannerError(f"planner returned {n} frames")
        out = qpos[0, :n]
        if not np.isfinite(out).all():
            raise PlannerError("planner output contains non-finite values")
        return out


# ----------------------------------------------------------------------------- motion buffers


@dataclass
class Motion:
    """A 50 Hz reference buffer: root position, root quaternion (wxyz), joints (MuJoCo order)."""

    pos: np.ndarray  # (T, 3)
    quat: np.ndarray  # (T, 4)
    joints: np.ndarray  # (T, 29)

    def __len__(self) -> int:
        return int(self.pos.shape[0])

    def qpos(self) -> np.ndarray:
        """(T, 36) MuJoCo qpos."""
        return np.concatenate([self.pos, self.quat, self.joints], axis=1)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """``quat_slerp_d`` (math_utils.hpp:343), vectorised: shortest path; lerp if nearly equal."""
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    lin = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.where(lin, 1.0, np.sin(theta))
    w0 = np.where(lin, 1.0 - t, np.sin((1.0 - t) * theta) / sin_theta)
    w1 = np.where(lin, t, np.sin(t * theta) / sin_theta)
    out = w0 * q0 + w1 * q1
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.where(lin, out / norm, out)


def resample_plan_50hz(qpos30: np.ndarray) -> Motion:
    """``ResampleGeneratedSequence50Hz`` via ``motion_convert.resample_motion``.

    The runtime emits ``floor(N / 30 * 50)`` frames and clamps the upper interpolation index to the
    last 30 Hz frame; appending one copy of the last frame lets ``resample_motion`` (same time
    mapping, same lerp/slerp) produce exactly that, after truncation.
    """
    n = int(qpos30.shape[0])
    num_out = math.floor(float(n) / PLANNER_FPS * CONTROL_FPS)
    q = np.asarray(qpos30, dtype=np.float64)
    raw = np.concatenate([q[:, 0:3], quat_wxyz_to_xyzw(q[:, 3:7]), q[:, 7:]], axis=1)
    raw = np.concatenate([raw, raw[-1:]], axis=0)
    m = resample_motion(raw, PLANNER_FPS, CONTROL_FPS)
    assert m.num_frames >= num_out, (m.num_frames, num_out)
    return Motion(m.root_pos[:num_out], m.root_quat[:num_out], m.joint_pos[:num_out])


def sample_context(motion: Motion, gen_frame: int) -> np.ndarray:
    """``UpdateContextFromMotion``: 4 frames at 30 Hz from 50 Hz ``gen_frame``; float32 (4, 36)."""
    T = len(motion)
    t = gen_frame / CONTROL_FPS + np.arange(4) / PLANNER_FPS
    f = t * CONTROL_FPS
    f0 = np.minimum(np.floor(f).astype(np.int64), T - 1)
    f1 = np.minimum(f0 + 1, T - 1)
    w1 = (f - f0)[:, None]
    w0 = 1.0 - w1
    ctx = np.empty((4, QPOS_DIM))
    ctx[:, 0:3] = w0 * motion.pos[f0] + w1 * motion.pos[f1]
    ctx[:, 3:7] = quat_slerp(motion.quat[f0], motion.quat[f1], f - f0)
    ctx[:, 7:] = w0 * motion.joints[f0] + w1 * motion.joints[f1]
    return ctx.astype(np.float32)


def blend_plan(
    motion: Motion, current_frame: int, plan: Motion, gen_frame: int, blend_frames: int = 8
) -> Motion | None:
    """``CurrentFrameAdvancement`` stitching; returns the rebased buffer (``current_frame`` -> 0).

    Returns None when the plan arrived too late to contribute (the runtime then keeps playing the
    old buffer). The C++ loop blends in place, so for ``f >= len(motion)`` the clamped "old" frame
    it reads is the already blended frame ``len(motion) - 1``; this is reproduced exactly.
    """
    t_old, t_new = len(motion), len(plan)
    length = gen_frame - current_frame + t_new
    if length <= 0:
        return None
    f = np.arange(length)
    f_old = np.clip(f + current_frame, 0, t_old - 1)
    f_new = np.clip(f + current_frame - gen_frame, 0, t_new - 1)
    blend_start = max(0, gen_frame - current_frame)
    w_new = np.clip((f - blend_start) / blend_frames, 0.0, 1.0)

    def mix(old: np.ndarray, new: np.ndarray, quat: bool) -> np.ndarray:
        if quat:
            return quat_slerp(old, new, w_new[: len(old)])
        w = w_new[: len(old), None]
        return (1.0 - w) * old + w * new

    out = []
    for old, new, quat in ((motion.pos, plan.pos, False), (motion.quat, plan.quat, True),
                           (motion.joints, plan.joints, False)):  # fmt: skip
        head = min(length, t_old)  # frames whose old source has not been overwritten yet
        res = np.empty((length, old.shape[1]))
        res[:head] = mix(old[f_old[:head]], new[f_new[:head]], quat)
        if length > head:
            src = np.repeat(res[t_old - 1 : t_old], length - head, axis=0)
            tail_w = w_new[head:]
            if quat:
                res[head:] = quat_slerp(src, new[f_new[head:]], tail_w)
            else:
                res[head:] = (1.0 - tail_w[:, None]) * src + tail_w[:, None] * new[f_new[head:]]
        out.append(res)
    return Motion(*out)


def initial_context(config: RuntimeConfig) -> np.ndarray:
    """``InitializeContext``: 4 identical frames at the origin, zero yaw, default height."""
    ctx = np.zeros((4, QPOS_DIM), np.float32)
    ctx[:, 2] = config.default_height
    ctx[:, 3] = 1.0
    ctx[:, 7:] = np.asarray(config.init_joint_pos, np.float32)
    return ctx


INIT_STATE = MovementState(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -1.0, -1.0)


# ----------------------------------------------------------------------------- command scripts


@dataclass
class Phase:
    """Constant operator input over ``[t0, t1)`` (facing may rotate at ``yaw_rate``)."""

    t0: float
    t1: float
    mode: str
    speed: float = -1.0
    height: float = -1.0
    facing: float = 0.0  # world yaw (rad) at t0
    yaw_rate: float = 0.0  # rad/s, right stick
    move_offset: float | None = None  # movement angle relative to facing; None: no movement


INPUT_DT = 0.01  # input thread period (planner_onnx.md "input thread (100 Hz)")


@dataclass
class CommandScript:
    """A piecewise-constant operator command timeline."""

    phases: list[Phase]

    def state_at(self, t: float) -> MovementState:
        """MovementState the input thread would hold at time ``t`` (facing updated at 100 Hz)."""
        ph = self.phases[-1]
        for p in self.phases:
            if t < p.t1:
                ph = p
                break
        spec = MODES[ph.mode]
        steps = math.floor((max(t, ph.t0) - ph.t0) / INPUT_DT + 1e-9)
        yaw = ph.facing + ph.yaw_rate * steps * INPUT_DT
        facing = (math.cos(yaw), math.sin(yaw), 0.0)
        if ph.move_offset is None or spec.static:
            move = (0.0, 0.0, 0.0)
        else:
            a = yaw + ph.move_offset
            move = (math.cos(a), math.sin(a), 0.0)
        return MovementState(spec.index, move, facing, float(ph.speed), float(ph.height))

    def end_facing(self) -> float:
        p = self.phases[-1]
        return p.facing + p.yaw_rate * (p.t1 - p.t0)

    def mode_seconds(self, duration: float) -> dict[str, float]:
        """Commanded time per mode, clipped to ``[0, duration]``."""
        out: dict[str, float] = {}
        for p in self.phases:
            dt = max(0.0, min(p.t1, duration) - p.t0)
            if dt > 0:
                out[p.mode] = out.get(p.mode, 0.0) + dt
        return out

    def to_json(self) -> str:
        rows = []
        for p in self.phases:
            d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(p).items()}
            rows.append(d)
        return json.dumps(rows, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> CommandScript:
        return cls([Phase(**d) for d in json.loads(text)])


@dataclass(frozen=True)
class ScriptConfig:
    """Distribution of randomized command scripts."""

    modes: tuple[str, ...] = V0_MODES
    mode_weights: tuple[float, ...] | None = None  # None: uniform over ``modes``
    first_idle_s: tuple[float, float] = (0.5, 2.0)
    idle_s: tuple[float, float] = (1.0, 3.0)  # idle between non-idle segments
    locomotion_s: tuple[float, float] = (2.0, 6.0)
    ground_s: tuple[float, float] = (3.0, 7.0)  # squat / kneel / lying (incl. getting down)
    max_phases: int = 3  # sub-phases with new speed / direction / height per segment
    p_keyboard_walk_speed: float = 0.25  # walk at mode default speed (-1), as the keyboard sends
    p_turn: float = 0.4  # continuous right-stick turning during a phase
    yaw_rate: tuple[float, float] = (0.3, 1.5)  # |rad/s|; gamepad max is 0.02 rad/tick at 100 Hz
    p_facing_step: float = 0.3  # discrete facing step (keyboard Q/E: pi/6) at a phase start
    p_idle_turn: float = 0.3  # facing change while idle (turn in place)


def _sample_move_offset(rng: np.random.Generator) -> float:
    r = rng.random()
    if r < 0.5:
        return 0.0  # forward
    if r < 0.6:
        return math.pi  # backward
    if r < 0.8:
        return float(rng.choice([math.pi / 2, -math.pi / 2]))  # strafe
    return float(rng.uniform(-math.pi, math.pi))  # arbitrary stick direction


def _facing_step(rng: np.random.Generator) -> float:
    return float(rng.choice([-1, 1]) * rng.integers(1, 4) * math.pi / 6)


def sample_command_script(
    rng: np.random.Generator, seconds: float, cfg: ScriptConfig | None = None
) -> CommandScript:
    """Random script: idle, then {mode segment, idle} pairs until ``seconds`` is covered."""
    cfg = cfg or ScriptConfig()
    weights = np.asarray(cfg.mode_weights or [1.0] * len(cfg.modes), dtype=np.float64)
    weights = weights / weights.sum()
    phases: list[Phase] = []
    t, yaw = 0.0, 0.0

    def add(p: Phase) -> None:
        nonlocal t, yaw
        phases.append(p)
        t = p.t1
        yaw = p.facing + p.yaw_rate * (p.t1 - p.t0)

    def idle(duration: float) -> None:
        face = yaw
        rate = 0.0
        if phases and rng.random() < cfg.p_idle_turn:
            if rng.random() < 0.5:
                face += _facing_step(rng)
            else:
                rate = float(rng.choice([-1, 1]) * rng.uniform(*cfg.yaw_rate))
                # Turn for part of the idle segment, then hold.
                turn = float(rng.uniform(0.3, 1.0)) * duration
                add(Phase(t, t + turn, "idle", facing=face, yaw_rate=rate))
                duration -= turn
                face, rate = yaw, 0.0
        add(Phase(t, t + duration, "idle", facing=face, yaw_rate=rate))

    idle(float(rng.uniform(*cfg.first_idle_s)))
    while t < seconds:
        spec = MODES[str(rng.choice(cfg.modes, p=weights))]
        span = cfg.ground_s if not spec.standing else cfg.locomotion_s
        duration = float(rng.uniform(*span))
        n_phases = int(rng.integers(1, cfg.max_phases + 1))
        cuts = np.sort(rng.uniform(0.0, duration, size=n_phases - 1))
        bounds = [0.0, *cuts.tolist(), duration]
        t_seg = t
        keyboard_speed = spec.name == "walk" and rng.random() < cfg.p_keyboard_walk_speed
        for a, b in itertools.pairwise(bounds):
            face = yaw
            if rng.random() < cfg.p_facing_step and (not spec.static or a == 0.0):
                face += _facing_step(rng)
            rate = 0.0
            if not spec.static and rng.random() < cfg.p_turn:
                rate = float(rng.choice([-1, 1]) * rng.uniform(*cfg.yaw_rate))
            speed, height, offset = -1.0, -1.0, None
            if spec.speed_range is not None and not spec.static:
                speed = -1.0 if keyboard_speed else float(rng.uniform(*spec.speed_range))
            if spec.height_range is not None and not spec.standing:
                height = float(rng.uniform(*spec.height_range))
            if not spec.static:
                offset = _sample_move_offset(rng)
            add(Phase(t_seg + a, t_seg + b, spec.name, speed, height, face, rate, offset))
        idle(float(rng.uniform(*cfg.idle_s)))
    return CommandScript(phases)


# ----------------------------------------------------------------------------- rollout


@dataclass
class StitchRecord:
    """One plan pickup by the control thread."""

    frame: int  # global 50 Hz index of the tick that picked up the plan
    blend_start: int  # global index where the cross-fade starts (weight 0)
    reason: str
    mode: int
    plan_frames_30hz: int
    old_new_joint_gap: float  # max |q_old - q_new| over the blend window (rad), before blending
    old_new_root_gap: float  # max |p_old - p_new| over the blend window (m)


@dataclass
class Rollout:
    """Result of :func:`rollout`."""

    stream: Motion  # realized 50 Hz reference
    stitches: list[StitchRecord] = field(default_factory=list)
    planner_calls: int = 0
    planner_seconds: float = 0.0


def _replan_reason(state: MovementState, last: MovementState, time_to_replan: bool) -> str | None:
    """Replan decision of ``Planner()`` (g1_deploy_onnx_ref.cpp:3727-3731)."""
    if state.mode != last.mode:
        return "mode"
    if state.facing_direction != last.facing_direction:
        return "facing"
    if state.height != last.height:
        return "height"
    if state.mode in STATIC_INDICES:
        return None
    if state.speed != last.speed:
        return "speed"
    if state.movement_direction != last.movement_direction:
        return "direction"
    if time_to_replan and state.speed != 0:
        return "timer"
    return None


def rollout(
    planner: PlannerLike,
    script: CommandScript,
    seconds: float,
    *,
    planner_phase: int = 0,
    pickup_delay: int = 0,
    seed_fn: Callable[[int], int] | None = None,
) -> Rollout:
    """Replay the runtime's planner/control loop for ``seconds`` and return the realized stream.

    ``planner_phase`` (0-4) is the control tick after which the planner thread ticks;
    ``pickup_delay`` is the number of extra control ticks before a plan is picked up;
    ``seed_fn(call_index)`` overrides the runtime's constant planner seed.
    """
    cfg = planner.config
    calls0, secs0 = planner.num_calls, planner.total_seconds
    seed_of = seed_fn or (lambda _i: cfg.random_seed)
    n_frames = round(seconds * CONTROL_FPS) + 1
    plan0 = planner(initial_context(cfg), INIT_STATE, seed_of(0))
    motion = resample_plan_50hz(plan0)  # first plan is copied verbatim
    cf = 0
    last = INIT_STATE
    counter = np.float32(0.0)
    pending: tuple[int, Motion, int, str, int, int] | None = None
    stream = np.empty((n_frames, QPOS_DIM))
    stitches: list[StitchRecord] = []
    n_calls = 1
    for k in range(n_frames):
        stream[k, 0:3] = motion.pos[cf]
        stream[k, 3:7] = motion.quat[cf]
        stream[k, 7:] = motion.joints[cf]
        if pending is not None and pending[0] == k:
            _, plan, fgen, reason, mode, n30 = pending
            pending = None
            new = blend_plan(motion, cf, plan, fgen, cfg.blend_frames)
            if new is not None:
                start = max(0, fgen - cf)
                win = slice(start, min(start + cfg.blend_frames + 1, len(new)))
                f_old = np.clip(np.arange(win.start, win.stop) + cf, 0, len(motion) - 1)
                f_new = np.clip(np.arange(win.start, win.stop) + cf - fgen, 0, len(plan) - 1)
                jgap = float(np.abs(motion.joints[f_old] - plan.joints[f_new]).max())
                pgap = float(np.linalg.norm(motion.pos[f_old] - plan.pos[f_new], axis=1).max())
                stitches.append(StitchRecord(k, k + start, reason, mode, n30, jgap, pgap))
                motion, cf = new, 0
        cf = min(cf + 1, len(motion) - 1)
        if (k - planner_phase) % cfg.planner_ticks != 0 or k < planner_phase:
            continue
        # ---- planner thread tick (after control tick k) ----
        state = script.state_at((k + 1) * cfg.control_dt)
        counter = np.float32(float(counter) + cfg.planner_dt)  # float += double (line 3695)
        if state.mode == RUN_INDEX:
            interval = cfg.replan_interval_running
        elif state.mode == CRAWL_INDEX:
            interval = cfg.replan_interval_crawling
        elif state.mode in PUNCH_INDICES:
            interval = cfg.replan_interval_boxing
        else:
            interval = cfg.replan_interval
        time_to_replan = bool(counter >= np.float32(interval))
        if time_to_replan:
            counter = np.float32(0.0)
        reason = _replan_reason(state, last, time_to_replan)
        if reason is None:
            continue
        last = state
        fgen = cf + cfg.look_ahead_steps
        qpos30 = planner(sample_context(motion, fgen), state, seed_of(n_calls))
        n_calls += 1
        pending = (k + 1 + pickup_delay, resample_plan_50hz(qpos30), fgen, reason, state.mode,
                   int(qpos30.shape[0]))  # fmt: skip
    root_q = stream[:, 3:7] / np.linalg.norm(stream[:, 3:7], axis=1, keepdims=True)
    return Rollout(
        stream=Motion(stream[:, 0:3], root_q, stream[:, 7:]),
        stitches=stitches,
        planner_calls=planner.num_calls - calls0,
        planner_seconds=planner.total_seconds - secs0,
    )


# ----------------------------------------------------------------------------- npz export


def joint_limit_excess(joints: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Per frame, the largest amount (rad) by which a joint leaves its ``g1.xml`` range."""
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    return np.maximum(lo - joints, joints - hi).max(axis=1)


def motion_to_npz_arrays(
    stream: Motion, model: mujoco.MjModel, source: str, clip_joint_limits: bool = True
) -> dict[str, np.ndarray]:
    """Arrays with exactly the keys/dtypes ``convert_csv_to_npz`` writes, for a 50 Hz stream.

    Velocities and body states come from ``resample_motion`` (50 -> 50 Hz, i.e. only the velocity
    computation) and ``forward_kinematics``; ``input_fps`` records the 50 Hz stream rate.
    ``clip_joint_limits`` clamps joints to the ``g1.xml`` ranges first: planner poses leave the
    ankle ranges (kneeling, and ankle roll in some walking steps) by up to ~0.26 rad, which no
    tracker can follow (the runtime itself does not clamp; the rollout context is always the
    unclamped stream).
    """
    joints = stream.joints
    if clip_joint_limits:
        joints = np.clip(joints, model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    raw = np.concatenate([stream.pos, quat_wxyz_to_xyzw(stream.quat), joints], axis=1)
    motion = resample_motion(raw, CONTROL_FPS, CONTROL_FPS)
    bodies = forward_kinematics(model, motion)
    # Same keys, order and dtypes as convert_csv_to_npz's np.savez call.
    return {
        "fps": np.array([CONTROL_FPS], dtype=np.float32),
        "joint_pos": motion.joint_pos.astype(np.float32),
        "joint_vel": motion.joint_vel.astype(np.float32),
        "body_pos_w": bodies.pos_w.astype(np.float32),
        "body_quat_w": bodies.quat_w.astype(np.float32),
        "body_lin_vel_w": bodies.lin_vel_w.astype(np.float32),
        "body_ang_vel_w": bodies.ang_vel_w.astype(np.float32),
        "joint_names": np.array(G1_JOINT_NAMES),
        "body_names": np.array(bodies.body_names),
        "source": np.array(source),
        "input_fps": np.array([CONTROL_FPS], dtype=np.float32),
    }


# ----------------------------------------------------------------------------- one clip


@dataclass
class ClipResult:
    """One generated clip and its metadata."""

    name: str
    seed: int
    arrays: dict[str, np.ndarray]
    script: CommandScript
    rollout: Rollout
    planner_phase: int
    pickup_delay: int
    raw_limit_excess: np.ndarray  # per-frame joint-limit excess of the unclamped stream (rad)

    @property
    def duration_s(self) -> float:
        return (self.arrays["joint_pos"].shape[0] - 1) / CONTROL_FPS

    def manifest_row(self) -> dict[str, str | int | float]:
        """One row of the dataset manifest (``name``/``keep`` as ``discover_npz`` expects)."""
        st = self.rollout.stitches
        reasons: dict[str, int] = {}
        for r in st:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
        modes = self.script.mode_seconds(self.duration_s)
        return {
            "name": self.name,
            "keep": 1,
            "seed": self.seed,
            "duration_s": round(self.duration_s, 4),
            "num_frames": int(self.arrays["joint_pos"].shape[0]),
            "num_replans": len(st),
            "replan_reasons": json.dumps(reasons, separators=(",", ":")),
            "mode_seconds": json.dumps(
                {k: round(v, 3) for k, v in modes.items()}, separators=(",", ":")
            ),
            "max_stitch_joint_gap": round(max((r.old_new_joint_gap for r in st), default=0.0), 4),
            "max_raw_limit_excess": round(float(max(self.raw_limit_excess.max(), 0.0)), 4),
            "planner_phase": self.planner_phase,
            "pickup_delay": self.pickup_delay,
            "command_script": self.script.to_json(),
        }


def clip_seed(base_seed: int, index: int) -> int:
    """Per-clip seed derived from the run seed (independent of the number of workers)."""
    return int(np.random.SeedSequence([base_seed, index]).generate_state(1)[0])


def generate_clip(
    planner: PlannerSession,
    seed: int,
    seconds: float,
    name: str,
    *,
    model: mujoco.MjModel | None = None,
    script_cfg: ScriptConfig | None = None,
    random_planner_seed: bool = False,
    clip_joint_limits: bool = True,
) -> ClipResult:
    """Sample a command script and runtime phase from ``seed``, roll out, and convert.

    ``random_planner_seed`` draws a fresh planner ``random_seed`` per call instead of the runtime's
    constant 1234 (more diverse target poses, but no longer what the robot will see).
    """
    rng = np.random.default_rng(seed)
    script = sample_command_script(rng, seconds, script_cfg)
    phase = int(rng.integers(0, planner.config.planner_ticks))
    delay = int(rng.integers(0, 2))

    def fresh_seed(i: int) -> int:
        return clip_seed(seed, i) % (2**31 - 1)

    seed_fn = fresh_seed if random_planner_seed else None

    ro = rollout(planner, script, seconds, planner_phase=phase, pickup_delay=delay, seed_fn=seed_fn)
    model = model or load_g1_model()
    source = f"sonic_planner:planner_sonic.onnx:seed={seed}"
    arrays = motion_to_npz_arrays(ro.stream, model, source, clip_joint_limits)
    excess = joint_limit_excess(ro.stream.joints, model)
    return ClipResult(name, seed, arrays, script, ro, phase, delay, excess)


def save_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]


# ----------------------------------------------------------------------------- quality stats


def _foot_and_body_geoms(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    feet, other = [], []
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        (feet if "_foot" in name else other).append(g)
    return feet, other


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by wxyz quaternions ``q`` (..., 4)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def lowest_points(model: mujoco.MjModel, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Lowest collision point per frame: z of the whole robot (``lowest``) and, for each foot,
    the world position of its lowest sole point (``{left,right}_foot``, (T, 3)).

    All G1 collision geoms are spheres or capsules, so the lowest point is the lowest end-sphere
    centre minus the radius.
    """
    pos, quat = arrays["body_pos_w"].astype(np.float64), arrays["body_quat_w"].astype(np.float64)
    feet, other = _foot_and_body_geoms(model)

    def geom_lowest(g: int) -> np.ndarray:
        b = model.geom_bodyid[g] - 1  # npz bodies exclude the world body
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            local = model.geom_pos[g][None]
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = _rotate(model.geom_quat[g], np.array([0.0, 0.0, model.geom_size[g, 1]]))
            local = np.stack([model.geom_pos[g] + axis, model.geom_pos[g] - axis])
        else:
            raise ValueError(f"unsupported collision geom type {model.geom_type[g]}")
        pts = pos[:, b, None, :] + _rotate(quat[:, b, None, :], local[None])
        low = pts[np.arange(len(pts)), pts[..., 2].argmin(axis=1)]
        low[:, 2] -= model.geom_size[g, 0]
        return low

    def group_lowest(geoms: list[int]) -> np.ndarray:
        cand = np.stack([geom_lowest(g) for g in geoms], axis=1)  # (T, G, 3)
        return cand[np.arange(cand.shape[0]), cand[..., 2].argmin(axis=1)]

    out = {}
    for side in ("left", "right"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        out[f"{side}_foot"] = group_lowest([g for g in feet if model.geom_bodyid[g] == body])
    out["lowest"] = np.minimum(group_lowest(other)[:, 2], np.minimum(
        out["left_foot"][:, 2], out["right_foot"][:, 2]))  # fmt: skip
    return out


def foot_contact_stats(
    model: mujoco.MjModel, arrays: dict[str, np.ndarray], contact_height: float = 0.0
) -> dict[str, np.ndarray]:
    """Per-frame ground-contact diagnostics (heights in m above the z = 0 floor).

    ``lowest``: lowest collision point of the whole robot (> 0 floating, < 0 penetrating);
    ``{left,right}_foot_z``: lowest sole point of each foot; ``slide``: horizontal speed of the
    lowest sole point of the stance foot (the lower foot, when below ``contact_height``; NaN
    otherwise), from the rigid-body velocity of the ankle roll link. The default threshold is the
    floor itself: planner stance soles sit ~8 mm below it (g1.xml collision capsules), and a swing
    foot only counts once it reaches the floor.
    """
    lp = lowest_points(model, arrays)
    names = list(arrays["body_names"])
    speed, z = {}, {}
    for side in ("left", "right"):
        b = names.index(f"{side}_ankle_roll_link")
        r = lp[f"{side}_foot"] - arrays["body_pos_w"][:, b].astype(np.float64)
        v = arrays["body_lin_vel_w"][:, b].astype(np.float64) + np.cross(
            arrays["body_ang_vel_w"][:, b].astype(np.float64), r
        )
        speed[side] = np.linalg.norm(v[:, :2], axis=1)
        z[side] = lp[f"{side}_foot"][:, 2]
    left_low = z["left"] <= z["right"]
    stance_z = np.where(left_low, z["left"], z["right"])
    stance_v = np.where(left_low, speed["left"], speed["right"])
    return {
        "lowest": lp["lowest"],
        "left_foot_z": z["left"],
        "right_foot_z": z["right"],
        "slide": np.where(stance_z < contact_height, stance_v, np.nan),
    }


def frame_modes(script: CommandScript, num_frames: int) -> np.ndarray:
    """Commanded mode index at each 50 Hz frame."""
    t = np.arange(num_frames) / CONTROL_FPS
    out = np.zeros(num_frames, np.int64)
    for p in script.phases:
        out[(t >= p.t0) & (t < p.t1)] = MODES[p.mode].index
    last = script.phases[-1]
    out[t >= last.t1] = MODES[last.mode].index
    return out


def stitch_continuity(
    arrays: dict[str, np.ndarray], stitches: list[StitchRecord], blend_frames: int = 8
) -> dict[str, np.ndarray]:
    """Per-frame steps of an exported clip, split into stitch windows and the rest.

    A stitch window runs from the frame before a cross-fade starts to one frame after it ends.
    Keys ``{joint_step, joint_acc, root_step, root_rot_step}_{stitch, other}``: max |dq|
    (rad/frame), max |d2q| (rad/frame^2), root displacement (m/frame), root rotation (rad/frame).
    """
    jp = arrays["joint_pos"].astype(np.float64)
    root = arrays["body_pos_w"][:, 0].astype(np.float64)
    quat = arrays["body_quat_w"][:, 0].astype(np.float64)
    dot = np.clip(np.abs(np.sum(quat[1:] * quat[:-1], axis=1)), 0.0, 1.0)
    steps = {
        "joint_step": np.abs(np.diff(jp, axis=0)).max(axis=1),
        "joint_acc": np.pad(np.abs(np.diff(jp, 2, axis=0)).max(axis=1), (0, 1)),
        "root_step": np.linalg.norm(np.diff(root, axis=0), axis=1),
        "root_rot_step": 2.0 * np.arccos(dot),
    }
    mask = np.zeros(jp.shape[0] - 1, bool)
    for s in stitches:
        mask[max(0, s.blend_start - 1) : s.blend_start + blend_frames + 1] = True
    out = {}
    for k, v in steps.items():
        out[f"{k}_stitch"], out[f"{k}_other"] = v[mask], v[~mask]
    return out
