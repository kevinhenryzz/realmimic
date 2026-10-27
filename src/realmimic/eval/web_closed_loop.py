"""CPU MuJoCo closed loop of the browser demo (the reference for ``tools/web_validate.mjs``).

Same loop as ``site/demo``: reset to the ``init_state`` keyframe with the history filled by the
reset state and zero actions; each 20 ms tick builds the reference window, runs the policy, sets the
actuators through ``actuator_trnid`` and takes ``decimation`` physics steps with the scripted
external forces of that tick in ``xfrc_applied``.

The "stand" reference is the default pose at ``stand_root_height`` with zero velocities and the
pelvis heading of the robot (yaw only), so the robot is asked to stand upright wherever it faces.

``track_clips`` plays clips with the same loop under the benchmark protocol (``supermimic.eval.benchmark``):
reset to the clip's first frame (joints clipped to the soft limits, no noise), target frame 1 at the
first tick, one tick per frame, the clip's last frame held (``freeze_clip_end``) for the final tick,
stop at the training termination.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import mujoco
import numpy as np

# (start s, duration s, body, force N (world frame))
DEFAULT_PUSHES = (
    (2.0, 0.1, "torso_link", (150.0, 0.0, 0.0)),
    (4.0, 0.1, "torso_link", (0.0, 120.0, 0.0)),
    (6.0, 0.1, "torso_link", (-150.0, 0.0, 0.0)),
    (8.0, 0.1, "torso_link", (0.0, -120.0, 0.0)),
)


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Heading (yaw-only) quaternion of a wxyz quaternion, as mjlab's ``yaw_quat``."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def stand_window(default_joint_pos: np.ndarray, root_height: float, root_quat: np.ndarray, frames: int) -> np.ndarray:
    """(frames, 71) reference rows: the default pose held still, heading of ``root_quat``."""
    row = np.zeros(71)
    row[2] = root_height
    row[3:7] = yaw_quat(root_quat)
    row[7:36] = default_joint_pos
    return np.repeat(row[None], frames, axis=0)


def fallen(qpos: np.ndarray, ref_row: np.ndarray, height: float = 0.25, tilt: float = 0.8) -> bool:
    """Training termination (pelvis height / tilt vs the reference)."""

    def gz(q):  # z of world gravity (0, 0, -1) in the body frame
        _, x, y, _ = q / np.linalg.norm(q)
        return -(1 - 2 * (x * x + y * y))

    return abs(ref_row[2] - qpos[2]) > height or abs(gz(ref_row[3:7]) - gz(qpos[3:7])) > tilt


@dataclass
class Trace:
    t: list[float] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    inputs: list[dict[str, np.ndarray]] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    fall_tick: int | None = None


def run(
    model: mujoco.MjModel,
    act: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
    meta: dict,
    seconds: float = 10.0,
    pushes=DEFAULT_PUSHES,
    record_inputs: bool = False,
) -> Trace:
    """``act(inputs) -> (action, joint_target)`` with the web-export inputs (float32, batch 1)."""
    H = meta["inputs"]["state_hist"]["shape"][1]
    K = meta["inputs"]["ref_window"]["shape"][1]
    q0 = np.asarray(meta["default_joint_pos"])
    dec = int(meta["decimation"])
    assert abs(model.opt.timestep * dec - meta["control_dt"]) < 1e-12
    act_jnt = model.actuator_trnid[:, 0] - 1  # actuator i drives joint act_jnt[i] (free joint = 0)
    bodies = {b: model.body("robot/" + b).id for _, _, b, _ in pushes}
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    state = np.concatenate([data.qpos, data.qvel])
    states, actions = [state] * H, [np.zeros(29)] * H
    out = Trace()
    for k in range(round(seconds / meta["control_dt"])):
        t = k * meta["control_dt"]
        ref = stand_window(q0, meta["stand_root_height"], data.qpos[3:7], K)
        x = {
            "state_hist": np.stack(states[-H:])[None].astype(np.float32),
            "action_hist": np.stack(actions[-H:])[None].astype(np.float32),
            "ref_window": ref[None].astype(np.float32),
        }
        a, target = act(x)
        a, target = np.asarray(a, dtype=np.float64).reshape(29), np.asarray(target, dtype=np.float64).reshape(29)
        data.ctrl[:] = target[act_jnt]
        data.xfrc_applied[:] = 0.0
        for start, dur, body, force in pushes:
            if start <= t < start + dur - 1e-9:
                data.xfrc_applied[bodies[body], :3] += force
        for _ in range(dec):
            mujoco.mj_step(model, data)
        states.append(np.concatenate([data.qpos, data.qvel]))
        actions.append(a)
        out.t.append(t + meta["control_dt"])
        out.qpos.append(data.qpos.copy())
        out.actions.append(a)
        if record_inputs:
            out.inputs.append(x)
        if out.fall_tick is None and fallen(data.qpos, ref[0]):
            out.fall_tick = k
    return out


SOFT_JOINT_LIMIT_FACTOR = 0.9  # mjlab's G1 articulation: reset joint positions are clipped to these limits


@dataclass
class ClipRef:
    windows: np.ndarray  # (T, K, 71) reference window of each tick (row 0: that tick's target frame)
    qpos0: np.ndarray  # (36,) reset state: the clip's first frame
    qvel0: np.ndarray  # (35,)


def clip_reference(npz, default_joint_pos: np.ndarray, frames: int, freeze_end: bool) -> ClipRef:
    """A clip's windows as the env builds them (``MotionLibrary.goal_window`` rows): tick k targets frame
    min(k + 1, T - 1), frames past the end held still if ``freeze_end``."""
    import torch

    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    lib = MotionLibrary([npz], TRACKED_BODIES, default_joint_pos, device="cpu", freeze_end=freeze_end)
    n = lib.num_frames
    idx, frozen = lib._window_index(torch.clamp(torch.arange(n) + 1, max=n - 1), frames)
    qpos, qvel = lib.ref_qpos_qvel(torch.zeros(1, dtype=torch.long))
    return ClipRef(lib.frozen_ref71(lib._ref[idx], frozen).double().numpy(), qpos[0].double().numpy(), qvel[0].double().numpy())


@dataclass
class TrackResult:
    frames: int
    steps: int
    fail_step: int  # step of the termination (1-based, as the benchmark), -1: completed
    joint_err: float  # mean |q - q_ref| over joints and steps [rad]
    pelvis_err: float  # mean pelvis position error over steps [m]


def track_clips(
    model: mujoco.MjModel,
    act: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]],
    clips: Iterable[tuple[object, Callable[[], ClipRef], float, int]],
    meta: dict,
    batch: int = 1,
) -> Iterator[tuple[object, TrackResult]]:
    """Play clips in lockstep, ``batch`` at a time (a slot takes the next clip when its clip ends); ``act``
    maps web-export inputs of that batch to (action, joint_target), (batch, 29) each. A clip is (key,
    reference loader, joint_noise, seed); joint_noise: std of a reset joint-position perturbation [rad]
    (run-to-run noise estimate). Yields (key, result) as clips end."""
    H = meta["inputs"]["state_hist"]["shape"][1]
    K = meta["inputs"]["ref_window"]["shape"][1]
    dec = int(meta["decimation"])
    term = meta["termination"]
    act_jnt = model.actuator_trnid[:, 0] - 1
    lo, hi = model.jnt_range[1:].T
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_JOINT_LIMIT_FACTOR
    queue = iter(clips)

    def start() -> dict | None:
        item = next(queue, None)
        if item is None:
            return None
        key, load, noise, seed = item
        ref = load()
        data = mujoco.MjData(model)
        data.qpos[:] = ref.qpos0
        data.qpos[7:] = np.clip(ref.qpos0[7:], mid - half, mid + half)
        if noise > 0:
            data.qpos[7:] += np.random.default_rng(seed).normal(0.0, noise, 29)
        data.qvel[:] = ref.qvel0
        mujoco.mj_forward(model, data)
        state = np.concatenate([data.qpos, data.qvel])
        return {"key": key, "ref": ref, "data": data, "states": np.repeat(state[None], H, 0), "actions": np.zeros((H, 29)), "k": 0, "jerr": 0.0, "perr": 0.0}

    x = {"state_hist": np.zeros((batch, H, 71), np.float32), "action_hist": np.zeros((batch, H, 29), np.float32), "ref_window": np.zeros((batch, K, 71), np.float32)}
    slots = [start() for _ in range(batch)]
    while any(slots):
        live = [i for i, s in enumerate(slots) if s]
        for i in range(batch):  # idle slots repeat a live one
            s = slots[i] or slots[live[0]]
            x["state_hist"][i], x["action_hist"][i], x["ref_window"][i] = s["states"], s["actions"], s["ref"].windows[s["k"]]
        a, target = (np.asarray(y, dtype=np.float64).reshape(batch, 29) for y in act(x))
        for i in live:
            s, d = slots[i], slots[i]["data"]
            win = s["ref"].windows[s["k"]]
            d.ctrl[:] = target[i, act_jnt]
            for _ in range(dec):
                mujoco.mj_step(model, d)
            s["states"] = np.concatenate([s["states"][1:], np.concatenate([d.qpos, d.qvel])[None]])
            s["actions"] = np.concatenate([s["actions"][1:], a[i][None]])
            s["k"] += 1
            s["jerr"] += np.abs(d.qpos[7:] - win[0, 7:36]).mean()
            s["perr"] += np.linalg.norm(d.qpos[:3] - win[0, :3])
            failed = fallen(d.qpos, win[0], term["height"], term["tilt"])
            if failed or s["k"] == len(s["ref"].windows):
                n = s["k"]
                yield s["key"], TrackResult(len(s["ref"].windows), n, n if failed else -1, s["jerr"] / n, s["perr"] / n)
                slots[i] = start()


