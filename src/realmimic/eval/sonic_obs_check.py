"""Contract check: the runtime's logged policy inputs vs ``realmimic.obs.sonic`` on the same states.

The runtime writes its observation buffer once per CONTROL tick (``--policy-input-logfile``, default
stream precision: 6 significant digits) and, with ``--enable-csv-logs``, the state it built it from
(StateLogger: pelvis IMU quaternion and gyro, joint q / dq in hardware order, last action in
IsaacLab order; 9 decimals). Row ``t`` of both is the ``t``-th CONTROL tick, so the history terms
are fully determined by rows ``0..t``: this module recomputes them with our reference
implementation and compares slice by slice. For reference terms only frame 0 of the window is
checked, against ``--target-motion-logfile`` (the frame shown that tick); later frames and the
anchor orientation depend on the planner / heading state, which no log records.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from realmimic.obs import sonic

HISTORY_FRAMES = 10
RTOL = 1e-5  # 6 significant digits in the policy-input log
ATOL = 1e-8  # absolute floor for values near zero (StateLogger values carry 9 decimals)


def observation_order(obs_config: str | Path) -> tuple[list[str], dict[str, int]]:
    """Enabled observation names in file order, and dims of terms not in ``sonic.TERM_DIMS``
    (``token_state`` = the encoder dimension)."""
    cfg = yaml.safe_load(Path(obs_config).read_text())
    order = [o["name"] for o in cfg["observations"] if o.get("enabled") is True]
    extra = {}
    if "token_state" in order:
        extra["token_state"] = int(cfg["encoder"]["dimension"])
    return order, extra


def history_windows(
    x: np.ndarray, frames: int = HISTORY_FRAMES
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per tick t the rows t-frames+1..t (clamped at 0) as (T, frames, D), and the number of real
    ticks so far (t+1), for ``sonic.his_*``'s ``num_valid`` padding."""
    t = np.arange(len(x))
    idx = np.clip(t[:, None] + np.arange(-frames + 1, 1)[None, :], 0, None)
    return torch.from_numpy(x[idx]), torch.from_numpy(t + 1)


def expected_terms(
    q: np.ndarray,
    dq: np.ndarray,
    base_quat: np.ndarray,
    base_ang_vel: np.ndarray,
    last_action: np.ndarray,
    target: np.ndarray | None,
) -> dict[str, tuple[torch.Tensor, slice]]:
    """Recomputable terms -> (expected values (T, n), the slice of the term they cover).

    ``q``/``dq`` hardware order (absolute q), ``last_action`` IsaacLab order (raw network output of
    the previous tick), ``target`` the target-motion log rows (pelvis pos, quat, 29 joint pos)."""
    terms: dict[str, tuple[torch.Tensor, slice]] = {}
    his: dict[str, tuple[Callable[..., torch.Tensor], np.ndarray]] = {
        "base_angular_velocity": (sonic.his_base_angular_velocity, base_ang_vel),
        "body_joint_positions": (sonic.his_body_joint_positions, q),
        "body_joint_velocities": (sonic.his_body_joint_velocities, dq),
        "last_actions": (sonic.his_last_actions, last_action[:, list(sonic.ISAACLAB_TO_MUJOCO)]),
        "gravity_dir": (sonic.his_gravity_dir, base_quat),
    }
    for stem, (fn, x) in his.items():
        window, valid = history_windows(x)
        for frames in (4, 10):
            out = fn(window, valid, num_frames=frames)
            terms[f"his_{stem}_{frames}frame_step1"] = (out, slice(0, out.shape[-1]))
    if target is not None:
        ref_q = torch.from_numpy(target[:, None, 7:36])
        jp = sonic.motion_joint_positions(ref_q)
        z = torch.from_numpy(target[:, 2:3])
        for name in sonic.TERM_DIMS:
            if name.startswith("motion_joint_positions") and sonic.TERM_DIMS[name] % 29 == 0:
                terms[name] = (jp, slice(0, 29))
            elif name.startswith("motion_root_z_position"):
                terms[name] = (z, slice(0, 1))
    return terms


def compare(
    inputs: np.ndarray,
    order: Sequence[str],
    expected: dict[str, tuple[torch.Tensor, slice]],
    extra_dims: dict[str, int] | None = None,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> dict[str, dict[str, Any]]:
    """Per term in ``order``: max abs / relative error and the fraction of elements out of
    tolerance (``|ours - logged| > rtol |ours| + atol``), or why it was not checked."""
    dims = {**sonic.TERM_DIMS, **(extra_dims or {})}
    total = sum(dims[name] for name in order)
    if inputs.shape[1] != total:
        raise ValueError(f"log rows have {inputs.shape[1]} values, observation config {total}")
    n = len(inputs)
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in order:
        logged = inputs[:, offset : offset + dims[name]]
        offset += dims[name]
        if name not in expected:
            out[name] = {"checked": False}
            continue
        values, part = expected[name]
        ours = values[:n].double().numpy()
        got = logged[: len(ours), part]
        err = np.abs(ours - got)
        bad = ~(err <= rtol * np.abs(ours) + atol)  # NaN on either side counts as a mismatch
        out[name] = {
            "checked": "full"
            if part == slice(0, dims[name])
            else f"elements {part.start}:{part.stop}",
            "ticks": len(ours),
            "max_abs_err": float(err.max()),
            "max_rel_err": float((err / np.maximum(np.abs(ours), atol)).max()),
            "frac_out_of_tol": float(bad.mean()),
            "first_bad_tick": int(np.flatnonzero(bad.any(1))[0]) if bad.any() else None,
        }
    return out


# ---------------------------------------------------------------------------- release policy


def runtime_frames(playing: np.ndarray, timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame shown per CONTROL tick of a reference clip, and the ticks that re-initialise the
    heading, from the logged ``play`` flag (StateLogger ``motion_playing``): the frame advances
    after each playing tick and returns to 0 (paused) at the clip end, which re-inits the heading
    on the next tick (cpp:3409-3427); the first CONTROL tick re-inits too (cpp:255)."""
    frame = np.zeros(len(playing), dtype=np.int64)
    reinit = np.zeros(len(playing), dtype=bool)
    reinit[0] = True
    for t in range(len(playing) - 1):
        frame[t + 1] = frame[t] + int(playing[t] > 0.5)
        if frame[t + 1] >= timesteps:
            frame[t + 1] = 0
            reinit[t + 1] = True
    return frame, reinit


def runtime_alignment(
    base_quat: np.ndarray, ref_root_quat: np.ndarray, frame: np.ndarray, reinit: np.ndarray
) -> np.ndarray:
    """Heading alignment in effect per tick (UpdateHeadingState, cpp:559-585, then
    ComputeApplyDeltaHeading, cpp:593-606; no operator yaw offset), (T, 4)."""
    init_base = np.zeros_like(base_quat)
    init_ref = np.zeros_like(base_quat)
    for t in range(len(frame)):
        init_base[t] = base_quat[t] if reinit[t] else init_base[t - 1]
        init_ref[t] = ref_root_quat[frame[t]] if reinit[t] or frame[t] == 0 else init_ref[t - 1]
    return sonic.heading_alignment(torch.from_numpy(init_base), torch.from_numpy(init_ref)).numpy()


def _error_stats(ours: np.ndarray, logged: np.ndarray, rtol: float, atol: float) -> dict[str, Any]:
    err = np.abs(ours - logged)
    bad = ~(err <= rtol * np.abs(logged) + atol)
    return {
        "ticks": len(ours),
        "max_abs_err": float(err.max()),
        "max_rel_err": float((err / np.maximum(np.abs(logged), atol)).max()),
        "frac_out_of_tol": float(bad.mean()),
        "ticks_out_of_tol": int(bad.any(1).sum()),
        "first_bad_tick": int(np.flatnonzero(bad.any(1))[0]) if bad.any() else None,
    }


def bundle_policy_check(
    onnx_path: str | Path, inputs: np.ndarray, actions: np.ndarray, rtol: float = 1e-3, atol: float = 1e-3
) -> dict[str, Any]:
    """Our exported decoder (ONNX Runtime, CPU) on the runtime's logged inputs against the actions
    its inference engine produced: row ``t`` of ``--policy-input-logfile`` against StateLogger's
    ``action`` of tick ``t + 1`` (the previous tick's raw network output, IsaacLab order). A mismatch
    means the deployed engine (TensorRT build, precision, version) does not compute our network."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    ours = np.concatenate([session.run(None, {name: x[None].astype(np.float32)})[0] for x in inputs[:-1]])
    return {"action": _error_stats(ours, actions[1 : len(inputs)], rtol, atol)}


def release_policy_check(
    policy: Any,
    q: np.ndarray,
    dq: np.ndarray,
    base_quat: np.ndarray,
    base_ang_vel: np.ndarray,
    last_action: np.ndarray,
    token: np.ndarray,
    playing: np.ndarray,
    inputs: np.ndarray,
    target: np.ndarray,
    clip: dict[str, np.ndarray],
    action_rtol: float = 1e-4,
    action_atol: float = 1e-5,
) -> dict[str, dict[str, Any]]:
    """SONIC's release policy (``realmimic.eval.sonic_release_policy.SonicReleasePolicy``) rerun on
    the runtime's logged states of a clip session, against what the runtime logged.

    Recomputed per tick from StateLogger rows and the clip CSVs (``clip``: ``joint_pos``,
    ``joint_vel`` (T, 29) IsaacLab order, ``root_quat`` (T, 4) = body 0 of ``body_quat.csv``): the
    shown frame and heading state, the encoder input, the token (vs StateLogger ``token_state``
    and the logged decoder input), the decoder input's history terms, and the action (vs the
    ``last_action`` logged one tick later; the last tick has none). The action is checked both
    from our token and from the logged one, to separate encoder from decoder differences."""
    from realmimic.eval.sonic_release_policy import (
        WINDOW_FRAMES,
        WINDOW_STEP,
        decoder_terms,
        encoder_terms,
    )

    n = len(q)
    timesteps = len(clip["joint_pos"])
    frame, reinit = runtime_frames(playing, timesteps)
    out: dict[str, dict[str, Any]] = {}
    shown = clip["joint_pos"][frame][:, list(sonic.ISAACLAB_TO_MUJOCO)]
    out["frame"] = {
        "reinit_ticks": np.flatnonzero(reinit).tolist(),
        **_error_stats(shown, target[:n, 7:36], RTOL, ATOL),
    }

    t64 = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    play = t64(playing > 0.5)
    alignment = runtime_alignment(base_quat, clip["root_quat"], frame, reinit)
    idx = sonic.future_frame_indices(
        t64(frame), torch.full((n,), timesteps - 1), WINDOW_FRAMES, WINDOW_STEP, play
    ).numpy()
    terms = encoder_terms(
        t64(base_quat),
        t64(clip["joint_pos"][idx][..., list(sonic.ISAACLAB_TO_MUJOCO)]),
        t64(clip["joint_vel"][idx][..., list(sonic.ISAACLAB_TO_MUJOCO)]),
        t64(clip["root_quat"][idx]),
        t64(alignment),
        play,
    )
    nets = policy.nets.cpu()
    ours_token = nets.encode(policy.encoder_input(terms).float()).double().numpy()
    out["token_state"] = _error_stats(ours_token, token[:n], 0.0, 1e-6)
    tok = policy.config.token_dim
    out["token_state_input"] = _error_stats(ours_token, inputs[:n, :tok], 0.0, 1e-6)

    his = {
        name: history_windows(x)
        for name, x in (
            ("ang", base_ang_vel), ("q", q), ("dq", dq),
            ("act", last_action[:, list(sonic.ISAACLAB_TO_MUJOCO)]), ("quat", base_quat),
        )
    }  # fmt: skip
    valid = his["q"][1]
    actions = {}
    for source, tokens in (("action", ours_token), ("action_from_logged_token", token[:n])):
        dec = policy.decoder_input(
            decoder_terms(
                t64(tokens),
                *(his[k][0] for k in ("ang", "q", "dq", "act", "quat")),
                num_valid=valid,
            )
        )
        if source == "action":
            out["history_terms_input"] = _error_stats(
                dec[:, tok:].numpy(), inputs[:n, tok:], RTOL, ATOL
            )
        actions[source] = nets.decode(dec.float()).double().numpy()
        out[source] = _error_stats(actions[source][:-1], last_action[1:n], action_rtol, action_atol)
    return out
