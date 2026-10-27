"""Export a policy for the browser demo (``site/assets/policy``): observation construction inside the graph.

Inputs are raw MuJoCo-order quantities, batch 1, float32:
  ``state_hist``  (1, H, 71)  [qpos 36 | qvel 35] of the last H control ticks, oldest -> newest;
  ``action_hist`` (1, H, 29)  the action applied before each of those states (zeros after a reset);
  ``ref_window``  (1, K, 71)  reference frames t..t+K-1 in MotionLibrary row layout
                              [root_pos 3 | root_quat 4 | joint_pos 29 | root_lin_vel_w 3 | root_ang_vel_w 3 | joint_vel 29].
Outputs: ``action`` (1, 29) in training units (the next ``action_hist`` entry) and ``joint_target``
(1, 29) = action * scale + q0, the position set-points in joint order.

The graph reuses the training functions: ``analytic_observation`` + ``flatten_history_term_major``
(policy history = the env's ``policy`` group without noise and encoder bias), ``sonic_goal_features``
(= ``MotionLibrary.goal_window`` in "sonic" mode) and ``anchor_window_obs`` (= the env's ``anchor_ori10``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from realmimic.export.sonic_bundle import ONNX_OPSET, load_policy
from realmimic.export.web_scene import ROBOT_PREFIX, physics_model, training_scene
from supermimic.envs.mdp.motion_library import sonic_goal_features
from supermimic.envs.sim_context import XI_DIM
from supermimic.kinematics.rotations import anchor_window_obs
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

INPUTS = ("state_hist", "action_hist", "ref_window")
OUTPUTS = ("action", "joint_target")


class WebPolicy(nn.Module):
    def __init__(self, policy: nn.Module, default_joint_pos: torch.Tensor, action_scale: torch.Tensor):
        super().__init__()
        self.policy = policy
        self.register_buffer("q0", default_joint_pos.float())
        self.register_buffer("scale", action_scale.float())
        self.register_buffer("xi", torch.zeros(1, 1, XI_DIM))  # sim context: no encoder bias at deployment

    def forward(self, state_hist: torch.Tensor, action_hist: torch.Tensor, ref_window: torch.Tensor):
        obs = analytic_observation(state_hist, action_hist, self.xi, self.q0)  # (1, H, 93)
        hist = flatten_history_term_major(obs.transpose(0, 1))
        anchor = anchor_window_obs(state_hist[:, -1, 3:7], ref_window[..., 3:7])
        action = self.policy(hist, torch.cat([sonic_goal_features(ref_window), anchor], dim=-1))
        return action, action * self.scale + self.q0


@dataclass(frozen=True)
class RobotConstants:
    joint_names: list[str]
    default_joint_pos: np.ndarray  # (29,) joint order (keyframe "init_state")
    action_scale: np.ndarray  # (29,) joint order (the env's action term)
    stand_root_height: float  # pelvis height in the default pose with the feet on the ground


def robot_constants(robot: str = "g1", self_collision: bool = True) -> RobotConstants:
    import mujoco
    from mjlab.utils.lab_api.string import resolve_matching_names_values

    from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX

    scene, cfg = training_scene(robot, self_collision)
    model = physics_model(scene, cfg)
    names = [model.joint(j).name.removeprefix(ROBOT_PREFIX) for j in range(1, model.njnt)]
    scale = np.zeros(len(names))
    idx, _, val = resolve_matching_names_values(cfg.actions["joint_pos"].scale, names)
    scale[idx] = val
    assert (scale > 0).all()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_kinematics(model, data)
    bottoms = []
    for g in range(model.ngeom):
        if re.fullmatch(FOOT_GEOM_REGEX, model.geom(g).name.removeprefix(ROBOT_PREFIX)):
            assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            radius, half = model.geom_size[g, :2]
            axis_z = data.geom_xmat[g].reshape(3, 3)[2, 2]
            bottoms.append(data.geom_xpos[g, 2] - abs(axis_z) * half - radius)
    return RobotConstants(names, data.qpos[7:].copy(), scale, float(data.qpos[2] - min(bottoms)))


def parse_encoding(encoding: str) -> tuple[int, int]:
    """Weight encoding -> (bits, group): "fp32" (32, 0), "fp16" (16, 0), "int<b>" (b = 2..8, one scale per
    output channel: group 0) or "int<b>g<G>" (one scale per group of G inputs of an output channel)."""
    m = re.fullmatch(r"fp(32|16)|int([2-8])(?:g([1-9]\d*))?", encoding)
    if m is None:
        raise ValueError(f"unknown weight encoding {encoding!r} (fp32, fp16, int<b>, int<b>g<G>)")
    return (int(m[1]), 0) if m[1] else (int(m[2]), int(m[3] or 0))


def quantize(w: np.ndarray, bits: int, group: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric quantization of an (out, in) weight, per output channel (``group`` 0) or per group of
    ``group`` inputs (zero-padded to a multiple): int8 codes (out, in_pad) in [-(2^(b-1)-1), 2^(b-1)-1] and
    float32 scales (out, groups, 1). Each scale minimises the squared error over clipping ratios 0.5..1 of
    the absolute maximum."""
    qmax = 2 ** (bits - 1) - 1
    g = group or w.shape[1]
    wg = np.pad(w.astype(np.float32), ((0, 0), (0, -w.shape[1] % g))).reshape(w.shape[0], -1, g)
    amax = np.maximum(np.abs(wg).max(-1, keepdims=True), np.float32(1e-12))
    best_q, best_s, best_err = None, None, None
    for ratio in np.linspace(1.0, 0.5, 11, dtype=np.float32):
        s = amax * ratio / np.float32(qmax)
        q = np.clip(np.round(wg / s), -qmax, qmax)
        err = np.square(q * s - wg).sum(-1, keepdims=True)
        if best_err is None:
            best_q, best_s, best_err = q, s, err
        else:
            better = err < best_err
            best_q, best_s, best_err = np.where(better, q, best_q), np.where(better, s, best_s), np.minimum(err, best_err)
    return best_q.astype(np.int8).reshape(w.shape[0], -1), best_s.astype(np.float32)


def dequantize(q: np.ndarray, s: np.ndarray, cols: int) -> np.ndarray:
    """The graph's weight decoding (float32): codes * scale per group, padding dropped."""
    return (q.astype(np.float32).reshape(*s.shape[:2], -1) * s).reshape(q.shape[0], -1)[:, :cols]


def encode_weights(model, encoding: str = "fp16", min_size: int = 1024) -> dict[str, np.ndarray]:
    """Re-encode large float initializers in place; compute stays float32. fp16: float16 behind a Cast.
    int: 2-D (Gemm) weights as int8 codes decoded by Cast -> Mul(scale) (with Reshape / Slice for groups),
    other large initializers float16. Decoding uses only foldable ops (no DequantizeLinear, which ONNX
    Runtime leaves in the graph), so the session holds float32 weights. Returns the decoded value of each
    re-encoded initializer (float32), i.e. what the graph computes with."""
    from onnx import TensorProto, helper, numpy_helper

    bits, group = parse_encoding(encoding)
    decoded, nodes, extra = {}, [], []

    def const(name, value):
        extra.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    for init in model.graph.initializer:
        w = numpy_helper.to_array(init)
        if bits == 32 or init.data_type != TensorProto.FLOAT or w.size < min_size:
            continue
        name = init.name
        if bits == 16 or w.ndim != 2:
            init.CopyFrom(numpy_helper.from_array(w.astype(np.float16), name + "_fp16"))
            nodes.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT))
            decoded[name] = w.astype(np.float16).astype(np.float32)
            continue
        q, s = quantize(w, bits, group)
        decoded[name] = dequantize(q, s, w.shape[1])
        init.CopyFrom(numpy_helper.from_array(q, name + "_q"))
        nodes.append(helper.make_node("Cast", [name + "_q"], [name + "_qf"], to=TensorProto.FLOAT))
        if s.shape[1] == 1:  # per output channel
            nodes.append(helper.make_node("Mul", [name + "_qf", const(name + "_scale", s[:, 0])], [name]))
            continue
        nodes.append(helper.make_node("Reshape", [name + "_qf", const(name + "_gshape", np.array(q.shape[:1] + s.shape[1:2] + (-1,)))], [name + "_qg"]))
        nodes.append(helper.make_node("Mul", [name + "_qg", const(name + "_scale", s)], [name + "_wg"]))
        cols = name + "_w" if q.shape[1] != w.shape[1] else name
        nodes.append(helper.make_node("Reshape", [name + "_wg", const(name + "_shape", np.array(q.shape))], [cols]))
        if cols != name:
            nodes.append(helper.make_node("Slice", [cols, const(name + "_s0", np.array([0])), const(name + "_s1", np.array([w.shape[1]])), const(name + "_ax", np.array([1]))], [name]))
    model.graph.initializer.extend(extra)
    graph_nodes = nodes + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(graph_nodes)
    return decoded


def encoding_label(encoding: str) -> str:
    bits, group = parse_encoding(encoding)
    if bits == 32:
        return "float32"
    if bits == 16:
        return "float16, float32 compute"
    scale = f"per group of {group} inputs" if group else "per output channel"
    return f"{encoding}: {bits}-bit symmetric codes (int8 storage), float32 scale {scale}, other large tensors float16; float32 compute"


def sample_inputs(c: RobotConstants, n: int, history: int, frames: int, seed: int = 0) -> list[dict[str, np.ndarray]]:
    """Random inputs around the default pose (graph parity checks)."""
    rng = np.random.default_rng(seed)

    def rows(k: int) -> np.ndarray:
        r = np.zeros((1, k, 71))
        r[..., 2] = c.stand_root_height + 0.05 * rng.standard_normal((1, k))
        q = np.concatenate([np.ones((1, k, 1)), 0.3 * rng.standard_normal((1, k, 3))], axis=-1)
        r[..., 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        r[..., 7:36] = c.default_joint_pos + 0.3 * rng.standard_normal((1, k, 29))
        r[..., 36:] = rng.standard_normal((1, k, 35))
        return r.astype(np.float32)

    return [
        {"state_hist": rows(history), "action_hist": rng.standard_normal((1, history, 29)).astype(np.float32), "ref_window": rows(frames)}
        for _ in range(n)
    ]


def run_onnx(path: str | Path, inputs: list[dict[str, np.ndarray]]) -> list[list[np.ndarray]]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return [sess.run(list(OUTPUTS), x) for x in inputs]


def build(checkpoint: str | Path) -> tuple[WebPolicy, RobotConstants, dict]:
    policy, ck = load_policy(checkpoint)
    env = ck["config"]["env"]
    c = robot_constants(env.get("robot", "g1"), env.get("self_collision", True))
    web = WebPolicy(policy, torch.as_tensor(c.default_joint_pos), torch.as_tensor(c.action_scale)).eval()
    return web, c, ck


def export_onnx(web: WebPolicy, c: RobotConstants, path: str | Path, encoding: str = "fp16", batch: int = 1) -> dict[str, np.ndarray]:
    """Write the graph with ``encoding`` weights; returns the decoded weights (``encode_weights``).
    ``batch`` > 1: the same graph and weights for a fixed batch (offline evaluation only)."""
    import onnx

    x = sample_inputs(c, 1, web.policy.cfg.history, web.policy.cfg.future_frames)[0]
    args = tuple(torch.from_numpy(np.repeat(x[k], batch, axis=0)) for k in INPUTS)
    torch.onnx.export(web, args, str(path), input_names=list(INPUTS), output_names=list(OUTPUTS), opset_version=ONNX_OPSET, dynamo=False)
    model = onnx.load(str(path))
    decoded = encode_weights(model, encoding)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return decoded


def with_weights(web: WebPolicy, decoded: dict[str, np.ndarray]) -> WebPolicy:
    """A copy of ``web`` holding the decoded weights (initializer names = state-dict keys)."""
    import copy

    out = copy.deepcopy(web)
    state = out.state_dict()
    unknown = set(decoded) - set(state)
    if unknown:
        raise KeyError(f"encoded initializers without a state-dict entry: {sorted(unknown)}")
    out.load_state_dict({**state, **{k: torch.from_numpy(v) for k, v in decoded.items()}})
    return out


def export_web_policy(checkpoint: str | Path, out_dir: str | Path, encoding: str = "fp16", parity_samples: int = 64, vectors: str | Path | None = None) -> dict:
    """Write ``policy.onnx`` + ``meta.json`` and return the ORT-vs-torch parity: against torch with the
    decoded weights (graph check, float32 tolerance) and against the trained weights (encoding error).
    ``vectors``: also write the parity inputs and ONNX Runtime outputs as JSON (JavaScript parity check)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    web, c, ck = build(checkpoint)
    cfg = web.policy.cfg
    onnx_path = out_dir / "policy.onnx"
    decoded = export_onnx(web, c, onnx_path, encoding)
    inputs = sample_inputs(c, parity_samples, cfg.history, cfg.future_frames, seed=1)
    got = run_onnx(onnx_path, inputs)

    def max_err(module):
        with torch.no_grad():
            ref = [[t.numpy() for t in module(*(torch.from_numpy(x[k]) for k in INPUTS))] for x in inputs]
        return {name: float(max(np.abs(g[i] - r[i]).max() for g, r in zip(got, ref))) for i, name in enumerate(OUTPUTS)}

    err = max_err(with_weights(web, decoded))
    if err["action"] > 1e-4:
        raise RuntimeError(f"ONNX Runtime/torch mismatch: max abs action error {err['action']}")
    enc_err = max_err(web) if decoded else err
    env = ck["config"]["env"]
    meta = {
        "format": "realmimic-web-policy/1",
        "inputs": {
            "state_hist": {"shape": [1, cfg.history, 71], "layout": "qpos 36 | qvel 35 (MuJoCo order), oldest -> newest"},
            "action_hist": {"shape": [1, cfg.history, 29], "layout": "action applied before each state of state_hist (zeros after a reset)"},
            "ref_window": {"shape": [1, cfg.future_frames, 71], "layout": "root_pos 3 | root_quat 4 (wxyz) | joint_pos 29 | root_lin_vel_w 3 | root_ang_vel_w 3 | joint_vel 29; frames t..t+K-1"},
        },
        "outputs": {"action": "training units, joint order", "joint_target": "action * action_scale + default_joint_pos, joint order"},
        "history_reset": "every slot holds the reset state and a zero action (mjlab observation history)",
        "joint_names": c.joint_names,
        "default_joint_pos": c.default_joint_pos.tolist(),
        "action_scale": c.action_scale.tolist(),
        "stand_root_height": c.stand_root_height,
        "control_dt": 0.02,
        "physics_dt": 0.005,
        "decimation": 4,
        "termination": {"height": 0.25, "tilt": 0.8},
        "robot": env.get("robot", "g1"),
        "weights": encoding_label(encoding),
        "weight_encoding": encoding,
        "onnx_opset": ONNX_OPSET,
        "checkpoint": Path(checkpoint).name,
        "training_iteration": ck.get("iteration"),
        "goal_features": env.get("goal_features"),
        "anchor_obs": env.get("anchor_obs"),
        "freeze_clip_end": env.get("freeze_clip_end", False),  # reference past a clip's end: last frame, zero velocities
        "parity_ort_vs_torch_max_abs": err,  # torch with the decoded weights
        "encoding_vs_trained_weights_max_abs": enc_err,  # ONNX Runtime vs torch with the trained weights
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    if vectors is not None:
        rows = [{**{k: x[k].ravel().tolist() for k in INPUTS}, **{k: g[i].ravel().tolist() for i, k in enumerate(OUTPUTS)}} for x, g in zip(inputs, got)]
        Path(vectors).write_text(json.dumps({"onnx": str(onnx_path), "samples": rows}))
    return {"onnx": onnx_path, "bytes": onnx_path.stat().st_size, "max_abs_err": err, "encoding_max_abs_err": enc_err}
