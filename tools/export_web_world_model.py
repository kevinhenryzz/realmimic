"""Export a checkpoint's world model for the browser demo (world_model.onnx + world_model.json) and measure
world-model-vs-reality drift: the policy driving the world model next to the policy driving CPU MuJoCo.

    uv run python tools/export_web_world_model.py logs/<run>/ckpt_XXXXXXX.pt site
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from realmimic.eval.web_closed_loop import DEFAULT_PUSHES, run, stand_window
from realmimic.export.web_world_model import export_web_world_model


def body_error(model: mujoco.MjModel, qa: np.ndarray, qb: np.ndarray) -> float:
    """Mean distance (m) between the robot bodies of two poses (forward kinematics)."""
    da, db = mujoco.MjData(model), mujoco.MjData(model)
    da.qpos[:], db.qpos[:] = qa, qb
    mujoco.mj_kinematics(model, da)
    mujoco.mj_kinematics(model, db)
    return float(np.linalg.norm(da.xpos[1:] - db.xpos[1:], axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("site")
    parser.add_argument("--weights", default="fp16", help="weight encoding (see tools/export_web_policy.py)")
    parser.add_argument("--horizon", type=int, default=32, help="world-model rollout length in ticks for the drift check")
    parser.add_argument("--vectors", help="also write parity inputs/outputs as JSON (JavaScript check, tools/web_validate.mjs)")
    args = parser.parse_args()
    site = Path(args.site)
    meta = json.loads((site / "assets/policy/meta.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(site / "assets/robot/scene.mjb"))
    policy = ort.InferenceSession(str(site / "assets/policy/policy.onnx"), providers=["CPUExecutionProvider"])
    act = lambda x: policy.run(["action", "joint_target"], x)  # noqa: E731

    # Reality: stand + scripted pushes, recording the policy inputs (state history) and actions.
    trace = run(model, act, meta, 10.0, DEFAULT_PUSHES, record_inputs=True)
    states = np.stack([x["state_hist"][0, -1] for x in trace.inputs])
    actions = np.stack(trace.actions)
    report = export_web_world_model(args.checkpoint, site / "assets/policy", states, actions, args.weights)
    print(f"world model: {report['onnx']} ({report['bytes'] / 1e6:.1f} MB)")
    print(f"ONNX Runtime vs torch max abs next-state error: decoded weights {report['err_graph']:.2e}, trained weights {report['err_enc']:.2e}")

    wm = ort.InferenceSession(str(site / "assets/policy/world_model.onnx"), providers=["CPUExecutionProvider"])
    if args.vectors:
        idx = np.linspace(0, len(states) - 1, 64).astype(int)
        mus = [1.0, 0.3]
        rows = [{"state": states[i].tolist(), "action": actions[i].tolist(), "friction": mus[n % 2],
                 "next_state": wm.run(["next_state"], {"state": states[i : i + 1].astype(np.float32), "action": actions[i : i + 1].astype(np.float32), "friction": np.full((1, 1), mus[n % 2], np.float32)})[0][0].tolist()}
                for n, i in enumerate(idx)]  # fmt: skip
        Path(args.vectors).parent.mkdir(parents=True, exist_ok=True)
        Path(args.vectors).write_text(json.dumps({"onnx": "assets/policy/world_model.onnx", "samples": rows}))

    # World model: from each start tick, the policy drives the world model for `horizon` ticks; compare with reality.
    H, K = meta["inputs"]["state_hist"]["shape"][1], meta["inputs"]["ref_window"]["shape"][1]
    q0 = np.asarray(meta["default_joint_pos"])
    pushed = lambda k: any(s <= k * meta["control_dt"] < s + d + 1.0 for s, d, _, _ in DEFAULT_PUSHES)  # noqa: E731
    for label, starts in (("standing", [k for k in range(0, len(states) - args.horizon, 10) if not pushed(k) and not pushed(k + args.horizon)]),
                          ("pushed (the world model does not see the push)", [k for k in range(0, len(states) - args.horizon, 10) if pushed(k + args.horizon // 2)])):
        errs = []
        for k in starts:
            x = trace.inputs[k]
            sh, ah = list(x["state_hist"][0]), list(x["action_hist"][0])
            s = sh[-1]
            for j in range(args.horizon):
                ref = stand_window(q0, meta["stand_root_height"], s[3:7], K)
                a, _ = act({"state_hist": np.stack(sh[-H:])[None].astype(np.float32), "action_hist": np.stack(ah[-H:])[None].astype(np.float32), "ref_window": ref[None].astype(np.float32)})
                s = wm.run(["next_state"], {"state": s[None].astype(np.float32), "action": a.astype(np.float32), "friction": np.ones((1, 1), np.float32)})[0][0]
                sh.append(s)
                ah.append(a[0])
            errs.append(body_error(model, s[:36].astype(np.float64), trace.qpos[k + args.horizon - 1]))
        if errs:
            print(f"world model vs reality after {args.horizon} ticks, {label}: mean body position error {100 * np.mean(errs):.1f} cm (median {100 * np.median(errs):.1f}, max {100 * np.max(errs):.1f}; {len(errs)} starts)")


if __name__ == "__main__":
    main()
