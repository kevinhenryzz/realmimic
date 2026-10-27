"""Run the site's policy in CPU MuJoCo (stand + scripted pushes) and write the trace for tools/web_validate.mjs.

    uv run python tools/web_closed_loop.py site exports/web/trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from realmimic.eval.web_closed_loop import DEFAULT_PUSHES, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site")
    parser.add_argument("trace")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--push-scale", type=float, default=1.0)
    args = parser.parse_args()
    site = Path(args.site)
    meta = json.loads((site / "assets/policy/meta.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(site / "assets/robot/scene.mjb"))
    sess = ort.InferenceSession(str(site / "assets/policy/policy.onnx"), providers=["CPUExecutionProvider"])
    pushes = [(s, d, b, tuple(args.push_scale * f for f in force)) for s, d, b, force in DEFAULT_PUSHES]
    trace = run(model, lambda x: sess.run(["action", "joint_target"], x), meta, args.seconds, pushes)
    q = np.stack(trace.qpos)
    out = {
        "seconds": args.seconds,
        "pushes": [{"start": s, "duration": d, "body": b, "force": list(f)} for s, d, b, f in pushes],
        "fall_tick": trace.fall_tick,
        "t": trace.t,
        "pelvis_z": q[:, 2].tolist(),
        "qpos": q.tolist(),
    }
    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace).write_text(json.dumps(out))
    status = "no fall" if trace.fall_tick is None else f"FELL at t={trace.t[trace.fall_tick]:.2f} s"
    print(f"{status}; pelvis z min {q[:, 2].min():.3f} max {q[:, 2].max():.3f} final {q[-1, 2]:.3f} m; final xy {q[-1, :2].round(3).tolist()}")


if __name__ == "__main__":
    main()
