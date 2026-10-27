"""Export the trained world model for the browser demo (``site/assets/policy/world_model.onnx``).

One imagination step, as in training (``ImagStep``: the policy acts, then ``WorldModel.step``):
  ``state``  (1, 71)  [qpos 36 | qvel 35] in MuJoCo order (= the world model's state layout);
  ``action`` (1, 29)  the policy action in training units (the web policy's ``action`` output);
  ``friction`` (1, 1)  foot friction of the scene (1.0 in the evaluation protocol; the demo's friction switch);
  -> ``next_state`` (1, 71), 20 ms later.
The rest of the simulation context xi is the nominal robot: no encoder bias, no torso COM offset. The page runs the policy and this graph in a loop next
to MuJoCo ("reality").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from realmimic.export.sonic_bundle import ONNX_OPSET
from realmimic.export.web_policy import encode_weights, encoding_label
from supermimic.envs.sim_context import XI_DIM
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

INPUTS = ("state", "action", "friction")
OUTPUTS = ("next_state",)
FOOT_FRICTION = 1.0  # site/assets/robot/scene.mjb (realmimic.export.web_scene): parity inputs


class WebWorldModel(nn.Module):
    def __init__(self, wm: WorldModel):
        super().__init__()
        self.wm = wm
        self.register_buffer("xi_rest", torch.zeros(1, XI_DIM - 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor, friction: torch.Tensor) -> torch.Tensor:
        return self.wm.step(state, action, torch.cat([friction, self.xi_rest], dim=-1))


def load_world_model(checkpoint: str | Path) -> tuple[WorldModel, dict]:
    """The checkpoint's world model (eval mode, CPU, float32; no analytic heads)."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = dict(ck["config"]["world_model"])
    spec = {k: tuple(v) if isinstance(v, list) else v for k, v in c.pop("state_spec").items()}
    cfg = WorldModelCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in c.items()}, state_spec=StateSpecCfg(**spec))
    wm = WorldModel(cfg, None, "cpu")
    wm.load_state_dict(ck["wm"])
    return wm.eval().float(), ck


def export_web_world_model(checkpoint: str | Path, out_dir: str | Path, states: np.ndarray, actions: np.ndarray, encoding: str = "fp16") -> dict:
    """Write ``world_model.onnx`` + ``world_model.json``. ``states`` / ``actions`` (N, 71) / (N, 29):
    realistic inputs for the parity check (e.g. from a CPU MuJoCo closed loop)."""
    import onnx
    import onnxruntime as ort

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm, ck = load_world_model(checkpoint)
    web = WebWorldModel(wm).eval()
    path = out_dir / "world_model.onnx"
    mu = torch.full((1, 1), FOOT_FRICTION)
    args = (torch.from_numpy(states[:1].astype(np.float32)), torch.from_numpy(actions[:1].astype(np.float32)), mu)
    torch.onnx.export(web, args, str(path), input_names=list(INPUTS), output_names=list(OUTPUTS), opset_version=ONNX_OPSET, dynamo=False)
    model = onnx.load(str(path))
    decoded = encode_weights(model, encoding)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))

    # Parity: ONNX Runtime vs torch with the decoded weights (graph), and vs the trained weights (encoding).
    exact = WebWorldModel(load_world_model(checkpoint)[0]).eval()
    if decoded:
        sd = web.state_dict()
        web.load_state_dict({**sd, **{k: torch.from_numpy(v) for k, v in decoded.items() if k in sd}})
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    err_graph = err_enc = 0.0
    with torch.no_grad():
        for s, a in zip(states.astype(np.float32), actions.astype(np.float32)):
            x = {"state": s[None], "action": a[None], "friction": mu.numpy()}
            got = sess.run(list(OUTPUTS), x)[0]
            ts, ta = torch.from_numpy(s[None]), torch.from_numpy(a[None])
            err_graph = max(err_graph, float(np.abs(got - web(ts, ta, mu).numpy()).max()))
            err_enc = max(err_enc, float(np.abs(got - exact(ts, ta, mu).numpy()).max()))
    if err_graph > 1e-4:
        raise RuntimeError(f"ONNX Runtime/torch mismatch: max abs next-state error {err_graph}")
    meta = {
        "format": "realmimic-web-world-model/1",
        "inputs": {"state": {"shape": [1, 71], "layout": "qpos 36 | qvel 35 (MuJoCo order)"}, "action": {"shape": [1, 29], "layout": "policy action, training units, joint order"}, "friction": {"shape": [1, 1], "layout": "foot friction"}},
        "outputs": {"next_state": "state 20 ms later, same layout"},
        "dt": wm.spec.dt,
        "frame": wm.spec.frame,
        "xi": {"foot_friction": "input", "encoder_bias": 0.0, "com_offset": 0.0},
        "training_horizon": ck["config"].get("policy_trainer", {}).get("horizon"),
        "weights": encoding_label(encoding),
        "onnx_opset": ONNX_OPSET,
        "checkpoint": Path(checkpoint).name,
        "training_iteration": ck.get("iteration"),
        "parity_ort_vs_torch_max_abs": {"decoded_weights": err_graph, "trained_weights": err_enc},
    }
    (out_dir / "world_model.json").write_text(json.dumps(meta, indent=1))
    return {"onnx": path, "bytes": path.stat().st_size, "err_graph": err_graph, "err_enc": err_enc}
