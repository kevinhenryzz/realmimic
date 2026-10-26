"""Export a trained policy as a SONIC decoder drop-in bundle.

The ONNX graph exposes exactly SONIC's decoder interface — one float32 input ``obs_dict`` [1, 1580]
built from ``layout.SONIC_OBS_ORDER`` and one output ``action`` [1, 29] in IsaacLab order, scaled by
SONIC's compile-time action scale — and adapts it to the training network inside the graph
(``layout.decoder_layout``). Robot constants that cannot be folded into the graph (PD gains of motors
that differ from SONIC's assumption) become runtime flags in ``launch.env``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from realmimic.export.layout import SONIC_OBS_ORDER, DecoderLayout, decoder_layout
from realmimic.obs import sonic
from realmimic.robots import get_robot_spec
from realmimic.robots.g1 import joint_gains

ONNX_OPSET = 17  # conservative: accepted by TensorRT 10.x and ONNX Runtime 1.16 (SONIC's pin)
TRAIN_HISTORY_DIM = 930


class SonicDecoder(nn.Module):
    """``obs_dict`` (B, 1580) -> ``action`` (B, 29) around a training-layout policy."""

    def __init__(self, policy: nn.Module, layout: DecoderLayout):
        super().__init__()
        self.policy = policy
        self.register_buffer("gather", torch.from_numpy(layout.gather))
        self.register_buffer("gain", torch.from_numpy(layout.gain))
        self.register_buffer("action_perm", torch.from_numpy(layout.action_perm))
        self.register_buffer("action_gain", torch.from_numpy(layout.action_gain))

    def forward(self, obs_dict: torch.Tensor) -> torch.Tensor:
        x = torch.index_select(obs_dict, 1, self.gather) * self.gain
        action = self.policy(x[:, :TRAIN_HISTORY_DIM], x[:, TRAIN_HISTORY_DIM:])
        return torch.index_select(action, 1, self.action_perm) * self.action_gain


def load_policy(checkpoint: str | Path) -> tuple[nn.Module, dict]:
    """Rebuild the trained policy (eval mode, CPU, fp32) and return it with the run config."""
    from supermimic.policy.nets import Policy, PolicyCfg

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    env = cfg["env"]
    if env.get("goal_features") != "sonic" or env.get("anchor_obs") != "pelvis_ori10":
        raise ValueError(
            f"{checkpoint} was trained with goal_features={env.get('goal_features')!r}, "
            f"anchor_obs={env.get('anchor_obs')!r}; SONIC export needs 'sonic' / 'pelvis_ori10'"
        )
    pcfg = PolicyCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["policy"].items()})
    policy = Policy(pcfg)
    policy.load_state_dict(ck["policy"])
    return policy.eval().float(), ck


def motor_gain_scale_flags(robot_name: str) -> list[str]:
    """``--motor-kp-scale`` / ``--motor-kd-scale`` arguments mapping SONIC's compile-time gains onto
    the robot's (hardware/MuJoCo motor indices; equal factors grouped)."""
    kp, kd, _ = joint_gains(get_robot_spec(robot_name), list(sonic.JOINT_NAMES_MUJOCO))
    flags = []
    for flag, ours, theirs in (("--motor-kp-scale", kp, sonic.KPS), ("--motor-kd-scale", kd, sonic.KDS)):
        groups: dict[str, list[int]] = {}
        for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            ratio = a / np.float32(b)
            if abs(ratio - 1.0) > 1e-4:
                if not 0.0 < ratio <= 4.0:
                    raise ValueError(f"motor {i}: gain ratio {ratio} outside SONIC's accepted (0, 4]")
                groups.setdefault(f"{ratio:.6f}", []).append(i)
        for factor, motors in groups.items():
            flags += [flag, f"{','.join(map(str, motors))}={factor}"]
    return flags


@dataclass(frozen=True)
class ExportReport:
    out_dir: Path
    max_abs_err: float
    flags: list[str]


def _git_head() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"]) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export_bundle(checkpoint: str | Path, out_dir: str | Path, parity_samples: int = 256) -> ExportReport:
    import onnx
    import onnxruntime as ort

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy, ck = load_policy(checkpoint)
    robot_name = ck["config"]["env"].get("robot", "g1")
    _, _, scale = joint_gains(get_robot_spec(robot_name), list(sonic.JOINT_NAMES_MUJOCO))
    layout = decoder_layout(scale)
    model = SonicDecoder(policy, layout).eval()

    onnx_path = out_dir / "policy.onnx"
    example = torch.zeros(1, layout.sonic_dim)
    torch.onnx.export(
        model, (example,), str(onnx_path), input_names=["obs_dict"], output_names=["action"],
        opset_version=ONNX_OPSET, dynamo=False,
    )  # fmt: skip
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # Parity: ONNX Runtime vs torch on the same inputs (one row at a time, static batch 1).
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(parity_samples, layout.sonic_dim, generator=gen)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(["action"], {"obs_dict": x[i : i + 1].numpy()})[0] for i in range(len(x))])
    max_err = float(np.abs(got - ref).max())
    if max_err > 1e-4:
        raise RuntimeError(f"ONNX Runtime/torch mismatch: max abs error {max_err}")

    (out_dir / "observation_config.yaml").write_text(
        "# Generated by realmimic.export.sonic_bundle; decoder-only policy (no encoder).\n"
        f"# Total dimension: {layout.sonic_dim}\n\nobservations:\n"
        + "".join(f'  - name: "{name}"\n    enabled: true\n' for name in SONIC_OBS_ORDER)
    )
    flags = motor_gain_scale_flags(robot_name)
    (out_dir / "launch.env").write_text(f'REALMIMIC_SONIC_FLAGS="{" ".join(flags)}"\n')
    spec = get_robot_spec(robot_name)
    manifest = {
        "format": "realmimic-sonic-bundle/1",
        "sonic_commit": sonic.SONIC_COMMIT,
        "robot": robot_name,
        "robot_hash": spec.content_hash(),
        "observation_order": list(SONIC_OBS_ORDER),
        "obs_dim": layout.sonic_dim,
        "onnx_opset": ONNX_OPSET,
        "runtime_flags": flags,
        "checkpoint": str(Path(checkpoint).resolve()),
        "training_iteration": ck.get("iteration"),
        "training_seconds": ck.get("train_seconds"),
        "export_git": _git_head(),
        "train_config": ck["config"],
        "parity_ort_vs_torch_max_abs": max_err,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return ExportReport(out_dir=out_dir, max_abs_err=max_err, flags=flags)
