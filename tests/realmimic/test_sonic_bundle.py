"""Exporting a (randomly initialised) SONIC-mode policy yields a valid bundle with ORT/torch parity."""

from __future__ import annotations

import json
from dataclasses import asdict

import onnx
import torch

from realmimic.export.sonic_bundle import ONNX_OPSET, export_bundle
from supermimic.config import TrainConfig
from supermimic.goal_spec import GOAL_FEAT_DIMS, goal_dim
from supermimic.policy.nets import Policy


def _fake_checkpoint(path, robot: str) -> None:
    cfg = TrainConfig()
    cfg.env.goal_features, cfg.env.anchor_obs, cfg.env.robot = "sonic", "pelvis_ori10", robot
    cfg.policy.goal_features = "sonic"
    cfg.policy.goal_frame_dim = GOAL_FEAT_DIMS["sonic"]
    cfg.policy.goal_dim = goal_dim(cfg.env.future_frames, "sonic", "pelvis_ori10")
    torch.manual_seed(0)
    policy = Policy(cfg.policy)
    torch.save({"policy": policy.state_dict(), "config": asdict(cfg), "iteration": 0}, path)


def test_export_bundle(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    _fake_checkpoint(ckpt, "g1_brainco")
    report = export_bundle(ckpt, tmp_path / "bundle", parity_samples=8)
    assert report.max_abs_err < 1e-4
    model = onnx.load(str(tmp_path / "bundle" / "policy.onnx"))
    assert [i.name for i in model.graph.input] == ["obs_dict"]
    assert [o.name for o in model.graph.output] == ["action"]
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 1580]
    assert model.opset_import[0].version == ONNX_OPSET
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert manifest["robot"] == "g1_brainco" and manifest["obs_dim"] == 1580
    # 7520-14 hip pitch vs SONIC's 7520-22: kp/kd scaled on hardware motors 0 and 6 only.
    assert report.flags == ["--motor-kp-scale", "0,6=0.405448", "--motor-kd-scale", "0,6=0.405448"]


def test_rejects_non_sonic_checkpoint(tmp_path):
    import pytest

    cfg = TrainConfig()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"policy": {}, "config": asdict(cfg)}, ckpt)
    with pytest.raises(ValueError, match="SONIC export needs"):
        export_bundle(ckpt, tmp_path / "bundle")
