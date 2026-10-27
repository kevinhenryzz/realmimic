"""The browser demo's reference window (site/demo/motion.js, ClipMotion.window) must target the frames training and the
benchmark target: control tick k aims at clip frame k + 1 (the env advances the target before the first action), past
the clip's end the last frame held still. Checked against the training library's windows (CPU) and against the frame
run_benchmark actually feeds the policy at each step (GPU)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MOTION_JS = ROOT / "site/demo/motion.js"
K = 10  # reference window length of the released policy
Q0 = np.full(29, 0.1)  # base joint pose of the synthetic clip (the reference rows do not depend on the default pose)
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


def _clip(path: Path, frames: int = 40) -> np.ndarray:
    """A synthetic clip whose first joint encodes the frame index (0.3 + 0.01 f); returns the library's 71-d rows."""
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES
    from test_web_policy import _synthetic_clip  # same folder (pytest puts it on the path)

    _synthetic_clip(path, Q0, frames)
    data = dict(np.load(path))
    data["joint_pos"][:, 0] = 0.3 + 0.01 * np.arange(frames)
    data["joint_vel"][:, 0] = 0.5
    np.savez(path, **data)
    lib = MotionLibrary([path], TRACKED_BODIES, Q0, device="cpu", freeze_end=True)
    return lib._ref.double().numpy()


def _browser_windows(tmp_path: Path, rows: np.ndarray, ticks: int) -> np.ndarray:
    """ClipMotion.window(k) for k < ticks, run by Node on the page's own motion.js: (ticks, K, 71)."""
    shutil.copy(MOTION_JS, tmp_path / "motion.mjs")
    (tmp_path / "rows.bin").write_bytes(rows.astype(np.float32).tobytes())
    script = f"""
import {{ ClipMotion }} from './motion.mjs';
import {{ readFileSync }} from 'node:fs';
const b = readFileSync('./rows.bin');
const m = new ClipMotion(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4), false);
const out = new Float32Array({K} * 71), all = [];
for (let k = 0; k < {ticks}; k++) {{ m.window(k, [1, 0, 0, 0], {K}, out); all.push(...out); }}
process.stdout.write(JSON.stringify(all));
"""
    (tmp_path / "run.mjs").write_text(script)
    res = subprocess.run(["node", "run.mjs"], cwd=tmp_path, capture_output=True, text=True, check=True)
    return np.asarray(json.loads(res.stdout)).reshape(ticks, K, 71)


def test_browser_window_matches_training_library(tmp_path):
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

    from realmimic.eval.web_closed_loop import clip_reference

    rows = _clip(tmp_path / "clip.npz")
    T = len(rows)
    win = _browser_windows(tmp_path, rows, T + 30)
    # Inside the clip: the windows the env builds (web_closed_loop.clip_reference: tick k -> frames k+1 ...).
    ref = clip_reference(tmp_path / "clip.npz", Q0, K, freeze_end=True).windows
    np.testing.assert_allclose(win[: T - 1], ref[: T - 1], atol=1e-5)
    np.testing.assert_allclose(win[:, 0, 7], 0.3 + 0.01 * np.minimum(np.arange(T + 30) + 1, T - 1), atol=1e-5)
    # Past the end: training's freeze_clip_end (the last frame, counting the ticks already past it, held still).
    lib = MotionLibrary([tmp_path / "clip.npz"], TRACKED_BODIES, Q0, device="cpu", freeze_end=True)
    for k in range(T - 1, T + 30):
        idx, frozen = lib._window_index(torch.tensor([T - 1]), K, beyond=torch.tensor([k + 1 - (T - 1)]))
        expect = lib.frozen_ref71(lib._ref[idx], frozen)[0].double().numpy()
        np.testing.assert_allclose(win[k], expect, atol=1e-5, err_msg=f"tick {k}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_browser_target_frame_matches_run_benchmark(tmp_path):
    """The frame run_benchmark puts in row 0 of the goal at step k is the frame the browser targets at tick k."""
    from supermimic.eval.benchmark import run_benchmark

    rows = _clip(tmp_path / "clip.npz")
    seen = []

    def probe(obs, goal):  # the sonic goal starts with row 0's joint positions
        seen.append(float(goal[0, 0]))
        return torch.zeros(goal.shape[0], 29, device=goal.device)

    (r,) = run_benchmark(probe, [str(tmp_path / "clip.npz")], num_envs=1, verbose=False, goal_features="sonic",
                         anchor_obs="pelvis_ori10", freeze_clip_end=True, terminate=False)  # fmt: skip
    bench = np.round((np.asarray(seen[: r.steps]) - 0.3) / 0.01).astype(int)
    win = _browser_windows(tmp_path, rows, r.steps)
    browser = np.round((win[:, 0, 7] - 0.3) / 0.01).astype(int)
    assert bench[0] == 1  # the first action already aims at frame 1
    np.testing.assert_array_equal(browser, bench)
