"""run_benchmark's recording hook: one qpos sample per env step for the robot and the reference, and
the same results as without it."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not NPZ.exists(), reason="LAFAN1 npz not built"),
]


def zero_policy(obs, goal):
    return torch.zeros(obs.shape[0], 29, device=obs.device)


def test_recording_matches_results():
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import Recording, run_benchmark

    files = [str(f) for f in discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])]
    rec = Recording()
    # One env for two clips: the second clip's samples must start after the queue reset.
    results = run_benchmark(zero_policy, files, num_envs=1, verbose=False, recording=rec)
    assert sorted(rec.qpos) == sorted(r.name for r in results)
    nq = rec.spec.compile().nq
    for r in results:
        q, ref = rec.qpos[r.name], rec.ref_qpos[r.name]
        assert r.failed  # zero actions fall
        assert r.max_height_error > 0.25 or r.max_tilt_error > 0.8  # the failing check is in the margins
        assert q.shape == ref.shape == (r.steps, nq)
        np.testing.assert_allclose(q[0], ref[0], atol=1e-3)  # reset to the reference's first frame
        assert np.isfinite(q).all()
    plain = run_benchmark(zero_policy, files, num_envs=1, verbose=False)
    assert [(r.steps, r.failed) for r in plain] == [(r.steps, r.failed) for r in results]
    # Same steps and outcome; returns agree to float32 accumulation noise (GPU contact solver is not
    # bitwise deterministic run to run: 1e-6 relative was exceeded once on an RTX 5090).
    np.testing.assert_allclose([r.norm_return for r in plain], [r.norm_return for r in results], rtol=1e-5)
