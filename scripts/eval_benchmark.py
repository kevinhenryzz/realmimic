"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/release/ckpt_latest.pt --npz-dir data/npz/lafan1 \\
      --out runs/benchmark/release_lafan1.json
  # SONIC's released encoder + decoder, with its own gains and action scale, on a RobotSpec body
  uv run python scripts/eval_benchmark.py --sonic-release third_party/sonic_release \\
      --robot g1 --npz-dir data/npz/bones_seed --clip-list data/bench/hard.txt --num-envs 256 --out runs/benchmark/sonic_release_hard.json
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
import tyro

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import (
    env_kwargs,
    load_policy_from_checkpoint,
    run_benchmark,
    save_results,
    summarize,
)


def main(ckpt: Path | None = None, sonic_release: Path | None = None, robot: str | None = None, npz_dir: Path = Path("data/npz/lafan1"), clips: list[str] | None = None, clip_list: Path | None = None, manifest: Path | None = None, num_envs: int = 1024, chunk_size: int | None = None, device: str = "cuda:0", out: Path | None = None, ema: bool = False) -> None:
    """``--ckpt``: one of our checkpoints (its config picks the robot unless ``--robot`` is given).
    ``--sonic-release``: SONIC's release policy directory (model_encoder.onnx, model_decoder.onnx,
    observation_config.yaml); ``--robot`` defaults to g1 and ``--out`` is required.
    ``--chunk-size``: evaluate the clips in chunks of this many (one env per chunk), so large sets
    such as the full BONES-SEED benchmark fit in GPU memory.
    ``--ema``: evaluate the checkpoint's policy_ema weights (policy-trainer.policy-ema > 0) instead of the raw ones."""
    if (ckpt is None) == (sonic_release is None):
        raise SystemExit("give exactly one of --ckpt and --sonic-release")
    if clip_list is not None:
        clips = [line.strip() for line in clip_list.read_text().splitlines() if line.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
    if sonic_release is not None:
        from realmimic.eval.sonic_release_policy import SonicReleasePolicy

        if out is None:
            raise SystemExit("--out is required with --sonic-release")
        policy = SonicReleasePolicy(sonic_release, device)
        meta = {"policy": "sonic_release", "policy_dir": str(policy.policy_dir), "robot": robot or "g1", "actuator_gains": "sonic"}
        results = _run_chunked(policy, files, chunk_size, device=device, num_envs=num_envs, robot=robot or "g1")
    else:
        policy, cfg = load_policy_from_checkpoint(ckpt, device, weights="policy_ema" if ema else "policy")
        kwargs = {**env_kwargs(cfg), **({"robot": robot} if robot else {})}
        results = _run_chunked(policy, files, chunk_size, device=device, num_envs=num_envs, **kwargs)
        meta = {**cfg.get("_meta", {}), "policy": "ours", "run_name": cfg.get("run_name"), "checkpoint_path": str(ckpt), "robot": robot or cfg["env"].get("robot", "g1"), "freeze_clip_end": cfg["env"].get("freeze_clip_end", False), "weights": "policy_ema" if ema else "policy"}
        hours = (meta.get("train_seconds") or 0.0) / 3600
        print(f"\ncheckpoint: iteration {meta.get('iteration')} ({meta.get('iteration_unit') or 'env steps (v1)'}), env steps {meta.get('env_steps')}, train time {hours:.2f} h")
    meta["clip_list"] = str(clip_list) if clip_list is not None else None
    s = summarize(results)
    print(f"{s['clips']} clips | failed {s['failed']} (at the clip end {s['failed_at_end']}) | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f} | v1 (survived-steps) {s['mean_norm_return_survived_v1']:.4f}")
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}.json"
    save_results(results, out, meta)
    print("saved", out)


def _run_chunked(policy, files: list[str], chunk_size: int | None, **kwargs) -> list:
    results = []
    for part in _chunks(files, chunk_size):
        results += run_benchmark(policy, part, **kwargs)
        # A chunk's environment (motion library, simulator) sits in reference cycles; free it before
        # the next one is built, or full-set runs accumulate GBs per chunk.
        gc.collect()
        torch.cuda.empty_cache()
        if chunk_size:
            print(f"[INFO] {len(results)}/{len(files)} clips; CUDA memory allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return results


def _chunks(files: list[str], size: int | None) -> list[list[str]]:
    size = size or len(files)
    return [files[i : i + size] for i in range(0, len(files), size)]


if __name__ == "__main__":
    tyro.cli(main)
