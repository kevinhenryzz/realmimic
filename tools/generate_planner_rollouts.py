"""Generate SONIC-planner reference clips (50 Hz npz) plus a manifest and a summary.

Example::

    uv run tools/generate_planner_rollouts.py --out data/npz/planner_v0 --num-clips 200 \\
        --seconds 20 --seed 0 --workers 8

Each clip ``planner_<index>`` is reproducible from ``(--seed, index)`` alone, independent of
``--workers``. Writes ``<out>/<name>.npz``, ``<out>/manifest.csv`` (one row per clip; ``name`` and
``keep`` columns as ``discover_npz`` expects) and ``<out>/summary.json`` (aggregate statistics).
The planner model is read from ``--planner`` (default: ``$REALMIMIC_PLANNER_ONNX`` or
``third_party/gear_sonic/planner_sonic.onnx``); it is never copied.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from realmimic.data import planner_rollouts as pr

MAX_ATTEMPTS = 3

_planner: pr.PlannerSession | None = None
_model = None
_opts: argparse.Namespace | None = None


def _init_worker(opts: argparse.Namespace) -> None:
    global _planner, _model, _opts
    from supermimic.data.motion_convert import load_g1_model

    _opts = opts
    _planner = pr.PlannerSession(opts.planner, num_threads=opts.threads)
    _model = load_g1_model()


def _clip_stats(res: pr.ClipResult) -> dict:
    """Per-frame arrays the parent aggregates into summary.json."""
    arrays = res.arrays
    n = arrays["joint_pos"].shape[0]
    foot = pr.foot_contact_stats(_model, arrays)
    cont = pr.stitch_continuity(arrays, res.rollout.stitches)
    return {
        "modes": pr.frame_modes(res.script, n),
        "root_z": arrays["body_pos_w"][:, 0, 2].astype(np.float64),
        "lowest": foot["lowest"],
        "slide": foot["slide"],
        "raw_limit_excess": res.raw_limit_excess,
        "gaps": np.array([s.old_new_joint_gap for s in res.rollout.stitches]),
        "root_gaps": np.array([s.old_new_root_gap for s in res.rollout.stitches]),
        **cont,
    }


def _work(index: int) -> tuple[int, dict | None, dict | None, str | None]:
    assert _planner is not None and _opts is not None
    name = f"{_opts.prefix}_{index:05d}"
    script_cfg = pr.ScriptConfig(modes=tuple(_opts.modes))
    err = None
    base_seed = pr.clip_seed(_opts.seed, index)
    for attempt in range(MAX_ATTEMPTS):
        # Retries (after an invalid plan) use a seed derived from the clip's own seed.
        seed = base_seed if attempt == 0 else pr.clip_seed(base_seed, attempt)
        t0 = time.perf_counter()
        calls0, secs0 = _planner.num_calls, _planner.total_seconds
        try:
            res = pr.generate_clip(
                _planner, seed, _opts.seconds, name, model=_model, script_cfg=script_cfg,
                random_planner_seed=_opts.random_planner_seed,
                clip_joint_limits=not _opts.no_clip_joint_limits,
            )  # fmt: skip
        except pr.PlannerError as e:
            err = f"{name} seed {seed}: {e}"
            continue
        pr.save_npz(Path(_opts.out) / f"{name}.npz", res.arrays)
        row = res.manifest_row()
        stats = _clip_stats(res)
        stats["planner_calls"] = _planner.num_calls - calls0
        stats["planner_seconds"] = _planner.total_seconds - secs0
        stats["wall_seconds"] = time.perf_counter() - t0
        return index, row, stats, err
    return index, None, None, err


def _pct(x: np.ndarray, qs=(1, 50, 99, 100)) -> dict[str, float] | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}


def summarize(stats: list[dict], rows: list[dict], failures: list[str]) -> dict:
    """Aggregate per-clip statistics."""
    cat = {k: np.concatenate([s[k] for s in stats]) for k in stats[0] if isinstance(stats[0][k],
           np.ndarray)}  # fmt: skip
    per_mode = {}
    for mi in np.unique(cat["modes"]):
        sel = cat["modes"] == mi
        slide = cat["slide"][sel]
        contact = np.isfinite(slide)
        per_mode[pr.MODE_NAMES[int(mi)]] = {
            "seconds": round(float(sel.sum()) / pr.CONTROL_FPS, 2),
            "root_z": _pct(cat["root_z"][sel]),
            "lowest_point_z": _pct(cat["lowest"][sel]),
            "floating_frac_gt_2cm": round(float((cat["lowest"][sel] > 0.02).mean()), 4),
            "penetration_frac_lt_-2cm": round(float((cat["lowest"][sel] < -0.02).mean()), 4),
            "stance_contact_frac": round(float(contact.mean()), 4),
            "stance_foot_speed": _pct(slide, (50, 90, 99)),
            "stance_slide_frac_gt_0.2mps": round(float((slide[contact] > 0.2).mean()), 4)
            if contact.any()
            else None,
            "raw_limit_excess_frac": round(float((cat["raw_limit_excess"][sel] > 1e-4).mean()), 4),
            "raw_limit_excess_max": round(float(max(cat["raw_limit_excess"][sel].max(), 0.0)), 4),
        }
    total_calls = sum(s["planner_calls"] for s in stats)
    total_secs = sum(s["planner_seconds"] for s in stats)
    reasons: dict[str, int] = {}
    for r in rows:
        for k, v in json.loads(r["replan_reasons"]).items():
            reasons[k] = reasons.get(k, 0) + v
    cont = {}
    for k in ("joint_step", "joint_acc", "root_step", "root_rot_step"):
        cont[k] = {"stitch": _pct(cat[f"{k}_stitch"]), "other": _pct(cat[f"{k}_other"])}
    return {
        "num_clips": len(rows),
        "num_failed": len(failures),
        "failures": failures,
        "total_seconds": round(sum(r["duration_s"] for r in rows), 2),
        "per_mode": per_mode,
        "replans": {
            "total": int(sum(r["num_replans"] for r in rows)),
            "per_clip_second": round(
                sum(r["num_replans"] for r in rows) / max(sum(r["duration_s"] for r in rows), 1e-9),
                3,
            ),
            "reasons": reasons,
        },
        "stitch_gap_joint_rad": _pct(cat["gaps"]),
        "stitch_gap_root_m": _pct(cat["root_gaps"]),
        "continuity": cont,
        "planner": {
            "calls": int(total_calls),
            "seconds_per_call": round(total_secs / max(total_calls, 1), 4),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-clips", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4, help="onnxruntime threads per worker")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--prefix", default="planner")
    default_planner = os.environ.get("REALMIMIC_PLANNER_ONNX", str(pr.DEFAULT_PLANNER_PATH))
    ap.add_argument("--planner", type=Path, default=Path(default_planner))
    ap.add_argument(
        "--modes",
        default=",".join(pr.V0_MODES),
        help=f"non-idle modes to sample (default v0). Available: {', '.join(pr.MODES)}",
    )
    ap.add_argument(
        "--random-planner-seed",
        action="store_true",
        help="fresh planner random_seed per call instead of the runtime's constant 1234",
    )
    ap.add_argument("--no-clip-joint-limits", action="store_true")
    opts = ap.parse_args(argv)
    opts.modes = [m.strip() for m in opts.modes.split(",") if m.strip()]
    unknown = [m for m in opts.modes if m not in pr.MODES or m == "idle"]
    if unknown:
        ap.error(f"unknown or invalid modes: {unknown}")
    if not opts.planner.exists():
        ap.error(f"planner model not found: {opts.planner}")
    opts.out.mkdir(parents=True, exist_ok=True)

    indices = list(range(opts.start_index, opts.start_index + opts.num_clips))
    rows: dict[int, dict] = {}
    stats: dict[int, dict] = {}
    failures: list[str] = []
    t0 = time.perf_counter()
    if opts.workers <= 1:
        _init_worker(opts)
        results = map(_work, indices)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(opts.workers, _init_worker, (opts,))
        results = pool.imap_unordered(_work, indices)
    for done, (index, row, st, err) in enumerate(results, 1):
        if err:
            failures.append(err)
        if row is not None and st is not None:
            rows[index], stats[index] = row, st
        if done % 10 == 0 or done == len(indices):
            print(f"[{done}/{len(indices)}] {time.perf_counter() - t0:.0f}s", flush=True)
    if pool is not None:
        pool.close()
        pool.join()

    ordered = [rows[i] for i in sorted(rows)]
    if not ordered:
        print("no clips generated", file=sys.stderr)
        return 1
    with open(opts.out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    summary = summarize([stats[i] for i in sorted(stats)], ordered, failures)
    summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
    summary["args"] = {k: str(v) for k, v in vars(opts).items()}
    (opts.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("num_clips", "num_failed", "total_seconds",
                                              "planner", "wall_seconds")}))  # fmt: skip
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
