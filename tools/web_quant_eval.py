"""Closed-loop tracking of web-policy exports (e.g. weight encodings) in CPU MuJoCo with the site's loop.

    uv run python tools/web_quant_eval.py site/assets/robot/scene.mjb <results.json> --checkpoint logs/<run>/ckpt_X.pt \
        --variant fp32=<dir> --variant int8=<dir> --replicas 3 \
        --sample bones data/bench/full.txt data/npz/bones_seed 300 --list lafan1 data/bench/lafan1.txt data/npz/lafan1

Each ``--variant`` directory holds an export (``policy.onnx`` + ``meta.json``). Every variant plays the
same clips (``realmimic.eval.web_closed_loop.track_clips``: benchmark protocol, training termination)
and the stand + scripted pushes loop at several push scales (the file itself, batch 1).
Batch-1 inference streams all weights from memory on every call, so clips run in lockstep, ``--batch``
at a time, on a twin of each file exported from ``--checkpoint`` with the same weight encoding and a
fixed batch; the twin must match the file on the export's parity inputs (``--batch 1``: the file).
``--replicas R`` replays the first variant R times with a reset perturbation of the joints
(``--joint-noise`` rad): the spread of those runs is the noise level.
``results.json`` holds per-clip results (clip names): keep it out of the repository. It is saved after
every task; clips already in it for the same variant, export and noise are skipped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

PUSH_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
_W: dict = {}


def _init(scene: str) -> None:
    import mujoco
    import torch

    torch.set_num_threads(1)
    _W["model"] = mujoco.MjModel.from_binary_path(scene)
    _W["sessions"] = {}


def _session(onnx: str):
    if onnx not in _W["sessions"]:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
        _W["sessions"][onnx] = lambda x: sess.run(["action", "joint_target"], x)
    return _W["sessions"][onnx]


def _clips(task):
    from realmimic.eval.web_closed_loop import clip_reference, track_clips

    label, onnx, meta, batch, clips, noise, seed = task
    q0, K, freeze = np.asarray(meta["default_joint_pos"]), meta["inputs"]["ref_window"]["shape"][1], meta.get("freeze_clip_end", False)
    items = [((s, f), (lambda f=f: clip_reference(f, q0, K, freeze)), noise, seed) for s, f in clips]
    return label, [(s, Path(f).stem, asdict(r)) for (s, f), r in track_clips(_W["model"], _session(onnx), items, meta, batch)]


def _stand(task):
    from realmimic.eval.web_closed_loop import DEFAULT_PUSHES, run

    label, onnx, meta, scale = task
    sess = _session(onnx)
    pushes = [(s, d, b, tuple(scale * f for f in force)) for s, d, b, force in DEFAULT_PUSHES]
    tr = run(_W["model"], lambda x: [y[0] for y in sess(x)], meta, 10.0, pushes)
    return label, scale, {"fall_t": None if tr.fall_tick is None else tr.t[tr.fall_tick], "min_z": float(min(q[2] for q in tr.qpos))}


def clip_sets(args) -> dict[str, list[str]]:
    sets = {}
    for name, listfile, directory, n in args.sample or []:
        names = sorted(c for c in Path(listfile).read_text().splitlines() if c.strip())
        pick = np.random.default_rng(args.seed).choice(len(names), size=min(int(n), len(names)), replace=False)
        sets[name] = [str(Path(directory) / f"{names[i]}.npz") for i in sorted(pick)]
    for name, listfile, directory in args.list or []:
        sets[name] = [str(Path(directory) / f"{c}.npz") for c in Path(listfile).read_text().splitlines() if c.strip()]
    return sets


def batched_twins(checkpoint: str, variants: dict[str, str], batch: int, out: Path) -> dict[str, str]:
    """Export each variant's encoding with a fixed ``batch``; check it against the file (batch 1)."""
    import onnxruntime as ort

    from realmimic.export.web_policy import INPUTS, build, export_onnx, sample_inputs

    web, c, _ = build(checkpoint)
    cfg = web.policy.cfg
    x = sample_inputs(c, batch, cfg.history, cfg.future_frames, seed=1)
    xb = {k: np.concatenate([s[k] for s in x]) for k in INPUTS}
    twins = {}
    for label, path in variants.items():
        enc = json.loads((Path(path) / "meta.json").read_text()).get("weight_encoding", "fp16")
        twins[label] = str(out / f"{label}.onnx")
        export_onnx(web, c, twins[label], enc, batch=batch)
        file = ort.InferenceSession(str(Path(path) / "policy.onnx"), providers=["CPUExecutionProvider"])
        ref = np.concatenate([file.run(["action"], s)[0] for s in x])
        err = float(np.abs(ort.InferenceSession(twins[label], providers=["CPUExecutionProvider"]).run(["action"], xb)[0] - ref).max())
        print(f"  {label}: batch-{batch} twin ({enc}) vs file max abs action {err:.2e}")
        if err > 1e-4:
            raise RuntimeError(f"{label}: batched twin does not match {path}")
    return twins


def summarize(res: dict, sets: list[str]) -> None:
    labels = list(res)
    print(f"{'variant':16s}" + "".join(f" | {s + ' fail':>16s} {'joint rad':>9s} {'pelvis m':>8s}" for s in sets) + " | stand+pushes (push scale: fall t)")
    for v in labels:
        row = f"{v:16s}"
        for s in sets:
            r = list(res[v]["clips"].get(s, {}).values())
            if not r:
                row += f" | {'-':>16s} {'':>9s} {'':>8s}"
                continue
            fails = sum(c["fail_step"] >= 0 for c in r)
            row += f" | {f'{fails}/{len(r)}':>16s} {np.mean([c['joint_err'] for c in r]):9.4f} {np.mean([c['pelvis_err'] for c in r]):8.4f}"
        stand = res[v].get("stand", {})
        row += " | " + " ".join(f"{k}:{'ok' if x['fall_t'] is None else 'FELL ' + format(x['fall_t'], '.2f')}" for k, x in sorted(stand.items(), key=lambda t: float(t[0])))
        print(row)
    base = labels[0]
    for v in labels[1:]:
        diff = []
        for s in sets:
            a, b = res[base]["clips"].get(s, {}), res[v]["clips"].get(s, {})
            diff.append(f"{s} {sum((c['fail_step'] >= 0) != (a[n]['fail_step'] >= 0) for n, c in b.items() if n in a)}")
        print(f"  {v} vs {base}: clips with a different outcome: " + ", ".join(diff))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene")
    parser.add_argument("results")
    parser.add_argument("--variant", action="append", required=True, help="label=export_dir")
    parser.add_argument("--checkpoint", help="checkpoint of the exports (needed for --batch > 1)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sample", nargs=4, action="append", metavar=("NAME", "LIST", "DIR", "N"), help="random sample of N of the clip names (one per line) in LIST, files in DIR")
    parser.add_argument("--list", nargs=3, action="append", metavar=("NAME", "LIST", "DIR"), help="clip names (one per line) in DIR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicas", type=int, default=0)
    parser.add_argument("--joint-noise", type=float, default=1e-3)
    parser.add_argument("--no-stand", action="store_true")
    parser.add_argument("--procs", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.batch > 1 and not args.checkpoint:
        parser.error("--batch > 1 needs --checkpoint")
    variants = dict(v.split("=", 1) for v in args.variant)
    metas = {label: json.loads((Path(p) / "meta.json").read_text()) for label, p in variants.items()}
    sets = clip_sets(args)
    out = Path(args.results)
    res = json.loads(out.read_text()) if out.exists() else {}
    first = next(iter(variants))
    runs = [(label, label, 0.0, 0) for label in variants] + [(f"{first}~r{r}", first, args.joint_noise, r) for r in range(1, args.replicas + 1)]
    for label, v, noise, _ in runs:
        key = {"export": str(Path(variants[v]).resolve()), "joint_noise": noise}
        if {k: res.get(label, {}).get(k) for k in key} != key:
            res[label] = {**key, "clips": {}, "stand": {}}
        for s in sets:
            res[label]["clips"].setdefault(s, {})
    lengths = {}
    for f in {f for files in sets.values() for f in files}:
        with np.load(f) as d:
            lengths[f] = d["joint_pos"].shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        onnx = batched_twins(args.checkpoint, variants, args.batch, Path(tmp)) if args.batch > 1 else {k: str(Path(p) / "policy.onnx") for k, p in variants.items()}
        tasks = []
        for label, v, noise, seed in runs:
            todo = sorted(((s, f) for s, files in sets.items() for f in files if Path(f).stem not in res[label]["clips"][s]), key=lambda t: -lengths[t[1]])
            # tasks of 2 * batch clips of similar length (a batch with few live slots costs as much as a full one)
            tasks += [(label, onnx[v], metas[v], args.batch, todo[j : j + 2 * args.batch], noise, seed) for j in range(0, len(todo), 2 * args.batch)]
        tasks.sort(key=lambda t: -lengths[t[4][0][1]])
        stand = [] if args.no_stand else [(label, str(Path(p) / "policy.onnx"), metas[label], k) for label, p in variants.items() for k in PUSH_SCALES if str(k) not in res[label]["stand"]]
        print(f"{len(tasks)} clip tasks ({sum(len(t[4]) for t in tasks)} clip runs), {len(stand)} stand runs", flush=True)
        with mp.get_context("spawn").Pool(args.procs, _init, (args.scene,)) as pool:
            for label, k, r in pool.imap_unordered(_stand, stand):
                res[label]["stand"][str(k)] = r
            out.write_text(json.dumps(res))
            for i, (label, rows) in enumerate(pool.imap_unordered(_clips, tasks)):
                for s, name, r in rows:
                    res[label]["clips"][s][name] = r
                out.write_text(json.dumps(res))
                print(f"  {i + 1}/{len(tasks)} tasks", flush=True)
    summarize({k: v for k, v in res.items() if k in {r[0] for r in runs}}, list(sets))


if __name__ == "__main__":
    main()
