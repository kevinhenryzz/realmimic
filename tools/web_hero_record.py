"""Record rollouts on benchmark clips for the site's videos, rendered in the page's own scene by
tools/web_hero_render.mjs: the opening montage (the released policy alone) and the side-by-side comparisons
(the released policy and SONIC's release policy on the same clip).

    uv run python tools/web_hero_record.py --ckpt logs/release/ckpt_final_86000.pt --npz-dir data/npz/bones_seed \\
        --montage montage.json --out <scratch dir> [--sonic-release <SONIC release policy dir>]

``montage.json``: [{"clip": <benchmark clip>, "caption": "360 flip", "start": s, "end": s}] (seconds of the clip;
optional). Rollouts are ``run_benchmark``'s (Appendix F protocol, deterministic, from the clip's first frame).

- Montage (no ``--sonic-release``): failure terminations on; a clip the policy fails is refused.
- Comparison (``--sonic-release``): failure terminations off, as in tools/render_rollouts.py, so both robots
  play to the end of the clip; the times at which each goes off track (the benchmark's failure) and falls
  are found from the recorded states with render_rollouts' ``events`` and written for the page's labels.
- ``--world-model <site>``: our pane comes from the browser demo's own loop instead (CPU MuJoCo, the site's
  policy.onnx and world_model.onnx; ``realmimic.eval.web_closed_loop``), with the world model re-synced to the
  simulation every training horizon (32 ticks) as in the live demo; its pose is recorded as a third track and
  drawn as the demo's blue overlay. A clip our policy fails in that loop is refused.

Output: ``<out>/<pane>/NN.f32`` per montage entry and pane (``ours``, and ``sonic`` for a comparison): robot
qpos, then reference qpos (then the world model's qpos with ``--world-model``), per 50 Hz step (float32, 2 or 3 x nq), rotated so the clip starts at the origin heading
+x, with a ``montage.json`` per pane (captions, frame counts); for a comparison also ``<out>/events.json``
(per entry: the caption and, per pane, the off-track and fall times in seconds or null).

BONES-SEED licence: these files are derived from the dataset; they stay on this machine, are deleted after
rendering, and only the rendered pixels are published. Captions are generic; clip names never go into
anything published.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FPS = 50


def heading_frame(qpos: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate and translate both trajectories so the reference starts at the origin facing +x."""
    w, x, y, z = ref[0, 3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    qz = np.array([np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2)])  # yaw rotation as a quaternion (wxyz)
    origin = ref[0, :2].copy()
    out = []
    for q in (qpos, ref):
        q = q.copy()
        q[:, :2] = (q[:, :2] - origin) @ rot.T
        a, b = qz, q[:, 3:7]
        q[:, 3:7] = np.stack([
            a[0] * b[:, 0] - a[3] * b[:, 3], a[0] * b[:, 1] - a[3] * b[:, 2],
            a[0] * b[:, 2] + a[3] * b[:, 1], a[0] * b[:, 3] + a[3] * b[:, 0],
        ], 1)  # fmt: skip
        out.append(q)
    return out[0], out[1]


def write_pane(out: Path, montage: list[dict], trajectories: list[tuple[np.ndarray, ...]]) -> None:
    """Per entry: robot qpos and reference qpos, and the world model's qpos when the trajectory has one."""
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, (m, tr) in enumerate(zip(montage, trajectories)):
        q, ref = heading_frame(tr[0], tr[1])
        tracks = [q, ref]
        if len(tr) > 2:  # the world model, in the same frame as the robot (same transform, from the reference)
            tracks.append(heading_frame(tr[2], tr[1])[0])
        a = int(round(m.get("start", 0.0) * FPS))
        b = len(q) if m.get("end") is None else min(len(q), int(round(m["end"] * FPS)))
        np.concatenate([t[a:b] for t in tracks], 1).astype(np.float32).tofile(out / f"{i:02d}.f32")
        index.append({"file": f"{i:02d}.f32", "caption": m["caption"], "frames": b - a, "nq": q.shape[1], "tracks": len(tracks)})
    (out / "montage.json").write_text(json.dumps(index, indent=1))


def browser_loop(site: Path, npz: str, horizon: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    """The live demo's loop on a clip: robot, reference and world-model qpos at the start of every tick, and the
    tick of a training termination (None: the clip completes)."""
    import mujoco
    import onnxruntime as ort

    from realmimic.eval.web_closed_loop import SOFT_JOINT_LIMIT_FACTOR, clip_reference, fallen

    meta = json.loads((site / "assets/policy/meta.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(site / "assets/robot/scene.mjb"))
    pol = ort.InferenceSession(str(site / "assets/policy/policy.onnx"), providers=["CPUExecutionProvider"])
    wm = ort.InferenceSession(str(site / "assets/policy/world_model.onnx"), providers=["CPUExecutionProvider"])
    H, K = meta["inputs"]["state_hist"]["shape"][1], meta["inputs"]["ref_window"]["shape"][1]
    ref = clip_reference(npz, np.asarray(meta["default_joint_pos"]), K, freeze_end=True)
    act_jnt = model.actuator_trnid[:, 0] - 1
    lo, hi = model.jnt_range[1:].T
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_JOINT_LIMIT_FACTOR
    d = mujoco.MjData(model)
    d.qpos[:] = ref.qpos0
    d.qpos[7:] = np.clip(ref.qpos0[7:], mid - half, mid + half)
    d.qvel[:] = ref.qvel0
    mujoco.mj_forward(model, d)
    state = lambda: np.concatenate([d.qpos, d.qvel]).astype(np.float32)  # noqa: E731
    hist_s, hist_a = np.repeat(state()[None], H, 0), np.zeros((H, 29), np.float32)
    act = lambda s_, a_, k: pol.run(["action", "joint_target"], {"state_hist": s_[None], "action_hist": a_[None], "ref_window": ref.windows[k][None].astype(np.float32)})  # noqa: E731
    w_s = w_a = None
    age = horizon
    robot, refq, wmq, fell = [], [], [], None
    one = np.ones((1, 1), np.float32)
    for k in range(len(ref.windows)):
        if age >= horizon:  # re-sync before either steps, as the page does
            w_s, w_a, age = hist_s.copy(), hist_a.copy(), 0
        robot.append(d.qpos.copy())
        refq.append(ref.qpos0[:36] if k == 0 else ref.windows[k - 1][0, :36])
        wmq.append(w_s[-1, :36].astype(np.float64))
        a, target = act(hist_s, hist_a, k)
        d.ctrl[:] = np.asarray(target).reshape(29)[act_jnt]
        for _ in range(int(meta["decimation"])):
            mujoco.mj_step(model, d)
        hist_s = np.concatenate([hist_s[1:], state()[None]])
        hist_a = np.concatenate([hist_a[1:], np.asarray(a, np.float32).reshape(1, 29)])
        aw, _ = act(w_s, w_a, k)  # the world model's own rollout: the policy on its imagined history
        nxt = wm.run(["next_state"], {"state": w_s[-1][None], "action": np.asarray(aw, np.float32).reshape(1, 29), "friction": one})[0][0]
        w_s = np.concatenate([w_s[1:], nxt[None].astype(np.float32)])
        w_a = np.concatenate([w_a[1:], np.asarray(aw, np.float32).reshape(1, 29)])
        age += 1
        if fell is None and fallen(d.qpos, ref.windows[k][0], meta["termination"]["height"], meta["termination"]["tilt"]):
            fell = k
    return np.stack(robot), np.stack(refq), np.stack(wmq), fell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sonic-release", type=Path, default=None, help="SONIC's release policy directory: record a comparison")
    parser.add_argument("--sonic-robot", default="g1", help="robot the SONIC release policy runs on")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-model", type=Path, default=None, metavar="SITE", help="our pane from the browser loop with the world model (the site directory)")
    args = parser.parse_args()

    import gc

    import torch

    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import Recording, env_kwargs, load_policy_from_checkpoint, run_benchmark

    montage = json.loads(args.montage.read_text())
    names = sorted({m["clip"] for m in montage})
    files = [str(f) for f in discover_npz(args.npz_dir, names=names)]
    missing = set(names) - {Path(f).stem for f in files}
    if missing:
        raise SystemExit(f"{len(missing)} clips not found under {args.npz_dir}")
    policy, cfg = load_policy_from_checkpoint(args.ckpt, args.device)

    if args.world_model is not None:  # our pane from the browser demo's loop, with the world model
        by_name = {Path(f).stem: f for f in files}
        tracks = {}
        for m in montage:
            if m["clip"] not in tracks:
                robot, refq, wmq, fell = browser_loop(args.world_model, by_name[m["clip"]])
                if fell is not None:
                    raise SystemExit(f"{m['caption']}: our policy falls at {fell / FPS:.2f} s in the browser loop")
                tracks[m["clip"]] = (robot, refq, wmq)
        write_pane(args.out / "ours", montage, [tracks[m["clip"]] for m in montage])
        print(f"{len(montage)} entries recorded with the world model")
        if args.sonic_release is None:
            return

    if args.sonic_release is None:  # the montage: our policy, benchmark terminations
        rec = Recording()
        results = {r.name: r for r in run_benchmark(policy, files, device=args.device, num_envs=min(len(files), 8), verbose=False, recording=rec, **env_kwargs(cfg))}
        for i, m in enumerate(montage):
            if results[m["clip"]].failed:
                raise SystemExit(f"montage entry {i} ({m['caption']}): the policy fails this clip")
        write_pane(args.out / "ours", montage, [(rec.qpos[m["clip"]], rec.ref_qpos[m["clip"]]) for m in montage])
        print(f"{len(montage)} entries recorded")
        return

    # A comparison: both policies play every clip to its end (render_rollouts' recording and events).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_rollouts import events, record

    from realmimic.eval.sonic_release_policy import SonicReleasePolicy

    ours, _ = record(policy, files, args.device, **env_kwargs(cfg))
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    sonic, _ = record(SonicReleasePolicy(args.sonic_release, args.device), files, args.device, robot=args.sonic_robot)
    report = []
    for pane, rec in (("ours", ours), ("sonic", sonic)):
        if pane == "ours" and args.world_model is not None:
            continue  # written above, from the browser loop with the world model
        write_pane(args.out / pane, montage, [(rec.qpos[m["clip"]], rec.ref_qpos[m["clip"]]) for m in montage])
    for i, m in enumerate(montage):
        a = int(round(m.get("start", 0.0) * FPS))
        entry = {"caption": m["caption"], "panes": {}}
        for pane, rec in (("ours", ours), ("sonic", sonic)):
            off, down = events(rec.qpos[m["clip"]], rec.ref_qpos[m["clip"]])
            entry["panes"][pane] = {k: None if v is None else round((v - a) / FPS, 2) for k, v in (("off_track", off), ("fell", down))}
        report.append(entry)
        print(f"{i:02d} {m['caption']}: " + ", ".join(f"{p} off track {e['off_track']} fell {e['fell']}" for p, e in entry["panes"].items()))
    (args.out / "events.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
