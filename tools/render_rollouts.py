"""Render benchmark rollouts to MP4: our checkpoint and, side by side, SONIC's release policy on the
same clip, each with a translucent ghost of the reference motion.

Both policies are rolled out by ``run_benchmark`` (deterministic, from the clip's first frame) with
the failure terminations off, so a clip plays on after the robot loses track. The benchmark's failure
(pelvis height error > 0.25 m or tilt error > 0.8 against the reference) is computed from the recorded
states and labelled "off track"; "fell" marks, once off track, the pelvis below 0.35 m or tilted more
than 1 rad from upright (the sim-to-sim fall test). Every pane plays to the end of the clip, falls included,
so side-by-side panes stay in step. Rendering is offline with MuJoCo (EGL by default), 1280x720 at the 50 Hz control rate, H.264.
Videos are numbered, never named after clips (BONES-SEED licence); the index -> clip mapping and the
benchmark results go to ``<out-dir>/clips.json``, which is for local use only.

    uv run python tools/render_rollouts.py --ckpt logs/release/ckpt_latest.pt \\
        --sonic-release ~/realmimic-sonic/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release \\
        --npz-dir data/npz/bones_seed --clips <clip names from your copy> --out-dir runs/videos
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import gc
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import (
    ClipResult,
    Recording,
    env_kwargs,
    load_policy_from_checkpoint,
    run_benchmark,
)

W, H, FPS = 1280, 720, 50  # FPS = control rate: one frame per env step
GHOST_RGBA = (0.2, 0.6, 1.0, 0.3)
MAX_ENVS = 8
HEIGHT_ERR, TILT_ERR = 0.25, 0.8  # the benchmark's failure thresholds (training terminations)
FALL_HEIGHT, FALL_TILT = 0.35, 1.0  # deploy/sonic sim-to-sim fall test


def record(policy, files: list[str], device: str, **kwargs) -> tuple[Recording, dict[str, ClipResult]]:
    rec = Recording()
    results = run_benchmark(policy, files, device=device, num_envs=min(len(files), MAX_ENVS), verbose=False, recording=rec, terminate=False, **kwargs)
    gc.collect()
    torch.cuda.empty_cache()
    return rec, {r.name: r for r in results}


def up_z(quat: np.ndarray) -> np.ndarray:
    """z component of the body z axis, R[2, 2], for (N, 4) wxyz quaternions."""
    return 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)


def first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def events(qpos: np.ndarray, ref_qpos: np.ndarray) -> tuple[int | None, int | None]:
    """Steps at which the robot goes off track (the benchmark's failure) and falls, or None."""
    height = ref_qpos[:, 2] - qpos[:, 2]
    tilt = up_z(qpos[:, 3:7]) - up_z(ref_qpos[:, 3:7])  # the termination's projected-gravity z difference
    off = first((np.abs(height) > HEIGHT_ERR) | (np.abs(tilt) > TILT_ERR))
    if off is None:  # inverted or low poses on track are the motion itself (flips, rolls), not falls
        return None, None
    down = first((qpos[off:, 2] < FALL_HEIGHT) | (up_z(qpos[off:, 3:7]) < np.cos(FALL_TILT)))
    return off, None if down is None else off + down


def yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def render_model(spec, width: int, height: int):
    """The benchmark scene restyled for video: light floor, sky gradient, collision geoms hidden."""
    spec = spec.copy()
    for tex in spec.textures:
        if tex.name == "groundplane":  # mjlab's plane terrain
            tex.rgb1, tex.rgb2, tex.markrgb = [0.82, 0.84, 0.86], [0.70, 0.72, 0.75], [0.55, 0.57, 0.6]
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.reflectance = 0.0
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.95, 0.98], rgb2=[0.75, 0.80, 0.88], width=512, height=3072)  # fmt: skip
    spec.visual.rgba.haze = [0.9, 0.92, 0.95, 1.0]
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    model = spec.compile()
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and (model.geom_contype[g] or model.geom_conaffinity[g]):
            model.geom_rgba[g, 3] = 0.0
    return model


class Pane:
    """One policy's view: robot opaque, reference ghost translucent, camera following the pelvis."""

    def __init__(self, spec, label: str, width: int, height: int, distance: float, elevation: float):
        model = render_model(spec, width, height)
        self.model, self.label = model, label
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.robot, self.ghost = mujoco.MjData(model), mujoco.MjData(model)
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation = distance, elevation
        self.lookat = None

    def render(self, qpos: np.ndarray, ref_qpos: np.ndarray, azimuth: float) -> np.ndarray:
        self.robot.qpos[:], self.ghost.qpos[:] = qpos, ref_qpos
        mujoco.mj_kinematics(self.model, self.robot)
        mujoco.mj_kinematics(self.model, self.ghost)
        target = np.array([qpos[0], qpos[1], 0.8])
        self.lookat = target if self.lookat is None else self.lookat + 0.1 * (target - self.lookat)
        self.cam.lookat[:], self.cam.azimuth = self.lookat, azimuth
        self.renderer.update_scene(self.robot, camera=self.cam, scene_option=self.opt)
        scene = self.renderer.scene
        first = scene.ngeom
        mujoco.mjv_addGeoms(self.model, self.ghost, self.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for g in range(first, scene.ngeom):
            scene.geoms[g].rgba[:] = GHOST_RGBA
            scene.geoms[g].matid = -1
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def render_clip(path: Path, panes: list[Pane], recs: list[Recording], clip: str, azimuth_offset: float) -> tuple[int, list[tuple[int | None, int | None]]]:
    """Write one clip's video; returns its number of frames and each pane's (off-track, fall) steps."""
    qs, refs = [r.qpos[clip] for r in recs], [r.ref_qpos[clip] for r in recs]
    ev = [events(q, ref) for q, ref in zip(qs, refs)]
    ends = [len(q) for q in qs]  # play every pane to the clip end; the labels mark off-track and fall times
    azimuth = np.degrees(yaw(refs[0][0, 3:7])) + azimuth_offset  # same view of the motion in every pane
    font = ImageFont.load_default(size=30)
    n = max(ends)
    for p in panes:
        p.lookat = None
    text = {"font": font, "stroke_width": 2, "stroke_fill": (255, 255, 255)}
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8) as writer:
        for s in range(n):
            imgs = []
            for p, q, ref, end, (off, down) in zip(panes, qs, refs, ends, ev):
                k = min(s, end - 1)
                img = Image.fromarray(p.render(q[k], ref[k], azimuth))
                draw = ImageDraw.Draw(img)
                draw.text((20, 16), p.label, fill=(20, 20, 20), **text)
                if off is not None and k >= off:
                    draw.text((20, 56), f"off track at {off / FPS:.1f} s (benchmark failure)", fill=(200, 110, 0), **text)
                if down is not None and k >= down:
                    draw.text((20, 96), f"fell at {down / FPS:.1f} s", fill=(200, 30, 30), **text)
                imgs.append(np.asarray(img))
            writer.append_data(np.concatenate(imgs, axis=1))
    return n, ev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True, help="our checkpoint")
    parser.add_argument("--sonic-release", type=Path, default=None, help="SONIC's release policy directory (second pane)")
    parser.add_argument("--sonic-robot", default="g1", help="robot the SONIC release policy runs on")
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=100.0, help="camera azimuth relative to the clip's initial heading (deg)")
    parser.add_argument("--distance", type=float, default=3.6)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    files = [str(f) for f in discover_npz(args.npz_dir, names=args.clips)]
    missing = set(args.clips) - {Path(f).stem for f in files}
    if missing:
        raise SystemExit(f"clips not found under {args.npz_dir}: {sorted(missing)}")

    policy, cfg = load_policy_from_checkpoint(args.ckpt, args.device)
    sides = [("realmimic", *record(policy, files, args.device, **env_kwargs(cfg)))]
    del policy
    if args.sonic_release is not None:
        from realmimic.eval.sonic_release_policy import SonicReleasePolicy

        sides.append(("SONIC release", *record(SonicReleasePolicy(args.sonic_release, args.device), files, args.device, robot=args.sonic_robot)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panes = [Pane(rec.spec, label, W // len(sides), H, args.distance, args.elevation) for label, rec, _ in sides]
    index = {}
    try:
        for i, clip in enumerate(args.clips):
            path = args.out_dir / f"{i:02d}.mp4"
            n, ev = render_clip(path, panes, [rec for _, rec, _ in sides], clip, args.azimuth)
            index[path.name] = {"clip": clip, "frames": sides[0][2][clip].frames, "video_frames": n,
                                "panes": [{"label": label, "off_track_step": off, "fall_step": down} for (label, _, _), (off, down) in zip(sides, ev)]}  # fmt: skip
            print(f"{path}: {n} frames, " + ", ".join(f"{p['label']} off track {p['off_track_step']} fell {p['fall_step']}" for p in index[path.name]["panes"]))
    finally:
        for p in panes:
            p.close()
    meta = {"checkpoint": str(args.ckpt), "sonic_release": None if args.sonic_release is None else str(args.sonic_release), "videos": index}
    (args.out_dir / "clips.json").write_text(json.dumps(meta, indent=1))
    print("wrote", args.out_dir / "clips.json")


if __name__ == "__main__":
    main()
