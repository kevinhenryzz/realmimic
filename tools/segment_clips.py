"""Split converted npz clips into fixed-length segments, each its own clip (``<name>__seg<k>.npz``).

Scoring long clips segment by segment, each started from its own reference state, separates tracking skill
from slow drift over minutes (e.g. global position, which a goal without root position cannot correct).
Segments are local evaluation inputs only; they are as redistributable as their source clips, i.e. not.

    uv run python tools/segment_clips.py data/npz/lafan1 data/npz/lafan1_seg10 --seconds 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def segment(path: Path, out_dir: Path, frames: int, min_frames: int) -> int:
    with np.load(path) as d:
        data = {k: d[k] for k in d.files}
    n = data["joint_pos"].shape[0]
    per_frame = [k for k, v in data.items() if v.ndim >= 1 and v.shape[0] == n]
    count = 0
    for k, start in enumerate(range(0, n, frames)):
        stop = min(start + frames, n)
        if stop - start < min_frames:
            break
        seg = {key: (val[start:stop] if key in per_frame else val) for key, val in data.items()}
        np.savez(out_dir / f"{path.stem}__seg{k:03d}.npz", **seg)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=50.0, help="frame rate of the converted clips")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = round(args.seconds * args.fps)
    total = sum(segment(p, args.out_dir, frames, frames // 2) for p in sorted(args.npz_dir.glob("*.npz")))
    print(f"{total} segments of {frames} frames in {args.out_dir}")


if __name__ == "__main__":
    main()
