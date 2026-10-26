"""Summarize benchmark result files (``scripts/eval_benchmark.py --out``) per Table A.8 motion group.

    uv run python tools/benchmark_report.py runs/benchmark/*.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from realmimic.eval.motion_groups import motion_group


def group_table(clips: list[dict]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        groups[motion_group(clip["name"])].append(clip)
    return {
        g: {
            "clips": len(cs),
            "failed": sum(c["failed"] for c in cs),
            "mean_norm_return": sum(c["norm_return"] for c in cs) / len(cs),
        }
        for g, cs in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.results:
        data = json.loads(path.read_text())
        s, meta = data["summary"], data.get("checkpoint", {})
        print(f"\n{path.name}: iteration {meta.get('iteration')}, "
              f"{(meta.get('train_seconds') or 0) / 3600:.2f} h | {s['clips']} clips | "
              f"failed {s['failed']} | mean norm return {s['mean_norm_return']:.4f}")
        for group, row in group_table(data["clips"]).items():
            print(f"  {group:24s} {row['clips']:6d} clips  failed {row['failed']:4d}  "
                  f"return {row['mean_norm_return']:.3f}")


if __name__ == "__main__":
    main()
