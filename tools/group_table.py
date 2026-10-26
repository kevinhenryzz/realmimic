"""Failed motions per Table A.8 group for full-set benchmark results, next to the paper's figures (counts only).

    uv run python tools/group_table.py "SONIC release=runs/benchmark/sonic_release_g1_full.json" \\
        "ours=runs/benchmark/release/ckpt_final_full.json" [--manifest data/npz/bones_seed/manifest_filtered.csv]

Group sizes are ours, by name pattern on the manifest's kept clips; the paper's sizes and figures (its evaluator,
its 129,514-clip set) are quoted from Table A.8. No clip names are printed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

from realmimic.eval.motion_groups import OTHER, TABLE_A8_GROUPS, motion_group

GROUP_LABELS = {
    "side_rolls": "side rolls",
    "flips": "flips (360 / 180 / 90)",
    "fire_in_the_hole_turns": "fire-in-the-hole turns",
    "crossed_legs": "crossed legs",
    "lying_resting_standing_up": "lying, resting and standing up",
    "broad_jumps": "broad jumps",
    "dance_routine": "dance routine",
    OTHER: "all other",
}
# Paper Table A.8: (group size, SONIC failed, SuperMimic 1 GPU x 3 h failed), in the paper's evaluator.
PAPER = {
    "side_rolls": (22, 22, 2),
    "flips": (44, 44, 1),
    "fire_in_the_hole_turns": (20, 3, 3),
    "crossed_legs": (10, 9, 0),
    "lying_resting_standing_up": (99, 28, 8),
    "broad_jumps": (24, 15, 0),
    "dance_routine": (8, 8, 0),
    OTHER: (129_287, 50, 11),
}


def failures(path: str) -> Counter:
    with open(path) as f:
        clips = json.load(f)["clips"]
    return Counter(motion_group(c["name"]) for c in clips if c["failed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="LABEL=path to a full-set benchmark JSON")
    parser.add_argument("--manifest", default="data/npz/bones_seed/manifest_filtered.csv")
    args = parser.parse_args()
    with open(args.manifest) as f:
        sizes = Counter(motion_group(r["name"]) for r in csv.DictReader(f) if r.get("keep", "1") in ("1", "True", "true"))
    runs = [(label, failures(path)) for label, path in (r.split("=", 1) for r in args.results)]
    groups = [*TABLE_A8_GROUPS, OTHER]
    head = ["group", "clips, ours (paper)", "SONIC, paper", "SuperMimic 3 h, paper", *(label for label, _ in runs)]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for g in groups:
        size, sonic, sm = PAPER[g]
        row = [GROUP_LABELS[g], f"{sizes[g]:,} ({size:,})", str(sonic), str(sm), *(str(f[g]) for _, f in runs)]
        print("| " + " | ".join(row) + " |")
    total = ["total", f"{sum(sizes.values()):,} (129,514)", "179", "25", *(str(sum(f.values())) for _, f in runs)]
    print("| " + " | ".join(total) + " |")


if __name__ == "__main__":
    main()
