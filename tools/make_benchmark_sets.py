"""Write the clip lists used to evaluate checkpoints during training (Appendix F protocol).

    uv run python tools/make_benchmark_sets.py data/npz/bones_seed/manifest_filtered.csv data/bench

- ``rand2000.txt``: 2,000 clips drawn uniformly (seed 0) from the paper-filtered BONES-SEED set, the
  cheap per-checkpoint proxy for the full benchmark.
- ``hard.txt``: every clip of the Table A.8 groups we can identify (``realmimic.eval.motion_groups``),
  where SONIC and early SuperMimic checkpoints fail.
- ``full.txt``: the whole paper-filtered set (the paper's benchmark).

The final numbers are always run on the full set.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from realmimic.eval.motion_groups import HARD_SET_GROUPS, motion_group


def make_sets(names: list[str], n_random: int = 2000, seed: int = 0) -> dict[str, list[str]]:
    names = sorted(names)
    return {
        "rand2000": sorted(random.Random(seed).sample(names, n_random)),
        "hard": [n for n in names if motion_group(n) in HARD_SET_GROUPS],
        "full": names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    with open(args.manifest) as f:
        names = [row["name"] for row in csv.DictReader(f) if row.get("keep", "1") == "1"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for set_name, clips in make_sets(names).items():
        (args.out_dir / f"{set_name}.txt").write_text("\n".join(clips) + "\n")
        print(f"{len(clips):6d} clips  {args.out_dir / set_name}.txt")


if __name__ == "__main__":
    main()
