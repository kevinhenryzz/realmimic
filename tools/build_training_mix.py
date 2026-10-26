"""Assemble a training mix: symlink the kept clips of several npz datasets into one directory and write
a combined manifest (``name,keep,source``) that ``discover_npz`` / ``build_library.py`` accept.

    uv run python tools/build_training_mix.py data/npz/mix_v1 \
        data/npz/bones_seed:data/npz/bones_seed/manifest_hwsafe.csv \
        data/npz/lafan1 data/npz/planner_v1:data/npz/planner_v1/manifest.csv

Each source is ``DIR`` (all clips) or ``DIR:MANIFEST`` (clips with keep == 1). Clip names must be
unique across sources. No data is copied.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def kept_names(directory: Path, manifest: Path | None) -> list[str]:
    names = sorted(p.stem for p in directory.glob("*.npz"))
    if manifest is None:
        return names
    with open(manifest) as f:
        keep = {row["name"] for row in csv.DictReader(f) if row.get("keep", "1") in ("1", "True", "true")}
    return [n for n in names if n in keep]


def build_mix(out: Path, sources: list[tuple[Path, Path | None]]) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=False)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for directory, manifest in sources:
        names = kept_names(directory, manifest)
        dup = seen.intersection(names)
        if dup:
            raise ValueError(f"clip names repeated across sources, e.g. {sorted(dup)[:3]}")
        seen.update(names)
        for name in names:
            (out / f"{name}.npz").symlink_to((directory / f"{name}.npz").resolve())
            rows.append((name, directory.name))
        counts[str(directory)] = len(names)
    with open(out / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "keep", "source"])
        writer.writerows((name, 1, source) for name, source in rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("sources", nargs="+", help="DIR or DIR:MANIFEST")
    args = parser.parse_args()
    sources = []
    for spec in args.sources:
        directory, _, manifest = spec.partition(":")
        sources.append((Path(directory), Path(manifest) if manifest else None))
    for source, n in build_mix(args.out, sources).items():
        print(f"{n:7d} clips  {source}")


if __name__ == "__main__":
    main()
