"""Build the hardware-safe BONES-SEED subset (realmimic.data.hw_safe) from the Appendix-E set.

Inputs are the converted 50 Hz npz clips, the Appendix-E manifest written by
``scripts/data/filter_bones_seed.py`` and the BONES-SEED metadata parquet. Writes, next to the
npz clips unless ``--out-dir`` is given:

* ``manifest_<tag>.csv``: ``name,keep`` for every clip (``DataCfg.manifest`` /
  ``scripts/data/build_library.py --manifest``);
* ``tags_<tag>.csv``: category, start/end posture, duration, exclusion reasons and metrics;
* ``summary_<tag>.md``: counts and hours per category and per exclusion reason.

  env -u PYTHONPATH uv run python tools/build_hw_safe_manifest.py \
      --npz-dir data/npz/bones_seed \
      --metadata data/raw/bones_seed/metadata/seed_metadata_v004.parquet --workers 16
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from tqdm import tqdm

from realmimic.data import hw_safe

_GEOMS: tuple[hw_safe.CollisionGeom, ...] = ()
_TH = hw_safe.Thresholds()


def _init(th: hw_safe.Thresholds) -> None:
    global _GEOMS, _TH
    _GEOMS = hw_safe.g1_collision_geoms()
    _TH = th


def _kinematics(path: str) -> tuple[str, dict | None]:
    name = Path(path).stem
    try:
        with np.load(path) as z:
            clip = {k: z[k] for k in z.files}
        return name, hw_safe.compute_kinematics(clip, _GEOMS, _TH).as_dict()
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {path}: {e!r}")
        return name, None


def _compute_all(paths: list[str], th: hw_safe.Thresholds, workers: int) -> pd.DataFrame:
    with mp.Pool(workers, initializer=_init, initargs=(th,)) as pool:
        rows = list(tqdm(pool.imap_unordered(_kinematics, paths, chunksize=32), total=len(paths)))
    ok = {n: k for n, k in rows if k is not None}
    return pd.DataFrame.from_dict(ok, orient="index").rename_axis("name")


def _load_metrics(
    paths: list[str], th: hw_safe.Thresholds, workers: int, cache: Path
) -> pd.DataFrame:
    """Kinematics for ``paths``, reusing ``cache`` when computed with the same metric settings."""
    settings = {f: getattr(th, f) for f in hw_safe.METRIC_SETTINGS}
    settings["fields"] = [f.name for f in fields(hw_safe.ClipKinematics)]
    sig = json.dumps(settings, sort_keys=True)
    sig_path = cache.with_suffix(".thresholds.json")
    if cache.exists() and sig_path.exists() and sig_path.read_text() == sig:
        df = pd.read_parquet(cache)
        missing = [p for p in paths if Path(p).stem not in df.index]
        if not missing:
            return df
        df = pd.concat([df, _compute_all(missing, th, workers)])
    else:
        df = _compute_all(paths, th, workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    sig_path.write_text(sig)
    return df


def _summary(tags: pd.DataFrame, th: hw_safe.Thresholds, keep_categories: tuple[str, ...]) -> str:
    kept = tags[tags.keep == 1]
    lines = ["# Hardware-safe BONES-SEED subset", ""]
    lines.append(
        f"Input clips: {len(tags)} (Appendix-E kept: {int((tags.appendix_e == '').sum())}). "
        f"**Kept: {len(kept)} clips, {kept.duration_s.sum() / 3600:.2f} h.**"
    )
    lines += ["", f"Kept categories: {', '.join(keep_categories)}", "", "## Per category", ""]
    lines.append("| category | candidates | candidate h | kept | kept h |")
    lines.append("|---|---:|---:|---:|---:|")
    cand = tags[tags.appendix_e == ""]
    for cat in hw_safe.ALL_CATEGORIES:
        c, k = cand[cand.category == cat], kept[kept.category == cat]
        lines.append(
            f"| {cat} | {len(c)} | {c.duration_s.sum() / 3600:.2f} | {len(k)} "
            f"| {k.duration_s.sum() / 3600:.2f} |"
        )
    lines += ["", "## Start -> end posture of kept clips", ""]
    pt = pd.crosstab(kept.start_posture, kept.end_posture)
    lines.append("| start \\ end | " + " | ".join(pt.columns) + " |")
    lines.append("|---|" + "---:|" * len(pt.columns))
    for idx, row in pt.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Exclusion reasons (Appendix-E candidates in kept categories)", ""]
    lines.append("First failing reason, then every reason a clip fails (a clip can fail several).")
    lines += ["", "| reason | first (clips) | first (h) | any (clips) |", "|---|---:|---:|---:|"]
    pool = cand[cand.category.isin(keep_categories) & (cand.keep == 0)]
    first = pool.reason.str.split(";").str[0]
    anyc = Counter(r for rs in pool.reason for r in rs.split(";"))
    groups = first.str.replace(r"^(meta:[a-z]+):.*$", r"\1", regex=True)
    for g in sorted(set(groups) | {r for r in anyc if r.startswith("kin:")}):
        m = groups == g
        n_any = sum(v for r, v in anyc.items() if r == g or r.startswith(g + ":"))
        lines.append(f"| {g} | {int(m.sum())} | {pool.duration_s[m].sum() / 3600:.2f} | {n_any} |")
    lines += ["", "## Thresholds", "", "```json", json.dumps(asdict(th), indent=2), "```", ""]
    return "\n".join(lines)


def main(
    npz_dir: Path,
    metadata: Path,
    filtered_manifest: Path | None = None,
    out_dir: Path | None = None,
    tag: str = "hwsafe",
    workers: int = 16,
    include: tuple[str, ...] = hw_safe.KEEP_CATEGORIES,
    metrics_cache: Path | None = None,
) -> None:
    filtered_manifest = filtered_manifest or npz_dir / "manifest_filtered.csv"
    out_dir = out_dir or npz_dir
    metrics_cache = metrics_cache or out_dir / f"metrics_{tag}.parquet"
    unknown = set(include) - set(hw_safe.ALL_CATEGORIES)
    if unknown:
        raise SystemExit(f"unknown categories {sorted(unknown)}; known: {hw_safe.ALL_CATEGORIES}")
    th = hw_safe.Thresholds()

    filt = pd.read_csv(filtered_manifest, dtype={"name": str}, keep_default_na=False)
    meta = pd.read_parquet(metadata).set_index("filename")
    paths = [str(npz_dir / f"{n}.npz") for n in filt.name]
    print(f"{len(filt)} clips in {filtered_manifest}, computing kinematics ({workers} workers)")
    kin = _load_metrics(paths, th, workers, metrics_cache)
    kin_fields = [f.name for f in fields(hw_safe.ClipKinematics)]

    rows = []
    for name, keep_e, reason_e in zip(filt.name, filt.keep, filt.reason, strict=True):
        row = meta.loc[name].to_dict() | {"filename": name} if name in meta.index else {}
        if row:
            category, meta_reason = hw_safe.classify_metadata(hw_safe.ClipMeta.from_row(row))
        else:
            category, meta_reason = "other", "meta:missing"
        k = None
        if name in kin.index:
            k = hw_safe.ClipKinematics(**{f: kin.at[name, f] for f in kin_fields})
        category = hw_safe.refine_category(category, k, th)
        reasons = [] if int(keep_e) == 1 else [f"appendix_e:{reason_e}"]
        reasons += hw_safe.decide(category, meta_reason, k, th, include)
        rec = {
            "name": name,
            "category": category,
            "start_posture": k.start_posture if k else "",
            "end_posture": k.end_posture if k else "",
            "duration_s": round(k.duration_s, 3) if k else 0.0,
            "keep": int(not reasons),
            "reason": ";".join(reasons),
            "appendix_e": "" if int(keep_e) == 1 else reason_e,
        }
        if k:
            rec |= {f: getattr(k, f) for f in kin_fields if f not in rec}
        rows.append(rec)
    tags = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags[["name", "keep"]].to_csv(out_dir / f"manifest_{tag}.csv", index=False)
    tags.to_csv(
        out_dir / f"tags_{tag}.csv", index=False, float_format="%.4f", quoting=csv.QUOTE_MINIMAL
    )
    summary = _summary(tags, th, include)
    (out_dir / f"summary_{tag}.md").write_text(summary)
    print(summary)
    print(f"saved {out_dir}/manifest_{tag}.csv, tags_{tag}.csv, summary_{tag}.md")


if __name__ == "__main__":
    tyro.cli(main)
