"""Figures for the project website, from the benchmark summaries (aggregate counts only; no clip names).

    uv run python tools/web_figures.py site/assets/figures --runs runs/benchmark --web runs/web --data site/assets/results \
        [--end-of-clip <run>="no fix, run A" <run>="no fix, run B" <run>="with the fix"]

Figures are drawn wide (one per row on the page), with a narrow ``*_m.svg`` version for phones (the page picks it below
600 px).
end_of_clip.svg (with ``--end-of-clip``): full-set failures within 3 frames of the clip end (``failed_at_end`` of
``runs/benchmark/<run>/ckpt_XXXXXXX_full.json``, scripts/eval_benchmark.py) against training iterations, for
two runs without the clip-end fix and one with it (run directory = legend label). Log axis of failures + 1.
With ``--web`` (the project's aggregate results, runs/web: results.json, curve.json, group_table.md): writes results.json pruned to
what the page shows (``PUBLIC``) and the motion-type table to ``--data``, and draws failures_by_iteration.svg /
failures_by_hours.svg (full-set failures of the runs in curve.json, the released one first, SONIC's release, the
paper's one-GPU point). curve.json itself is not copied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

END_COLORS = ("#ff3b30", "#ff8a3d", "#3d8bff")  # the two runs without the fix, then the one with it
FONT = "JetBrains Mono, ui-monospace, monospace"


def along(xy: list[tuple[float, float]]) -> list[float]:
    """Each vertex's distance along the polyline as a fraction of its length: when the page's draw-in reaches it."""
    d = [0.0]
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        d.append(d[-1] + math.hypot(x1 - x0, y1 - y0))
    return [v / (d[-1] or 1) for v in d]


def series(runs_dir: Path, run: str) -> list[tuple[int, int]]:
    out = []
    for f in sorted((runs_dir / run).glob("ckpt_*_full.json")):
        s = json.loads(f.read_text())["summary"]
        out.append((int(re.search(r"ckpt_(?:final_)?(\d+)_full", f.name).group(1)), int(s.get("failed_at_end", 0))))
    return out


def end_of_clip_svg(runs_dir: Path, runs: list[tuple[str, str, str]], compact: bool = False) -> tuple[str, dict]:
    """``compact``: a narrow version for phones (legend on top, coarser ticks), text at the same size."""
    data = {run: series(runs_dir, run) for run, _, _ in runs}
    xmax = max(it for pts in data.values() for it, _ in pts)
    xmax = math.ceil(xmax / 8000) * 8000
    ymax = 4  # log10(failures + 1): up to 10,000
    W, H, L, R, T, B = (360, 330, 50, 34, 74, 40) if compact else (1100, 340, 56, 170, 18, 40)
    x = lambda it: L + (W - L - R) * it / xmax  # noqa: E731
    y = lambda v: T + (H - T - B) * (1 - math.log10(v + 1) / ymax)  # noqa: E731
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" font-family="{FONT}" font-size="11">',
          "<title>End-of-clip failures on the full set against training iterations</title>"]  # fmt: skip
    for v, label in ((0, "0"), (9, "10"), (99, "100"), (999, "1,000"), (9999, "10,000")):
        el.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#ffffff" stroke-opacity="0.08"/>')
        el.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#85888f">{label}</text>')
    for it in range(0, xmax + 1, 16000 if compact else 8000):
        el.append(f'<text x="{x(it):.1f}" y="{H - B + 18}" text-anchor="middle" fill="#85888f">{it // 1000}k</text>')
    el.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 4}" text-anchor="middle" fill="#85888f">training iterations</text>')
    for k, (run, label, color) in enumerate(runs):
        pts = data[run]
        if not pts:
            continue
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(it):.1f},{y(v):.1f}" for i, (it, v) in enumerate(pts))
        el.append(f'<path class="series" style="--k:{k}" pathLength="1" d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for (it, v), d in zip(pts, along([(x(it), y(v)) for it, v in pts])):
            el.append(f'<circle class="pt" style="--k:{k};--d:{d:.3f}" cx="{x(it):.1f}" cy="{y(v):.1f}" r="3.2" fill="{color}"/>')
        it, v = pts[-1]
        el.append(f'<text class="end" style="--k:{k}" x="{x(it) + 8:.1f}" y="{y(v) + 4:.1f}" fill="{color}">{v:,}</text>')
        lx, ly = (L - 40, 8 + 20 * k) if compact else (W - R + 34, T + 8 + 22 * k)
        el.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="{color}"/>')
        el.append(f'<text x="{lx + 16}" y="{ly + 9}" fill="#b8b9bd">{label}</text>')
    el.append("</svg>")
    return "\n".join(el) + "\n", data


# ---------------------------------------------------------------- results from runs/web

CURVE_COLORS = ("#ff6a1a", "#8d96a3")  # curve.json's runs in order: the released one, then the comparison run
CURVE_LABELS = ("realmimic, released", "same recipe, alternative reward shape")  # their legend labels, in plain words


def failure_curve_svg(curve: dict, by: str, compact: bool = False) -> str:
    """Full-set failures (log axis) against training iteration (``by="iteration"``) or 4090-hours (``"hours"``)
    for the runs in curve.json, SONIC's release as a dashed line, the paper's one-GPU point (hours only; named in
    the legend, so no label crosses the curves). ``compact``: a narrow version for phones, text at the same size.
    Classes for the page's draw-in (demo/reveal.js): series (pathLength 1), pt (--d: see along), end, ref."""
    rows = len(curve["runs"]) + (by == "hours")  # legend rows: above the plot on phones, else bottom left
    w, h, l, r, t, b = (360, 300 + 18 * rows, 44, 12, 16 + 18 * rows, 44) if compact else (1100, 340, 52, 16, 16, 44)
    xs = [p[by] for run in curve["runs"].values() for p in run["points"]]
    xmax = (math.ceil(max(xs) / 8000) * 8000) if by == "iteration" else math.ceil(max(xs))
    y0, y1 = 0.0, 3.0  # log10 failures: 1 .. 1000
    x = lambda v: l + (w - l - r) * v / xmax  # noqa: E731
    y = lambda v: t + (h - t - b) * (1 - (math.log10(max(v, 1)) - y0) / (y1 - y0))  # noqa: E731
    what = "training iterations" if by == "iteration" else "training hours on one RTX 4090"
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" font-family="{FONT}" font-size="11">',
          f"<title>Full-set failures against {what}</title>"]  # fmt: skip
    for v, label in ((1, "1"), (10, "10"), (100, "100"), (1000, "1,000")):
        el.append(f'<line x1="{l}" x2="{w - r}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#ffffff" stroke-opacity="0.08"/>')
        el.append(f'<text x="{l - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#85888f">{label}</text>')
    step = (32000 if compact else 16000) if by == "iteration" else (4 if compact else 2)
    for v in range(0, xmax + 1, step):
        el.append(f'<text x="{x(v):.1f}" y="{h - b + 18}" text-anchor="middle" fill="#85888f">{f"{v // 1000}k" if by == "iteration" else f"{v} h"}</text>')
    el.append(f'<text x="{(l + w - r) / 2:.1f}" y="{h - 6}" text-anchor="middle" fill="#85888f">{what}</text>')
    s = curve["sonic_release_failed"]
    el.append(f'<line class="ref" x1="{l}" x2="{w - r}" y1="{y(s):.1f}" y2="{y(s):.1f}" stroke="#eceae6" stroke-opacity="0.55" stroke-dasharray="5 4"/>')
    sonic = f"SONIC release: {s}" if compact else f"SONIC release, the same evaluation: {s}"
    el.append(f'<text class="ref" x="{w - r}" y="{y(s) - 6:.1f}" text-anchor="end" fill="#b8b9bd">{sonic}</text>')
    ly0 = 4 if compact else h - b - 10 - 18 * rows  # bottom left on the wide chart: the curves start high on the left
    if by == "hours":
        p = curve["paper_1gpu_failed_as_reported"]
        diamond = lambda cx, cy: f"M{cx - 6:.1f},{cy:.1f} l6,-6 l6,6 l-6,6 z"  # noqa: E731
        el.append(f'<path class="ref" d="{diamond(x(3), y(p))}" fill="none" stroke="#eceae6" stroke-width="1.5"/>')
        ly = ly0 + 18 * len(curve["runs"])
        el.append(f'<path d="M{l + 12:.1f},{ly + 5:.1f} l5,-5 l5,5 l-5,5 z" fill="none" stroke="#eceae6" stroke-width="1.5"/>')
        paper = f"paper: {p} at 3 h (one H100 80 GB)" if compact else f"paper: {p} at 3 h (one H100 80 GB, its own evaluation)"
        el.append(f'<text x="{l + 28}" y="{ly + 9}" fill="#b8b9bd">{paper}</text>')
    for k, (run, color) in enumerate(zip(curve["runs"], CURVE_COLORS)):
        pts = curve["runs"][run].get("points", [])
        if not pts:
            continue
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(p[by]):.1f},{y(p['failed']):.1f}" for i, p in enumerate(pts))
        el.append(f'<path class="series" style="--k:{k}" pathLength="1" d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for p, d in zip(pts, along([(x(p[by]), y(p["failed"])) for p in pts])):
            el.append(f'<circle class="pt" style="--k:{k};--d:{d:.3f}" cx="{x(p[by]):.1f}" cy="{y(p["failed"]):.1f}" r="3" fill="{color}"/>')
        last = pts[-1]
        el.append(f'<text class="end" style="--k:{k}" x="{x(last[by]) - 4:.1f}" y="{y(last["failed"]) + (16 if k == 0 else -8):.1f}" text-anchor="end" fill="{color}">{last["failed"]}</text>')
        ly = ly0 + 18 * k
        el.append(f'<rect x="{l + 12}" y="{ly}" width="10" height="10" rx="2" fill="{color}"/>')
        el.append(f'<text x="{l + 28}" y="{ly + 9}" fill="#b8b9bd">{CURVE_LABELS[k]}</text>')
    el.append("</svg>")
    return "\n".join(el) + "\n"


# The README's motion-type table: its column names, "everything else", and no kernel-tail column.
GROUP_HEADER = {"group": "motion type", "SONIC release": "SONIC release, ours", "realmimic, released": "realmimic"}
GROUP_ROW = {"all other": "everything else"}


def group_table(md: str) -> dict:
    """The markdown group table -> {"header": [...], "rows": [[...], ...]}, as the page shows it."""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip().startswith("|")]
    cells = lambda ln: [c.strip() for c in ln.strip("|").split("|")]  # noqa: E731
    header = cells(lines[0])
    keep = [j for j, h in enumerate(header) if "kernel-tail" not in h]
    rows = [[(GROUP_ROW.get(r[j], r[j]) if j == 0 else r[j]) for j in keep] for r in map(cells, lines[2:])]
    return {"header": [GROUP_HEADER.get(header[j], header[j]) for j in keep], "rows": rows}


# The public results file (site/assets/results/results.json): exactly what demo/results.js reads, nothing else.
PUBLIC = {
    "released": ["failed", "clips", "return", "last_three", "training", "scored_on"],
    "second_seed": ["label", "training", "failed", "return", "last_three", "scored_on"],
    "kernel_tail_run": ["label", "note", "training", "failed", "return", "scored_on"],
    "sonic_release": ["failed", "return", "scored_on"],
    "paper": ["failed", "clips", "sonic_failed"],
    "hardware_note": None,
    "lafan1": ["whole_clips", "segments_10s", "note", "hardware_note"],
    "practice_fix": None,
    "not_measurable": None,
    "compute": ["rows", "total", "not_included"],
}


def public_results(results: dict) -> dict:
    """Keep the keys the page renders (PUBLIC); a missing one is an error, anything else is dropped."""
    out = {}
    for key, fields in PUBLIC.items():
        value = results[key]
        out[key] = value if fields is None else {f: value[f] for f in fields}
    return out


def web_results(web_dir: Path, figures: Path, data: Path) -> None:
    """Copy the project's aggregate results (runs/web) into the site (pruned to what the page shows) and draw their
    figures."""
    data.mkdir(parents=True, exist_ok=True)
    results = json.loads((web_dir / "results.json").read_text())
    curve = json.loads((web_dir / "curve.json").read_text())
    (data / "results.json").write_text(json.dumps(public_results(results), indent=1, ensure_ascii=False) + "\n")
    (data / "groups.json").write_text(json.dumps(group_table((web_dir / "group_table.md").read_text()), indent=1, ensure_ascii=False) + "\n")
    for suffix, compact in (("", False), ("_m", True)):  # _m: the phone versions
        (figures / f"failures_by_iteration{suffix}.svg").write_text(failure_curve_svg(curve, "iteration", compact))
        (figures / f"failures_by_hours{suffix}.svg").write_text(failure_curve_svg(curve, "hours", compact))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument("--runs", default="runs/benchmark", help="benchmark results directory")
    parser.add_argument("--web", help="the project's aggregate results (runs/web)")
    parser.add_argument("--data", default="site/assets/results", help="where the page reads the results")
    parser.add_argument("--end-of-clip", nargs=3, metavar="RUN=LABEL", help="end_of_clip.svg: two runs without the clip-end fix, then one with it")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.end_of_clip:
        runs = [(*spec.split("=", 1), color) for spec, color in zip(args.end_of_clip, END_COLORS)]
        svg, data = end_of_clip_svg(Path(args.runs), runs)
        (out / "end_of_clip.svg").write_text(svg)
        (out / "end_of_clip_m.svg").write_text(end_of_clip_svg(Path(args.runs), runs, compact=True)[0])
        for (run, label, _), pts in zip(runs, data.values()):
            print(f"{label}: " + ", ".join(f"{it // 1000}k {v}" for it, v in pts))
    if args.web:
        web_results(Path(args.web), out, Path(args.data))
        print(f"results: {args.web} -> {args.data} (pruned to what the page shows); figures: failures_by_iteration.svg, failures_by_hours.svg")


if __name__ == "__main__":
    main()
