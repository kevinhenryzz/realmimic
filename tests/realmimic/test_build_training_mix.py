"""tools/build_training_mix.py: manifests filter, names stay unique, nothing is copied."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("build_training_mix", Path(__file__).parents[2] / "tools/build_training_mix.py")
mix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mix)


def _dataset(root: Path, names: list[str]) -> Path:
    root.mkdir()
    for n in names:
        (root / f"{n}.npz").write_bytes(b"x")
    return root


def test_mix_respects_manifest_and_links(tmp_path):
    a = _dataset(tmp_path / "a", ["a1", "a2", "a3"])
    b = _dataset(tmp_path / "b", ["b1"])
    manifest = tmp_path / "a.csv"
    manifest.write_text("name,keep\na1,1\na2,0\na3,1\n")
    counts = mix.build_mix(tmp_path / "out", [(a, manifest), (b, None)])
    assert counts == {str(a): 2, str(b): 1}
    out = tmp_path / "out"
    assert sorted(p.stem for p in out.glob("*.npz")) == ["a1", "a3", "b1"]
    assert all(p.is_symlink() for p in out.glob("*.npz"))
    assert (out / "manifest.csv").read_text().splitlines()[0] == "name,keep,source"


def test_duplicate_names_rejected(tmp_path):
    a = _dataset(tmp_path / "a", ["x"])
    b = _dataset(tmp_path / "b", ["x"])
    with pytest.raises(ValueError, match="repeated"):
        mix.build_mix(tmp_path / "out", [(a, None), (b, None)])
