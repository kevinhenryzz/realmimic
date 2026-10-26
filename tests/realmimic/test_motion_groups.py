"""Table A.8 group matching and the evaluation clip lists."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from realmimic.eval.motion_groups import OTHER, motion_group

_spec = importlib.util.spec_from_file_location(
    "make_benchmark_sets", Path(__file__).parents[2] / "tools/make_benchmark_sets.py"
)
sets_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sets_tool)


def test_groups_match_prefixes_only():
    assert motion_group("side_roll_R_999__A000") == "side_rolls"
    assert motion_group("flip_360_synthetic_999__A000_M") == "flips"
    assert motion_group("broad_jump_999__A000") == "broad_jumps"
    assert motion_group("dance_synthetic_R_999__A000") == OTHER
    assert motion_group("crossed_arms_idle_R_999__A000") == OTHER
    assert motion_group("stand_up_lying_side_R_999__A000") == "lying_resting_standing_up"
    assert motion_group("faint_stand_up_lying_flat_R_999__A000_M") == "lying_resting_standing_up"
    assert motion_group("faint_R_999__A000") == OTHER  # collapses without getting up are left out
    assert motion_group("mayahi_999__A000_M") == "dance_routine"


def test_sets_are_deterministic_and_hard_set_complete():
    names = [f"walk_{i:04d}__A000" for i in range(5000)] + ["flip_090_999__A000", "side_roll_R_999__A000"]
    a, b = sets_tool.make_sets(names), sets_tool.make_sets(list(reversed(names)))
    assert a == b
    assert len(a["rand2000"]) == 2000 and len(set(a["rand2000"])) == 2000
    assert a["hard"] == ["flip_090_999__A000", "side_roll_R_999__A000"]
    # The hard set keeps its original five groups; later groups do not enter it.
    assert sets_tool.make_sets([*names, "dozing_cross_legged_R_999__A000", "mayahi_999__A000"])["hard"] == a["hard"]
    assert a["full"] == sorted(names)
