"""BONES-SEED motion groups of the SuperMimic paper's Table A.8, recovered from clip names.

The paper reports failed motions per group on its 129,514-clip benchmark. Each comment gives the paper's group size
next to ours on the paper-filtered set (``data/npz/bones_seed/manifest_filtered.csv``, 129,491 clips). Group sizes
are ours, by name pattern; the paper's set is 23 clips larger and its own grouping cannot be recovered exactly.
"""

from __future__ import annotations

import re

TABLE_A8_GROUPS: dict[str, str] = {
    "side_rolls": r"^side_roll_",  # paper 22, ours 22
    # paper 44, ours 54: our filter drops one flip take for floating feet and the wall-assisted flips (flip_from_wall,
    # elevated support); the paper's feasibility pass evidently dropped more, which we cannot recover.
    "flips": r"^flip_(090|180|360)",
    "fire_in_the_hole_turns": r"^fire_in_the_hole_",  # paper 20, ours 20
    "crossed_legs": r"^crossed_legs_",  # paper 10, ours 10
    # paper 99, ours 102: stems that describe lying or resting on the floor, or standing up from lying. Our filter
    # drops one actor's dozing/rest takes (ground penetration); faint_R / faint_side_R (collapses that end on the
    # floor without getting up) are left out (with them: 117).
    "lying_resting_standing_up": r"^(dozing_|rest_R_|spread_out_|fetal_position_|stand_up_lying_|faint_stand_up_lying_)",
    "broad_jumps": r"^broad_jump_",  # paper 24, ours 24
    # paper 8, ours 8: the paper's "dance routine" SONIC fails entirely; identified as the only 8-clip dance (one
    # routine, four actors with mirrors) that SONIC's release policy fails entirely in our evaluator.
    "dance_routine": r"^mayahi_",
}
# The 130-clip hard set (data/bench/hard.txt) predates the last two groups and keeps its five.
HARD_SET_GROUPS = ("side_rolls", "flips", "fire_in_the_hole_turns", "crossed_legs", "broad_jumps")
OTHER = "all_other"


def motion_group(name: str) -> str:
    """Table A.8 group of a clip name, or ``OTHER``."""
    for group, pattern in TABLE_A8_GROUPS.items():
        if re.match(pattern, name):
            return group
    return OTHER
