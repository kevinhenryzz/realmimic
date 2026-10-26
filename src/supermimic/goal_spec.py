"""Goal-layout settings shared by the environment, the imagination and the networks (no heavy imports).

``anchor_obs``: reference anchor pose relative to the robot anchor appended to
the goal: "none" (pure reference goal), "ori" (rot6d, 6, IMU-observable), "pos_ori" (pos 3 + rot6d 6,
needs state estimation). ``goal_features``: per-frame features "v1" (68) or "traj" (77).
"""

from __future__ import annotations

ANCHOR_OBS_DIMS = {"none": 0, "ori": 6, "pos_ori": 9, "pelvis_ori10": 60}
GOAL_FEAT_DIMS = {"v1": 68, "traj": 77, "sonic": 59}
# Per-frame goal feature layout (size, fixed scale divisor) per mode.
GOAL_FEAT_LAYOUTS = {
    "v1": ((29, 1.0), (29, 5.0), (3, 1.0), (3, 2.0), (3, 1.0), (1, 1.0)),
    "traj": ((29, 1.0), (29, 5.0), (3, 1.0), (3, 2.0), (3, 1.0), (1, 1.0), (3, 1.0), (6, 1.0)),
    # SONIC deploy runtime terms (realmimic.obs.sonic): absolute reference joint positions, joint
    # velocities and pelvis height; the anchor term "pelvis_ori10" adds the per-frame reference
    # pelvis orientation relative to the robot pelvis (rot6d, 10 frames).
    "sonic": ((29, 1.0), (29, 5.0), (1, 1.0)),
}
ANCHOR_WINDOW_MODES = ("pelvis_ori10",)  # anchor terms with one rot6d per goal frame


def anchor_mode(value) -> str:
    """Normalise the ``anchor_obs`` setting; old configs stored a bool (True = pos + ori)."""
    if value is True:
        return "pos_ori"
    if value is False or value is None:
        return "none"
    value = str(value)
    if value not in ANCHOR_OBS_DIMS:
        raise ValueError(f"anchor_obs must be one of {tuple(ANCHOR_OBS_DIMS)}, got {value!r}")
    return value


def goal_dim(future_frames: int, goal_features: str = "v1", anchor_obs="none") -> int:
    return future_frames * GOAL_FEAT_DIMS[goal_features] + ANCHOR_OBS_DIMS[anchor_mode(anchor_obs)]
