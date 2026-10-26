"""Hand models as lumped rigid inertias of ``<side>_wrist_yaw_link`` (hand + adapter + wrist link).

Fingers are welded: our tracker never commands them, so for whole-body dynamics a hand is one rigid
body. Values are derived with ``tools/derive_hand_inertia.py`` from the cited upstream model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LumpedHand:
    """Inertia of the whole distal link in ``<side>_wrist_yaw_link`` frame (COM-centred)."""

    # Finger pose the values were derived in: SDK actuator positions (0 open, 1 closed) for
    # thumb rotation, thumb flexion, index, middle, ring, pinky.
    pose: tuple[float, float, float, float, float, float]
    mass: float
    com: tuple[float, float, float]
    fullinertia: tuple[float, float, float, float, float, float]  # ixx iyy izz ixy ixz iyz
    # Collision capsules (fromto [m], radius [m]): the palm and fingers replace mjlab's rubber-hand
    # capsule; the thumb gets its own.
    capsule_fromto: tuple[float, float, float, float, float, float]
    capsule_radius: float
    thumb_capsule_fromto: tuple[float, float, float, float, float, float]
    thumb_capsule_radius: float


# Source: unitreerobotics/unitree_ros @ 15ebe53f4cd77e7d745ae83b9d7ff90e685e8998,
# robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf (hand masses, inertias, kinematics and
# mount orientation), derived in the pose below with the hand base moved to the measured mount:
# `tools/derive_hand_inertia.py <urdf> --hand-base-x 0.0675`. Mounted with BrainCo's 26 mm adapter ring
# past the 41.5 mm wrist flange, the hand has its base 67.5 mm along the wrist-yaw link's x axis, not the
# URDF's 59.1 mm (measured on a G1 with calipers; pinky knuckle and fingertip distances cross-check within 1 mm;
# the URDF's palm-inward orientation matches the robot). Open middle fingertip at ~227 mm.
BRAINCO_REVO2: dict[str, LumpedHand] = {
    "left": LumpedHand(
        pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        mass=0.49238282,
        com=(0.10528124, 0.00073126, 0.00801149),
        fullinertia=(3.7661584e-4, 1.6194785e-3, 1.3103827e-3, 1.6177125e-5, 3.0366087e-5, 3.8868197e-5),
        capsule_fromto=(0.0784, 0.0, 0.0, 0.1777, -0.0135, 0.0001),
        capsule_radius=0.04,
        thumb_capsule_fromto=(0.0953, -0.0069, 0.0330, 0.1069, -0.0097, 0.1107),
        thumb_capsule_radius=0.012,
    ),
    "right": LumpedHand(
        pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        mass=0.49238282,
        com=(0.10527855, -0.00050175, 0.00812185),
        fullinertia=(3.8474585e-4, 1.6256175e-3, 1.3122940e-3, -1.5622301e-5, 3.0892655e-5, -4.5282122e-5),
        capsule_fromto=(0.0784, 0.0, 0.0, 0.1777, 0.0135, 0.0001),
        capsule_radius=0.04,
        thumb_capsule_fromto=(0.0953, 0.0081, 0.0339, 0.1069, 0.0117, 0.1115),
        thumb_capsule_radius=0.012,
    ),
}


# Lumped hand data per ``RobotSpec.hands`` value (the stock hand is mjlab's own model).
HAND_MODELS: dict[str, dict[str, LumpedHand]] = {"brainco_revo2": BRAINCO_REVO2}
