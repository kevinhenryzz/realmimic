"""RobotSpec: the single description of one physical G1 configuration.

Everything robot-specific that differs between G1 units lives here: hip-pitch motor revision, hands,
rigid payloads and known calibration uncertainty. ``realmimic.robots.g1`` turns a spec into the
simulation model and actuator config; the export step turns it into deploy constants. Its
``content_hash`` is stamped into every downstream artifact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal

HipPitchMotor = Literal["7520_14", "7520_22"]
HandType = Literal["stock", "brainco_revo2"]


@dataclass(frozen=True)
class PayloadItem:
    """A rigid point mass fixed to a body (sensors, brackets). Position in the body frame [m]."""

    name: str
    body: str
    mass: float
    pos: tuple[float, float, float]
    source: str  # where the mass/position come from ("spec sheet", "measured", "estimate: ...")


@dataclass(frozen=True)
class RobotSpec:
    name: str
    description: str
    # G1 29-DoF rev 1.0 shipped with 7520-14 hip-pitch motors; later units use 7520-22. The robot
    # reports its variant in LowState.mode_machine (5 == g1_29dof_rev_1_0).
    hip_pitch_motor: HipPitchMotor = "7520_14"
    hands: HandType = "stock"
    payloads: tuple[PayloadItem, ...] = ()
    # Encoder-bias randomization half-range [rad]: default for all joints, overrides by regex.
    encoder_bias_default: float = 0.01
    encoder_bias_overrides: tuple[tuple[str, float], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def content_hash(self) -> str:
        """Stable short hash of every field and of the hand model data they select (sorted JSON)."""
        from realmimic.robots.hands import HAND_MODELS

        content = {**dataclasses.asdict(self), "hand_model": HAND_MODELS.get(self.hands)}
        blob = json.dumps(content, sort_keys=True, separators=(",", ":"), default=dataclasses.asdict)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


ROBOTS: dict[str, RobotSpec] = {
    spec.name: spec
    for spec in (
        RobotSpec(
            name="g1",
            description="Stock G1 29-DoF rev 1.0 as modelled by mjlab (rubber hands, 7520-14 hip pitch).",
        ),
        RobotSpec(
            name="g1_brainco",
            description="G1 29-DoF rev 1.0 with BrainCo Revo 2 hands.",
            hands="brainco_revo2",
        ),
    )
}


def get_robot_spec(name: str) -> RobotSpec:
    try:
        return ROBOTS[name]
    except KeyError:
        raise KeyError(f"unknown robot {name!r}; known: {sorted(ROBOTS)}") from None


def encoder_bias_half_ranges(spec: RobotSpec, joint_names: list[str]) -> list[float]:
    """Encoder-bias half-range per joint (first matching override wins, else the default)."""
    out = []
    for name in joint_names:
        half = spec.encoder_bias_default
        for pattern, value in spec.encoder_bias_overrides:
            if re.fullmatch(pattern, name):
                half = value
                break
        out.append(half)
    return out
