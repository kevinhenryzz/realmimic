"""Build the mjlab robot config (MJCF + actuators) for a :class:`RobotSpec`.

Starts from mjlab's G1 (rev 1.0 body, BeyondMimic-style actuators, ``KNEES_BENT_KEYFRAME`` default
pose — identical to SONIC's ``default_angles``) and applies the spec. Only inertials and collision
geoms change; the kinematic tree is untouched, so motion data and forward kinematics built from
mjlab's ``g1.xml`` stay valid.
"""

from __future__ import annotations

import dataclasses
import re
from functools import partial
from typing import Literal

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from realmimic.obs import sonic
from realmimic.robots.hands import BRAINCO_REVO2
from realmimic.robots.spec import RobotSpec

SIDES = ("left", "right")


def _body_inertia_matrix(body: mujoco.MjsBody) -> np.ndarray:
    """Inertia tensor about the body COM, in the body frame."""
    if not np.isnan(body.fullinertia[0]):
        ixx, iyy, izz, ixy, ixz, iyz = body.fullinertia
        return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(body.iquat, dtype=float))
    rot = rot.reshape(3, 3)
    return rot @ np.diag(body.inertia) @ rot.T


def _set_full_inertia(body: mujoco.MjsBody, mass: float, com: np.ndarray, inertia: np.ndarray) -> None:
    """Set a COM-centred inertia tensor (body frame) as principal moments + principal-axes quat."""
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, axes.flatten())
    body.mass = float(mass)
    body.ipos = np.asarray(com, dtype=float)
    body.iquat = quat
    body.inertia = moments
    body.fullinertia = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
    body.explicitinertial = True


def add_point_masses(body: mujoco.MjsBody, items: list[tuple[float, np.ndarray]]) -> None:
    """Rigidly add point masses ``(mass, pos_in_body_frame)`` to ``body``'s inertial."""
    m0, c0, i0 = body.mass, np.asarray(body.ipos, dtype=float), _body_inertia_matrix(body)
    parts = [(m0, c0, i0)] + [(m, np.asarray(p, dtype=float), np.zeros((3, 3))) for m, p in items]
    mass = sum(m for m, _, _ in parts)
    com = sum(m * c for m, c, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in parts:
        d = c - com
        inertia += i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    _set_full_inertia(body, mass, com, inertia)


def build_mjspec(robot: RobotSpec) -> mujoco.MjSpec:
    spec = g1.get_spec()
    if robot.hands == "brainco_revo2":
        for side in SIDES:
            hand = BRAINCO_REVO2[side]
            ixx, iyy, izz, ixy, ixz, iyz = hand.fullinertia
            inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            _set_full_inertia(spec.body(f"{side}_wrist_yaw_link"), hand.mass, np.array(hand.com), inertia)
            geom = spec.geom(f"{side}_hand_collision")
            geom.fromto = np.array(hand.capsule_fromto, dtype=float)
            geom.size = np.array([hand.capsule_radius, 0.0, 0.0])
            body = spec.body(f"{side}_wrist_yaw_link")
            # Same class and contact settings as the palm capsule; the name puts it in mjlab's
            # ``.*_collision`` collision sets.
            thumb = body.add_geom(name=f"{side}_thumb_collision", default=geom.classname)
            for attr in ("type", "group", "contype", "conaffinity", "condim", "friction", "rgba"):
                setattr(thumb, attr, getattr(geom, attr))
            thumb.fromto = np.array(hand.thumb_capsule_fromto, dtype=float)
            thumb.size = np.array([hand.thumb_capsule_radius, 0.0, 0.0])
            # Drop the rubber-hand visual mesh (it is not this hand).
            for g in [g for g in body.geoms if g.meshname == f"{side}_rubber_hand"]:
                spec.delete(g)
    elif robot.hands != "stock":
        raise ValueError(f"unsupported hands {robot.hands!r}")
    by_body: dict[str, list[tuple[float, np.ndarray]]] = {}
    for item in robot.payloads:
        by_body.setdefault(item.body, []).append((item.mass, np.array(item.pos)))
    for body_name, items in by_body.items():
        add_point_masses(spec.body(body_name), items)
    return spec


def articulation(robot: RobotSpec) -> EntityArticulationInfoCfg:
    """mjlab's G1 actuators, with the hip-pitch joints moved to the spec's motor type."""
    if robot.hip_pitch_motor == "7520_14":
        return g1.G1_ARTICULATION
    hip_pitch = ".*_hip_pitch_joint"
    actuators = []
    for act in g1.G1_ARTICULATION.actuators:
        names = tuple(n for n in act.target_names_expr if n != hip_pitch)
        if act is g1.G1_ACTUATOR_7520_22:
            names = names + (hip_pitch,)
        if names:
            actuators.append(dataclasses.replace(act, target_names_expr=names))
    return dataclasses.replace(g1.G1_ARTICULATION, actuators=tuple(actuators))


def sonic_gains_articulation(robot: RobotSpec) -> EntityArticulationInfoCfg:
    """The spec's actuators (effort limits, armature) with the PD gains SONIC's runtime commands,
    ``realmimic.obs.sonic.KPS``/``KDS`` (hip pitch: 7520-22 gains on every robot), one per joint."""
    base = articulation(robot)
    actuators = []
    for name, kp, kd in zip(sonic.JOINT_NAMES_MUJOCO, sonic.KPS, sonic.KDS, strict=True):
        act = next(
            a for a in base.actuators if any(re.fullmatch(p, name) for p in a.target_names_expr)
        )
        actuators.append(
            dataclasses.replace(act, target_names_expr=(name,), stiffness=kp, damping=kd)
        )
    return dataclasses.replace(base, actuators=tuple(actuators))


def robot_cfg(
    robot: RobotSpec, collisions=None, gains: Literal["spec", "sonic"] = "spec"
) -> EntityCfg:
    """``gains``: the spec's motor gains (training) or SONIC's (``sonic_gains_articulation``)."""
    return EntityCfg(
        init_state=g1.KNEES_BENT_KEYFRAME,
        collisions=collisions if collisions is not None else (g1.FULL_COLLISION,),
        spec_fn=partial(build_mjspec, robot),
        articulation=articulation(robot) if gains == "spec" else sonic_gains_articulation(robot),
    )


def action_scale(robot: RobotSpec) -> dict[str, float]:
    """0.25 × effort_limit / kp per joint-name pattern (BeyondMimic/SONIC convention)."""
    scale: dict[str, float] = {}
    for act in articulation(robot).actuators:
        assert isinstance(act, BuiltinPositionActuatorCfg) and act.effort_limit is not None
        for name in act.target_names_expr:
            scale[name] = 0.25 * act.effort_limit / act.stiffness
    return scale


def joint_gains(robot: RobotSpec, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-joint (kp, kd, action_scale) in the order of ``joint_names``."""
    kp, kd, sc = np.zeros(len(joint_names)), np.zeros(len(joint_names)), np.zeros(len(joint_names))
    found = np.zeros(len(joint_names), dtype=bool)
    for act in articulation(robot).actuators:
        for pattern in act.target_names_expr:
            for i, name in enumerate(joint_names):
                if re.fullmatch(pattern, name):
                    if found[i]:
                        raise ValueError(f"joint {name} matched by two actuators")
                    kp[i], kd[i] = act.stiffness, act.damping
                    sc[i] = 0.25 * act.effort_limit / act.stiffness
                    found[i] = True
    if not found.all():
        raise ValueError(f"joints without actuator: {[n for n, f in zip(joint_names, found) if not f]}")
    return kp, kd, sc
