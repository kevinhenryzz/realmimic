"""Lump a hand subtree of a G1 URDF into one rigid inertia expressed in ``<side>_wrist_yaw_link``.

Used to derive the BrainCo Revo 2 numbers in ``realmimic/robots/hands.py`` from Unitree's official
``g1_with_brainco_hand`` URDF (unitreerobotics/unitree_ros, robots/g1_with_brainco_hand), in a given
finger pose. The pose is the hand's six actuator positions as the BrainCo SDK reports them
(0 = fully open, 1 = fully closed), each mapped linearly onto its URDF joint range; mimic joints follow.

    uv run python tools/derive_hand_inertia.py <path/to/g1_29dof_mode_15_brainco_hand.urdf> \
        [--pose THUMB_ROT,THUMB_FLEX,INDEX,MIDDLE,RING,PINKY] [--hand-base-x METERS]

``--hand-base-x`` moves the hand's base along the wrist-yaw link's x axis (the URDF's
``<side>_base_joint``, 0.0591 m) to a measured mount: our hands sit on a 26 mm adapter ring past the
0.0415 m flange, i.e. 0.0675 m (``revo2_mount.json``, measured on the robot).
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin(el: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if el is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array([float(v) for v in el.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in el.get("rpy", "0 0 0").split()])
    return xyz, _rpy_matrix(rpy)


# Actuated URDF joint per SDK actuator, in SDK order (the rest of each finger is a URDF mimic joint).
ACTUATED = (
    "thumb_metacarpal",
    "thumb_proximal",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
)


def _axis_rotation(axis: np.ndarray, q: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


def joint_angles(robot: ET.Element, side: str, pose: tuple[float, ...]) -> dict[str, float]:
    """URDF joint angles for a normalized six-actuator pose, mimic joints included."""
    joints = {j.get("name"): j for j in robot.findall("joint")}
    q: dict[str, float] = {}
    for name, value in zip(ACTUATED, pose, strict=True):
        joint = joints[f"{side}_{name}_joint"]
        limit = joint.find("limit")
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        q[joint.get("name")] = lo + float(np.clip(value, 0.0, 1.0)) * (hi - lo)
    for name, joint in joints.items():
        mimic = joint.find("mimic")
        if mimic is not None and mimic.get("joint") in q:
            q[name] = float(mimic.get("multiplier", 1)) * q[mimic.get("joint")] + float(
                mimic.get("offset", 0)
            )
    return q


def lump_subtree(urdf_path: str, root_link: str, angles: dict[str, float] | None = None, origins: dict[str, float] | None = None) -> dict:
    """Mass, COM and inertia about the COM of ``root_link`` plus all descendants, in ``root_link``'s
    frame, with joints at ``angles`` (radians, by joint name; missing joints at zero) and the x of the
    named joints' origins replaced by ``origins`` (metres). Also returns every descendant link's origin
    in that frame."""
    angles = angles or {}
    origins = origins or {}
    robot = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    children: dict[str, list[ET.Element]] = {}
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    frames: dict[str, list[float]] = {}

    parts = []  # (mass, com in root frame, inertia about own com in root frame)

    def visit(link_name: str, pos: np.ndarray, rot: np.ndarray) -> None:
        frames[link_name] = pos.tolist()
        inertial = links[link_name].find("inertial")
        if inertial is not None:
            m = float(inertial.find("mass").get("value"))
            c_pos, c_rot = _origin(inertial.find("origin"))
            i = inertial.find("inertia")
            ixx, iyy, izz, ixy, ixz, iyz = (
                float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            )
            inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            r = rot @ c_rot
            parts.append((m, pos + rot @ c_pos, r @ inertia @ r.T))
        for joint in children.get(link_name, []):
            j_pos, j_rot = _origin(joint.find("origin"))
            if joint.get("name") in origins:
                j_pos = np.array([origins[joint.get("name")], j_pos[1], j_pos[2]])
            q = angles.get(joint.get("name"), 0.0)
            if q and joint.get("type") in ("revolute", "continuous"):
                axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                j_rot = j_rot @ _axis_rotation(axis / np.linalg.norm(axis), q)
            visit(joint.find("child").get("link"), pos + rot @ j_pos, rot @ j_rot)

    visit(root_link, np.zeros(3), np.eye(3))
    mass = sum(m for m, _, _ in parts)
    com = sum(m * c for m, c, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in parts:
        d = c - com
        inertia += i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {
        "root_link": root_link,
        "num_bodies": len(parts),
        "mass": mass,
        "com": com.tolist(),
        # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz.
        "fullinertia": [
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2],
        ],
        "link_origins": frames,
    }


PALM_START_PAST_BASE = 0.0109  # palm capsule starts just past the hand base flange
PALM_RADIUS = 0.04  # half the palm width (index to pinky spread)
THUMB_RADIUS = 0.012


def hand_capsules(link_origins: dict[str, list[float]], side: str) -> dict:
    """Collision capsules (fromto, radius) in the wrist-yaw frame for the hand as posed.

    Palm: from the base flange towards the mean fingertip origin, shortened by the radius so the
    capsule ends at the fingertips. Thumb: proximal-joint origin to thumb-tip origin."""

    def origin(part: str) -> np.ndarray:
        (name,) = [n for n in link_origins if n.lower().startswith(f"{side}_{part}")]
        return np.array(link_origins[name])

    palm_start = np.array([link_origins[f"{side}_base_link"][0] + PALM_START_PAST_BASE, 0.0, 0.0])
    tips = np.mean([origin(f"{f}_tip") for f in ("index", "middle", "ring", "pinky")], axis=0)
    direction = (tips - palm_start) / np.linalg.norm(tips - palm_start)
    palm_end = tips - PALM_RADIUS * direction
    return {
        "palm": {"fromto": [*palm_start.tolist(), *palm_end.tolist()], "radius": PALM_RADIUS},
        "thumb": {
            "fromto": [*origin("thumb_proximal").tolist(), *origin("thumb_tip").tolist()],
            "radius": THUMB_RADIUS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("urdf")
    parser.add_argument(
        "--pose", default="0,0,0,0,0,0", help="six SDK actuator positions in [0, 1]"
    )
    parser.add_argument("--hand-base-x", type=float, default=None, help="hand base on the wrist-yaw x axis [m]")
    args = parser.parse_args()
    pose = tuple(float(v) for v in args.pose.split(","))
    robot = ET.parse(args.urdf).getroot()
    out = {}
    for side in ("left", "right"):
        mount = {f"{side}_base_joint": args.hand_base_x} if args.hand_base_x is not None else None
        lumped = lump_subtree(args.urdf, f"{side}_wrist_yaw_link", joint_angles(robot, side, pose), mount)
        out[side] = {
            "pose": pose,
            **lumped,
            "capsules": hand_capsules(lumped["link_origins"], side),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
