"""RobotSpec → MJCF/actuator assembly."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1

from realmimic.robots import ROBOTS, get_robot_spec
from realmimic.robots.spec import PayloadItem, RobotSpec
from realmimic.robots.g1 import articulation, build_mjspec, joint_gains, robot_cfg
from realmimic.robots.hands import BRAINCO_REVO2

STOCK_HAND_MASS = 0.254576  # mjlab g1.xml wrist_yaw_link incl. rubber hand


def _compile(name: str | RobotSpec) -> mujoco.MjModel:
    return build_mjspec(get_robot_spec(name) if isinstance(name, str) else name).compile()


# A synthetic payload and encoder-bias override, exercising the spec fields no built-in robot uses.
_PAYLOAD_SPEC = RobotSpec(
    name="test_payload",
    description="G1 with BrainCo hands and two point masses on the torso (test only).",
    hands="brainco_revo2",
    payloads=(
        PayloadItem("box_a", "torso_link", 0.30, (0.0, 0.0, 0.50), "test"),
        PayloadItem("box_b", "torso_link", 0.10, (0.05, 0.0, 0.45), "test"),
    ),
    encoder_bias_overrides=((r".*_ankle_(pitch|roll)_joint", 0.09),),
)


def _total_mass(model: mujoco.MjModel) -> float:
    return float(model.body_mass.sum())


def test_stock_spec_is_mjlab_g1():
    stock = g1.get_spec().compile()
    ours = _compile("g1")
    np.testing.assert_allclose(ours.body_mass, stock.body_mass)
    np.testing.assert_allclose(ours.body_inertia, stock.body_inertia)


def test_brainco_hand_mass():
    expected = _total_mass(_compile("g1")) + 2 * (BRAINCO_REVO2["left"].mass - STOCK_HAND_MASS)
    assert _total_mass(_compile("g1_brainco")) == pytest.approx(expected, abs=1e-9)


def test_payload_mass_and_com():
    spec = _PAYLOAD_SPEC
    payload = sum(p.mass for p in spec.payloads)
    base, ours = _compile("g1_brainco"), _compile(spec)
    assert _total_mass(ours) == pytest.approx(_total_mass(base) + payload, abs=1e-9)
    torso = mujoco.mj_name2id(ours, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    m0 = base.body_mass[torso]
    com0 = base.body_ipos[torso]  # nonzero iquat in base is irrelevant for the COM
    expected = (m0 * com0 + sum(p.mass * np.array(p.pos) for p in spec.payloads)) / (m0 + payload)
    np.testing.assert_allclose(ours.body_ipos[torso], expected, atol=1e-9)


@pytest.mark.parametrize("name", sorted(ROBOTS))
def test_kinematics_unchanged(name):
    """Only inertials/collision geoms change: body poses match mjlab's G1 for random configurations."""
    stock, ours = g1.get_spec().compile(), _compile(name)
    d0, d1 = mujoco.MjData(stock), mujoco.MjData(ours)
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = stock.qpos0.copy()
        lo, hi = stock.jnt_range[1:, 0], stock.jnt_range[1:, 1]
        q[7:] = rng.uniform(lo, hi)
        d0.qpos[:], d1.qpos[:] = q, q
        mujoco.mj_kinematics(stock, d0)
        mujoco.mj_kinematics(ours, d1)
        np.testing.assert_allclose(d1.xpos, d0.xpos, atol=1e-12)
        np.testing.assert_allclose(d1.xquat, d0.xquat, atol=1e-12)


# SONIC's compile-time motor assignment (gear_sonic_deploy policy_parameters.hpp @ 7f151314).
SONIC_MOTOR = {
    "hip_pitch": "7520_22", "hip_roll": "7520_22", "knee": "7520_22", "hip_yaw": "7520_14",
    "waist_yaw": "7520_14", "ankle": "5020x2", "waist_roll": "5020x2", "waist_pitch": "5020x2",
    "shoulder": "5020", "elbow": "5020", "wrist_roll": "5020", "wrist_pitch": "4010", "wrist_yaw": "4010",
}
STIFFNESS = {
    "7520_22": g1.STIFFNESS_7520_22, "7520_14": g1.STIFFNESS_7520_14, "5020": g1.STIFFNESS_5020,
    "5020x2": 2 * g1.STIFFNESS_5020, "4010": g1.STIFFNESS_4010,
}


def _sonic_kp(joint: str) -> float:
    for key, motor in SONIC_MOTOR.items():
        if key in joint:
            return STIFFNESS[motor]
    raise KeyError(joint)


def test_gains_match_sonic_except_hip_pitch():
    model = _compile("g1")
    names = [model.joint(i).name for i in range(1, model.njnt)]
    kp, _, _ = joint_gains(get_robot_spec("g1"), names)
    for name, k in zip(names, kp):
        expected = g1.STIFFNESS_7520_14 if "hip_pitch" in name else _sonic_kp(name)
        assert k == pytest.approx(expected), name


def test_hip_pitch_7520_22_variant():
    import dataclasses

    spec = dataclasses.replace(get_robot_spec("g1"), name="g1_new_hip", hip_pitch_motor="7520_22")
    model = _compile("g1")
    names = [model.joint(i).name for i in range(1, model.njnt)]
    kp, _, scale = joint_gains(spec, names)
    for name, k in zip(names, kp):
        assert k == pytest.approx(_sonic_kp(name)), name
    hp = names.index("left_hip_pitch_joint")
    assert scale[hp] == pytest.approx(0.25 * 139.0 / g1.STIFFNESS_7520_22)
    assert len(articulation(spec).actuators) == len(g1.G1_ARTICULATION.actuators)


def test_content_hash_distinguishes_specs():
    hashes = {spec.content_hash() for spec in ROBOTS.values()}
    assert len(hashes) == len(ROBOTS)
    assert get_robot_spec("g1").content_hash() == get_robot_spec("g1").content_hash()


@pytest.mark.parametrize("name", sorted(ROBOTS))
def test_mjlab_entity_builds(name):
    from mjlab.entity.entity import Entity

    entity = Entity(robot_cfg(get_robot_spec(name)))
    entity.spec.compile()


def test_hand_inertia_tensor_preserved():
    """The principal-axes encoding written into the spec reproduces the lumped tensor exactly."""
    model = _compile("g1_brainco")
    for side in ("left", "right"):
        hand = BRAINCO_REVO2[side]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_iquat[bid])
        rot = rot.reshape(3, 3)
        tensor = rot @ np.diag(model.body_inertia[bid]) @ rot.T
        ixx, iyy, izz, ixy, ixz, iyz = hand.fullinertia
        expected = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        np.testing.assert_allclose(tensor, expected, atol=1e-10)
        np.testing.assert_allclose(model.body_ipos[bid], hand.com, atol=1e-12)


def test_encoder_bias_ranges_and_env_groups():
    from realmimic.robots import encoder_bias_half_ranges
    from supermimic.envs.tracking_env_cfg import encoder_bias_groups

    model = _compile("g1")
    names = [model.joint(i).name for i in range(1, model.njnt)]
    spec = _PAYLOAD_SPEC
    half = dict(zip(names, encoder_bias_half_ranges(spec, names)))
    assert half["left_ankle_roll_joint"] == 0.09 and half["right_ankle_pitch_joint"] == 0.09
    assert half["left_knee_joint"] == 0.01
    import re

    groups = encoder_bias_groups(spec)
    for name in names:
        hits = [g for g, (pats, _) in groups.items() if any(re.fullmatch(p, name) for p in pats)]
        assert len(hits) == 1, (name, hits)  # every joint randomized by exactly one event term
        assert groups[hits[0]][1] == half[name]
    assert set(encoder_bias_half_ranges(get_robot_spec("g1"), names)) == {0.01}


def test_brainco_thumb_capsule_in_collision_set():
    from mjlab.entity.entity import Entity

    model = Entity(robot_cfg(get_robot_spec("g1_brainco"))).spec.compile()
    for side in ("left", "right"):
        palm, thumb = model.geom(f"{side}_hand_collision"), model.geom(f"{side}_thumb_collision")
        assert thumb.bodyid == palm.bodyid
        assert (thumb.contype, thumb.conaffinity, thumb.condim) == (palm.contype, palm.conaffinity, palm.condim)
        hand = BRAINCO_REVO2[side]
        a, b = np.array(hand.thumb_capsule_fromto[:3]), np.array(hand.thumb_capsule_fromto[3:])
        np.testing.assert_allclose(thumb.pos, (a + b) / 2, atol=1e-9)
        assert thumb.size[0] == hand.thumb_capsule_radius
    stock = _compile("g1")
    assert mujoco.mj_name2id(stock, mujoco.mjtObj.mjOBJ_GEOM, "left_thumb_collision") == -1


def test_content_hash_covers_hand_model(monkeypatch):
    import dataclasses

    from realmimic.robots import hands

    spec = get_robot_spec("g1_brainco")
    before = spec.content_hash()
    moved = dataclasses.replace(BRAINCO_REVO2["left"], thumb_capsule_radius=0.02)
    monkeypatch.setitem(hands.HAND_MODELS, "brainco_revo2", {**BRAINCO_REVO2, "left": moved})
    assert spec.content_hash() != before
