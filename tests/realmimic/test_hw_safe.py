"""Hardware-safe subset filters (realmimic.data.hw_safe) on small synthetic clips (CPU, no data)."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

from realmimic.data import hw_safe

FPS = 50.0


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    import mjlab

    xml = Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    return mujoco.MjModel.from_xml_path(str(xml))


@pytest.fixture(scope="module")
def geoms() -> tuple[hw_safe.CollisionGeom, ...]:
    return hw_safe.g1_collision_geoms()


def contact_geoms(geoms):
    """The geoms the contact checks look at (as in ``compute_kinematics``)."""
    return [g for g in geoms if g.name not in hw_safe.CONTACT_IGNORED_GEOMS]


def _names(model: mujoco.MjModel, obj: mujoco.mjtObj, n: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj, i) for i in range(1, n)]  # skip world / free joint


def body_names(model: mujoco.MjModel) -> list[str]:
    return _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)


def joint_names(model: mujoco.MjModel) -> list[str]:
    return _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)


def make_clip(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    T: int = 100,
    clearance: float = 0.005,
    support: tuple[str, ...] = hw_safe.FOOT_BODIES,
):
    """A static ``T``-frame clip at one G1 configuration, its lowest ``support`` point ``clearance``
    above the ground. Tests then edit the arrays to create the behaviour under test."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    pos, quat = data.xpos[1:].copy(), data.xquat[1:].copy()
    names = body_names(model)
    geoms = contact_geoms(hw_safe.g1_collision_geoms())
    low = hw_safe.lowest_point(pos[None], quat[None], names, geoms, support)[0]
    pos[:, 2] -= low - clearance
    nb, nj = pos.shape[0], model.nq - 7
    return {
        "fps": np.array([FPS], dtype=np.float32),
        "body_names": np.array(names),
        "joint_names": np.array(joint_names(model)),
        "body_pos_w": np.repeat(pos[None], T, 0),
        "body_quat_w": np.repeat(quat[None], T, 0),
        "body_lin_vel_w": np.zeros((T, nb, 3)),
        "body_ang_vel_w": np.zeros((T, nb, 3)),
        "joint_pos": np.repeat(qpos[None, 7:], T, 0),
        "joint_vel": np.zeros((T, nj)),
    }


def stand_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = np.zeros(model.nq)
    q[2], q[3] = 0.8, 1.0
    return q


def squat_qpos(model: mujoco.MjModel, knee: float = 2.0) -> np.ndarray:
    """Feet flat, shins and thighs folded symmetrically, pelvis upright."""
    q = stand_qpos(model)
    j = {n: 7 + i for i, n in enumerate(joint_names(model))}
    for side in ("left", "right"):
        q[j[f"{side}_hip_pitch_joint"]] = -knee / 2
        q[j[f"{side}_knee_joint"]] = knee
        q[j[f"{side}_ankle_pitch_joint"]] = -knee / 2
    return q


def kneel_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Upright double kneel: knees folded 90 deg, thighs vertical, shins on the ground."""
    q = stand_qpos(model)
    j = {n: 7 + i for i, n in enumerate(joint_names(model))}
    for side in ("left", "right"):
        q[j[f"{side}_knee_joint"]] = np.pi / 2
        q[j[f"{side}_ankle_pitch_joint"]] = 0.5
    return q


def bi(clip: dict, name: str) -> int:
    return list(clip["body_names"]).index(name)


def ji(clip: dict, name: str) -> int:
    return list(clip["joint_names"]).index(name)


def kin(clip: dict, geoms) -> hw_safe.ClipKinematics:
    return hw_safe.compute_kinematics(clip, geoms)


def reasons(clip: dict, geoms, category: str = "walk") -> list[str]:
    return hw_safe.kinematic_reasons(kin(clip, geoms), category)


def drop_to_ground(clip: dict, geoms, bodies: tuple[str, ...], frames: slice, depth=0.0) -> None:
    """Translate ``bodies`` down so their lowest collision point is at ``depth`` in ``frames``."""
    names = list(clip["body_names"])
    for b in bodies:
        k = names.index(b)
        low = hw_safe.lowest_point(
            clip["body_pos_w"][frames],
            clip["body_quat_w"][frames],
            names,
            contact_geoms(geoms),
            (b,),
        )
        clip["body_pos_w"][frames, k, 2] -= low - depth


# --------------------------------------------------------------------------- robot constants


def test_velocity_limits_match_mjlab_actuators(model):
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as c

    motor_of = [
        (c.G1_ACTUATOR_5020, c.ACTUATOR_5020),
        (c.G1_ACTUATOR_7520_14, c.ACTUATOR_7520_14),
        (c.G1_ACTUATOR_7520_22, c.ACTUATOR_7520_22),
        (c.G1_ACTUATOR_4010, c.ACTUATOR_4010),
        (c.G1_ACTUATOR_WAIST, c.ACTUATOR_5020),  # two 5020s through a nominal 1:1 linkage
        (c.G1_ACTUATOR_ANKLE, c.ACTUATOR_5020),
    ]
    assert len(motor_of) == len(c.G1_ARTICULATION.actuators)
    names = joint_names(model)
    limits = dict(zip(names, hw_safe.joint_velocity_limits(names), strict=True))
    covered = set()
    for cfg, motor in motor_of:
        for expr in cfg.target_names_expr:
            for n in names:
                if re.fullmatch(expr, n):
                    assert limits[n] == motor.velocity_limit, n
                    covered.add(n)
    assert covered == set(names)


def test_lowest_point_matches_mujoco(model, geoms):
    rng = np.random.default_rng(0)
    q = stand_qpos(model)
    q[3:7] = rng.normal(size=4)
    q[3:7] /= np.linalg.norm(q[3:7])
    q[7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    names = body_names(model)
    for body in ("left_knee_link", "right_wrist_yaw_link", "torso_link", "left_ankle_roll_link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        ref = np.inf
        for g in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if model.geom_bodyid[g] != bid or not gname.endswith("_collision"):
                continue
            capsule = model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
            half = model.geom_size[g, 1] if capsule else 0.0
            axis_z = data.geom_xmat[g].reshape(3, 3)[2, 2]
            ref = min(ref, data.geom_xpos[g, 2] - half * abs(axis_z) - model.geom_size[g, 0])
        got = hw_safe.lowest_point(
            data.xpos[None, 1:], data.xquat[None, 1:], names, geoms, (body,)
        )[0]
        assert got == pytest.approx(ref, abs=1e-9)


def test_moving_average_and_longest_run():
    x = np.zeros(11)
    x[5] = 5.0
    np.testing.assert_allclose(hw_safe.moving_average(x, 5)[3:8], 1.0)
    assert hw_safe.moving_average(x, 5).shape == x.shape
    assert hw_safe.longest_run(np.array([1, 1, 0, 1, 1, 1, 0], bool)) == 3
    assert hw_safe.longest_run(np.zeros(4, bool)) == 0


# --------------------------------------------------------------------------- kinematic rules


def test_standing_clip_passes_every_check(model, geoms):
    k = kin(make_clip(model, stand_qpos(model)), geoms)
    assert hw_safe.kinematic_reasons(k, "idle") == []
    assert (k.start_posture, k.end_posture) == ("standing", "standing")
    assert k.duration_s == pytest.approx(99 / FPS)


@pytest.mark.parametrize(
    ("speed", "failed"),
    [(1.5, False), (hw_safe.Thresholds().max_pelvis_speed_xy + 0.5, True)],
)
def test_peak_pelvis_speed_xy(model, geoms, speed, failed):
    clip = make_clip(model, stand_qpos(model))
    clip["body_lin_vel_w"][40:60, :, 0] = speed
    assert ("kin:pelvis_speed_xy" in reasons(clip, geoms)) is failed


def test_peak_pelvis_speed_z_and_angular_speed(model, geoms):
    th = hw_safe.Thresholds()
    clip = make_clip(model, stand_qpos(model))
    clip["body_lin_vel_w"][40:60, :, 2] = -(th.max_pelvis_speed_z + 0.3)
    clip["body_ang_vel_w"][40:60, :, 2] = th.max_pelvis_ang_speed + 1.0
    r = reasons(clip, geoms)
    assert "kin:pelvis_speed_z" in r and "kin:pelvis_ang_speed" in r


def test_single_frame_velocity_spike_is_smoothed_out(model, geoms):
    clip = make_clip(model, stand_qpos(model))
    clip["body_lin_vel_w"][50, :, 0] = 8.0  # one-frame retarget glitch
    clip["joint_vel"][50, ji(clip, "left_knee_joint")] = 30.0
    assert reasons(clip, geoms) == []


def test_joint_speed_is_relative_to_each_motor_limit(model, geoms):
    frac = hw_safe.Thresholds().joint_speed_fraction
    clip = make_clip(model, stand_qpos(model))
    speed = 0.9 * frac * 37.0  # fine for an elbow (5020, 37 rad/s) ...
    clip["joint_vel"][40:60, ji(clip, "left_elbow_joint")] = speed
    k = kin(clip, geoms)
    assert "kin:joint_speed" not in hw_safe.kinematic_reasons(k, "walk")
    clip["joint_vel"][40:60, ji(clip, "right_knee_joint")] = speed  # ... too fast for a knee (20)
    k = kin(clip, geoms)
    assert "kin:joint_speed" in hw_safe.kinematic_reasons(k, "walk")
    assert k.joint_speed_joint == "right_knee_joint"
    assert k.joint_speed_ratio == pytest.approx(speed / 20.0)


def test_minimum_pelvis_height(model, geoms):
    clip = make_clip(model, stand_qpos(model))
    pel = bi(clip, "pelvis")
    clip["body_pos_w"][40:60, pel, 2] = 0.15  # sitting / lying height
    assert "kin:pelvis_low" in reasons(clip, geoms)
    clip["body_pos_w"][40:60, pel, 2] = 0.35  # deep squat height is fine
    assert "kin:pelvis_low" not in reasons(clip, geoms)


@pytest.mark.parametrize(("flight_s", "failed"), [(0.1, False), (0.4, True)])
def test_both_feet_airborne_duration(model, geoms, flight_s, failed):
    clip = make_clip(model, stand_qpos(model))
    n = round(flight_s * FPS)
    clip["body_pos_w"][40 : 40 + n, :, 2] += 0.15
    assert ("kin:airborne" in reasons(clip, geoms)) is failed


def test_single_support_is_not_airborne(model, geoms):
    clip = make_clip(model, stand_qpos(model))
    clip["body_pos_w"][20:80, bi(clip, "left_ankle_roll_link"), 2] += 0.2  # standing on one leg
    assert reasons(clip, geoms) == []


@pytest.mark.parametrize("body", ["left_knee_link", "right_hip_yaw_link"])  # shin, thigh end
def test_knee_contact_is_allowed_only_for_kneel_clips(model, geoms, body):
    clip = make_clip(model, stand_qpos(model))
    drop_to_ground(clip, geoms, (body,), slice(30, 70))
    assert "kin:knee_contact" in reasons(clip, geoms, "walk")
    assert reasons(clip, geoms, "kneel") == []


def test_hand_contact_only_briefly(model, geoms):
    th = hw_safe.Thresholds()
    clip = make_clip(model, stand_qpos(model), T=200)
    brief = round(0.8 * th.max_hand_contact_run_s * FPS)
    drop_to_ground(clip, geoms, ("left_wrist_yaw_link",), slice(20, 20 + brief))
    assert reasons(clip, geoms) == []
    long = round(1.5 * th.max_hand_contact_run_s * FPS)
    drop_to_ground(clip, geoms, ("left_wrist_yaw_link",), slice(100, 100 + long))
    assert "kin:hand_contact" in reasons(clip, geoms)


def test_hand_contact_total_duration(model, geoms):
    th = hw_safe.Thresholds()
    clip = make_clip(model, stand_qpos(model), T=400)
    run = round(0.8 * th.max_hand_contact_run_s * FPS)
    for start in range(0, 400, 2 * run):  # many short touches add up
        drop_to_ground(clip, geoms, ("right_wrist_yaw_link",), slice(start, start + run))
    k = kin(clip, geoms)
    assert k.hand_contact_run_s <= th.max_hand_contact_run_s < k.hand_contact_s
    assert "kin:hand_contact" in hw_safe.kinematic_reasons(k, "walk")


@pytest.mark.parametrize("body", ["pelvis", "torso_link", "left_elbow_link", "right_hip_roll_link"])
def test_other_body_contact(model, geoms, body):
    clip = make_clip(model, stand_qpos(model))
    drop_to_ground(clip, geoms, (body,), slice(40, 60))
    k = kin(clip, geoms)
    assert "kin:body_contact" in hw_safe.kinematic_reasons(k, "kneel")
    assert k.body_contact_body == body
    clip = make_clip(model, stand_qpos(model))
    drop_to_ground(clip, geoms, (body,), slice(40, 43))  # 3 frames of retarget noise
    assert "kin:body_contact" not in reasons(clip, geoms)


# --------------------------------------------------------------------------- posture tagger


def test_posture_squat(model, geoms):
    k = kin(make_clip(model, squat_qpos(model)), geoms)
    assert (k.start_posture, k.end_posture) == ("squat", "squat")
    assert k.pelvis_z_min < hw_safe.Thresholds().stand_min_pelvis_z


def test_posture_kneel(model, geoms):
    clip = make_clip(model, kneel_qpos(model), support=hw_safe.KNEE_BODIES)
    k = kin(clip, geoms)
    assert (k.start_posture, k.end_posture) == ("kneel", "kneel")
    assert hw_safe.kinematic_reasons(k, "kneel") == []
    assert "kin:knee_contact" in hw_safe.kinematic_reasons(k, "idle")


def test_posture_one_knee_kneel(model, geoms):
    q = squat_qpos(model, knee=1.6)
    clip = make_clip(model, q)
    drop_to_ground(clip, geoms, ("right_knee_link",), slice(None))
    assert kin(clip, geoms).start_posture == "kneel"


def test_posture_other_when_lying(model, geoms):
    q = stand_qpos(model)
    q[3:7] = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]  # pitched 90 deg: lying
    clip = make_clip(model, q, clearance=0.0)
    k = kin(clip, geoms)
    assert (k.start_posture, k.end_posture) == ("other", "other")
    assert {"kin:pelvis_low", "kin:body_contact"} <= set(hw_safe.kinematic_reasons(k, "idle"))


def test_posture_transition_stand_to_squat(model, geoms):
    stand = make_clip(model, stand_qpos(model), T=60)
    squat = make_clip(model, squat_qpos(model), T=60)
    clip = {
        k: np.concatenate([stand[k], squat[k]]) if stand[k].ndim >= 2 else stand[k] for k in stand
    }
    k = kin(clip, geoms)
    assert (k.start_posture, k.end_posture) == ("standing", "squat")


def test_posture_other_when_both_feet_in_the_air(model, geoms):
    clip = make_clip(model, stand_qpos(model))
    clip["body_pos_w"][:, :, 2] += 0.3
    assert kin(clip, geoms).start_posture == "other"


def test_posture_standing_mid_stride(model, geoms):
    q = stand_qpos(model)
    q[7 + joint_names(model).index("left_knee_joint")] = 1.3  # swing leg folded
    clip = make_clip(model, q)
    clip["body_pos_w"][:, bi(clip, "left_ankle_roll_link"), 2] += 0.1
    assert kin(clip, geoms).start_posture == "standing"


def test_kneeling_with_feet_off_the_ground_is_not_airborne(model, geoms):
    clip = make_clip(model, kneel_qpos(model), support=hw_safe.KNEE_BODIES)
    for foot in hw_safe.FOOT_BODIES:
        clip["body_pos_w"][:, bi(clip, foot), 2] += 0.1
    k = kin(clip, geoms)
    assert k.airborne_s == 0.0 and k.start_posture == "kneel"


def test_squat_hold_promotes_category(model, geoms):
    th = hw_safe.Thresholds()
    stand = make_clip(model, stand_qpos(model), T=50)
    squat = make_clip(model, squat_qpos(model), T=round(2 * th.squat_hold_s * FPS))
    clip = {
        k: np.concatenate([stand[k], squat[k], stand[k]]) if stand[k].ndim >= 2 else stand[k]
        for k in stand
    }
    k = kin(clip, geoms)
    assert k.squat_s == pytest.approx(2 * th.squat_hold_s, abs=0.1)
    assert (k.start_posture, k.end_posture) == ("standing", "standing")
    assert hw_safe.refine_category("manipulation", k) == "squat"
    assert hw_safe.refine_category("walk", k) == "walk"  # locomotion keeps its category
    k_stand = kin(make_clip(model, stand_qpos(model)), geoms)
    assert k_stand.squat_s == 0.0
    assert hw_safe.refine_category("manipulation", k_stand) == "manipulation"


# --------------------------------------------------------------------------- metadata rules


def meta(name: str, **kw) -> hw_safe.ClipMeta:
    return hw_safe.ClipMeta(name=name, **kw)


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("walk_ff_loop_090_999__A000", "walk"),
        ("walk_sideway_left_start_999__A000_M", "walk"),
        ("jog_ff_start_180_999__A000", "jog"),
        ("idle_turn_270_999__A000", "turn"),
        ("idle_loop_999__A000", "idle"),
        ("kneeling_start_999__A000", "kneel"),
        ("crouch_ff_loop_180_999__A000", "crouch"),
        ("squat_999__A000", "squat"),
    ],
)
def test_category_from_name(name, category):
    assert hw_safe.classify_metadata(meta(name)) == (category, None)


def test_category_from_labels_and_bones_category():
    assert hw_safe.classify_metadata(meta("salute_999__A000", category="Gestures"))[0] == "gesture"
    assert hw_safe.classify_metadata(meta("x_999__A000", movement="kneeling"))[0] == "kneel"
    m = meta("salsa_basic_999__A000", category="Dancing")
    assert hw_safe.classify_metadata(m)[0] == "dance"
    assert "category:dance" in hw_safe.decide("dance", None, None)
    m = meta("pick_up_low_999__A000", category="Object Manipulation")
    assert hw_safe.classify_metadata(m)[0] == "manipulation"
    assert hw_safe.decide("manipulation", None, None, keep_categories=("manipulation",)) == [
        "kin:missing"
    ]


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("front_flip_999__A000", "acrobatics"),
        ("safety_roll_045_999__A000", "acrobatics"),
        ("cartwheel_left_999__A000", "acrobatics"),
        ("handstand_walk_999__A000", "acrobatics"),
        ("breakdance_freeze_999__A000", "acrobatics"),
        ("fall_and_get_up_999__A000", "fall"),
        ("lying_idle_999__A000", "floor"),
        ("crawl_ff_loop_999__A000", "floor"),
        ("high_jump_999__A000", "jump"),
        ("roundhouse_kick_999__A000", "fight"),
        ("walk_with_crutches_999__A000", "support"),
    ],
)
def test_exclusion_keywords_in_name(name, group):
    _, reason = hw_safe.classify_metadata(meta(name))
    assert reason is not None and reason.startswith(f"meta:{group}:")


def test_exclusion_keywords_in_labels_and_descriptions():
    m = meta("idle_999__A000", body_position="sitting on floor")
    assert hw_safe.classify_metadata(m)[1] == "meta:floor:sitting_on"
    m = meta("walk_ff_999__A000", descriptions=("A person walks forward, trips and falls.",))
    assert hw_safe.classify_metadata(m)[1].startswith("meta:fall:")
    m = meta("idle_999__A000", movement="standing, lying down")
    assert hw_safe.classify_metadata(m)[1] == "meta:floor:lying"


def test_keywords_match_whole_words_only():
    for name in ("scroll_phone_999__A000", "flipchart_point_999__A000", "trolley_walk_999__A000"):
        assert hw_safe.classify_metadata(meta(name))[1] is None, name


def test_decide_combines_metadata_and_kinematics(model, geoms):
    k = kin(make_clip(model, stand_qpos(model)), geoms)
    assert hw_safe.decide("idle", None, k) == []
    assert hw_safe.decide("idle", "meta:fall:falls", k) == ["meta:fall:falls"]
    assert hw_safe.decide("stunt", None, k) == ["category:stunt"]
