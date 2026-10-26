"""Torque-driven sim-to-sim scene: what SONIC's DefaultEnv assumes about motor/joint/actuator order."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mjlab.entity import Entity

from realmimic.obs.sonic import JOINT_NAMES_MUJOCO
from realmimic.robots import ROBOTS, get_robot_spec
from realmimic.robots.g1 import robot_cfg
from realmimic.robots.sim2sim import sim2sim_spec, write_sim2sim_mjcf

# base_sim.py DefaultEnv.init_scene: a joint is body joint i if its name contains one of these.
SONIC_BODY_PARTS = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")


@pytest.fixture(scope="module", params=sorted(ROBOTS))
def scene(request, tmp_path_factory) -> tuple[str, mujoco.MjModel]:
    path = write_sim2sim_mjcf(get_robot_spec(request.param), tmp_path_factory.mktemp("scene"))
    return request.param, mujoco.MjModel.from_xml_path(str(path))


def test_sonic_motor_mapping(scene):
    """Motor i -> i-th body joint (by SONIC's name rule) -> qpos 7+i, actuator i, all hardware order."""
    _, m = scene
    assert m.joint(0).name == "floating_base_joint"
    body = [i for i in range(m.njnt) if any(p in m.joint(i).name for p in SONIC_BODY_PARTS)]
    assert [m.joint(i).name for i in body] == list(JOINT_NAMES_MUJOCO)
    assert body == list(range(1, 30))
    assert list(m.jnt_qposadr[body]) == list(range(7, 36))  # DefaultEnv: qpos[joint_id + 7 - 1]
    assert m.nu == 29
    assert list(m.actuator_trnid[:, 0]) == body  # DefaultEnv: ctrl[joint_id - 1]
    assert (m.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED).all()
    assert (m.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE).all()
    np.testing.assert_array_equal(m.actuator_gainprm[:, 0], 1.0)  # ctrl is torque [Nm]
    for name in ("pelvis", "torso_link"):
        m.body(name)
    m.geom("floor")


def test_matches_training_model(scene):
    """Same inertials, armature, joint limits and collision geoms as the mjlab training entity
    (MuJoCo's XML writer keeps 6 significant digits); effort limits = the position actuators'
    force ranges (hip pitch 88 Nm for 7520-14)."""
    name, m = scene
    train = Entity(robot_cfg(get_robot_spec(name))).spec.compile()
    for attr in ("body_mass", "body_inertia", "body_ipos", "dof_armature", "jnt_range"):
        np.testing.assert_allclose(getattr(m, attr), getattr(train, attr), rtol=5e-6, atol=1e-9)
    effort = {train.joint(train.actuator_trnid[a, 0]).name: train.actuator_forcerange[a, 1]
              for a in range(train.nu)}  # fmt: skip
    for a in range(m.nu):
        assert m.actuator_ctrlrange[a, 1] == effort[m.joint(m.actuator_trnid[a, 0]).name]
    assert m.actuator("left_hip_pitch").ctrlrange[1] == 88.0
    assert m.actuator("left_knee").ctrlrange[1] == 139.0
    assert m.ngeom == train.ngeom + 1 and m.geom(0).name == "floor"  # world geoms come first
    for attr in ("geom_contype", "geom_conaffinity", "geom_condim", "geom_priority", "geom_size"):
        np.testing.assert_allclose(getattr(m, attr)[1:], getattr(train, attr), rtol=5e-6)


def test_foot_friction_and_options():
    m = sim2sim_spec(get_robot_spec("g1")).compile()
    feet = [i for i in range(m.ngeom) if "_foot" in m.geom(i).name]
    assert feet and (m.geom_friction[feet, 0] == 1.0).all()
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert (m.opt.iterations, m.opt.ls_iterations) == (10, 20)
