"""Stand-alone MuJoCo scene of a :class:`RobotSpec` for SONIC's sim-to-sim loop.

SONIC's simulator (``gear_sonic/utils/mujoco_sim/base_sim.py``, ``DefaultEnv``) loads an MJCF file,
computes each motor's PD torque from ``rt/lowcmd`` itself and writes it to ``data.ctrl``. It needs:

- a free joint named ``floating_base_joint`` first, then the 29 body joints in hardware motor order
  (it maps motor ``i`` to the ``i``-th joint whose name contains hip/knee/ankle/waist/shoulder/
  elbow/wrist, and actuator ``joint_id - 1``);
- one torque (``<motor>``) actuator per joint, in joint order;
- bodies ``pelvis`` and ``torso_link`` (IMU sources).

The robot is the training model: mjlab's Entity for :func:`realmimic.robots.g1.robot_cfg` (spec
inertials, hands, payloads, armature, collision setup), with its ``<position>`` actuators replaced
by ``<motor>`` actuators whose ``ctrlrange`` is the training effort limit. Physics options are the
training env's (``supermimic.envs.tracking_env_cfg``); foot friction is the evaluation protocol's 1.0.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mujoco
from mjlab.entity import Entity

from realmimic.obs.sonic import JOINT_NAMES_MUJOCO
from realmimic.robots.g1 import robot_cfg
from realmimic.robots.spec import RobotSpec

FOOT_FRICTION = 1.0  # tracking_env_cfg evaluation protocol (training randomizes 0.3-1.2)
TRAIN_TIMESTEP = 0.005
TRAIN_ITERATIONS = 10
TRAIN_LS_ITERATIONS = 20


def sim2sim_spec(robot: RobotSpec) -> mujoco.MjSpec:
    """Torque-driven scene (floor, light, robot) for SONIC's ``DefaultEnv``."""
    spec = Entity(robot_cfg(robot)).spec
    effort = {}
    for act in list(spec.actuators):
        lo, hi = act.forcerange
        if act.trntype != mujoco.mjtTrn.mjTRN_JOINT or lo != -hi:
            raise ValueError(f"unexpected actuator {act.name}: {act.trntype}, range {lo, hi}")
        effort[act.target] = float(hi)
        spec.delete(act)
    joints = [j.name for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    if tuple(joints) != JOINT_NAMES_MUJOCO:
        raise ValueError(f"joint order {joints} is not the hardware motor order")
    for name in joints:
        spec.add_actuator(
            name=name.removesuffix("_joint"),
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=(-effort[name], effort[name]),
        )
    for geom in spec.geoms:
        if geom.name.endswith("_collision") and "_foot" in geom.name:
            geom.friction[0] = FOOT_FRICTION
    for key in list(spec.keys):
        spec.delete(key)

    spec.option.timestep = TRAIN_TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations = TRAIN_ITERATIONS
    spec.option.ls_iterations = TRAIN_LS_ITERATIONS
    spec.worldbody.add_light(
        pos=(0, 0, 3), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05))
    return spec


def write_sim2sim_mjcf(robot: RobotSpec, out_dir: str | Path) -> Path:
    """Write ``scene.xml`` + the meshes it references to ``out_dir``; returns the scene path."""
    out_dir = Path(out_dir)
    spec = sim2sim_spec(robot)
    spec.compile()  # validates, and resolves the mesh directory
    mesh_src = Path(spec.modelfiledir) / spec.meshdir
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    for mesh in spec.meshes:
        shutil.copyfile(mesh_src / mesh.file, out_dir / "assets" / mesh.file)
    spec.meshdir = "assets"
    path = out_dir / "scene.xml"
    path.write_text(spec.to_xml())
    return path
