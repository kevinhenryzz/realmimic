"""Headless, scriptable SONIC sim-to-sim simulator for any G1 MJCF scene.

Reuses SONIC's own simulator code (``gear_sonic.utils.mujoco_sim``): ``DefaultEnv`` loads the scene,
maps motors to joints, computes each motor's torque from ``rt/lowcmd`` (``tau + kp (q* - q) +
kd (dq* - dq)``) and packs ``rt/lowstate``; ``UnitreeSdk2Bridge`` does the DDS I/O; ``ElasticBand``
holds the robot up until it is dropped. Only ``BaseSimulator``'s loop is replaced, to:

- run headless in real time with the PD recomputed every physics step (``--sim-dt``) and LowState
  published at its own rate (``--lowstate-dt``); SONIC's loop does both at 200 Hz;
- start standing in ``--init-joint-pos`` on the floor, held by the band until ``release`` arrives on
  stdin (SONIC: hang at z = 1 m, press 9 in the viewer);
- record the state to an ``.npz`` and report falls on stdout instead of silently resetting the sim;
- optionally degrade the simulation towards the real robot (``--state-delay``, ``--cmd-delay``,
  IMU gyro noise/bias and mounting tilt, encoder offsets and velocity noise, weaker motors, extra
  torso mass, floor friction): the published LowState and the applied PD then differ from the ideal
  ones the way the hardware's do. All default to off; draws are seeded (``--seed``).

Runs in SONIC's sim venv (Python 3.10, see ``setup_sim.sh``) with the SONIC checkout root on
``PYTHONPATH``. stdin commands, one per line: ``release`` (drop the band), ``quit``. stdout events:
``SIM ready``, ``SIM released t=<s>``, ``SIM fall t=<s> z=<m> tilt=<rad>``, ``SIM saved <path>``.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def make_config(scene: Path, sim_dt: float) -> dict[str, Any]:
    """SONIC's sim config (g1_29dof_sonic_model12.yaml + SimLoopConfig overrides) for ``scene``."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    num_hand = sum("left_hand" in model.joint(i).name for i in range(model.njnt))
    cfg = SimLoopConfig(interface="sim", enable_onscreen=False, sim_frequency=round(1 / sim_dt))
    wbc = cfg.load_wbc_yaml()
    wbc["ROBOT_SCENE"] = str(scene.resolve())  # DefaultEnv joins it to the repo root; abs wins
    wbc["SIMULATE_DT"] = sim_dt
    wbc["NUM_HAND_MOTORS"] = wbc["NUM_HAND_JOINTS"] = num_hand
    wbc["ENABLE_ELASTIC_BAND"] = True
    if len(wbc["motor_effort_limit_list"]) != model.nu:
        # SONIC's list is for its 43-actuator stock model; other scenes carry their limits.
        wbc["motor_effort_limit_list"] = model.actuator_ctrlrange[:, 1].tolist()
    return wbc


def standing_height(env: DefaultEnv) -> float:
    """Pelvis height at which the lowest robot geom touches the floor (current joint pose)."""
    m, d = env.mj_model, env.mj_data
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    d.qpos[2] = 1.0
    mujoco.mj_forward(m, d)
    robot = [
        g for g in range(m.ngeom)
        if g != floor and m.geom_bodyid[g] != 0
        and (m.geom_contype[g] & m.geom_conaffinity[floor] or m.geom_conaffinity[g] & m.geom_contype[floor])
    ]  # fmt: skip
    fromto = np.zeros(6)
    gap = min(mujoco.mj_geomDistance(m, d, g, floor, 2.0, fromto) for g in robot)
    return 1.0 - gap


class Sim:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = make_config(Path(args.scene), args.sim_dt)
        self.env = DefaultEnv(self.config, onscreen=False, offscreen=False)
        ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
        self.bridge = UnitreeSdk2Bridge(self.config)
        self.env.set_unitree_bridge(self.bridge)
        m, d = self.env.mj_model, self.env.mj_data
        self.qadr = m.jnt_qposadr[self.env.body_joint_index]
        self.vadr = m.jnt_dofadr[self.env.body_joint_index]
        self.pelvis = m.body("pelvis").id
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [m.body(n).id for n in FOOT_BODIES]
        self.limit = np.asarray(self.config["motor_effort_limit_list"]) * args.torque_scale
        self.setup_perturbations()

        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = np.asarray(args.init_joint_pos, dtype=float)
        d.qpos[2] = standing_height(self.env) + 0.002
        mujoco.mj_forward(m, d)
        band = self.env.elastic_band
        band.point = d.xpos[self.env.band_attached_link].copy()
        band.length = 0.0
        band.enable = True

        self.perturbed = any((
            args.state_delay, args.imu_tilt, args.gyro_bias, args.gyro_noise, args.encoder_offset,
            args.encoder_offset_joint, args.joint_vel_noise,
        ))  # fmt: skip
        self.running = True
        self.fallen = False
        self.release_time: float | None = None
        self.rows: dict[str, list[np.ndarray]] = {k: [] for k in self.FIELDS}

    def setup_perturbations(self) -> None:
        """Seeded constants of the degraded robot (model changes, sensor offsets) and delay queues."""
        a, m = self.args, self.env.mj_model
        rng = np.random.default_rng(a.seed)
        self.rng = rng
        n = len(self.env.body_joint_index)
        names = [m.joint(j).name for j in self.env.body_joint_index]
        self.encoder_offset = rng.uniform(-a.encoder_offset, a.encoder_offset, n)
        for item in a.encoder_offset_joint:
            name, value = item.split("=")
            self.encoder_offset[names.index(name)] = float(value)
        axis = rng.normal(size=3)
        axis[2] = 0.0  # mounting tilt about a horizontal axis of the IMU frame
        axis /= np.linalg.norm(axis)
        self.imu_tilt = np.concatenate([[np.cos(a.imu_tilt / 2)], np.sin(a.imu_tilt / 2) * axis])
        self.gyro_bias = rng.uniform(-a.gyro_bias, a.gyro_bias, 3)
        if a.torso_mass:
            torso = m.body("torso_link").id
            m.body_mass[torso] += a.torso_mass  # a point mass at the body COM
        if a.friction is not None:
            floor_feet = [g for g in range(m.ngeom) if g == self.floor or m.geom_bodyid[g] in self.feet]
            m.geom_friction[floor_feet, 0] = a.friction
        self.state_queue: collections.deque = collections.deque()
        self.cmd_queue: collections.deque = collections.deque()

    def perturbed_obs(self) -> dict[str, Any]:
        """``DefaultEnv.prepare_obs`` with sensor errors, delayed by ``--state-delay``."""
        a = self.args
        obs = copy.deepcopy(self.env.prepare_obs())  # some entries are views into mj_data
        obs["body_q"] = obs["body_q"] + self.encoder_offset
        if a.joint_vel_noise:
            obs["body_dq"] = obs["body_dq"] + self.rng.normal(0.0, a.joint_vel_noise, obs["body_dq"].shape)
        q = np.empty(4)
        mujoco.mju_mulQuat(q, obs["floating_base_pose"][3:7], self.imu_tilt)
        obs["floating_base_pose"][3:7] = q
        gyro = obs["floating_base_vel"][3:6] + self.gyro_bias  # free-joint angular velocity: body frame
        if a.gyro_noise:
            gyro = gyro + self.rng.normal(0.0, a.gyro_noise, 3)
        obs["floating_base_vel"][3:6] = gyro
        now = self.env.mj_data.time
        self.state_queue.append((now, obs))
        while len(self.state_queue) > 1 and self.state_queue[1][0] <= now - a.state_delay + 1e-9:
            self.state_queue.popleft()
        return self.state_queue[0][1]

    def delayed_command(self) -> np.ndarray:
        """LowCmd (q, dq, kp, kd, tau) per body motor as it was ``--cmd-delay`` ago, (5, n)."""
        cmd = self.bridge.low_cmd.motor_cmd
        n = len(self.env.body_joint_index)
        now = self.env.mj_data.time
        self.cmd_queue.append((now, np.array([[c.q, c.dq, c.kp, c.kd, c.tau] for c in cmd[:n]]).T))
        while len(self.cmd_queue) > 1 and self.cmd_queue[1][0] <= now - self.args.cmd_delay + 1e-9:
            self.cmd_queue.popleft()
        return self.cmd_queue[0][1]

    FIELDS = (
        "t_mono", "sim_time", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel", "q", "dq",
        "tau", "q_cmd", "kp", "kd", "foot_contact", "other_floor_contacts", "band", "lag",
    )  # fmt: skip

    def apply_band(self) -> None:
        env, m, d = self.env, self.env.mj_model, self.env.mj_data
        link = env.band_attached_link
        if env.elastic_band.enable:
            pose = np.concatenate([d.xpos[link], d.xquat[link], np.zeros(6)])
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, link, pose[7:13], 0)
            pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
            d.xfrc_applied[link] = env.elastic_band.Advance(pose)
        else:
            d.xfrc_applied[link] = 0.0

    def apply_torques(self) -> np.ndarray:
        """DefaultEnv.sim_step's torque path: per-motor PD from the latest LowCmd, clipped. With
        ``--cmd-delay`` the same PD runs on the LowCmd of ``--cmd-delay`` ago."""
        env = self.env
        if self.args.cmd_delay > 0 and self.bridge.low_cmd:
            q_des, dq_des, kp, kd, tau = self.delayed_command()
            d = env.mj_data
            body = tau + kp * (q_des - d.qpos[self.qadr]) + kd * (dq_des - d.qvel[self.vadr])
            env.torques[env.body_joint_index - 1] = body
        else:
            env.torques[env.body_joint_index - 1] = env.compute_body_torques()
        if env.num_hand_dof > 0:
            hand = env.compute_hand_torques()
            env.torques[env.left_hand_index - 1] = hand[: env.num_hand_dof]
            env.torques[env.right_hand_index - 1] = hand[env.num_hand_dof :]
        env.torques = np.clip(env.torques, -self.limit, self.limit)
        env.mj_data.ctrl[:] = env.torques
        return env.torques[env.body_joint_index - 1]

    def floor_contacts(self) -> tuple[np.ndarray, int]:
        m, d = self.env.mj_model, self.env.mj_data
        feet = np.zeros(2, dtype=bool)
        other = 0
        for c in d.contact[: d.ncon]:
            if self.floor not in (c.geom1, c.geom2):
                continue
            body = m.geom_bodyid[c.geom2 if c.geom1 == self.floor else c.geom1]
            if body in self.feet:
                feet[self.feet.index(body)] = True
            else:
                other += 1
        return feet, other

    def record(self, tau: np.ndarray, lag: float) -> None:
        d, cmd = self.env.mj_data, self.bridge.low_cmd.motor_cmd
        n = len(self.env.body_joint_index)
        feet, other = self.floor_contacts()
        rot = d.xmat[self.pelvis].reshape(3, 3)
        lin_vel = rot.T @ d.qvel[0:3]  # free-joint linear velocity is world-frame
        row = {
            "t_mono": time.monotonic(), "sim_time": d.time, "base_pos": d.qpos[0:3],
            "base_quat": d.qpos[3:7], "base_lin_vel": lin_vel, "base_ang_vel": d.qvel[3:6],
            "q": d.qpos[self.qadr], "dq": d.qvel[self.vadr], "tau": tau,
            "q_cmd": [cmd[i].q for i in range(n)], "kp": [cmd[i].kp for i in range(n)],
            "kd": [cmd[i].kd for i in range(n)], "foot_contact": feet,
            "other_floor_contacts": other, "band": self.env.elastic_band.enable, "lag": lag,
        }  # fmt: skip
        for k, v in row.items():
            self.rows[k].append(np.array(v, dtype=np.float64 if k == "t_mono" else np.float32))
        if self.release_time is not None and not self.fallen:
            z, tilt = float(d.qpos[2]), float(np.arccos(np.clip(rot[2, 2], -1.0, 1.0)))
            if z < self.args.fall_height or tilt > self.args.fall_tilt:
                self.fallen = True
                print(f"SIM fall t={d.time:.3f} z={z:.3f} tilt={tilt:.3f}", flush=True)

    def commands(self) -> None:
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "release":
                self.env.elastic_band.enable = False
                self.release_time = self.env.mj_data.time
                print(f"SIM released t={self.release_time:.3f}", flush=True)
            elif cmd == "quit":
                break
        self.running = False

    def run(self) -> None:
        a = self.args
        lowstate_every = round(a.lowstate_dt / a.sim_dt)
        record_every = round(a.record_dt / a.sim_dt)
        threading.Thread(target=self.commands, daemon=True).start()
        print("SIM ready", flush=True)
        m, d = self.env.mj_model, self.env.mj_data
        t0, step, lag = time.monotonic(), 0, 0.0
        while self.running:
            if step % lowstate_every == 0:
                self.bridge.PublishLowState(self.perturbed_obs() if self.perturbed else self.env.prepare_obs())
            self.apply_band()
            tau = self.apply_torques()
            mujoco.mj_step(m, d)
            step += 1
            if step % record_every == 0:
                self.record(tau, lag)
            lag = time.monotonic() - (t0 + step * a.sim_dt)
            if lag < 0:
                time.sleep(-lag)

    def save(self) -> None:
        out = Path(self.args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "scene": str(Path(self.args.scene).resolve()),
            "sim_dt": self.args.sim_dt, "lowstate_dt": self.args.lowstate_dt,
            "record_dt": self.args.record_dt, "release_sim_time": self.release_time,
            "joint_names": [self.env.mj_model.joint(j).name for j in self.env.body_joint_index],
            "effort_limit": self.limit[self.env.body_joint_index - 1].tolist(),
            "init_joint_pos": list(self.args.init_joint_pos),
            "perturbations": {k: getattr(self.args, k) for k in PERTURBATIONS},
            "encoder_offset": self.encoder_offset.tolist(), "gyro_bias": self.gyro_bias.tolist(),
        }  # fmt: skip
        arrays = {k: np.stack(v) if v else np.zeros(0) for k, v in self.rows.items()}
        np.savez_compressed(out, meta=json.dumps(meta), **arrays)
        print(f"SIM saved {out}", flush=True)


PERTURBATIONS = (
    "seed", "state_delay", "cmd_delay", "imu_tilt", "gyro_bias", "gyro_noise", "encoder_offset",
    "encoder_offset_joint", "joint_vel_noise", "torque_scale", "torso_mass", "friction",
)  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description="Headless SONIC sim-to-sim simulator.")
    p.add_argument("--scene", required=True, help="MJCF scene (floor geom named 'floor')")
    p.add_argument("--out", required=True, help="recording .npz")
    p.add_argument(
        "--init-joint-pos", required=True,
        type=lambda s: [float(x) for x in s.split(",")], help="29 joint angles, motor order",
    )  # fmt: skip
    p.add_argument("--sim-dt", type=float, default=0.001)
    p.add_argument("--lowstate-dt", type=float, default=0.002)
    p.add_argument("--record-dt", type=float, default=0.005)
    p.add_argument("--fall-height", type=float, default=0.35)
    p.add_argument("--fall-tilt", type=float, default=1.0, help="pelvis tilt from upright [rad]")
    g = p.add_argument_group("degradation towards the real robot (all off by default)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--state-delay", type=float, default=0.0, help="LowState age when published [s]")
    g.add_argument("--cmd-delay", type=float, default=0.0, help="LowCmd age when applied [s]")
    g.add_argument("--imu-tilt", type=float, default=0.0, help="IMU mounting tilt [rad]")
    g.add_argument("--gyro-bias", type=float, default=0.0, help="per-axis bias half-range [rad/s]")
    g.add_argument("--gyro-noise", type=float, default=0.0, help="white noise std [rad/s]")
    g.add_argument("--encoder-offset", type=float, default=0.0, help="per-joint half-range [rad]")
    g.add_argument(
        "--encoder-offset-joint", action="append", default=[], metavar="JOINT=RAD",
        help="fixed offset of one joint (repeatable), e.g. left_ankle_roll_joint=0.087",
    )  # fmt: skip
    g.add_argument("--joint-vel-noise", type=float, default=0.0, help="white noise std [rad/s]")
    g.add_argument("--torque-scale", type=float, default=1.0, help="effort limits x this")
    g.add_argument("--torso-mass", type=float, default=0.0, help="extra mass on torso_link [kg]")
    g.add_argument("--friction", type=float, default=None, help="floor + feet sliding friction")
    args = p.parse_args()
    for name in ("lowstate_dt", "record_dt"):
        ratio = getattr(args, name) / args.sim_dt
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            p.error(f"--{name.replace('_', '-')} must be a multiple of --sim-dt")
    sim = Sim(args)

    def stop(*_: object) -> None:
        sim.running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        sim.run()
    finally:
        sim.save()


if __name__ == "__main__":
    main()
