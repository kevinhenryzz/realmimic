"""Author the browser demo's reference motions from the robot's default pose (our own data, no third-party
motion capture) and check that the site's policy tracks them in CPU MuJoCo.

    uv run python tools/web_author_motions.py site            # writes site/assets/motions/*.bin + index.json
    uv run python tools/web_author_motions.py site --check    # also runs the closed-loop check

Motions are kinematic references at 50 Hz in the training library's row layout (see assets/motions/
index.json). Legs are solved with damped least-squares inverse kinematics so the feet stay where the
script puts them (planted, or stepping); every motion starts and ends in the default pose, so it loops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

FPS = 50
PREFIX = "robot/"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def ease(x: float) -> float:
    """Smooth 0 -> 1 (cosine)."""
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def rot_z(v: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


class Author:
    def __init__(self, model: mujoco.MjModel, meta: dict):
        self.m = model
        self.d = mujoco.MjData(model)
        self.names = meta["joint_names"]
        self.q0 = np.asarray(meta["default_joint_pos"], dtype=float)
        self.z0 = float(meta["stand_root_height"])
        self.jid = {n: i for i, n in enumerate(self.names)}
        self.feet = [model.body(PREFIX + f).id for f in FEET]
        self.leg_dofs = [6 + self.jid[n] for n in self.names if "hip" in n or "knee" in n or "ankle" in n]
        lo, hi = model.jnt_range[1:].T
        self.lo, self.hi = lo, hi
        # Feet in the default pose, relative to the pelvis's ground projection (heading frame).
        q = self.qpos(np.array([0.0, 0.0, self.z0]), np.array([1.0, 0, 0, 0]), self.q0)
        self.d.qpos[:] = q
        mujoco.mj_kinematics(model, self.d)
        self.foot0 = [self.d.xpos[b].copy() for b in self.feet]
        self.foot0_quat = [self.d.xquat[b].copy() for b in self.feet]

    @staticmethod
    def qpos(p: np.ndarray, quat: np.ndarray, joints: np.ndarray) -> np.ndarray:
        return np.concatenate([p, quat, joints])

    def legs(self, p: np.ndarray, quat: np.ndarray, joints: np.ndarray, foot_pos, foot_quat, iters: int = 60) -> np.ndarray:
        """Leg joints that put both feet at (foot_pos, foot_quat) for the pelvis pose (p, quat)."""
        m, d = self.m, self.d
        q = self.qpos(p, quat, joints.copy())
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err, rows = [], []
            for b, tp, tq in zip(self.feet, foot_pos, foot_quat):
                dq, dw = np.zeros(3), np.zeros(3)
                mujoco.mju_subQuat(dq, tq, d.xquat[b])  # in the foot's frame; the Jacobian is in the world frame
                mujoco.mju_rotVecQuat(dw, dq, d.xquat[b])
                err += [tp - d.xpos[b], dw]
                mujoco.mj_jacBody(m, d, jacp, jacr, b)
                rows += [jacp[:, self.leg_dofs].copy(), jacr[:, self.leg_dofs].copy()]
            e, J = np.concatenate(err), np.concatenate(rows)
            if np.abs(e).max() < 1e-7:
                break
            step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            for k, dof in enumerate(self.leg_dofs):
                i = dof - 6
                q[7 + i] = np.clip(q[7 + i] + step[k], self.lo[i], self.hi[i])
        return q

    def rows(self, frames: list[np.ndarray]) -> np.ndarray:
        """qpos frames -> (T, 71) rows: root pos, quat, joints, root lin vel (world), root ang vel (world), joint vel."""
        Q = np.stack(frames)
        T = len(Q)
        out = np.zeros((T, 71), dtype=np.float32)
        out[:, :36] = Q
        for t in range(T):
            a, b = Q[max(t - 1, 0)], Q[min(t + 1, T - 1)]
            h = (min(t + 1, T - 1) - max(t - 1, 0)) / FPS
            if h == 0:
                continue
            out[t, 36:39] = (b[:3] - a[:3]) / h
            dq = np.zeros(3)  # world-frame angular velocity: rotation from a to b, in the world frame
            mujoco.mju_subQuat(dq, b[3:7], a[3:7])  # body-frame rotation vector of a^-1 b
            w = np.zeros(3)
            mujoco.mju_rotVecQuat(w, dq / h, Q[t, 3:7])
            out[t, 39:42] = w
            out[t, 42:71] = (b[7:] - a[7:]) / h
        out[0, 36:], out[-1, 36:] = 0, 0  # starts and ends at rest (loops)
        return out

    # ---------------------------------------------------------------- motions

    def squat(self) -> list[np.ndarray]:
        """Slow squat and rise: pelvis down 16 cm, hips back, torso kept upright by the waist."""
        frames, plan = [], [(1.0, 0), (2.0, 1), (0.8, 0), (2.0, -1), (1.2, 0)]
        s = 0.0
        for dur, direction in plan:
            n = int(dur * FPS)
            for k in range(n):
                x = s + direction * ease((k + 1) / n) if direction else s
                p = np.array([-0.05 * x, 0.0, self.z0 - 0.16 * x])
                pitch = 0.30 * x
                quat = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
                j = self.q0.copy()
                j[self.jid["waist_pitch_joint"]] = -0.25 * x
                frames.append(self.legs(p, quat, j, self.foot0, self.foot0_quat))
            s = s + direction if direction else s
        return frames

    def weight_shift(self) -> list[np.ndarray]:
        """Pelvis over the left foot, then the right, then back to the middle; feet planted."""
        frames, keys = [], [(1.0, 0.0), (1.6, 0.07), (0.8, 0.07), (2.4, -0.07), (0.8, -0.07), (1.6, 0.0), (0.8, 0.0)]
        y = 0.0
        for dur, target in keys:
            n = int(dur * FPS)
            for k in range(n):
                yy = y + (target - y) * ease((k + 1) / n)
                roll = -0.6 * yy  # lean the pelvis slightly with the shift
                p = np.array([0.0, yy, self.z0 - 0.01 * abs(yy) / 0.07])
                quat = np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
                j = self.q0.copy()
                j[self.jid["waist_roll_joint"]] = -roll
                frames.append(self.legs(p, quat, j, self.foot0, self.foot0_quat))
            y = target
        return frames

    def wave(self) -> list[np.ndarray]:
        """Right arm up beside the head, three waves, arm down; legs and torso in the default pose."""
        sp, sr, sy, el = (self.jid[f"right_{n}_joint"] for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"))
        up = {sp: -1.2, sr: -1.25, sy: 0.0, el: 0.15}
        frames = []
        def pose(alpha: float, wave: float) -> np.ndarray:
            j = self.q0.copy()
            for i, v in up.items():
                j[i] = self.q0[i] + (v - self.q0[i]) * alpha
            j[el] += 0.45 * wave
            j[sy] += 0.35 * wave
            return self.qpos(np.array([0.0, 0.0, self.z0]), np.array([1.0, 0, 0, 0]), j)
        for k in range(int(1.0 * FPS)):
            frames.append(pose(0.0, 0.0))
        for k in range(int(1.5 * FPS)):
            frames.append(pose(ease((k + 1) / (1.5 * FPS)), 0.0))
        n = int(3 * 0.9 * FPS)
        for k in range(n):
            frames.append(pose(1.0, np.sin(2 * np.pi * (k + 1) / (0.9 * FPS))))
        for k in range(int(1.5 * FPS)):
            frames.append(pose(1.0 - ease((k + 1) / (1.5 * FPS)), 0.0))
        for k in range(int(0.8 * FPS)):
            frames.append(pose(0.0, 0.0))
        return frames

    def turn(self, steps: int = 3, step_yaw: float = np.radians(30)) -> list[np.ndarray]:
        """Turn left in place by steps * step_yaw (stepping: left foot, then right), then back."""
        frames = [self._turn_frame(0.0, (0.0, 0.0), None, 0.0, 0.0) for _ in range(int(0.8 * FPS))]
        yaw_feet = [0.0, 0.0]
        sequence = []
        for k in range(steps):
            sequence += [(0, +step_yaw), (1, +step_yaw)]
        for k in range(steps):
            sequence += [(1, -step_yaw), (0, -step_yaw)]
        for foot, dyaw in sequence:
            stance = 1 - foot
            shift_n, swing_n, back_n = int(0.4 * FPS), int(0.5 * FPS), int(0.3 * FPS)
            y0 = list(yaw_feet)
            for k in range(shift_n):  # weight onto the stance foot
                frames.append(self._turn_frame(np.mean(y0), tuple(y0), stance, ease((k + 1) / shift_n), 0.0))
            for k in range(swing_n):  # swing: lift, rotate the swing foot about the turning axis
                a = ease((k + 1) / swing_n)
                yf = list(y0)
                yf[foot] = y0[foot] + dyaw * a
                frames.append(self._turn_frame(np.mean(yf), tuple(yf), stance, 1.0, np.sin(np.pi * (k + 1) / swing_n)))
            yaw_feet[foot] += dyaw
            for k in range(back_n):  # weight back to the middle
                frames.append(self._turn_frame(np.mean(yaw_feet), tuple(yaw_feet), stance, 1.0 - ease((k + 1) / back_n), 0.0))
        frames += [self._turn_frame(0.0, (0.0, 0.0), None, 0.0, 0.0) for _ in range(int(0.8 * FPS))]
        return frames

    def _turn_frame(self, pelvis_yaw: float, foot_yaw: tuple[float, float], stance, shift: float, lift: float) -> np.ndarray:
        foot_pos, foot_quat = [], []
        for i in range(2):
            pos = rot_z(self.foot0[i], foot_yaw[i])
            if stance is not None and i != stance:
                pos = pos + np.array([0.0, 0.0, 0.04 * lift])
            foot_pos.append(pos)
            foot_quat.append(quat_mul(yaw_quat(foot_yaw[i]), self.foot0_quat[i]))
        p = np.array([0.0, 0.0, self.z0 - 0.04])
        if stance is not None:
            over = rot_z(self.foot0[stance], foot_yaw[stance])
            p[:2] += 0.4 * shift * over[:2]
        return self.legs(p, yaw_quat(pelvis_yaw), self.q0.copy(), foot_pos, foot_quat)


MOTIONS = {
    "wave": ("Wave", "right arm raised beside the head, three waves", Author.wave),
    "squat": ("Squat", "slow squat and rise, feet planted", Author.squat),
    "weight_shift": ("Weight shift", "pelvis over the left foot, then the right; feet planted", Author.weight_shift),
    "turn": ("Turn in place", "90 degrees left in three stepping cycles, then back", Author.turn),
}


def track(model: mujoco.MjModel, meta: dict, policy, rows: np.ndarray, loops: int = 2) -> dict:
    """The site's loop on a looping clip: fall (training termination), mean joint and pelvis position error."""
    H, K = meta["inputs"]["state_hist"]["shape"][1], meta["inputs"]["ref_window"]["shape"][1]
    act_jnt = model.actuator_trnid[:, 0] - 1
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("init_state").id)
    mujoco.mj_forward(model, d)
    s = np.concatenate([d.qpos, d.qvel])
    states, actions = [s] * H, [np.zeros(29)] * H
    T = len(rows)
    jerr, perr, yerr, fell = [], [], [], None
    yaw = lambda q: np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))  # noqa: E731

    def gz(q):
        _, x, y, _ = q / np.linalg.norm(q)
        return -(1 - 2 * (x * x + y * y))

    for k in range(loops * T):
        win = rows[[(k + 1 + i) % T for i in range(K)]]  # tick k targets frame k + 1, as training and the benchmark
        a, target = policy({"state_hist": np.stack(states[-H:])[None].astype(np.float32), "action_hist": np.stack(actions[-H:])[None].astype(np.float32), "ref_window": win[None].astype(np.float32)})
        d.ctrl[:] = np.asarray(target).reshape(29)[act_jnt]
        for _ in range(meta["decimation"]):
            mujoco.mj_step(model, d)
        states.append(np.concatenate([d.qpos, d.qvel]))
        actions.append(np.asarray(a).reshape(29))
        ref = rows[(k + 1) % T]
        jerr.append(np.abs(d.qpos[7:] - ref[7:36]).mean())
        perr.append(np.linalg.norm(d.qpos[:3] - ref[:3]))
        yerr.append(abs((yaw(d.qpos[3:7]) - yaw(ref[3:7]) + np.pi) % (2 * np.pi) - np.pi))
        if fell is None and (abs(ref[2] - d.qpos[2]) > meta["termination"]["height"] or abs(gz(ref[3:7]) - gz(d.qpos[3:7])) > meta["termination"]["tilt"]):
            fell = k
    return {"fell_at_s": None if fell is None else round(fell / FPS, 2), "joint_err_rad": float(np.mean(jerr)), "pelvis_err_m": float(np.mean(perr)), "max_pelvis_err_m": float(np.max(perr)), "heading_err_deg": float(np.degrees(np.mean(yerr)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site")
    parser.add_argument("--check", action="store_true", help="track every motion with the site's policy in CPU MuJoCo")
    args = parser.parse_args()
    site = Path(args.site)
    meta = json.loads((site / "assets/policy/meta.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(site / "assets/robot/scene.mjb"))
    author = Author(model, meta)
    out = site / "assets/motions"
    index = json.loads((out / "index.json").read_text())
    poses = [m for m in index["motions"] if m["kind"] == "pose"]
    others = [m for m in index["motions"] if m["kind"] != "pose" and m["id"] not in MOTIONS]  # e.g. retargeted clips
    index["motions"] = []
    index["fields"]["loop"] = "clip restarts from its first frame after the last (clips start and end in the default pose)"
    clips = {}
    for mid, (name, desc, fn) in MOTIONS.items():
        rows = author.rows(fn(author))
        (out / f"{mid}.bin").write_bytes(rows.astype(np.float32).tobytes())
        clips[mid] = rows
        index["motions"].append({
            "id": mid, "name": name, "kind": "clip", "file": f"{mid}.bin", "loop": True,
            "source": f"authored for this site from the robot model's default pose ({desc}); tools/web_author_motions.py",
            "source_url": None, "licence": "Apache-2.0 (own data, no third-party motion capture)", "retargeting": None,
        })  # fmt: skip
        print(f"{mid}: {len(rows) / FPS:.1f} s, {len(rows)} frames, {(out / f'{mid}.bin').stat().st_size / 1e3:.0f} kB")
    for m in index["motions"]:
        m["group"] = "Made for this page"
    index["motions"] += others + poses  # authored motions first, then retargeted clips, then the poses
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    if args.check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(site / "assets/policy/policy.onnx"), providers=["CPUExecutionProvider"])
        pol = lambda x: sess.run(["action", "joint_target"], x)  # noqa: E731
        for mid, rows in clips.items():
            r = track(model, meta, pol, rows)
            print(f"check {mid}: {'no fall' if r['fell_at_s'] is None else 'FELL at ' + str(r['fell_at_s']) + ' s'}; "
                  f"joint error {r['joint_err_rad']:.3f} rad, pelvis error mean {100 * r['pelvis_err_m']:.1f} cm, max {100 * r['max_pelvis_err_m']:.1f} cm, heading error {r['heading_err_deg']:.1f} deg")  # fmt: skip


if __name__ == "__main__":
    main()
