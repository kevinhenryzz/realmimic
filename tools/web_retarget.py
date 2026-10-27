"""Retarget openly licensed human motion capture to the G1 for the browser demo, and check each clip with the
site's policy in CPU MuJoCo.

    uv run python tools/web_retarget.py site <out_dir> --cmu <dir with NN.asf / NN_TT.amc> --clips clips.json [--check]

Sources: CMU Graphics Lab Motion Capture Database (original ASF/AMC from mocap.cs.cmu.edu; "free for all uses",
not to be resold, even converted) and 100STYLE (BVH from Zenodo record 8127870, CC BY 4.0).
``clips.json``: [{"id", "name", "trial": "NN_TT", "fps", "start", "end"}] for CMU, [{"id", "name",
"source": "100style", "file": "Style_XX.bvh", "start", "end"}] for 100STYLE (seconds; the window kept).

Retargeting: forward kinematics of the ASF skeleton; Y-up -> Z-up; the start heading aligned with +x. The G1
follows the human's bone directions with its own segment lengths (so every target is reachable): pelvis
position (scaled by leg length) and orientation, torso orientation, knees, ankles and foot direction, elbows and
wrists, solved per 50 Hz frame by damped least-squares inverse kinematics. Ground: no part of the robot below
the floor, and a foot the human has planted is put on the floor; the root is then shifted horizontally so that
feet on the floor do not slide.

The check (``--check``) runs the site's loop from the clip's first frame (state and velocities of the reference),
as the benchmark does, through the whole clip: falls (training termination), joint / pelvis / heading error,
how dynamic the clip is, and robustness (initial-state noise and a mid-clip push).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_author_motions import FPS, PREFIX, Author  # noqa: E402

UNIT = 0.0254 / 0.45  # ASF "length 0.45" units -> metres
YUP_TO_ZUP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


# ---------------------------------------------------------------- ASF / AMC


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]  # fmt: skip


def _euler(x: float, y: float, z: float) -> np.ndarray:
    """ASF/AMC rotations: static XYZ (R = Rz Ry Rx)."""
    return _rot("z", z) @ _rot("y", y) @ _rot("x", x)


def parse_asf(path: Path) -> tuple[dict, dict]:
    bones, hierarchy, cur, section = {}, {}, None, None
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            section = line.split()[0]
            continue
        tok = line.split()
        if section == ":bonedata":
            if tok[0] == "begin":
                cur = {"dof": []}
            elif tok[0] == "end":
                bones[cur["name"]] = cur
            elif tok[0] == "name":
                cur["name"] = tok[1]
            elif tok[0] == "direction":
                cur["direction"] = np.array([float(v) for v in tok[1:4]])
            elif tok[0] == "length":
                cur["length"] = float(tok[1]) * UNIT
            elif tok[0] == "axis":
                cur["C"] = _euler(*[float(v) for v in tok[1:4]])
            elif tok[0] == "dof":
                cur["dof"] = tok[1:]
        elif section == ":hierarchy" and tok[0] not in ("begin", "end"):
            hierarchy[tok[0]] = tok[1:]
    return bones, hierarchy


def parse_amc(path: Path) -> list[dict[str, list[float]]]:
    frames, cur = [], None
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ":")):
            continue
        tok = line.split()
        if len(tok) == 1 and tok[0].isdigit():
            cur = {}
            frames.append(cur)
        elif cur is not None:
            cur[tok[0]] = [float(v) for v in tok[1:]]
    return frames


def skeleton_positions(bones: dict, hierarchy: dict, frame: dict) -> dict[str, np.ndarray]:
    """End point of every bone (and "root"), metres, Y-up."""
    r = frame["root"]
    pos = {"root": np.array(r[:3]) * UNIT}
    mats = {"root": _euler(*r[3:6])}

    def visit(parent: str) -> None:
        for name in hierarchy.get(parent, []):
            b = bones[name]
            angles = dict(zip(b["dof"], frame.get(name, [])))
            m = _euler(angles.get("rx", 0.0), angles.get("ry", 0.0), angles.get("rz", 0.0))
            mats[name] = mats[parent] @ b["C"] @ m @ b["C"].T
            pos[name] = pos[parent] + b["length"] * (mats[name] @ b["direction"])
            visit(name)

    visit("root")
    return pos


def human_track(asf: Path, amc: Path, fps: float, start: float, end: float | None) -> dict[str, np.ndarray]:
    """Joint positions (T, 3) at 50 Hz, Z-up metres, heading at the first frame along +x, start at the origin."""
    bones, hierarchy = parse_asf(asf)
    frames = parse_amc(amc)
    names = ["root"] + list(bones)
    raw = np.stack([np.stack([skeleton_positions(bones, hierarchy, f)[n] for n in names]) for f in frames])
    return _align(raw, names, fps, start, end, bones)


# ---------------------------------------------------------------- BVH (100STYLE)

# 100STYLE joint (or "<joint>/end" for its End Site) for each CMU name the retargeter reads (a CMU name is the end
# point of that bone: "lfemur" is the knee).
BVH_TO_CMU = {
    "root": "Hips", "lowerback": "Chest", "upperback": "Chest3", "lowerneck": "Neck",
    "lhipjoint": "LeftHip", "lfemur": "LeftKnee", "ltibia": "LeftAnkle", "lfoot": "LeftToe", "ltoes": "LeftToe/end",
    "rhipjoint": "RightHip", "rfemur": "RightKnee", "rtibia": "RightAnkle", "rfoot": "RightToe", "rtoes": "RightToe/end",
    "lclavicle": "LeftShoulder", "lhumerus": "LeftElbow", "lradius": "LeftWrist",
    "rclavicle": "RightShoulder", "rhumerus": "RightElbow", "rradius": "RightWrist",
}  # fmt: skip


def bvh_track(path: Path, start: float, end: float | None, unit: float = 0.01) -> dict[str, np.ndarray]:
    """``human_track`` for a BVH file (Y-up; 100STYLE: centimetres, 60 fps, Euler channels in the listed order)."""
    joints, stack, motion = [], [], None  # joints: (name, parent index, offset, channels)
    lines = path.read_text(errors="ignore").splitlines()
    for i, raw_line in enumerate(lines):
        tok = raw_line.split()
        if not tok:
            continue
        if tok[0] in ("ROOT", "JOINT"):
            joints.append([tok[1], stack[-1] if stack else -1, None, []])
            pending = len(joints) - 1
        elif tok[0] == "End":
            joints.append([joints[stack[-1]][0] + "/end", stack[-1], None, []])
            pending = len(joints) - 1
        elif tok[0] == "{":
            stack.append(pending)
        elif tok[0] == "}":
            stack.pop()
        elif tok[0] == "OFFSET":
            joints[stack[-1]][2] = np.array([float(x) for x in tok[1:4]]) * unit
        elif tok[0] == "CHANNELS":
            joints[stack[-1]][3] = tok[2:]
        elif tok[0] == "Frame" and tok[1] == "Time:":
            fps = 1.0 / float(tok[2])
            motion = i + 1
            break
    t0 = int(start * fps)
    t1 = len(lines) - motion if end is None else min(int(end * fps) + 2, len(lines) - motion)
    data = np.array([[float(x) for x in lines[motion + t].split()] for t in range(t0, t1)])
    index = {j[0]: k for k, j in enumerate(joints)}
    pos, rot, col = [None] * len(joints), [None] * len(joints), 0
    for k, (name, parent, offset, channels) in enumerate(joints):
        n = len(data)
        r, p = np.broadcast_to(np.eye(3), (n, 3, 3)), np.broadcast_to(offset, (n, 3))
        for ch in channels:
            v = data[:, col]
            col += 1
            if ch.endswith("position"):
                p = p + np.outer(v * unit, np.eye(3)["XYZ".index(ch[0])])
            else:
                r = r @ np.stack([_rot(ch[0].lower(), a) for a in v])
        if parent < 0:
            pos[k], rot[k] = p, r
        else:
            pos[k] = pos[parent] + np.einsum("tij,tj->ti", rot[parent], p)
            rot[k] = rot[parent] @ r
    names = list(BVH_TO_CMU)
    raw = np.stack([pos[index[BVH_TO_CMU[n]]] for n in names], 1)
    length = lambda a, b: float(np.linalg.norm(raw[0, names.index(a)] - raw[0, names.index(b)]))  # noqa: E731
    bones = {"lfemur": {"length": length("lfemur", "lhipjoint")}, "ltibia": {"length": length("ltibia", "lfemur")}}
    return _align(raw, names, fps, start - t0 / fps, None if end is None else end - t0 / fps, bones)


def _align(raw: np.ndarray, names: list[str], fps: float, start: float, end: float | None, bones: dict) -> dict:
    """(frames, joints, 3) Y-up metres at ``fps`` -> 50 Hz Z-up track, start heading +x, root at the origin, floor 0."""
    raw = raw @ YUP_TO_ZUP.T
    t_src = np.arange(len(raw)) / fps
    t_end = min(end if end is not None else t_src[-1], t_src[-1])
    t = np.arange(start, t_end, 1.0 / FPS)
    out = np.stack([np.stack([np.interp(t, t_src, raw[:, j, k]) for k in range(3)], -1) for j in range(len(names))], 1)
    track = {n: out[:, i] for i, n in enumerate(names)}
    left = track["lhipjoint"][0] - track["rhipjoint"][0]
    fwd = np.cross(left, [0, 0, 1.0])
    yaw = -np.arctan2(fwd[1], fwd[0])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    origin = np.array([track["root"][0, 0], track["root"][0, 1], 0.0])
    floor = min(min(track[f"{s}{p}"][:, 2].min() for p in ("foot", "toes", "tibia") if f"{s}{p}" in track) for s in "lr")
    for n in track:
        track[n] = (track[n] - origin) @ rz.T - [0, 0, floor]
    track["_bones"] = bones
    return track


def load_track(c: dict, cmu: str | None, style: str | None) -> dict:
    if c.get("source") == "100style":
        return bvh_track(Path(style) / c["file"], c.get("start", 0.0), c.get("end"))
    subj = c["trial"].split("_")[0]
    return human_track(Path(cmu) / f"{subj}.asf", Path(cmu) / f"{c['trial']}.amc", c["fps"], c.get("start", 0.0), c.get("end"))


def _frame(x: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation with columns forward / left / up from a left-ish vector and an up-ish vector."""
    z = up / np.linalg.norm(up)
    y = x - z * (x @ z)
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], 1)


def _quat(m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, m.ravel())
    return q


# ---------------------------------------------------------------- retargeting


class Retargeter(Author):
    """The G1 following a human track (see the module docstring)."""

    SIDES = (("l", "left"), ("r", "right"))

    def __init__(self, model: mujoco.MjModel, meta: dict):
        super().__init__(model, meta)
        m, d = model, self.d
        self.body = lambda n: m.body(PREFIX + n).id  # noqa: E731
        d.qpos[:] = self.qpos(np.array([0, 0, self.z0]), np.array([1.0, 0, 0, 0]), self.q0)
        mujoco.mj_kinematics(m, d)
        p = lambda n: d.xpos[self.body(n)].copy()  # noqa: E731
        self.len = {
            "thigh": np.linalg.norm(p("left_knee_link") - p("left_hip_pitch_link")),
            "shin": np.linalg.norm(p("left_ankle_roll_link") - p("left_knee_link")),
            "upper": np.linalg.norm(p("left_elbow_link") - p("left_shoulder_pitch_link")),
            "fore": np.linalg.norm(p("left_wrist_roll_link") - p("left_elbow_link")),
        }
        self.leg = self.len["thigh"] + self.len["shin"]
        self.free = list(range(6, model.nv))  # joints only: the root pose comes from the human
        self.feet_geoms = [g for g in range(m.ngeom) if "foot" in m.geom(g).name and m.geom_bodyid[g] > 0]
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 1 and m.geom_contype[g] | m.geom_conaffinity[g]]

    def retarget(self, h: dict, iters: int = 25) -> list[np.ndarray]:
        bones = h["_bones"]
        human_leg = bones["lfemur"]["length"] + bones["ltibia"]["length"]
        scale = self.leg / human_leg
        m, d = self.m, self.d
        q = self.qpos(np.array([0, 0, self.z0]), np.array([1.0, 0, 0, 0]), self.q0)
        frames = []
        root0 = h["root"][0].copy()
        for t in range(len(h["root"])):
            hp = {k: v[t] for k, v in h.items() if k != "_bones"}
            left = hp["lhipjoint"] - hp["rhipjoint"]
            pelvis = _frame(left, hp["lowerback"] - hp["root"])
            torso = _frame(hp["lclavicle"] - hp["rclavicle"], hp["lowerneck"] - hp["upperback"])
            root = np.array([hp["root"][0] * scale, hp["root"][1] * scale, self.z0 + (hp["root"][2] - root0[2]) * scale])
            q[:3], q[3:7] = root, _quat(pelvis)
            dirs = {}
            for s, side in self.SIDES:
                dirs[side] = {
                    "thigh": hp[f"{s}femur"] - hp[f"{s}hipjoint"], "shin": hp[f"{s}tibia"] - hp[f"{s}femur"],
                    "foot": hp[f"{s}toes"] - hp[f"{s}tibia"], "upper": hp[f"{s}humerus"] - hp[f"{s}clavicle"],
                    "fore": hp[f"{s}radius"] - hp[f"{s}humerus"],
                }  # fmt: skip
                for k, v in dirs[side].items():
                    dirs[side][k] = v / max(np.linalg.norm(v), 1e-9)
            for _ in range(iters):
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                res, jac = [], []

                def pos_term(body: str, target: np.ndarray, w: float) -> None:
                    b = self.body(body)
                    jp = np.zeros((3, m.nv))
                    mujoco.mj_jacBody(m, d, jp, None, b)
                    res.append(w * (target - d.xpos[b]))
                    jac.append(w * jp[:, self.free])

                def rot_term(body: str, target: np.ndarray, w: float, axes=(0, 1, 2)) -> None:
                    b = self.body(body)
                    jr = np.zeros((3, m.nv))
                    mujoco.mj_jacBody(m, d, None, jr, b)
                    cur = d.xmat[b].reshape(3, 3)
                    err = np.zeros(3)
                    mujoco.mju_subQuat(err, _quat(target), _quat(cur))
                    err = cur @ err  # world frame
                    res.append(w * err[list(axes)])
                    jac.append(w * jr[list(axes)][:, self.free])

                rot_term("torso_link", torso, 1.0)
                for side, dv in dirs.items():
                    hip = d.xpos[self.body(f"{side}_hip_pitch_link")]
                    knee = hip + self.len["thigh"] * dv["thigh"]
                    pos_term(f"{side}_knee_link", knee, 0.7)
                    pos_term(f"{side}_ankle_roll_link", knee + self.len["shin"] * dv["shin"], 1.0)
                    # Sole: forward from ankle to toe tip, sideways like the pelvis.
                    rot_term(f"{side}_ankle_roll_link", _frame(pelvis[:, 1], np.cross(dv["foot"], pelvis[:, 1])), 0.3)
                    sh = d.xpos[self.body(f"{side}_shoulder_pitch_link")]
                    elbow = sh + self.len["upper"] * dv["upper"]
                    pos_term(f"{side}_elbow_link", elbow, 0.6)
                    pos_term(f"{side}_wrist_roll_link", elbow + self.len["fore"] * dv["fore"], 0.8)
                r, J = np.concatenate(res), np.concatenate(jac)
                reg = 0.02
                step = np.linalg.solve(J.T @ J + reg * np.eye(len(self.free)), J.T @ r)
                for k, dof in enumerate(self.free):
                    i = dof - 6
                    q[7 + i] = np.clip(q[7 + i] + step[k], self.lo[i], self.hi[i])
            frames.append(q.copy())
        return self.ground(frames, h, scale)

    def lowest(self, q: np.ndarray, geoms: list[int]) -> float:
        d = self.d
        d.qpos[:] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        z = np.inf
        for g in geoms:
            size, typ = self.m.geom_size[g], self.m.geom_type[g]
            c, rot = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                z = min(z, c[2] - size[0] - abs(rot[2, 2]) * size[1])
            elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
                z = min(z, c[2] - size[0])
            elif typ == mujoco.mjtGeom.mjGEOM_BOX:
                z = min(z, c[2] - np.abs(rot[2]) @ size)
            elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
                z = min(z, c[2] - abs(rot[2, 2]) * size[1] - np.sqrt(max(0.0, 1 - rot[2, 2] ** 2)) * size[0])
        return z

    def ground(self, frames: list[np.ndarray], h: dict, scale: float) -> list[np.ndarray]:
        """Shift the root height per frame: no geom below the floor; a planted human foot on the floor."""
        planted = np.minimum(*(np.minimum(h[f"{s}toes"][:, 2], h[f"{s}foot"][:, 2]) for s in "lr")) < 0.04
        low_all = np.array([self.lowest(q, self.robot_geoms) for q in frames])
        low_feet = np.array([self.lowest(q, self.feet_geoms) for q in frames])
        dz = np.where(planted, -low_feet, np.maximum(-low_all, 0.0))
        k = np.ones(5) / 5
        dz = np.convolve(np.pad(dz, 2, mode="edge"), k, mode="valid")
        dz = np.maximum(dz, -low_all)  # smoothing must not push anything into the floor
        out = []
        for q, z in zip(frames, dz):
            q = q.copy()
            q[2] += z
            out.append(q)
        return self.deskate(out)

    def deskate(self, frames: list[np.ndarray], floor: float = 0.02) -> list[np.ndarray]:
        """Shift the root horizontally so that a foot on the floor does not slide: the human's proportions differ
        from the G1's, so its planted feet move under the scaled root, and a physical robot cannot follow that."""
        m, d = self.m, self.d
        feet = [self.body(f"{side}_ankle_roll_link") for side in ("left", "right")]
        geoms = [[g for g in self.feet_geoms if m.geom_bodyid[g] == b] for b in feet]
        xy, low = [], []
        for q in frames:
            low.append([self.lowest(q, g) for g in geoms])  # also runs the kinematics for xpos
            xy.append([d.xpos[b][:2].copy() for b in feet])
        xy, low = np.array(xy), np.array(low)
        stance = (low[1:] < floor) & (low[:-1] < floor)
        slide = np.diff(xy, axis=0) * stance[..., None]
        n = stance.sum(1, keepdims=True)
        shift = np.concatenate([np.zeros((1, 2)), np.cumsum(np.where(n > 0, slide.sum(1) / np.maximum(n, 1), 0.0), 0)])
        out = []
        for q, s in zip(frames, shift):
            q = q.copy()
            q[:2] -= s
            out.append(q)
        return out

    def settle(self, frames: list[np.ndarray], seconds: float) -> list[np.ndarray]:
        """A stop appended to a clip that ends in motion (a walk cut mid-stride): over ``seconds`` the root's
        horizontal velocity eases to zero, the pelvis levels to its heading at the standing height and the joints
        blend into the standing pose, the feet kept on the floor. The last frame is then a pose the robot can
        hold, which is what the policy is asked to do past a clip's end (freeze_clip_end)."""
        n = int(round(seconds * FPS))
        a, b = frames[-2], frames[-1]
        v = (b[:2] - a[:2]) * FPS
        w, x, y, z = b[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        level = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        out, pos = list(frames), b[:2].copy()
        for k in range(1, n + 1):
            u = k / n
            s = u * u * (3 - 2 * u)  # smoothstep
            pos = pos + v * (1 - u) ** 2 / FPS
            q = b.copy()
            q[:2] = pos
            q[2] = b[2] + (self.z0 - b[2]) * s
            q[3:7] = _slerp(b[3:7], level, s)
            q[7:] = b[7:] + (self.q0 - b[7:]) * s
            q[2] -= self.lowest(q, self.feet_geoms)  # feet on the floor
            out.append(q)
        return out


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q1 = q1 if q0 @ q1 >= 0 else -q1
    d = float(np.clip(q0 @ q1, -1.0, 1.0))
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------- check


def start_state(model: mujoco.MjModel, row: np.ndarray) -> np.ndarray:
    """[qpos | qvel] of a reference row: angular velocity from world to body frame (MuJoCo free joint)."""
    qvel = np.zeros(model.nv)
    qvel[:3] = row[36:39]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, row[3:7].astype(float))
    qvel[3:6] = rot.reshape(3, 3).T @ row[39:42]
    qvel[6:] = row[42:71]
    return np.concatenate([row[:36], qvel])


def track(model, meta, policy, rows: np.ndarray, hold: float = 1.5, noise: float = 0.0, push=None, seed: int = 0) -> dict:
    """The site's loop from the clip's first frame through its end (then the last frame held still)."""
    H, K = meta["inputs"]["state_hist"]["shape"][1], meta["inputs"]["ref_window"]["shape"][1]
    act_jnt = model.actuator_trnid[:, 0] - 1
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(model)
    s0 = start_state(model, rows[0].astype(float))
    d.qpos[:], d.qvel[:] = s0[: model.nq], s0[model.nq :]
    d.qpos[7:] += noise * rng.standard_normal(29)
    mujoco.mj_forward(model, d)
    states, actions = [np.concatenate([d.qpos, d.qvel])] * H, [np.zeros(29)] * H
    T, last = len(rows), len(rows) - 1
    held = rows[last].copy()
    held[36:] = 0
    ref_at = lambda f: rows[f] if f <= last else held  # noqa: E731
    jerr, perr, fell = [], [], None
    gz = lambda q: -(1 - 2 * (q[1] ** 2 + q[2] ** 2) / (q @ q))  # noqa: E731
    body = model.body(PREFIX + "torso_link").id
    for k in range(T + int(hold * FPS)):
        win = np.stack([ref_at(k + 1 + i) for i in range(K)])  # tick k targets frame k + 1, as training and the benchmark
        a, tgt = policy({"state_hist": np.stack(states[-H:])[None].astype(np.float32), "action_hist": np.stack(actions[-H:])[None].astype(np.float32), "ref_window": win[None].astype(np.float32)})
        d.ctrl[:] = np.asarray(tgt).reshape(29)[act_jnt]
        d.xfrc_applied[:] = 0
        if push and push[0] <= k / FPS < push[0] + push[1]:
            d.xfrc_applied[body, :3] = push[2]
        for _ in range(meta["decimation"]):
            mujoco.mj_step(model, d)
        states.append(np.concatenate([d.qpos, d.qvel]))
        actions.append(np.asarray(a).reshape(29))
        ref = ref_at(k + 1)
        jerr.append(np.abs(d.qpos[7:] - ref[7:36]).mean())
        perr.append(np.linalg.norm(d.qpos[:3] - ref[:3]))
        if fell is None and (abs(ref[2] - d.qpos[2]) > meta["termination"]["height"] or abs(gz(ref[3:7]) - gz(d.qpos[3:7])) > meta["termination"]["tilt"]):
            fell = k
    return {"fell_at_s": None if fell is None else round(fell / FPS, 2), "joint_err": float(np.mean(jerr)), "pelvis_err": float(np.mean(perr)), "pelvis_err_max": float(np.max(perr))}


def dynamism(model: mujoco.MjModel, rt: Retargeter, rows: np.ndarray) -> dict:
    """How dynamic a reference is: root speed, airtime (both feet off the floor), rotation, tilt, height range."""
    feet_low = np.array([rt.lowest(r[:36].astype(float), rt.feet_geoms) for r in rows])
    ang = np.linalg.norm(rows[:, 39:42], axis=1)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (rows[:, 4] ** 2 + rows[:, 5] ** 2), -1, 1)))
    return {
        "seconds": round(len(rows) / FPS, 1),
        "root_speed": round(float(np.linalg.norm(rows[:, 36:38], axis=1).mean()), 2),
        "airtime_s": round(float((feet_low > 0.03).sum() / FPS), 2),
        "rotation_rad": round(float(ang.sum() / FPS), 1),
        "max_tilt_deg": round(float(tilt.max()), 0),
        "root_z_range": round(float(rows[:, 2].max() - rows[:, 2].min()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site")
    parser.add_argument("out_dir", help="where the clip row files go (e.g. a scratch dir; the page reads site/assets/motions)")
    parser.add_argument("--cmu", help="dir with the CMU NN.asf / NN_TT.amc files")
    parser.add_argument("--style", help="dir with the 100STYLE .bvh files (clips with source 100style)")
    parser.add_argument("--clips", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--robust", type=int, default=0, help="robustness runs per clip (initial noise + a mid-clip push)")
    parser.add_argument("--only", nargs="*", help="clip ids to process")
    parser.add_argument("--install", nargs="*", help="clip ids to copy into site/assets/motions and list in index.json")
    parser.add_argument("--default", help="the clip the page opens on")
    parser.add_argument("--settle", type=float, default=1.2, help="s of a stop appended to clips that end in motion (0: none)")
    args = parser.parse_args()
    site, out = Path(args.site), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((site / "assets/policy/meta.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(site / "assets/robot/scene.mjb"))
    rt = Retargeter(model, meta)
    clips = [c for c in json.loads(Path(args.clips).read_text()) if not args.only or c["id"] in args.only]
    policy = None
    if args.check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(site / "assets/policy/policy.onnx"), providers=["CPUExecutionProvider"])
        policy = lambda x: sess.run(["action", "joint_target"], x)  # noqa: E731
    report = {}
    for c in clips:
        h = load_track(c, args.cmu, args.style)
        frames = rt.retarget(h)
        end_speed = np.linalg.norm(np.diff(np.stack(frames[-26:])[:, :2], axis=0), axis=1).mean() * FPS
        if args.settle > 0 and end_speed > SETTLE_ABOVE:  # ends in motion: append a stop the robot can hold
            frames = rt.settle(frames, args.settle)
            SETTLED.add(c["id"])
        rows = rt.rows(frames)
        rows[0, 36:] = rows[1, 36:]  # keep the first frame's velocities (rows() zeroes them for looping clips)
        rows[-1, 36:] = rows[-2, 36:]
        (out / f"{c['id']}.bin").write_bytes(rows.astype(np.float32).tobytes())
        r = {"dynamism": dynamism(model, rt, rows)}
        if policy:
            r["track"] = track(model, meta, policy, rows)
            if args.robust:
                rng = np.random.default_rng(1)
                runs = []
                for k in range(args.robust):
                    ang = rng.uniform(0, 2 * np.pi)
                    push = (len(rows) / FPS / 2, 0.3, (100 * np.cos(ang), 100 * np.sin(ang), 0.0))
                    runs.append(track(model, meta, policy, rows, noise=0.02, push=push, seed=k)["fell_at_s"])
                r["robust_falls"] = sum(x is not None for x in runs)
                r["robust_runs"] = args.robust
        report[c["id"]] = r
        print(c["id"], json.dumps(r), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    if args.install is not None:
        install(site, out, [c for c in json.loads(Path(args.clips).read_text()) if c["id"] in args.install], args.default)


SETTLE_ABOVE = 0.10  # m/s, root speed over a clip's last 0.5 s above which it gets a stop appended
SETTLED: set[str] = set()  # clip ids that got one (this run)

CMU_LICENCE = "CMU Graphics Lab Motion Capture Database terms: free for all uses; may be copied, modified and redistributed; not to be resold, even converted"
STYLE_URL = "https://doi.org/10.5281/zenodo.8127870"
STYLE_LICENCE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"


def install(site: Path, rows_dir: Path, clips: list[dict], default: str | None) -> None:
    """Copy the clips' rows into site/assets/motions and list them in index.json (replacing earlier entries of this
    tool), between the authored motions and the poses."""
    import shutil

    motions = site / "assets/motions"
    index = json.loads((motions / "index.json").read_text())
    ours = {c["id"] for c in clips}
    kept = [m for m in index["motions"] if m["id"] not in ours and not m.get("retargeted_by") == "tools/web_retarget.py"]
    for m in kept:
        m.pop("default", None)
    entries = []
    for c in clips:
        shutil.copyfile(rows_dir / f"{c['id']}.bin", motions / f"{c['id']}.bin")
        window = f", {c.get('start', 0.0):g}-{c['end']:g} s" if c.get("end") is not None else ""
        if c.get("source") == "100style":
            src = {
                "group": "Walking styles (100STYLE)",
                "source": f"100STYLE dataset (Mason, Starke, Komura 2022), {c['file']}{window}",
                "source_url": STYLE_URL, "licence": STYLE_LICENCE,
            }  # fmt: skip
        else:
            subj, trial = c["trial"].split("_")
            src = {
                "group": "Motion capture (CMU)",
                "source": f"CMU Graphics Lab Motion Capture Database, subject {int(subj)}, trial {int(trial)}{window}",
                "source_url": f"http://mocap.cs.cmu.edu/search.php?subjectnumber={int(subj)}", "licence": CMU_LICENCE,
            }  # fmt: skip
        entries.append({
            "id": c["id"], "name": c["name"], "kind": "clip", "file": f"{c['id']}.bin", "loop": False,
            "default": c["id"] == default, **src,
            "retargeting": "keypoint IK to the G1 with its own segment lengths, root scaled by leg length; ground contact fixed, foot sliding removed"
                           + ("; a short stop into a standing pose appended (the source keeps moving)" if c["id"] in SETTLED else "")
                           + "; modified from the original",
            "retargeted_by": "tools/web_retarget.py",
        })  # fmt: skip
    poses = [m for m in kept if m["kind"] == "pose"]
    others = [m for m in kept if m["kind"] != "pose"]
    index["motions"] = others + entries + poses
    index["fields"]["group"] = "menu group on the page"
    index["fields"]["default"] = "the motion the page opens on"
    (motions / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
