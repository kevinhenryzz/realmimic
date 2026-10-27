"""MuJoCo scene of the browser demo (``site/assets/robot``).

The training scene (mjlab's G1 from the RobotSpec + ground plane, ``supermimic.envs.make_env_cfg``)
with the training solver options and the evaluation-protocol foot friction (1.0), saved as a
compiled binary model (``scene.mjb``, MuJoCo 3.16): MJCF export rounds numbers to 6 significant
digits (armature, gains), the binary keeps them exact. The visual meshes carry no mass and no
collision; they are left out of the physics model (``check_physics_unchanged`` verifies the rollout
is bit-identical) and shipped to the renderer welded, quantized and meshopt-compressed (``visual.json`` +
``meshes.bin``).
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import mujoco
import numpy as np

ROBOT_PREFIX = "robot/"
# Fields that must match the training scene exactly (geoms: the non-mesh ones).
PHYSICS_FIELDS = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "jnt_range", "dof_armature",
    "dof_damping", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_trnid",
    "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority",
)  # fmt: skip


def eval_env_cfg(robot: str = "g1", self_collision: bool = True):
    """Environment config of the benchmark protocol (no noise, no domain randomization)."""
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg

    args = SuperMimicEnvCfgArgs(
        npz_files=[], num_envs=1, obs_noise=False, reset_noise=False, domain_randomization=False,
        action_noise_std=0.0, record_transitions=False, robot=robot, self_collision=self_collision,
    )  # fmt: skip
    return make_env_cfg(args)


def training_scene(robot: str = "g1", self_collision: bool = True):
    """(mjlab Scene, env cfg) with solver options and foot friction applied to the scene's spec."""
    from mjlab.scene import Scene

    from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX

    cfg = eval_env_cfg(robot, self_collision)
    scene = Scene(cfg.scene, device="cpu")
    spec = scene.spec
    cfg.sim.mujoco.apply(types.SimpleNamespace(opt=spec.option))  # MujocoCfg only writes ``opt.*``
    for geom in spec.geoms:
        if re.fullmatch(FOOT_GEOM_REGEX, geom.name.removeprefix(ROBOT_PREFIX)):
            geom.friction[0] = 1.0
    return scene, cfg


def reference_model(robot: str = "g1", self_collision: bool = True) -> mujoco.MjModel:
    """The training scene compiled with ``MujocoCfg.apply`` on the model (how mjlab sets the options)."""
    scene, cfg = training_scene(robot, self_collision)
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    return model


def weld(vertices: np.ndarray, faces: np.ndarray, fraction: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices (STL stores three per triangle), then optionally simplify to ``fraction``
    of the faces with meshoptimizer's simplifier (after welding, so it sees the connected surface)."""
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    v, f = np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32)
    if fraction >= 1.0:
        return v, f
    mo = _meshopt()
    out = np.zeros(f.size, dtype=np.uint32)
    n = mo.simplify(out, f.ravel(), v, target_index_count=3 * max(200, round(len(f) * fraction)), target_error=0.01)
    used, idx = np.unique(out[:n], return_inverse=True)  # drop the vertices the simplified mesh no longer uses
    return v[used], idx.reshape(-1, 3).astype(np.uint32)


def _meshopt():
    try:
        import meshoptimizer
    except ImportError as e:
        raise ImportError("mesh export needs meshoptimizer: uv run --with meshoptimizer==0.2.30a0 ...") from e
    return meshoptimizer


def encode_mesh(v: np.ndarray, f: np.ndarray) -> tuple[bytes, bytes, list[float], list[float]]:
    """meshopt-encoded vertex and index buffers. Positions are quantized to uint16 inside the mesh's
    bounding box (padded to 4 components, the codec's stride unit): (vertices, indices, min, extent)."""
    mo = _meshopt()
    idx = np.ascontiguousarray(f.ravel(), dtype=np.uint32)
    opt = np.zeros_like(idx)
    mo.optimize_vertex_cache(opt, idx, vertex_count=len(v))
    verts = np.zeros((len(v), 3), dtype=np.float32)
    nv = mo.optimize_vertex_fetch(verts, opt, np.ascontiguousarray(v, dtype=np.float32))
    verts = verts[:nv]
    lo = verts.min(0)
    ext = np.maximum(verts.max(0) - lo, 1e-9)
    q = np.zeros((nv, 4), dtype=np.uint16)
    q[:, :3] = np.round((verts - lo) / ext * 65535)
    return mo.encode_vertex_buffer(q), mo.encode_index_buffer(opt, vertex_count=nv), lo.tolist(), ext.tolist()


def mesh_sources(scene) -> dict[str, Path]:
    """Mesh file name -> source path (entity meshes compile straight from disk)."""
    return {
        mesh.file: Path(ent.spec.modelfiledir) / mesh.compiler.meshdir / mesh.file
        for ent in scene.entities.values()
        for mesh in ent.spec.meshes
    }


def visual_meshes(scene, fraction: float = 1.0) -> tuple[list[dict], bytes]:
    """Visual meshes as compiled by MuJoCo (vertices in the mesh frame of the compiled model, pose
    relative to the body), welded and meshopt-encoded: (entries, concatenated buffers)."""
    import trimesh

    spec = scene.spec.copy()
    if fraction < 1.0:
        for name, path in mesh_sources(scene).items():
            mesh = trimesh.load(path, file_type="stl", process=False)
            v, f = weld(np.asarray(mesh.vertices), np.asarray(mesh.faces), fraction)
            spec.assets[name] = trimesh.Trimesh(v, f, process=False).export(file_type="stl")
    m = spec.compile()
    entries, blob = [], bytearray()

    def put(b: bytes) -> list[int]:
        blob.extend(b"\0" * (-len(blob) % 4))
        blob.extend(b)
        return [len(blob) - len(b), len(b)]

    for g in np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        f = m.mesh_face[m.mesh_faceadr[k] : m.mesh_faceadr[k] + m.mesh_facenum[k]]
        v, f = weld(v, f)
        assert len(v) < 65536, m.mesh(k).name
        vbuf, ibuf, lo, ext = encode_mesh(v, f)
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        entries.append({
            "body": m.body(m.geom_bodyid[g]).name.removeprefix(ROBOT_PREFIX),
            "mesh": m.mesh(k).name.removeprefix(ROBOT_PREFIX),
            "pos": m.geom_pos[g].tolist(), "quat": m.geom_quat[g].tolist(), "rgba": [round(float(c), 4) for c in rgba],
            "vert_count": int(len(v)), "face_count": int(len(f)), "pos_min": lo, "pos_extent": ext,
            "vertices": put(vbuf), "indices": put(ibuf),
        })  # fmt: skip
    return entries, bytes(blob)


def physics_model(scene, cfg) -> mujoco.MjModel:
    """The training scene without its visual mesh geoms, options applied (mjlab ``MujocoCfg.apply``)."""
    spec = scene.spec.copy()
    for geom in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]:
        if geom.contype or geom.conaffinity:
            raise AssertionError(f"mesh geom {geom.name} collides; it cannot be dropped from the physics model")
        spec.delete(geom)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    model = spec.compile()
    cfg.sim.mujoco.apply(model)
    return model


def check_physics_unchanged(model: mujoco.MjModel, robot: str = "g1", self_collision: bool = True, steps: int = 400) -> float:
    """Compare ``model`` with the training scene: options, inertial/actuator fields and a PD-driven
    rollout from the default pose must agree exactly. Returns the max |qpos| difference."""
    ref = reference_model(robot, self_collision)
    for name in ("timestep", "integrator", "cone", "jacobian", "solver", "iterations", "tolerance", "ls_iterations", "ls_tolerance", "ccd_iterations", "impratio", "gravity"):
        a, b = np.asarray(getattr(ref.opt, name)), np.asarray(getattr(model.opt, name))
        if not np.array_equal(a, b):
            raise AssertionError(f"option {name}: training {a} vs exported {b}")
    for name in PHYSICS_FIELDS:
        a, b = getattr(ref, name), getattr(model, name)
        if name.startswith("geom_"):
            a = a[ref.geom_type != mujoco.mjtGeom.mjGEOM_MESH]
        if not np.array_equal(a, b):
            raise AssertionError(f"model field {name} differs from the training scene")
    traj = []
    for m in (ref, model):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        q0 = d.qpos[7:].copy()
        jnt = m.actuator_trnid[:, 0] - 1
        out = np.zeros((steps, m.nq))
        for k in range(steps):
            d.ctrl[:] = q0[jnt] + 0.3 * np.sin(0.02 * k + np.arange(m.nu))
            mujoco.mj_step(m, d)
            out[k] = d.qpos
        traj.append(out)
    return float(np.abs(traj[0] - traj[1]).max())


def export_scene(out_dir: str | Path, robot: str = "g1", self_collision: bool = True, mesh_fraction: float = 1.0) -> dict:
    """Write ``scene.mjb`` (physics, exact), ``visual.json`` + ``meshes.bin`` (compressed visual meshes)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene, cfg = training_scene(robot, self_collision)
    model = physics_model(scene, cfg)
    diff = check_physics_unchanged(model, robot, self_collision)
    if diff != 0.0:
        raise AssertionError(f"exported scene rollout differs from the training scene by {diff}")
    mujoco.mj_saveModel(model, str(out_dir / "scene.mjb"), None)
    entries, blob = visual_meshes(scene, mesh_fraction)
    (out_dir / "meshes.bin").write_bytes(blob)
    visual = {
        "format": "realmimic-web-visual/2",
        "layout": (
            "per geom, [offset, bytes] into meshes.bin of meshoptimizer-encoded buffers (vertex / index codec): "
            "vertices = vert_count x 4 uint16 (x, y, z = pos_min + q / 65535 * pos_extent; 4th unused), "
            "indices = face_count x 3 triangle indices local to the mesh"
        ),
        "num_verts": sum(e["vert_count"] for e in entries), "num_faces": sum(e["face_count"] for e in entries),
        "geoms": entries,
    }  # fmt: skip
    (out_dir / "visual.json").write_text(json.dumps(visual, indent=1))
    return {
        "faces": visual["num_faces"],
        "mjb_bytes": (out_dir / "scene.mjb").stat().st_size,
        "mesh_bytes": (out_dir / "meshes.bin").stat().st_size,
        "rollout_max_abs_diff": diff,
    }


# Stand-in shapes for payload items that have no visual mesh: box half-sizes [m] from the spec sheets
# (Livox Mid-360 65 x 65 x 60 mm, RealSense D435 90 x 25 x 25 mm, ReSpeaker XVF3800 board ~70 x 70 x 10 mm).
PAYLOAD_BOXES = {
    "livox_mid360": (0.0325, 0.0325, 0.03),
    "realsense_d435": (0.0125, 0.045, 0.0125),
    "respeaker_xvf3800": (0.035, 0.035, 0.005),
}


HAND_BASE_X = 0.0675  # measured hand mount on the wrist-yaw x axis (realmimic.robots.hands)
WRIST_FLANGE_X = 0.0415  # wrist flange on the same axis; a 26 mm adapter ring fills the gap to the hand
ADAPTER_RADIUS = 0.026  # approximate


def _rpy(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def hand_mesh(urdf: str | Path, side: str, pose: tuple[float, ...], base_x: float = HAND_BASE_X, fraction: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """The hand below ``<side>_wrist_yaw_link`` of a URDF (Unitree's g1_with_brainco_hand) as one mesh in
    that link's frame: fingers at ``pose`` (six SDK actuator positions, 0 open, 1 closed, mapped onto the
    actuated joints' ranges; mimic joints follow), hand base at ``base_x`` on the x axis as mounted."""
    import xml.etree.ElementTree as ET

    import trimesh

    urdf = Path(urdf)
    robot = ET.parse(urdf).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = {j.get("name"): j for j in robot.findall("joint")}
    q = {}
    actuated = ("thumb_metacarpal", "thumb_proximal", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal")
    for name, value in zip(actuated, pose, strict=True):
        lim = joints[f"{side}_{name}_joint"].find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        q[f"{side}_{name}_joint"] = lo + float(np.clip(value, 0, 1)) * (hi - lo)
    for name, j in joints.items():
        mimic = j.find("mimic")
        if mimic is not None and mimic.get("joint") in q:
            q[name] = float(mimic.get("multiplier", 1)) * q[mimic.get("joint")] + float(mimic.get("offset", 0))
    children = {}
    for j in joints.values():
        children.setdefault(j.find("parent").get("link"), []).append(j)

    def origin(el) -> tuple[np.ndarray, np.ndarray]:
        if el is None:
            return np.zeros(3), np.eye(3)
        return np.array([float(v) for v in el.get("xyz", "0 0 0").split()]), _rpy([float(v) for v in el.get("rpy", "0 0 0").split()])

    verts, faces, n = [], [], 0

    def visit(link: str, pos: np.ndarray, rot: np.ndarray, root: bool) -> None:
        nonlocal n
        visual = links[link].find("visual")
        if not root and visual is not None and visual.find("geometry/mesh") is not None:
            v_pos, v_rot = origin(visual.find("origin"))
            m = trimesh.load(urdf.parent / visual.find("geometry/mesh").get("filename"), file_type="stl", process=False)
            v = (rot @ (v_rot @ np.asarray(m.vertices).T + v_pos[:, None])).T + pos
            verts.append(v)
            faces.append(np.asarray(m.faces) + n)
            n += len(v)
        for j in children.get(link, []):
            j_pos, j_rot = origin(j.find("origin"))
            if j.get("name") == f"{side}_base_joint":
                j_pos = np.array([base_x, j_pos[1], j_pos[2]])
            if j.get("type") in ("revolute", "continuous") and q.get(j.get("name")):
                axis = np.array([float(a) for a in j.find("axis").get("xyz").split()])
                axis /= np.linalg.norm(axis)
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                a = q[j.get("name")]
                j_rot = j_rot @ (np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k)
            visit(j.find("child").get("link"), pos + rot @ j_pos, rot @ j_rot, False)

    visit(f"{side}_wrist_yaw_link", np.zeros(3), np.eye(3), True)
    return weld(np.concatenate(verts), np.concatenate(faces), fraction)


def export_variant(out_dir: str | Path, robot: str, base: str = "g1", self_collision: bool = True, hand_urdf: str | Path | None = None) -> dict:
    """A robot variant for the demo's robot switch, drawn with the ``base`` robot's meshes: ``scene.mjb``
    (its exact physics) + ``variant.json`` (base meshes it lacks; replaced hands as meshes from
    ``hand_urdf`` posed and mounted like the physics model, else their collision capsules; boxes for
    payload items) + ``meshes.bin`` when there are hand meshes."""
    from realmimic.robots.hands import HAND_MODELS
    from realmimic.robots.spec import get_robot_spec

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene, cfg = training_scene(robot, self_collision)
    model = physics_model(scene, cfg)
    diff = check_physics_unchanged(model, robot, self_collision)
    if diff != 0.0:
        raise AssertionError(f"exported scene rollout differs from the training scene by {diff}")
    mujoco.mj_saveModel(model, str(out_dir / "scene.mjb"), None)

    def drawn(s) -> set[str]:
        return {g.meshname.removeprefix(ROBOT_PREFIX) for g in s.spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}

    hide = sorted(drawn(training_scene(base, self_collision)[0]) - drawn(scene))
    spec = get_robot_spec(robot)
    extras, meshes, blob = [], [], bytearray()
    if spec.hands != "stock" and hand_urdf is not None:
        for side in ("left", "right"):
            v, f = hand_mesh(hand_urdf, side, HAND_MODELS[spec.hands][side].pose)
            vbuf, ibuf, lo, ext = encode_mesh(v, f)
            entry = {"body": f"{side}_wrist_yaw_link", "mesh": f"{side}_{spec.hands}_hand", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
                     "rgba": [0.2, 0.2, 0.2, 1.0], "vert_count": int(len(v)), "face_count": int(len(f)), "pos_min": lo, "pos_extent": ext}  # fmt: skip
            for key, b in (("vertices", vbuf), ("indices", ibuf)):
                blob.extend(b"\0" * (-len(blob) % 4))
                entry[key] = [len(blob), len(b)]
                blob.extend(b)
            meshes.append(entry)
            # The adapter ring between the wrist flange (0.0415 m) and the hand base has no mesh: a cylinder.
            extras.append({"name": f"{side}_hand_adapter", "body": f"{side}_wrist_yaw_link", "type": "cylinder",
                           "pos": [(WRIST_FLANGE_X + HAND_BASE_X) / 2, 0.0, 0.0], "quat": [0.7071068, 0.0, 0.7071068, 0.0],
                           "size": [ADAPTER_RADIUS, (HAND_BASE_X - WRIST_FLANGE_X) / 2]})  # fmt: skip
        (out_dir / "meshes.bin").write_bytes(bytes(blob))
    elif spec.hands != "stock":
        for g in range(model.ngeom):
            name = model.geom(g).name.removeprefix(ROBOT_PREFIX)
            if re.fullmatch(r"(left|right)_(hand|thumb)_collision", name):
                assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                extras.append({
                    "name": name, "body": model.body(model.geom_bodyid[g]).name.removeprefix(ROBOT_PREFIX), "type": "capsule",
                    "pos": model.geom_pos[g].tolist(), "quat": model.geom_quat[g].tolist(), "size": model.geom_size[g, :2].tolist(),
                })  # fmt: skip
    for item in spec.payloads:
        if item.name in PAYLOAD_BOXES:
            extras.append({"name": item.name, "body": item.body, "type": "box", "pos": list(item.pos), "quat": [1.0, 0.0, 0.0, 0.0], "size": list(PAYLOAD_BOXES[item.name])})
    variant = {
        "format": "realmimic-web-variant/1",
        "robot": robot,
        "description": spec.description,
        "mass": float(model.body_mass.sum()),
        "hide_meshes": hide,
        "geoms": meshes,
        "geoms_note": "hand meshes from the hand URDF (open pose, measured mount), in meshes.bin; layout as visual.json" if meshes else None,
        "extras": extras,
        "extras_note": "stand-in shapes (no visual mesh): hand adapter rings, or hand collision capsules without hand meshes; payload boxes sized from spec sheets",
    }
    (out_dir / "variant.json").write_text(json.dumps(variant, indent=1))
    return {"mjb_bytes": (out_dir / "scene.mjb").stat().st_size, "hide": hide, "extras": len(extras), "hand_faces": sum(m["face_count"] for m in meshes),
            "mesh_bytes": len(blob), "rollout_max_abs_diff": diff}  # fmt: skip
