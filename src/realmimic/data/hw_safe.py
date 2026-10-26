"""Hardware-safe BONES-SEED subset for a first real-G1 campaign (gantry, SONIC planner modes).

A clip is kept only if (a) its metadata puts it in one of the behaviours the SONIC kinematic
planner produces (idle, walk, turn, moderate jog, squat, kneel, crouch, standing gestures) and no
exclusion keyword fires, and (b) its kinematics, computed from the converted 50 Hz npz arrays,
pass every check in :class:`Thresholds`. Clips that hold a squat are moved to the squat category
by :func:`refine_category` (BONES names almost none of its squats).

Thresholds were checked against the per-category distributions of the full Appendix-E set
(numbers quoted in the field comments); every reference value cited there is from BONES-SEED
converted with ``scripts/data/csv_to_npz_batch.py``.

Ground contact is evaluated on the G1 collision geometry (spheres and capsules of the mjlab
MJCF) posed with the npz link states, so no MuJoCo call is needed per frame.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FPS = 50.0

# --------------------------------------------------------------------------- robot constants

# Motor no-load velocity limits (rad/s) of mjlab's G1 actuators
# (mjlab/asset_zoo/robots/unitree_g1/g1_constants.py, ElectricActuator.velocity_limit):
# 5020 = 37, 7520-14 = 32, 7520-22 = 20, 4010 = 22. Waist roll/pitch and the ankles are
# 4-bar linkages driven by two 5020 motors at a nominal 1:1 ratio (mjlab's own assumption),
# so they inherit the 5020 limit.
JOINT_VELOCITY_LIMITS: tuple[tuple[str, float], ...] = (
    (r".*_hip_pitch_joint", 32.0),
    (r".*_hip_roll_joint", 20.0),
    (r".*_hip_yaw_joint", 32.0),
    (r".*_knee_joint", 20.0),
    (r".*_ankle_(pitch|roll)_joint", 37.0),
    (r"waist_yaw_joint", 32.0),
    (r"waist_(roll|pitch)_joint", 37.0),
    (r".*_shoulder_(pitch|roll|yaw)_joint", 37.0),
    (r".*_elbow_joint", 37.0),
    (r".*_wrist_roll_joint", 37.0),
    (r".*_wrist_(pitch|yaw)_joint", 22.0),
)

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
# Knee contact, per side: the shin capsule and the thigh capsule, whose lower end covers the knee
# cap (it is ~1.5 cm above the floor when kneeling upright on the shin).
KNEE_GROUPS = (("left_knee_link", "left_hip_yaw_link"), ("right_knee_link", "right_hip_yaw_link"))
KNEE_BODIES = KNEE_GROUPS[0] + KNEE_GROUPS[1]
# The ankle linkage-brace capsule hangs off the knee link just above the ankle: it sits 2-4 cm
# above the floor in any flat-footed pose (standing or squatting), so it would read as "knee
# contact". It can only reach the ground with the shin flat, i.e. when the shin capsule touches
# too, so ground-contact checks ignore it.
CONTACT_IGNORED_GEOMS = ("left_linkage_brace_collision", "right_linkage_brace_collision")
HAND_BODIES = (
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)
KNEE_JOINTS = ("left_knee_joint", "right_knee_joint")


def joint_velocity_limits(joint_names: Sequence[str]) -> np.ndarray:
    """Per-joint velocity limit (rad/s) in ``joint_names`` order."""
    out = np.empty(len(joint_names))
    for i, name in enumerate(joint_names):
        for pattern, limit in JOINT_VELOCITY_LIMITS:
            if re.fullmatch(pattern, name):
                out[i] = limit
                break
        else:
            raise KeyError(f"no velocity limit for joint {name!r}")
    return out


@dataclass(frozen=True)
class CollisionGeom:
    """A sphere (``half_length == 0``) or capsule rigidly attached to a body."""

    name: str
    body: str
    pos: tuple[float, float, float]  # centre in the body frame
    axis: tuple[float, float, float]  # capsule axis (unit) in the body frame
    radius: float
    half_length: float


def g1_collision_geoms(xml_path: str | Path | None = None) -> tuple[CollisionGeom, ...]:
    """The ``*_collision`` spheres/capsules of the mjlab G1 MJCF, in body frames."""
    import mujoco

    if xml_path is None:
        import mjlab

        xml_path = Path(mjlab.__file__).parent / "asset_zoo/robots/unitree_g1/xmls/g1.xml"
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    geoms = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.endswith("_collision"):
            continue
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        gtype = int(m.geom_type[g])
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            half = 0.0
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            half = float(m.geom_size[g, 1])
        else:
            raise ValueError(f"unsupported collision geom type for {name}")
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, m.geom_quat[g])
        axis = mat.reshape(3, 3)[:, 2]
        geoms.append(
            CollisionGeom(
                name=name,
                body=body,
                pos=tuple(float(x) for x in m.geom_pos[g]),
                axis=tuple(float(x) for x in axis),
                radius=float(m.geom_size[g, 0]),
                half_length=half,
            )
        )
    return tuple(geoms)


def _rot_z_row(quat_wxyz: np.ndarray) -> np.ndarray:
    """Third row of the rotation matrix of wxyz quaternions: (..., 3)."""
    w, x, y, z = (quat_wxyz[..., i] for i in range(4))
    return np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


def lowest_point(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_names: Sequence[str],
    geoms: Sequence[CollisionGeom],
    bodies: Sequence[str],
) -> np.ndarray:
    """Lowest world z over the collision geoms of ``bodies`` per frame: (T,)."""
    index = {n: i for i, n in enumerate(body_names)}
    out = np.full(body_pos_w.shape[0], np.inf)
    for g in geoms:
        if g.body not in bodies:
            continue
        b = index[g.body]
        rz = _rot_z_row(body_quat_w[:, b])
        cz = body_pos_w[:, b, 2] + rz @ np.asarray(g.pos)
        z = cz - g.half_length * np.abs(rz @ np.asarray(g.axis)) - g.radius
        out = np.minimum(out, z)
    return out


# --------------------------------------------------------------------------- kinematics


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average along axis 0 with edge padding (same length as ``x``)."""
    if n <= 1 or x.shape[0] < 2:
        return x
    pad = [(n // 2, n - 1 - n // 2)] + [(0, 0)] * (x.ndim - 1)
    c = np.cumsum(np.pad(x, pad, mode="edge"), axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    return (c[n:] - c[:-n]) / n


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    if not mask.any():
        return 0
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return int((np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]).max())


@dataclass(frozen=True)
class Thresholds:
    """Kinematic limits of the hardware-safe subset (see module docstring and README of the tool).

    Every limit is applied to the 50 Hz npz arrays after a centred moving average of
    ``smooth_s`` (0.1 s = 5 frames): this removes single-frame retargeting spikes while keeping
    everything the 50 Hz policy could actually be asked to track.
    """

    smooth_s: float = 0.1
    # Peak horizontal pelvis speed. SONIC's planner walks at 0.5-1.5 m/s and Unitree rates the
    # G1 at 2 m/s top speed, so 2.0 m/s keeps walks and a moderate jog and drops runs/sprints
    # (BONES jog clips: median peak 1.8 m/s, p90 2.4 m/s).
    max_pelvis_speed_xy: float = 2.0
    # Peak vertical pelvis speed: squatting/kneeling down stays below 1.0 m/s (BONES kneel and
    # crouch clips: p99 0.96-1.0 m/s); dropping to the floor or jumping exceeds it (a 7 cm free
    # fall already reaches 1.2 m/s).
    max_pelvis_speed_z: float = 1.2
    # Peak pelvis angular speed: walking turns and turns in place stay below ~4.5 rad/s (BONES
    # turn clips: p90 4.4 rad/s); spins and pirouettes are faster.
    max_pelvis_ang_speed: float = 5.0
    # Peak joint speed as a fraction of the motor no-load velocity limit. Motor torque falls
    # roughly linearly to zero at the no-load speed, so a reference at 60 % of it leaves ~40 % of
    # peak torque for feedback and for correcting tracking error. Walking stays below it (BONES
    # walk clips: p99 0.61, the knee is almost always the limiting joint).
    joint_speed_fraction: float = 0.6
    # Lowest pelvis height. Kneeling upright puts the G1 pelvis at ~0.40 m (BONES kneel clips)
    # and a deep squat at ~0.3 m; sitting on heels, on the floor or lying puts it below ~0.25 m.
    min_pelvis_height: float = 0.25
    # Foot/ground contact: lowest point of the foot capsules within this height (retarget noise:
    # standing BONES clips sit at 0-2 cm, the paper's floating-feet filter uses 2 cm).
    foot_contact_z: float = 0.03
    # Longest flight phase (no foot, knee or hand on the ground). Jogs have flight phases
    # <= ~0.12 s (BONES jog clips: p99); 0.2 s is a ballistic apex of g t^2 / 8 = 5 cm, our
    # "modest" jump height.
    max_airborne_s: float = 0.2
    # Non-foot contact: lowest point of a body's collision capsules below this height.
    contact_z: float = 0.02
    # Hands may touch the ground briefly (e.g. steadying while kneeling down).
    max_hand_contact_run_s: float = 0.5
    max_hand_contact_s: float = 1.0
    # Any other non-foot body (pelvis, hips, torso, head, arms): only a few frames of
    # retargeting noise are tolerated.
    max_body_contact_s: float = 0.1
    # Posture tagging (first/last ``posture_window_s`` of the clip, medians). Standing: pelvis
    # >= 0.62 m (nominal G1 stance 0.76-0.79 m), pelvis tilt < 37 deg, a foot on the ground and the
    # straighter knee < 0.9 rad (so mid-stride and one-leg stances count). Squat: pelvis lower,
    # both knees >= 1.0 rad, both feet on the ground, no knee contact. Kneel: a shin/knee capsule
    # within ``kneel_contact_z`` of the ground.
    posture_window_s: float = 0.3
    stand_min_pelvis_z: float = 0.62
    stand_max_knee: float = 0.9
    squat_min_knee: float = 1.0
    upright_min: float = 0.8
    kneel_contact_z: float = 0.03
    # A manipulation/idle/gesture/other clip that holds a squat posture this long is a squat clip
    # (picking things up from the floor is where BONES keeps most of its squats).
    squat_hold_s: float = 0.5


# Threshold fields that change :func:`compute_kinematics` output (the rest only change decisions).
METRIC_SETTINGS = (
    "smooth_s",
    "foot_contact_z",
    "contact_z",
    "min_pelvis_height",
    "posture_window_s",
    "stand_min_pelvis_z",
    "stand_max_knee",
    "squat_min_knee",
    "upright_min",
    "kneel_contact_z",
)


@dataclass
class ClipKinematics:
    """Per-clip kinematic summary (all peaks after smoothing)."""

    frames: int
    duration_s: float
    pelvis_speed_xy: float
    pelvis_speed_z: float
    pelvis_ang_speed: float
    joint_speed_ratio: float
    joint_speed_joint: str
    pelvis_z_min: float
    airborne_s: float
    knee_contact_s: float
    hand_contact_run_s: float
    hand_contact_s: float
    body_contact_s: float
    body_contact_body: str
    squat_s: float  # longest continuous squat posture
    start_posture: str
    end_posture: str

    def as_dict(self) -> dict:
        return asdict(self)


POSTURES = ("standing", "squat", "kneel", "other")


def _postures(
    pelvis_z: np.ndarray,
    upright: np.ndarray,
    knee: np.ndarray,
    feet_contact: np.ndarray,
    knee_low: np.ndarray,
    th: Thresholds,
) -> np.ndarray:
    """Posture label per row of (already windowed or per-frame) signals; see :class:`Thresholds`."""
    ok = (pelvis_z >= th.min_pelvis_height) & (upright >= 0.5)
    kneel = ok & (knee_low < th.kneel_contact_z).any(axis=-1)
    stand = (
        ok
        & ~kneel
        & (pelvis_z >= th.stand_min_pelvis_z)
        & (upright >= th.upright_min)
        & (knee.min(axis=-1) <= th.stand_max_knee)
        & feet_contact.any(axis=-1)
    )
    squat = (
        ok
        & ~kneel
        & (pelvis_z < th.stand_min_pelvis_z)
        & (knee.min(axis=-1) >= th.squat_min_knee)
        & feet_contact.all(axis=-1)
    )
    return np.select([kneel, stand, squat], ["kneel", "standing", "squat"], "other")


def _window_posture(sig: Sequence[np.ndarray], th: Thresholds) -> str:
    """Posture of a short window: medians of the continuous signals, majority of the contacts."""
    z, up, knee, feet, knee_low = sig
    feet = feet.mean(axis=0) >= 0.5
    med = (np.median(z), np.median(up), np.median(knee, 0), feet, np.median(knee_low, 0))
    return str(_postures(*(np.asarray(s)[None] for s in med), th)[0])


def compute_kinematics(
    clip: Mapping[str, np.ndarray],
    geoms: Sequence[CollisionGeom],
    th: Thresholds | None = None,
) -> ClipKinematics:
    """Kinematic summary of one clip given its npz arrays (mjlab layout, 50 Hz)."""
    th = th or Thresholds()
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0]) if "fps" in clip else FPS
    names = [str(n) for n in clip["body_names"]]
    jnames = [str(n) for n in clip["joint_names"]]
    pos = np.asarray(clip["body_pos_w"], dtype=np.float64)
    quat = np.asarray(clip["body_quat_w"], dtype=np.float64)
    lin = np.asarray(clip["body_lin_vel_w"], dtype=np.float64)
    ang = np.asarray(clip["body_ang_vel_w"], dtype=np.float64)
    qpos = np.asarray(clip["joint_pos"], dtype=np.float64)
    qvel = np.asarray(clip["joint_vel"], dtype=np.float64)
    T = qpos.shape[0]
    n = max(1, round(th.smooth_s * fps))
    pel = names.index("pelvis")

    v = moving_average(lin[:, pel], n)
    w = moving_average(ang[:, pel], n)
    qd = np.abs(moving_average(qvel, n)) / joint_velocity_limits(jnames)
    jpeak = qd.max(axis=0)
    pelvis_z = moving_average(pos[:, pel, 2], n)
    upright = _rot_z_row(quat[:, pel])[:, 2]  # world-z . pelvis-z

    geoms = [g for g in geoms if g.name not in CONTACT_IGNORED_GEOMS]
    feet = np.stack([lowest_point(pos, quat, names, geoms, (b,)) for b in FOOT_BODIES], 1)
    knees = np.stack([lowest_point(pos, quat, names, geoms, g) for g in KNEE_GROUPS], 1)
    hands = lowest_point(pos, quat, names, geoms, HAND_BODIES)
    feet_contact = feet < th.foot_contact_z
    knee_contact = (knees < th.contact_z).any(axis=1)
    hand_contact = hands < th.contact_z
    special = set(FOOT_BODIES + KNEE_BODIES + HAND_BODIES)
    other_bodies = sorted({g.body for g in geoms} - special)
    body_s, body_name = 0.0, ""
    for b in other_bodies:
        run = longest_run(lowest_point(pos, quat, names, geoms, (b,)) < th.contact_z) / fps
        if run > body_s:
            body_s, body_name = run, b
    supported = feet_contact.any(axis=1) | knee_contact | hand_contact

    knee = qpos[:, [jnames.index(j) for j in KNEE_JOINTS]]
    sig = (pelvis_z, upright, knee, feet_contact, knees)
    per_frame = _postures(*sig, th)
    m = max(1, min(T, round(th.posture_window_s * fps)))

    return ClipKinematics(
        frames=T,
        duration_s=(T - 1) / fps,
        pelvis_speed_xy=float(np.linalg.norm(v[:, :2], axis=1).max()),
        pelvis_speed_z=float(np.abs(v[:, 2]).max()),
        pelvis_ang_speed=float(np.linalg.norm(w, axis=1).max()),
        joint_speed_ratio=float(jpeak.max()),
        joint_speed_joint=jnames[int(jpeak.argmax())],
        pelvis_z_min=float(pelvis_z.min()),
        airborne_s=longest_run(~supported) / fps,
        knee_contact_s=float(knee_contact.sum()) / fps,
        hand_contact_run_s=longest_run(hand_contact) / fps,
        hand_contact_s=float(hand_contact.sum()) / fps,
        body_contact_s=body_s,
        body_contact_body=body_name,
        squat_s=longest_run(per_frame == "squat") / fps,
        start_posture=_window_posture([s[:m] for s in sig], th),
        end_posture=_window_posture([s[-m:] for s in sig], th),
    )


# --------------------------------------------------------------------------- metadata

# Behaviour categories kept by default: what SONIC's planner can command on the gantry.
KEEP_CATEGORIES = ("idle", "walk", "turn", "jog", "squat", "kneel", "crouch", "gesture")
ALL_CATEGORIES = KEEP_CATEGORIES + ("manipulation", "dance", "sport", "stunt", "unusual", "other")


def _words(text: str) -> tuple[str, ...]:
    """Comma-separated word list -> tuple (keeps the keyword tables readable)."""
    return tuple(w.strip() for w in text.split(",") if w.strip())


# Exclusion keywords, matched as whole words (``_`` and spaces delimit) against the clip name,
# the movement-type / body-position labels and all natural-language descriptions.
EXCLUDE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "acrobatics": _words(
        "flip, flips, flipping, somersault, cartwheel, cartwheels, handstand, handstanding,"
        "handspring, standing on hands, roll, rolls, rolling, acrobatic, acrobatics, breakdance,"
        "breakdancing, break dance, headspin, kip, vault, vaulting, parkour, backflip, frontflip"
    ),
    "fall": _words(
        "fall, falls, falling, fallen, faint, faints, fainting, trip, trips, tripping, stumble,"
        "stumbles, stumbling, slip, slips, slipping, collapse, collapses, collapsing, knocked,"
        "postmortem, convulsions, dies, dying"
    ),
    "floor": _words(
        "lying, lie, lies, lay, lays, laying, crawl, crawls, crawling, all fours, hands and knees,"
        "plank, planking, push up, pushup, push ups, burpee, prone, supine, sit on, sits on,"
        "sitting on, sit down, sits down, sitting down, seated"
    ),
    "jump": _words("jump, jumps, jumping, hop, hops, hopping, leap, leaps, leaping"),
    "fight": _words(
        "kick, kicks, kicking, punch, punches, punching, fight, fighting, combat, martial, karate,"
        "kungfu, boxing, sword, knife, stab, stabbing, attack, attacks, uppercut, jab, dodge,"
        "dodges, dodging"
    ),
    "support": _words(
        "crutch, crutches, leash, wall, leaning on, lean on, leans on, ladder, rope, box, stairs,"
        "stair, chair, bench, table"
    ),
}

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kneel", ("kneel", "kneeling", "kneels")),
    ("squat", ("squat", "squats", "squatting", "squating")),
    ("crouch", ("crouch", "crouching", "crouches", "croaching")),
    ("jog", ("jog", "jogging", "run", "running")),
    ("walk", ("walk", "walking", "ff", "sideway", "sideways", "strafe", "backward", "step")),
    ("turn", ("turn", "turning", "rotate", "pivot")),
)
_METADATA_CATEGORY = {
    "Dancing": "dance",
    "Sports": "sport",
    "Stunts": "stunt",
    "Magic": "stunt",
    "Martial Arts": "stunt",
    "Unusual Locomotion": "unusual",
    "Other": "other",
    "Object Manipulation": "manipulation",
    "Object Interaction": "manipulation",
    "Household": "manipulation",
    "Consuming": "manipulation",
}
_GESTURE_CATEGORIES = ("Gestures", "Communication", "Looking and Pointing")
_GESTURE_WORDS = ("gesture", "pointing", "point", "waving", "wave", "looking", "talking", "clap")
_IDLE_WORDS = ("idle", "standing", "stand", "transition", "body check")

DESCRIPTION_COLUMNS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
)


def _normalise(text: str) -> str:
    """Lower-case, ``_``/punctuation/digits -> single spaces, padded so whole-word search works."""
    return " " + " ".join(re.sub(r"[^a-z]+", " ", text.lower()).split()) + " "


def _has_word(text: str, words: Sequence[str]) -> str | None:
    for w in words:
        if f" {w} " in text:
            return w
    return None


@dataclass(frozen=True)
class ClipMeta:
    """The metadata fields the classifier reads (BONES-SEED parquet columns)."""

    name: str
    category: str = ""  # BONES ``category``
    movement: str = ""  # ``content_type_of_movement``
    body_position: str = ""  # ``content_body_position``
    descriptions: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> ClipMeta:
        def s(key: str) -> str:
            v = row.get(key)
            return v if isinstance(v, str) else ""

        return cls(
            name=s("filename"),
            category=s("category"),
            movement=s("content_type_of_movement"),
            body_position=s("content_body_position"),
            descriptions=tuple(s(c) for c in DESCRIPTION_COLUMNS),
        )


def classify_metadata(meta: ClipMeta) -> tuple[str, str | None]:
    """Behaviour category and the metadata exclusion reason (``None`` if none fires)."""
    stem = meta.name.split("__")[0]  # drop the actor / mirror suffix
    name = _normalise(stem)
    labels = _normalise(f"{meta.movement} {meta.body_position}")
    text = _normalise(" ".join((stem, meta.movement, meta.body_position) + meta.descriptions))

    category = _METADATA_CATEGORY.get(meta.category)
    if category is None and _has_word(text, ("dance", "dancing", "dancer")):
        category = "dance"
    if category is None:
        for cat, words in _CATEGORY_RULES:
            if _has_word(name, words) or _has_word(labels, words):
                category = cat
                break
    if category is None:
        if meta.category in _GESTURE_CATEGORIES or _has_word(labels, _GESTURE_WORDS):
            category = "gesture"
        elif _has_word(name, _IDLE_WORDS) or _has_word(labels, _IDLE_WORDS):
            category = "idle"
        else:
            category = "other"

    for reason, words in EXCLUDE_KEYWORDS.items():
        hit = _has_word(text, words)
        if hit:
            return category, f"meta:{reason}:{hit.replace(' ', '_')}"
    return category, None


# --------------------------------------------------------------------------- decision

SQUAT_SOURCE_CATEGORIES = ("idle", "gesture", "manipulation", "other")


def refine_category(category: str, kin: ClipKinematics | None, th: Thresholds | None = None) -> str:
    """Promote clips that hold a squat (``squat_s >= squat_hold_s``) to the squat category.

    BONES has almost no clips *named* squat; its squats are in pick-up / put-down and similar
    clips. Exclusion keywords and every kinematic check still apply to them.
    """
    th = th or Thresholds()
    if kin is not None and category in SQUAT_SOURCE_CATEGORIES and kin.squat_s >= th.squat_hold_s:
        return "squat"
    return category


def kinematic_reasons(
    kin: ClipKinematics, category: str, th: Thresholds | None = None
) -> list[str]:
    """Names of the kinematic checks a clip fails (empty list = passes)."""
    th = th or Thresholds()
    out = []
    if kin.pelvis_speed_xy > th.max_pelvis_speed_xy:
        out.append("kin:pelvis_speed_xy")
    if kin.pelvis_speed_z > th.max_pelvis_speed_z:
        out.append("kin:pelvis_speed_z")
    if kin.pelvis_ang_speed > th.max_pelvis_ang_speed:
        out.append("kin:pelvis_ang_speed")
    if kin.joint_speed_ratio > th.joint_speed_fraction:
        out.append("kin:joint_speed")
    if kin.pelvis_z_min < th.min_pelvis_height:
        out.append("kin:pelvis_low")
    if kin.airborne_s > th.max_airborne_s:
        out.append("kin:airborne")
    if kin.knee_contact_s > 0 and category != "kneel":
        out.append("kin:knee_contact")
    if (
        kin.hand_contact_run_s > th.max_hand_contact_run_s
        or kin.hand_contact_s > th.max_hand_contact_s
    ):
        out.append("kin:hand_contact")
    if kin.body_contact_s > th.max_body_contact_s:
        out.append("kin:body_contact")
    return out


def decide(
    category: str,
    meta_reason: str | None,
    kin: ClipKinematics | None,
    th: Thresholds | None = None,
    keep_categories: Sequence[str] = KEEP_CATEGORIES,
) -> list[str]:
    """All exclusion reasons of a clip, metadata first; empty list = hardware-safe."""
    th = th or Thresholds()
    reasons = []
    if category not in keep_categories:
        reasons.append(f"category:{category}")
    if meta_reason:
        reasons.append(meta_reason)
    if kin is None:
        reasons.append("kin:missing")
    else:
        reasons += kinematic_reasons(kin, category, th)
    return reasons
