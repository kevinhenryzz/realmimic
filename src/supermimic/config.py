"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    # Reset sampler bin estimator: "count" (visit-count aware; needed once the library has ~1M bins) or "ema" (the
    # optimistic moving average of the reimplementation this code started from).
    bin_estimator: str = "count"
    # Reference past a clip's end = its last frame held still, in goals, imagined rewards and the critic, so
    # the objective covers the clip ends (with policy_trainer.start_margin 0). False before this option (and in
    # older checkpoints): the last frame is repeated with its velocities and imagination stops 32 frames short.
    freeze_clip_end: bool = True
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    # Reference-relative anchor pose in the goal: "none" (pure reference goal),
    # "ori" (reference anchor orientation in the robot anchor frame, 6-d, IMU-observable; default),
    # "pos_ori" (+ anchor position, 9-d, needs state estimation).
    anchor_obs: str = "ori"
    # Per-frame goal features: "v1" (68-d local reference quantities) or "traj" (77-d: + root
    # displacement and rotation relative to the current target frame's heading frame).
    goal_features: str = "traj"
    # supermimic_vibed option: False drops robot self-collisions (feet / body vs ground only).
    self_collision: bool = False
    # realmimic RobotSpec name (see realmimic.robots.ROBOTS); "g1" is mjlab's stock G1.
    robot: str = "g1"
    # Weight of the normalised joint-torque penalty in the policy objective (0 = paper reward).
    effort_penalty: float = 0.0
    # Per-joint weights of that penalty, "regex=w,regex=w" over joint names (others 1), e.g.
    # "waist_(roll|pitch)_joint=5".
    effort_joint_weights: str = ""
    # Fraction of each joint's effort limit below which that penalty is zero (relu(|tau|/tau_max - t)^2).
    effort_threshold: float = 0.0
    # SONIC's shaping terms in the policy objective (0 = off; SONIC: energy -1e-4 (v1.1 only),
    # ankle acceleration -2.5e-6, anti-shake -5e-3).
    energy_penalty: float = 0.0
    ankle_acc_penalty: float = 0.0
    anti_shake_penalty: float = 0.0
    # Sustained per-motor load penalty (TrackingRewardCfg.w_sustained_load; 0 = off), its threshold as a fraction
    # of each motor's limit, and the averaging time constant [s]. Weights per joint: effort_joint_weights.
    sustained_load_penalty: float = 0.0
    sustained_load_threshold: float = 0.1
    sustained_load_tau: float = 0.5
    # Per-joint thresholds overriding sustained_load_threshold, "regex=t,regex=t" (e.g. a reference policy's mean loads).
    sustained_load_thresholds: str = ""
    # Sustained torso-lean penalty (TrackingRewardCfg.w_waist_lean; 0 = off) and its threshold on the averaged
    # |waist pitch| [rad]; averaging time constant sustained_load_tau.
    waist_lean_penalty: float = 0.0
    waist_lean_threshold: float = 0.035
    # Its averaging time constant [s]; None = sustained_load_tau; 0 = penalise the instantaneous lean.
    waist_lean_tau: float | None = None
    # Pelvis-tilt reward (TrackingRewardCfg.w_pelvis_tilt; 0 = off) and its kernel width [rad].
    pelvis_tilt_reward: float = 0.0
    pelvis_tilt_std: float = 0.05
    # Per-body weights of the body position / orientation tracking terms, "regex=w,..." over the tracked body names
    # (unmatched bodies 1; empty = the paper's uniform mean), and the anchor (torso) orientation term's weight.
    body_pos_weights: str = ""
    body_ori_weights: str = ""
    anchor_ori_weight: float = 0.5
    # Imagined-reward-only linear tail on the tracking kernels (0 = paper kernel; TrackingRewardCfg).
    kernel_tail: float = 0.0
    # MuJoCo Warp per-world contact / constraint capacity (contact-rich clips such as rolls need more).
    nconmax: int = 35
    njmax: int = 250


@dataclass
class ScheduleCfg:
    # One iteration = one world-model update + one policy/critic update (Algorithm A.1: "Both omega and
    # theta update every iteration"), preceded by 1/UTD vectorised env steps (fractional accumulator).
    total_iterations: int = 320_000
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_env_steps: int = 40  # env steps collected before the first iteration (fills k-step windows)
    wm_warmup_iterations: int = 500  # world-model-only iterations before the first policy update
    lr_final_ratio: float = 0.03
    lr_decay_iterations: int = 250_000
    # Warm restart (resumed runs): from this iteration a second cosine from lr_restart_ratio x lr down to
    # lr_final_ratio x lr over lr_restart_decay_iterations. None = off.
    lr_restart_iteration: int | None = None
    lr_restart_ratio: float = 0.3
    lr_restart_decay_iterations: int = 60_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
    # Warm start: load model weights (world model, policy, critic, input normalisers) from a
    # checkpoint; optimisers, schedule and reset-sampling state start fresh (e.g. new dataset or robot).
    init_from: Path | None = None
    amp: str = "bf16"  # "bf16" (autocast for MLP matmuls; ~31% faster on RTX 5090) | "none" (TF32 matmuls, as in the paper)
