"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from realmimic.robots import encoder_bias_half_ranges, get_robot_spec
from realmimic.robots.g1 import joint_gains
from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.rewards import EffortModel, TrackingRewardCfg, joint_weights
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.goal_spec import GOAL_FEAT_DIMS, anchor_mode, goal_dim
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import (
    allreduce_buffers,
    allreduce_mean,
    broadcast_params,
    finalize,
    init_distributed,
)
from supermimic.training.schedule import scheduled_lr
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        cfg.env.anchor_obs = anchor_mode(cfg.env.anchor_obs)
        cfg.policy.future_frames = cfg.env.future_frames
        cfg.policy.goal_frame_dim = GOAL_FEAT_DIMS[cfg.env.goal_features]
        cfg.policy.goal_features = cfg.env.goal_features
        cfg.policy.goal_dim = goal_dim(cfg.env.future_frames, cfg.env.goal_features, cfg.env.anchor_obs)
        cfg.policy_trainer.future_frames = cfg.env.future_frames
        cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
        cfg.policy_trainer.goal_features = cfg.env.goal_features
        from supermimic.world_model import dynamics

        dynamics.AMP_DTYPE = {"none": None, "bf16": torch.bfloat16}[cfg.amp]
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                bin_estimator=cfg.env.bin_estimator,
                freeze_clip_end=cfg.env.freeze_clip_end,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                anchor_obs=cfg.env.anchor_obs,
                goal_features=cfg.env.goal_features,
                self_collision=cfg.env.self_collision,
                robot=cfg.env.robot,
                nconmax=cfg.env.nconmax,
                njmax=cfg.env.njmax,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(load_g1_model(), device=self.device)
        robot_spec = get_robot_spec(cfg.env.robot)
        effort = None
        e = cfg.env
        if (e.effort_penalty, e.energy_penalty, e.ankle_acc_penalty, e.sustained_load_penalty) != (0.0, 0.0, 0.0, 0.0):
            names = list(robot.joint_names)
            kp, kd, scale = joint_gains(robot_spec, names)
            as_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
            effort = EffortModel(
                robot.data.default_joint_pos[0].clone(), as_t(scale), as_t(kp), as_t(kd), as_t(4.0 * scale * kp),
                joint_weight=as_t(joint_weights(names, e.effort_joint_weights)),
                ankle_mask=torch.tensor(["ankle" in n for n in names], device=self.device),
                effort_threshold=e.effort_threshold,
                load_threshold=as_t(joint_weights(names, e.sustained_load_thresholds, e.sustained_load_threshold)) if e.sustained_load_thresholds else None,
                motor_pairs=torch.tensor(
                    [[names.index(f"{p}_pitch_joint"), names.index(f"{p}_roll_joint")] for p in ("left_ankle", "right_ankle", "waist")],
                    device=self.device,
                ),
            )
        reward_cfg = TrackingRewardCfg(
            w_joint_torque=e.effort_penalty, w_energy=e.energy_penalty, w_ankle_acc=e.ankle_acc_penalty, w_anti_shake=e.anti_shake_penalty,
            kernel_tail=e.kernel_tail, w_sustained_load=e.sustained_load_penalty,
            sustained_load_threshold=e.sustained_load_threshold, sustained_load_tau=e.sustained_load_tau,
            w_pelvis_tilt=e.pelvis_tilt_reward, std_pelvis_tilt=e.pelvis_tilt_std, pelvis_index=TRACKED_BODIES.index(ROOT_BODY),
            body_pos_weights=tuple(joint_weights(list(TRACKED_BODIES), e.body_pos_weights)) if e.body_pos_weights else None,
            body_ori_weights=tuple(joint_weights(list(TRACKED_BODIES), e.body_ori_weights)) if e.body_ori_weights else None,
            w_root_ori=e.anchor_ori_weight,
            w_waist_lean=e.waist_lean_penalty, waist_lean_threshold=e.waist_lean_threshold, waist_lean_tau=e.waist_lean_tau,
            waist_lean_index=list(robot.joint_names).index("waist_pitch_joint"),
        )
        heads = AnalyticHeads(
            fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(),
            reward_cfg=reward_cfg, effort_model=effort, state_frame=cfg.world_model.state_spec.frame,
        )
        # Normalise the encoder-bias part of xi by the robot's actual randomization ranges.
        bias_scale = encoder_bias_half_ranges(robot_spec, list(robot.joint_names))
        xs = cfg.world_model.state_spec.xi_scale
        cfg.world_model.state_spec.xi_scale = (xs[0], *bias_scale, *xs[1 + len(bias_scale) :])
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, feat_dim=self.wm.spec.feat_dim, pcfg=cfg.policy, state_frame=self.wm.spec.frame).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        cfg.policy_trainer.value_cap = heads.reward_cfg.max_step_reward / (1.0 - cfg.policy_trainer.gamma)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())
        if self.is_main:
            # Written once every derived setting (compact library, value cap, ...) is resolved.
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0  # = world-model updates (Algorithm A.1 iterations)
        self.num_updates = 0  # policy updates
        self.env_steps = 0  # vectorised env steps
        self._step_acc = 0.0
        self._time_offset = 0.0  # training seconds before a resume
        self._t_start: float | None = None
        if cfg.resume is not None and cfg.init_from is not None:
            raise ValueError("use either resume (continue a run) or init_from (warm start), not both")
        if cfg.resume is not None:
            self.load(cfg.resume)
        elif cfg.init_from is not None:
            self.init_weights(cfg.init_from)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "env_steps": self.env_steps,
                "step_acc": self._step_acc,
                "iteration_unit": "update",
                "train_seconds": (time.time() - self._t_start) if self._t_start is not None else self._time_offset,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                **({"policy_ema": self.policy_trainer.policy_ema} if self.policy_trainer.policy_ema is not None else {}),
                "bin_return": self.command.bin_return,
                "bin_count": self.command.bin_count,
                "bin_memory": self.command.bin_memory,
                "config": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def init_weights(self, path: Path) -> None:
        """Warm start from ``path``: model weights only; the observation/goal layout must match."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        src = ck["config"]["env"]
        for key in ("goal_features", "anchor_obs", "future_frames"):
            if src.get(key) != getattr(self.cfg.env, key):
                raise ValueError(f"init_from {path}: env.{key}={src.get(key)!r} != {getattr(self.cfg.env, key)!r}")
        self._check_state_frame(ck, path)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        print(f"[INFO] initialised weights from {path} (iteration {ck.get('iteration')}, robot {src.get('robot', 'g1')})")

    def _check_state_frame(self, ck: dict, path: Path) -> None:
        # Checkpoints from before StateSpecCfg.frame existed use the heading frame.
        frame = ck["config"]["world_model"]["state_spec"].get("frame", "heading")
        if frame != self.wm.spec.frame:
            raise ValueError(f"{path}: world model in the {frame!r} state frame, this run uses {self.wm.spec.frame!r} (world_model.state_spec.frame)")

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self._check_state_frame(ck, path)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        if "policy_ema" in ck:
            self.policy_trainer.policy_ema = ck["policy_ema"]
        self.command.bin_return.copy_(ck["bin_return"])
        self.command.bin_memory.copy_(ck["bin_memory"])
        if "bin_count" in ck:
            self.command.bin_count.copy_(ck["bin_count"])
            self.command._bin_count_synced.copy_(ck["bin_count"])
        elif self.cfg.env.bin_estimator == "count":
            raise ValueError(f"{path} has no bin visit counts (fork sampler); resume it with env.bin_estimator='ema'")
        if ck.get("iteration_unit") != "update":
            raise ValueError(f"{path} counts iterations in env steps (v1); resuming it would misalign the schedule")
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        self.env_steps = ck["env_steps"]
        self._step_acc = ck["step_acc"]
        self._time_offset = ck.get("train_seconds", 0.0)
        print(f"[INFO] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        self.policy.update_input_norms(obs["policy"], obs["goal"])  # no-op unless policy.input_norm == "running"
        return self.policy(obs["policy"], obs["goal"])

    # MuJoCo Warp OverflowType bits (mujoco_warp/_src/types.py): constraint / contact capacity and
    # solver iteration limits. The printf warnings are disabled (QuietSimulationCfg), so count them here.
    _OVERFLOW_BITS = {"nefc": 1 << 0, "broadphase": 1 << 2, "narrowphase": 1 << 3, "ccd": 1 << 4, "iterations": 1 << 9, "ls_iterations": 1 << 10}

    def _overflow_fractions(self) -> dict[str, float]:
        """Fraction of worlds that hit each overflow since the last call (the bitmask is sticky; reset it)."""
        ov = getattr(self.env.sim.data, "overflow", None)
        if ov is None:
            return {}
        try:
            t = ov[:].clone()
            ov[:] = 0
        except Exception:  # noqa: BLE001 - diagnostics only
            return {}
        return {k: float(((t & b) != 0).float().mean()) for k, b in self._OVERFLOW_BITS.items()}

    def _env_step(self) -> None:
        """One vectorised env step with the current policy; tracks training-episode returns."""
        action = self.act(self._obs)
        self._obs, rew, term, tout, _ = self.env.step(action)
        self.env_steps += 1
        self.ep_reward_sum += rew
        self.ep_len += 1
        done = term | tout
        if done.any():
            ids = done.nonzero().squeeze(-1)
            lens = self.ep_len[ids]
            norm_ret = self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))
            self._finished_returns.append(norm_ret)
            self._finished_lens.append(lens)
            self.ep_reward_sum[ids] = 0.0
            self.ep_len[ids] = 0.0

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        self._obs, _ = self.env.reset()
        self._finished_returns: list[torch.Tensor] = []
        self._finished_lens: list[torch.Tensor] = []
        t_start = self._t_start = time.time() - self._time_offset
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        last_log = time.time()

        # Fill the replay buffer far enough for k-step windows (also after a resume: the buffer is not checkpointed).
        for _ in range(sch.warmup_env_steps):
            self._env_step()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect 1/UTD vectorised env steps (fractional accumulator)
            t0 = time.perf_counter()
            self._step_acc += 1.0 / sch.updates_per_env_step
            n_steps = int(self._step_acc)
            self._step_acc -= n_steps
            for _ in range(n_steps):
                self._env_step()
            if self.world > 1 and cfg.policy.input_norm == "running":
                allreduce_buffers(self.policy)  # each rank updates the input normalisers from its own envs
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0

            # ---------------- (2) fit the world model, (3) improve policy + critic
            lr = scheduled_lr(self.iteration, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_iterations, sch.lr_restart_iteration, sch.lr_restart_ratio, sch.lr_restart_decay_iterations)
            self.wm_trainer.set_lr(lr)
            self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
            t0 = time.perf_counter()
            m_wm = self.wm_trainer.update(self.buffer)
            torch.cuda.synchronize()
            timers["wm"] += time.perf_counter() - t0
            for k, v in m_wm.items():
                agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
            agg["lr"] = lr
            agg_n += 1
            if self.iteration >= sch.wm_warmup_iterations:
                t0 = time.perf_counter()
                for _ in range(sch.policy_updates_per_wm_update):
                    m_pi = self.policy_trainer.update(self.buffer)
                    self.num_updates += 1
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                agg["_pi_n"] = agg.get("_pi_n", 0.0) + sch.policy_updates_per_wm_update
                torch.cuda.synchronize()
                timers["policy"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                if self._finished_returns:
                    rets = torch.cat(self._finished_returns)
                    lens = torch.cat(self._finished_lens)
                    ret_sum, ret_cnt, len_sum = float(rets.sum()), float(rets.numel()), float(lens.sum())
                else:
                    ret_sum = ret_cnt = len_sum = 0.0
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    ret_sum = allreduce_mean(ret_sum) * self.world
                    len_sum = allreduce_mean(len_sum) * self.world
                    ret_cnt = allreduce_mean(ret_cnt) * self.world
                tr = ret_sum / ret_cnt if ret_cnt > 0 else float("nan")
                pi_n = max(agg.pop("_pi_n", 0.0), 1.0)
                if self.world > 1:
                    self.command.sync_bins()
                samp = self.command.sampler_stats()
                with torch.no_grad():
                    err = self.wm_trainer.online_error(self.buffer, n=1024, k=8) if self.is_main else None
                if self.is_main:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v if k == "lr" else v / (pi_n if k.startswith("policy/") else agg_n), it)
                    if ret_cnt > 0:
                        self.writer.add_scalar("train/episode_norm_return", tr, it)
                        self.writer.add_scalar("train/episode_len", len_sum / ret_cnt, it)
                    for k, v in self._overflow_fractions().items():
                        self.writer.add_scalar(f"sim/overflow_{k}", v, it)
                        if v > 0 and k in ("nefc", "broadphase", "narrowphase", "ccd"):
                            print(f"[WARN] MuJoCo Warp {k} overflow in {v:.2%} of worlds since the last log", flush=True)
                    self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                    self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/env_steps", self.env_steps, it)
                    self.writer.add_scalar("time/policy_updates", self.num_updates, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    for k, v in samp.items():
                        self.writer.add_scalar(f"sampler/{k}", v, it)
                    for k, v in zip(("nonfinite", "joint_vel", "root_vel"), self.recorder.unphysical_counts.tolist()):
                        self.writer.add_scalar(f"sampler/unphysical_{k}", v, it)
                    # Resets on the Table A.8 groups relative to a uniform sampler (rank 0's resets).
                    hard = [g for g in self.command.group_names if g != "all_other"]
                    hard_x = sum(samp.get(f"{g}/reset_share", 0.0) for g in hard) / max(sum(samp[f"{g}/uniform_share"] for g in hard), 1e-12)
                    self.writer.add_scalar("sampler/hard_groups_reset_x_uniform", hard_x, it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.2f} it/s | env steps {self.env_steps} | "
                        f"train_ret {tr:.3f} (n={int(ret_cnt)}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/agg_n:.4f} nse8 {float(err[-1]):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/pi_n:.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/pi_n:.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s | "
                        f"untouched {samp['untouched_frac']:.3f} hard x{hard_x:.1f} unphys {int(samp['unphysical_episodes'])} "
                        f"(nonfinite/qd/v {'/'.join(str(c) for c in self.recorder.unphysical_counts.tolist())})"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                self._finished_returns, self._finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
