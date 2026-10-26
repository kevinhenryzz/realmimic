"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"
    anchor_obs: str = "ori"  # set from EnvCfg by the trainer
    goal_features: str = "v1"  # set from EnvCfg by the trainer
    # "sum": loss = -J (Eq. 2, batch mean) so Table A.5's clipping acts on the paper's objective;
    # "per_step": loss = -J / H (the v1 behaviour).
    objective_scale: str = "sum"
    # Upper bound on critic values used as bootstrap and GAE targets: r_max / (1 - gamma) (set by the
    # trainer from the reward config; None = unbounded). Guards against critic blow-ups feeding back
    # through the targets, reported in other reproductions.
    value_cap: float | None = None
    # supermimic_vibed option: only start imagined rollouts at states whose target frame is at least
    # ``start_margin`` frames before the clip end (0 = any state; past the end the last frame is held).
    start_margin: int = 0
    # Policy gradient accumulated over this many independent batches of batch_size imagined rollouts before one
    # optimiser step: an effective batch of grad_accumulation x batch_size within one GPU's memory (the paper's
    # 1-GPU run uses 4096, its 2-GPU run 8192). The critic is fitted once on all of them.
    grad_accumulation: int = 1
    # Exponential moving average of the policy weights (decay per policy update; 0 = off), saved as policy_ema.
    policy_ema: float = 0.0


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        fused = torch.cuda.is_available()
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas, fused=fused)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, fused=fused)
        from supermimic.imagination.compiled import ImagHeads, ImagStep

        self.step_fn = ImagStep(wm, policy, lib.default_joint_pos, anchor_obs=cfg.anchor_obs, obs_noise=cfg.obs_noise, compile_mode=cfg.compile)
        self.heads_fn = ImagHeads(wm, critic_features=critic.error_features, compile_mode=cfg.compile)
        self.policy_ema: dict[str, torch.Tensor] | None = None  # cfg.policy_ema > 0: averaged policy weights

    @torch.no_grad()
    def _critic_data(self, ro: ImagRollout, feat, goals, err, xi_norm):
        """Flattened critic inputs and GAE targets of one rollout, plus its values (for the metrics)."""
        cfg = self.cfg
        feat_d, goals_d, err_d = feat.detach(), goals.detach(), err.detach()
        H1, n, _ = feat_d.shape
        xi_rep = xi_norm[None].expand(H1, n, -1)
        values = self.critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1), err_d.reshape(H1 * n, -1)).reshape(H1, n)
        if cfg.value_cap is not None:
            values = values.clamp(max=cfg.value_cap)
        targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)  # (H, n)
        return (
            feat_d[:-1].reshape(-1, feat_d.shape[-1]),
            goals_d[:-1].reshape(-1, goals_d.shape[-1]),
            xi_rep[:-1].reshape(-1, xi_rep.shape[-1]),
            err_d[:-1].reshape(-1, err_d.shape[-1]),
            targets.reshape(-1),
            values.reshape(-1),
        )

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.cfg.policy_ema <= 0.0:
            return
        if self.policy_ema is None:
            self.policy_ema = {k: v.detach().clone() for k, v in self.policy.state_dict().items()}
            return
        d = self.cfg.policy_ema
        for k, v in self.policy.state_dict().items():
            if torch.is_floating_point(v):
                self.policy_ema[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.policy_ema[k].copy_(v)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.crit_feat

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        extra = None
        if cfg.start_margin > 0:
            extra = buffer.frame + cfg.start_margin <= self.lib.last_frame_of(buffer.frame)

        # ---- actor: BPTT through the frozen world model, gradient summed over grad_accumulation batches ----
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        k = max(1, cfg.grad_accumulation)
        metrics: dict[str, torch.Tensor] = {}
        fits = []  # per batch: detached critic inputs and GAE targets
        for _ in range(k):
            start = buffer.sample_starts(cfg.batch_size, cfg.history, extra_mask=extra)
            xi_norm = self.wm.spec.normalize_xi(start.xi)
            ro = imagine(self.lib, start, cfg.horizon, self.step_fn, self.heads_fn, cfg.future_frames, cfg.goal_features)
            feat, goals, err = self._critic_inputs(ro)
            if cfg.use_critic:
                v_H = self.critic(feat[-1], goals[-1], xi_norm, err[-1])
                if cfg.value_cap is not None:
                    v_H = v_H.clamp(max=cfg.value_cap)
            else:
                v_H = None
            J = policy_objective(ro, v_H, cfg.gamma)
            loss = -J if cfg.objective_scale == "sum" else -J / cfg.horizon
            (loss / k).backward()
            # Metrics stay on-device until one host sync at the end of the update (means over the batches).
            batch_metrics = {
                "imag_objective": J.detach(),
                "imag_return_undisc": (ro.survival[:-1] * ro.rewards).sum(0).mean().detach(),
                "imag_reward_mean": ro.rewards.mean().detach(),
                "imag_survival_H": ro.survival[-1].mean().detach(),
                "imag_dhat_mean": ro.dhat[1:].mean().detach(),
                "action_abs_mean": ro.actions.abs().mean().detach(),
                **{f"imag_r/{name}": v for name, v in ro.reward_terms.items()},
            }
            for name, v in batch_metrics.items():
                metrics[name] = metrics.get(name, 0.0) + v / k
            if cfg.use_critic:
                fits.append(self._critic_data(ro, feat, goals, err, xi_norm))
        allreduce_grads(self.policy)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self._update_ema()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        metrics["policy_grad_norm"] = gn.detach()

        # ---- critic: fit to the GAE targets of these rollouts (targets fixed) ------------------
        if cfg.use_critic:
            X_feat, X_goal, X_xi, X_err, Y, values = (torch.cat(t) for t in zip(*fits))
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_err[idx])
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = last
            metrics["critic_value_mean"] = values.mean()
            metrics["critic_target_mean"] = Y.mean()
        keys = list(metrics)
        vals = torch.stack([metrics[k].float().reshape(()) for k in keys]).tolist()
        return dict(zip(keys, vals))
