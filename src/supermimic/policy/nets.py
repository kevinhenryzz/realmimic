"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from supermimic.goal_spec import GOAL_FEAT_LAYOUTS
from supermimic.world_model.dynamics import MLP, RunningNorm
from supermimic.world_model.heads import OBS_TERMS, critic_err_dim

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos 29, joint_vel 29, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # future_frames * goal_frame_dim + anchor dims (set by the trainer)
    future_frames: int = 10
    goal_frame_dim: int = 68  # 68 ("v1"), 77 ("traj") or 59 ("sonic")
    goal_features: str = "v1"  # per-frame goal layout (supermimic.goal_spec.GOAL_FEAT_LAYOUTS)
    # Options from supermimic_vibed: "fixed" per-term input scales, or "running" mean/std of the
    # observation history and goal (updated from simulator observations during collection).
    input_norm: str = "running"
    activation: str = "elu"  # "silu" | "elu"


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)
    # Privileged tracking-error features (critic only); False = root errors only (4).
    error_features: bool = True


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    frame_dim = getattr(cfg, "goal_frame_dim", 68)
    if getattr(cfg, "goal_features", "v1") == "sonic":
        per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in GOAL_FEAT_LAYOUTS["sonic"]])
    else:
        sizes = (29, 29, 3, 3, 3, 1)
        per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)] + [torch.ones(frame_dim - 68)])
    k = getattr(cfg, "future_frames", 10)
    extra = goal_dim - k * frame_dim
    if extra < 0:  # configs from before future_frames/goal_frame_dim were stored
        k, extra = divmod(goal_dim, frame_dim)
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


def _running(cfg: PolicyCfg) -> bool:
    mode = getattr(cfg, "input_norm", "fixed")
    if mode not in ("fixed", "running"):
        raise ValueError(f"unknown policy input_norm {mode!r}")
    return mode == "running"


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim, act=getattr(cfg, "activation", "silu"))
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))
        self.norm = RunningNorm(goal_dim) if _running(cfg) else None

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(self.norm(goal) if self.norm is not None else goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int = 930):
        super().__init__()
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, ACT_DIM, act=getattr(cfg, "activation", "silu"), last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))
        self.norm = RunningNorm(obs_dim) if _running(cfg) else None

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, self.norm(obs_hist) if self.norm is not None else obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        return self.decoder(self.encoders[interface](goal), obs_hist)

    @torch.no_grad()
    def update_input_norms(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> None:
        """Feed simulator observations to the running input normalisers (no-op for "fixed")."""
        enc = self.encoders[interface]
        if enc.norm is not None:
            enc.norm.update(goal)
        if self.decoder.norm is not None:
            self.decoder.norm.update(obs_hist)


class Critic(nn.Module):
    """V(world-model state features, goal, xi 33, privileged tracking errors | root error) -> scalar
    (one per interface). ``feat_dim`` and ``state_frame`` follow the world model's StateSpec."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 68, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None, state_frame: str = "body"):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.error_features = cfg.error_features
        self.err_dim = critic_err_dim(state_frame, cfg.error_features)
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + self.err_dim, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, err], dim=-1)).squeeze(-1)
