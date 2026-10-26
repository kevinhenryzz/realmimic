"""Compiled pieces of an imagined rollout (launch-overhead control).

The rollout is split the way supermimic_vibed does it, which minimises kernel launches:

* ``ImagStep`` — the only per-step work: goal anchor term (FK of the pelvis -> torso chain only),
  policy forward, world-model step, next noise-free observation + sensor noise, history shift.
  Policy, world model and observation are fused into one compiled graph per step.
* ``ImagHeads`` — everything that does not feed back into the rollout, evaluated once on all
  (H+1)·n imagined states: full-body FK, reward, termination logits and the critic's privileged
  tracking errors. Gradients reach every state through the stacked tensors exactly as if they
  were computed step by step (the policy's actions never depend on rewards).
"""

from __future__ import annotations

import math

import torch

from supermimic.envs.mdp.rewards import motor_load, sustained_excess_penalty, sustained_load_penalty
from supermimic.goal_spec import ANCHOR_WINDOW_MODES, anchor_mode
from supermimic.kinematics.rotations import anchor_window_obs
from supermimic.world_model.heads import (
    ReferenceFrame,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)
from supermimic.world_model.world_model import WorldModel

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
    "joint_torque",
    "energy",
    "ankle_acc",
    "anti_shake",
    "sustained_load",
    "pelvis_tilt",
    "waist_lean",
)

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
# Goal-side anchor terms (mjlab tracking defaults): position 0.25 m, orientation 0.05.
ANCHOR_NOISE_HALF = {
    "pos_ori": torch.tensor([0.25] * 3 + [0.05] * 6),
    "ori": torch.tensor([0.05] * 6),
    "pelvis_ori10": torch.tensor([0.05] * 60),
}


def _compile(fn, mode: str | None):
    if not mode:
        return fn
    return torch.compile(fn, mode=None if mode == "default" else mode, dynamic=False)


class ImagStep:
    """(s_t, noisy obs history, reference goal part of g_t, reference anchor) -> (s_{t+1}, a_t, history', g_t)."""

    def __init__(self, wm: WorldModel, policy, default_joint_pos: torch.Tensor, anchor_obs: str | bool = "none", obs_noise: bool = True, compile_mode: str | None = None):
        self.wm, self.policy, self.q0 = wm, policy, default_joint_pos
        self.mode = anchor_mode(anchor_obs)
        self.obs_noise = obs_noise
        dev = default_joint_pos.device
        self.noise_half = OBS_NOISE_HALF.to(dev)
        self.anchor_noise_half = ANCHOR_NOISE_HALF[self.mode].to(dev) if self.mode != "none" else None
        heads = wm.heads
        self.torso = int(heads.tracked_idx[heads.anchor_index])  # FK body index of the anchor (torso_link)
        self._fn = _compile(self._step, compile_mode)
        self._goal_fn = _compile(self.goal, compile_mode)

    def noisy(self, o: torch.Tensor) -> torch.Tensor:
        return o + (torch.rand_like(o) * 2 - 1) * self.noise_half if self.obs_noise else o

    def goal(self, s: torch.Tensor, goal_ref: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
        """Full goal g_t: reference features + (noisy) reference anchor pose in the robot anchor frame of s_t."""
        if self.mode == "none":
            return goal_ref
        if self.mode in ANCHOR_WINDOW_MODES:  # ref_anchor_quat: (N, k, 4) window of reference pelvis quats
            rel = anchor_window_obs(s[..., 3:7], ref_anchor_quat)
        else:
            pos, quat = self.wm.heads.fk.chain_pose(s[..., :36], self.torso)
            rel = anchor_relative_obs(pos, quat, ref_anchor_pos, ref_anchor_quat, self.mode)
        if self.obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * self.anchor_noise_half
        return torch.cat([goal_ref, rel], dim=-1)

    def _step(self, s, hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi):
        g = self.goal(s, goal_ref, ref_anchor_pos, ref_anchor_quat)
        a = self.policy(flatten_history_term_major(hist), g)
        s_next = self.wm.step(s, a, xi)
        obs = self.noisy(analytic_observation(s_next, a, xi, self.q0))
        return s_next, a, torch.cat([hist[1:], obs[None]], dim=0), g

    def __call__(self, s, hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi):
        return self._fn(s, hist, goal_ref, ref_anchor_pos, ref_anchor_quat, xi)


class ImagHeads:
    """Batched reward / termination / critic features over a whole rollout."""

    def __init__(self, wm: WorldModel, critic_features: bool = True, compile_mode: str | None = None):
        self.wm = wm
        self.critic_features = critic_features
        self._fn = _compile(self._heads, compile_mode)

    def _sustained_load(self, states, actions, prev_actions):
        """Sustained per-motor load term (H, n) along the rollout (TrackingRewardCfg.w_sustained_load): the
        load of action t at state t+1, as the torque term, averaged from the start state's load under the
        previous action."""
        heads = self.wm.heads
        cfg, model = heads.reward_cfg, heads.effort_model
        H, n = actions.shape[:2]
        if cfg.w_sustained_load == 0.0:
            return torch.zeros(H, n, device=states.device)
        if model is None:
            raise ValueError("w_sustained_load needs an effort model")
        s1 = states[1:]
        loads = motor_load(model, s1[..., 7:36], s1[..., 42:71], actions)
        init = motor_load(model, states[0, :, 7:36], states[0, :, 42:71], prev_actions[0])
        alpha = math.exp(-cfg.dt / cfg.sustained_load_tau)
        threshold = model.load_threshold if model.load_threshold is not None else cfg.sustained_load_threshold
        return cfg.dt * cfg.w_sustained_load * sustained_load_penalty(model, loads, init, threshold, alpha)

    def _waist_lean(self, states, n_steps):
        """Sustained torso-lean term (H, n) along the rollout (TrackingRewardCfg.w_waist_lean): the held waist pitch
        of states 1..H, averaged from the start state's."""
        cfg = self.wm.heads.reward_cfg
        if cfg.w_waist_lean == 0.0:
            return torch.zeros(n_steps, states.shape[1], device=states.device)
        if cfg.waist_lean_index < 0:
            raise ValueError("w_waist_lean needs waist_lean_index")
        q = states[..., 7 + cfg.waist_lean_index].abs()
        tau = cfg.sustained_load_tau if cfg.waist_lean_tau is None else cfg.waist_lean_tau
        alpha = math.exp(-cfg.dt / tau) if tau > 0 else 0.0  # tau 0: the instantaneous lean
        return cfg.dt * cfg.w_waist_lean * sustained_excess_penalty(q[1:], q[0], cfg.waist_lean_threshold, alpha)

    def _heads(self, states, actions, prev_actions, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat, ref_joint_pos):
        """states (H+1, n, 71); actions / prev_actions (H, n, 29); reference tensors for frames 0..H."""
        H1, n = states.shape[:2]
        H = H1 - 1
        heads = self.wm.heads
        flat = states.reshape(H1 * n, -1)
        bodies = heads.robot_bodies(flat)
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat, ref_joint_pos)
        # Reward r_{t+1} of reaching s_{t+1} under target frame t (frames 0..H-1, states 1..H).
        tgt = ReferenceFrame(*(x[: H * n] for x in (ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat, ref_joint_pos)))
        nxt = {k: v[n:] for k, v in bodies.items()}
        terms = heads.reward(flat[n:], actions.reshape(H * n, -1), prev_actions.reshape(H * n, -1), tgt, nxt, s_prev=flat[: H * n])
        terms["sustained_load"] = self._sustained_load(states, actions, prev_actions).reshape(H * n)
        terms["waist_lean"] = self._waist_lean(states, H).reshape(H * n)
        terms["total"] = terms["total"] + terms["sustained_load"] + terms["waist_lean"]
        term_vec = torch.stack([terms[k].reshape(H, n) for k in REWARD_TERM_NAMES], dim=0)  # (terms, H, n)
        logits = self.wm.termination_logits(flat[n:], tgt).reshape(H, n)
        # Critic features of s_t against target frame t (t = 0..H).
        if self.critic_features:
            crit = heads.critic_error_features(flat, ref, bodies)
        else:
            crit = heads.root_error_features(flat, ref)
        return terms["total"].reshape(H, n), term_vec, logits, crit.reshape(H1, n, -1)

    def __call__(self, states, actions, prev_actions, ref: ReferenceFrame):
        return self._fn(states, actions, prev_actions, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w, ref.joint_pos)
