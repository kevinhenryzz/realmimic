"""Benchmark evaluation (paper Appendix F.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends.

Normalised return (paper §4): the undiscounted reward sum over the episode divided by the
largest value it can take, i.e. max step reward * the number of steps of the *complete* clip,
so a policy that terminates early scores low. ``norm_return_survived`` keeps the v1 definition
(divided by the steps actually taken) for comparison with the old experiment log only.

``policy`` is one of our checkpoints or SONIC's release policy
(``realmimic.eval.sonic_release_policy``); each runs with its own actuator gains and action scale.

``recording`` (optional, meant for a handful of clips) keeps the robot and reference qpos of every
step for videos (``tools/render_rollouts.py``); it does not change the results. ``terminate=False``
(videos only) disables the failure terminations so a clip plays on after the robot loses track.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.goal_spec import anchor_mode
from supermimic.policy.nets import Policy, PolicyCfg

if TYPE_CHECKING:
    from realmimic.eval.sonic_release_policy import SonicReleasePolicy


# A failure this many frames or fewer before the clip's end counts as an end-of-clip failure (reported
# separately; it is still a failure under the protocol).
END_FAILURE_FRAMES = 3


def failed_at_end(frames: int, failed: bool, fail_step: int) -> bool:
    return failed and fail_step >= frames - END_FAILURE_FRAMES


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    norm_return_survived: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    # Env steps in which MuJoCo Warp hit a capacity overflow (contacts / constraints) and the first
    # such step (-1: none); overflowed steps have dropped contacts, i.e. non-physical dynamics.
    overflow_steps: int = 0
    first_overflow_step: int = -1
    # Largest pelvis-height and tilt errors over the clip, as checked by the terminations (thresholds 0.25 m
    # and 0.8): how close a completed clip came to failing. NaN for results saved before these fields existed.
    max_height_error: float = float("nan")
    max_tilt_error: float = float("nan")


@dataclass
class Recording:
    """Filled by ``run_benchmark(recording=...)``: a copy of the scene's MjSpec and, per clip name, the
    robot and reference qpos (model layout, env origin removed) at the start of every env step,
    i.e. ``steps`` samples, sample k against reference frame k. The terminal state is not kept."""

    spec: Any = None  # mujoco.MjSpec; compile it (after any restyling) for a render model
    qpos: dict[str, np.ndarray] = field(default_factory=dict)
    ref_qpos: dict[str, np.ndarray] = field(default_factory=dict)


def env_kwargs(cfg: dict) -> dict:
    """``run_benchmark`` keyword arguments that reproduce a checkpoint's training env."""
    e = cfg["env"]
    return {"future_frames": e["future_frames"], "anchor_obs": anchor_mode(e.get("anchor_obs", False)), "goal_features": e.get("goal_features", "v1"),
            "self_collision": e.get("self_collision", True), "robot": e.get("robot", "g1"), "freeze_clip_end": e.get("freeze_clip_end", False)}  # fmt: skip


def load_policy_from_checkpoint(path: str | Path, device: str, weights: str = "policy") -> tuple[Policy, dict]:
    """``weights``: "policy" (the trained weights) or "policy_ema" (their moving average, if the run kept one)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = dict(ck["config"]["policy"])
    pc.setdefault("input_norm", "fixed")  # configs saved before these options existed (v1 / early v2)
    pc.setdefault("activation", "silu")
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    if weights not in ck:
        raise KeyError(f"{path}: no {weights!r} weights in this checkpoint")
    policy.load_state_dict(ck[weights])
    policy.eval()
    cfg = dict(ck["config"])
    cfg["_meta"] = {k: ck.get(k) for k in ("iteration", "iteration_unit", "env_steps", "num_updates", "train_seconds")}
    return policy, cfg


@torch.no_grad()
def run_benchmark(policy: Policy | SonicReleasePolicy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: str | bool = "none", goal_features: str = "v1", self_collision: bool = True, robot: str = "g1", freeze_clip_end: bool = False, recording: Recording | None = None, terminate: bool = True) -> list[ClipResult]:
    from realmimic.eval.sonic_release_policy import SonicReleasePolicy

    sonic = isinstance(policy, SonicReleasePolicy)
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_mode(anchor_obs),
        goal_features=goal_features,
        self_collision=self_collision,
        freeze_clip_end=freeze_clip_end,
        # terminate=False (videos only): clips play to the end whatever happens; nothing counts as failed.
        **({} if terminate else {"root_height_threshold": float("inf"), "root_tilt_threshold": float("inf")}),
        robot=robot,
        actuator_gains="sonic" if sonic else "spec",
    )
    env = make_env(args, device=device)
    try:
        if sonic:
            actor = policy.actor(env)  # reads the robot state and the reference from env itself
            act_fn = lambda obs: actor()
        else:
            act_fn = lambda obs: policy(obs["policy"], obs["goal"])
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        overflow = getattr(env.sim.data, "overflow", None)  # per-world sticky bitmask
        ov_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        ov_first = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        max_h = torch.zeros(num_envs, device=device)
        max_t = torch.zeros(num_envs, device=device)
        results: dict[int, ClipResult] = {}
        if recording is not None:
            recording.spec = env.scene.spec.copy()
            idx = env.scene["robot"].indexing
            root_q, robot_q = idx.free_joint_q_adr[:3].long(), torch.cat([idx.free_joint_q_adr, idx.joint_q_adr]).long()
            rec_q, rec_ref = [], []
        remaining = n_clips
        while remaining > 0:
            if recording is not None:
                q = env.sim.data.qpos.clone()
                q[:, root_q] -= env.scene.env_origins
                clip = torch.clamp(current, max=n_clips - 1)
                ref = torch.zeros_like(q)
                ref[:, robot_q] = lib.ref_qpos_qvel(lib.clip_start[clip] + torch.minimum(steps, lib.clip_len[clip] - 1))[0].to(q.dtype)
                rec_q.append(q)
                rec_ref.append(ref)
            act = act_fn(obs)
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            max_h = torch.maximum(max_h, cmd.root_height_error)
            max_t = torch.maximum(max_t, cmd.root_tilt_error)
            if overflow is not None:
                hit = overflow[:] != 0
                overflow[:] = 0
                ov_steps += hit.long()
                ov_first = torch.where(hit & (ov_first < 0), steps, ov_first)
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        # A complete clip of L frames takes L env steps here (the reset already
                        # advances the target to frame 1; the clip-end time-out fires one step after
                        # the last frame is reached, which is scored against the last frame again).
                        full = lib.clips[ci].length
                        if not bool(term[e]) and st != full:
                            print(f"[WARN] {lib.clips[ci].name}: completed in {st} steps, expected {full}")
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=full,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(full, st, 1)),
                            norm_return_survived=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            overflow_steps=int(ov_steps[e]),
                            first_overflow_step=int(ov_first[e]),
                            max_height_error=float(max_h[e]),
                            max_tilt_error=float(max_t[e]),
                        )
                        if recording is not None:
                            recording.qpos[lib.clips[ci].name] = torch.stack([x[e] for x in rec_q[-st:]]).cpu().numpy()
                            recording.ref_qpos[lib.clips[ci].name] = torch.stack([x[e] for x in rec_ref[-st:]]).cpu().numpy()
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
                ov_steps[ids] = 0
                ov_first[ids] = -1
                max_h[ids] = 0.0
                max_t[ids] = 0.0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return: total reward over total attainable reward of the whole benchmark.
    tot_frames = sum(r.frames for r in results)
    w_ret = sum(r.norm_return * r.frames for r in results) / max(tot_frames, 1)
    old_mean = sum(r.norm_return_survived for r in results) / max(n, 1)
    tot_steps = sum(r.steps for r in results)
    at_end = sum(failed_at_end(r.frames, r.failed, r.fail_step) for r in results)
    out = {"clips": n, "failed": failed, "failed_at_end": at_end, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret, "mean_norm_return_survived_v1": old_mean}
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"checkpoint": meta or {}, "summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
