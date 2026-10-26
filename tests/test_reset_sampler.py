"""Return-based reset sampler (paper D.1): per-bin return estimates, bootstraps, exclusions, sync.

The sampler state lives in ``MultiClipMotionCommand``; these tests drive its bookkeeping directly on a
synthetic clip index (no simulator, no motion data), the way the env does at every reset: per-step
rewards and bins are logged, then ``_finalize_episodes`` folds the finished episodes into the bins.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from supermimic.envs.mdp.motion_library import ClipInfo, MotionLibrary

G_MAX = 10.0  # r_max 0.1 / (1 - gamma 0.99)


def make_command(clips: list[tuple[str, int]], num_envs: int, estimator: str = "count", max_steps: int = 512):
    lib = MotionLibrary.__new__(MotionLibrary)
    lib.device = torch.device("cpu")
    lib.bin_frames = 50
    start, infos = 0, []
    for name, length in clips:
        infos.append(ClipInfo(name, start, length))
        start += length
    lib.clips, lib.num_clips, lib.num_frames = infos, len(infos), start
    lib._index_clips()

    cmd = MultiClipMotionCommand.__new__(MultiClipMotionCommand)
    cmd.cfg = MultiClipMotionCommandCfg(
        npz_files=[], anchor_body_name="torso_link", body_names=("torso_link",), bin_estimator=estimator  # type: ignore[arg-type]
    )
    cmd._env = SimpleNamespace(
        num_envs=num_envs, device="cpu", termination_manager=SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool))
    )
    cmd.library = lib
    cmd.metrics = {k: torch.zeros(num_envs) for k in ("sampling_entropy", "episode_return_to_go")}
    cmd._init_sampler(num_envs)
    cmd.max_ep_steps = max_steps
    cmd.ep_rewards = torch.zeros(num_envs, max_steps)
    cmd.ep_bins = torch.zeros(num_envs, max_steps, dtype=torch.long)
    cmd.ep_len = torch.zeros(num_envs, dtype=torch.long)
    cmd._gamma_pow = cmd.cfg.gamma ** torch.arange(max_steps, dtype=torch.float64)
    cmd.bootstrap_fn = None
    cmd.clip_end = torch.zeros(num_envs, dtype=torch.bool)
    return cmd


def finish(cmd, starts: torch.Tensor, lengths: torch.Tensor, reward: torch.Tensor, failed: torch.Tensor) -> None:
    """Episodes on envs 0..n-1 that ran ``lengths`` steps from frame ``starts`` earning ``reward`` per step,
    ended by a failure (``failed``) or otherwise by the clip end / a time-out."""
    n = starts.numel()
    ids = torch.arange(n)
    steps = torch.arange(cmd.max_ep_steps)
    frames = torch.minimum(starts[:, None] + steps[None, :], cmd.library.last_frame_of(starts)[:, None])
    cmd.ep_bins[ids] = cmd.library.bin_of_frame(frames.reshape(-1)).reshape(n, -1)
    cmd.ep_rewards[ids] = reward[:, None].expand(n, cmd.max_ep_steps).float()
    cmd.ep_len[ids] = lengths
    cmd._env.termination_manager.terminated[ids] = failed
    cmd.clip_end[ids] = ~failed
    cmd._finalize_episodes(ids)
    cmd.clip_end[ids] = False


def rtg(rewards: list[float], gamma: float = 0.99, tail: float = 0.0) -> torch.Tensor:
    """Discounted return-to-go of each step, plus ``tail`` (the value after the last step)."""
    out, g = [], tail
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return torch.tensor(out[::-1])


def test_unvisited_bins_start_hard_and_first_visit_sets_the_value():
    cmd = make_command([("walk_a", 500)], num_envs=1)
    assert torch.all(cmd.bin_return == 0.0) and torch.all(cmd.bin_count == 0)
    # A failure after 30 steps in bin 0 at reward 0.05: the bin reads the mean return-to-go of its steps.
    finish(cmd, torch.tensor([0]), torch.tensor([30]), torch.tensor([0.05]), torch.tensor([True]))
    expected = rtg([0.05] * 30).mean()
    assert cmd.bin_return[0] == pytest.approx(float(expected), rel=1e-5)
    assert cmd.bin_count[0] == 1 and torch.all(cmd.bin_return[1:] == 0.0)
    # The second visit averages with the first (running mean), not a 5 % step towards it.
    finish(cmd, torch.tensor([0]), torch.tensor([30]), torch.tensor([0.08]), torch.tensor([True]))
    second = rtg([0.08] * 30).mean()
    assert cmd.bin_return[0] == pytest.approx(float((expected + second) / 2), rel=1e-5)


def test_running_mean_turns_into_the_ema_after_twenty_visits():
    cmd = make_command([("walk_a", 500)], num_envs=1)
    cmd.bin_count[0] = 40.0
    cmd.bin_return[0] = 5.0
    finish(cmd, torch.tensor([0]), torch.tensor([30]), torch.tensor([0.05]), torch.tensor([True]))
    obs = float(rtg([0.05] * 30).mean())
    assert cmd.bin_return[0] == pytest.approx(0.95 * 5.0 + 0.05 * obs, rel=1e-5)


def test_well_tracked_clip_reads_g_max_up_to_its_last_bin():
    # 120 frames: bins of 50, 50 and a 20-frame tail. Perfect tracking to the clip end.
    cmd = make_command([("walk_a", 120), ("walk_b", 500)], num_envs=1)
    finish(cmd, torch.tensor([0]), torch.tensor([120]), torch.tensor([0.1]), torch.tensor([False]))
    torch.testing.assert_close(cmd.bin_return[:3], torch.full((3,), G_MAX), atol=1e-4, rtol=0)
    # The fork's estimator leaves the clip end unbootstrapped: its last bin reads as a failure would.
    old = make_command([("walk_a", 120), ("walk_b", 500)], num_envs=1, estimator="ema")
    old.bin_return.zero_()  # isolate the per-visit observation (alpha 1 would be the running mean's first step)
    old.cfg.ema_alpha = 1.0
    finish(old, torch.tensor([0]), torch.tensor([120]), torch.tensor([0.1]), torch.tensor([False]))
    assert old.bin_return[2] < 0.3 * G_MAX


def test_time_out_is_bootstrapped_with_the_final_reward_rate():
    cmd = make_command([("walk_long", 1500)], num_envs=1)
    finish(cmd, torch.tensor([0]), torch.tensor([500]), torch.tensor([0.06]), torch.tensor([False]))
    tail = 0.06 / 0.01
    expected = rtg([0.06] * 500, tail=tail).view(10, 50).mean(1)
    torch.testing.assert_close(cmd.bin_return[:10], expected.float(), rtol=1e-5, atol=1e-4)


def test_failures_read_low_and_negative_returns_are_clamped():
    cmd = make_command([("flip_a", 150), ("walk_b", 500)], num_envs=2)
    finish(cmd, torch.tensor([0, 150]), torch.tensor([20, 40]), torch.tensor([0.02, -0.3]), torch.tensor([True, True]))
    assert 0.0 < cmd.bin_return[0] < 0.5
    assert cmd.bin_return[3] == 0.0  # walk_b's first bin: negative return clamped to 0


def test_unphysical_episodes_are_left_out():
    cmd = make_command([("walk_a", 500)], num_envs=2)
    cmd.unphysical[1] = True
    finish(cmd, torch.tensor([0, 100]), torch.tensor([30, 30]), torch.tensor([0.05, 0.05]), torch.tensor([True, True]))
    assert cmd.bin_count[0] == 1 and cmd.bin_count[2] == 0 and cmd.bin_return[2] == 0.0
    assert cmd.unphysical_episodes == 1 and not cmd.unphysical.any()


def test_unvisited_bins_outweigh_visited_easy_bins():
    cmd = make_command([("walk_a", 500), ("walk_b", 500)], num_envs=1)
    finish(cmd, torch.tensor([0]), torch.tensor([500]), torch.tensor([0.095]), torch.tensor([False]))
    assert torch.all(cmd.bin_count[:10] == 1) and torch.all(cmd.bin_count[10:] == 0)
    f = cmd._sample_frames(20000)
    on_b = (cmd.library.frame_clip[f] == 1).float().mean()
    assert on_b > 0.65  # uniform alone would give 0.5


def run_sampler(estimator: str, easy_n: int, n_envs: int, iters: int, hard_n: int = 130) -> dict[str, float]:
    """Train-like reset loop on ``hard_n`` short acrobatic clips (fail after 20 steps) among ``easy_n``
    easy ones (tracked to the clip end); sampler statistics over the last third of the iterations."""
    torch.manual_seed(0)
    cmd = make_command([(f"flip_360_{i:03d}", 150) for i in range(hard_n)] + [(f"walk_{i}", 300) for i in range(easy_n)], n_envs, estimator)
    lib = cmd.library
    for it in range(iters):
        if it == 2 * iters // 3:
            cmd.reset_group_counts.zero_()
        f = cmd._sample_frames(n_envs)
        hard = lib.frame_clip[f] < hard_n
        left = lib.last_frame_of(f) - f + 1
        fails = hard & (left > 20)
        lengths = torch.where(fails, torch.full_like(left, 20), left.clamp_max(cmd.max_ep_steps))
        finish(cmd, f, lengths, torch.where(hard, 0.02, 0.095), fails)
    stats = cmd.sampler_stats()
    stats["flips_x_uniform"] = stats["flips/reset_share"] / stats["flips/uniform_share"]
    # The fork keeps no visit counts: a visited bin is one whose estimate left g_max.
    seen = cmd.bin_count > 0 if estimator == "count" else cmd.bin_return < cmd.cfg.g_max
    stats["untouched_frac"] = float((~seen).float().mean())
    flips = cmd.bin_group == cmd.group_names.index("flips")
    stats["visited_flips_bin_return"] = float(cmd.bin_return[seen & flips].mean())
    return stats


def test_fixed_estimator_covers_the_library_and_finds_the_hard_clips():
    """~25k bins, ~3 episode starts per bin: everything gets visited and the 390 acrobatic bins take the
    hard mass (>= 10x a uniform sampler's resets there)."""
    stats = run_sampler("count", easy_n=4000, n_envs=512, iters=150)
    assert stats["untouched_frac"] < 0.05, stats
    assert stats["flips/bin_return"] < stats["all_other/bin_return"]
    assert stats["flips_x_uniform"] >= 10.0, stats


def test_fork_estimator_misses_the_hard_clips_when_bins_are_rarely_visited():
    """~48k bins and ~0.3 episode starts per bin (the ratio of a full-library run at 8k iterations: ~1M bins). The fork's
    optimistic EMA still rates the visited acrobatic bins as easy and sends no extra resets there; the
    fixed estimator rates them as failures after one visit and is still sweeping the unvisited bins
    (which it treats as hard), so it covers the library faster."""
    old = run_sampler("ema", easy_n=8000, n_envs=128, iters=120)
    new = run_sampler("count", easy_n=8000, n_envs=128, iters=120)
    assert old["visited_flips_bin_return"] > 0.8 * G_MAX, old
    assert old["flips_x_uniform"] < 3.0, old
    assert new["visited_flips_bin_return"] < 0.1 * G_MAX, new
    assert new["untouched_frac"] < old["untouched_frac"], (new, old)


def _sync_worker(rank: int, world: int, path: str, q) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=world)
    cmd = make_command([("walk_a", 500)], num_envs=1)
    # Rank 0 visits bin 0 three times (value 6), rank 1 once (value 2); bin 5 only on rank 1.
    cmd.bin_count[0] = 3.0 if rank == 0 else 1.0
    cmd.bin_return[0] = 6.0 if rank == 0 else 2.0
    if rank == 1:
        cmd.bin_count[5], cmd.bin_return[5] = 1.0, 7.0
    cmd.sync_bins()
    q.put((rank, float(cmd.bin_return[0]), float(cmd.bin_count[0]), float(cmd.bin_return[5]), float(cmd.bin_count[5])))
    dist.destroy_process_group()


def test_bins_merge_across_ranks(tmp_path):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    path = str(tmp_path / "pg")
    procs = [ctx.Process(target=_sync_worker, args=(r, 2, path, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    res = sorted(q.get() for _ in range(2))
    for _, g0, n0, g5, n5 in res:
        assert g0 == pytest.approx((3 * 6.0 + 1 * 2.0) / 4) and n0 == 4.0
        assert g5 == pytest.approx(7.0) and n5 == 1.0


def test_sampler_stats_report_every_group():
    cmd = make_command([("flip_a", 150), ("side_roll_a", 100), ("walk_a", 500)], num_envs=4)
    cmd._sample_frames(1000)
    stats = cmd.sampler_stats()
    for g in cmd.group_names:
        assert f"{g}/uniform_share" in stats and f"{g}/reset_share" in stats
    assert sum(stats[f"{g}/reset_share"] for g in cmd.group_names) == pytest.approx(1.0)
    assert stats["untouched_frac"] == 1.0
    assert cmd.reset_group_counts.sum() == 0  # cleared for the next interval
