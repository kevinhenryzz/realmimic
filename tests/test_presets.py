"""The release preset reproduces the released run's configuration (fields that differ from the defaults)."""

from pathlib import Path

from supermimic.config import TrainConfig
from supermimic.presets import PRESETS, release


def test_release_preset_matches_the_released_run():
    cfg = release()
    assert cfg.data.library == Path("data/library/bones_paper.pt")
    assert (cfg.env.num_envs, cfg.env.robot, cfg.env.self_collision) == (4096, "g1", True)
    assert (cfg.env.goal_features, cfg.env.anchor_obs) == ("sonic", "pelvis_ori10")
    assert cfg.env.kernel_tail == 0.0  # the paper's kernel
    assert cfg.env.freeze_clip_end and cfg.env.bin_estimator == "count"
    assert cfg.world_model.state_spec.frame == "body"
    assert cfg.policy_trainer.batch_size == 2048 and cfg.policy_trainer.start_margin == 0
    assert (cfg.schedule.total_iterations, cfg.schedule.lr_decay_iterations) == (86_000, 69_000)
    assert cfg.amp == "none" and cfg.seed == 0


def test_presets_are_fresh_configs():
    a, b = PRESETS["release"][1](), PRESETS["release"][1]()
    a.env.num_envs = 1
    assert b.env.num_envs == 4096 and TrainConfig().env.num_envs == 4096
