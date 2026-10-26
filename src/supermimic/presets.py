"""Named training configurations: ``sm-train <preset> [--overrides ...]`` starts from one of these.

Only the fields that differ from the TrainConfig defaults are set here; everything derived from them (goal
dimensions, value cap, compact library storage) is filled in by the trainer as for any other run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from supermimic.config import TrainConfig


def release() -> TrainConfig:
    """The released model's recipe: paper kernel, one RTX 4090, 86,000 iterations (about 10.5 h).

    Paper-filtered BONES-SEED library (``tools/prepare_data.sh``), SONIC-format goals with the pelvis
    orientation anchor (the deploy runtime's observation layout), self-collisions on, fp32 throughout.
    Batch 2048 imagined rollouts fits the 4090's 24 GB next to 4096 environments; the learning rate decays
    over the first 69,000 iterations and holds its floor for the rest.
    """
    cfg = TrainConfig(run_name="release")
    cfg.data.library = Path("data/library/bones_paper.pt")
    cfg.env.num_envs = 4096
    cfg.env.robot = "g1"
    cfg.env.self_collision = True
    cfg.env.goal_features = "sonic"
    cfg.env.anchor_obs = "pelvis_ori10"
    cfg.policy_trainer.batch_size = 2048
    cfg.schedule.total_iterations = 86_000
    cfg.schedule.lr_decay_iterations = 69_000
    cfg.amp = "none"
    return cfg


PRESETS: dict[str, tuple[str, Callable[[], TrainConfig]]] = {
    "release": ("the released model's recipe (one RTX 4090, 86k iterations)", release),
}
