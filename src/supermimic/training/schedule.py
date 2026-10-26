"""Cosine learning-rate schedule (paper Table A.5): decay to `final_ratio` over `decay_iterations`, then constant.

Optionally a warm restart: from `restart_iteration` a second cosine from `restart_ratio` down to `final_ratio`
over `restart_decay_iterations`, then constant again (continuing a finished run from its checkpoint)."""

from __future__ import annotations

import math


def cosine_lr(iteration: int, base_lr: float, final_ratio: float = 0.03, decay_iterations: int = 250_000) -> float:
    decay = max(1, int(decay_iterations))
    if iteration >= decay:
        return base_lr * final_ratio
    cos = 0.5 * (1.0 + math.cos(math.pi * iteration / decay))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cos)


def scheduled_lr(iteration: int, base_lr: float, final_ratio: float, decay_iterations: int, restart_iteration: int | None = None, restart_ratio: float = 0.3, restart_decay_iterations: int = 60_000) -> float:
    if restart_iteration is None or iteration < restart_iteration:
        return cosine_lr(iteration, base_lr, final_ratio, decay_iterations)
    # cosine_lr from restart_ratio * base_lr to final_ratio * base_lr: rescale the ratios to the restart peak.
    peak = base_lr * restart_ratio
    return cosine_lr(iteration - restart_iteration, peak, final_ratio / restart_ratio, restart_decay_iterations)
