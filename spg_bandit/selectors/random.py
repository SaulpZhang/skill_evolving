"""Random sampling baseline."""

import numpy as np
from spg_bandit.core.interfaces import TaskPool
from .base import BaseSelector


class RandomSelector(BaseSelector):
    """Uniformly random task selection."""

    def __init__(self, seed: int | None = None):
        super().__init__(name="random")
        self._rng = np.random.default_rng(seed)

    def select(self, task_pool: TaskPool, profile: np.ndarray) -> int:
        return int(self._rng.integers(len(task_pool)))

    def reset(self):
        self._rng = np.random.default_rng()
