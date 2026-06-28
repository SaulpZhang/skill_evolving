"""Uniform sampling baseline: cycle through tasks in fixed order."""

import numpy as np
from spg_bandit.core.interfaces import TaskPool
from .base import BaseSelector


class UniformSelector(BaseSelector):
    """Cycle through tasks in order (t % M)."""

    def __init__(self):
        super().__init__(name="uniform")
        self._t = 0

    def select(self, task_pool: TaskPool, profile: np.ndarray) -> int:
        return self._t % len(task_pool)

    def update(self, task_id, profile_before, profile_after, success):
        self._t += 1

    def reset(self):
        self._t = 0
