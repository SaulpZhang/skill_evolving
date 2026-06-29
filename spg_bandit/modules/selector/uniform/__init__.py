"""Uniform selector: cycle through tasks in order."""

from spg_bandit.modules.dataset.base import TaskPool
from spg_bandit.modules.selector.base import BaseSelector


class UniformSelector(BaseSelector):
    """Cycle through tasks in order (t % M)."""

    def __init__(self):
        self._t = 0

    def select(self, task_pool: TaskPool) -> int:
        tid = self._t % task_pool.M
        self._t += 1
        return tid

    def update(self, task_id: int, result: dict):
        pass

    def reset(self):
        self._t = 0
