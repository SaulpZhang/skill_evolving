"""Seeded uniform selector with no replacement within each epoch."""

import numpy as np

from spg_bandit.modules.dataset.base import TaskPool
from spg_bandit.modules.selector.base import BaseSelector


class UniformSelector(BaseSelector):
    """Sample a fresh uniform permutation for each pass over the task pool."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self._position = 0
        self._pool_size: int | None = None

    def select(self, task_pool: TaskPool) -> int:
        if task_pool.M < 1:
            raise ValueError("Cannot select from an empty task pool")
        if self._pool_size != task_pool.M or self._position >= len(self._order):
            self._pool_size = task_pool.M
            self._order = self._rng.permutation(task_pool.M).tolist()
            self._position = 0
        task_id = self._order[self._position]
        self._position += 1
        return task_id

    def update(self, task_id: int, result: dict):
        pass

    def reset(self):
        self._rng = np.random.default_rng(self._seed)
        self._order = []
        self._position = 0
        self._pool_size = None

    def save_checkpoint(self) -> dict:
        return {
            "rng_state": self._rng.bit_generator.state,
            "order": self._order,
            "position": self._position,
            "pool_size": self._pool_size,
        }

    def load_checkpoint(self, state: dict):
        if not state:
            return
        self._rng.bit_generator.state = state["rng_state"]
        self._order = [int(task_id) for task_id in state.get("order", [])]
        self._position = int(state.get("position", 0))
        pool_size = state.get("pool_size")
        self._pool_size = int(pool_size) if pool_size is not None else None
