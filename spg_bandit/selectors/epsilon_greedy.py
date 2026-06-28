"""Epsilon-greedy: explore randomly with prob epsilon, else exploit best estimated task.

Uses rolling average success rate per task as the value estimate.
"""

import numpy as np
from spg_bandit.core.interfaces import TaskPool
from .base import BaseSelector


class EpsilonGreedy(BaseSelector):
    def __init__(self, epsilon: float = 0.1, seed: int | None = None):
        super().__init__(name=f"epsilon_greedy_{epsilon}")
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._counts: dict[int, int] = {}       # task_id -> times chosen
        self._successes: dict[int, int] = {}     # task_id -> times succeeded
        self._step = 0

    def select(self, task_pool: TaskPool, profile: np.ndarray) -> int:
        M = len(task_pool)
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(M))
        # Exploit: pick task with highest observed success rate
        best_task = 0
        best_rate = -1.0
        for tau in range(M):
            c = self._counts.get(tau, 0)
            if c == 0:
                return tau  # try unseen tasks first
            rate = self._successes[tau] / c
            if rate > best_rate:
                best_rate = rate
                best_task = tau
        return best_task

    def update(self, task_id, profile_before, profile_after, success):
        if task_id not in self._counts:
            self._counts[task_id] = 0
            self._successes[task_id] = 0
        self._counts[task_id] += 1
        self._successes[task_id] += int(success)
        self._step += 1

    def reset(self):
        self._rng = np.random.default_rng()
        self._counts.clear()
        self._successes.clear()
        self._step = 0
