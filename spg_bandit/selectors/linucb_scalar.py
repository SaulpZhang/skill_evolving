"""Standard LinUCB on scalar reward (overall success rate).

Context = task embedding + current skill profile.
Reward = scalar success ∈ {0, 1}.
"""

import numpy as np
from spg_bandit.core.interfaces import TaskPool
from .base import BaseSelector


class LinUCBScalar(BaseSelector):
    """LinUCB on scalar reward (K=1)."""

    def __init__(self, alpha: float = 0.1, lambda_reg: float = 1.0):
        super().__init__(name=f"linucb_scalar_alpha{alpha}")
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._theta: np.ndarray | None = None
        self._last_context: np.ndarray | None = None

    def _build_context(self, task_pool: TaskPool, profile: np.ndarray) -> np.ndarray:
        """Concatenate task embedding with current profile."""
        M = len(task_pool)
        d_c = task_pool.d_c
        K = len(profile)
        contexts = np.zeros((M, d_c + K))
        for tau in range(M):
            contexts[tau] = np.concatenate([task_pool.get_embedding(tau), profile])
        return contexts

    def select(self, task_pool: TaskPool, profile: np.ndarray) -> int:
        contexts = self._build_context(task_pool, profile)
        M = contexts.shape[0]
        d = contexts.shape[1]

        if self._A is None:
            self._A = self.lambda_reg * np.eye(d)
            self._b = np.zeros(d)
            # Explore first task randomly
            chosen = int(np.random.randint(M))
            self._last_context = contexts[chosen]
            return chosen

        if self._theta is None:
            self._theta = np.linalg.solve(self._A, self._b)

        scores = np.zeros(M)
        for tau in range(M):
            x = contexts[tau]
            mu = x @ self._theta
            uncertainty = self.alpha * np.sqrt(x @ np.linalg.solve(self._A, x))
            scores[tau] = mu + uncertainty

        chosen = int(np.argmax(scores))
        self._last_context = contexts[chosen]
        return chosen

    def update(
        self,
        task_id: int,
        profile_before: np.ndarray,
        profile_after: np.ndarray,
        success: bool,
    ) -> None:
        if self._last_context is not None:
            x = self._last_context
            self._A += np.outer(x, x)
            self._b += float(success) * x
            self._theta = np.linalg.solve(self._A, self._b)
            self._last_context = None

    def reset(self):
        self._A = None
        self._b = None
        self._theta = None
        self._last_context = None
