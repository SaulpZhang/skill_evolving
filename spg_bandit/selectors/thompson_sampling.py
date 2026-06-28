"""Thompson Sampling on scalar reward (Bayesian linear regression).

Models P(success | x) = Φ(x^T θ) with Gaussian prior on θ.
At each step, sample θ ~ posterior, then pick task with highest expected reward.
"""

import numpy as np
from spg_bandit.core.interfaces import TaskPool
from .base import BaseSelector


class ThompsonSampling(BaseSelector):
    """Thompson Sampling with Bayesian linear regression on scalar reward."""

    def __init__(self, lambda_reg: float = 1.0, sigma_noise: float = 1.0):
        super().__init__(name=f"ts_scalar_lambda{lambda_reg}")
        self.lambda_reg = lambda_reg
        self.sigma_noise = sigma_noise
        self._A: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._last_context: np.ndarray | None = None

    def _build_context(self, task_pool: TaskPool, profile: np.ndarray) -> np.ndarray:
        d_c = task_pool.d_c
        K = len(profile)
        M = len(task_pool)
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
            chosen = int(np.random.randint(M))
            self._last_context = contexts[chosen]
            return chosen

        # Sample θ from posterior N(θ_hat, Σ)
        theta_hat = np.linalg.solve(self._A, self._b)
        cov = self.sigma_noise**2 * np.linalg.inv(self._A)
        theta_sample = np.random.multivariate_normal(theta_hat, cov)

        scores = contexts @ theta_sample
        chosen = int(np.argmax(scores))
        self._last_context = contexts[chosen]
        return chosen

    def update(self, task_id, profile_before, profile_after, success):
        if self._last_context is not None:
            x = self._last_context
            self._A += np.outer(x, x)
            self._b += float(success) * x
            self._last_context = None

    def reset(self):
        self._A = None
        self._b = None
        self._last_context = None
