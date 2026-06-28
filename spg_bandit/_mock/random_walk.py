"""Random walk skill evolution method (Block 7, Method D, negative control).

s_t evolves via a random walk with drift based on task success,
unrelated to MIRT structure. Tests: SPG-Bandit should NOT help
when there's no learnable structure (validation of the framework).
"""

import numpy as np
from scipy.special import expit as sigmoid
from spg_bandit.core.interfaces import SkillEvolvingMethod


class RandomWalkMethod(SkillEvolvingMethod):
    """Skill evolution via random walk (negative control).

    Profile evolves mostly randomly, with a small bias from task outcome.
    Task success probability is still σ(a^T s - d), but profile updates
    are dominated by noise — so task selection shouldn't matter much.

    Args:
        A: (M, K) loading matrix.
        d: (M,) difficulty vector.
        s0: (K,) initial profile.
        noise_scale: Standard deviation of random walk step.
        drift: Small bias per step (default 0, pure random walk).
        seed: Random seed.
    """

    def __init__(
        self,
        A: np.ndarray,
        d: np.ndarray,
        s0: np.ndarray,
        noise_scale: float = 0.1,
        drift: float = 0.0,
        seed: int | None = None,
    ):
        self._A = A
        self._d = d
        self._s = s0.copy()
        self._K = len(s0)
        self.noise_scale = noise_scale
        self.drift = drift
        self._rng = np.random.default_rng(seed)
        self._history: list[np.ndarray] = [s0.copy()]

    def execute(self, task_id: int) -> bool:
        theta = self._A[task_id] @ self._s - self._d[task_id]
        p = sigmoid(theta)
        return bool(self._rng.random() < p)

    def get_profile(self) -> np.ndarray:
        return self._s.copy()

    def update_profile(self, task_id, success):
        # Random walk: mostly noise, tiny signal from task outcome
        noise = self._rng.normal(0, self.noise_scale, self._K)
        signal = self.drift * float(success) * np.ones(self._K)
        self._s = np.clip(self._s + noise + signal, 0.0, 1.0)
        self._history.append(self._s.copy())

    def get_K(self) -> int:
        return self._K

    def get_history(self) -> list[np.ndarray]:
        return self._history.copy()
