"""Delta-increment skill evolution method (Block 7, Method B).

Simpler dynamics without MIRT Bayesian update:
After success on task τ, s[k] += η · a_τk (increment along task's loaded dimensions).
After failure, s[k] -= η · a_τk.
Clamped to [0, 1].
"""

import numpy as np
from scipy.special import expit as sigmoid
from spg_bandit.core.interfaces import SkillEvolvingMethod


class DeltaIncrementMethod(SkillEvolvingMethod):
    """Skill evolution via simple delta increments along task loadings.

    Args:
        A: (M, K) loading matrix — how task τ loads on each dimension.
        d: (M,) difficulty vector — determines success probability.
        s0: (K,) initial profile.
        eta: Step size for each update. If None, use 0.05 / max(A[task_id]).
        seed: Random seed.
    """

    def __init__(
        self,
        A: np.ndarray,
        d: np.ndarray,
        s0: np.ndarray,
        eta: float | None = None,
        seed: int | None = None,
    ):
        self._A = A
        self._d = d
        self._s = s0.copy()
        self._eta = eta
        self._K = len(s0)
        self._rng = np.random.default_rng(seed)
        self._history: list[np.ndarray] = [s0.copy()]

    def execute(self, task_id: int) -> bool:
        theta = self._A[task_id] @ self._s - self._d[task_id]
        p = sigmoid(theta)
        return bool(self._rng.random() < p)

    def get_profile(self) -> np.ndarray:
        return self._s.copy()

    def update_profile(self, task_id: int, success: bool) -> None:
        a = self._A[task_id]
        eta = self._eta if self._eta is not None else 0.05 / max(np.abs(a).max(), 1e-10)
        if success:
            self._s += eta * a
        else:
            self._s -= eta * a
        self._s = np.clip(self._s, 0.0, 1.0)
        self._history.append(self._s.copy())

    def get_K(self) -> int:
        return self._K

    def get_history(self) -> list[np.ndarray]:
        return self._history.copy()
