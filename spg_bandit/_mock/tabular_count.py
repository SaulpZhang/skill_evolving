"""Tabular count-based skill evolution method (Block 7, Method C).

Each task has a per-dimension "practice count" c_τk.
s[k] = f(Σ_τ c_τk) where f is a saturating function 1 - exp(-γ · count).
Executing task τ increments its count vector along loaded dimensions.
"""

import numpy as np
from scipy.special import expit as sigmoid
from spg_bandit.core.interfaces import SkillEvolvingMethod


class TabularCountMethod(SkillEvolvingMethod):
    """Skill evolution via practice counts with saturating function.

    Args:
        A: (M, K) loading matrix — used to define which dimensions a task practices.
        d: (M,) difficulty vector.
        s0: (K,) initial profile (determines initial counts).
        gamma: Saturation rate for the practice→skill function.
        seed: Random seed.
    """

    def __init__(
        self,
        A: np.ndarray,
        d: np.ndarray,
        s0: np.ndarray,
        gamma: float = 0.1,
        seed: int | None = None,
    ):
        self._A = A
        self._d = d
        self._M, self._K = A.shape
        self.gamma = gamma
        self._rng = np.random.default_rng(seed)

        # Initialize counts from s0: c_k = -ln(1 - s0_k) / γ
        self._counts: np.ndarray = -np.log(np.clip(1.0 - s0, 1e-10, 1.0)) / gamma

        # current s from counts
        self._s = self._counts_to_profile()
        self._history: list[np.ndarray] = [self._s.copy()]

    def _counts_to_profile(self) -> np.ndarray:
        """s[k] = 1 - exp(-γ * count_k)"""
        return 1.0 - np.exp(-self.gamma * self._counts)

    def execute(self, task_id: int) -> bool:
        theta = self._A[task_id] @ self._s - self._d[task_id]
        p = sigmoid(theta)
        return bool(self._rng.random() < p)

    def get_profile(self) -> np.ndarray:
        return self._s.copy()

    def update_profile(self, task_id: int, success: bool) -> None:
        if success:
            # Increment counts along task's loaded dimensions
            a = np.abs(self._A[task_id])
            self._counts += a / (a.sum() + 1e-10)  # normalize contribution
        # (failure doesn't reduce counts — only success builds skill)
        self._s = self._counts_to_profile()
        self._history.append(self._s.copy())

    def get_K(self) -> int:
        return self._K

    def get_history(self) -> list[np.ndarray]:
        return self._history.copy()
