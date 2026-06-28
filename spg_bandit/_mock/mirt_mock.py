"""MIRT-mock skill evolution method (Block 7, Method A / default Env-Synth).

Agent profile s_t ∈ [0,1]^K evolves via the proposal's MIRT Bayesian update
(Equation 4: P(success) = σ(a_τ^⊤ s - d_τ)).
"""

import numpy as np
from scipy.special import expit as sigmoid
from spg_bandit.core.interfaces import SkillEvolvingMethod


class MIRTMockMethod(SkillEvolvingMethod):
    """Skill evolution via MIRT model with Bayesian online profile updates.

    Args:
        A: (M, K) loading matrix.
        d: (M,) difficulty vector.
        s0: (K,) initial profile.
        sigma_s: Smoothing prior std for online update.
        seed: Random seed.
    """

    def __init__(
        self,
        A: np.ndarray,
        d: np.ndarray,
        s0: np.ndarray,
        sigma_s: float = 1.0,
        seed: int | None = None,
    ):
        self._A = A
        self._d = d
        self._s = s0.copy()
        self.sigma_s = sigma_s
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
        """Bayesian one-step Newton update (proposal §3.1.4)."""
        from spg_bandit.mirt import online_update_profile

        s_new = online_update_profile(
            self._s,
            self._A[task_id],
            self._d[task_id],
            success,
            sigma_s=self.sigma_s,
        )
        self._s = s_new
        self._history.append(s_new.copy())

    def get_K(self) -> int:
        return self._K

    def get_history(self) -> list[np.ndarray]:
        return self._history.copy()
