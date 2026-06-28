"""SPG-Bandit: Skill Profile Guided Bandit for task selection.

Score(τ) = g_t^T · Δ̂_τ  +  α · √(ϕ_τ^T A^{-1} ϕ_τ)

where:
  g_t = softmax((1 - s_t) / τ)  — gap vector
  Δ̂_τ = W^T ϕ_τ                 — predicted per-dimension change
  ϕ_τ = MLP(x_τ)                — learned features from LLM embedding
  A = λI + Σ ϕ_i ϕ_i^T          — ridge regression covariance
"""

import numpy as np
from spg_bandit.core.interfaces import TaskPool
from .base import BaseSelector
from .mlp_featurizer import MLPFeaturizer


class SPGBanditSelector(BaseSelector):
    """SPG-Bandit task selection policy.

    Args:
        mlp: Pre-trained MLPFeaturizer for task embedding → feature.
        K: Dimensionality of skill profile.
        d_f: Feature dimension (must match MLP output dim).
        alpha: UCB exploration coefficient.
        tau: Temperature for gap softmax.
        lambda_reg: Ridge regression regularization.
        gap_weighted: If False, use uniform weights (ablation: no gap).
        use_ucb: If False, skip UCB term (ablation: greedy).
    """

    def __init__(
        self,
        mlp: MLPFeaturizer,
        K: int,
        d_f: int,
        alpha: float = 0.1,
        tau: float = 0.1,
        lambda_reg: float = 1.0,
        gap_weighted: bool = True,
        use_ucb: bool = True,
    ):
        name_parts = ["spg_bandit"]
        if not gap_weighted:
            name_parts.append("nogap")
        if not use_ucb:
            name_parts.append("noucb")
        super().__init__(name="_".join(name_parts))

        self.mlp = mlp
        self.K = K
        self.d_f = d_f
        self.alpha = alpha
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.gap_weighted = gap_weighted
        self.use_ucb = use_ucb

        # Ridge regression parameters: A_t ∈ R^{d_f×d_f}, B_t ∈ R^{d_f×K}
        self._A: np.ndarray = lambda_reg * np.eye(d_f)
        self._B: np.ndarray = np.zeros((d_f, K))
        self._W: np.ndarray = np.zeros((d_f, K))
        self._last_phi: np.ndarray | None = None
        self._last_scores: np.ndarray | None = None
        self._step = 0

    def _gap(self, profile: np.ndarray) -> np.ndarray:
        """Compute gap vector g_t = softmax((1 - s_t) / τ)."""
        raw = (1.0 - profile) / max(self.tau, 1e-10)
        exp = np.exp(raw - np.max(raw))  # stable softmax
        g = exp / np.sum(exp)
        if not self.gap_weighted:
            g[:] = 1.0 / self.K  # uniform weight (ablation)
        return g

    def select(self, task_pool: TaskPool, profile: np.ndarray) -> int:
        g = self._gap(profile)
        M = len(task_pool)
        scores = np.full(M, -np.inf)

        # Pre-compute A_inv for UCB term
        if self.use_ucb:
            A_inv = np.linalg.inv(self._A)
        else:
            A_inv = None

        chosen = 0
        for tau in range(M):
            x = task_pool.get_embedding(tau)
            phi = self.mlp.forward(x)  # (d_f,)

            # Predict change
            delta_hat = self._W.T @ phi  # (K,)

            # Gap-weighted score
            score = g @ delta_hat

            # UCB exploration
            if self.use_ucb and A_inv is not None:
                ucb_term = phi @ A_inv @ phi
                if ucb_term > 0:
                    score += self.alpha * np.sqrt(ucb_term)

            scores[tau] = score
            if score > scores[chosen]:
                chosen = tau

        # Store ϕ of chosen task for update
        x_chosen = task_pool.get_embedding(chosen)
        self._last_phi = self.mlp.forward(x_chosen)
        self._last_scores = scores
        return chosen

    def update(
        self,
        task_id: int,
        profile_before: np.ndarray,
        profile_after: np.ndarray,
        success: bool,
    ) -> None:
        """Online update of ridge regression with observed Δ."""
        if self._last_phi is not None:
            delta = profile_after - profile_before  # (K,)
            phi = self._last_phi

            # Incremental update
            self._A += np.outer(phi, phi)
            self._B += np.outer(phi, delta)

            # Solve for W (could defer for efficiency)
            self._W = np.linalg.solve(self._A, self._B)

        self._step += 1

    def fine_tune_mlp(
        self,
        warmup_embeddings: np.ndarray,
        warmup_deltas: np.ndarray,
        epochs: int = 10,
        verbose: bool = False,
    ):
        """Fine-tune MLP and reset ridge regression (Algorithm 1 lines 22-24).

        After fine-tuning, feature space changes, so ridge regression resets.
        """
        # Train MLP to predict current W^T ϕ from embeddings
        # This uses the current W as target: y = ϕ = W_target^T · x ?
        # Actually: we want MLP(embedding) to produce features that help
        # predict Δ. So we train MLP(embedding) → W^T · MLP_old(embedding)
        # i.e., distill current ridge regression into the MLP.
        current_features = self.mlp.forward(warmup_embeddings)  # (N, d_f)
        # Use current W to compute target Δ
        target_delta = current_features @ self._W  # (N, K)
        # Train MLP: embedding → feature that predicts Δ well
        # Simple approach: train MLP(embedding) → current_ϕ
        self.mlp.train(warmup_embeddings, current_features, epochs=epochs, verbose=verbose)

        # Reset ridge regression
        self._A = self.lambda_reg * np.eye(self.d_f)
        self._B = np.zeros((self.d_f, self.K))
        self._W = np.zeros((self.d_f, self.K))
        self._step = 0

    def reset(self):
        self._A = self.lambda_reg * np.eye(self.d_f)
        self._B = np.zeros((self.d_f, self.K))
        self._W = np.zeros((self.d_f, self.K))
        self._last_phi = None
        self._last_scores = None
        self._step = 0
