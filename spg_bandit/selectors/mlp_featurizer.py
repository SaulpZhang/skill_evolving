"""MLP featurizer for SPG-Bandit.

Architecture:
  ϕ = W2 · ReLU(W1 · x + b1) + b2

Pre-trained on warmup data to predict observed ∆ via an auxiliary
linear readout head (d_f → K). After training, the readout head is
discarded and only the d_f-dim features ϕ are used by ridge regression.
Fine-tuned every 50 steps during online phase.
"""

import numpy as np
from numpy.typing import NDArray


class MLPFeaturizer:
    """Two-layer MLP: R^d_c → R^d_h → R^d_f.

    During train(), an auxiliary linear head R^d_f → R^K predicts ∆.
    During forward(), only the d_f-dim features are returned.
    """

    def __init__(
        self,
        d_c: int,
        d_h: int = 128,
        d_f: int = 64,
        lr: float = 1e-3,
        seed: int | None = None,
    ):
        self.d_c = d_c
        self.d_h = d_h
        self.d_f = d_f
        self.lr = lr
        self.rng = np.random.default_rng(seed)

        # Xavier init for base MLP
        bound1 = np.sqrt(6.0 / (d_c + d_h))
        self.W1 = self.rng.uniform(-bound1, bound1, (d_c, d_h))
        self.b1 = np.zeros(d_h)

        bound2 = np.sqrt(6.0 / (d_h + d_f))
        self.W2 = self.rng.uniform(-bound2, bound2, (d_h, d_f))
        self.b2 = np.zeros(d_f)

        # Auxiliary readout head (created lazily in train())
        self._head_W: np.ndarray | None = None
        self._head_b: np.ndarray | None = None

    def forward(self, x: NDArray) -> NDArray:
        """ϕ = W2 · ReLU(W1 · x + b1) + b2

        Args:
            x: (d_c,) or (N, d_c) input.
        Returns:
            ϕ: (d_f,) or (N, d_f) output features.
        """
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0.0)
        return h @ self.W2 + self.b2

    def _forward_with_head(self, X: NDArray) -> NDArray:
        """Full forward pass including readout head: MLP(x) @ W_head + b_head."""
        phi = self.forward(X)  # (N, d_f)
        return phi @ self._head_W + self._head_b  # (N, K)

    def train(
        self,
        X: NDArray,
        y: NDArray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        """Train MLP features + readout head to predict ∆ (K-dim) from embeddings.

        Args:
            X: (N, d_c) task embeddings.
            y: (N, K) observed deltas.
            epochs: Number of SGD epochs.
            batch_size: Mini-batch size.
        """
        N, K = X.shape[0], y.shape[1]

        # Lazy init readout head
        if self._head_W is None:
            bound = np.sqrt(6.0 / (self.d_f + K))
            self._head_W = self.rng.uniform(-bound, bound, (self.d_f, K))
            self._head_b = np.zeros(K)

        for epoch in range(epochs):
            perm = self.rng.permutation(N)
            X_shuf = X[perm]
            y_shuf = y[perm]
            total_loss = 0.0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                loss = self._train_step(X_batch, y_batch)
                total_loss += loss

            if verbose and epoch % 10 == 9:
                avg_loss = total_loss / max(N // batch_size, 1)
                print(f"  MLP epoch {epoch+1:3d}  MSE = {avg_loss:.6f}")

    def _train_step(self, X_batch: NDArray, y_batch: NDArray) -> float:
        """Single SGD step with readout head. Returns batch loss."""
        B = X_batch.shape[0]

        # Forward through MLP
        z1 = X_batch @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        phi = h1 @ self.W2 + self.b2  # (B, d_f) features

        # Forward through readout head
        pred = phi @ self._head_W + self._head_b  # (B, K)

        # MSE loss
        diff = pred - y_batch
        loss = np.mean(diff ** 2)

        # Gradients through readout head
        d_pred = 2 * diff / B  # (B, K)
        d_head_W = phi.T @ d_pred  # (d_f, K)
        d_head_b = d_pred.sum(axis=0)  # (K,)
        d_phi = d_pred @ self._head_W.T  # (B, d_f)

        # Gradients through MLP
        d_h1 = d_phi @ self.W2.T  # (B, d_h)
        d_z1 = d_h1 * (z1 > 0).astype(float)
        d_W2 = h1.T @ d_phi  # (d_h, d_f)
        d_b2 = d_phi.sum(axis=0)  # (d_f,)
        d_W1 = X_batch.T @ d_z1  # (d_c, d_h)
        d_b1 = d_z1.sum(axis=0)  # (d_h,)

        # Update MLP weights
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self._head_W -= self.lr * d_head_W
        self._head_b -= self.lr * d_head_b

        return float(loss)

    def get_params(self) -> dict:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_params(self, params: dict):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
