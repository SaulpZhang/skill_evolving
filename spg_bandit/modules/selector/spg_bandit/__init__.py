"""SPG-Bandit selector: gap-weighted multi-output bandit.

Maintains its own skill profile internally, independent of skill evolving method.
"""

import json
import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize

from spg_bandit.modules.dataset.base import TaskPool
from spg_bandit.utils.wandb import log_metrics
from spg_bandit.modules.selector.base import BaseSelector


# ── MLP Featurizer ──────────────────────────────────────────────────────────

class MLPFeaturizer:
    """Two-layer MLP: R^d_c → R^d_h → R^d_f."""

    def __init__(self, d_c: int, d_h: int = 32, d_f: int = 16, seed: int = 42):
        self.d_c, self.d_h, self.d_f = d_c, d_h, d_f
        self.lr = 1e-3
        self.rng = np.random.default_rng(seed)
        b1 = np.sqrt(6.0 / (d_c + d_h))
        self.W1 = self.rng.uniform(-b1, b1, (d_c, d_h))
        self.b1 = np.zeros(d_h)
        b2 = np.sqrt(6.0 / (d_h + d_f))
        self.W2 = self.rng.uniform(-b2, b2, (d_h, d_f))
        self.b2 = np.zeros(d_f)
        self._head_W: np.ndarray | None = None
        self._head_b: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.W1 + self.b1, 0.0)
        return h @ self.W2 + self.b2

    def train(self, X, y, epochs=50, batch_size=32, wandb_prefix="mlp"):
        N, K = X.shape[0], y.shape[1]
        if self._head_W is None:
            bound = np.sqrt(6.0 / (self.d_f + K))
            self._head_W = self.rng.uniform(-bound, bound, (self.d_f, K))
            self._head_b = np.zeros(K)
        loss_history = []
        for epoch in range(epochs):
            perm = self.rng.permutation(N)
            X_s, y_s = X[perm], y[perm]
            total_loss = 0.0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                total_loss += self._train_step(X_s[start:end], y_s[start:end])
            avg_loss = total_loss / max(N // batch_size, 1)
            loss_history.append(avg_loss)
            if epoch % 10 == 9:
                print(f"  MLP epoch {epoch+1}: MSE = {avg_loss:.6f}")
            log_metrics({f"{wandb_prefix}/mse": avg_loss, f"{wandb_prefix}/step": epoch})
        return loss_history

    def _train_step(self, X_batch, y_batch):
        B = X_batch.shape[0]
        z1 = X_batch @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        phi = h1 @ self.W2 + self.b2
        pred = phi @ self._head_W + self._head_b
        diff = pred - y_batch
        loss = np.mean(diff ** 2)
        d_pred = 2 * diff / B
        self._head_W -= self.lr * (phi.T @ d_pred)
        self._head_b -= self.lr * d_pred.sum(axis=0)
        d_phi = d_pred @ self._head_W.T
        d_h1 = d_phi @ self.W2.T
        d_z1 = d_h1 * (z1 > 0).astype(float)
        self.W1 -= self.lr * (X_batch.T @ d_z1)
        self.b1 -= self.lr * d_z1.sum(axis=0)
        self.W2 -= self.lr * (h1.T @ d_phi)
        self.b2 -= self.lr * d_phi.sum(axis=0)
        return float(loss)


# ── MIRT EM ─────────────────────────────────────────────────────────────────

def fit_mirt_em(R, K, max_iter=200, tol=1e-4, verbose=False):
    N_warm, M = R.shape
    obs_mask = ~np.isnan(R)
    R_filled = np.nan_to_num(R, nan=0.0)
    s_hist = np.random.randn(N_warm, K) * 0.1
    A = np.random.randn(M, K) * 0.1
    d_vec = np.zeros(M)
    prev_ll = -np.inf
    ll_history = []

    for it in range(max_iter):
        for t in range(N_warm):
            obs_t = np.where(obs_mask[t])[0]
            if len(obs_t) == 0:
                continue
            s = s_hist[t].copy()
            for _ in range(20):
                theta = A[obs_t] @ s - d_vec[obs_t]
                p = sigmoid(theta)
                grad = A[obs_t].T @ (R_filled[t, obs_t] - p) - s
                Wd = p * (1 - p)
                hess = -A[obs_t].T @ (A[obs_t] * Wd[:, np.newaxis]) - np.eye(K)
                s -= 0.5 * np.linalg.solve(hess, grad)
            s_hist[t] = np.clip(s, -3.0, 3.0)

        for tau in range(M):
            t_idx = np.where(obs_mask[:, tau])[0]
            if len(t_idx) < 2:
                continue
            X, y = s_hist[t_idx], R_filled[t_idx, tau]

            def nll(params):
                a, b = params[:-1], params[-1]
                p = sigmoid(X @ a - b)
                ll = y @ np.log(p + 1e-15) + (1 - y) @ np.log(1 - p + 1e-15)
                return -(ll - 0.01 * np.sum(a ** 2))

            res = minimize(nll, np.concatenate([A[tau], [d_vec[tau]]]), method="L-BFGS-B", options={"maxiter": 50})
            A[tau], d_vec[tau] = res.x[:-1], res.x[-1]

        ll = 0.0
        for t in range(N_warm):
            for tau in range(M):
                if obs_mask[t, tau]:
                    p = sigmoid(A[tau] @ s_hist[t] - d_vec[tau])
                    ll += R_filled[t, tau] * np.log(p + 1e-15) + (1 - R_filled[t, tau]) * np.log(1 - p + 1e-15)
        ll_history.append(ll)
        if verbose:
            print(f"  EM iter {it}: LL = {ll:.4f}")
        if abs(ll - prev_ll) < tol:
            if verbose:
                print(f"  Converged at iter {it}")
            break
        prev_ll = ll

    return sigmoid(s_hist), A, ll, ll_history


# ── MIRT Online Bayesian Update ─────────────────────────────────────────────

def online_profile_update(s_t, a_tau, d_tau, success, sigma_s=1.0):
    """One-step MIRT Bayesian profile update (proposal §3.1.4)."""
    K = len(s_t)
    s = s_t.copy()
    for _ in range(5):
        theta = a_tau @ s - d_tau
        p = sigmoid(theta)
        grad = (float(success) - p) * a_tau - (1.0 / sigma_s ** 2) * (s - s_t)
        W = p * (1 - p)
        hess = -W * np.outer(a_tau, a_tau) - (1.0 / sigma_s ** 2) * np.eye(K)
        s -= np.linalg.solve(hess, grad)
    return np.clip(s, 0.0, 1.0)


# ── SPG-Bandit Selector ────────────────────────────────────────────────────

TASK_TYPES = [
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
]


class SPGBanditSelector(BaseSelector):
    """SPG-Bandit: gap-weighted task selection. Profile maintained internally."""

    def __init__(self, task_pool: TaskPool, n_warm: int = 30,
                 alpha: float = 0.1, tau: float = 0.1,
                 d_f: int = 16, d_h: int = 32,
                 lambda_reg: float = 1.0, seed: int = 42,
                 K: int = 6):
        self._K = K
        self._n_warm = n_warm
        self._alpha = alpha
        self._tau = tau
        self._lambda = lambda_reg
        self._seed = seed
        self._d_f, self._d_h = d_f, d_h
        self._step = 0
        self._warmup_ready = False
        self._mlp: MLPFeaturizer | None = None
        self._A = self._lambda * np.eye(d_f)
        self._B = np.zeros((d_f, K))
        self._W = np.zeros((d_f, K))
        self._last_phi = None

        # Internal profile (SPG own concept)
        self._profile = np.zeros(K)

        # Metrics for logging
        self._metrics = {}

        # Warmup data
        self._warmup_task_ids = []
        self._warmup_successes = []
        self._warmup_deltas = []
        self._warmup_embeds = []

        # MIRT fitted params for online update
        self._A_fit = None
        self._d_fit = None

    @property
    def needs_warmup(self):
        return True

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def save_warmup_data(self, path: str):
        """Save warmup data to JSON for future --warmup-data runs."""
        data = {
            "task_ids": self._warmup_task_ids,
            "successes": self._warmup_successes,
            "deltas": [d.tolist() for d in self._warmup_deltas],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_warmup_data(self, path: str, task_pool: TaskPool):
        """Load warmup data, skip task execution, run MIRT EM + MLP."""
        with open(path) as f:
            data = json.load(f)
        self._warmup_task_ids = data["task_ids"]
        self._warmup_successes = data["successes"]
        self._warmup_deltas = [np.array(d) for d in data["deltas"]]
        self._n_warm = len(self._warmup_task_ids)
        self._finalize_warmup(task_pool)
        self._step = self._n_warm

    def select(self, task_pool: TaskPool) -> int:
        if self._step < self._n_warm:
            tid = self._step % task_pool.M
            self._last_phi = None
            self._step += 1
            return tid

        if not self._warmup_ready:
            self._finalize_warmup(task_pool)

        g = self._compute_gap(self._profile)

        # Log profile to wandb
        wb = {f"profile/dim_{i}": float(self._profile[i]) for i in range(len(self._profile))}
        wb["profile/mean"] = float(np.mean(self._profile))
        wb["profile/min"] = float(np.min(self._profile))
        wb["profile/max"] = float(np.max(self._profile))
        wb["evolving/step"] = self._step - self._n_warm + 1
        log_metrics(wb)

        A_inv = np.linalg.inv(self._A)
        best_score, best_tid = -np.inf, 0
        for tau in range(task_pool.M):
            phi = self._mlp.forward(task_pool.get_embedding(tau))
            delta_hat = self._W.T @ phi
            ucb = self._alpha * np.sqrt(max(phi @ A_inv @ phi, 1e-10))
            score = g @ delta_hat + ucb
            if score > best_score:
                best_score, best_tid, self._last_phi = score, tau, phi

        self._step += 1
        return best_tid

    def update(self, task_id: int, result: dict):
        success = result["success"]

        if self._step <= self._n_warm and not self._warmup_ready:
            self._warmup_task_ids.append(task_id)
            self._warmup_successes.append(success)
            self._warmup_deltas.append(result.get("delta", np.zeros(self._K)))
        elif self._warmup_ready:
            # MIRT Bayesian online update
            if self._A_fit is not None and task_id < len(self._A_fit):
                a_tau = self._A_fit[task_id]
                d_tau = self._d_fit[task_id] if task_id < len(self._d_fit) else 0.0
                self._profile = online_profile_update(self._profile, a_tau, d_tau, success)
            else:
                # Fallback heuristic
                dim = TASK_TYPES.index(
                    task_pool.metadata[task_id]["task_type"]) if task_id < len(task_pool.metadata) else 0
                self._profile[dim] += 0.05 if success else -0.01
                self._profile = np.clip(self._profile, 0.0, 1.0)

            # Ridge regression update
            if self._last_phi is not None:
                delta = result.get("delta", np.zeros(self._K))
                self._A += np.outer(self._last_phi, self._last_phi)
                self._B += np.outer(self._last_phi, delta)
                self._W = np.linalg.solve(self._A, self._B)

    def _compute_gap(self, profile):
        raw = (1.0 - profile) / max(self._tau, 1e-10)
        exp = np.exp(raw - np.max(raw))
        return exp / np.sum(exp)

    def _finalize_warmup(self, task_pool: TaskPool):
        print(f"\n  [SPG] Finalizing warmup ({self._n_warm} tasks)...")
        for tid in self._warmup_task_ids:
            self._warmup_embeds.append(task_pool.get_embedding(tid))

        N = len(self._warmup_task_ids)
        R = np.full((N, task_pool.M), np.nan)
        for t, tid in enumerate(self._warmup_task_ids):
            R[t, tid] = float(self._warmup_successes[t])

        # Sequential MIRT EM: run EM with cumulative data to compute per-step deltas
        profile = np.zeros(self._K)
        deltas = []
        for t in range(N):
            s_hist_t, *_ = fit_mirt_em(R[:t + 1], self._K, verbose=False)
            new_profile = s_hist_t[-1]  # sigmoid-transformed ability, (K,)
            deltas.append(new_profile - profile)
            profile = new_profile

        # Final EM on all N (verbose, for logging + item params)
        s_hist, self._A_fit, ll, ll_history = fit_mirt_em(R, self._K, verbose=True)
        self._profile = s_hist[-1].copy()
        self._d_fit = np.zeros(task_pool.M)
        self._metrics["mirt_ll_history"] = [round(v, 4) for v in ll_history]
        for i, ll_val in enumerate(ll_history):
            log_metrics({"mirt/ll": ll_val, "mirt/step": i})

        # MLP training with proper sequential deltas
        self._warmup_deltas = deltas
        self._mlp = MLPFeaturizer(task_pool.d_c, self._d_h, self._d_f, self._seed)
        loss_hist = self._mlp.train(np.array(self._warmup_embeds), np.array(self._warmup_deltas), 50, wandb_prefix="spg")
        self._metrics["mlp_loss_history"] = [round(v, 6) for v in loss_hist]
        print(f"  [SPG] MLP final MSE: {loss_hist[-1]:.6f}")

        self._A = self._lambda * np.eye(self._d_f)
        self._B = np.zeros((self._d_f, self._K))
        self._W = np.zeros((self._d_f, self._K))
        self._warmup_ready = True

    def reset(self):
        self._step = 0
        self._warmup_ready = False
        self._mlp = None
        self._profile = np.zeros(self._K)
        self._A = self._lambda * np.eye(self._d_f)
        self._B = np.zeros((self._d_f, self._K))
        self._W = np.zeros((self._d_f, self._K))
        self._last_phi = None
        self._warmup_task_ids.clear()
        self._warmup_successes.clear()
        self._warmup_deltas.clear()
        self._warmup_embeds.clear()
        self._A_fit = None
        self._d_fit = None
