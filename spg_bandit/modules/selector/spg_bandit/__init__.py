"""SPG-Bandit selector: gap-weighted multi-output bandit.

Maintains its own skill profile internally, independent of skill evolving method.
"""

import json
from collections import deque
import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from spg_bandit.modules.dataset.base import TaskPool
from spg_bandit.modules.skill_evolving.base import SelectionContext
from spg_bandit.utils.wandb import log_metrics
from spg_bandit.modules.selector.base import BaseSelector


def _resolve_torch_device(requested: str = "auto", min_free_memory_mb: int = 0) -> str:
    """Resolve warmup acceleration, safely falling back for ``device=auto``.

    The actor/vLLM owns the GPU budget.  SPG's MLP and ridge regression are
    optional accelerators, so automatic mode only uses CUDA when its currently
    *free* memory clears a configurable safety floor.  An explicit ``cuda``
    request remains an opt-in override for users who want strict placement.
    """
    if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError("spg_bandit.device must be auto, cpu, cuda, or cuda:<index>")
    if min_free_memory_mb < 0:
        raise ValueError("spg_bandit.gpu_min_free_memory_mb must be non-negative")
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 ** 2)
        if free_mb < min_free_memory_mb:
            print(
                "  [SPG] CUDA has only "
                f"{free_mb:.0f}/{total_bytes / (1024 ** 2):.0f} MiB free; "
                f"need {min_free_memory_mb} MiB for auto mode, using CPU",
                flush=True,
            )
            return "cpu"
        print(
            "  [SPG] CUDA auto mode enabled "
            f"({free_mb:.0f}/{total_bytes / (1024 ** 2):.0f} MiB free)",
            flush=True,
        )
        return "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("  [SPG] CUDA requested but unavailable; falling back to CPU", flush=True)
        return "cpu"
    return requested


def _ridge_predict(X_seen, y_seen, X_all, alpha: float, device: str) -> np.ndarray:
    """Multi-output ridge with an optional CUDA implementation.

    sklearn's default Ridge fits an intercept.  The torch path implements the
    same centered closed-form solution and returns predictions for ``X_all``.
    """
    if device != "cpu":
        try:
            import torch
            dtype = torch.float32
            x = torch.as_tensor(X_seen, dtype=dtype, device=device)
            y = torch.as_tensor(y_seen, dtype=dtype, device=device)
            query = torch.as_tensor(X_all, dtype=dtype, device=device)
            x_mean, y_mean = x.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)
            xc, yc = x - x_mean, y - y_mean
            gram = xc.T @ xc + float(alpha) * torch.eye(xc.shape[1], dtype=dtype, device=device)
            coef = torch.linalg.solve(gram, xc.T @ yc)
            prediction = (query - x_mean) @ coef + y_mean
            return prediction.detach().cpu().numpy()
        except Exception as exc:
            print(f"  [SPG] GPU ridge failed ({type(exc).__name__}); using sklearn CPU", flush=True)
    reg = Ridge(alpha=alpha)
    reg.fit(X_seen, y_seen)
    return reg.predict(X_all)


# ── MLP Featurizer ──────────────────────────────────────────────────────────

class MLPFeaturizer:
    """Two-layer MLP: R^d_c → R^d_h → R^d_f."""

    def __init__(self, d_c: int, d_h: int = 32, d_f: int = 16, seed: int = 42,
                 device: str = "cpu"):
        self.d_c, self.d_h, self.d_f = d_c, d_h, d_f
        self.lr = 1e-3
        self.device = _resolve_torch_device(device)
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

    def get_state(self) -> dict:
        return {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "head_W": self._head_W.tolist() if self._head_W is not None else None,
            "head_b": self._head_b.tolist() if self._head_b is not None else None,
        }

    def set_state(self, state: dict):
        import numpy as np
        self.W1 = np.array(state["W1"])
        self.b1 = np.array(state["b1"])
        self.W2 = np.array(state["W2"])
        self.b2 = np.array(state["b2"])
        self._head_W = np.array(state["head_W"]) if state.get("head_W") is not None else None
        self._head_b = np.array(state["head_b"]) if state.get("head_b") is not None else None

    def train(self, X, y, epochs=50, batch_size=32, wandb_prefix="mlp"):
        """Train on CUDA when available, retaining NumPy as a dependency-free fallback."""
        if self.device != "cpu":
            try:
                return self._train_torch(X, y, epochs, batch_size, wandb_prefix)
            except Exception as exc:
                print(f"  [SPG] GPU MLP failed ({type(exc).__name__}); using NumPy CPU", flush=True)
        return self._train_numpy(X, y, epochs, batch_size, wandb_prefix)

    def _train_torch(self, X, y, epochs, batch_size, wandb_prefix):
        import torch
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        N, K = x.shape[0], target.shape[1]
        if self._head_W is None:
            bound = np.sqrt(6.0 / (self.d_f + K))
            self._head_W = self.rng.uniform(-bound, bound, (self.d_f, K))
            self._head_b = np.zeros(K)
        params = [
            torch.tensor(self.W1, dtype=torch.float32, device=self.device, requires_grad=True),
            torch.tensor(self.b1, dtype=torch.float32, device=self.device, requires_grad=True),
            torch.tensor(self.W2, dtype=torch.float32, device=self.device, requires_grad=True),
            torch.tensor(self.b2, dtype=torch.float32, device=self.device, requires_grad=True),
            torch.tensor(self._head_W, dtype=torch.float32, device=self.device, requires_grad=True),
            torch.tensor(self._head_b, dtype=torch.float32, device=self.device, requires_grad=True),
        ]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_history = []
        for epoch in range(epochs):
            permutation = torch.randperm(N, device=self.device)
            total_loss, examples = 0.0, 0
            for start in range(0, N, batch_size):
                idx = permutation[start:min(start + batch_size, N)]
                xb, yb = x[idx], target[idx]
                phi = torch.relu(xb @ params[0] + params[1]) @ params[2] + params[3]
                prediction = phi @ params[4] + params[5]
                loss = torch.mean((prediction - yb) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(idx)
                examples += len(idx)
            avg_loss = total_loss / max(examples, 1)
            loss_history.append(avg_loss)
            if epoch % 10 == 9:
                print(f"  MLP epoch {epoch + 1}: MSE = {avg_loss:.6f}", flush=True)
            log_metrics({f"{wandb_prefix}/mse": avg_loss, "_step_spg": epoch})
        self.W1, self.b1, self.W2, self.b2, self._head_W, self._head_b = [
            value.detach().cpu().numpy() for value in params
        ]
        return loss_history

    def _train_numpy(self, X, y, epochs=50, batch_size=32, wandb_prefix="mlp"):
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
            log_metrics({f"{wandb_prefix}/mse": avg_loss, "_step_spg": epoch})
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

def fit_mirt_em(R, K, trials=None, max_iter=200, tol=1e-4, verbose=False, seed=None):
    """Fit MIRT to grouped Bernoulli observations.

    ``R[t, tau]`` is the number of successful rollouts selected at round t;
    ``trials[t, tau]`` is the corresponding rollout count.  With ``trials``
    omitted this is the original single-Bernoulli formulation.  Grouping is
    important: repeated rollouts of one selected task share one pre-update
    skill profile instead of being treated as sequential profile states.
    """
    N_warm, M = R.shape
    obs_mask = ~np.isnan(R)
    R_filled = np.nan_to_num(R, nan=0.0)
    if trials is None:
        trials_filled = obs_mask.astype(float)
    else:
        if trials.shape != R.shape:
            raise ValueError("trials must have the same shape as R")
        trials_filled = np.nan_to_num(trials, nan=0.0)
        obs_mask &= trials_filled > 0
    s_hist = np.full((N_warm, K), 0.5)
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.5, 1.5, (M, K))
    d_vec = np.full(M, 0.5)
    prev_ll = -np.inf
    ll_history = []

    for it in range(max_iter):
        # E-step: MAP estimate of s_t ∈ [0,1]^K
        for t in range(N_warm):
            obs_t = np.where(obs_mask[t])[0]
            if len(obs_t) == 0:
                continue
            s = s_hist[t].copy()
            for _ in range(20):
                theta = A[obs_t] @ s - d_vec[obs_t]
                p = sigmoid(theta)
                n = trials_filled[t, obs_t]
                dif = R_filled[t, obs_t] - n * p
                grad = A[obs_t].T @ dif - 1.0 * (s - 0.5)
                Wd = n * p * (1 - p)
                hess = -A[obs_t].T @ (A[obs_t] * Wd[:, np.newaxis]) - np.eye(K)
                s -= 0.5 * np.linalg.solve(hess, grad)
            s_hist[t] = np.clip(s, 0.0, 1.0)

        # M-step: non-negative discrimination and difficulty keep the MIRT
        # parameters semantically interpretable for the [0, 1] skill profile.
        for tau in range(M):
            t_idx = np.where(obs_mask[:, tau])[0]
            if len(t_idx) == 0:
                continue
            X, y = s_hist[t_idx], R_filled[t_idx, tau]
            n = trials_filled[t_idx, tau]

            def nll(params):
                a, b = params[:-1], params[-1]
                p = sigmoid(X @ a - b)
                ll = y @ np.log(p + 1e-15) + (n - y) @ np.log(1 - p + 1e-15)
                return -(ll - 0.01 * np.sum(a ** 2))

            res = minimize(
                nll, np.concatenate([A[tau], [d_vec[tau]]]),
                method="L-BFGS-B",
                bounds=[(0.0, None)] * (K + 1),
                options={"maxiter": 50},
            )
            A[tau], d_vec[tau] = res.x[:-1], res.x[-1]

        # Log-likelihood
        ll = 0.0
        for t in range(N_warm):
            for tau in range(M):
                if obs_mask[t, tau]:
                    p = sigmoid(A[tau] @ s_hist[t] - d_vec[tau])
                    n = trials_filled[t, tau]
                    ll += R_filled[t, tau] * np.log(p + 1e-15) + (n - R_filled[t, tau]) * np.log(1 - p + 1e-15)
        ll_history.append(ll)
        if verbose:
            print(f"  EM iter {it}: LL = {ll:.4f}")
        if abs(ll - prev_ll) < tol:
            if verbose:
                print(f"  Converged at iter {it}")
            break
        prev_ll = ll

    return s_hist, A, d_vec, ll, ll_history


# ── MIRT Online Bayesian Update ─────────────────────────────────────────────

def online_profile_update(s_t, a_tau, d_tau, successes, trials=1, sigma_s=1.0):
    """Grouped MIRT Bayesian profile update (proposal §3.1.4)."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials] and trials must be positive")
    K = len(s_t)
    s = s_t.copy()
    for _ in range(5):
        theta = a_tau @ s - d_tau
        p = sigmoid(theta)
        grad = (float(successes) - trials * p) * a_tau - (1.0 / sigma_s ** 2) * (s - s_t)
        W = trials * p * (1 - p)
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
                 K: int = 6, warmup_ids: list[int] | None = None,
                 window_size: int = 20, device: str = "auto",
                 task_state_dim: int = 0, gpu_min_free_memory_mb: int = 2048,
                 gain_measurement: str = "mirt_transition"):
        self._K = K
        self._n_warm = n_warm
        self._alpha = alpha
        self._tau = tau
        self._lambda = lambda_reg
        self._seed = seed
        self._d_f, self._d_h = d_f, d_h
        self._requested_device = device
        self._gpu_min_free_memory_mb = int(gpu_min_free_memory_mb)
        self._device = _resolve_torch_device(
            device, self._gpu_min_free_memory_mb,
        )
        self._task_state_dim = int(task_state_dim)
        if self._task_state_dim < 0:
            raise ValueError("task_state_dim must be non-negative")
        self._selection_context_provider = None
        self._last_task_state = np.zeros(self._task_state_dim)
        self._last_selection_diagnostics: dict = {}
        self._legacy_state_in_mlp = False
        self._window_size = window_size
        if self._window_size <= 0:
            raise ValueError("window_size must be positive")
        self._gain_measurement = str(gain_measurement)
        if self._gain_measurement not in {"mirt_transition", "probe"}:
            raise ValueError(
                "spg_bandit.gain_measurement must be mirt_transition or probe"
            )
        self._step = 0
        self._warmup_ready = False
        self._task_pool = task_pool
        self._warmup_ids = list(warmup_ids) if warmup_ids else list(range(task_pool.M))
        self._mlp: MLPFeaturizer | None = None
        bandit_dim = d_f + self._task_state_dim
        self._A = self._lambda * np.eye(bandit_dim)
        self._B = np.zeros((bandit_dim, K))
        self._W = np.zeros((bandit_dim, K))
        self._last_phi = None
        # Probe attribution is retained as an explicit ablation only.  The
        # proposal/default path supervises the MLP with the selected task's
        # immediate MIRT profile transition.
        self._pending_skill_observations = deque()
        self._window = deque()

        # Internal profile (SPG own concept)
        self._profile = np.zeros(K)

        # Metrics for logging
        self._metrics = {}

        # Warmup data
        self._warmup_task_ids = []
        self._warmup_successes = []
        self._warmup_trials = []
        self._warmup_outcomes = []
        self._warmup_deltas = []
        self._warmup_embeds = []
        # Raw before/after probe observations exist only in probe-ablation
        # mode and are replayed once MIRT is identifiable.
        self._warmup_gain_measurements = []

        # MIRT fitted params for online update
        self._A_fit = None
        self._d_fit = None

    @property
    def needs_warmup(self):
        return True

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def get_profile(self) -> np.ndarray:
        """Return a copy of the current MIRT ability profile."""
        return self._profile.copy()

    def set_selection_context_provider(self, provider, feature_dim: int):
        """Attach the evolving method's context adapter at the selector seam."""
        feature_dim = int(feature_dim)
        if feature_dim != self._task_state_dim:
            if self._mlp is not None:
                raise ValueError("Cannot change task-state feature size after warmup")
            self._task_state_dim = feature_dim
            self._last_task_state = np.zeros(feature_dim)
            bandit_dim = self._d_f + feature_dim
            self._A = self._lambda * np.eye(bandit_dim)
            self._B = np.zeros((bandit_dim, self._K))
            self._W = np.zeros((bandit_dim, self._K))
        self._selection_context_provider = provider

    def set_task_state_provider(self, provider, feature_dim: int):
        """Compatibility adapter for legacy providers that return arrays."""
        def context_provider(task_id: int):
            return SelectionContext(features=np.asarray(provider(task_id), dtype=float))

        self.set_selection_context_provider(context_provider, feature_dim)

    def _selection_context(self, task_id: int) -> SelectionContext:
        if self._selection_context_provider is None:
            context = SelectionContext(features=np.zeros(self._task_state_dim))
        else:
            context = self._selection_context_provider(int(task_id))
            if not isinstance(context, SelectionContext):
                context = SelectionContext(features=np.asarray(context, dtype=float))
        if context.features.shape != (self._task_state_dim,):
            raise ValueError(
                "Selection context returned feature shape "
                f"{context.features.shape}; expected ({self._task_state_dim},)"
            )
        return context

    def _task_state(self, task_id: int) -> np.ndarray:
        return self._selection_context(task_id).features

    def _model_input(self, task_id: int, profile: np.ndarray, task_state: np.ndarray | None = None):
        values = [self._task_pool.get_embedding(task_id), profile]
        if self._legacy_state_in_mlp:
            state = self._task_state(task_id) if task_state is None else task_state
            values.append(state)
        return np.concatenate(values)

    def _bandit_features(
        self, task_id: int, profile: np.ndarray, task_state: np.ndarray,
    ) -> np.ndarray:
        semantic = self._mlp.forward(self._model_input(task_id, profile, task_state))
        if self._legacy_state_in_mlp or self._task_state_dim == 0:
            return semantic
        return np.concatenate([semantic, task_state])

    def get_last_selection_diagnostics(self) -> dict:
        return dict(self._last_selection_diagnostics)

    def save_warmup_data(self, path: str):
        """Save warmup data to JSON for future --warmup-data runs."""
        data = {
            "gain_measurement": self._gain_measurement,
            "task_ids": self._warmup_task_ids,
            "successes": self._warmup_successes,
            "trials": self._warmup_trials,
            "outcomes": self._warmup_outcomes,
            "deltas": [d.tolist() for d in self._warmup_deltas],
            "gain_measurements": self._warmup_gain_measurements,
            "task_states": [state.tolist() for state in self._warmup_embeds],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_warmup_data(self, path: str, task_pool: TaskPool | None = None):
        """Load warmup data, skip task execution, run MIRT EM + MLP."""
        with open(path) as f:
            data = json.load(f)
        data_measurement = data.get("gain_measurement")
        if data_measurement is None:
            data_measurement = (
                "probe" if data.get("gain_measurements") else "mirt_transition"
            )
        if data_measurement != self._gain_measurement:
            raise ValueError(
                "Warmup data gain_measurement does not match current SPG config "
                f"({data_measurement!r} != {self._gain_measurement!r})"
            )
        self._warmup_task_ids = data["task_ids"]
        self._warmup_successes = data["successes"]
        self._warmup_trials = data.get("trials", [1] * len(self._warmup_task_ids))
        self._warmup_outcomes = data.get(
            "outcomes",
            [[bool(success)] for success in self._warmup_successes],
        )
        self._warmup_deltas = [np.array(d) for d in data["deltas"]]
        self._warmup_gain_measurements = list(data.get("gain_measurements", []))
        self._warmup_embeds = [
            np.asarray(state, dtype=float)
            for state in data.get("task_states", [np.zeros(self._task_state_dim)] * len(self._warmup_task_ids))
        ]
        self._n_warm = len(self._warmup_task_ids)
        self._finalize_warmup()
        self._step = self._n_warm

    def select(self, task_pool: TaskPool) -> int:
        if self._step < self._n_warm:
            tid = self._warmup_ids[self._step % len(self._warmup_ids)]
            self._last_phi = None
            context = self._selection_context(tid)
            self._last_task_state = context.features.copy()
            self._last_selection_diagnostics = {
                "phase": "warmup",
                "task_id": int(tid),
                "eligible": bool(context.eligible),
                "eligibility_reason": context.reason,
                "policy_version": context.policy_version,
                "task_state": context.features.tolist(),
            }
            self._step += 1
            return tid

        if not self._warmup_ready:
            self._finalize_warmup()

        g = self._compute_gap(self._profile)

        # Log profile to wandb
        wb = {f"profile/dim_{i}": float(self._profile[i]) for i in range(len(self._profile))}
        wb["_step_evolving"] = self._step - self._n_warm + 1
        log_metrics(wb)

        A_inv = np.linalg.inv(self._A)
        contexts = [self._selection_context(tau) for tau in range(task_pool.M)]
        eligible_count = sum(bool(context.eligible) for context in contexts)
        fallback_all_ineligible = eligible_count == 0
        best_score, best_tid = -np.inf, 0
        ranked: list[dict] = []
        for tau, context in enumerate(contexts):
            if not fallback_all_ineligible and not context.eligible:
                continue
            task_state = context.features
            phi = self._bandit_features(tau, self._profile, task_state)
            delta_hat = self._W.T @ phi
            ucb = self._alpha * np.linalg.norm(g) * np.sqrt(max(phi @ A_inv @ phi, 1e-10))
            predicted_gain = float(g @ delta_hat)
            score = predicted_gain + ucb
            ranked.append({
                "task_id": int(tau),
                "score": float(score),
                "predicted_gain": predicted_gain,
                "ucb": float(ucb),
                "predicted_delta": delta_hat.tolist(),
                "eligible": bool(context.eligible),
                "eligibility_reason": context.reason,
                "policy_version": context.policy_version,
                "task_state": task_state.tolist(),
            })
            if score > best_score:
                best_score, best_tid, self._last_phi = score, tau, phi
                self._last_task_state = task_state.copy()

        ranked.sort(key=lambda item: (-item["score"], item["task_id"]))
        selected_context = contexts[best_tid]
        self._last_selection_diagnostics = {
            "phase": "bandit",
            "task_id": int(best_tid),
            "score": float(best_score),
            "eligible_count": int(eligible_count),
            "pool_size": int(task_pool.M),
            "fallback_all_ineligible": fallback_all_ineligible,
            "eligibility_reason": selected_context.reason,
            "policy_version": selected_context.policy_version,
            "top_candidates": ranked[:5],
        }
        self._step += 1
        return best_tid

    def update(self, task_id: int, result: dict):
        outcomes = result.get("rollout_successes")
        if outcomes is None:
            outcomes = [bool(result["success"])]
        outcomes = [bool(outcome) for outcome in outcomes]
        if not outcomes:
            raise ValueError("result.rollout_successes must not be empty")
        successes, trials = sum(outcomes), len(outcomes)

        if self._step <= self._n_warm and not self._warmup_ready:
            self._warmup_task_ids.append(task_id)
            self._warmup_successes.append(successes)
            self._warmup_trials.append(trials)
            self._warmup_outcomes.append(outcomes)
            self._warmup_deltas.append(result.get("delta", np.zeros(self._K)))
            self._warmup_embeds.append(self._last_task_state.copy())
        elif self._warmup_ready:
            a_tau = self._A_fit[task_id]
            d_tau = self._d_fit[task_id]
            profile_before = self._profile.copy()
            self._profile = online_profile_update(
                self._profile, a_tau, d_tau, successes, trials,
            )
            if self._last_phi is not None:
                if self._gain_measurement == "probe":
                    self._pending_skill_observations.append((task_id, self._last_phi.copy()))
                else:
                    delta = self._profile - profile_before
                    self._append_window_observation(self._last_phi, delta)
                    self._W = np.linalg.solve(self._A, self._B)

    @staticmethod
    def _compact_probe_results(results: list[dict]) -> list[dict]:
        """Store only grouped success observations, never large trajectories."""
        compact = []
        for item in results:
            outcomes = item.get("rollout_successes")
            if outcomes is None:
                outcomes = [bool(item.get("success", False))]
            compact.append({
                "task_id": int(item["task_id"]),
                "rollout_successes": [bool(value) for value in outcomes],
            })
        return compact

    def record_warmup_skill_measurement(self, task_id: int, before_results: list[dict],
                                        after_results: list[dict] | None, *, updated: bool):
        """Record an ExpeL update's raw probe evidence during warmup.

        The warmup schedule is fixed, so no MIRT estimate is needed yet.  At
        warmup end the final item parameters are used to reconstruct these
        causal labels on one consistent latent scale.
        """
        if (
            self._gain_measurement != "probe"
            or self._warmup_ready
            or not self._warmup_task_ids
        ):
            return
        self._warmup_gain_measurements.append({
            "task_id": int(task_id),
            "warmup_index": len(self._warmup_task_ids) - 1,
            "updated": bool(updated),
            "before": self._compact_probe_results(before_results),
            "after": self._compact_probe_results(after_results or []),
        })

    def _profile_from_probe_results(self, results: list[dict], base_profile: np.ndarray) -> np.ndarray:
        profile = np.asarray(base_profile, dtype=float).copy()
        for item in results:
            outcomes = [bool(value) for value in item.get("rollout_successes", [])]
            if not outcomes:
                continue
            task_id = int(item["task_id"])
            profile = online_profile_update(
                profile, self._A_fit[task_id], self._d_fit[task_id],
                sum(outcomes), len(outcomes),
            )
        return profile

    def discard_latest_pending_observation(self):
        """Drop an unlabelled immediate-attribution selection safely."""
        if self._pending_skill_observations:
            self._pending_skill_observations.pop()

    def estimate_profile_from_results(self, results: list[dict], *, base_profile=None):
        """Estimate ability from probe outcomes without mutating selector state."""
        if not self._warmup_ready:
            raise RuntimeError("probe profiles require completed SPG warmup")
        profile = np.array(
            self._profile if base_profile is None else base_profile, dtype=float,
        ).copy()
        for item in results:
            task_id = int(item["task_id"])
            outcomes = item.get("rollout_successes")
            if outcomes is None:
                outcomes = [bool(item["success"])]
            outcomes = [bool(value) for value in outcomes]
            if not outcomes:
                raise ValueError("probe result.rollout_successes must not be empty")
            profile = online_profile_update(
                profile, self._A_fit[task_id], self._d_fit[task_id],
                sum(outcomes), len(outcomes),
            )
        return profile

    def commit_skill_update(self, profile_before, profile_after, *, updated: bool):
        """Assign a post-reflection probe gain to tasks since the last update.

        The MLP target remains an ability-profile delta.  Unlike the old
        immediate outcome label, this delta is measured after the skill bank
        has actually changed, so it estimates the selected batch's effect on
        skill-mediated competence.
        """
        if self._gain_measurement != "probe" or not self._warmup_ready:
            return {"committed": 0, "delta": [0.0] * self._K}
        delta = np.asarray(profile_after, dtype=float) - np.asarray(profile_before, dtype=float)
        if not updated:
            delta = np.zeros(self._K)
        count = len(self._pending_skill_observations)
        while self._pending_skill_observations:
            _task_id, phi = self._pending_skill_observations.popleft()
            self._append_window_observation(phi, delta)
        if count:
            self._W = np.linalg.solve(self._A, self._B)
        # Post-update probes are the freshest ability evidence available.
        if updated:
            self._profile = np.asarray(profile_after, dtype=float).copy()
        return {"committed": count, "delta": delta.tolist()}

    def _compute_gap(self, profile):
        raw = (1.0 - profile) / max(self._tau, 1e-10)
        exp = np.exp(raw - np.max(raw))
        return exp / np.sum(exp)

    def _finalize_warmup(self):
        print(f"\n  [SPG] Finalizing warmup ({self._n_warm} tasks)...")

        # Re-check just before the only GPU-heavy work.  vLLM may have
        # allocated more KV cache after selector construction.
        self._device = _resolve_torch_device(
            self._requested_device, self._gpu_min_free_memory_mb,
        )

        N = len(self._warmup_task_ids)
        if N == 0:
            raise ValueError("Cannot finalize SPG-Bandit warmup without observations")
        # Compatibility with warmup JSON/checkpoints written before grouped
        # rollout support, and with callers that populate legacy fields.
        if len(self._warmup_trials) != N:
            self._warmup_trials = [1] * N
        if len(self._warmup_outcomes) != N:
            self._warmup_outcomes = [
                [bool(success)] for success in self._warmup_successes
            ]
        if self._window_size > N:
            raise ValueError("window_size cannot exceed the number of warmup observations")
        R = np.full((N, self._task_pool.M), np.nan)
        trials = np.full((N, self._task_pool.M), np.nan)
        for t, tid in enumerate(self._warmup_task_ids):
            R[t, tid] = float(self._warmup_successes[t])
            trials[t, tid] = float(self._warmup_trials[t])

        # Sequential MIRT EM: run EM with cumulative data to compute per-step deltas
        profiles_before = []
        profiles_after_task = []
        profile = np.zeros(self._K)
        deltas = []
        for t in range(N):
            s_hist_t, *_ = fit_mirt_em(
                R[:t + 1], self._K, trials=trials[:t + 1],
                verbose=False, seed=self._seed,
            )
            new_profile = s_hist_t[-1]
            profiles_before.append(profile.copy())
            deltas.append(new_profile - profile)
            profile = new_profile
            profiles_after_task.append(profile.copy())

        # Final EM on all N (verbose, for logging + item params)
        s_hist, self._A_fit, self._d_fit, ll, ll_history = fit_mirt_em(
            R, self._K, trials=trials, verbose=True, seed=self._seed,
        )
        self._profile = s_hist[-1].copy()
        self._metrics["mirt_ll_history"] = [round(v, 4) for v in ll_history]

        # Embedding → (a, d) predictor: infer parameters for unseen tasks
        X_seen = np.array([self._task_pool.get_embedding(tid) for tid in self._warmup_task_ids])
        warmup_ids = np.asarray(self._warmup_task_ids)
        y_seen = np.column_stack([
            self._A_fit[warmup_ids], self._d_fit[warmup_ids].reshape(-1, 1),
        ])
        y_pred_train = _ridge_predict(
            X_seen, y_seen, X_seen, self._lambda, self._device,
        )
        # Ridge is unconstrained, so project its extrapolated item parameters
        # back to the same non-negative MIRT domain used by EM.
        y_pred_train = np.maximum(y_pred_train, 0.0)
        pred_mse = float(np.mean((y_pred_train - y_seen) ** 2))
        log_metrics({"mirt/pred_mse": pred_mse, "_step_mirt": 0})
        X_all = self._task_pool.embeddings
        y_pred = np.maximum(
            _ridge_predict(X_seen, y_seen, X_all, self._lambda, self._device), 0.0,
        )
        self._A_fit = y_pred[:, :self._K]
        self._d_fit = y_pred[:, self._K]
        for i, ll_val in enumerate(ll_history):
            log_metrics({"mirt/ll": ll_val, "_step_mirt": i + 1})

        # Default/proposal labels are the sequential selected-task MIRT
        # transitions above. Probe replay remains an explicit ablation.
        train_ids = list(self._warmup_task_ids)
        train_profiles = list(profiles_before)
        train_deltas = list(deltas)
        train_task_states = list(self._warmup_embeds)
        if len(train_task_states) != len(train_ids):
            train_task_states = [np.zeros(self._task_state_dim) for _ in train_ids]
        last_measured_profile = None
        if self._gain_measurement == "probe" and self._warmup_gain_measurements:
            measured_ids, measured_profiles, measured_deltas, measured_task_states = [], [], [], []
            for measurement in self._warmup_gain_measurements:
                index = int(measurement.get("warmup_index", -1))
                if not 0 <= index < len(profiles_after_task):
                    continue
                anchor = profiles_after_task[index]
                before = self._profile_from_probe_results(measurement.get("before", []), anchor)
                if measurement.get("updated"):
                    after = self._profile_from_probe_results(measurement.get("after", []), anchor)
                    delta = after - before
                    last_measured_profile = after
                else:
                    delta = np.zeros(self._K)
                measured_ids.append(int(measurement["task_id"]))
                measured_profiles.append(before)
                measured_deltas.append(delta)
                measured_task_states.append(self._warmup_embeds[index].copy())
            if measured_ids:
                train_ids, train_profiles, train_deltas, train_task_states = (
                    measured_ids, measured_profiles, measured_deltas, measured_task_states,
                )
                self._metrics["warmup_causal_gain_labels"] = len(measured_ids)
        # MLP training learns semantic task/profile features only. Method state
        # is appended to the contextual ridge head below, because warmup sees
        # almost exclusively fresh-task states and cannot calibrate cumulative
        # retry features without extrapolating out of distribution.
        embeds = [self._task_pool.get_embedding(tid) for tid in train_ids]
        X = np.array([
            np.concatenate([e, p])
            for e, p in zip(embeds, train_profiles)
        ])
        y = np.array(train_deltas)
        self._warmup_deltas = train_deltas
        self._mlp = MLPFeaturizer(
            self._task_pool.d_c + self._K,
            self._d_h, self._d_f, self._seed,
            device=self._device,
        )
        loss_hist = self._mlp.train(X, y, 50, wandb_prefix="spg")
        self._metrics["mlp_loss_history"] = [round(v, 6) for v in loss_hist]
        print(f"  [SPG] MLP final MSE: {loss_hist[-1]:.6f}")

        # Algorithm 1 initializes the sliding-window ridge head with the final
        # window_size warmup tuples, rather than discarding calibration data.
        self._window.clear()
        bandit_dim = self._d_f + self._task_state_dim
        self._A = self._lambda * np.eye(bandit_dim)
        self._B = np.zeros((bandit_dim, self._K))
        for embedding, profile_before, task_state, delta in zip(
            embeds, train_profiles, train_task_states, train_deltas,
        ):
            semantic = self._mlp.forward(np.concatenate([embedding, profile_before]))
            phi = (
                semantic if self._task_state_dim == 0
                else np.concatenate([semantic, task_state])
            )
            self._append_window_observation(phi, delta)
        self._W = np.linalg.solve(self._A, self._B)
        if last_measured_profile is not None:
            self._profile = last_measured_profile.copy()
        self._warmup_ready = True

    def _append_window_observation(self, phi: np.ndarray, delta: np.ndarray):
        """Append one tuple and evict the oldest tuple beyond the sliding window."""
        if len(self._window) == self._window_size:
            old_phi, old_delta = self._window.popleft()
            self._A -= np.outer(old_phi, old_phi)
            self._B -= np.outer(old_phi, old_delta)
        self._window.append((phi.copy(), delta.copy()))
        self._A += np.outer(phi, phi)
        self._B += np.outer(phi, delta)

    def reset(self):
        self._step = 0
        self._warmup_ready = False
        self._mlp = None
        self._profile = np.zeros(self._K)
        bandit_dim = self._d_f + self._task_state_dim
        self._A = self._lambda * np.eye(bandit_dim)
        self._B = np.zeros((bandit_dim, self._K))
        self._W = np.zeros((bandit_dim, self._K))
        self._last_phi = None
        self._last_selection_diagnostics = {}
        self._legacy_state_in_mlp = False
        self._pending_skill_observations.clear()
        self._window.clear()
        self._warmup_task_ids.clear()
        self._warmup_successes.clear()
        self._warmup_trials.clear()
        self._warmup_outcomes.clear()
        self._warmup_deltas.clear()
        self._warmup_embeds.clear()
        self._warmup_gain_measurements.clear()
        self._A_fit = None
        self._d_fit = None

    def save_checkpoint(self) -> dict:
        return {
            "feature_layout_version": 2,
            "step": self._step, "n_warm": self._n_warm,
            "warmup_ready": self._warmup_ready,
            "profile": self._profile.tolist() if hasattr(self._profile, 'tolist') else self._profile,
            "A": self._A.tolist() if hasattr(self._A, 'tolist') else self._A,
            "B": self._B.tolist() if hasattr(self._B, 'tolist') else self._B,
            "W": self._W.tolist() if hasattr(self._W, 'tolist') else self._W,
            "A_fit": self._A_fit.tolist() if self._A_fit is not None else None,
            "d_fit": self._d_fit.tolist() if self._d_fit is not None else None,
            "task_ids": list(self._warmup_task_ids),
            "successes": list(self._warmup_successes),
            "trials": list(self._warmup_trials),
            "outcomes": list(self._warmup_outcomes),
            "gain_measurements": list(self._warmup_gain_measurements),
            "task_states": [state.tolist() for state in self._warmup_embeds],
            "task_state_dim": self._task_state_dim,
            "window": [
                {"phi": phi.tolist(), "delta": delta.tolist()}
                for phi, delta in self._window
            ],
            "pending_skill_observations": [
                {"task_id": task_id, "phi": phi.tolist()}
                for task_id, phi in self._pending_skill_observations
            ],
            "window_size": self._window_size,
            "gain_measurement": self._gain_measurement,
            "last_selection_diagnostics": self._last_selection_diagnostics,
            "mlp": self._mlp.get_state() if self._mlp is not None else None,
            "mlp_cfg": {"d_c": self._mlp.d_c, "d_h": self._d_h, "d_f": self._d_f, "seed": self._seed, "device": self._device} if self._mlp is not None else None,
        }

    def load_checkpoint(self, data: dict):
        self._step = data["step"]
        self._n_warm = data["n_warm"]
        self._warmup_ready = data["warmup_ready"]
        import numpy as np
        self._profile = np.array(data["profile"])
        self._A = np.array(data["A"])
        self._B = np.array(data["B"])
        self._W = np.array(data["W"])
        self._A_fit = (
            np.maximum(np.array(data["A_fit"]), 0.0)
            if data.get("A_fit") is not None else None
        )
        self._d_fit = (
            np.maximum(np.array(data["d_fit"]), 0.0)
            if data.get("d_fit") is not None else None
        )
        self._warmup_task_ids = list(data["task_ids"])
        self._warmup_successes = list(data["successes"])
        self._warmup_trials = list(data.get("trials", [1] * len(self._warmup_task_ids)))
        self._warmup_outcomes = list(data.get(
            "outcomes", [[bool(success)] for success in self._warmup_successes],
        ))
        self._warmup_gain_measurements = list(data.get("gain_measurements", []))
        self._task_state_dim = int(data.get("task_state_dim", self._task_state_dim))
        self._legacy_state_in_mlp = int(data.get("feature_layout_version", 1)) < 2
        self._last_selection_diagnostics = dict(
            data.get("last_selection_diagnostics", {})
        )
        self._warmup_embeds = [
            np.asarray(item, dtype=float)
            for item in data.get("task_states", [np.zeros(self._task_state_dim)] * len(self._warmup_task_ids))
        ]
        self._window_size = data.get("window_size", self._window_size)
        checkpoint_measurement = data.get("gain_measurement")
        if checkpoint_measurement is None:
            checkpoint_measurement = (
                "probe"
                if data.get("gain_measurements") or data.get("pending_skill_observations")
                else "mirt_transition"
            )
        if checkpoint_measurement != self._gain_measurement:
            raise ValueError(
                "Checkpoint gain_measurement does not match current SPG config "
                f"({checkpoint_measurement!r} != {self._gain_measurement!r})"
            )
        self._window = deque(
            (np.array(item["phi"]), np.array(item["delta"]))
            for item in data.get("window", [])
        )
        self._pending_skill_observations = deque(
            (int(item["task_id"]), np.array(item["phi"]))
            for item in data.get("pending_skill_observations", [])
        )
        if data.get("mlp") and self._mlp is None:
            mlp_cfg = data["mlp_cfg"]
            import numpy as np
            self._mlp = MLPFeaturizer(
                mlp_cfg["d_c"], mlp_cfg["d_h"], mlp_cfg["d_f"], mlp_cfg["seed"],
                device=mlp_cfg.get("device", self._device),
            )
        if data.get("mlp"):
            self._mlp.set_state(data["mlp"])
