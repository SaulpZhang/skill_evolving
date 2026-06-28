"""MIRT: Multidimensional Item Response Theory for skill profile estimation.

Implements:
- EM algorithm for offline fitting from warmup data
- Bayesian one-step Newton for online profile updates
- BIC-based K selection
"""

import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize


def fit_mirt_em(
    R: np.ndarray,
    K: int,
    max_iter: int = 200,
    tol: float = 1e-4,
    lr: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit MIRT model via EM algorithm.

    Args:
        R: (N_warm, M) binary matrix, R[t, tau] = success of task tau at time t.
           Unobserved entries should be NaN.
        K: Number of latent dimensions.
        max_iter: Maximum EM iterations.
        tol: Convergence threshold for log-likelihood.
        lr: Learning rate for M-step parameter updates (0 < lr <= 1).
        verbose: Print progress.

    Returns:
        A: (M, K) loading matrix — a_tau_k
        d: (M,) difficulty vector — d_tau
        s_hist: (N_warm, K) estimated skill profile at each time step
        log_likelihood: final log-likelihood
    """
    N_warm, M = R.shape
    obs_mask = ~np.isnan(R)
    R_filled = np.nan_to_num(R, nan=0.0)

    # Initialize parameters
    s_hist = np.random.randn(N_warm, K) * 0.1
    A = np.random.randn(M, K) * 0.1
    d = np.zeros(M)

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # --- E-step: estimate s_t given current A, d ---
        for t in range(N_warm):
            observed_tasks = np.where(obs_mask[t])[0]
            if len(observed_tasks) == 0:
                continue

            # Newton-Raphson for s_t
            s = s_hist[t].copy()
            for _ in range(20):
                theta = A[observed_tasks] @ s - d[observed_tasks]
                p = sigmoid(theta)
                grad = A[observed_tasks].T @ (R_filled[t, observed_tasks] - p) - s
                W = np.diag(p * (1 - p))
                hess = -A[observed_tasks].T @ W @ A[observed_tasks] - np.eye(K)
                s -= lr * np.linalg.solve(hess, grad)
            s_hist[t] = np.clip(s, -3.0, 3.0)  # keep in reasonable range

        # --- M-step: update A, d given s_hist ---
        for tau in range(M):
            time_steps = np.where(obs_mask[:, tau])[0]
            if len(time_steps) < 2:
                continue

            # Logistic regression: predict R[t, tau] from s_t
            X = s_hist[time_steps]  # (N_tau, K)
            y = R_filled[time_steps, tau]

            # L2-regularized logistic regression using scipy
            def neg_log_likelihood(params):
                a = params[:-1]
                b = params[-1]
                theta = X @ a - b
                p = sigmoid(theta)
                ll = y @ np.log(p + 1e-15) + (1 - y) @ np.log(1 - p + 1e-15)
                reg = 0.01 * np.sum(a**2)
                return -(ll - reg)

            result = minimize(
                neg_log_likelihood,
                x0=np.concatenate([A[tau], [d[tau]]]),
                method="L-BFGS-B",
                options={"maxiter": 50},
            )
            A[tau] = result.x[:-1]
            d[tau] = result.x[-1]

        # --- Compute log-likelihood ---
        ll = 0.0
        for t in range(N_warm):
            for tau in range(M):
                if obs_mask[t, tau]:
                    theta = A[tau] @ s_hist[t] - d[tau]
                    p = sigmoid(theta)
                    ll += R_filled[t, tau] * np.log(p + 1e-15) + (1 - R_filled[t, tau]) * np.log(1 - p + 1e-15)

        if verbose:
            print(f"  EM iter {iteration:3d}  log-likelihood = {ll:.2f}")

        if abs(ll - prev_ll) < tol:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break
        prev_ll = ll

    # Map s_hist to [0, 1] via sigmoid for the bandit layer
    s_hist_01 = sigmoid(s_hist)
    # Normalize A columns for interpretability
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)

    return A, d, s_hist_01, ll


def select_k_bic(
    R: np.ndarray,
    K_candidates: list[int] = None,
    verbose: bool = False,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Select optimal K via BIC.

    Args:
        R: (N_warm, M) observation matrix.
        K_candidates: List of K values to try (default: 1..7).
        verbose: Print BIC scores.

    Returns:
        best_K, A, d, s_hist, ll for the best K
    """
    if K_candidates is None:
        K_candidates = list(range(1, 8))

    N_warm, M = R.shape
    obs_count = np.sum(~np.isnan(R))

    best_K = K_candidates[0]
    best_bic = np.inf
    best_params = None
    best_ll = -np.inf

    for K in K_candidates:
        A, d, s_hist, ll = fit_mirt_em(R, K, verbose=verbose)
        # BIC: -2*LL + n_params * log(n_obs)
        # n_params: (M*K) for A + M for d + (N_warm*K) for s
        #   Note: s is "free" in EM but penalized in BIC
        n_params = M * K + M + N_warm * K
        bic = -2 * ll + n_params * np.log(obs_count)
        if verbose:
            print(f"  K={K}: BIC={bic:.1f}  LL={ll:.1f}")
        if bic < best_bic:
            best_bic = bic
            best_K = K
            best_params = (A, d, s_hist)
            best_ll = ll

    return best_K, *best_params, best_ll


def online_update_profile(
    s_t: np.ndarray,
    A_tau: np.ndarray,
    d_tau: float,
    success: bool,
    sigma_s: float = 1.0,
    max_iter: int = 5,
) -> np.ndarray:
    """One-step Bayesian online profile update (proposal Section 3.1.4).

    Args:
        s_t: current profile (K,)
        A_tau: loading vector for task tau (K,)
        d_tau: difficulty for task tau
        success: whether execution was successful
        sigma_s: smoothing prior std (larger = more responsive)

    Returns:
        s_{t+1}: updated profile (K,)
    """
    K = len(s_t)
    s = s_t.copy()

    for _ in range(max_iter):
        theta = A_tau @ s - d_tau
        p = sigmoid(theta)

        # Gradient
        grad = (success - p) * A_tau - (1.0 / sigma_s**2) * (s - s_t)

        # Hessian
        W = p * (1 - p)
        hess = -W * np.outer(A_tau, A_tau) - (1.0 / sigma_s**2) * np.eye(K)

        # Newton step
        delta = np.linalg.solve(hess, grad)
        s = s - delta

    return np.clip(s, 0.0, 1.0)
