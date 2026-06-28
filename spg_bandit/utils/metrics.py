"""Metrics for SPG-Bandit evaluation."""

import numpy as np


def cumulative_regret(
    history: list[dict],
    oracle_delta: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cumulative regret RT from experiment history.

    Uses the actual reward (delta g_k^T Δ̂_k) at each step.
    The oracle is approximated as the maximum observed delta at each step,
    or from a provided oracle_delta array.

    Args:
        history: List of step records from orchestrator.
        oracle_delta: (T,) array of best possible delta at each step.
                      If None, use max observed delta per step.

    Returns:
        (T,) array of cumulative regret values.
    """
    T = len(history)
    regrets = np.zeros(T)

    for t, record in enumerate(history):
        chosen_delta = np.max(record["delta"])  # max dimension improvement
        if oracle_delta is not None:
            best_delta = oracle_delta[t]
        else:
            best_delta = chosen_delta  # no external oracle
        regrets[t] = best_delta - chosen_delta

    return np.cumsum(regrets)


def profile_quality(
    history: list[dict],
    target_profile: np.ndarray | None = None,
) -> np.ndarray:
    """Compute L2 distance from recovered profile to target at each step.

    Args:
        history: List of step records.
        target_profile: (K,) target profile. If None, use (1, 1, ..., 1).

    Returns:
        (T+1,) array of distances (including initial state at t=0).
    """
    if target_profile is None:
        target_profile = np.ones(len(history[0]["profile_before"]))

    distances = []
    distances.append(np.linalg.norm(history[0]["profile_before"] - target_profile))
    for record in history:
        distances.append(np.linalg.norm(record["profile_after"] - target_profile))

    return np.array(distances)
