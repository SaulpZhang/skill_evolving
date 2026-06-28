"""Experiment runner: orchestrates warmup, MIRT fitting, and online evaluation."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from spg_bandit.core import TaskPool, SPGBandit
from spg_bandit.selectors import (
    BaseSelector,
    MLPFeaturizer,
    UniformSelector,
    RandomSelector,
    EpsilonGreedy,
    LinUCBScalar,
    ThompsonSampling,
    SPGBanditSelector,
)
from spg_bandit._mock import (
    MIRTMockMethod,
    DeltaIncrementMethod,
    TabularCountMethod,
    RandomWalkMethod,
)
from spg_bandit.mirt import fit_mirt_em, select_k_bic, online_update_profile
from spg_bandit.utils import cumulative_regret, profile_quality
from .configs import ExperimentConfig, SelectorConfig, EnvConfig


# ── Synthetic data generators ──────────────────────────────────────────────

def generate_synthetic_task_pool(
    M: int,
    K: int,
    d_c: int,
    A: np.ndarray | None = None,
    d: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[TaskPool, np.ndarray, np.ndarray]:
    """Generate a synthetic task pool with MIRT-modeled tasks.

    Returns:
        task_pool, loading_matrix_A, difficulty_d
    """
    rng = np.random.default_rng(seed)

    # Generate LLM-like embeddings
    embeddings = rng.normal(0, 1, (M, d_c))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Generate MIRT parameters
    if A is None:
        A = rng.dirichlet(np.full(K, 0.5), M)  # (M, K), rows sum to 1
    if d is None:
        d = rng.uniform(0.0, 1.0, M)

    task_pool = TaskPool(embeddings)
    return task_pool, A, d


def generate_warmup_data(
    A: np.ndarray,
    d: np.ndarray,
    s0: np.ndarray,
    M: int,
    N_warm: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run warmup phase: uniformly sample tasks and record outcomes.

    Returns:
        R: (N_warm, M) observation matrix (NaN for unobserved).
        warmup_task_ids: (N_warm,) sequence of selected task IDs.
        s_history: (N_warm + 1, K) true profile history.
    """
    rng = np.random.default_rng(seed)
    K = len(s0)
    R = np.full((N_warm, M), np.nan)
    warmup_task_ids = np.zeros(N_warm, dtype=int)
    s = s0.copy()
    s_history = [s0.copy()]

    for t in range(N_warm):
        # Uniform sample
        tau = int(rng.integers(M))
        warmup_task_ids[t] = tau

        # Execute
        theta = A[tau] @ s - d[tau]
        p = 1.0 / (1.0 + np.exp(-theta))
        success = 1 if rng.random() < p else 0
        R[t, tau] = success

        # Update profile (using same dynamics as MIRTMockMethod)
        s = online_update_profile(s, A[tau], d[tau], bool(success))
        s_history.append(s.copy())

    return R, warmup_task_ids, np.array(s_history)


# ── Selector factory ────────────────────────────────────────────────────────

def create_selector(
    cfg: SelectorConfig,
    env_cfg: EnvConfig,
    mlp: MLPFeaturizer | None = None,
) -> BaseSelector:
    """Create a selector from config."""
    params = cfg.params

    if cfg.name == "uniform":
        return UniformSelector()
    elif cfg.name == "random":
        return RandomSelector()
    elif cfg.name == "epsilon_greedy":
        return EpsilonGreedy(epsilon=params.get("epsilon", 0.1))
    elif cfg.name == "linucb_scalar":
        return LinUCBScalar(alpha=params.get("alpha", 0.1))
    elif cfg.name == "thompson_sampling":
        return ThompsonSampling()
    elif cfg.name.startswith("spg_bandit"):
        assert mlp is not None, "SPG-Bandit requires MLP featurizer"
        return SPGBanditSelector(
            mlp=mlp,
            K=env_cfg.K,
            d_f=params.get("d_f", 64),
            alpha=params.get("alpha", 0.1),
            tau=params.get("tau", 0.1),
            gap_weighted=params.get("gap_weighted", True),
            use_ucb=params.get("use_ucb", True),
        )
    else:
        raise ValueError(f"Unknown selector: {cfg.name}")


# ── Method factory ─────────────────────────────────────────────────────────

def create_method(
    method_name: str,
    A: np.ndarray,
    d: np.ndarray,
    s0: np.ndarray,
    seed: int = 42,
) -> Any:
    """Create a skill evolving method by name."""
    if method_name == "mirt_mock":
        return MIRTMockMethod(A, d, s0, seed=seed)
    elif method_name == "delta_increment":
        return DeltaIncrementMethod(A, d, s0, seed=seed)
    elif method_name == "tabular_count":
        return TabularCountMethod(A, d, s0, seed=seed)
    elif method_name == "random_walk":
        return RandomWalkMethod(A, d, s0, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ── Core experiment function ────────────────────────────────────────────────

def run_experiment(
    env_cfg: EnvConfig,
    method_name: str,
    selector_cfgs: list[SelectorConfig],
    n_seeds: int = 5,
    output_dir: str = "results",
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Run a full experiment: warmup → MIRT → online evaluation.

    Args:
        env_cfg: Environment configuration.
        method_name: Which SkillEvolvingMethod to use.
        selector_cfgs: List of selector configurations to compare.
        n_seeds: Number of random seeds.
        output_dir: Directory to save results.
        verbose: Print progress.

    Returns:
        {selector_name: [seed_results]}, where each seed_result is
        a dict with keys: "history", "cumulative_regret", etc.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {
        sc.name: [] for sc in selector_cfgs
    }

    base_seed = env_cfg.seed

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        rng = np.random.default_rng(seed)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Seed {seed}  ({seed_offset + 1}/{n_seeds})")
            print(f"{'='*60}")

        # ── Generate shared environment ──
        task_pool, A_gt, d_gt = generate_synthetic_task_pool(
            env_cfg.M, env_cfg.K, env_cfg.d_c, seed=seed,
        )
        s0 = np.random.uniform(0.1, 0.3, env_cfg.K)  # initial low skill

        # ── Warmup phase ──
        R, warmup_task_ids, _ = generate_warmup_data(
            A_gt, d_gt, s0, env_cfg.M, env_cfg.N_warm, seed=seed + 1000,
        )

        # ── MIRT fitting (on warmup data) ──
        best_K = env_cfg.K  # use ground-truth for now
        A_mirt, d_mirt, s_hist_mirt, ll = fit_mirt_em(
            R, best_K, verbose=verbose,
        )
        s0_mirt = s_hist_mirt[-1]  # use final warmup profile as initial online profile

        if verbose:
            print(f"  MIRT fitted K={best_K}, LL={ll:.1f}")

        # ── Pre-train MLP on warmup data ──
        warmup_embeddings = np.array([
            task_pool.get_embedding(tid) for tid in warmup_task_ids
        ])
        warmup_deltas = s_hist_mirt[1:] - s_hist_mirt[:-1]  # (N_warm, K)

        mlp = MLPFeaturizer(
            d_c=env_cfg.d_c, d_h=128, d_f=64, seed=seed,
        )
        # Warmup: predict deltas from embeddings
        mlp.train(warmup_embeddings, warmup_deltas, epochs=50, verbose=verbose)

        # ── Run each selector ──
        for sc in selector_cfgs:
            if verbose:
                print(f"\n  --- Selector: {sc.name} ---")

            method = create_method(method_name, A_gt, d_gt, s0_mirt, seed=seed)

            if sc.name.startswith("spg_bandit"):
                selector = create_selector(sc, env_cfg, mlp)
            else:
                selector = create_selector(sc, env_cfg)

            orchestrator = SPGBandit(task_pool, selector, method)
            history = orchestrator.run(env_cfg.T)

            regret = cumulative_regret(history)
            quality = profile_quality(history)

            if verbose:
                usage = orchestrator.get_total_usage()
                print(f"    Usage: {usage['api_calls']} calls, "
                      f"{usage['prompt_tokens']} prompt, "
                      f"{usage['completion_tokens']} completion, "
                      f"{usage['total_tokens']} total tokens")

            result = {
                "seed": seed,
                "history": [
                    {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in rec.items()}
                    for rec in history
                ],
                "cumulative_regret": regret.tolist(),
                "profile_quality": quality.tolist(),
                "final_regret": float(regret[-1]),
                "total_usage": orchestrator.get_total_usage(),
                "orchestrator": orchestrator,
                "selector": selector,
            }
            results[sc.name].append(result)

    # ── Save results ──
    save_results(results, output_dir, env_cfg, method_name, verbose)

    # ── Print usage summary ──
    if verbose:
        print(f"\n{'='*60}")
        print("USAGE SUMMARY (avg across seeds):")
        print(f"{'='*60}")
        for sc_name, seed_results in results.items():
            total = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            n = len(seed_results)
            for sr in seed_results:
                u = sr.get("total_usage", {})
                for k in total:
                    total[k] += u.get(k, 0)
            print(f"  {sc_name:25s}: {total['api_calls']/n:6.1f} calls/seed  "
                  f"{total['total_tokens']/n:8.0f} total tokens/seed")

    return results


def save_results(
    results: dict,
    output_dir: str,
    env_cfg: EnvConfig,
    method_name: str,
    verbose: bool = True,
):
    """Save experiment results to disk."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    env_tag = f"{env_cfg.name}_{method_name}"
    filename = f"results_{env_tag}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    # Strip non-serializable objects
    serializable = {}
    for sel_name, seed_results in results.items():
        serializable[sel_name] = []
        for sr in seed_results:
            usage = sr.get("total_usage", {})
            serializable[sel_name].append({
                "seed": sr["seed"],
                "cumulative_regret": sr["cumulative_regret"],
                "profile_quality": sr["profile_quality"],
                "final_regret": sr["final_regret"],
                "total_usage": usage,
                "history_len": len(sr["history"]),
            })

    with open(filepath, "w") as f:
        json.dump({
            "env_config": {
                "name": env_cfg.name,
                "M": env_cfg.M,
                "K": env_cfg.K,
                "T": env_cfg.T,
                "N_warm": env_cfg.N_warm,
            },
            "method": method_name,
            "results": serializable,
        }, f, indent=2)

    if verbose:
        print(f"\nResults saved to {filepath}")


# ── Convenience run function ────────────────────────────────────────────────

def run_block_1(verbose: bool = True):
    """B1: Main anchor result — SPG-Bandit vs baselines on Env-Synth."""
    env_cfg = EnvConfig(name="synth", M=100, K=5, T=500, N_warm=30)
    from .configs import EXPERIMENT_A_SELECTORS
    return run_experiment(
        env_cfg, "mirt_mock", EXPERIMENT_A_SELECTORS,
        n_seeds=5, verbose=verbose,
    )


def run_block_2(verbose: bool = True):
    """B2: Novelty isolation — gap weighting ablation."""
    env_cfg = EnvConfig(name="synth", M=100, K=5, T=500, N_warm=30)
    from .configs import EXPERIMENT_B_ABLATION_SELECTORS
    return run_experiment(
        env_cfg, "mirt_mock", EXPERIMENT_B_ABLATION_SELECTORS,
        n_seeds=5, verbose=verbose,
    )
