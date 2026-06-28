"""Quick sanity test: verify the entire framework runs end-to-end."""

import numpy as np
from spg_bandit.core import TaskPool, SPGBandit
from spg_bandit.selectors import (
    UniformSelector, RandomSelector, EpsilonGreedy,
    LinUCBScalar, ThompsonSampling, SPGBanditSelector, MLPFeaturizer,
)
from spg_bandit._mock import (
    MIRTMockMethod, DeltaIncrementMethod, TabularCountMethod, RandomWalkMethod,
)
from spg_bandit.mirt import fit_mirt_em, online_update_profile, select_k_bic
from spg_bandit.utils import cumulative_regret


def test_interface_contract():
    """Verify all selectors implement the ABC correctly."""
    from spg_bandit.core.interfaces import TaskSelector
    selectors = [
        UniformSelector(),
        RandomSelector(),
        EpsilonGreedy(),
        LinUCBScalar(),
        ThompsonSampling(),
    ]
    for s in selectors:
        assert isinstance(s, TaskSelector), f"{s.name} does not implement TaskSelector"


def test_task_pool():
    pool = TaskPool(np.random.randn(50, 16))
    assert len(pool) == 50
    assert pool.d_c == 16
    assert pool.get_embedding(0).shape == (16,)


def test_mirt_em():
    """MIRT EM can fit data generated from a known model."""
    rng = np.random.default_rng(42)
    M, K, N = 30, 3, 100
    A = rng.dirichlet(np.full(K, 0.5), M)
    d = rng.uniform(0.0, 1.0, M)
    s = np.random.uniform(0.2, 0.8, (N, K))

    R = np.full((N, M), np.nan)
    for t in range(N):
        for tau in range(M):
            if rng.random() < 0.3:  # 30% observation probability
                theta = A[tau] @ s[t] - d[tau]
                p = 1.0 / (1.0 + np.exp(-theta))
                R[t, tau] = 1 if rng.random() < p else 0

    A_fit, d_fit, s_fit, ll = fit_mirt_em(R, K, max_iter=50, verbose=False)
    assert not np.any(np.isnan(A_fit)), "MIRT EM produced NaN in A"
    assert not np.any(np.isnan(d_fit)), "MIRT EM produced NaN in d"
    print(f"  MIRT EM: LL = {ll:.1f}")


def test_online_update():
    s = np.array([0.3, 0.5, 0.7])
    a = np.array([0.8, 0.1, 0.1])
    d = 0.0
    s_new = online_update_profile(s, a, d, success=True, sigma_s=1.0)
    assert s_new.shape == s.shape
    assert np.all(s_new >= 0.0) and np.all(s_new <= 1.0)
    # Successful execution on a loaded dimension should increase that dimension
    assert s_new[0] >= s[0], f"Expected increase in loaded dim, got {s_new[0]} < {s[0]}"


def test_spg_bandit_orchestrator():
    """Full SPG-Bandit orchestrator runs for T steps."""
    M, K, d_c, T = 20, 3, 8, 50
    rng = np.random.default_rng(42)

    pool = TaskPool(rng.normal(0, 1, (M, d_c)))
    A = rng.dirichlet(np.full(K, 0.5), M)
    d_vec = rng.uniform(0.0, 1.0, M)
    s0 = np.array([0.2, 0.2, 0.2])

    method = MIRTMockMethod(A, d_vec, s0, seed=42)
    mlp = MLPFeaturizer(d_c, d_h=16, d_f=8, seed=42)

    # Quick MLP pre-training on synthetic data
    X_train = rng.normal(0, 1, (30, d_c))
    y_train = rng.normal(0, 0.1, (30, K))
    mlp.train(X_train, y_train, epochs=20, verbose=False)

    selector = SPGBanditSelector(mlp, K=K, d_f=8, alpha=0.1, tau=0.1)
    bandit = SPGBandit(pool, selector, method)
    history = bandit.run(T)

    assert len(history) == T
    assert all("delta" in h for h in history)
    assert all("success" in h for h in history)
    assert all("task_id" in h for h in history)

    regret = cumulative_regret(history)
    assert len(regret) == T
    print(f"  SPG-Bandit orchestrator: T={T}, final regret={regret[-1]:.3f}")


def test_all_methods():
    """All 4 SkillEvolvingMethods can run with uniform selector."""
    M, K, T = 20, 3, 30
    rng = np.random.default_rng(42)
    pool = TaskPool(rng.normal(0, 1, (M, 8)))
    A = rng.dirichlet(np.full(K, 0.5), M)
    d = rng.uniform(0.0, 1.0, M)
    s0 = np.array([0.2, 0.3, 0.4])

    methods = {
        "mirt_mock": MIRTMockMethod(A, d, s0, seed=42),
        "delta_increment": DeltaIncrementMethod(A, d, s0, eta=0.05, seed=42),
        "tabular_count": TabularCountMethod(A, d, s0, gamma=0.1, seed=42),
        "random_walk": RandomWalkMethod(A, d, s0, noise_scale=0.05, seed=42),
    }

    for name, method in methods.items():
        selector = UniformSelector()
        bandit = SPGBandit(pool, selector, method)
        history = bandit.run(T)
        assert len(history) == T, f"{name}: expected {T} steps, got {len(history)}"
        print(f"  {name}: ran {T} steps OK")


if __name__ == "__main__":
    print("Running sanity tests...")

    test_interface_contract()
    print("✓ interface_contract")

    test_task_pool()
    print("✓ task_pool")

    test_mirt_em()
    print("✓ mirt_em")

    test_online_update()
    print("✓ online_update")

    test_spg_bandit_orchestrator()
    print("✓ spg_bandit_orchestrator")

    test_all_methods()
    print("✓ all_methods")

    print("\nAll sanity tests passed!")
