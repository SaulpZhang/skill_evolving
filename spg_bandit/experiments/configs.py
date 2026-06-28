"""Default experiment configurations for SPG-Bandit evaluation."""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Configuration for a synthetic environment."""
    name: str
    M: int = 100            # number of tasks
    K: int = 5              # ground-truth latent dimensions
    d_c: int = 32           # LLM embedding dimension
    T: int = 500            # total steps
    N_warm: int = 30        # warmup steps
    seed: int = 42


@dataclass
class SelectorConfig:
    """Configuration for a task selector."""
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Top-level experiment config."""
    env: EnvConfig = field(default_factory=EnvConfig)
    selectors: list[SelectorConfig] = field(default_factory=list)
    n_seeds: int = 5


# Pre-built selector config sets
EXPERIMENT_A_SELECTORS = [
    SelectorConfig("uniform"),
    SelectorConfig("random"),
    SelectorConfig("epsilon_greedy", {"epsilon": 0.1}),
    SelectorConfig("linucb_scalar", {"alpha": 0.1}),
    SelectorConfig("thompson_sampling"),
    SelectorConfig("spg_bandit_nogap", {"alpha": 0.1, "gap_weighted": False}),
    SelectorConfig("spg_bandit", {"alpha": 0.1}),
]

EXPERIMENT_B_ABLATION_SELECTORS = [
    SelectorConfig("spg_bandit", {"alpha": 0.1}),
    SelectorConfig("spg_bandit_nogap", {"alpha": 0.1, "gap_weighted": False}),
    SelectorConfig("spg_bandit_noucb", {"alpha": 0.0, "use_ucb": False}),
    SelectorConfig("spg_bandit_nogap_noucb", {"alpha": 0.0, "gap_weighted": False, "use_ucb": False}),
]
