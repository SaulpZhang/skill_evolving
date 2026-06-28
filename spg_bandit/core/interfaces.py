"""Pluggable interfaces for SPG-Bandit framework."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class SkillEvolvingMethod(ABC):
    """Pluggable skill evolution dynamics.

    Any existing skill evolving method can be adapted by implementing
    this interface. The framework calls execute() and update_profile()
    each step, and get_profile() to query the current agent state.
    """

    @abstractmethod
    def execute(self, task_id: int) -> bool:
        """Execute a task, return whether it succeeded."""

    @abstractmethod
    def get_profile(self) -> np.ndarray:
        """Return current skill profile s_t in [0, 1]^K."""

    @abstractmethod
    def update_profile(self, task_id: int, success: bool) -> None:
        """Update the skill profile after executing task_id with given outcome."""

    @abstractmethod
    def get_K(self) -> int:
        """Return the dimensionality K of the skill profile."""

    def get_usage(self) -> dict:
        """Return token/call usage stats. Default: empty.

        Subclasses that make LLM API calls should override this.
        """
        return {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


class TaskSelector(ABC):
    """Pluggable task selection policy.

    Each selector receives the current skill profile and task pool,
    picks a task, and updates its internal state from the outcome.
    """

    @abstractmethod
    def select(
        self,
        task_pool: "TaskPool",  # noqa: F821
        profile: np.ndarray,
    ) -> int:
        """Select a task ID from the pool given current skill profile."""

    @abstractmethod
    def update(
        self,
        task_id: int,
        profile_before: np.ndarray,
        profile_after: np.ndarray,
        success: bool,
    ) -> None:
        """Update selector internal state after observing outcome."""

    @abstractmethod
    def reset(self) -> None:
        """Reset selector to initial state (for new runs)."""

    def get_usage(self) -> dict:
        """Return token/call usage stats. Default: empty.

        Subclasses that make LLM API calls should override this.
        """
        return {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


class TaskPool:
    """A fixed pool of M tasks with embeddings and metadata.

    Each task has:
    - task_id: int (0 .. M-1)
    - embedding: d_c-dim vector from a frozen LLM
    - metadata: optional dict (e.g., category, difficulty estimate)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ):
        assert embeddings.ndim == 2, "embeddings must be (M, d_c)"
        self.embeddings = embeddings
        self.M = embeddings.shape[0]
        self.d_c = embeddings.shape[1]
        self.metadata = metadata or [{} for _ in range(self.M)]

    def get_embedding(self, task_id: int) -> np.ndarray:
        return self.embeddings[task_id]

    def get_all_embeddings(self) -> np.ndarray:
        return self.embeddings

    def __len__(self) -> int:
        return self.M
