"""Base selector with common utilities."""

from abc import abstractmethod
import numpy as np
from spg_bandit.core.interfaces import TaskSelector, TaskPool


class BaseSelector(TaskSelector):
    """Base class providing common helpers for all selectors."""

    def __init__(self, name: str = "base"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def select(self, task_pool: TaskPool, profile: np.ndarray) -> int:
        ...

    def update(
        self,
        task_id: int,
        profile_before: np.ndarray,
        profile_after: np.ndarray,
        success: bool,
    ) -> None:
        """Default: no-op. Override in subclasses that learn from outcomes."""
        pass

    def reset(self) -> None:
        """Override in subclasses that maintain state."""
        pass
