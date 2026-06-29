"""Task selector interface."""

from abc import ABC, abstractmethod

from spg_bandit.modules.dataset.base import TaskPool


class BaseSelector(ABC):
    """Base class for task selection strategies."""

    @property
    def needs_warmup(self) -> bool:
        return False

    @abstractmethod
    def select(self, task_pool: TaskPool) -> int:
        ...

    @abstractmethod
    def update(self, task_id: int, result: dict):
        ...

    def reset(self):
        pass
