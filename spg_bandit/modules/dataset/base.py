"""Dataset interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TaskPool:
    """A fixed pool of M tasks with embeddings and metadata."""
    embeddings: np.ndarray       # (M, d_c) LLM embeddings
    metadata: list[dict] = field(default_factory=list)

    def __post_init__(self):
        assert self.embeddings.ndim == 2, "embeddings must be (M, d_c)"

    @property
    def M(self) -> int:
        return self.embeddings.shape[0]

    @property
    def d_c(self) -> int:
        return self.embeddings.shape[1]

    def get_embedding(self, i: int) -> np.ndarray:
        return self.embeddings[i]

    def get_goal(self, i: int) -> str:
        return self.metadata[i].get("goal", "") if i < len(self.metadata) else ""


class BaseDataset(ABC):
    """Base class for datasets."""

    @property
    @abstractmethod
    def task_pool(self) -> TaskPool:
        ...

    @abstractmethod
    def get_task_goal(self, task_id: int) -> str:
        ...

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def create_env(self, task_id: int):
        """Create an environment instance for executing a specific task."""
        ...
