from .base import BaseDataset, TaskPool
from .base import EnvironmentState, EnvironmentStep
from .embedding_cache import EmbeddingCache
from .registry import available_datasets, create_dataset, get_dataset_class, register_dataset

__all__ = [
    "ALFWorldDataset", "BaseDataset", "EnvironmentState", "EnvironmentStep",
    "TaskPool", "EmbeddingCache", "available_datasets", "create_dataset", "get_dataset_class",
    "register_dataset",
]


def __getattr__(name):
    """Avoid importing ALFWorld/TextWorld when only base interfaces are needed."""
    if name == "ALFWorldDataset":
        from .alfworld import ALFWorldDataset
        return ALFWorldDataset
    raise AttributeError(name)
