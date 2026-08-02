from .base import BaseDataset, TaskPool

__all__ = ["ALFWorldDataset", "BaseDataset", "TaskPool"]


def __getattr__(name):
    """Avoid importing ALFWorld/TextWorld when only base interfaces are needed."""
    if name == "ALFWorldDataset":
        from .alfworld import ALFWorldDataset
        return ALFWorldDataset
    raise AttributeError(name)
