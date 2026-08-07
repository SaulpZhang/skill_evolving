"""Dataset registry and factory.

Datasets are imported lazily so importing the framework interfaces does not
require every environment's optional dependencies.  A new dataset can either
be registered in this module or registered by an application before calling
``create_dataset``.
"""

from __future__ import annotations

import importlib
from typing import Type

from spg_bandit.modules.dataset.base import BaseDataset


_DATASET_IMPORTS = {
    "alfworld": (
        "spg_bandit.modules.dataset.alfworld",
        "ALFWorldDataset",
    ),
}
_DATASET_CLASSES: dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str, dataset_class: Type[BaseDataset]) -> None:
    """Register a dataset implementation under a config-friendly name."""
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("Dataset name must not be empty")
    if not isinstance(dataset_class, type) or not issubclass(dataset_class, BaseDataset):
        raise TypeError("dataset_class must be a BaseDataset subclass")
    _DATASET_CLASSES[normalized] = dataset_class


def _load_class(name: str) -> Type[BaseDataset]:
    raw_name = str(name).strip()
    normalized = raw_name.lower()
    if normalized in _DATASET_CLASSES:
        return _DATASET_CLASSES[normalized]

    if normalized in _DATASET_IMPORTS:
        module_name, class_name = _DATASET_IMPORTS[normalized]
    elif ":" in raw_name:
        module_name, class_name = raw_name.split(":", 1)
    elif "." in raw_name:
        module_name, class_name = raw_name.rsplit(".", 1)
    else:
        available = ", ".join(sorted(set(_DATASET_IMPORTS) | set(_DATASET_CLASSES)))
        raise ValueError(
            f"Unknown dataset '{name}'. Available datasets: {available or '(none)'}; "
            "or use a module.path:ClassName entry."
        )

    module = importlib.import_module(module_name)
    dataset_class = getattr(module, class_name)
    register_dataset(normalized, dataset_class)
    return dataset_class


def get_dataset_class(name: str) -> Type[BaseDataset]:
    """Return a lazily imported dataset class."""
    return _load_class(name)


def create_dataset(name: str, config: dict) -> BaseDataset:
    """Instantiate a configured dataset from its registry name."""
    dataset_class = get_dataset_class(name)
    return dataset_class(config)


def available_datasets() -> list[str]:
    """List built-in and already registered dataset names."""
    return sorted(set(_DATASET_IMPORTS) | set(_DATASET_CLASSES))
