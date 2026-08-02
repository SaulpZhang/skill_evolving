"""Load YAML config files with default override support."""

import os
import yaml
from pathlib import Path
from typing import Any


def resolve_config_path(config_path: str, default_name: str = "default") -> Path | None:
    """Return the YAML file selected by a CLI config argument, if it exists."""
    config_dir = Path(__file__).parents[1] / "config"
    path = Path(config_path)
    if path.exists():
        return path
    candidate = config_dir / f"{path.stem}.yaml"
    if candidate.exists():
        return candidate
    default_file = config_dir / f"{default_name}.yaml"
    return default_file if default_file.exists() else None


def load_config(config_path: str, default_name: str = "default") -> dict:
    config_dir = Path(__file__).parents[1] / "config"
    default_file = config_dir / f"{default_name}.yaml"
    config = {}
    if default_file.exists():
        with open(default_file) as f:
            config = yaml.safe_load(f) or {}
    cfg_path = resolve_config_path(config_path, default_name)
    if cfg_path is None:
        print(f"Config {config_path} not found, using defaults only")
        return config
    with open(cfg_path) as f:
        override = yaml.safe_load(f) or {}
    _deep_merge(config, override)
    print(f"Loaded config: {cfg_path}")
    return config


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def get_param(config: dict, *keys: str, default: Any = None) -> Any:
    current = config
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default
    return current
