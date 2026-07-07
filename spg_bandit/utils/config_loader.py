"""Load YAML config files with default override support."""

import os
import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str, default_name: str = "default") -> dict:
    config_dir = Path(__file__).parents[1] / "config"
    default_file = config_dir / f"{default_name}.yaml"
    config = {}
    if default_file.exists():
        with open(default_file) as f:
            config = yaml.safe_load(f) or {}
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        cfg_path = config_dir / f"{cfg_path.stem}.yaml"
        if not cfg_path.exists():
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
