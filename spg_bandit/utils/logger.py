"""File-based logger. Creates logs/<date>_<run_id>/run.log."""

import os
import sys
import time
import logging
from pathlib import Path


def setup_logger(run_id: str = None, run_name: str = None,
                 log_dir: str = "logs", log_file_enabled: bool = False):
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("spg_bandit")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler (always on)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # File handler (optional)
    log_file = None
    if log_file_enabled:
        log_subdir = f"{run_id}"
        if run_name:
            log_subdir += f"_{run_name}"
        log_path = Path(log_dir) / log_subdir
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "run.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        logger.info(f"Log file: {log_file}")

    logger.info(f"Run ID: {run_id}")
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("spg_bandit")
