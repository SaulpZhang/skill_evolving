#!/usr/bin/env python3
"""Run a saved experiment's final SkillBank on its evaluation split."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from spg_bandit.main import create_skill_evolving
from spg_bandit.modules.dataset import create_dataset
from spg_bandit.modules.selector.uniform import UniformSelector
from spg_bandit.utils.config_loader import load_config
from spg_bandit.utils.logger import setup_logger
from spg_bandit.utils.recorder import Recorder
from spg_bandit.utils.wandb import finish_wandb, init_wandb, log_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunPaths:
    """Artifacts required to evaluate one completed experiment run."""

    run_id: str
    run_dir: Path
    config_path: Path
    skills_dir: Path


def resolve_run_paths(run_id: str, project_root: Path = PROJECT_ROOT) -> RunPaths:
    """Resolve the saved config and evolved SkillBank for ``run_id``."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must not be empty")

    project_root = Path(project_root)
    run_dir = project_root / "logs" / run_id
    config_path = run_dir / "records" / "config.yaml"
    skills_dir = project_root / "skills" / run_id
    skill_path = skills_dir / "skills.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Saved run config not found: {config_path}")
    if not skill_path.is_file():
        raise FileNotFoundError(f"Evolved SkillBank not found: {skill_path}")
    return RunPaths(run_id, run_dir, config_path, skills_dir)


def _dataset_name_and_params(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    setting = config.get("dataset", "alfworld")
    if isinstance(setting, str):
        return setting, {}
    if not isinstance(setting, dict):
        raise ValueError("dataset must be a name or a mapping with a 'name' field")

    params = dict(setting.get("params", {}) or {})
    for key, value in setting.items():
        if key not in {"name", "params"}:
            params.setdefault(key, value)
    return setting.get("name", "alfworld"), params


def _make_dataset_config(
    config: dict[str, Any], dataset_params: dict[str, Any], overrides: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(dataset_params)
    cfg.setdefault("task_types", "all")
    cfg.setdefault("split", "valid_seen")
    cfg.update({
        "embedding_model": config.get(
            "embedding_model", cfg.get("embedding_model", "all-MiniLM-L6-v2")
        ),
        "embedding_type": config.get(
            "embedding_type", cfg.get("embedding_type", "local")
        ),
        "embedding_url": config.get("embedding_url", cfg.get("embedding_url", "")),
        "embedding_cache": config.get("embedding_cache", cfg.get("embedding_cache", True)),
        "embedding_cache_dir": config.get(
            "embedding_cache_dir", cfg.get("embedding_cache_dir")
        ),
        "embedding_cache_save_interval": config.get(
            "embedding_cache_save_interval",
            cfg.get("embedding_cache_save_interval", 100),
        ),
        "max_turns": config.get("max_turns", cfg.get("max_turns", 51)),
    })
    cfg.update(overrides)
    return cfg


def _evaluation_skill_config(skill_config: dict[str, Any]) -> dict[str, Any]:
    """Return the saved SkillRL/SkillOpt config in no-update evaluation mode."""
    evaluation_config = dict(skill_config)
    evaluation_config["enable_dynamic_update"] = False
    if "evaluation_temperature" in skill_config:
        evaluation_config["temperature"] = skill_config["evaluation_temperature"]
    if "evaluation_max_tokens" in skill_config:
        evaluation_config["max_tokens"] = skill_config["evaluation_max_tokens"]
    return evaluation_config


def run_evaluation(
    run_id: str,
    *,
    split: str | None = None,
    rollouts: int | None = None,
    seed: int | None = None,
    no_wandb: bool = False,
    output_dir: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Evaluate a completed run's saved final SkillBank.

    The experiment's saved config controls the dataset, model endpoint, prompt
    settings, and default evaluation split. Dynamic skill updates are always
    disabled so this command never mutates the saved ``skills.json``.
    """
    paths = resolve_run_paths(run_id, project_root=project_root)
    config = load_config(str(paths.config_path))
    if seed is not None:
        config.setdefault("experiment", {})["seed"] = seed
    run_seed = int(config.get("experiment", {}).get("seed", 42))
    random.seed(run_seed)

    dataset_name, dataset_params = _dataset_name_and_params(config)
    evaluate_cfg = config.get("evaluate", {})
    if not isinstance(evaluate_cfg, dict):
        raise ValueError("evaluate must be a mapping")
    dataset_overrides = dict(evaluate_cfg)
    if split is not None:
        dataset_overrides["split"] = split
    dataset_overrides.setdefault("split", "valid_seen")

    skill_config = dict(config.get("skill_evolving", {}) or {})
    agent_name = skill_config.get("name", "simple_agent")
    eval_skill_config = _evaluation_skill_config(skill_config)
    default_rollouts = int(skill_config.get("evaluation_rollouts_per_task", 1))
    evaluation_rollouts = int(rollouts if rollouts is not None else default_rollouts)
    if evaluation_rollouts < 1:
        raise ValueError("rollouts must be at least 1")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) if output_dir is not None else (
        paths.run_dir / "evaluations" / timestamp
    )
    records_dir = output_dir / "records"
    recorder = Recorder(str(records_dir))
    shutil.copyfile(paths.config_path, records_dir / "config.yaml")
    logger = setup_logger(
        f"{run_id}_evaluate_{timestamp}",
        f"{run_id}_evaluate_{timestamp}",
        log_file_enabled=False,
    )

    wandb_id = f"{run_id}_evaluate_{timestamp}"
    init_wandb(config, wandb_id, wandb_id, enabled=not no_wandb)
    try:
        logger.info("Evaluating saved run %s", run_id)
        logger.info("SkillBank: %s", paths.skills_dir / "skills.json")
        logger.info("Evaluation split: %s", dataset_overrides["split"])

        dataset = create_dataset(
            dataset_name,
            _make_dataset_config(config, dataset_params, dataset_overrides),
        )
        task_pool = dataset.task_pool
        if task_pool.M == 0:
            raise ValueError("Evaluation dataset contains no tasks")

        max_turns = int(config.get("max_turns", 51))
        method = create_skill_evolving(
            agent_name,
            dataset,
            max_turns=max_turns,
            records_dir=str(output_dir / "messages"),
            skill_config=eval_skill_config,
        )
        method.load_skills(str(paths.skills_dir))
        selector = UniformSelector(seed=run_seed)

        successes = 0
        total_rollouts = 0
        api_calls = 0
        for step in range(task_pool.M):
            task_id = selector.select(task_pool)
            started = time.time()
            result = method.execute(task_id, num_rollouts=evaluation_rollouts)
            elapsed = time.time() - started
            task_successes = int(result.get("successes", int(bool(result["success"]))))
            task_rollouts = int(result.get("num_rollouts", evaluation_rollouts))
            successes += task_successes
            total_rollouts += task_rollouts
            api_calls += int(result.get("api_calls", 0))
            log_metrics({
                "evaluating/success_rate": successes / total_rollouts,
                "_step_evaluating": step + 1,
            })
            recorder.append_jsonl("evaluating_steps", {
                "step": step,
                "task_id": task_id,
                "success": bool(result["success"]),
                "successes": task_successes,
                "num_rollouts": task_rollouts,
                "success_rate": task_successes / task_rollouts,
                "rollout_successes": result.get("rollout_successes"),
                "api_calls": int(result.get("api_calls", 0)),
                "duration_s": round(elapsed, 1),
            })
            if step % 5 == 0 or step == task_pool.M - 1:
                logger.info(
                    "  evaluating step %s/%s: task %s -> %s (%.0fs)",
                    step + 1, task_pool.M, task_id,
                    "OK" if result["success"] else "FAIL", elapsed,
                )

        summary = {
            "source_run_id": run_id,
            "split": dataset_overrides["split"],
            "success": successes,
            "total": total_rollouts,
            "success_rate": successes / total_rollouts,
            "api_calls": api_calls,
            "rollouts_per_task": evaluation_rollouts,
            "skills_path": str(paths.skills_dir / "skills.json"),
            "output_dir": str(output_dir),
        }
        recorder.save_json("evaluating_result", summary)
        logger.info(
            "Evaluation complete: %s/%s success (%.2f%%)",
            successes, total_rollouts, 100 * summary["success_rate"],
        )
        return summary
    finally:
        finish_wandb()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SPG-Bandit run's final SkillBank",
    )
    parser.add_argument("--run-id", required=True, help="Existing run id under logs/ and skills/")
    parser.add_argument("--split", default=None, help="Override the saved evaluate.split")
    parser.add_argument("--rollouts", type=int, default=None, help="Override evaluation rollouts per task")
    parser.add_argument("--seed", type=int, default=None, help="Override the saved experiment seed")
    parser.add_argument("--output-dir", default=None, help="Directory for this evaluation's records")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_evaluation(
        args.run_id,
        split=args.split,
        rollouts=args.rollouts,
        seed=args.seed,
        no_wandb=args.no_wandb,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Saved evaluation result: {summary['output_dir']}")


if __name__ == "__main__":
    main()
