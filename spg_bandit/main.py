#!/usr/bin/env python3
"""SPG-Bandit experiment runner with structured data saving."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from dotenv import load_dotenv
load_dotenv()

from spg_bandit.utils.config_loader import load_config, get_param
from spg_bandit.utils.logger import setup_logger
from spg_bandit.utils.recorder import Recorder
from spg_bandit.utils.wandb import init_wandb, log_metrics, finish_wandb
from spg_bandit.modules.dataset.alfworld import ALFWorldDataset
from spg_bandit.modules.skill_evolving import SimpleAgent
from spg_bandit.modules.selector import UniformSelector, SPGBanditSelector


def build_parser():
    p = argparse.ArgumentParser(description="SPG-Bandit experiment runner")
    p.add_argument("--config", "-c", default="default")
    p.add_argument("--run_id", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-file", action="store_true")
    p.add_argument("--warmup-data", default=None,
                   help="Path to warmup data JSON. Skip task execution, load data for MIRT+MLP.")
    return p


def create_selector(name, task_pool, config):
    params = config.get(name, {})
    if name == "uniform":
        return UniformSelector()
    elif name == "spg_bandit":
        exp = config.get("experiment", {})
        return SPGBanditSelector(
            task_pool=task_pool,
            n_warm=exp.get("n_warm", 30),
            alpha=params.get("alpha", 0.1),
            tau=params.get("tau", 0.1),
            d_f=params.get("d_f", 16),
            K=params.get("K", 6),
            seed=config.get("experiment", {}).get("seed", 42),
        )
    raise ValueError(f"Unknown selector: {name}")


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed

    sel_name = config.get("selector", "uniform")
    agent_name = config.get("skill_evolving", {}).get("name", "unknown")

    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{sel_name}_{agent_name}"
    if not args.run_name:
        args.run_name = run_id

    logger = setup_logger(run_id, args.run_name, log_file_enabled=args.log_file)
    logger.info(f"Config: {args.config}")
    logger.info(f"Selector: {sel_name}")

    wandb_active = init_wandb(config, run_id, args.run_name, enabled=not args.no_wandb)

    log_base = Path(__file__).parent.parent / "logs" / run_id
    recorder = Recorder(str(log_base / "records"))
    config_path = str(log_base / "records" / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Records: {log_base / 'records'}")

    logger.info("Loading dataset...")
    dataset = ALFWorldDataset(config.get("alfworld", {}))
    task_pool = dataset.task_pool

    n_bandit = config.get("experiment", {}).get("n_bandit", 50)

    skills_dir = str(Path(__file__).parent / "modules" / "skill_evolving" / "simple_agent" / "skills" / run_id / sel_name)
    records_dir = str(log_base / sel_name / "messages")
    method = SimpleAgent(dataset, max_turns=config.get("alfworld", {}).get("max_turns", 30),
                         records_dir=records_dir)
    method.load_skills(skills_dir)
    selector = create_selector(sel_name, task_pool, config)

    n_warm_config = config.get("experiment", {}).get("n_warm", 30)
    if args.warmup_data:
        if not hasattr(selector, "load_warmup_data"):
            logger.warning("Selector %s does not support warmup loading, ignoring --warmup-data", sel_name)
            warmup_steps = selector.needs_warmup * n_warm_config
        else:
            selector.load_warmup_data(args.warmup_data, task_pool)
            warmup_steps = 0
            logger.info("Warmup task execution skipped, loaded data from %s", args.warmup_data)
    else:
        warmup_steps = n_warm_config if selector.needs_warmup else 0

    total_steps = n_bandit + warmup_steps
    if warmup_steps > 0:
        logger.info(f"  (warmup: {warmup_steps} steps)")

    success_count = 0
    step_records = []

    for step in range(total_steps):
        task_id = selector.select(task_pool)
        t0 = time.time()
        result = method.execute(task_id)
        elapsed = time.time() - t0
        method.reflect(task_id, result)
        selector.update(task_id, result)

        is_warmup = step < warmup_steps
        if not is_warmup:
            if result["success"]:
                success_count += 1
            bandit_done = step - warmup_steps + 1
            log_metrics({f"{sel_name}/success_rate": success_count / bandit_done})

        record = {
            "step": step, "selector": sel_name, "task_id": task_id,
            "success": result["success"], "api_calls": result["api_calls"],
            "duration_s": round(elapsed, 1), "is_warmup": is_warmup,
        }
        step_records.append(record)
        recorder.append_jsonl(f"{sel_name}_steps", record)

        logger.info(f"  step {step+1}/{total_steps}: task {task_id} -> "
                    f"{'OK' if result['success'] else 'FAIL'} ({elapsed:.0f}s)")

    if hasattr(selector, "get_metrics"):
        metrics = selector.get_metrics()
        if metrics:
            recorder.save_json(f"{sel_name}_spg_metrics", metrics)

    # Save warmup data for future --warmup-data runs
    if warmup_steps > 0 and hasattr(selector, "save_warmup_data"):
        warmup_path = str(log_base / "records" / f"{sel_name}_warmup_data.json")
        selector.save_warmup_data(warmup_path)
        logger.info("Warmup data saved to %s", warmup_path)

    bandit_steps = [r for r in step_records if not r["is_warmup"]]
    bandit_success = sum(1 for r in bandit_steps if r["success"])
    total_api = sum(r["api_calls"] for r in bandit_steps)

    result_entry = {
        "name": sel_name, "success": bandit_success,
        "total": len(bandit_steps), "api_calls": total_api,
    }

    recorder.save_json("comparison", {
        "run_id": run_id,
        "config": args.config,
        "results": [result_entry],
    })

    logger.info(f"\n{'='*60}")
    logger.info(f"[{sel_name}] Done: {bandit_success}/{len(bandit_steps)} "
                f"success | {total_api} API calls")

    finish_wandb()
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
