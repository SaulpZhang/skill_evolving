#!/usr/bin/env python3
"""Evaluate selector performance — no reflection, just task execution."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from spg_bandit.utils.config_loader import load_config
from spg_bandit.utils.logger import setup_logger
from spg_bandit.modules.dataset.alfworld import ALFWorldDataset
from spg_bandit.modules.skill_evolving import SimpleAgent
from spg_bandit.modules.selector import UniformSelector, SPGBanditSelector


def build_parser():
    p = argparse.ArgumentParser(description="SPG-Bandit evaluation (no reflection)")
    p.add_argument("--config", "-c", default="default")
    p.add_argument("--run_id", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-file", action="store_true")
    p.add_argument("--skills", default=None, help="Path to pre-existing skills directory")
    return p


def create_selector(name, task_pool, config):
    params = config.get("selectors", {}).get(name, {})
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
            seed=config.get("experiment", {}).get("seed", 42),
        )
    raise ValueError(f"Unknown selector: {name}")


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed

    selectors_enabled = config.get("selectors", {}).get("enabled", ["uniform"])
    sel_tag = "_".join(selectors_enabled)
    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_eval_{sel_tag}"
    if not args.run_name:
        args.run_name = run_id

    logger = setup_logger(run_id, args.run_name, log_file_enabled=args.log_file)
    logger.info(f"Config: {args.config}")
    logger.info(f"EVAL mode — no reflection")
    logger.info(f"Selectors: {selectors_enabled}")

    logger.info("Loading dataset...")
    dataset = ALFWorldDataset(config.get("alfworld", {}))
    task_pool = dataset.task_pool

    enabled = selectors_enabled
    n_bandit = config.get("experiment", {}).get("n_bandit", 50)
    results = []

    for sel_name in enabled:
        logger.info(f"\n{'='*60}")
        logger.info(f"Selector: {sel_name}")
        logger.info(f"{'='*60}")

        records_dir = Path(__file__).parent.parent / "logs" / run_id / sel_name / "messages"
        method = SimpleAgent(dataset, max_turns=config.get("alfworld", {}).get("max_turns", 30),
                             records_dir=str(records_dir))

        # Load pre-existing skills if provided
        if args.skills:
            method.load_skills(args.skills)
            logger.info(f"  Skills loaded from: {args.skills}")

        selector = UniformSelector()

        success_count = 0
        step_records = []

        for step in range(n_bandit):
            task_id = selector.select(task_pool)
            t0 = time.time()
            result = method.execute(task_id)
            elapsed = time.time() - t0
            # No reflect() call — eval mode
            selector.update(task_id, result)

            is_warmup = selector.needs_warmup and step < config.get("experiment", {}).get("n_warm", 30)
            if not is_warmup and result["success"]:
                success_count += 1

            step_records.append({
                "step": step, "selector": sel_name, "task_id": task_id,
                "success": result["success"], "api_calls": result["api_calls"],
                "duration_s": round(elapsed, 1), "is_warmup": is_warmup,
            })

            if step % 5 == 0 or step == total_steps - 1:
                logger.info(f"  step {step+1}/{total_steps}: task {task_id} -> "
                           f"{'OK' if result['success'] else 'FAIL'} ({elapsed:.0f}s)")

        bandit_steps = [r for r in step_records if not r["is_warmup"]]
        bandit_success = sum(1 for r in bandit_steps if r["success"])
        total_api = sum(r["api_calls"] for r in bandit_steps)

        results.append({
            "name": sel_name, "success": bandit_success,
            "total": len(bandit_steps), "api_calls": total_api,
        })

        logger.info(f"  [{sel_name}] Done: {bandit_success}/{len(bandit_steps)} "
                    f"success | {total_api} API calls")

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"{'Selector':20s}  {'Success':>10s}  {'API Calls':>10s}")
    logger.info("-" * 45)
    for r in results:
        logger.info(f"{r['name']:20s}  {r['success']}/{r['total']:>7d}        {r['api_calls']:>6d}")


if __name__ == "__main__":
    main()
