#!/usr/bin/env python3
"""Evaluate skill evolution quality — no reflection, Uniform selection, loaded skills only."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import yaml
from spg_bandit.utils.config_loader import load_config
from spg_bandit.utils.logger import setup_logger
from spg_bandit.utils.recorder import Recorder
from spg_bandit.modules.dataset.alfworld import ALFWorldDataset
from spg_bandit.modules.skill_evolving import SimpleAgent
from spg_bandit.modules.selector import UniformSelector


def build_parser():
    p = argparse.ArgumentParser(description="SPG-Bandit evaluation (no reflection)")
    p.add_argument("--config", "-c", default="default")
    p.add_argument("--run_id", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-file", action="store_true")
    p.add_argument("--skills", default=None, help="Path to pre-existing skills directory")
    p.add_argument("--label", default="eval", help="Label for this eval run")
    return p


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed

    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_eval_{args.label}"
    if not args.run_name:
        args.run_name = run_id

    logger = setup_logger(run_id, args.run_name, log_file_enabled=args.log_file)
    logger.info(f"Config: {args.config}")
    logger.info(f"EVAL mode — no reflection, Uniform selection")
    if args.skills:
        logger.info(f"Skills: {args.skills}")

    log_base = Path(__file__).parent.parent / "logs" / run_id
    recorder = Recorder(str(log_base / "records"))
    config_path = str(log_base / "records" / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("Loading dataset...")
    dataset = ALFWorldDataset(config.get("alfworld", {}))
    task_pool = dataset.task_pool

    n_bandit = config.get("experiment", {}).get("n_bandit", 50)

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation: {args.label}")
    logger.info(f"{'='*60}")

    records_dir = Path(__file__).parent.parent / "logs" / run_id / args.label / "messages"
    method = SimpleAgent(dataset, max_turns=config.get("alfworld", {}).get("max_turns", 30),
                         records_dir=str(records_dir))

    if args.skills:
        method.load_skills(args.skills)

    selector = UniformSelector()
    success_count = 0
    step_records = []

    for step in range(n_bandit):
        task_id = selector.select(task_pool)
        t0 = time.time()
        result = method.execute(task_id)
        elapsed = time.time() - t0
        if result["success"]:
            success_count += 1

        step_records.append({
            "step": step, "task_id": task_id,
            "success": result["success"], "api_calls": result["api_calls"],
            "duration_s": round(elapsed, 1),
        })

        if step % 5 == 0 or step == n_bandit - 1:
            logger.info(f"  step {step+1}/{n_bandit}: task {task_id} -> "
                        f"{'OK' if result['success'] else 'FAIL'} ({elapsed:.0f}s)")

    total_api = sum(r["api_calls"] for r in step_records)

    # Save records
    for rec in step_records:
        recorder.append_jsonl(f"{args.label}_steps", rec)
    recorder.save_json("result", {
        "label": args.label, "success": success_count,
        "total": n_bandit, "api_calls": total_api,
    })
    logger.info(f"\n  [{args.label}] Done: {success_count}/{n_bandit} "
                f"success | {total_api} API calls")
    logger.info(f"{'='*60}")
    logger.info(f"Result: {success_count}/{n_bandit}  |  {total_api} API calls")


if __name__ == "__main__":
    main()
