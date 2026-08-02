#!/usr/bin/env python3
"""SPG-Bandit experiment runner with structured data saving."""

import argparse
import importlib
import inspect
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from json import JSONEncoder

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return {"__numpy__": True, "dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tobytes().hex()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

load_dotenv()

from spg_bandit.utils.config_loader import load_config, resolve_config_path
from spg_bandit.utils.logger import setup_logger
from spg_bandit.utils.recorder import Recorder
from spg_bandit.utils.wandb import init_wandb, log_metrics, finish_wandb
from spg_bandit.modules.dataset.alfworld import ALFWorldDataset
from spg_bandit.modules.skill_evolving import BaseSkillEvolving, SimpleAgent
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
    p.add_argument("--evaluating", action="store_true",
                   help="Run evaluation after main experiment (Uniform, no reflection)")
    p.add_argument("--resume", action="store_true",
                   help="Resume a previous run. Must also pass --run_id.")
    return p


def create_selector(name, task_pool, config, warmup_ids=None, n_warm=0,
                    warmup_pool=None):
    params = config.get(name, {})
    if name == "uniform":
        return UniformSelector()
    elif name == "spg_bandit":
        return SPGBanditSelector(
            task_pool=task_pool,
            n_warm=n_warm,
            alpha=params.get("alpha", 0.1),
            tau=params.get("tau", 0.1),
            d_f=params.get("d_f", 16),
            K=params.get("K", 6),
            seed=config.get("experiment", {}).get("seed", 42),
            warmup_ids=warmup_ids,
            warmup_pool=warmup_pool,
        )
    raise ValueError(f"Unknown selector: {name}")


def create_skill_evolving(name, dataset, max_turns, records_dir=None):
    """Instantiate a registered skill-evolving implementation by its package name."""
    if name == "simple_agent":
        implementation = SimpleAgent
    else:
        module = importlib.import_module(f"spg_bandit.modules.skill_evolving.{name}")
        implementations = [
            value for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, BaseSkillEvolving)
            and value is not BaseSkillEvolving
        ]
        if len(implementations) != 1:
            raise ValueError(
                f"Skill evolving module '{name}' must export exactly one "
                "BaseSkillEvolving implementation"
            )
        implementation = implementations[0]

    parameters = inspect.signature(implementation).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    kwargs = {}
    if accepts_kwargs or "max_turns" in parameters:
        kwargs["max_turns"] = max_turns
    if records_dir is not None and (accepts_kwargs or "records_dir" in parameters):
        kwargs["records_dir"] = records_dir
    return implementation(dataset, **kwargs)


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    config_source = resolve_config_path(args.config)
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed
    seed = config.get("experiment", {}).get("seed", 42)
    random.seed(seed)

    sel_name = config.get("selector", "uniform")
    agent_name = config.get("skill_evolving", {}).get("name", "unknown")

    run_id = args.run_id or f"{sel_name}_{agent_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not args.run_name:
        args.run_name = run_id

    logger = setup_logger(run_id, args.run_name, log_file_enabled=args.log_file)
    logger.info(f"Config: {args.config}")
    logger.info(f"Selector: {sel_name}")

    wandb_active = init_wandb(config, run_id, args.run_name, enabled=not args.no_wandb, resume=args.resume)

    log_base = Path(__file__).parent.parent / "logs" / run_id
    recorder = Recorder(str(log_base / "records"))
    config_path = log_base / "records" / "config.yaml"
    if config_source is not None:
        # Keep the exact user-authored YAML rather than serializing a normalized
        # runtime dictionary, so unknown future fields and formatting are retained.
        shutil.copyfile(config_source, config_path)
    else:
        logger.warning("No config source file found; no config.yaml was copied")
    logger.info(f"Records: {log_base / 'records'}")

    def _make_cfg(overrides):
        cfg = {
            "embedding_model": config.get("embedding_model", "all-MiniLM-L6-v2"),
            "embedding_type": config.get("embedding_type", "local"),
            "max_turns": config.get("max_turns", 51),
            "task_types": "all",
            "split": "valid_seen",
        }
        cfg.update(overrides)
        return cfg

    logger.info("Loading evolve dataset...")
    evo_cfg = config.get("evolve", {})
    evo_dataset = ALFWorldDataset(_make_cfg({
        "split": evo_cfg.get("split", "valid_seen"),
    }))
    evo_pool = evo_dataset.task_pool

    logger.info("Loading evaluate dataset...")
    eva_cfg = config.get("evaluate", {})
    eva_dataset = ALFWorldDataset(_make_cfg({
        "split": eva_cfg.get("split", "valid_seen"),
    }))
    eva_pool = eva_dataset.task_pool

    n_bandit = evo_pool.M
    n_eva = eva_pool.M
    max_turns = config.get("max_turns", 51)

    warmup_dataset = None
    warmup_pool = None
    warmup_ids = []
    n_warm = 0
    if sel_name == "spg_bandit":
        warmup_cfg = config.get("warmup", {})
        warmup_ratio = config.get("spg_bandit", {}).get("warmup_ratio", 0.3)
        if not 0 < warmup_ratio < 1:
            raise ValueError("spg_bandit.warmup_ratio must be between 0 and 1")
        n_warm = round(evo_pool.M * warmup_ratio)
        if n_warm == 0:
            raise ValueError("Warmup ratio produces zero steps; increase warmup_ratio")

        logger.info("Loading warmup dataset...")
        warmup_dataset = ALFWorldDataset(_make_cfg({
            "split": warmup_cfg.get("split", evo_cfg.get("split", "valid_seen")),
        }))
        warmup_pool = warmup_dataset.task_pool
        if warmup_pool.M == 0:
            raise ValueError("Warmup requested but the warmup dataset has no tasks")

        # Allocate warmup evenly by type and repeat anchor tasks so MIRT receives
        # multiple observations for each fitted item.
        from collections import defaultdict
        type_to_ids = defaultdict(list)
        for m in warmup_pool.metadata:
            type_to_ids[m["dim"]].append(m["id"])
        raw = {d: len(ids) / warmup_pool.M * n_warm for d, ids in type_to_ids.items()}
        alloc = {d: int(raw[d]) for d in raw}
        remainder = n_warm - sum(alloc.values())
        for d in sorted(raw, key=lambda d: raw[d] - int(raw[d]), reverse=True):
            if remainder <= 0:
                break
            alloc[d] += 1
            remainder -= 1
        for d in sorted(type_to_ids):
            pool_ids = type_to_ids[d]
            anchor_count = min(len(pool_ids), max(1, alloc[d] // 2))
            anchors = pool_ids[:anchor_count]
            warmup_ids.extend((anchors * (alloc[d] // len(anchors) + 1))[:alloc[d]])
        random.shuffle(warmup_ids)
        logger.info("Warmup: %s tasks, type distribution: %s", len(warmup_ids),
                    {d: alloc[d] for d in sorted(alloc)})

    skills_dir = str(Path(__file__).parent.parent / "skills" / run_id)
    records_dir = str(log_base / sel_name / "messages")
    method = create_skill_evolving(agent_name, evo_dataset, max_turns, records_dir)
    warmup_method = (
        create_skill_evolving(agent_name, warmup_dataset, max_turns, records_dir)
        if warmup_dataset is not None else None
    )
    method.load_skills(skills_dir)
    if warmup_method is not None:
        warmup_method.load_skills(skills_dir)
    selector = create_selector(
        sel_name, evo_pool, config, warmup_ids=warmup_ids, n_warm=n_warm,
        warmup_pool=warmup_pool,
    )

    if selector.needs_warmup:
        n_bandit = evo_pool.M - n_warm

    logger.info(f"Warmup pool: {warmup_pool.M if warmup_pool else 0} tasks, {n_warm} steps")
    logger.info(f"Evolve pool: {evo_pool.M} tasks, {n_bandit} steps")
    logger.info(f"Eval pool: {eva_pool.M} tasks, {n_eva} steps")

    if args.warmup_data:
        if not hasattr(selector, "load_warmup_data"):
            logger.warning("Selector %s does not support warmup loading, ignoring --warmup-data", sel_name)
            warmup_steps = selector.needs_warmup * n_warm
        else:
            selector.load_warmup_data(args.warmup_data)
            warmup_steps = 0
            logger.info("Warmup task execution skipped, loaded data from %s", args.warmup_data)
    else:
        warmup_steps = n_warm if selector.needs_warmup else 0

    total_steps = n_bandit + warmup_steps
    if warmup_steps > 0:
        logger.info(f"  (warmup: {warmup_steps} steps)")

    success_count = 0
    step_records = []
    ckpt_interval = 30
    start_step = 0

    # Resume: load checkpoint if --resume
    ckpt_path = str(log_base / "records" / "checkpoint.json")
    if args.resume:
        if not args.run_id:
            logger.error("--resume requires --run_id")
            return
        try:
            import json
            ckpt = json.load(open(ckpt_path))
            start_step = ckpt["step"]
            success_count = ckpt["success_count"]
            step_records = ckpt["step_records"]
            warmup_steps = ckpt["warmup_steps"]
            n_bandit = ckpt.get("n_bandit", n_bandit)
            n_eva = ckpt.get("n_eva", n_eva)
            if hasattr(selector, "load_checkpoint"):
                selector.load_checkpoint(ckpt.get("selector", {}))
            logger.info(f"Resumed from step {start_step}/{total_steps}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return

    def _save_ckpt(st, sw, sb, sr, phase="", path=None):
        try:
            data = {
                "step": st, "total_steps": total_steps,
                "warmup_steps": sw, "success_count": sb,
                "step_records": sr, "n_bandit": n_bandit,
                "n_eva": n_eva, "phase": phase,
            }
            if hasattr(selector, "save_checkpoint"):
                data["selector"] = selector.save_checkpoint()
            with open(path or ckpt_path, "w") as f:
                import json
                json.dump(data, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Checkpoint saved at step {st} ({phase})")
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")

    try:
        for step in range(start_step, total_steps):
            is_warmup = step < warmup_steps
            pool = warmup_pool if is_warmup else evo_pool
            active_method = warmup_method if is_warmup else method
            task_id = selector.select(pool)
            t0 = time.time()
            result = active_method.execute(task_id)
            elapsed = time.time() - t0
            active_method.reflect(task_id, result)
            selector.update(task_id, result)

            if result["success"]:
                success_count += 1
            bandit_done = step + 1
            log_metrics({"evolving/success_rate": success_count / bandit_done, "_step_evolving": bandit_done})

            record = {
                "step": step, "selector": sel_name, "task_id": task_id,
                "success": result["success"], "api_calls": result["api_calls"],
                "duration_s": round(elapsed, 1), "is_warmup": is_warmup,
            }
            step_records.append(record)
            recorder.append_jsonl(f"{sel_name}_steps", record)

            # Checkpoint: every 30 steps + warmup end
            if (step + 1) % ckpt_interval == 0:
                _save_ckpt(step + 1, warmup_steps, success_count, step_records, "interval")
            if warmup_steps > 0 and step == warmup_steps - 1:
                _save_ckpt(step + 1, warmup_steps, success_count, step_records, "warmup_end")

    except Exception as e:
        _save_ckpt(step, warmup_steps, success_count, step_records, "error")
        logger.error(f"Experiment interrupted at step {step}: {e}")
        finish_wandb()
        raise

    _save_ckpt(total_steps, warmup_steps, success_count, step_records, "evolving_end")

    if hasattr(selector, "get_metrics"):
        metrics = selector.get_metrics()
        if metrics:
            recorder.save_json(f"{sel_name}_spg_metrics", metrics)

    # Save warmup data for future --warmup-data runs
    if warmup_steps > 0 and hasattr(selector, "save_warmup_data"):
        warmup_path = str(log_base / "records" / f"{sel_name}_warmup_data.json")
        selector.save_warmup_data(warmup_path)
        logger.info("Warmup data saved to %s", warmup_path)

    # ── Evaluating phase ────────────────────────────────────────────────────
    if args.evaluating:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: Uniform, no reflection")
        logger.info(f"{'='*60}")

        method.reset()
        eval_method = create_skill_evolving(agent_name, eva_dataset, max_turns)
        eval_method.load_skills(skills_dir)
        evaluating_selector = UniformSelector()
        evaluating_success = 0
        evaluating_records = []

        eva_start = 0
        eval_ckpt_path = str(log_base / "records" / "evaluating_checkpoint.json")

        try:
            for step in range(eva_start, n_eva):
                task_id = evaluating_selector.select(eva_pool)
                t0 = time.time()
                result = eval_method.execute(task_id)
                elapsed = time.time() - t0
                if result["success"]:
                    evaluating_success += 1
                log_metrics({"evaluating/success_rate": evaluating_success / (step + 1), "_step_evaluating": step + 1})
                evaluating_records.append({
                    "step": step, "task_id": task_id,
                    "success": result["success"], "api_calls": result["api_calls"],
                    "duration_s": round(elapsed, 1),
                })
                if step % 5 == 0 or step == n_eva - 1:
                    logger.info(f"  evaluating step {step+1}/{n_eva}: task {task_id} -> "
                                f"{'OK' if result['success'] else 'FAIL'} ({elapsed:.0f}s)")
                if (step + 1) % ckpt_interval == 0:
                    _save_ckpt(step + 1, warmup_steps, success_count, step_records,
                               "eval_interval", path=eval_ckpt_path)
        except Exception as e:
            _save_ckpt(step, warmup_steps, success_count, step_records,
                       "eval_error", path=eval_ckpt_path)
            logger.error(f"Evaluating interrupted at step {step}: {e}")
            finish_wandb()
            raise

        _save_ckpt(n_eva, warmup_steps, success_count, step_records,
                   "eval_end", path=eval_ckpt_path)
        evaluating_api = sum(r["api_calls"] for r in evaluating_records)
        for rec in evaluating_records:
            recorder.append_jsonl("evaluating_steps", rec)
        recorder.save_json("evaluating_result", {
            "label": sel_name, "success": evaluating_success,
            "total": n_eva, "api_calls": evaluating_api,
        })
        logger.info(f"\n  [evaluating] Done: {evaluating_success}/{n_eva} "
                    f"success | {evaluating_api} API calls")

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
