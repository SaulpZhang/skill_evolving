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
from spg_bandit.utils.warmup import sample_type_balanced_task_ids
from spg_bandit.modules.dataset import create_dataset
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
                    window_size=20, task_type_count=None):
    params = config.get(name, {})
    if name == "uniform":
        return UniformSelector(
            seed=config.get("experiment", {}).get("seed", 42),
        )
    elif name == "spg_bandit":
        return SPGBanditSelector(
            task_pool=task_pool,
            n_warm=n_warm,
            alpha=params.get("alpha", 0.1),
            tau=params.get("tau", 0.1),
            d_f=params.get("d_f", 16),
            seed=config.get("experiment", {}).get("seed", 42),
            warmup_ids=warmup_ids,
            window_size=window_size,
            K=params.get("K", task_type_count or 6),
        )
    raise ValueError(f"Unknown selector: {name}")


def create_skill_evolving(
    name, dataset, max_turns, records_dir=None, skill_config=None,
    selection_dataset=None,
):
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
    if selection_dataset is not None and (
        accepts_kwargs or "selection_dataset" in parameters
    ):
        kwargs["selection_dataset"] = selection_dataset
    if skill_config is not None:
        if accepts_kwargs or "config" in parameters:
            kwargs["config"] = skill_config
        elif "skill_config" in parameters:
            kwargs["skill_config"] = skill_config
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
    skill_config = dict(config.get("skill_evolving", {}) or {})
    # Pass the experiment seed into batch-oriented methods as well as the
    # selector, unless a method explicitly requests its own seed.
    skill_config.setdefault("seed", config.get("experiment", {}).get("seed", 42))
    rollouts_per_task = int(skill_config.get("rollouts_per_task", 1))
    if rollouts_per_task < 1:
        raise ValueError("skill_evolving.rollouts_per_task must be at least 1")
    evaluation_rollouts_per_task = int(
        skill_config.get("evaluation_rollouts_per_task", rollouts_per_task)
    )
    if evaluation_rollouts_per_task < 1:
        raise ValueError(
            "skill_evolving.evaluation_rollouts_per_task must be at least 1"
        )

    run_id = args.run_id or f"{sel_name}_{agent_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not args.run_name:
        args.run_name = run_id

    logger = setup_logger(run_id, args.run_name, log_file_enabled=args.log_file)
    logger.info(f"Config: {args.config}")
    logger.info(f"Selector: {sel_name}")

    dataset_setting = config.get("dataset", "alfworld")
    if isinstance(dataset_setting, str):
        dataset_name = dataset_setting
        dataset_params = {}
    elif isinstance(dataset_setting, dict):
        dataset_name = dataset_setting.get("name", "alfworld")
        dataset_params = dict(dataset_setting.get("params", {}) or {})
        # Also accept flat dataset options for concise configs.  ``name`` and
        # ``params`` are registry metadata, not constructor arguments.
        for key, value in dataset_setting.items():
            if key not in {"name", "params"}:
                dataset_params.setdefault(key, value)
    else:
        raise ValueError("dataset must be a name or a mapping with a 'name' field")
    logger.info(f"Dataset: {dataset_name}")

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
        cfg = dict(dataset_params)
        cfg.setdefault("task_types", "all")
        cfg.setdefault("split", "valid_seen")
        cfg.update({
            "embedding_model": config.get(
                "embedding_model", cfg.get("embedding_model", "all-MiniLM-L6-v2")
            ),
            "embedding_type": config.get("embedding_type", cfg.get("embedding_type", "local")),
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

    logger.info("Loading evolve dataset...")
    evo_cfg = config.get("evolve", {})
    evo_overrides = dict(evo_cfg) if isinstance(evo_cfg, dict) else {}
    evo_overrides.setdefault("split", "valid_seen")
    evo_dataset = create_dataset(dataset_name, _make_cfg(evo_overrides))
    evo_pool = evo_dataset.task_pool

    logger.info("Loading evaluate dataset...")
    eva_cfg = config.get("evaluate", {})
    eva_overrides = dict(eva_cfg) if isinstance(eva_cfg, dict) else {}
    eva_overrides.setdefault("split", "valid_seen")
    eva_dataset = create_dataset(dataset_name, _make_cfg(eva_overrides))
    eva_pool = eva_dataset.task_pool

    # SkillOpt uses a fixed, non-selector pool for its validation gate.  Keep
    # this split separate from both the evolve and evaluate pools so neither
    # selector receives gate tasks as evidence and the final test remains
    # untouched.  Only dataset fields are forwarded; selection controls such
    # as ``selection_size`` stay in the SkillOpt config.
    skill_selection_dataset = None
    if agent_name == "skillopt":
        selection_cfg = config.get("skill_selection", {})
        if not isinstance(selection_cfg, dict):
            raise ValueError("skill_selection must be a mapping")
        selection_split = selection_cfg.get("split", "valid_seen")
        logger.info("Loading SkillOpt selection dataset (%s)...", selection_split)
        skill_selection_dataset = create_dataset(
            dataset_name,
            _make_cfg({"split": selection_split}),
        )
        # Force pool construction here so the fixed ids are known before the
        # agent starts consuming target-model calls.
        _ = skill_selection_dataset.task_pool

    n_bandit = evo_pool.M
    n_eva = eva_pool.M
    max_turns = config.get("max_turns", 51)

    warmup_ids = []
    n_warm = 0
    window_size = 20
    if sel_name == "spg_bandit":
        spg_cfg = config.get("spg_bandit", {})
        warmup_ratio = spg_cfg.get("warmup_ratio", 0.3)
        if not 0 < warmup_ratio < 1:
            raise ValueError("spg_bandit.warmup_ratio must be between 0 and 1")
        n_warm = round(evo_pool.M * warmup_ratio)
        if n_warm == 0:
            raise ValueError("Warmup ratio produces zero steps; increase warmup_ratio")
        window_size = spg_cfg.get("window_size", min(20, n_warm))
        if not 0 < window_size <= n_warm:
            raise ValueError("spg_bandit.window_size must be in [1, n_warm]")
        warmup_ids = sample_type_balanced_task_ids(evo_pool, n_warm, random)
        logger.info("Warmup: %s type-balanced samples from evolve pool", n_warm)

    skills_dir = str(Path(__file__).parent.parent / "skills" / run_id)
    records_dir = str(log_base / sel_name / "messages")
    method = create_skill_evolving(
        agent_name, evo_dataset, max_turns, records_dir, skill_config,
        selection_dataset=skill_selection_dataset,
    )
    method.load_skills(skills_dir)
    selector = create_selector(
        sel_name, evo_pool, config, warmup_ids=warmup_ids, n_warm=n_warm,
        window_size=window_size, task_type_count=len(evo_pool.task_types),
    )

    if selector.needs_warmup:
        n_bandit = evo_pool.M - n_warm

    # A fixed, type-balanced probe subset measures the ability-profile change
    # caused by a SkillRL/SkillOpt update.  It is never reflected on and is
    # therefore evaluation evidence rather than additional training data.
    probe_ids = []
    if sel_name == "spg_bandit":
        probe_size = int(spg_cfg.get("probe_size", 0))
        if probe_size:
            if probe_size < 1 or probe_size > evo_pool.M:
                raise ValueError("spg_bandit.probe_size must be in [1, evolve pool size]")
            probe_ids = sample_type_balanced_task_ids(evo_pool, probe_size, random)
            logger.info("SPG skill-gain probes: %s fixed type-balanced tasks", len(probe_ids))

    logger.info(f"Warmup pool: evolve pool ({evo_pool.M} tasks), {n_warm} steps")
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
    rollout_count = 0
    step_records = []
    ckpt_interval = 30
    start_step = 0

    # Resume: load checkpoint if --resume
    ckpt_path = str(log_base / "records" / "checkpoint.json")
    if args.resume:
        if not args.run_id:
            logger.error("--resume requires --run_id")
            return

    def _run_skill_gain_probes():
        """Execute the fixed probes without letting them modify the SkillBank."""
        observations = []
        for probe_task_id in probe_ids:
            probe_result = method.execute(probe_task_id, num_rollouts=rollouts_per_task)
            observations.append({"task_id": probe_task_id, **probe_result})
        return observations

    def _commit_skill_gain(events, anchor_profile, before_profile):
        """Train SPG on post-reflection probe gain for each completed batch."""
        if not events or not isinstance(selector, SPGBanditSelector):
            return
        for event in events:
            if not isinstance(event, dict) or not event.get("skill_update_completed"):
                continue
            updated = bool(event.get("skill_updated", False))
            measured_before = None
            measured_after = None
            if updated and probe_ids and anchor_profile is not None:
                after_profile = selector.estimate_profile_from_results(
                    _run_skill_gain_probes(), base_profile=anchor_profile,
                )
                measured_before = before_profile.tolist()
                measured_after = after_profile.tolist()
                committed = selector.commit_skill_update(
                    before_profile, after_profile, updated=True,
                )
            else:
                committed = selector.commit_skill_update(
                    anchor_profile if anchor_profile is not None else selector.get_profile(),
                    anchor_profile if anchor_profile is not None else selector.get_profile(),
                    updated=False,
                )
            record = {
                "event": "skill_gain_label",
                "update": event,
                "committed": committed,
                "probe_task_ids": probe_ids if updated else [],
                "profile_before": measured_before,
                "profile_after": measured_after,
            }
            recorder.append_jsonl("spg_skill_gain", record)
            log_metrics({
                **{f"spg/skill_gain_dim_{i}": value for i, value in enumerate(committed["delta"])},
                "spg/skill_gain_batch_size": committed["committed"],
                "_step_evolving": step + 1,
            })
    if args.resume:
        try:
            import json
            ckpt = json.load(open(ckpt_path))
            start_step = ckpt["step"]
            success_count = ckpt["success_count"]
            rollout_count = ckpt.get("rollout_count", start_step)
            step_records = ckpt["step_records"]
            warmup_steps = ckpt["warmup_steps"]
            n_bandit = ckpt.get("n_bandit", n_bandit)
            n_eva = ckpt.get("n_eva", n_eva)
            if hasattr(selector, "load_checkpoint"):
                selector.load_checkpoint(ckpt.get("selector", {}))
            method.load_checkpoint(ckpt.get("skill_evolving", {}))
            logger.info(f"Resumed from step {start_step}/{total_steps}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return

    def _save_ckpt(st, sw, sb, sr, phase="", path=None):
        try:
            data = {
                "step": st, "total_steps": total_steps,
                "warmup_steps": sw, "success_count": sb,
                "rollout_count": rollout_count,
                "step_records": sr, "n_bandit": n_bandit,
                "n_eva": n_eva, "phase": phase,
            }
            if hasattr(selector, "save_checkpoint"):
                data["selector"] = selector.save_checkpoint()
            data["skill_evolving"] = method.save_checkpoint()
            with open(path or ckpt_path, "w") as f:
                import json
                json.dump(data, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Checkpoint saved at step {st} ({phase})")
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")

    try:
        for step in range(start_step, total_steps):
            is_warmup = step < warmup_steps
            task_id = selector.select(evo_pool)
            t0 = time.time()
            result = method.execute(task_id, num_rollouts=rollouts_per_task)
            elapsed = time.time() - t0
            selector.update(task_id, result)
            # The selected-task outcome updates the current MIRT state.  If a
            # skill update is due, probe that same state before reflection;
            # probe again after reflection and use the profile difference as
            # the MLP supervision signal.
            anchor_profile = None
            before_profile = None
            if (
                probe_ids and isinstance(selector, SPGBanditSelector)
                and selector._warmup_ready
                and method.will_update_after_reflect(task_id, result)
            ):
                anchor_profile = selector.get_profile()
                before_profile = selector.estimate_profile_from_results(
                    _run_skill_gain_probes(), base_profile=anchor_profile,
                )
            update_events = method.reflect(task_id, result) or []
            _commit_skill_gain(update_events, anchor_profile, before_profile)

            successes = int(result.get("successes", int(bool(result["success"]))))
            num_rollouts = int(result.get("num_rollouts", 1))
            success_count += successes
            rollout_count += num_rollouts
            bandit_done = step + 1
            log_metrics({"evolving/success_rate": success_count / rollout_count, "_step_evolving": bandit_done})

            record = {
                "step": step, "selector": sel_name, "task_id": task_id,
                "success": result["success"], "api_calls": result["api_calls"],
                "successes": successes, "num_rollouts": num_rollouts,
                "success_rate": successes / num_rollouts,
                "rollout_successes": result.get("rollout_successes"),
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

    # Batch optimizers (SkillOpt) may have a partial minibatch left after the
    # final selected task.  Flush it before saving metrics or evaluating the
    # resulting skill.
    final_anchor_profile = None
    final_before_profile = None
    if (
        probe_ids and isinstance(selector, SPGBanditSelector)
        and selector._warmup_ready
        and method.get_usage().get("buffered_rollouts", 0) > 0
    ):
        final_anchor_profile = selector.get_profile()
        final_before_profile = selector.estimate_profile_from_results(
            _run_skill_gain_probes(), base_profile=final_anchor_profile,
        )
    final_events = method.finalize() or []
    _commit_skill_gain(final_events, final_anchor_profile, final_before_profile)

    # Save only after `finalize`: a final partial SkillOpt minibatch can
    # modify both the active skill and buffered optimizer state.
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
        eval_skill_config = dict(skill_config)
        eval_skill_config["enable_dynamic_update"] = False
        if "evaluation_temperature" in skill_config:
            eval_skill_config["temperature"] = skill_config["evaluation_temperature"]
        if "evaluation_max_tokens" in skill_config:
            eval_skill_config["max_tokens"] = skill_config["evaluation_max_tokens"]
        eval_method = create_skill_evolving(
            agent_name,
            eva_dataset,
            max_turns,
            records_dir=str(log_base / sel_name / "evaluating_messages"),
            skill_config=eval_skill_config,
        )
        eval_method.load_skills(skills_dir)
        evaluating_selector = UniformSelector(seed=seed)
        evaluating_success = 0
        evaluating_rollouts = 0
        evaluating_records = []

        eva_start = 0
        eval_ckpt_path = str(log_base / "records" / "evaluating_checkpoint.json")

        try:
            for step in range(eva_start, n_eva):
                task_id = evaluating_selector.select(eva_pool)
                t0 = time.time()
                result = eval_method.execute(
                    task_id, num_rollouts=evaluation_rollouts_per_task,
                )
                elapsed = time.time() - t0
                successes = int(result.get("successes", int(bool(result["success"]))))
                num_rollouts = int(result.get("num_rollouts", 1))
                evaluating_success += successes
                evaluating_rollouts += num_rollouts
                log_metrics({"evaluating/success_rate": evaluating_success / evaluating_rollouts, "_step_evaluating": step + 1})
                evaluating_record = {
                    "step": step, "task_id": task_id,
                    "success": result["success"], "api_calls": result["api_calls"],
                    "successes": successes, "num_rollouts": num_rollouts,
                    "success_rate": successes / num_rollouts,
                    "rollout_successes": result.get("rollout_successes"),
                    "duration_s": round(elapsed, 1),
                }
                evaluating_records.append(evaluating_record)
                # Emit and persist every evaluation task: evaluations can take
                # many minutes per rollout, so five-step batching hides liveness
                # and loses all task records if the job is interrupted early.
                recorder.append_jsonl("evaluating_steps", evaluating_record)
                logger.info(
                    f"  evaluating step {step + 1}/{n_eva}: task {task_id} -> "
                    f"{'OK' if result['success'] else 'FAIL'} ({elapsed:.1f}s); "
                    f"running SR={evaluating_success / evaluating_rollouts:.3f}"
                )
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
        recorder.save_json("evaluating_result", {
            "label": sel_name, "success": evaluating_success,
            "total": evaluating_rollouts, "api_calls": evaluating_api,
        })
        logger.info(f"\n  [evaluating] Done: {evaluating_success}/{evaluating_rollouts} "
                    f"success | {evaluating_api} API calls")

    bandit_steps = [r for r in step_records if not r["is_warmup"]]
    bandit_success = sum(int(r.get("successes", int(bool(r["success"])))) for r in bandit_steps)
    bandit_rollouts = sum(int(r.get("num_rollouts", 1)) for r in bandit_steps)
    total_api = sum(r["api_calls"] for r in bandit_steps)

    result_entry = {
        "name": sel_name, "success": bandit_success,
        "total": bandit_rollouts, "api_calls": total_api,
    }

    recorder.save_json("comparison", {
        "run_id": run_id,
        "config": args.config,
        "results": [result_entry],
    })

    logger.info(f"\n{'='*60}")
    logger.info(f"[{sel_name}] Done: {bandit_success}/{bandit_rollouts} "
                f"success | {total_api} API calls")

    finish_wandb()
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
