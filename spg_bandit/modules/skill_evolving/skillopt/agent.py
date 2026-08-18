"""SkillOpt-backed skill evolution inside the SPG-Bandit runner.

The upstream SkillOpt stages are used for trajectory reflection, patch
aggregation, edit selection, update, and validation-gate decisions.  The
runner still owns task selection and environment execution, which is the
important boundary for comparing SPG-Bandit with uniform selection.

Only the vendored package in ``resource/skillopt`` is imported at runtime.
The upstream checkout under ``docs`` is deliberately not part of this
adapter's import path.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from spg_bandit.modules.skill_evolving.simple_agent import SimpleAgent


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RESOURCE_ROOT = _PROJECT_ROOT / "resource"
_SKILLOPT_ROOT = _RESOURCE_ROOT / "skillopt"


def _bootstrap_vendored_skillopt() -> None:
    """Put the vendored source package first, without relying on ``docs``.

    A descriptive error is preferable to silently importing a globally
    installed SkillOpt version: prompt and API changes between versions can
    otherwise make experiments irreproducible.
    """
    if not (_SKILLOPT_ROOT / "__init__.py").is_file():
        raise ImportError(
            "SkillOpt runtime resources are missing: "
            f"{_SKILLOPT_ROOT}. Copy SkillOpt into resource/skillopt."
        )
    loaded = sys.modules.get("skillopt")
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin:
            origin_path = Path(origin).resolve()
            try:
                origin_path.relative_to(_SKILLOPT_ROOT)
            except ValueError as exc:
                raise ImportError(
                    "A different 'skillopt' package is already imported at "
                    f"{origin_path}; the SPG runtime requires the vendored "
                    f"package at {_SKILLOPT_ROOT}."
                ) from exc
    resource_text = str(_RESOURCE_ROOT)
    if resource_text not in sys.path:
        sys.path.insert(0, resource_text)


_bootstrap_vendored_skillopt()

from skillopt.evaluation.gate import evaluate_gate  # noqa: E402
from skillopt.gradient.aggregate import merge_patches  # noqa: E402
from skillopt.gradient.reflect import run_minibatch_reflect  # noqa: E402
from skillopt.model import (  # noqa: E402
    configure_openai_compatible,
    get_optimizer_backend,
    get_token_summary,
    set_optimizer_backend,
)
from skillopt.optimizer.clip import rank_and_select  # noqa: E402
from skillopt.optimizer.skill import apply_patch_with_report  # noqa: E402
from skillopt.optimizer.update_modes import get_payload_items  # noqa: E402
from skillopt.utils.scoring import compute_score  # noqa: E402


class SkillOptAgent(SimpleAgent):
    """Use SkillOpt's batch reflection pipeline with SPG task selection.

    ``dataset`` is the evolve pool used for target rollouts.  The optional
    ``selection_dataset`` is a fixed held-out pool used only by the
    validation gate; it is never sent to the selector or to the reflection
    buffer.  This keeps the comparison between selectors scientifically
    meaningful while avoiding evaluation-set leakage.
    """

    def __init__(
        self,
        dataset,
        max_turns: int = 30,
        records_dir: str | None = None,
        config: dict[str, Any] | None = None,
        selection_dataset=None,
    ):
        super().__init__(dataset, max_turns=max_turns, records_dir=records_dir)
        self._config = dict(config or {})
        self._selection_dataset = selection_dataset
        # Evaluation instances have no selection dataset and should execute
        # the target only; they must not create a second training evidence
        # stream under a shared ``logs/skillopt`` directory.
        self._collect_evidence = selection_dataset is not None
        self._active_skill = ""
        self._best_skill = ""
        self._current_score = 0.0
        self._best_score = 0.0
        self._best_step = 0
        self._global_step = 0
        self._gate_mode = False
        self._pending_triplets: dict[int, list[dict[str, Any]]] = {}
        self._evidence: list[dict[str, Any]] = []
        self._rejected_context: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._rollout_index = 0
        self._optimizer_updates = 0

        self._batch_size = int(self._config.get("batch_size", 8))
        self._reflection_minibatch_size = int(
            self._config.get("reflection_minibatch_size", self._batch_size)
        )
        self._edit_budget = int(self._config.get("edit_budget", 4))
        self._merge_batch_size = int(self._config.get("merge_batch_size", 8))
        self._reflection_workers = int(self._config.get("reflection_workers", 1))
        self._failure_only = bool(self._config.get("failure_only", False))
        self._update_mode = str(self._config.get("update_mode", "patch"))
        self._seed = int(self._config.get("seed", 42))
        self._selection_size = int(self._config.get("selection_size", 24))
        self._selection_rollouts = int(self._config.get("selection_rollouts", 1))
        # Candidate selection is a validation decision, not exploration.  A
        # fixed temperature makes the current/candidate comparison stable.
        self._gate_temperature = float(self._config.get("gate_temperature", 0.0))
        self._gate_metric = str(self._config.get("gate_metric", "hard"))
        self._mixed_weight = float(self._config.get("mixed_weight", 0.5))
        self._use_semantic_density = bool(
            self._config.get("use_semantic_density", False)
        )
        self._semantic_density_weight = float(
            self._config.get("semantic_density_weight", 0.05)
        )
        self._skill_aware_reflection = bool(
            self._config.get("skill_aware_reflection", False)
        )

        if self._batch_size < 1:
            raise ValueError("skill_evolving.batch_size must be at least 1")
        if self._reflection_minibatch_size < 1:
            raise ValueError(
                "skill_evolving.reflection_minibatch_size must be at least 1"
            )
        if self._edit_budget < 1:
            raise ValueError("skill_evolving.edit_budget must be at least 1")
        if self._merge_batch_size < 1:
            raise ValueError("skill_evolving.merge_batch_size must be at least 1")
        if self._reflection_workers < 1:
            raise ValueError("skill_evolving.reflection_workers must be at least 1")
        if self._selection_size < 0:
            raise ValueError("skill_evolving.selection_size must be non-negative")
        if self._selection_rollouts < 1:
            raise ValueError("skill_evolving.selection_rollouts must be at least 1")
        if self._update_mode.strip().lower() not in {
            "patch", "edits"
        }:
            raise ValueError(
                "The SPG SkillOpt adapter currently supports update_mode='patch'. "
                "Use the upstream full trainer for rewrite modes."
            )

        records_root = Path(records_dir).parent if records_dir else Path("logs")
        self._work_dir = records_root / "skillopt"
        self._prediction_dir = self._work_dir / "predictions"
        self._patches_dir = self._work_dir / "patches"
        self._reflections_dir = self._work_dir / "reflections"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._prediction_dir.mkdir(parents=True, exist_ok=True)
        self._patches_dir.mkdir(parents=True, exist_ok=True)
        self._reflections_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self._work_dir / "updates.jsonl"
        self._rollout_index = sum(
            1 for path in self._prediction_dir.iterdir()
            if path.is_dir() and path.name.startswith("task_")
        )

        self._selection_task_ids = self._build_selection_ids()
        self._configure_optimizer_backend()

    # ── Runtime setup and persistence ───────────────────────────────────

    def _configure_optimizer_backend(self) -> None:
        backend = str(
            self._config.get("optimizer_backend", "openai_compatible")
        ).strip()
        set_optimizer_backend(backend)
        if get_optimizer_backend() != "openai_compatible":
            return

        optimizer_url = (
            self._config.get("optimizer_base_url")
            or self._config.get("reflection_base_url")
            or os.getenv("OPTIMIZER_BASE_URL")
            or os.getenv("REFLECTION_BASE_URL")
            or os.getenv("LLM_BASE_URL")
        )
        optimizer_key = (
            self._config.get("optimizer_api_key")
            or self._config.get("reflection_api_key")
            or os.getenv("OPTIMIZER_API_KEY")
            or os.getenv("REFLECTION_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        optimizer_model = (
            self._config.get("optimizer_model")
            or self._config.get("reflection_model")
            or os.getenv("OPTIMIZER_MODEL")
            or os.getenv("REFLECTION_MODEL")
            or os.getenv("LLM_MODEL")
        )
        kwargs: dict[str, Any] = {
            "optimizer_base_url": optimizer_url,
            "optimizer_api_key": optimizer_key,
            "optimizer_model": optimizer_model,
        }
        if self._config.get("optimizer_temperature") is not None:
            kwargs["temperature"] = self._config["optimizer_temperature"]
        if self._config.get("optimizer_timeout_seconds") is not None:
            kwargs["timeout_seconds"] = self._config["optimizer_timeout_seconds"]
        if self._config.get("optimizer_max_tokens") is not None:
            kwargs["max_tokens"] = self._config["optimizer_max_tokens"]
        configure_openai_compatible(**kwargs)

    def _build_selection_ids(self) -> list[int]:
        selection_dataset = self._selection_dataset
        if selection_dataset is None:
            return []
        pool = selection_dataset.task_pool
        ids = list(range(pool.M))
        random.Random(self._seed + 7919).shuffle(ids)
        if self._selection_size:
            ids = ids[: min(self._selection_size, len(ids))]
        return ids

    @staticmethod
    def _resolve_path(value: str | os.PathLike | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else _PROJECT_ROOT / path

    def _initial_skill_content(self) -> str:
        inline = self._config.get("initial_skill")
        if isinstance(inline, str) and inline.strip():
            return inline.strip() + "\n"
        configured = self._resolve_path(
            self._config.get("initial_skill_path")
            or self._config.get("skill_path")
        )
        dataset_name = str(getattr(self._dataset, "name", "")).strip().lower()
        dataset_default = (
            _SKILLOPT_ROOT / "envs" / dataset_name / "skills" / "initial.md"
            if dataset_name
            else None
        )
        # Keep the ALFWorld seed used by the provided experiment configs, but
        # never inject it into another dataset that has no dataset-specific
        # SkillOpt resource.
        default = dataset_default if dataset_default and dataset_default.is_file() else None
        path = configured if configured and configured.is_file() else default
        if path is not None and path.is_file():
            return path.read_text(encoding="utf-8")
        return (
            "# Agent Skill\n\n"
            "Follow the task goal, use only admissible actions, keep track of "
            "progress, and verify completion before stopping.\n"
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _save_skill_state(self) -> None:
        if not self._skills_dir:
            return
        self._write_text(self._skills_dir / "current_skill.md", self._active_skill)
        self._write_text(self._skills_dir / "best_skill.md", self._best_skill)

    def load_skills(self, skills_dir: str):
        directory = Path(skills_dir)
        directory.mkdir(parents=True, exist_ok=True)
        current_path = directory / "current_skill.md"
        best_path = directory / "best_skill.md"
        # `current` is the policy that actually generated the next target
        # rollout.  Prefer it over the historical best so a resumed run does
        # not silently change policy before checkpoint state is restored.
        if current_path.is_file():
            skill = current_path.read_text(encoding="utf-8")
        elif best_path.is_file():
            skill = best_path.read_text(encoding="utf-8")
        else:
            skill = self._initial_skill_content()
        self._skills_dir = directory
        self._skill_mgr = None
        self._active_skill = skill
        self._best_skill = skill
        self._current_score = 0.0
        self._best_score = 0.0
        self._best_step = 0
        self._save_skill_state()
        if self._selection_task_ids:
            hard, soft = self._evaluate_skill(skill)
            self._current_score = self._gate_score(skill, hard, soft)
            self._best_score = self._current_score
            self._save_history({
                "event": "initial_gate",
                "hard": hard,
                "soft": soft,
                "score": self._current_score,
                "num_tasks": len(self._selection_task_ids),
                "selection_task_ids": self._selection_task_ids,
            })
        print(
            f"  >>> SkillOpt loaded skill ({len(skill)} chars) from {directory}",
            flush=True,
        )

    # ── SkillOpt rollout materialisation ─────────────────────────────────

    def _save_triplets(self, task_id: int, triplets: list, result: dict):
        """Capture the exact target conversation for SkillOpt's analyst."""
        if not self._gate_mode:
            self._pending_triplets.setdefault(task_id, []).append({
                "triplets": copy.deepcopy(triplets),
                "result": copy.deepcopy(result),
            })
            super()._save_triplets(task_id, triplets, result)

    def _materialize_item(
        self,
        task_id: int,
        result: dict,
        captured: dict[str, Any] | None,
    ) -> dict[str, Any]:
        captured = captured or {}
        triplets = captured.get("triplets", [])
        conversation: list[dict[str, str]] = []
        for turn in triplets:
            if not isinstance(turn, dict):
                continue
            user = str(turn.get("user", ""))
            assistant = str(turn.get("assistant", ""))
            if user:
                conversation.append({"role": "user", "content": user})
            if assistant:
                conversation.append({"role": "assistant", "content": assistant})
        if not conversation and result.get("trajectory"):
            conversation = [{
                "role": "assistant",
                "content": str(result.get("trajectory", "")),
            }]
        elif result.get("trajectory"):
            # Keep the normalized user/assistant turns, and append the
            # environment-side trace so the analyst sees observations and the
            # terminal outcome that are not present in the assistant text.
            conversation.append({
                "role": "system",
                "content": (
                    f"Outcome: {'success' if result.get('success') else 'failure'}\n"
                    f"Execution trace:\n{result.get('trajectory', '')}"
                ),
            })

        rollout_id = f"task_{task_id}_rollout_{self._rollout_index:06d}"
        self._rollout_index += 1
        target_dir = self._prediction_dir / rollout_id
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "conversation.json").write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        goal = self._dataset.get_task_goal(task_id)
        try:
            task_type = self._dataset.get_task_type(task_id)
        except (AttributeError, IndexError, KeyError):
            task_type = self._detect_task_type(goal, task_id=task_id)
        success = bool(result.get("success", False))
        trajectory = str(result.get("trajectory", ""))
        return {
            "id": rollout_id,
            "hard": 1.0 if success else 0.0,
            "soft": float(result.get("soft", 1.0 if success else 0.0)),
            "n_turns": len(triplets) or len(result.get("actions", [])),
            "fail_reason": "" if success else trajectory[-1000:],
            "task_type": str(task_type),
            "task_description": str(goal),
            "instruction": str(goal),
            "spg_task_id": task_id,
            "agent_ok": True,
            "api_calls": int(result.get("api_calls", 0)),
        }

    def _run_single_rollout(self, task_id: int) -> tuple[dict, dict]:
        if not self._collect_evidence:
            result = SimpleAgent.execute(self, task_id)
            self._pending_triplets.pop(task_id, None)
            return result, {}
        before = len(self._pending_triplets.get(task_id, []))
        result = SimpleAgent.execute(self, task_id)
        pending = self._pending_triplets.get(task_id, [])
        captured = pending.pop(before) if len(pending) > before else None
        if pending:
            self._pending_triplets[task_id] = pending
        else:
            self._pending_triplets.pop(task_id, None)
        item = self._materialize_item(task_id, result, captured)
        return result, item

    def execute(self, task_id: int, num_rollouts: int = 1) -> dict:
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be at least 1")
        rollout_results: list[dict] = []
        items: list[dict] = []
        for _ in range(num_rollouts):
            result, item = self._run_single_rollout(task_id)
            rollout_results.append(result)
            items.append(item)
        outcomes = [bool(r.get("success", False)) for r in rollout_results]
        successes = sum(outcomes)
        representative = rollout_results[0]
        return {
            "success": successes * 2 >= num_rollouts,
            "successes": successes,
            "num_rollouts": num_rollouts,
            "success_rate": successes / num_rollouts,
            "rollout_successes": outcomes,
            "rollout_results": rollout_results,
            "skillopt_items": items,
            "trajectory": representative.get("trajectory", ""),
            "trajectories": [r.get("trajectory", "") for r in rollout_results],
            "actions": representative.get("actions", []),
            "api_calls": sum(int(r.get("api_calls", 0)) for r in rollout_results),
            "loaded_skill": representative.get("loaded_skill"),
        }

    # ── SkillOpt stages ──────────────────────────────────────────────────

    def _step_buffer_context(self) -> str:
        if not self._rejected_context:
            return ""
        return json.dumps(self._rejected_context[-5:], ensure_ascii=False, indent=2)

    def reflect(self, task_id: int, result: dict):
        """Buffer all rollouts for a selected task and optimize in batches."""
        items = result.get("skillopt_items") or []
        self._evidence.extend(copy.deepcopy(items))
        events = []
        while len(self._evidence) >= self._batch_size:
            batch = self._evidence[: self._batch_size]
            del self._evidence[: self._batch_size]
            events.append(self._optimize_batch(batch))
        return events

    def will_update_after_reflect(self, task_id: int, result: dict) -> bool:
        del task_id
        return len(self._evidence) + len(result.get("skillopt_items") or []) >= self._batch_size

    def finalize(self):
        """Flush a final partial minibatch at the end of evolving."""
        if self._evidence:
            batch = list(self._evidence)
            self._evidence.clear()
            return [self._optimize_batch(batch)]
        return []

    def _gate_score(self, skill: str, hard: float, soft: float) -> float:
        from skillopt.evaluation.gate import select_gate_score

        return select_gate_score(
            hard,
            soft,
            metric=self._gate_metric,
            mixed_weight=self._mixed_weight,
            skill_content=skill,
            use_semantic_density=self._use_semantic_density,
            semantic_density_weight=self._semantic_density_weight,
        )

    def _evaluate_skill(self, skill: str) -> tuple[float, float]:
        """Evaluate a candidate on the fixed gate pool only."""
        if not self._selection_task_ids or self._selection_dataset is None:
            return 0.0, 0.0
        old_dataset = self._dataset
        old_skill = self._active_skill
        old_gate_mode = self._gate_mode
        old_temperature = self._generation_temperature
        self._dataset = self._selection_dataset
        self._active_skill = skill
        self._gate_mode = True
        self._generation_temperature = self._gate_temperature
        results: list[dict[str, float]] = []
        try:
            for task_id in self._selection_task_ids:
                for _ in range(self._selection_rollouts):
                    rollout = SimpleAgent.execute(self, task_id)
                    success = 1.0 if rollout.get("success") else 0.0
                    results.append({"hard": success, "soft": float(rollout.get("soft", success))})
        finally:
            self._dataset = old_dataset
            self._active_skill = old_skill
            self._gate_mode = old_gate_mode
            self._generation_temperature = old_temperature
        return compute_score(results)

    def _optimize_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self._optimizer_updates += 1
        self._global_step += 1
        update_step = self._global_step
        before = self._active_skill
        event = {
            "skill_update_completed": True,
            "skill_updated": False,
            "task_ids": sorted({
                int(item["spg_task_id"])
                for item in batch if item.get("spg_task_id") is not None
            }),
            "update_step": update_step,
        }
        try:
            raw_patches = run_minibatch_reflect(
                batch,
                before,
                str(self._prediction_dir),
                str(self._patches_dir / f"step_{update_step:06d}"),
                workers=self._reflection_workers,
                failure_only=self._failure_only,
                minibatch_size=self._reflection_minibatch_size,
                edit_budget=self._edit_budget,
                random_seed=self._seed + update_step,
                step_buffer_context=self._step_buffer_context(),
                update_mode="patch",
                skill_aware_reflection=self._skill_aware_reflection,
                reflection_log_dir=str(
                    self._reflections_dir / f"step_{update_step:06d}"
                ),
            )
            failure_patches = [
                item["patch"] for item in raw_patches
                if isinstance(item, dict) and item.get("source_type") == "failure"
                and isinstance(item.get("patch"), dict)
            ]
            success_patches = [
                item["patch"] for item in raw_patches
                if isinstance(item, dict) and item.get("source_type") == "success"
                and isinstance(item.get("patch"), dict)
            ]
            if not failure_patches and not success_patches:
                self._save_history({
                    "event": "no_patch",
                    "step": update_step,
                    "batch_size": len(batch),
                })
                event["reason"] = "no_patch"
                return event
            merged = merge_patches(
                before,
                failure_patches,
                success_patches,
                batch_size=self._merge_batch_size,
                verbose=False,
                workers=self._reflection_workers,
                update_mode="patch",
                meta_skill_context="",
            )
            selected = rank_and_select(
                before,
                merged,
                max_edits=self._edit_budget,
                update_mode="patch",
            )
            edits = get_payload_items(selected, "patch")
            candidate, reports = apply_patch_with_report(before, selected)
            if not edits or candidate == before:
                self._save_history({
                    "event": "no_change",
                    "step": update_step,
                    "batch_size": len(batch),
                    "num_edits": len(edits),
                    "reports": reports,
                })
                event["reason"] = "no_change"
                return event

            cand_hard, cand_soft = self._evaluate_skill(candidate)
            gate = evaluate_gate(
                candidate,
                cand_hard,
                before,
                self._current_score,
                self._best_skill,
                self._best_score,
                self._best_step,
                update_step,
                cand_soft=cand_soft,
                metric=self._gate_metric,
                mixed_weight=self._mixed_weight,
                use_semantic_density=self._use_semantic_density,
                semantic_density_weight=self._semantic_density_weight,
            )
            accepted = gate.action != "reject"
            self._active_skill = gate.current_skill
            self._current_score = gate.current_score
            self._best_skill = gate.best_skill
            self._best_score = gate.best_score
            self._best_step = gate.best_step
            if accepted:
                self._save_skill_state()
            else:
                self._rejected_context.append({
                    "step": update_step,
                    "hard": cand_hard,
                    "soft": cand_soft,
                    "score": self._gate_score(candidate, cand_hard, cand_soft),
                    "num_edits": len(edits),
                    "reports": reports,
                })
            self._save_history({
                "event": "gate",
                "step": update_step,
                "batch_size": len(batch),
                "task_ids": [item.get("spg_task_id") for item in batch],
                "num_edits": len(edits),
                "candidate_hard": cand_hard,
                "candidate_soft": cand_soft,
                "candidate_score": self._gate_score(candidate, cand_hard, cand_soft),
                "action": gate.action,
                "accepted": accepted,
                "best_score": self._best_score,
                "reports": reports,
            })
            print(
                f"  >>> SkillOpt update {update_step}: {gate.action} "
                f"(hard={cand_hard:.3f}, soft={cand_soft:.3f})",
                flush=True,
            )
            event["skill_updated"] = bool(accepted)
            event["reason"] = gate.action
            return event
        except Exception as exc:  # noqa: BLE001
            self._save_history({
                "event": "error",
                "step": update_step,
                "batch_size": len(batch),
                "error": repr(exc),
            })
            print(f"  >>> SkillOpt update failed: {exc}", flush=True)
            event["reason"] = f"error:{type(exc).__name__}"
            return event

    def _save_history(self, record: dict[str, Any]) -> None:
        record = {"timestamp": time.time(), **record}
        self._history.append(record)
        with self._history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_checkpoint(self) -> dict:
        """Serialize all state that affects future SkillOpt updates.

        Skill documents on disk alone are insufficient: the buffered
        trajectories, validation scores, update counter, and rejected-patch
        context change what the next optimization step will do.
        """
        return {
            "version": 1,
            "active_skill": self._active_skill,
            "best_skill": self._best_skill,
            "current_score": self._current_score,
            "best_score": self._best_score,
            "best_step": self._best_step,
            "global_step": self._global_step,
            "optimizer_updates": self._optimizer_updates,
            "rollout_index": self._rollout_index,
            "evidence": copy.deepcopy(self._evidence),
            "pending_triplets": copy.deepcopy(self._pending_triplets),
            "rejected_context": copy.deepcopy(self._rejected_context),
            "selection_task_ids": list(self._selection_task_ids),
        }

    def load_checkpoint(self, state: dict):
        """Restore a checkpoint written by :meth:`save_checkpoint`."""
        if not isinstance(state, dict) or not state:
            return
        self._active_skill = str(state.get("active_skill", self._active_skill))
        self._best_skill = str(state.get("best_skill", self._best_skill))
        self._current_score = float(state.get("current_score", self._current_score))
        self._best_score = float(state.get("best_score", self._best_score))
        self._best_step = int(state.get("best_step", self._best_step))
        self._global_step = int(state.get("global_step", self._global_step))
        self._optimizer_updates = int(
            state.get("optimizer_updates", self._optimizer_updates)
        )
        self._rollout_index = max(
            self._rollout_index, int(state.get("rollout_index", self._rollout_index))
        )
        evidence = state.get("evidence")
        if isinstance(evidence, list):
            self._evidence = copy.deepcopy(evidence)
        pending = state.get("pending_triplets")
        if isinstance(pending, dict):
            self._pending_triplets = copy.deepcopy(pending)
        rejected = state.get("rejected_context")
        if isinstance(rejected, list):
            self._rejected_context = copy.deepcopy(rejected)
        selection_ids = state.get("selection_task_ids")
        if isinstance(selection_ids, list):
            self._selection_task_ids = [int(task_id) for task_id in selection_ids]
        self._save_skill_state()
        self._save_history({
            "event": "checkpoint_restored",
            "step": self._global_step,
            "buffered_rollouts": len(self._evidence),
        })

    # ── Prompt/runner lifecycle ─────────────────────────────────────────

    def _get_skill_section(self, goal: str, task_type: str) -> str:
        del goal, task_type
        return self._active_skill

    def get_usage(self) -> dict:
        usage = super().get_usage()
        usage.update({
            "optimizer_backend": get_optimizer_backend(),
            "optimizer_updates": self._optimizer_updates,
            "buffered_rollouts": len(self._evidence),
            "accepted_skill_score": self._current_score,
            "best_skill_score": self._best_score,
            "optimizer_tokens": get_token_summary(),
        })
        return usage

    def reset(self):
        """Reset counters for evaluation while retaining the evolved skill."""
        self._total_calls = 0
        self._loaded_skill = None
        self._gate_mode = False
