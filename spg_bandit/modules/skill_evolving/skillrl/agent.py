"""Adapter that reuses SkillRL's SkillBank in the SPG-Bandit runtime.

The runner supplies environment rollouts through ``SimpleAgent``.  The small
SkillRL memory/updater subset required at runtime is vendored under
``resource/skillrl`` so experiments do not depend on the upstream source
checkout being present on the execution server.
"""

from __future__ import annotations

import shutil
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from spg_bandit.modules.skillrl_source import (
    PROJECT_ROOT as _PROJECT_ROOT,
    SKILLRL_ROOT as _SKILLRL_ROOT,
    SkillUpdater,
    SkillsOnlyMemory,
    alfworld_projection,
)
from spg_bandit.modules.skill_evolving.base import SelectionContext
from spg_bandit.modules.skill_evolving.simple_agent import SimpleAgent

class _OpenAICompatibleCompletions:
    """Translate upstream Azure completion calls to the configured endpoint."""

    def __init__(self, agent: SimpleAgent):
        self._agent = agent

    def create(self, *, model, messages, max_completion_tokens):
        content = self._agent._chat(
            messages,
            max_tokens=max_completion_tokens,
            client=self._agent._reflect_client,
            model=self._agent._reflect_model,
            # Upstream o3 request does not set a sampling temperature.
            temperature=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )


class _OpenAICompatibleSkillUpdater(SkillUpdater):
    """Run the unmodified upstream updater through a transport-only shim."""

    def __init__(self, agent: SimpleAgent, max_new_skills_per_update: int):
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=_OpenAICompatibleCompletions(agent)),
        )
        self.model = agent._reflect_model
        self.max_completion_tokens = 393216
        self.max_new_skills_per_update = max_new_skills_per_update
        self.update_history = []
        self.last_update_status = {"status": "not_called"}
        self.last_reflection = {"prompt": "", "response": ""}


class SkillRLAgent(SimpleAgent):
    """SkillRL SkillBank with grouped environment rollouts.

    This is deliberately an inference-time adapter: it uses the upstream
    SkillBank and recursive failure analysis, while policy optimisation (the
    upstream GRPO/FSDP trainer) remains outside this API-based runner.
    """

    def __init__(self, dataset, max_turns: int = 30, records_dir: str | None = None,
                 config: dict[str, Any] | None = None):
        super().__init__(dataset, max_turns=max_turns, records_dir=records_dir)
        self._config = config or {}
        # Match the upstream ALFWorld training rollout defaults.  These remain
        # configurable so a frozen-policy evaluation can deliberately use the
        # upstream validation settings instead.
        self._generation_temperature = float(self._config.get("temperature", 1.0))
        self._generation_max_tokens = int(self._config.get("max_tokens", 512))
        self._history_window = int(self._config.get("history_length", 2))
        self._skills_on_initial_step = False
        self._memory: SkillsOnlyMemory | None = None
        self._skill_path: Path | None = None
        self._retrieval_top_k = int(self._config.get("top_k", 6))
        legacy_interval = "update_interval" in self._config
        self._update_batch_size = int(self._config.get(
            "update_batch_size", self._config.get("update_interval", 16),
        ))
        self._skill_update_freq = int(self._config.get("skill_update_freq", 1))
        self._flush_partial_batch = bool(self._config.get(
            "flush_partial_batch", legacy_interval,
        ))
        self._update_threshold = float(self._config.get("update_threshold", 0.4))
        self._max_new_skills = int(self._config.get("max_new_skills", 3))
        self._dynamic_updates = bool(self._config.get("enable_dynamic_update", True))
        if self._update_batch_size < 1:
            raise ValueError("skill_evolving.update_batch_size must be at least 1")
        if self._skill_update_freq < 1:
            raise ValueError("skill_evolving.skill_update_freq must be at least 1")
        if not 0 <= self._update_threshold <= 1:
            raise ValueError("skill_evolving.update_threshold must be in [0, 1]")
        self._updater = None
        self._groups_since_update = 0
        self._virtual_batch_step = 0
        self._batch_task_ids: list[int] = []
        self._outcomes_by_type: dict[str, list[bool]] = defaultdict(list)
        self._failed_trajectories: list[dict[str, Any]] = []
        self._update_diagnostics = self._new_update_diagnostics()
        self._skill_version = 0
        self._task_last_skill_version: dict[int, int] = {}

    @staticmethod
    def _new_update_diagnostics() -> dict[str, int]:
        """Cumulative, resumable counters for the dynamic-update pipeline."""
        return {
            "batches_checked": 0,
            "gate_passed": 0,
            "skipped_no_poor_type": 0,
            "skipped_no_failures": 0,
            "teacher_calls": 0,
            "teacher_errors": 0,
            "teacher_empty": 0,
            "skills_generated": 0,
            "skills_added": 0,
        }

    def _record_update_diagnostic(self, event: dict[str, Any]):
        """Emit one concise, machine-readable record for a flush attempt."""
        reason = event["reason"]
        print(
            "  >>> [SkillRL diagnostic] "
            f"batch={event['virtual_batch_step']} rates={event['success_rates']} "
            f"poor_types={event['poor_types']} failures={event['failure_count']} "
            f"teacher_called={event['teacher_called']} "
            f"generated={event['generated']} added={event['added']} reason={reason}",
            flush=True,
        )
        if self._records_dir is None:
            return
        self._records_dir.mkdir(parents=True, exist_ok=True)
        with (self._records_dir / "skillrl_updates.jsonl").open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_skillrl_reflection(self, failure_count: int):
        """Persist the complete teacher request and response for one update."""
        if self._records_dir is None:
            return
        reflection = getattr(self._updater, "last_reflection", {})
        self._records_dir.mkdir(parents=True, exist_ok=True)
        path = self._records_dir / (
            f"skillrl_reflection_batch_{self._virtual_batch_step}_{time.time_ns()}.json"
        )
        payload = {
            "type": "skillrl_reflection",
            "virtual_batch_step": self._virtual_batch_step,
            "failure_count": failure_count,
            "prompt": reflection.get("prompt", ""),
            "response": reflection.get("response", ""),
            "error": reflection.get("error"),
            "updater_status": getattr(self._updater, "last_update_status", {}),
        }
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _project_action(response: str) -> str:
        """Use SkillRL's original ALFWorld projection semantics.

        A well-formed response contributes the text inside ``<action>`` tags.
        A malformed response is passed to ALFWorld as its final 30 characters,
        exactly as the upstream projection does before recording validity.
        """
        actions, _valids = alfworld_projection([response], [[]])
        return actions[0]

    def load_skills(self, skills_dir: str):
        directory = Path(skills_dir)
        directory.mkdir(parents=True, exist_ok=True)
        skill_path = directory / "skills.json"
        if not skill_path.exists():
            configured_bank = self._config.get("skill_bank_path")
            if configured_bank:
                seed_bank = Path(configured_bank)
                if not seed_bank.is_absolute():
                    seed_bank = _PROJECT_ROOT / seed_bank
            else:
                dataset_name = str(getattr(self._dataset, "name", "dataset")).lower()
                seed_bank = (
                    _SKILLRL_ROOT / "memory_data" / dataset_name
                    / "claude_style_skills.json"
                )
            if not seed_bank.is_file():
                raise FileNotFoundError(
                    f"SkillBank not found: {seed_bank}. "
                    "Set skill_evolving.skill_bank_path to a JSON SkillBank."
                )
            shutil.copyfile(seed_bank, skill_path)
        self._skills_dir = directory
        self._skill_path = skill_path
        self._skill_mgr = None  # Parent execution calls _get_skill_section().
        self._memory = SkillsOnlyMemory(
            str(skill_path),
            retrieval_mode=self._config.get("retrieval_mode", "template"),
            embedding_model_path=self._config.get("embedding_model_path"),
            task_specific_top_k=self._config.get("task_specific_top_k"),
        )
        if self._dynamic_updates:
            try:
                self._updater = SkillUpdater(
                    max_new_skills_per_update=self._max_new_skills,
                )
                print("  >>> SkillRL uses the upstream Azure SkillUpdater", flush=True)
            except EnvironmentError:
                self._updater = _OpenAICompatibleSkillUpdater(self, self._max_new_skills)
                print("  >>> SkillRL uses the OpenAI-compatible SkillUpdater adapter", flush=True)

    def _get_skill_section(self, goal: str, task_type: str) -> str:
        if self._memory is None:
            return ""
        retrieved = self._memory.retrieve(goal, top_k=self._retrieval_top_k)
        self._loaded_skill = {
            "task_type": retrieved.get("task_type", task_type),
            "skill_ids": [
                skill.get("skill_id")
                for key in ("general_skills", "task_specific_skills")
                for skill in retrieved.get(key, [])
            ],
        }
        return self._memory.format_for_prompt(retrieved)

    def execute(self, task_id: int, num_rollouts: int = 1) -> dict:
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be at least 1")
        # Do not use zero-argument ``super()`` inside a list comprehension.
        # Python 3 creates a separate comprehension scope, so the implicit
        # ``__class__``/``self`` pair is lost and raises
        # ``TypeError: super(type, obj)`` on the first rollout.
        rollout_results = [
            SimpleAgent.execute(self, task_id) for _ in range(num_rollouts)
        ]
        outcomes = [bool(result["success"]) for result in rollout_results]
        successes = sum(outcomes)
        representative = rollout_results[0]
        return {
            "success": successes * 2 >= num_rollouts,
            "successes": successes,
            "num_rollouts": num_rollouts,
            "success_rate": successes / num_rollouts,
            "rollout_successes": outcomes,
            "rollout_results": rollout_results,
            "trajectory": representative.get("trajectory", ""),
            "trajectories": [result.get("trajectory", "") for result in rollout_results],
            "trajectory_steps": representative.get("trajectory_steps", []),
            "actions": representative.get("actions", []),
            "api_calls": sum(result.get("api_calls", 0) for result in rollout_results),
            "loaded_skill": representative.get("loaded_skill"),
        }

    def reflect(self, task_id: int, result: dict):
        """Run SkillRL-style recursive updates periodically, not per rollout."""
        if not self._dynamic_updates or self._memory is None or self._updater is None:
            return
        goal = self._dataset.get_task_goal(task_id)
        # Upstream computes the update gate from the dataset/gamefile task
        # labels, while the trajectory sent to SkillUpdater uses the coarser
        # skill-bank category detected from the task text.
        metric_task_type = self._dataset.get_task_type(task_id)
        skill_task_type = self._memory.retrieve(goal, top_k=0).get(
            "task_type", "unknown"
        )
        rollout_results = result.get("rollout_results", [result])
        outcomes = result.get("rollout_successes", [result.get("success", False)])
        self._outcomes_by_type.setdefault(metric_task_type, []).extend(
            bool(value) for value in outcomes
        )
        for outcome, rollout in zip(outcomes, rollout_results):
            if not outcome:
                self._failed_trajectories.extend(
                    {
                        "task": goal,
                        "task_type": skill_task_type,
                        "trajectory": trajectory,
                    }
                    for trajectory in self._failed_trajectory_entries(rollout)
                )
        self._groups_since_update += 1
        self._batch_task_ids.append(int(task_id))
        # This task has now contributed evidence under the current SkillBank.
        # It becomes selectable again only after a real skill update changes
        # the policy version.
        self._task_last_skill_version[int(task_id)] = self._skill_version
        if self._groups_since_update < self._update_batch_size:
            return []
        self._virtual_batch_step += 1
        if self._virtual_batch_step % self._skill_update_freq == 0:
            return [self._flush_skill_update()]
        else:
            # Upstream inspects only the current training batch at a scheduled
            # global step; evidence from intervening batches is not carried
            # into the next update.
            task_ids = list(self._batch_task_ids)
            self._clear_update_batch()
            return [{
                "skill_update_completed": True,
                "skill_updated": False,
                "task_ids": task_ids,
                "reason": "unscheduled_update",
            }]

    def will_update_after_reflect(self, task_id: int, result: dict) -> bool:
        del task_id, result
        next_group = self._groups_since_update + 1
        next_batch = self._virtual_batch_step + 1
        return (
            self._dynamic_updates and self._memory is not None
            and self._updater is not None
            and next_group >= self._update_batch_size
            and next_batch % self._skill_update_freq == 0
        )

    def _flush_skill_update(self):
        """Apply SkillRL's training-batch update rule to buffered rollouts."""
        success_rates = {
            kind: round(sum(values) / len(values), 4)
            for kind, values in self._outcomes_by_type.items()
            if values
        }
        poor_types = {
            kind for kind, rate in success_rates.items()
            if rate < self._update_threshold
        }
        # Upstream uses per-type success only as a gate.  Once any type is
        # below threshold, every failed trajectory in the current batch is
        # eligible, preserving batch order and taking the first ten.
        failures = self._failed_trajectories[:10]
        self._update_diagnostics["batches_checked"] += 1
        event: dict[str, Any] = {
            "virtual_batch_step": self._virtual_batch_step,
            "success_rates": success_rates,
            "poor_types": sorted(poor_types),
            "failure_count": len(failures),
            "teacher_called": False,
            "generated": 0,
            "added": 0,
            "reason": "",
            "skill_update_completed": True,
            "skill_updated": False,
            "task_ids": list(self._batch_task_ids),
        }
        if poor_types and failures:
            self._update_diagnostics["gate_passed"] += 1
            self._update_diagnostics["teacher_calls"] += 1
            event["teacher_called"] = True
            try:
                new_skills = self._updater.analyze_failures(
                    failures, self._memory.skills,
                )
            except Exception as exc:
                # The vendored updater handles its own API exceptions, but
                # preserve a clear record if another updater implementation
                # leaks one through this boundary.
                self._update_diagnostics["teacher_errors"] += 1
                event["reason"] = f"teacher_exception:{type(exc).__name__}"
                new_skills = []
            self._save_skillrl_reflection(len(failures))
            status = getattr(self._updater, "last_update_status", {})
            if isinstance(status, dict):
                event["updater_status"] = status
            event["generated"] = len(new_skills)
            self._update_diagnostics["skills_generated"] += len(new_skills)
            added = self._memory.add_skills(new_skills, category="general")
            event["added"] = added
            self._update_diagnostics["skills_added"] += added
            if added and self._skill_path is not None:
                self._memory.save_skills(str(self._skill_path))
                print(f"  >>> SkillRL added {added} dynamic skills", flush=True)
            event["skill_updated"] = bool(added)
            if added:
                self._skill_version += 1
            if not event["reason"]:
                event["reason"] = (
                    status.get("status", "generated")
                    if isinstance(status, dict) and not new_skills
                    else "generated"
                )
            if not new_skills:
                self._update_diagnostics["teacher_empty"] += 1
                if isinstance(status, dict) and status.get("status") == "api_error":
                    self._update_diagnostics["teacher_errors"] += 1
        elif not poor_types:
            self._update_diagnostics["skipped_no_poor_type"] += 1
            event["reason"] = "no_poor_task_type"
        else:
            self._update_diagnostics["skipped_no_failures"] += 1
            event["reason"] = "no_failed_trajectory"
        self._record_update_diagnostic(event)
        self._clear_update_batch()
        return event

    def _clear_update_batch(self):
        self._groups_since_update = 0
        self._outcomes_by_type.clear()
        self._failed_trajectories.clear()
        self._batch_task_ids.clear()

    @property
    def selection_feature_dim(self) -> int:
        return 4

    def get_selection_context(self, task_id: int) -> SelectionContext:
        """Expose SkillRL batch-evidence value to SPG.

        SkillRL consumes a task once per SkillBank version.  A successful
        update opens a new evidence round; an empty/failed teacher update does
        not make repeating the same trajectory informative.
        """
        if not self._dynamic_updates:
            return SelectionContext(
                features=np.zeros(self.selection_feature_dim),
                eligible=True,
                reason="dynamic_updates_disabled",
                policy_version=f"skillrl:{self._skill_version}",
            )

        task_id = int(task_id)
        task_type = self._dataset.get_task_type(task_id)
        outcomes = self._outcomes_by_type.get(task_type, [])
        evidence_ratio = min(len(outcomes) / max(self._update_batch_size, 1), 1.0)
        failure_ratio = (
            1.0 - sum(outcomes) / len(outcomes) if outcomes else 0.0
        )
        observed = self._task_last_skill_version.get(task_id) == self._skill_version
        features = np.asarray([
            float(observed),
            min(self._groups_since_update / max(self._update_batch_size, 1), 1.0),
            evidence_ratio,
            failure_ratio,
        ])
        return SelectionContext(
            features=features,
            eligible=not observed,
            reason=(
                "already_observed_for_skill_version" if observed else "new_batch_evidence"
            ),
            policy_version=f"skillrl:{self._skill_version}",
        )

    def get_selection_features(self, task_id: int):
        """Compatibility view for callers predating SelectionContext."""
        return self.get_selection_context(task_id).features

    @staticmethod
    def _trajectory_steps(result: dict) -> list[dict[str, str]]:
        conversation_steps = result.get("trajectory_steps")
        if conversation_steps is not None:
            return [
                {
                    "action": str(step.get("action", ""))[:1500],
                    "observation": str(step.get("observation", ""))[:800],
                }
                for step in conversation_steps
            ]

        # Compatibility for records created before raw conversation steps were
        # stored.  Pair each observation with the immediately preceding action
        # instead of independently indexing the two lists (which shifted all
        # later observations after an invalid action).
        actions = result.get("actions", [])
        lines = result.get("trajectory", "").splitlines()
        steps = [{"action": str(action), "observation": ""} for action in actions]
        action_index = -1
        for line in lines:
            if line.startswith("Agent: "):
                action_index += 1
            elif line.startswith("Obs: ") and 0 <= action_index < len(steps):
                steps[action_index]["observation"] = line.removeprefix("Obs: ")
        return steps

    @classmethod
    def _failed_trajectory_entries(
        cls, result: dict,
    ) -> list[list[dict[str, str]]]:
        """Recreate the per-step entries consumed by upstream SkillUpdater.

        SkillRL flattens an environment rollout into one training-batch row per
        turn. Its parser sees each row as an input prompt plus the current
        response, so a failed multi-turn episode contributes one failure entry
        per turn rather than one synthesized whole-episode record.
        """
        conversation_steps = result.get("trajectory_steps")
        if conversation_steps is not None:
            return [
                [
                    {
                        "action": "",
                        "observation": str(step.get("observation", ""))[:3000],
                    },
                    {
                        "action": str(step.get("action", ""))[:2000],
                        "observation": "",
                    },
                ]
                for step in conversation_steps
            ]
        return [cls._trajectory_steps(result)]

    def finalize(self):
        """Optionally flush a final partial virtual batch."""
        if (
            self._dynamic_updates
            and self._memory is not None
            and self._updater is not None
            and self._flush_partial_batch
            and self._groups_since_update > 0
        ):
            return [self._flush_skill_update()]
        return []

    def save_checkpoint(self) -> dict:
        return {
            "groups_since_update": self._groups_since_update,
            "virtual_batch_step": self._virtual_batch_step,
            "outcomes_by_type": dict(self._outcomes_by_type),
            "failed_trajectories": list(self._failed_trajectories),
            "batch_task_ids": list(self._batch_task_ids),
            "update_diagnostics": dict(self._update_diagnostics),
            "skill_version": self._skill_version,
            "task_last_skill_version": {
                str(task_id): version
                for task_id, version in self._task_last_skill_version.items()
            },
        }

    def load_checkpoint(self, state: dict):
        if not state:
            return
        self._groups_since_update = int(state.get("groups_since_update", 0))
        self._virtual_batch_step = int(state.get("virtual_batch_step", 0))
        self._outcomes_by_type = defaultdict(
            list,
            {
                str(task_type): [bool(value) for value in values]
                for task_type, values in state.get("outcomes_by_type", {}).items()
            },
        )
        self._failed_trajectories = list(state.get("failed_trajectories", []))
        self._batch_task_ids = [int(value) for value in state.get("batch_task_ids", [])]
        self._update_diagnostics = self._new_update_diagnostics()
        self._update_diagnostics.update({
            key: int(value)
            for key, value in state.get("update_diagnostics", {}).items()
            if key in self._update_diagnostics
        })
        self._skill_version = int(state.get("skill_version", 0))
        self._task_last_skill_version = {
            int(task_id): int(version)
            for task_id, version in state.get("task_last_skill_version", {}).items()
        }

    def reset(self):
        super().reset()
        self._virtual_batch_step = 0
        self._clear_update_batch()
        self._update_diagnostics = self._new_update_diagnostics()
        self._skill_version = 0
        self._task_last_skill_version.clear()
