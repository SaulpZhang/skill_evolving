"""Adapter that reuses SkillRL's SkillBank in the SPG-Bandit runtime.

The runner supplies environment rollouts through ``SimpleAgent``.  The small
SkillRL memory/updater subset required at runtime is vendored under
``resource/skillrl`` so experiments do not depend on the upstream source
checkout being present on the execution server.
"""

from __future__ import annotations

import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

from spg_bandit.modules.skill_evolving.simple_agent import SimpleAgent


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SKILLRL_ROOT = _PROJECT_ROOT / "resource" / "skillrl"
if not _SKILLRL_ROOT.is_dir():
    raise ImportError(
        "SkillRL runtime resources are missing: "
        f"{_SKILLRL_ROOT}. Copy the required files into resource/skillrl."
    )


def _register_source_package(name: str, path: Path) -> None:
    """Expose a source package without executing its heavy ``__init__``.

    Upstream ``agent_system.memory.__init__`` eagerly imports FAISS-backed
    retrieval memory.  The SkillBank-only adapter does not use it, so register
    namespace packages and import the original individual modules directly.
    """
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules.setdefault(name, package)


_SOURCE_PACKAGE = "_spg_skillrl_source"
_register_source_package(_SOURCE_PACKAGE, _SKILLRL_ROOT / "agent_system")
_register_source_package(
    f"{_SOURCE_PACKAGE}.memory", _SKILLRL_ROOT / "agent_system" / "memory",
)

# These are intentionally the original SkillRL source modules.
from _spg_skillrl_source.memory.skills_only_memory import SkillsOnlyMemory  # noqa: E402
from _spg_skillrl_source.memory.skill_updater import SkillUpdater  # noqa: E402


class _OpenAICompatibleSkillUpdater(SkillUpdater):
    """Use SkillUpdater's prompt/parser with this project's API client.

    The upstream updater is retained unchanged when its Azure credentials are
    available.  This small transport adapter makes the same update protocol
    work with the OpenAI-compatible reflection endpoint used by SimpleAgent.
    """

    def __init__(self, agent: SimpleAgent, max_new_skills_per_update: int):
        self.agent = agent
        self.max_completion_tokens = 2048
        self.max_new_skills_per_update = max_new_skills_per_update
        self.update_history = []

    def analyze_failures(self, failed_trajectories, current_skills):
        if not failed_trajectories:
            return []
        next_dyn_idx = self._next_dyn_index(current_skills)
        prompt = self._build_analysis_prompt(
            failed_trajectories, current_skills, next_dyn_idx,
        )
        try:
            response = self.agent._chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_completion_tokens,
                client=self.agent._reflect_client,
                model=self.agent._reflect_model,
            )
            skills = self._reassign_dyn_ids(
                self._parse_skills_response(response), next_dyn_idx,
            )[:self.max_new_skills_per_update]
            self.update_history.append({
                "num_failures_analyzed": len(failed_trajectories),
                "num_skills_generated": len(skills),
                "skill_ids": [skill.get("skill_id") for skill in skills],
            })
            return skills
        except Exception as error:
            print(f"[SkillRL adapter] Skill update failed: {error}")
            return []


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
        self._memory: SkillsOnlyMemory | None = None
        self._skill_path: Path | None = None
        self._retrieval_top_k = int(self._config.get("top_k", 6))
        self._update_interval = int(self._config.get("update_interval", 10))
        self._update_threshold = float(self._config.get("update_threshold", 0.4))
        self._max_new_skills = int(self._config.get("max_new_skills", 3))
        self._dynamic_updates = bool(self._config.get("enable_dynamic_update", True))
        if self._update_interval < 1:
            raise ValueError("skill_evolving.update_interval must be at least 1")
        if not 0 <= self._update_threshold <= 1:
            raise ValueError("skill_evolving.update_threshold must be in [0, 1]")
        self._updater = None
        self._groups_since_update = 0
        self._outcomes_by_type: dict[str, list[bool]] = defaultdict(list)
        self._failed_trajectories: list[dict[str, Any]] = []

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
        rollout_results = [super().execute(task_id) for _ in range(num_rollouts)]
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
            "actions": representative.get("actions", []),
            "api_calls": sum(result.get("api_calls", 0) for result in rollout_results),
            "loaded_skill": representative.get("loaded_skill"),
        }

    def reflect(self, task_id: int, result: dict):
        """Run SkillRL-style recursive updates periodically, not per rollout."""
        if not self._dynamic_updates or self._memory is None or self._updater is None:
            return
        goal = self._dataset.get_task_goal(task_id)
        task_type = self._memory.retrieve(goal, top_k=0).get("task_type", "unknown")
        rollout_results = result.get("rollout_results", [result])
        outcomes = result.get("rollout_successes", [result.get("success", False)])
        self._outcomes_by_type[task_type].extend(bool(value) for value in outcomes)
        for outcome, rollout in zip(outcomes, rollout_results):
            if not outcome:
                self._failed_trajectories.append({
                    "task": goal,
                    "task_type": task_type,
                    "trajectory": self._trajectory_steps(rollout),
                })
        self._groups_since_update += 1
        if self._groups_since_update < self._update_interval:
            return

        poor_types = {
            kind for kind, values in self._outcomes_by_type.items()
            if values and sum(values) / len(values) < self._update_threshold
        }
        failures = [
            trajectory for trajectory in self._failed_trajectories
            if trajectory["task_type"] in poor_types
        ][-10:]
        if failures:
            new_skills = self._updater.analyze_failures(failures, self._memory.skills)
            added = self._memory.add_skills(new_skills, category="general")
            if added and self._skill_path is not None:
                self._memory.save_skills(str(self._skill_path))
                print(f"  >>> SkillRL added {added} dynamic skills", flush=True)
        self._groups_since_update = 0
        self._outcomes_by_type.clear()
        self._failed_trajectories.clear()

    @staticmethod
    def _trajectory_steps(result: dict) -> list[dict[str, str]]:
        actions = result.get("actions", [])
        lines = result.get("trajectory", "").splitlines()
        observations = [line.removeprefix("Obs: ") for line in lines if line.startswith("Obs: ")]
        return [
            {"action": str(action), "observation": observations[index] if index < len(observations) else ""}
            for index, action in enumerate(actions)
        ]

    def reset(self):
        super().reset()
        self._groups_since_update = 0
        self._outcomes_by_type.clear()
        self._failed_trajectories.clear()
