"""Embedded ExpeL for the SPG-Bandit experiment runtime.

This adapter follows the implementation in ``docs/ExpeL`` at the algorithmic
boundaries that matter:

* the actor uses ExpeL's ALFWorld ReAct protocol and official demonstrations;
* failed trials produce task-local ``New plan`` reflections used on retries;
* successful trials are retrieved as demonstrations by task similarity;
* global insights are extracted from success/failure pairs and groups of
  successful tasks; and
* AGREE/REMOVE/EDIT/ADD update rule strengths with ExpeL's original weights.

The only intentional adaptation is scheduling.  ``spg_online`` lets the outer
SPG selector choose one trial at a time and reuses a reflection when that task
is selected again.  ``paper_faithful`` performs the original contiguous retry
loop inside one selected task.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from spg_bandit.modules.dataset.base import BaseDataset
from spg_bandit.modules.skill_evolving.base import BaseSkillEvolving, SelectionContext
from spg_bandit.modules.skill_evolving.expel.prompts import (
    OfficialPromptAssets,
    build_actor_context,
    build_insight_prompt,
    build_reflection_prompt,
    format_alfworld_task,
    format_webshop_task,
)
from spg_bandit.modules.skill_evolving.expel.retrieval import retrieve_successes
from spg_bandit.modules.skill_evolving.expel.rules import RuleBank, parse_operations


load_dotenv(Path(__file__).resolve().parents[4] / ".env")


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ExpelAgent(BaseSkillEvolving):
    """ExpeL memory, execution, reflection, retrieval, and insight extraction."""

    _STATE_VERSION = 2
    _ACTION_PATTERN = re.compile(
        r"^(?:go to|open|take|put|use|heat|cool|look|clean|inventory)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        dataset: BaseDataset,
        max_turns: int = 30,
        records_dir: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        actor_chat: Callable[..., str] | None = None,
        reflection_chat: Callable[..., str] | None = None,
    ):
        self._dataset = dataset
        self.max_turns = int(max_turns)
        self._records_dir = Path(records_dir) if records_dir else None
        self._config = dict(config or {})
        self._mode = str(self._config.get("mode", "spg_online"))
        if self._mode not in {"spg_online", "paper_faithful"}:
            raise ValueError("skill_evolving.mode must be spg_online or paper_faithful")
        self._insight_strategy = str(self._config.get("insight_strategy", "incremental"))
        if self._insight_strategy not in {"incremental", "deferred"}:
            raise ValueError("insight_strategy must be incremental or deferred")

        self._dynamic_updates = bool(self._config.get("enable_dynamic_update", True))
        self._temperature = float(self._config.get("temperature", 0.3))
        self._max_tokens = int(self._config.get("max_tokens", 1024))
        self._reflection_temperature = float(self._config.get("reflection_temperature", 0.0))
        self._reflection_max_tokens = int(self._config.get("reflection_max_tokens", 4096))
        self._insight_max_tokens = int(
            self._config.get("insight_max_tokens", self._reflection_max_tokens)
        )
        self._top_k = int(self._config.get("top_k", 2))
        self._max_fewshot_tokens = int(self._config.get("max_fewshot_tokens", 1000))
        self._max_reflection_depth = int(self._config.get("max_reflection_depth", 3))
        self._success_critique_num = int(self._config.get("success_critique_num", 8))
        self._use_memory_during_evolve = bool(
            self._config.get("use_learned_memory_during_evolve", True)
        )
        self._max_prompt_trajectory_chars = int(
            self._config.get("max_prompt_trajectory_chars", 48000)
        )
        max_rules = int(self._config.get("max_num_rules", self._config.get("max_rules", 20)))
        if (
            self.max_turns < 1 or self._max_tokens < 1 or self._reflection_max_tokens < 1
            or self._top_k < 0
            or self._max_reflection_depth < 0 or self._success_critique_num < 1
            or self._max_fewshot_tokens < 1
        ):
            raise ValueError("Invalid ExpeL runtime configuration")

        # Custom datasets used by tests and downstream extensions retain the
        # original ALFWorld protocol unless they explicitly identify as WebShop.
        self._benchmark = (
            "webshop" if str(getattr(self._dataset, "name", "")).lower() == "webshop"
            else "alfworld"
        )
        self._assets = OfficialPromptAssets.load(
            self._config.get("official_source_dir"), benchmark=self._benchmark,
        )
        self._rule_bank = RuleBank(max_rules)
        self._skills_dir: Path | None = None
        self._expel_dir: Path | None = None

        self._actor_chat_override = actor_chat
        self._reflection_chat_override = reflection_chat
        self._actor_client = None
        self._reflection_client = None
        self._actor_model = os.getenv("LLM_MODEL")
        self._reflection_model = os.getenv("REFLECTION_MODEL", self._actor_model)
        if actor_chat is None:
            self._actor_client = OpenAI(
                base_url=os.getenv("LLM_BASE_URL"),
                api_key=os.getenv("LLM_API_KEY"),
                timeout=300,
            )
            if not self._actor_model:
                raise EnvironmentError("Expel requires LLM_MODEL")
        if reflection_chat is None and self._dynamic_updates:
            self._reflection_client = OpenAI(
                base_url=os.getenv("REFLECTION_BASE_URL", os.getenv("LLM_BASE_URL")),
                api_key=os.getenv("REFLECTION_API_KEY", os.getenv("LLM_API_KEY")),
                timeout=300,
            )
            if not self._reflection_model:
                raise EnvironmentError("Expel requires REFLECTION_MODEL for dynamic updates")

        self._experiences: list[dict[str, Any]] = []
        self._by_task: dict[str, list[str]] = defaultdict(list)
        self._task_reflections: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._processed_pairs: set[str] = set()
        self._processed_success_chunks: set[str] = set()
        self._decision_id = 0
        self._experience_id = 0
        self._reflection_id = 0
        self._update_step = 0
        self._total_calls = 0
        self._actor_calls = 0
        self._reflection_calls = 0
        self._insight_calls = 0
        self._diagnostics: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Durable state and event streams

    def load_skills(self, skills_dir: str):
        self._skills_dir = Path(skills_dir)
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._expel_dir = (
            self._records_dir.parent / "expel"
            if self._records_dir is not None else self._skills_dir / "expel_logs"
        )
        for name in ("prompts", "rule_snapshots"):
            (self._expel_dir / name).mkdir(parents=True, exist_ok=True)
        if self._state_path.is_file():
            try:
                self._restore_state(json.loads(self._state_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid ExpeL state file: {self._state_path}") from exc
        self._append("lifecycle", {
            "event": "load",
            "state_version": self._STATE_VERSION,
            "mode": self._mode,
            "insight_strategy": self._insight_strategy,
            "official_source": self._assets.source_path,
            "official_source_sha256": self._assets.source_sha256,
            "timestamp": time.time(),
        })

    @property
    def _state_path(self) -> Path:
        if self._skills_dir is None:
            raise RuntimeError("load_skills must be called before persisting ExpeL state")
        return self._skills_dir / "expel_state.json"

    @staticmethod
    def _atomic_json(path: Path, payload: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def _append(self, stream: str, payload: dict[str, Any]):
        if self._expel_dir is None:
            return
        path = self._expel_dir / f"{stream}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _state_payload(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "mode": self._mode,
            "official_source_sha256": self._assets.source_sha256,
            "rule_bank": self._rule_bank.to_state(),
            "experiences": self._experiences,
            "task_reflections": dict(self._task_reflections),
            "processed_pairs": sorted(self._processed_pairs),
            "processed_success_chunks": sorted(self._processed_success_chunks),
            "decision_id": self._decision_id,
            "experience_id": self._experience_id,
            "reflection_id": self._reflection_id,
            "update_step": self._update_step,
            "diagnostics": dict(self._diagnostics),
        }

    def _restore_state(self, payload: dict[str, Any]):
        # Version-one migration preserves prior rules/experiences, then starts
        # the source-faithful reflection/insight bookkeeping from that point.
        saved_source = payload.get("official_source_sha256")
        if (
            saved_source
            and saved_source != self._assets.source_sha256
            and not self._config.get("allow_official_source_mismatch", False)
        ):
            raise ValueError(
                "Bundled ExpeL prompt source changed since this state was saved. "
                "Set allow_official_source_mismatch=true only for an intentional migration."
            )
        self._rule_bank.load_state(payload.get("rule_bank", payload.get("rules", [])))
        self._experiences = list(payload.get("experiences", []))
        legacy_state = int(payload.get("version", 1)) < self._STATE_VERSION
        legacy_types = {
            "pick_and_place": "pick_and_place_simple",
            "clean": "pick_clean_then_place_in_recep",
            "heat": "pick_heat_then_place_in_recep",
            "cool": "pick_cool_then_place_in_recep",
            "look_at_obj_in_light": "look_at_obj_in_light",
            "pick_two_obj_and_place": "pick_two_obj_and_place",
        }
        for item in self._experiences:
            if legacy_state:
                item["task_type"] = legacy_types.get(
                    str(item.get("task_type", "")), str(item.get("task_type", "default"))
                )
            elif "task_type" not in item and "task_id" in item:
                item["task_type"] = self._raw_task_type(int(item["task_id"]))
            if legacy_state or "task_key" not in item:
                item["task_key"] = self._task_key(
                    str(item.get("task_goal", "")), str(item.get("task_type", "default"))
                )
            if not item.get("task_embedding") and "task_id" in item:
                task_id = int(item["task_id"])
                if (
                    0 <= task_id < self._dataset.task_pool.M
                    and self._dataset.get_task_goal(task_id).strip().casefold()
                    == str(item.get("task_goal", "")).strip().casefold()
                ):
                    item["task_embedding"] = self._dataset.task_pool.get_embedding(task_id).tolist()
        self._task_reflections = defaultdict(list, {
            str(key): list(value)
            for key, value in payload.get("task_reflections", {}).items()
        })
        self._processed_pairs = set(payload.get("processed_pairs", []))
        self._processed_success_chunks = set(payload.get("processed_success_chunks", []))
        self._decision_id = int(payload.get("decision_id", 0))
        self._experience_id = int(payload.get("experience_id", len(self._experiences)))
        self._reflection_id = int(payload.get("reflection_id", 0))
        self._update_step = int(payload.get("update_step", 0))
        self._diagnostics = defaultdict(int, payload.get("diagnostics", {}))
        self._by_task = defaultdict(list)
        for item in self._experiences:
            self._by_task[str(item["task_key"])].append(str(item["experience_id"]))

    def _persist(self):
        if self._skills_dir is None:
            return
        self._atomic_json(self._state_path, self._state_payload())

    def _snapshot_rules(self):
        if self._expel_dir is None:
            return
        payload = {
            "update_step": self._update_step,
            "rule_hash": self._rule_bank.hash(),
            "rules": [vars(rule) for rule in self._rule_bank.rules],
        }
        self._atomic_json(self._expel_dir / "current_rules.json", payload)
        self._atomic_json(
            self._expel_dir / "rule_snapshots" / f"rules_{self._update_step:06d}.json",
            payload,
        )

    # ------------------------------------------------------------------
    # LLM and actor protocol

    def _chat(
        self, *, kind: str, system: str, user: str, max_tokens: int,
        temperature: float, stop: list[str] | None = None,
    ) -> str:
        override = self._actor_chat_override if kind == "actor" else self._reflection_chat_override
        self._total_calls += 1
        if kind == "actor":
            self._actor_calls += 1
        elif kind == "reflection":
            self._reflection_calls += 1
        else:
            self._insight_calls += 1
        if override is not None:
            value = override(
                kind=kind, system=system, user=user, max_tokens=max_tokens,
                temperature=temperature, stop=stop,
            )
            return str(value or "").strip()
        client = self._actor_client if kind == "actor" else self._reflection_client
        model = self._actor_model if kind == "actor" else self._reflection_model
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            request["stop"] = stop
        for attempt in range(3):
            try:
                response = client.chat.completions.create(**request)
                return str(response.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        raise AssertionError("unreachable")

    @classmethod
    def _parse_react_output(cls, output: str) -> tuple[str, str]:
        text = (output or "").strip()
        thought = re.match(r"^(?:>\s*)?think\s*:?[\s]*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if thought:
            return "thought", thought.group(1).strip()
        if text.startswith(">"):
            return "action", re.sub(r"^>\s*", "", text).lower().strip()
        if cls._ACTION_PATTERN.match(text):
            return "action", re.sub(r"^>\s*", "", text).lower().strip()
        # This fallback is exactly the source parser's behaviour: malformed
        # output is treated as another thought, not silently executed.
        return "thought", text

    @staticmethod
    def _parse_webshop_output(output: str) -> tuple[str, str]:
        """Match ExpeL's WebShop parser: every response is one environment action."""
        text = re.sub(r"(?i)^\s*action\s*\d*\s*:\s*", "", (output or "").strip())
        if "[" not in text:
            text = f"think[{text.removeprefix('Observation:').strip()}]"
        elif not text.endswith("]"):
            text += "]"
        return "action", text

    def _raw_task_type(self, task_id: int) -> str:
        return str(self._dataset.get_task_type(task_id))

    @staticmethod
    def _task_key(goal: str, task_type: str) -> str:
        value = f"{task_type}\0{goal.strip().casefold()}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()[:20]

    def _task_experiences(self, task_key: str) -> list[dict[str, Any]]:
        wanted = set(self._by_task.get(task_key, []))
        return [item for item in self._experiences if item.get("experience_id") in wanted]

    def _policy_hash(self) -> str:
        visible_successes = sorted(
            item["experience_id"] for item in self._experiences if item.get("success")
        )
        reflections = {
            key: [item.get("text", "") for item in values]
            for key, values in sorted(self._task_reflections.items())
        }
        payload = {
            "rules": self._rule_bank.hash(),
            "successes": visible_successes,
            "reflections": reflections,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

    def _global_knowledge_hash(self) -> str:
        """Hash reusable knowledge, excluding task-local retry reflections."""
        first_successes = sorted(
            str(item["task_key"])
            for item in self._first_successes()
        )
        payload = {
            "rules": self._rule_bank.hash(),
            "successful_task_keys": first_successes,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]

    def _retrieved_demonstrations(self, task_id: int, goal: str, task_type: str):
        if not self._actor_uses_learned_memory or self._top_k == 0:
            return []
        return retrieve_successes(
            experiences=self._experiences,
            query_embedding=self._dataset.task_pool.get_embedding(task_id),
            task_type=task_type,
            current_goal=goal,
            # The bundled ALFWorld prompt and the reference configs use two
            # fewshots. Retrieved trials replace, rather than extend, them.
            top_k=min(self._top_k, 2),
            max_fewshot_tokens=self._max_fewshot_tokens,
        )

    @property
    def _actor_uses_learned_memory(self) -> bool:
        # The option controls training/evolving only. Evaluation constructs a
        # read-only agent (dynamic updates disabled) and must always consume
        # the rules and successful demonstrations learned during training.
        return not self._dynamic_updates or self._use_memory_during_evolve

    def _run_trial(self, task_id: int, extra_reflections: list[str] | None = None) -> dict[str, Any]:
        goal = self._dataset.get_task_goal(task_id)
        task_type = self._raw_task_type(task_id)
        task_key = self._task_key(goal, task_type)
        global_knowledge_hash_before = self._global_knowledge_hash()
        stored_reflections = [
            str(item.get("text", "")) for item in self._task_reflections.get(task_key, [])
        ]
        reflections = stored_reflections + list(extra_reflections or [])
        demonstrations = self._retrieved_demonstrations(task_id, goal, task_type)
        rules = self._rule_bank.render() if self._actor_uses_learned_memory else ""

        env = self._dataset.create_env(task_id)
        started_calls = self._total_calls
        try:
            state = self._dataset.reset_env(env)
            initial_observation = str(state.observation)
            system, base_context = build_actor_context(
                assets=self._assets,
                task_type=task_type,
                task_goal=goal,
                initial_observation=initial_observation,
                max_steps=self.max_turns,
                rules=rules,
                retrieved_demonstrations=demonstrations,
                reflections=reflections,
            )
            transcript: list[str] = []
            steps: list[dict[str, Any]] = []
            actions: list[str] = []
            model_outputs: list[str] = []
            success = bool(state.success)
            done = bool(state.done)
            current_observation = str(state.observation)
            admissible = list(state.admissible_actions)
            for step_index in range(self.max_turns):
                if done or success:
                    break
                thought_count = 0
                action = "N/A"
                while True:
                    prompt = base_context
                    if transcript:
                        prompt += "\n" + "\n".join(transcript)
                    prompt += "\nAction:" if self._benchmark == "webshop" else "\n>"
                    output = self._chat(
                        kind="actor", system=system, user=prompt,
                        max_tokens=self._max_tokens, temperature=self._temperature,
                        stop=["\n"],
                    )
                    model_outputs.append(output)
                    message_type, content = (
                        self._parse_webshop_output(output)
                        if self._benchmark == "webshop" else self._parse_react_output(output)
                    )
                    if message_type == "action":
                        action = content
                        break
                    thought_count += 1
                    transcript.append(
                        f"Action: think[{content}]\nObservation: OK."
                        if self._benchmark == "webshop" else f"> think: {content}\nOK."
                    )
                    if thought_count > 2:
                        action = "N/A"
                        break
                before = current_observation
                step = self._dataset.step_env(env, action)
                current_observation = (
                    "You are thinking too many times without taking action."
                    if action == "N/A" and thought_count > 2 else str(step.observation)
                )
                transcript.append(
                    f"Action: {action}\nObservation: {current_observation}"
                    if self._benchmark == "webshop" else f"> {action}\n{current_observation}"
                )
                actions.append(action)
                steps.append({
                    "step": step_index + 1,
                    "observation": before,
                    "admissible_actions": admissible,
                    "action": action,
                    "next_observation": current_observation,
                    "success": bool(step.success),
                    "done": bool(step.done),
                })
                admissible = list(step.admissible_actions)
                success = bool(step.success)
                done = bool(step.done)
        finally:
            self._dataset.close_env(env)

        trajectory = "\n".join(transcript)
        result = {
            "success": success,
            "trajectory": trajectory,
            "trajectory_steps": steps,
            "actions": actions,
            "api_calls": self._total_calls - started_calls,
            "model_outputs": model_outputs,
            "task_goal": goal,
            "task_type": task_type,
            "task_key": task_key,
            "global_knowledge_hash_before": global_knowledge_hash_before,
            "initial_observation": initial_observation,
            "retrieved_experience_ids": [item["experience_id"] for item in demonstrations],
            "rule_ids": [rule.rule_id for rule in self._rule_bank.rules],
            "reflection_ids": [item.get("reflection_id") for item in self._task_reflections.get(task_key, [])],
            "prompt_manifest": {
                "official_source_sha256": self._assets.source_sha256,
                "rules": [vars(rule) for rule in self._rule_bank.rules],
                "retrieved_experience_ids": [item["experience_id"] for item in demonstrations],
                "task_reflections": reflections,
            },
        }
        self._append("retrievals", {
            "event": "retrieval",
            "decision_id": self._decision_id,
            "task_id": int(task_id),
            "task_key": task_key,
            "experience_ids": result["retrieved_experience_ids"],
            "rule_ids": result["rule_ids"],
            "reflection_ids": result["reflection_ids"],
            "timestamp": time.time(),
        })
        self._append("trials", {
            "event": "trial",
            "decision_id": self._decision_id,
            "task_id": int(task_id),
            **result,
            "timestamp": time.time(),
        })
        return result

    def execute(self, task_id: int, num_rollouts: int = 1) -> dict:
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be at least 1")
        self._decision_id += 1
        started_calls = self._total_calls
        rollouts: list[dict[str, Any]] = []
        logical_outcomes: list[bool] = []
        representative_rollouts: list[dict[str, Any]] = []
        attempt_counts: list[int] = []
        staged_reflections: list[dict[str, Any]] = []
        for _ in range(num_rollouts):
            local_reflections: list[str] = []
            if self._mode == "paper_faithful":
                goal = self._dataset.get_task_goal(task_id)
                task_key = self._task_key(goal, self._raw_task_type(task_id))
                remaining_reflections = max(
                    0,
                    self._max_reflection_depth
                    - len(self._task_reflections.get(task_key, [])),
                )
                attempts = 1 + remaining_reflections
            else:
                attempts = 1
            attempt_count = 0
            for attempt in range(attempts):
                rollout = self._run_trial(task_id, local_reflections)
                rollouts.append(rollout)
                attempt_count += 1
                if rollout["success"] or self._mode != "paper_faithful" or attempt + 1 >= attempts:
                    break
                reflection = self._generate_task_reflection(task_id, rollout, staged=True)
                if reflection is None:
                    break
                staged_reflections.append(reflection)
                local_reflections.append(reflection["text"])
            logical_outcomes.append(bool(rollout["success"]))
            representative_rollouts.append(rollout)
            attempt_counts.append(attempt_count)

        representative = representative_rollouts[0]
        return {
            "success": sum(logical_outcomes) * 2 >= len(logical_outcomes),
            "successes": sum(logical_outcomes),
            "num_rollouts": len(logical_outcomes),
            "success_rate": sum(logical_outcomes) / len(logical_outcomes),
            "rollout_successes": logical_outcomes,
            "rollout_results": rollouts,
            "trajectory": representative["trajectory"],
            "trajectories": [item["trajectory"] for item in rollouts],
            "trajectory_steps": representative["trajectory_steps"],
            "actions": representative["actions"],
            "api_calls": self._total_calls - started_calls,
            "expel_decision_id": self._decision_id,
            "staged_reflections": staged_reflections,
            "trial_attempt_counts": attempt_counts,
        }

    # ------------------------------------------------------------------
    # Task reflection and experience memory

    def _generate_task_reflection(
        self, task_id: int, rollout: dict[str, Any], *, staged: bool,
    ) -> dict[str, Any] | None:
        if not self._dynamic_updates:
            return None
        task_key = str(rollout["task_key"])
        existing_count = len(self._task_reflections.get(task_key, []))
        if existing_count >= self._max_reflection_depth and not staged:
            return None
        system, user = build_reflection_prompt(
            assets=self._assets,
            initial_observation=str(rollout.get("initial_observation", "")),
            task_goal=str(rollout["task_goal"]),
            trajectory=str(rollout["trajectory"])[-self._max_prompt_trajectory_chars:],
        )
        started = time.time()
        response = ""
        error = None
        try:
            response = self._chat(
                kind="reflection", system=system, user=user,
                max_tokens=self._reflection_max_tokens,
                temperature=self._reflection_temperature,
            )
            prefix = "Next" if self._benchmark == "webshop" else "New"
            reflection_text = re.sub(
                rf"^\s*(?:STATUS:\s*FAIL\s*)?(?:{prefix}\s+plan\s*:\s*)?",
                "", response, flags=re.IGNORECASE,
            ).strip()
            if not reflection_text:
                raise ValueError(f"reflection model returned an empty {prefix} plan")
        except Exception as exc:
            reflection_text = ""
            error = f"{type(exc).__name__}: {exc}"
        payload = {
            "event": "task_reflection",
            "decision_id": self._decision_id,
            "task_id": int(task_id),
            "task_key": task_key,
            "task_goal": rollout["task_goal"],
            "prompt": {"system": system, "user": user},
            "response": response,
            "text": reflection_text,
            "status": "ok" if error is None else "error",
            "error": error,
            "staged": bool(staged),
            "latency_s": round(time.time() - started, 4),
            "model": self._reflection_model,
            "timestamp": time.time(),
        }
        self._append("task_reflections", payload)
        if error is not None:
            self._append("errors", payload)
            self._diagnostics["task_reflection_errors"] += 1
            return None
        return payload

    def _store_reflection(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        task_key = str(payload["task_key"])
        if len(self._task_reflections[task_key]) >= self._max_reflection_depth:
            return None
        self._reflection_id += 1
        item = {
            "reflection_id": f"reflection_{self._reflection_id:06d}",
            "decision_id": int(payload.get("decision_id", self._decision_id)),
            "task_id": int(payload["task_id"]),
            "task_key": task_key,
            "task_goal": payload["task_goal"],
            "text": str(payload["text"]),
            "timestamp": time.time(),
        }
        self._task_reflections[task_key].append(item)
        self._append("reflection_memory", {"event": "reflection_stored", **item})
        return item

    def _store_experience(
        self, task_id: int, rollout: dict[str, Any],
    ) -> dict[str, Any] | None:
        task_key = str(rollout["task_key"])
        if rollout.get("success") and any(
            item.get("success") for item in self._task_experiences(task_key)
        ):
            # One successful demonstration per task is sufficient for ExpeL
            # retrieval and success/failure comparison. Re-storing identical
            # solved-task evidence made policy hashes change forever and gave
            # the selector a false signal that repeated successes add skills.
            self._diagnostics["duplicate_successes_skipped"] += 1
            return None
        self._experience_id += 1
        item = {
            "experience_id": f"experience_{self._experience_id:07d}",
            "decision_id": self._decision_id,
            "task_id": int(task_id),
            "source_task_key": task_key,
            "task_key": task_key,
            "task_goal": str(rollout["task_goal"]),
            "task_type": str(rollout["task_type"]),
            "task_embedding": self._dataset.task_pool.get_embedding(task_id).tolist(),
            "initial_observation": str(rollout.get("initial_observation", "")),
            "success": bool(rollout["success"]),
            "trajectory": str(rollout.get("trajectory", "")),
            "trajectory_steps": list(rollout.get("trajectory_steps", [])),
            "actions": list(rollout.get("actions", [])),
            "api_calls": int(rollout.get("api_calls", 0)),
            "global_knowledge_hash_before": str(
                rollout.get("global_knowledge_hash_before", "")
            ),
            "timestamp": time.time(),
        }
        self._experiences.append(item)
        self._by_task[item["task_key"]].append(item["experience_id"])
        self._append("experiences", {"event": "experience_stored", **item})
        return item

    # ------------------------------------------------------------------
    # Global ExpeL insight extraction

    def _run_insight_job(
        self,
        *,
        kind: str,
        experience_ids: list[str],
        success_history: str,
        failure_history: str | None = None,
        task_goal: str | None = None,
    ) -> dict[str, Any]:
        self._update_step += 1
        system, user = build_insight_prompt(
            kind=kind,
            rules=self._rule_bank.render(),
            success_history=success_history[-self._max_prompt_trajectory_chars:],
            failure_history=(failure_history or "")[-self._max_prompt_trajectory_chars:],
            task_goal=task_goal,
            list_full=self._rule_bank.is_full,
        )
        prompt_payload = {
            "event": "insight_prompt",
            "update_step": self._update_step,
            "kind": kind,
            "experience_ids": experience_ids,
            "system": system,
            "user": user,
            "timestamp": time.time(),
        }
        self._append("insight_prompts", prompt_payload)
        if self._expel_dir is not None:
            self._atomic_json(
                self._expel_dir / "prompts" / f"insight_{self._update_step:06d}.json",
                prompt_payload,
            )

        before_hash = self._rule_bank.hash()
        response = ""
        error = None
        started = time.time()
        try:
            response = self._chat(
                kind="insight", system=system, user=user,
                max_tokens=self._insight_max_tokens,
                temperature=self._reflection_temperature,
            )
            if not response.strip():
                raise ValueError("insight model returned an empty response")
            operations = parse_operations(response)
            changed, applied = self._rule_bank.apply(operations, self._update_step)
            status = "updated" if changed else "no_change"
        except Exception as exc:
            operations = []
            changed = False
            applied = []
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            self._diagnostics["insight_errors"] += 1
        event = {
            "event": "insight_update",
            "update_step": self._update_step,
            "kind": kind,
            "status": status,
            "experience_ids": experience_ids,
            "response": response,
            "operations": [vars(operation) for operation in operations],
            "applied_operations": applied,
            "skill_updated": changed,
            "before_rule_hash": before_hash,
            "after_rule_hash": self._rule_bank.hash(),
            "error": error,
            "latency_s": round(time.time() - started, 4),
            "model": self._reflection_model,
            "timestamp": time.time(),
        }
        self._append("insight_updates", event)
        if error:
            self._append("errors", event)
        if changed:
            self._diagnostics["rule_updates"] += 1
            self._snapshot_rules()
        else:
            self._diagnostics["rule_no_change"] += 1
        return event

    def _pair_jobs(self) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        jobs = []
        for task_key in sorted(self._by_task):
            trials = self._task_experiences(task_key)
            successes = [item for item in trials if item["success"]]
            failures = [item for item in trials if not item["success"]]
            for success in successes:
                for failure in failures:
                    pair_id = f"{success['experience_id']}::{failure['experience_id']}"
                    if pair_id not in self._processed_pairs:
                        jobs.append((pair_id, success, failure))
        return jobs

    def _first_successes(self) -> list[dict[str, Any]]:
        first = []
        for task_key in sorted(self._by_task):
            successes = [item for item in self._task_experiences(task_key) if item["success"]]
            if successes:
                first.append(min(successes, key=lambda item: item["experience_id"]))
        first.sort(key=lambda item: item["experience_id"])
        return first

    def _success_chunks(self, *, include_partial: bool) -> list[tuple[str, list[dict[str, Any]]]]:
        successes = self._first_successes()
        chunks = []
        for start in range(0, len(successes), self._success_critique_num):
            chunk = successes[start:start + self._success_critique_num]
            if len(chunk) < self._success_critique_num and not include_partial:
                continue
            chunk_id = "::".join(item["experience_id"] for item in chunk)
            if chunk and chunk_id not in self._processed_success_chunks:
                chunks.append((chunk_id, chunk))
        return chunks

    def _extract_available_insights(self, *, include_partial_success: bool) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for pair_id, success, failure in self._pair_jobs():
            event = self._run_insight_job(
                kind="compare",
                experience_ids=[success["experience_id"], failure["experience_id"]],
                success_history=success["trajectory"],
                failure_history=failure["trajectory"],
                task_goal=self._format_task_for_insight(success),
            )
            if event["status"] != "error":
                self._processed_pairs.add(pair_id)
            events.append(event)
        for chunk_id, chunk in self._success_chunks(include_partial=include_partial_success):
            success_history = "\n\n".join(
                f"{self._format_task_for_insight(item)}"
                f"\n{item['trajectory']}"
                for item in chunk
            )
            event = self._run_insight_job(
                kind="all_success",
                experience_ids=[item["experience_id"] for item in chunk],
                success_history=success_history,
            )
            if event["status"] != "error":
                self._processed_success_chunks.add(chunk_id)
            events.append(event)
        return events

    def _format_task_for_insight(self, item: dict[str, Any]) -> str:
        if self._benchmark == "webshop":
            return format_webshop_task(str(item["task_goal"]))
        return format_alfworld_task(
            str(item.get("initial_observation", "")), str(item["task_goal"]),
        )

    def will_update_after_reflect(self, task_id: int, result: dict) -> bool:
        del task_id, result
        # Used only by the optional probe ablation.  Every stored success or
        # task reflection can alter ExpeL's future policy, even without a rule
        # operation.
        return self._dynamic_updates

    def reflect(self, task_id: int, result: dict):
        before_hash = self._policy_hash()
        stored: list[dict[str, Any]] = []
        stored_reflections: list[dict[str, Any]] = []
        for rollout in result.get("rollout_results", [result]):
            item = self._store_experience(task_id, rollout)
            if item is not None:
                stored.append(item)

        for payload in result.get("staged_reflections", []):
            item = self._store_reflection(payload)
            if item:
                stored_reflections.append(item)

        if self._dynamic_updates and self._mode == "spg_online":
            for rollout in result.get("rollout_results", [result]):
                if rollout.get("success"):
                    continue
                task_key = str(rollout["task_key"])
                if len(self._task_reflections[task_key]) >= self._max_reflection_depth:
                    continue
                payload = self._generate_task_reflection(task_id, rollout, staged=False)
                if payload:
                    item = self._store_reflection(payload)
                    if item:
                        stored_reflections.append(item)

        insight_events: list[dict[str, Any]] = []
        if self._dynamic_updates and self._insight_strategy == "incremental":
            insight_events = self._extract_available_insights(include_partial_success=False)
        after_hash = self._policy_hash()
        event = {
            "event": "expel_learning_step",
            "decision_id": int(result.get("expel_decision_id", self._decision_id)),
            "task_id": int(task_id),
            "task_ids": [int(task_id)],
            "experience_ids": [item["experience_id"] for item in stored],
            "reflection_ids": [item["reflection_id"] for item in stored_reflections],
            "insight_update_steps": [item["update_step"] for item in insight_events],
            "skill_update_completed": True,
            "skill_updated": before_hash != after_hash,
            "bandit_label_ready": True,
            "before_policy_hash": before_hash,
            "after_policy_hash": after_hash,
            "timestamp": time.time(),
        }
        self._append("learning_steps", event)
        self._persist()
        return [event]

    # ------------------------------------------------------------------
    # Selector features and lifecycle

    @property
    def selection_feature_dim(self) -> int:
        return 5

    def get_selection_context(self, task_id: int) -> SelectionContext:
        goal = self._dataset.get_task_goal(task_id)
        task_type = self._raw_task_type(task_id)
        task_key = self._task_key(goal, task_type)
        trials = self._task_experiences(task_key)
        reflections = self._task_reflections.get(task_key, [])
        remaining_ratio = max(
            0.0,
            (self._max_reflection_depth - len(reflections))
            / max(self._max_reflection_depth, 1),
        )
        if not trials:
            return SelectionContext(
                features=np.asarray([0.0, 0.0, 0.0, remaining_ratio, 0.0]),
                eligible=True,
                reason="unseen",
                policy_version=self._global_knowledge_hash(),
            )

        latest = trials[-1]
        latest_decision = int(latest.get("decision_id", -1))
        latest_succeeded = bool(latest.get("success"))
        untested_reflection = any(
            int(item.get("decision_id", -2)) == latest_decision
            for item in reflections
        )
        current_policy = self._global_knowledge_hash()
        previous_policy = str(latest.get("global_knowledge_hash_before", ""))
        global_policy_changed = bool(previous_policy and previous_policy != current_policy)
        features = np.asarray([
            1.0,
            float(latest_succeeded),
            float(untested_reflection),
            remaining_ratio,
            float(global_policy_changed),
        ])

        if latest_succeeded:
            eligible, reason = False, "solved"
        elif untested_reflection:
            eligible, reason = True, "new_reflection"
        elif global_policy_changed:
            eligible, reason = True, "global_policy_changed"
        elif len(reflections) >= self._max_reflection_depth:
            eligible, reason = False, "reflection_budget_exhausted"
        else:
            eligible, reason = False, "no_new_learning_signal"
        return SelectionContext(
            features=features,
            eligible=eligible,
            reason=reason,
            policy_version=current_policy,
        )

    def get_selection_features(self, task_id: int):
        """Compatibility view for callers predating SelectionContext."""
        return self.get_selection_context(task_id).features

    def get_usage(self) -> dict:
        return {
            "api_calls": self._total_calls,
            "actor_calls": self._actor_calls,
            "reflection_calls": self._reflection_calls,
            "insight_calls": self._insight_calls,
            "rule_count": len(self._rule_bank.rules),
            "experience_count": len(self._experiences),
            "task_reflection_count": sum(len(value) for value in self._task_reflections.values()),
            "buffered_rollouts": 0,
        }

    def save_checkpoint(self) -> dict:
        return self._state_payload()

    def load_checkpoint(self, state: dict):
        if state:
            self._restore_state(state)
            self._persist()

    def finalize(self):
        events: list[dict[str, Any]] = []
        if self._dynamic_updates:
            # Official ExpeL also analyses the final (possibly short) group of
            # successful tasks.  Deferred mode performs all pair jobs here;
            # incremental mode only has the last success chunk left.
            insight_events = self._extract_available_insights(include_partial_success=True)
            if insight_events:
                events.append({
                    "event": "expel_finalize",
                    "skill_update_completed": True,
                    "skill_updated": any(item["skill_updated"] for item in insight_events),
                    "bandit_label_ready": True,
                    "task_ids": [],
                    "insight_update_steps": [item["update_step"] for item in insight_events],
                    "timestamp": time.time(),
                })
        self._persist()
        self._append("lifecycle", {
            "event": "finalize", "usage": self.get_usage(), "timestamp": time.time(),
        })
        return events

    def reset(self):
        # Learned memory persists; only per-process usage counters reset.
        self._total_calls = 0
        self._actor_calls = 0
        self._reflection_calls = 0
        self._insight_calls = 0

    def record_gain_measurement(self, payload: dict[str, Any]):
        """Keep optional probe-ablation measurements in ExpeL's audit log."""
        self._append("gain_measurements", {**payload, "timestamp": time.time()})
