"""Online ExpeL-style skill evolution at the existing agent seam.

The runner owns task selection and evaluation.  This adapter owns only the
state ExpeL needs: successful/failed trajectories, retrieved demonstrations,
reflection prompts, and an editable rule bank.  It intentionally does not
instantiate a second ALFWorld loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from spg_bandit.modules.skill_evolving.simple_agent import SimpleAgent


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ExpelAgent(SimpleAgent):
    """A single-task-update online adaptation of ExpeL.

    A selected task may contain several independent rollouts.  Their evidence
    is stored immediately; when sufficient task-local evidence exists, the
    reflection model proposes JSON rule operations.  The returned event tells
    the runner whether a causal post-update probe label is available.
    """

    _STATE_VERSION = 1

    def __init__(self, dataset, max_turns: int = 30, records_dir: str | None = None,
                 config: dict[str, Any] | None = None):
        super().__init__(dataset, max_turns=max_turns, records_dir=records_dir)
        self._config = dict(config or {})
        self._dynamic_updates = bool(self._config.get("enable_dynamic_update", True))
        self._generation_temperature = float(self._config.get("temperature", 0.3))
        self._generation_max_tokens = int(self._config.get("max_tokens", 1024))
        self._history_window = int(self._config.get("history_length", 5))
        self._top_k = int(self._config.get("top_k", 2))
        self._max_rules = int(self._config.get("max_rules", 10))
        self._max_examples = int(self._config.get("max_examples_per_prompt", 2))
        self._min_evidence = int(self._config.get("min_evidence", 2))
        self._min_failures = int(self._config.get("min_failures_for_critique", 2))
        self._max_task_history = int(self._config.get("max_task_history", 12))
        self._reflection_max_tokens = int(self._config.get("reflection_max_tokens", 4096))
        self._reflection_temperature = float(self._config.get("reflection_temperature", 0.0))
        if self._top_k < 0 or self._max_rules < 1 or self._min_evidence < 1:
            raise ValueError("Invalid ExpeL retrieval/rule/evidence configuration")
        if not os.getenv("REFLECTION_MODEL") and self._dynamic_updates:
            raise EnvironmentError("Expel requires REFLECTION_MODEL when dynamic updates are enabled")

        self._skills_dir: Path | None = None
        self._expel_dir: Path | None = None
        self._rules: list[dict[str, Any]] = []
        self._experiences: list[dict[str, Any]] = []
        self._by_task: dict[int, list[str]] = defaultdict(list)
        self._experience_id = 0
        self._update_step = 0
        self._decision_id = 0
        self._reflection_calls = 0
        self._diagnostics = defaultdict(int)

    # ── durable state and logs ────────────────────────────────────────

    def load_skills(self, skills_dir: str):
        self._skills_dir = Path(skills_dir)
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        # ``records_dir`` is intentionally still reserved for per-action
        # messages.  Keep ExpeL's event stream beside it, under one run.
        self._expel_dir = (
            self._records_dir.parent / "expel"
            if self._records_dir is not None else self._skills_dir / "expel_logs"
        )
        for name in ("snapshots", "prompts"):
            (self._expel_dir / name).mkdir(parents=True, exist_ok=True)
        self._load_persisted_state()

    @property
    def _state_path(self) -> Path:
        assert self._skills_dir is not None
        return self._skills_dir / "expel_state.json"

    def _load_persisted_state(self):
        if self._skills_dir is None or not self._state_path.is_file():
            return
        try:
            payload = json.loads(self._state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid ExpeL state file: {self._state_path}") from exc
        self._restore_state(payload)

    def _persist_state(self):
        if self._skills_dir is None:
            return
        self._atomic_json(self._state_path, self._state_payload())

    def _append(self, stream: str, payload: dict[str, Any]):
        if self._expel_dir is None:
            return
        path = self._expel_dir / f"{stream}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any] | list[Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)

    def _state_payload(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "rules": self._rules,
            "experiences": self._experiences,
            "experience_id": self._experience_id,
            "update_step": self._update_step,
            "decision_id": self._decision_id,
            "diagnostics": dict(self._diagnostics),
        }

    def _restore_state(self, payload: dict[str, Any]):
        self._rules = list(payload.get("rules", []))
        self._experiences = list(payload.get("experiences", []))
        self._experience_id = int(payload.get("experience_id", len(self._experiences)))
        self._update_step = int(payload.get("update_step", 0))
        self._decision_id = int(payload.get("decision_id", 0))
        self._diagnostics = defaultdict(int, payload.get("diagnostics", {}))
        self._by_task = defaultdict(list)
        for item in self._experiences:
            self._by_task[int(item["task_id"])].append(str(item["experience_id"]))

    @staticmethod
    def _hash_rules(rules: list[dict[str, Any]]) -> str:
        data = json.dumps(rules, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]

    # ── execution and retrieval ───────────────────────────────────────

    def _get_skill_section(self, goal: str, task_type: str) -> str:
        rules = self._relevant_rules(task_type)
        examples = self._retrieve_successes(goal, task_type)
        # Count actual prompt use only.  ``_relevant_rules`` is also queried
        # by the selector for every candidate task, so mutating there would
        # turn the learned context into an accidental selection-frequency
        # feature.
        for rule in rules:
            rule["use_count"] = int(rule.get("use_count", 0)) + 1
        self._loaded_skill = {
            "rule_ids": [rule["rule_id"] for rule in rules],
            "experience_ids": [item["experience_id"] for item in examples],
        }
        if self._expel_dir is not None:
            self._append("retrievals", {
                "event": "retrieval",
                "decision_id": self._decision_id,
                "task_goal": goal,
                "task_type": task_type,
                "rule_ids": self._loaded_skill["rule_ids"],
                "experience_ids": self._loaded_skill["experience_ids"],
                "timestamp": time.time(),
            })
        sections: list[str] = []
        if rules:
            sections.append("### ExpeL Insights\n" + "\n".join(
                f"- **{rule['title']}**: {rule['principle']}"
                + (f" _Apply when: {rule['when_to_apply']}_" if rule.get("when_to_apply") else "")
                for rule in rules
            ))
        if examples:
            lines = ["### Retrieved Successful Experiences"]
            for item in examples:
                text = str(item.get("trajectory", ""))[-1000:]
                lines.append(f"- Task: {item['task_goal']}\n  Successful trace: {text}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections) if sections else "(none)"

    def _relevant_rules(self, task_type: str) -> list[dict[str, Any]]:
        specific = [r for r in self._rules if r.get("task_type") in (task_type, "general")]
        return sorted(specific, key=lambda r: (-int(r.get("use_count", 0)), r["rule_id"]))[:self._max_rules]

    @property
    def selection_feature_dim(self) -> int:
        # Counts, evidence completeness, and relevant-rule coverage.  These
        # are learned context, not an externally imposed repeat penalty.
        return 5

    @property
    def immediate_gain_attribution(self) -> bool:
        return True

    def get_selection_features(self, task_id: int):
        task_id = int(task_id)
        evidence = self._task_experiences(task_id)
        successes = sum(bool(item.get("success")) for item in evidence)
        failures = len(evidence) - successes
        task_type = self._dataset.get_skill_task_type(task_id)
        return np.asarray([
            np.log1p(successes),
            np.log1p(failures),
            float(successes > 0 and failures > 0),
            min(len(evidence), self._max_task_history) / self._max_task_history,
            min(len(self._relevant_rules(task_type)), self._max_rules) / self._max_rules,
        ], dtype=float)

    def _retrieve_successes(self, goal: str, task_type: str) -> list[dict[str, Any]]:
        candidates = [
            item for item in self._experiences
            if item.get("success") and item.get("task_type") == task_type
        ]
        if not candidates or self._top_k == 0:
            return []
        query = self._goal_vector(goal)
        ranked = []
        for item in candidates:
            value = float(query @ self._goal_vector(item["task_goal"]))
            ranked.append((value, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:self._top_k]]

    @staticmethod
    def _goal_vector(text: str) -> np.ndarray:
        # Dependency-free lexical retrieval: stable across actor/reflection
        # workers and easy to reconstruct after a checkpoint.  The framework's
        # task embeddings remain the selector representation.
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = np.zeros(256, dtype=float)
        for token in tokens:
            index = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "little")
            vector[index % len(vector)] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def execute(self, task_id: int, num_rollouts: int = 1) -> dict:
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be at least 1")
        self._decision_id += 1
        rollouts = [SimpleAgent.execute(self, task_id) for _ in range(num_rollouts)]
        outcomes = [bool(item["success"]) for item in rollouts]
        representative = rollouts[0]
        return {
            "success": sum(outcomes) * 2 >= num_rollouts,
            "successes": sum(outcomes),
            "num_rollouts": num_rollouts,
            "success_rate": sum(outcomes) / num_rollouts,
            "rollout_successes": outcomes,
            "rollout_results": rollouts,
            "trajectory": representative.get("trajectory", ""),
            "trajectories": [item.get("trajectory", "") for item in rollouts],
            "trajectory_steps": representative.get("trajectory_steps", []),
            "actions": representative.get("actions", []),
            "api_calls": sum(int(item.get("api_calls", 0)) for item in rollouts),
            "loaded_skill": representative.get("loaded_skill"),
            "expel_decision_id": self._decision_id,
        }

    # ── online insight extraction ─────────────────────────────────────

    def will_update_after_reflect(self, task_id: int, result: dict) -> bool:
        if not self._dynamic_updates:
            return False
        incoming = len(result.get("rollout_results", [result]))
        return len(self._by_task[int(task_id)]) + incoming >= self._min_evidence

    def reflect(self, task_id: int, result: dict):
        goal = self._dataset.get_task_goal(task_id)
        task_type = self._dataset.get_skill_task_type(task_id)
        decision_id = int(result.get("expel_decision_id", self._decision_id))
        new_items = self._store_experiences(task_id, goal, task_type, decision_id, result)
        if not self._dynamic_updates:
            return []
        evidence = self._task_experiences(task_id)
        if len(evidence) < self._min_evidence:
            self._append("rule_updates", {
                "event": "pending_evidence", "decision_id": decision_id,
                "task_id": task_id, "experience_ids": [x["experience_id"] for x in new_items],
                "evidence_count": len(evidence), "min_evidence": self._min_evidence,
                "bandit_label_ready": False, "timestamp": time.time(),
            })
            self._persist_state()
            return []
        return [self._attempt_update(task_id, goal, task_type, decision_id, evidence)]

    def _store_experiences(self, task_id, goal, task_type, decision_id, result):
        items = result.get("rollout_results", [result])
        stored = []
        for rollout in items:
            self._experience_id += 1
            item = {
                "experience_id": f"exp_{self._experience_id:06d}",
                "decision_id": decision_id,
                "task_id": int(task_id),
                "task_goal": goal,
                "task_type": task_type,
                "success": bool(rollout.get("success", False)),
                "trajectory": rollout.get("trajectory", ""),
                "actions": list(rollout.get("actions", [])),
                "api_calls": int(rollout.get("api_calls", 0)),
                "timestamp": time.time(),
            }
            self._experiences.append(item)
            self._by_task[int(task_id)].append(item["experience_id"])
            stored.append(item)
            self._append("experiences", item)
        # Bounded in-memory context while retaining full JSONL provenance.
        ids = self._by_task[int(task_id)]
        if len(ids) > self._max_task_history:
            self._by_task[int(task_id)] = ids[-self._max_task_history:]
        return stored

    def _task_experiences(self, task_id: int) -> list[dict[str, Any]]:
        wanted = set(self._by_task[int(task_id)])
        return [item for item in self._experiences if item["experience_id"] in wanted]

    def _attempt_update(self, task_id, goal, task_type, decision_id, evidence):
        self._update_step += 1
        before_hash = self._hash_rules(self._rules)
        successes = sum(bool(item["success"]) for item in evidence)
        failures = len(evidence) - successes
        # A small amount of mixed evidence is informative; a failure-only
        # trace needs repeated failures before we turn it into a general rule.
        # This makes ``min_failures_for_critique`` a real data-quality gate
        # rather than an inert config field.
        if failures and not successes and failures < self._min_failures:
            event = {
                "event": "rule_update", "status": "insufficient_failure_evidence",
                "skill_update_completed": False, "skill_updated": False,
                "bandit_label_ready": False, "update_step": self._update_step,
                "decision_id": decision_id, "task_id": task_id, "task_ids": [task_id],
                "experience_ids": [x["experience_id"] for x in evidence],
                "reason": "insufficient_failure_evidence", "timestamp": time.time(),
            }
            self._append("rule_updates", event)
            self._diagnostics["insufficient_failure_evidence"] += 1
            self._persist_state()
            return event
        reflection_mode = (
            "contrast_success_and_failure" if successes and failures
            else "repeated_failure_critique" if failures
            else "successful_pattern_extraction"
        )
        prompt = self._build_reflection_prompt(goal, task_type, evidence, reflection_mode)
        prompt_path = ""
        if self._expel_dir is not None:
            path = self._expel_dir / "prompts" / f"reflect_{self._update_step:06d}.json"
            self._atomic_json(path, {"prompt": prompt, "experience_ids": [x["experience_id"] for x in evidence]})
            prompt_path = str(path)
        raw_response = ""
        error = None
        started = time.time()
        try:
            self._reflection_calls += 1
            raw_response = self._chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self._reflection_max_tokens,
                client=self._reflect_client,
                model=self._reflect_model,
                temperature=self._reflection_temperature,
            )
            operations = self._parse_operations(raw_response)
        except Exception as exc:  # Persist failure; never turn it into zero gain.
            operations = None
            error = f"{type(exc).__name__}: {exc}"

        reflection = {
            "event": "reflection", "update_step": self._update_step,
            "decision_id": decision_id, "task_id": task_id,
            "experience_ids": [x["experience_id"] for x in evidence],
            "prompt_path": prompt_path, "response": raw_response,
            "error": error, "latency_s": round(time.time() - started, 4),
            "reflection_model": self._reflect_model, "timestamp": time.time(),
            "reflection_mode": reflection_mode,
        }
        if error:
            reflection["status"] = "api_error"
            self._append("reflections", reflection)
            self._append("rule_updates", {**reflection, "skill_update_completed": False,
                                           "bandit_label_ready": False})
            self._diagnostics["reflection_errors"] += 1
            self._persist_state()
            return {"skill_update_completed": False, "skill_updated": False,
                    "bandit_label_ready": False, "reason": "reflection_api_error",
                    "task_ids": [task_id], "decision_id": decision_id}
        if operations is None:
            reflection["status"] = "parse_error"
            self._append("reflections", reflection)
            self._append("rule_updates", {**reflection, "skill_update_completed": False,
                                           "bandit_label_ready": False})
            self._diagnostics["parse_errors"] += 1
            self._persist_state()
            return {"skill_update_completed": False, "skill_updated": False,
                    "bandit_label_ready": False, "reason": "reflection_parse_error",
                    "task_ids": [task_id], "decision_id": decision_id}

        changed, applied = self._apply_operations(operations, task_type)
        after_hash = self._hash_rules(self._rules)
        status = "updated" if changed else "attempted_no_change"
        reflection["status"] = status
        reflection["operations"] = operations
        self._append("reflections", reflection)
        event = {
            "event": "rule_update", "status": status,
            "skill_update_completed": True, "skill_updated": changed,
            "bandit_label_ready": True, "update_step": self._update_step,
            "decision_id": decision_id, "task_id": task_id, "task_ids": [task_id],
            "experience_ids": [x["experience_id"] for x in evidence],
            "operations": operations, "applied_operations": applied,
            "before_rule_hash": before_hash, "after_rule_hash": after_hash,
            "reason": status, "timestamp": time.time(),
        }
        self._append("rule_updates", event)
        if changed:
            self._snapshot_rules()
            self._diagnostics["rule_updates"] += 1
        else:
            self._diagnostics["no_change"] += 1
        self._persist_state()
        return event

    def _build_reflection_prompt(self, goal, task_type, evidence, reflection_mode):
        existing = [{"rule_id": r["rule_id"], "title": r["title"], "principle": r["principle"]}
                    for r in self._rules]
        traces = []
        for item in evidence[-self._max_task_history:]:
            traces.append({
                "experience_id": item["experience_id"], "outcome": "success" if item["success"] else "failure",
                "trajectory": item["trajectory"][-3000:],
            })
        return (
            "You are improving a reusable embodied-agent rule bank. Analyze the "
            "successful and failed trajectories for one task. Extract only rules that "
            "generalize beyond concrete object names. Avoid duplicating existing rules.\n\n"
            f"TASK: {goal}\nTASK TYPE: {task_type}\n"
            f"REFLECTION MODE: {reflection_mode}\n"
            f"EXISTING RULES: {json.dumps(existing, ensure_ascii=False)}\n"
            f"EVIDENCE: {json.dumps(traces, ensure_ascii=False)}\n\n"
            "Return ONLY a JSON array. Each element must be one operation: "
            "{\"op\": \"ADD\", \"title\": \"3-5 words\", \"principle\": \"1-2 sentences\", "
            "\"when_to_apply\": \"condition\", \"task_type\": \"general or current type\"}, "
            "or {\"op\": \"EDIT\", \"rule_id\": \"...\", \"title\": \"...\", "
            "\"principle\": \"...\", \"when_to_apply\": \"...\"}, "
            "or {\"op\": \"REMOVE\", \"rule_id\": \"...\"}. Return [] when no change is justified."
        )

    @staticmethod
    def _parse_operations(response: str) -> list[dict[str, Any]] | None:
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        if not text.startswith("["):
            start, end = text.find("["), text.rfind("]")
            if start >= 0 and end > start:
                text = text[start:end + 1]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            return None
        return parsed

    def _apply_operations(self, operations, default_task_type):
        changed = False
        applied = []
        rules_by_id = {str(rule["rule_id"]): rule for rule in self._rules}
        for operation in operations:
            op = str(operation.get("op", "")).upper()
            if op == "ADD":
                title = str(operation.get("title", "")).strip()
                principle = str(operation.get("principle", "")).strip()
                if not title or not principle or any(r["title"].lower() == title.lower() for r in self._rules):
                    continue
                if len(self._rules) >= self._max_rules:
                    continue
                rule_id = f"rule_{self._update_step:05d}_{len(self._rules):03d}"
                rule = {"rule_id": rule_id, "title": title, "principle": principle,
                        "when_to_apply": str(operation.get("when_to_apply", "")).strip(),
                        "task_type": str(operation.get("task_type") or default_task_type),
                        "use_count": 0, "created_update_step": self._update_step}
                self._rules.append(rule)
                rules_by_id[rule_id] = rule
                changed = True
                applied.append({"op": "ADD", "rule_id": rule_id})
            elif op == "EDIT":
                rule = rules_by_id.get(str(operation.get("rule_id", "")))
                if rule is None:
                    continue
                dirty = False
                for key in ("title", "principle", "when_to_apply"):
                    if operation.get(key) and str(operation[key]).strip() != rule.get(key, ""):
                        rule[key] = str(operation[key]).strip()
                        dirty = True
                if dirty:
                    changed = True
                    applied.append({"op": "EDIT", "rule_id": rule["rule_id"]})
            elif op == "REMOVE":
                rule_id = str(operation.get("rule_id", ""))
                rule = rules_by_id.get(rule_id)
                if rule is not None:
                    self._rules.remove(rule)
                    rules_by_id.pop(rule_id, None)
                    changed = True
                    applied.append({"op": "REMOVE", "rule_id": rule_id})
        return changed, applied

    def _snapshot_rules(self):
        if self._expel_dir is None:
            return
        payload = {"update_step": self._update_step, "rules": self._rules,
                   "rule_hash": self._hash_rules(self._rules)}
        self._atomic_json(self._expel_dir / "current_rules.json", payload)
        self._atomic_json(self._expel_dir / "snapshots" / f"rules_{self._update_step:06d}.json", payload)

    def record_gain_measurement(self, payload: dict[str, Any]):
        """Receive runner-owned probe/MIRT evidence into the ExpeL log stream."""
        self._append("gain_measurements", {**payload, "timestamp": time.time()})

    # ── lifecycle ─────────────────────────────────────────────────────

    def get_usage(self) -> dict:
        return {"api_calls": self._total_calls, "reflection_calls": self._reflection_calls,
                "rule_count": len(self._rules), "experience_count": len(self._experiences)}

    def save_checkpoint(self) -> dict:
        return self._state_payload()

    def load_checkpoint(self, state: dict):
        if state:
            self._restore_state(state)
            self._persist_state()

    def finalize(self):
        self._persist_state()
        return []

    def reset(self):
        # Evaluation creates a new adapter then loads this persisted skill
        # state.  Do not clear learned rules or demonstrations here.
        super().reset()
        self._reflection_calls = 0
