"""ExpeL insight rules and the original strength-update semantics.

The reference implementation stores rules as ``(text, strength)`` tuples and
updates them through four textual operations.  This module keeps that public
behaviour while adding stable rule identifiers and serialisable audit data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RuleOperation:
    op: str
    text: str
    rule_number: int | None = None


@dataclass
class Rule:
    rule_id: str
    text: str
    strength: int
    created_update: int
    updated_update: int


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def parse_operations(llm_text: str) -> list[RuleOperation]:
    """Parse ExpeL's ``<OP> <N>: <RULE>`` format.

    The official parser is line-oriented.  JSON-array support is accepted as
    a compatibility fallback for checkpoints made by the previous adapter;
    generated prompts always request the official textual format.
    """
    text = (llm_text or "").strip()
    operations: list[RuleOperation] = []
    pattern = re.compile(
        r"^\s*(REMOVE|EDIT|ADD|AGREE)(?:\s+(\d+))?\s*:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(text):
        op = match.group(1).upper()
        rule_text = match.group(3).strip().strip("`")
        if not rule_text or any(
            re.search(rf"\b{word}\b", rule_text, re.IGNORECASE)
            for word in ("ADD", "AGREE", "EDIT")
        ):
            continue
        # The reference code rejects cut-off generations by requiring a full
        # sentence.  Accept the common terminal punctuation variants too.
        if rule_text[-1] not in ".!?。！？":
            continue
        operations.append(RuleOperation(
            op=op,
            text=rule_text,
            rule_number=int(match.group(2)) if match.group(2) else None,
        ))
    if operations:
        # The prompt requests at most four operations, but the reference
        # implementation does not silently discard additional parsed lines.
        return operations

    # Backward-compatible parse of the old adapter's JSON response format.
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    payload = fenced.group(1) if fenced else text
    if not payload.startswith("["):
        start, end = payload.find("["), payload.rfind("]")
        payload = payload[start:end + 1] if start >= 0 and end > start else ""
    try:
        values = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    for item in values:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "")).upper()
        if op not in {"ADD", "EDIT", "REMOVE", "AGREE"}:
            continue
        rule_text = str(
            item.get("text") or item.get("principle") or item.get("rule") or ""
        ).strip()
        if rule_text and rule_text[-1] not in ".!?。！？":
            rule_text += "."
        number = item.get("rule_number")
        operations.append(RuleOperation(
            op=op,
            text=rule_text,
            rule_number=int(number) if number is not None else None,
        ))
    return operations


class RuleBank:
    """Ordered ExpeL rule list with source-faithful counter updates."""

    def __init__(self, max_num_rules: int = 20):
        if max_num_rules < 1:
            raise ValueError("max_num_rules must be positive")
        self.max_num_rules = int(max_num_rules)
        self._rules: list[Rule] = []
        self._next_id = 1

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)

    @property
    def is_full(self) -> bool:
        # ExpeL starts asking the teacher to prefer REMOVE as soon as the
        # configured target size is reached.  This remains a soft limit.
        return len(self._rules) >= self.max_num_rules

    @property
    def uses_strong_remove(self) -> bool:
        # The source has a second, deliberately later threshold: REMOVE is
        # worth -3 instead of -1 only once the bank grows five rules beyond
        # its target size.
        return len(self._rules) >= self.max_num_rules + 5

    def render(self) -> str:
        return "\n".join(f"{index}. {rule.text}" for index, rule in enumerate(self._rules, 1))

    def hash(self) -> str:
        payload = [asdict(rule) for rule in self._rules]
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]

    def to_state(self) -> dict[str, Any]:
        return {
            "max_num_rules": self.max_num_rules,
            "next_id": self._next_id,
            "rules": [asdict(rule) for rule in self._rules],
        }

    def load_state(self, state: dict[str, Any] | list[Any]):
        self._rules = []
        if isinstance(state, list):
            raw_rules = state
        else:
            raw_rules = state.get("rules", [])
            self._next_id = int(state.get("next_id", 1))
        for index, item in enumerate(raw_rules, 1):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                text, strength = item
                raw = {}
            elif isinstance(item, dict):
                raw = item
                # Migration from the previous online adapter.
                text = raw.get("text") or raw.get("principle") or raw.get("title")
                strength = raw.get("strength", max(1, int(raw.get("use_count", 0)) + 1))
            else:
                continue
            if not str(text or "").strip():
                continue
            self._rules.append(Rule(
                rule_id=str(raw.get("rule_id", f"rule_{index:06d}")),
                text=str(text).strip(),
                strength=int(strength),
                created_update=int(raw.get("created_update", raw.get("created_update_step", 0))),
                updated_update=int(raw.get("updated_update", raw.get("created_update_step", 0))),
            ))
        self._rules = [rule for rule in self._rules if rule.strength > 0]
        self._rules.sort(key=lambda rule: rule.strength, reverse=True)
        self._next_id = max(self._next_id, len(self._rules) + 1)

    def _find_matching_index(self, operation: RuleOperation) -> int | None:
        wanted = _normalise(operation.text)
        if operation.rule_number is not None:
            index = operation.rule_number - 1
            if 0 <= index < len(self._rules):
                candidate = _normalise(self._rules[index].text)
                if not wanted or candidate in wanted:
                    return index
        for index, rule in enumerate(self._rules):
            candidate = _normalise(rule.text)
            if candidate in wanted:
                return index
        return None

    def _is_duplicate(self, text: str) -> bool:
        wanted = _normalise(text)
        return any(
            _normalise(rule.text) in wanted
            for rule in self._rules
        )

    def apply(self, operations: Iterable[RuleOperation], update_step: int) -> tuple[bool, list[dict[str, Any]]]:
        """Apply operations in ExpeL's REMOVE→AGREE→EDIT→ADD order."""
        strong_remove = self.uses_strong_remove
        prepared: list[RuleOperation] = []
        for operation in operations:
            op = operation.op.upper()
            if op == "ADD":
                if operation.text and not self._is_duplicate(operation.text):
                    prepared.append(operation)
                continue
            if op == "EDIT":
                if self._is_duplicate(operation.text):
                    match = self._find_matching_index(operation)
                    if match is None:
                        match = next(
                            (i for i, rule in enumerate(self._rules)
                             if _normalise(rule.text) in _normalise(operation.text)),
                            None,
                        )
                    if match is not None:
                        prepared.append(RuleOperation("AGREE", self._rules[match].text, match + 1))
                elif operation.rule_number is not None and 0 < operation.rule_number <= len(self._rules):
                    prepared.append(operation)
                continue
            if op in {"REMOVE", "AGREE"} and self._find_matching_index(operation) is not None:
                prepared.append(operation)

        before = self.hash()
        applied: list[dict[str, Any]] = []
        for op_name in ("REMOVE", "AGREE", "EDIT", "ADD"):
            for operation in prepared:
                if operation.op != op_name:
                    continue
                if op_name == "ADD":
                    rule = Rule(
                        rule_id=f"rule_{self._next_id:06d}",
                        text=operation.text,
                        strength=2,
                        created_update=update_step,
                        updated_update=update_step,
                    )
                    self._next_id += 1
                    self._rules.append(rule)
                    applied.append({"op": "ADD", "rule_id": rule.rule_id, "strength_delta": 2})
                    continue
                index = self._find_matching_index(operation)
                if op_name == "EDIT":
                    index = (operation.rule_number or 0) - 1
                    if not 0 <= index < len(self._rules):
                        continue
                if index is None:
                    continue
                rule = self._rules[index]
                if op_name == "REMOVE":
                    delta = -3 if strong_remove else -1
                    rule.strength += delta
                elif op_name == "AGREE":
                    delta = 1
                    rule.strength += delta
                else:
                    delta = 1
                    rule.text = operation.text
                    rule.strength += 1
                rule.updated_update = update_step
                applied.append({"op": op_name, "rule_id": rule.rule_id, "strength_delta": delta})

        self._rules = [rule for rule in self._rules if rule.strength > 0]
        # Python's stable sort preserves the reference implementation's order
        # for ties.
        self._rules.sort(key=lambda rule: rule.strength, reverse=True)
        return before != self.hash(), applied
