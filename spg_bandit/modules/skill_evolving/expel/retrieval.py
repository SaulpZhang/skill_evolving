"""Successful-trajectory retrieval for the embedded ExpeL runtime."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re
from typing import Any

import numpy as np


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(left @ right / (left_norm * right_norm))


def _lexical_vector(text: str) -> np.ndarray:
    vector = np.zeros(256, dtype=float)
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        vector[int.from_bytes(digest, "little") % len(vector)] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def approximate_tokens(text: str) -> int:
    # Exact tokenisation depends on the configured actor model.  This
    # conservative estimate is used only for ExpeL's per-fewshot length gate.
    return max(1, (len(text) + 2) // 3)


def retrieve_successes(
    *,
    experiences: list[dict[str, Any]],
    query_embedding: np.ndarray,
    task_type: str,
    current_goal: str,
    top_k: int,
    max_fewshot_tokens: int,
) -> list[dict[str, Any]]:
    """Retrieve one shortest successful trajectory per prior similar task.

    This mirrors ExpeL's task-similarity path: filter to the same ALFWorld
    environment/task family, exclude the exact current task, rank tasks by
    embedding similarity, and use the shortest successful trial for each.
    """
    if top_k <= 0:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in experiences:
        if not item.get("success") or str(item.get("task_type")) != str(task_type):
            continue
        goal = str(item.get("task_goal", ""))
        if goal.strip().casefold() == current_goal.strip().casefold():
            continue
        grouped[(goal, str(item.get("source_task_key", item.get("task_id", ""))))].append(item)

    ranked: list[tuple[float, dict[str, Any]]] = []
    query = np.asarray(query_embedding, dtype=float)
    lexical_query = _lexical_vector(current_goal)
    for trials in grouped.values():
        shortest = min(trials, key=lambda item: len(str(item.get("trajectory", ""))))
        trajectory = str(shortest.get("trajectory", ""))
        if approximate_tokens(trajectory) > max_fewshot_tokens:
            continue
        embedding = np.asarray(shortest.get("task_embedding", []), dtype=float)
        score = (
            _cosine(query, embedding)
            if embedding.shape == query.shape
            else _cosine(lexical_query, _lexical_vector(str(shortest.get("task_goal", ""))))
        )
        ranked.append((score, shortest))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[:top_k]]
