"""Warmup task sampling utilities."""

import random

from spg_bandit.modules.dataset.base import TaskPool


def sample_type_balanced_task_ids(task_pool: TaskPool, n_samples: int,
                                  rng: random.Random) -> list[int]:
    """Sample tasks with near-equal coverage of every available task type.

    Each task is selected at most once. The sampler gives the next draw to a
    currently least-covered type that still has unselected tasks, so coverage
    is as balanced as type capacities permit.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")

    type_to_ids: dict[int, list[int]] = {}
    for item in task_pool.metadata:
        type_to_ids.setdefault(item["dim"], []).append(item["id"])
    if n_samples and not type_to_ids:
        raise ValueError("Cannot sample warmup tasks from an empty task pool")
    if n_samples > task_pool.M:
        raise ValueError("Warmup samples cannot exceed the number of evolve tasks")

    task_types = sorted(type_to_ids)
    counts = {task_type: 0 for task_type in task_types}
    for _ in range(n_samples):
        available = [
            task_type for task_type in task_types
            if counts[task_type] < len(type_to_ids[task_type])
        ]
        min_count = min(counts[task_type] for task_type in available)
        candidates = [task_type for task_type in available if counts[task_type] == min_count]
        counts[rng.choice(candidates)] += 1

    sampled = []
    for task_type in task_types:
        sampled.extend(rng.sample(type_to_ids[task_type], counts[task_type]))
    rng.shuffle(sampled)
    return sampled
