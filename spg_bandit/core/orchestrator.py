"""SPG-Bandit framework orchestrator.

Wraps a TaskSelector, a SkillEvolvingMethod, and a TaskPool
into a single step() → observe() → repeat loop.
"""

import numpy as np
from .interfaces import SkillEvolvingMethod, TaskSelector, TaskPool


class SPGBandit:
    """Main framework orchestrator.

    Usage:
        bandit = SPGBandit(task_pool, selector, method)
        for t in range(T):
            result = bandit.step()
        results = bandit.get_history()
    """

    def __init__(
        self,
        task_pool: TaskPool,
        selector: TaskSelector,
        method: SkillEvolvingMethod,
    ):
        self.task_pool = task_pool
        self.selector = selector
        self.method = method

        self._history: list[dict] = []
        self._t = 0

    @property
    def profile(self) -> np.ndarray:
        """Current skill profile s_t."""
        return self.method.get_profile()

    @property
    def t(self) -> int:
        return self._t

    def step(self) -> dict:
        """One complete iteration: select → execute → update profile → update selector.

        Returns a record dict with all step information.
        """
        profile_before = self.method.get_profile().copy()

        # 1. Select task
        task_id = self.selector.select(self.task_pool, profile_before)

        # 2. Execute task
        success = self.method.execute(task_id)

        # 3. Update skill profile
        self.method.update_profile(task_id, success)

        # 4. Observe outcome and update selector
        profile_after = self.method.get_profile().copy()
        self.selector.update(task_id, profile_before, profile_after, success)

        record = {
            "t": self._t,
            "task_id": task_id,
            "success": success,
            "profile_before": profile_before,
            "profile_after": profile_after,
            "delta": profile_after - profile_before,
            "usage": {
                "method": self.method.get_usage(),
                "selector": self.selector.get_usage(),
            },
        }
        self._history.append(record)
        self._t += 1
        return record

    def get_total_usage(self) -> dict:
        """Aggregate token/call usage across all steps."""
        total = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for rec in self._history:
            usage = rec.get("usage", {})
            for source in ("method", "selector"):
                u = usage.get(source, {})
                total["api_calls"] += u.get("api_calls", 0)
                total["prompt_tokens"] += u.get("prompt_tokens", 0)
                total["completion_tokens"] += u.get("completion_tokens", 0)
        total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
        return total

    def run(self, T: int) -> list[dict]:
        """Run for T steps; return all records."""
        for _ in range(T):
            self.step()
        return self._history

    def get_history(self) -> list[dict]:
        return self._history.copy()

    def reset(self) -> None:
        self.selector.reset()
        self._history = []
        self._t = 0
