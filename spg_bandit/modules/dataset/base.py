"""Dataset and environment interfaces.

The selector only needs a :class:`TaskPool`, while skill-evolving agents need
an environment with a small, normalized protocol.  Keeping that protocol in
this module prevents agents from depending on one environment library (for
example TextWorld's batched ALFWorld API).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskPool:
    """A fixed pool of M tasks with embeddings and metadata."""
    embeddings: np.ndarray       # (M, d_c) LLM embeddings
    metadata: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.asarray(self.embeddings)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D array with shape (M, d_c)")
        if self.metadata and len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                "metadata must contain one entry per embedding "
                f"({len(self.metadata)} != {self.embeddings.shape[0]})"
            )
        # Dataset adapters may omit an explicit id because the pool index is
        # already the stable task id.  Fill it in once at the boundary.
        for task_id, item in enumerate(self.metadata):
            item.setdefault("id", task_id)

    @property
    def M(self) -> int:
        return self.embeddings.shape[0]

    @property
    def d_c(self) -> int:
        return self.embeddings.shape[1]

    def get_embedding(self, i: int) -> np.ndarray:
        return self.embeddings[i]

    def get_goal(self, i: int) -> str:
        return self.metadata[i].get("goal", "") if i < len(self.metadata) else ""

    def get_task_type(self, i: int) -> Any:
        """Return a task category used by warmup/skill memory.

        ``dim`` is retained for compatibility with the original ALFWorld
        implementation.  New datasets should prefer the readable
        ``task_type`` field; either value may be a string or integer.
        """
        if i >= len(self.metadata):
            return "default"
        item = self.metadata[i]
        return item.get("task_type", item.get("dim", "default"))

    @property
    def task_types(self) -> list[Any]:
        """Ordered unique task categories present in this pool."""
        values = [self.get_task_type(i) for i in range(self.M)]
        return list(dict.fromkeys(values))


@dataclass
class EnvironmentState:
    """Normalized state returned by ``BaseDataset.reset_env``."""

    observation: Any
    admissible_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    success: bool = False


@dataclass
class EnvironmentStep:
    """Normalized result returned by ``BaseDataset.step_env``."""

    observation: Any
    admissible_actions: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    success: bool = False


class BaseDataset(ABC):
    """Base class for datasets and their environment adapters.

    A concrete dataset must provide task loading and ``create_env``.  The
    default environment methods understand the common Gym/Gymnasium API;
    datasets with a batched or tool-specific API can override only the
    relevant methods.
    """

    name = "dataset"

    @property
    @abstractmethod
    def task_pool(self) -> TaskPool:
        ...

    @abstractmethod
    def get_task_goal(self, task_id: int) -> str:
        ...

    def get_task_type(self, task_id: int) -> str:
        """Return the normalized category for a task.

        Metadata is the source of truth, so task-type detection does not need
        to be duplicated in every agent implementation.
        """
        return str(self.task_pool.get_task_type(task_id))

    def get_skill_task_type(self, task_id: int) -> str:
        """Return the category used by skill retrieval/evolution.

        Most datasets can use their task type directly.  A dataset may map a
        fine-grained evaluation label to a coarser skill-bank category by
        overriding this method (ALFWorld does this for its six task labels).
        """
        return self.get_task_type(task_id)

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def create_env(self, task_id: int):
        """Create an environment instance for executing a specific task."""
        ...

    @staticmethod
    def _env_object(env_handle):
        """Return the actual environment from a dataset-specific handle."""
        return env_handle

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, (list, tuple, np.ndarray)):
            return bool(value[0]) if len(value) else False
        return bool(value)

    @staticmethod
    def _actions_from_info(info: dict[str, Any] | None) -> list[str]:
        info = info or {}
        # TextWorld calls this field ``admissible_commands`` while most Gym
        # adapters use ``admissible_actions`` (or the older ``actions``
        # alias).  Keep all three names in the normalized interface; without
        # the TextWorld spelling, ALFWorld prompts incorrectly contain [] at
        # the initial state.
        actions = info.get("admissible_actions")
        if actions is None:
            actions = info.get("admissible_commands", info.get("actions", []))
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        if isinstance(actions, (list, tuple)) and actions and isinstance(actions[0], (list, tuple)):
            actions = actions[0]
        if actions is None:
            return []
        return [str(action) for action in actions]

    def reset_env(self, env_handle) -> EnvironmentState:
        """Reset a standard Gym/Gymnasium environment and normalize its state."""
        env = self._env_object(env_handle)
        result = env.reset()
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            observation, info = result
        else:
            observation, info = result, {}
        return EnvironmentState(
            observation=observation,
            admissible_actions=self._actions_from_info(info),
            info=info,
        )

    def step_env(self, env_handle, action: str) -> EnvironmentStep:
        """Step a standard Gym/Gymnasium environment and normalize its result."""
        env = self._env_object(env_handle)
        result = env.step(action)
        if not isinstance(result, tuple):
            raise TypeError("Environment step must return a tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            observation, reward, done, info = result
            done = bool(done)
        else:
            raise ValueError(f"Unsupported environment step result with {len(result)} fields")
        info = info if isinstance(info, dict) else {}
        success = self._as_bool(info.get("success", info.get("won", False)))
        return EnvironmentStep(
            observation=observation,
            admissible_actions=self._actions_from_info(info),
            info=info,
            reward=float(reward) if isinstance(reward, (int, float, np.number)) else 0.0,
            done=done,
            success=success,
        )

    def close_env(self, env_handle):
        """Close an environment handle.  Dataset adapters may override this."""
        env = self._env_object(env_handle)
        close = getattr(env, "close", None)
        if close is not None:
            close()

    def build_action_prompt(
        self,
        *,
        task_goal: str,
        skill_section: str,
        observation: Any,
        admissible_actions: list[str],
        step: int,
        recent: list[tuple[str, str]],
        history_window: int,
    ) -> str:
        """Build a generic text-action prompt.

        Environments with a specialized prompt (ALFWorld, WebShop, tool
        use, etc.) can override this method without changing the agent.
        """
        history = "\n".join(
            f"[Observation {i}: '{obs}', Action {i}: '{action}']"
            for i, (obs, action) in enumerate(recent[-history_window:], 1)
        ) or "(none)"
        skills = skill_section or "(none)"
        actions = "; ".join(admissible_actions)
        return (
            "You are an agent operating in an interactive environment.\n"
            f"Your task is: {task_goal}\n\n"
            f"## Retrieved Relevant Experience\n\n{skills}\n\n"
            "## Current Progress\n\n"
            f"You have taken {step} step(s). Recent history: {history}\n"
            f"Current observation: {observation}\n"
            f"Available actions: [{actions}]\n\n"
            "Reason step-by-step inside <think> </think> tags, then put one "
            "action inside <action> </action> tags."
        )

    def build_reflection_prompt(
        self, *, outcome: str, task_goal: str, task_type: str,
        trajectory: str, existing_titles: list[str],
    ) -> str:
        """Build a generic reflection prompt for skill extraction."""
        return (
            "Analyze the trajectory below.\n\n"
            f"OUTCOME: {outcome}\nTASK: {task_goal}\nTASK TYPE: {task_type}\n"
            f"TRAJECTORY (last steps):\n{trajectory}\n\n"
            f"EXISTING SKILL TITLES (avoid duplicating these): {existing_titles}\n\n"
            "Return ONLY JSON. For success use planning_pattern, title and "
            "principle; for failure use mistakes_to_avoid with trigger_condition "
            "and bad_action."
        )
