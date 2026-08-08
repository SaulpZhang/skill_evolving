"""Skill evolving method interface."""

from abc import ABC, abstractmethod


class BaseSkillEvolving(ABC):
    """Base class for skill evolving methods."""

    @abstractmethod
    def execute(self, task_id: int, num_rollouts: int = 1) -> dict:
        """Execute a task.

        Returns a dict with at minimum:
            {"success": bool, "trajectory": str, "api_calls": int}.

        Implementations that support grouped rollouts should additionally
        return ``rollout_successes`` (one Boolean per rollout), ``successes``
        and ``num_rollouts``.  All rollout outcomes belong to the same task
        selection and therefore share the profile before that selection.
        """

    def load_skills(self, skills_dir: str):
        """Load existing skills from a directory.
        Called before execution to make skills available to the agent.
        Default no-op.
        """

    def reflect(self, task_id: int, result: dict):
        """Optional: reflect on execution and evolve skills.
        Called by orchestrator after execute(). Default no-op.
        """

    def finalize(self):
        """Optional end-of-run hook.

        Batch-oriented methods can use this hook to flush buffered evidence
        after the orchestrator has selected its last task.  Existing methods
        remain step-local and therefore inherit the no-op implementation.
        """

    def save_checkpoint(self) -> dict:
        """Return resumable method state. Stateless methods return no data."""
        return {}

    def load_checkpoint(self, state: dict):
        """Restore method state written by :meth:`save_checkpoint`."""

    @abstractmethod
    def get_usage(self) -> dict:
        """Return token/call usage info."""

    @abstractmethod
    def reset(self):
        """Reset to initial state (for new runs)."""
