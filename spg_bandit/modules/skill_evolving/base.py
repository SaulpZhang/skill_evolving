"""Skill evolving method interface."""

from abc import ABC, abstractmethod


class BaseSkillEvolving(ABC):
    """Base class for skill evolving methods."""

    @abstractmethod
    def execute(self, task_id: int) -> dict:
        """Execute a task.

        Returns a dict with at minimum:
            {"success": bool, "trajectory": str, "api_calls": int}
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

    @abstractmethod
    def get_usage(self) -> dict:
        """Return token/call usage info."""

    @abstractmethod
    def reset(self):
        """Reset to initial state (for new runs)."""
