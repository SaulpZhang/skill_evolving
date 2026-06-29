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

    @abstractmethod
    def get_usage(self) -> dict:
        """Return token/call usage info."""

    @abstractmethod
    def reset(self):
        """Reset to initial state (for new runs)."""
