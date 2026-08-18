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

    @property
    def selection_feature_dim(self) -> int:
        """Number of task-state features exposed to a stateful selector."""
        return 0

    def get_selection_features(self, task_id: int):
        """Return current task-local learning-state features.

        The default deliberately exposes no method-specific state.  ExpeL
        overrides this so SPG can learn marginal value without hand-written
        repetition penalties.
        """
        del task_id
        return []

    @property
    def immediate_gain_attribution(self) -> bool:
        """Whether an unlabeled selection must not be credited to a later update."""
        return False

    def reflect(self, task_id: int, result: dict):
        """Optional: reflect on execution and evolve skills.
        Called by orchestrator after execute(). Default no-op.

        Implementations may return a list of update events.  Each event has
        ``skill_update_completed``, ``skill_updated``, and the selected
        ``task_ids`` whose trajectories were used by that update.
        """

    def will_update_after_reflect(self, task_id: int, result: dict) -> bool:
        """Whether ``reflect`` will apply a skill update for this selection.

        The runner uses this to take a pre-update probe measurement.  The
        default keeps non-batched methods backwards compatible.
        """
        return False

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
