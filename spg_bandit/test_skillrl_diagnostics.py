import json
from pathlib import Path

from spg_bandit.modules.skill_evolving.skillrl.agent import SkillRLAgent
from spg_bandit.modules.skillrl_source import SkillUpdater


class _Dataset:
    def get_task_goal(self, task_id):
        return "put the object away"

    def get_task_type(self, task_id):
        return "pick_and_place"


class _Memory:
    skills = {"general_skills": [], "task_specific_skills": {}}

    def retrieve(self, goal, top_k=0):
        return {"task_type": "pick_and_place"}

    def add_skills(self, skills, category="general"):
        return 0


class _EmptyUpdater:
    last_update_status = {"status": "empty_parse"}

    def analyze_failures(self, failures, skills):
        return []


def _agent(tmp_path: Path):
    agent = SkillRLAgent.__new__(SkillRLAgent)
    agent._dynamic_updates = True
    agent._memory = _Memory()
    agent._updater = _EmptyUpdater()
    agent._dataset = _Dataset()
    agent._update_batch_size = 1
    agent._skill_update_freq = 1
    agent._update_threshold = 0.8
    agent._groups_since_update = 0
    agent._virtual_batch_step = 0
    agent._outcomes_by_type = {}
    agent._failed_trajectories = []
    agent._records_dir = tmp_path
    agent._skill_path = None
    agent._update_diagnostics = agent._new_update_diagnostics()
    return agent


def test_reflection_flush_persists_empty_teacher_diagnostic(tmp_path):
    agent = _agent(tmp_path)

    agent.reflect(
        7,
        {
            "rollout_successes": [False],
            "rollout_results": [{"trajectory_steps": [{"action": "look", "observation": "x"}]}],
        },
    )

    assert agent._update_diagnostics["batches_checked"] == 1
    assert agent._update_diagnostics["gate_passed"] == 1
    assert agent._update_diagnostics["teacher_calls"] == 1
    assert agent._update_diagnostics["teacher_empty"] == 1
    event = json.loads((tmp_path / "skillrl_updates.jsonl").read_text().strip())
    assert event["reason"] == "empty_parse"
    assert event["generated"] == 0
    assert event["added"] == 0


def test_skill_updater_accepts_skills_without_model_supplied_ids():
    updater = SkillUpdater.__new__(SkillUpdater)

    parsed = updater._parse_skills_response(
        '{"skills": [{"title": "Check State First", '
        '"principle": "Verify the object state before acting.", '
        '"when_to_apply": "Before irreversible actions."}]}'
    )

    assert [skill["title"] for skill in parsed] == ["Check State First"]
