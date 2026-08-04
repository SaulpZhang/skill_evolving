import random
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spg_bandit.modules.dataset.base import TaskPool
from spg_bandit.modules.selector.spg_bandit import (
    SPGBanditSelector, online_profile_update,
)
from spg_bandit.utils.config_loader import load_config, resolve_config_path
from spg_bandit.utils.warmup import sample_type_balanced_task_ids


class _CapturingRidge:
    last_y = None

    def __init__(self, alpha):
        self.output_width = None

    def fit(self, X, y):
        type(self).last_y = y.copy()
        self.output_width = y.shape[1]
        return self

    def predict(self, X):
        return np.zeros((len(X), self.output_width))


class SPGBanditTests(unittest.TestCase):
    def setUp(self):
        self.evolve_pool = TaskPool(
            embeddings=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            metadata=[{"id": i, "dim": 0, "goal": str(i)} for i in range(3)],
        )

    def test_finalize_uses_warmup_task_parameters_not_task_type_indexes(self):
        selector = SPGBanditSelector(
            self.evolve_pool, n_warm=2, K=2, d_f=2, window_size=2,
        )
        selector._warmup_task_ids = [1, 2]
        selector._warmup_successes = [True, False]

        def fake_mirt(R, K, **_kwargs):
            n_rows, n_tasks = R.shape
            profiles = np.full((n_rows, K), 0.5)
            discrimination = np.array([[10 * i, 10 * i + 1] for i in range(n_tasks)], dtype=float)
            difficulty = np.arange(n_tasks, dtype=float)
            return profiles, discrimination, difficulty, 0.0, [0.0]

        with patch("spg_bandit.modules.selector.spg_bandit.fit_mirt_em", fake_mirt), \
             patch("spg_bandit.modules.selector.spg_bandit.Ridge", _CapturingRidge), \
             patch("spg_bandit.modules.selector.spg_bandit.MLPFeaturizer.train", return_value=[0.0]):
            selector._finalize_warmup()

        expected = np.array([[10.0, 11.0, 1.0], [20.0, 21.0, 2.0]])
        np.testing.assert_array_equal(_CapturingRidge.last_y, expected)

    def test_empty_warmup_has_clear_error(self):
        selector = SPGBanditSelector(self.evolve_pool, n_warm=0, K=2)
        with self.assertRaisesRegex(ValueError, "without observations"):
            selector.select(self.evolve_pool)

    def test_sliding_window_evicts_the_oldest_observation(self):
        selector = SPGBanditSelector(
            self.evolve_pool, n_warm=3, K=2, d_f=2, window_size=2,
        )
        selector._append_window_observation(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        selector._append_window_observation(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        selector._append_window_observation(np.array([1.0, 1.0]), np.array([2.0, 3.0]))

        self.assertEqual(len(selector._window), 2)
        np.testing.assert_array_equal(
            selector._A,
            np.eye(2) + np.array([[0.0, 0.0], [0.0, 1.0]]) + np.ones((2, 2)),
        )
        np.testing.assert_array_equal(
            selector._B,
            np.array([[2.0, 3.0], [2.0, 4.0]]),
        )

    def test_grouped_rollouts_are_one_warmup_round(self):
        selector = SPGBanditSelector(
            self.evolve_pool, n_warm=1, K=2, d_f=2, window_size=1,
        )
        selector.select(self.evolve_pool)
        selector.update(0, {"rollout_successes": [True, False, True, True]})

        self.assertEqual(selector._warmup_task_ids, [0])
        self.assertEqual(selector._warmup_successes, [3])
        self.assertEqual(selector._warmup_trials, [4])
        self.assertEqual(selector._warmup_outcomes, [[True, False, True, True]])

    def test_grouped_profile_update_uses_all_rollout_outcomes(self):
        initial = np.array([0.5])
        discrimination = np.array([1.0])
        failure = online_profile_update(initial, discrimination, 0.0, 0, trials=4)
        success = online_profile_update(initial, discrimination, 0.0, 4, trials=4)

        self.assertLess(failure[0], initial[0])
        self.assertGreater(success[0], initial[0])


class ConfigResolutionTests(unittest.TestCase):
    def test_explicit_config_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "custom.yaml"
            source.write_text("future_field:\n  arbitrary: [keep, this]\n")
            self.assertEqual(resolve_config_path(str(source)), source)

    def test_skillrl_configs_use_the_initial_skillbank(self):
        project_root = Path(__file__).resolve().parents[1]
        bank_path = project_root / "resource" / "skillrl" / "memory_data" / "alfworld" / "claude_style_skills.json"

        self.assertTrue(bank_path.is_file())
        bank = json.loads(bank_path.read_text())
        skill_ids = [skill.get("skill_id", "") for skill in bank["general_skills"]]
        skill_ids.extend(
            skill.get("skill_id", "")
            for skills in bank["task_specific_skills"].values()
            for skill in skills
        )
        self.assertFalse(any(skill_id.startswith("dyn_") for skill_id in skill_ids))

        for config_name, selector_name in (("skillrl", "spg_bandit"), ("skillrl_uniform", "uniform")):
            config = load_config(config_name)
            self.assertEqual(config["selector"], selector_name)
            configured_path = Path(config["skill_evolving"]["skill_bank_path"])
            self.assertEqual(configured_path.name, "claude_style_skills.json")
            self.assertEqual(configured_path.parts[:2], ("resource", "skillrl"))

    def test_skillrl_runtime_sources_are_vendored_under_resource(self):
        project_root = Path(__file__).resolve().parents[1]
        resource_root = project_root / "resource" / "skillrl"
        for relative_path in (
            "agent_system/memory/base.py",
            "agent_system/memory/skills_only_memory.py",
            "agent_system/memory/skill_updater.py",
        ):
            self.assertTrue((resource_root / relative_path).is_file())


class WarmupSamplingTests(unittest.TestCase):
    def test_sampling_balances_task_types_without_repeating_tasks(self):
        pool = TaskPool(
            embeddings=np.zeros((5, 2)),
            metadata=[
                {"id": 0, "dim": 0}, {"id": 1, "dim": 0},
                {"id": 2, "dim": 1}, {"id": 3, "dim": 1},
                {"id": 4, "dim": 2},
            ],
        )
        sampled = sample_type_balanced_task_ids(pool, 5, random.Random(7))
        counts = {task_type: 0 for task_type in range(3)}
        for task_id in sampled:
            counts[pool.metadata[task_id]["dim"]] += 1

        self.assertEqual(len(sampled), 5)
        self.assertEqual(len(sampled), len(set(sampled)))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
