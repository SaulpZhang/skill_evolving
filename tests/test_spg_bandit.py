import random
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spg_bandit.modules.dataset import create_dataset, register_dataset
from spg_bandit.modules.dataset.base import BaseDataset, TaskPool
from spg_bandit.modules.dataset.embedding_cache import EmbeddingCache
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


class EmbeddingCacheTests(unittest.TestCase):
    def test_embedding_cache_persists_and_isolated_by_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            first = EmbeddingCache(
                directory, namespace="alfworld",
                config={"embedding_model": "model-a", "embedding_type": "local"},
            )
            first.put("task A", [1.0, 2.0])
            self.assertTrue(first.save())

            second = EmbeddingCache(
                directory, namespace="alfworld",
                config={"embedding_model": "model-a", "embedding_type": "local"},
            )
            np.testing.assert_array_equal(second.get("task A"), [1.0, 2.0])
            self.assertIsNone(second.get("task B"))

            different_model = EmbeddingCache(
                directory, namespace="alfworld",
                config={"embedding_model": "model-b", "embedding_type": "local"},
            )
            self.assertIsNone(different_model.get("task A"))

    def test_disabled_embedding_cache_never_reads_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(
                directory, namespace="alfworld", enabled=False,
            )
            cache.put("task A", [1.0, 2.0])
            self.assertFalse(cache.save())
            self.assertIsNone(cache.get("task A"))

    def test_skillrl_runtime_sources_are_vendored_under_resource(self):
        project_root = Path(__file__).resolve().parents[1]
        resource_root = project_root / "resource" / "skillrl"
        for relative_path in (
            "agent_system/memory/base.py",
            "agent_system/memory/skills_only_memory.py",
            "agent_system/memory/skill_updater.py",
        ):
            self.assertTrue((resource_root / relative_path).is_file())

    def test_skillopt_configs_use_vendored_runtime_and_gate_split(self):
        project_root = Path(__file__).resolve().parents[1]
        skillopt_root = project_root / "resource" / "skillopt"
        self.assertTrue((skillopt_root / "gradient" / "reflect.py").is_file())
        self.assertTrue((skillopt_root / "envs" / "alfworld" / "skills" / "initial.md").is_file())

        for config_name, selector_name in (("skillopt", "spg_bandit"), ("skillopt_uniform", "uniform")):
            config = load_config(config_name)
            self.assertEqual(config["selector"], selector_name)
            self.assertEqual(config["skill_evolving"]["name"], "skillopt")
            self.assertEqual(config["skill_selection"]["split"], "valid_seen")
            self.assertEqual(config["evaluate"]["split"], "valid_unseen")

        # The execution package must not contain a hard-coded checkout path.
        for source in skillopt_root.rglob("*.py"):
            self.assertNotIn("docs/SkillOpt", source.read_text(encoding="utf-8"))


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

    def test_sampling_supports_string_types_without_ids_in_metadata(self):
        pool = TaskPool(
            embeddings=np.zeros((4, 2)),
            metadata=[
                {"task_type": "web", "goal": "a"},
                {"task_type": "web", "goal": "b"},
                {"task_type": "search", "goal": "c"},
                {"task_type": "tool", "goal": "d"},
            ],
        )
        sampled = sample_type_balanced_task_ids(pool, 4, random.Random(3))
        self.assertEqual(sorted(sampled), [0, 1, 2, 3])
        self.assertEqual([item["id"] for item in pool.metadata], [0, 1, 2, 3])


class _FakeEnv:
    def __init__(self):
        self.closed = False

    def reset(self):
        return "start", {"actions": ["finish"]}

    def step(self, action):
        self.action = action
        return "done", 1.0, True, {"success": True, "actions": []}

    def close(self):
        self.closed = True


class _FakeDataset(BaseDataset):
    name = "fake"

    def __init__(self, config):
        self._pool = TaskPool(
            embeddings=np.zeros((1, 2)),
            metadata=[{"goal": "finish the task", "task_type": "toy"}],
        )

    @property
    def task_pool(self):
        return self._pool

    def get_task_goal(self, task_id):
        return self._pool.get_goal(task_id)

    def load(self):
        return None

    def create_env(self, task_id):
        return _FakeEnv()


class DatasetInterfaceTests(unittest.TestCase):
    def test_batch_boolean_normalization_does_not_treat_false_container_as_true(self):
        self.assertFalse(BaseDataset._as_bool([False]))
        self.assertFalse(BaseDataset._as_bool(np.array([False])))
        self.assertTrue(BaseDataset._as_bool([True]))

    def test_registry_and_generic_gym_protocol(self):
        register_dataset("fake", _FakeDataset)
        dataset = create_dataset("fake", {})
        handle = dataset.create_env(0)
        state = dataset.reset_env(handle)
        self.assertEqual(state.observation, "start")
        self.assertEqual(state.admissible_actions, ["finish"])
        transition = dataset.step_env(handle, "finish")
        self.assertTrue(transition.success)
        self.assertTrue(transition.done)
        dataset.close_env(handle)
        self.assertTrue(handle.closed)


class SkillRLAdapterTests(unittest.TestCase):
    def test_grouped_rollouts_do_not_call_zero_argument_super_in_comprehension(self):
        from spg_bandit.modules.skill_evolving.skillrl.agent import SkillRLAgent

        fake_rollout = {
            "success": True,
            "trajectory": "ok",
            "actions": [],
            "api_calls": 1,
        }
        with patch("spg_bandit.modules.skill_evolving.simple_agent.agent.OpenAI"):
            agent = SkillRLAgent(
                _FakeDataset({}),
                config={"enable_dynamic_update": False},
            )
        with patch(
            "spg_bandit.modules.skill_evolving.skillrl.agent.SimpleAgent.execute",
            return_value=fake_rollout,
        ) as execute:
            result = agent.execute(0, num_rollouts=2)

        self.assertEqual(execute.call_count, 2)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["num_rollouts"], 2)


if __name__ == "__main__":
    unittest.main()
