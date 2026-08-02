import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spg_bandit.modules.dataset.base import TaskPool
from spg_bandit.modules.selector.spg_bandit import SPGBanditSelector
from spg_bandit.utils.config_loader import resolve_config_path


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
        self.warmup_pool = TaskPool(
            embeddings=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            metadata=[{"id": i, "dim": 0, "goal": str(i)} for i in range(3)],
        )
        self.evolve_pool = TaskPool(
            embeddings=np.array([[3.0, 0.0], [4.0, 0.0]]),
            metadata=[{"id": i, "dim": 0, "goal": str(i)} for i in range(2)],
        )

    def test_finalize_uses_warmup_task_parameters_not_task_type_indexes(self):
        selector = SPGBanditSelector(
            self.evolve_pool, n_warm=2, K=2, d_f=2,
            warmup_pool=self.warmup_pool,
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


class ConfigResolutionTests(unittest.TestCase):
    def test_explicit_config_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "custom.yaml"
            source.write_text("future_field:\n  arbitrary: [keep, this]\n")
            self.assertEqual(resolve_config_path(str(source)), source)


if __name__ == "__main__":
    unittest.main()
