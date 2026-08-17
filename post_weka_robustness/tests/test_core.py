from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from setm33robust.config import load_config, load_tasks, umap_jobs
from setm33robust.io import (
    begin_output, completed_output, quantile_subset_indices,
    write_completed_output,
)
from setm33robust.metrics import (
    jaccard, knn_indices, mean_knn_overlap, regression_metrics,
)


class CoreTests(unittest.TestCase):
    def test_canonical_workload(self):
        cfg = load_config()
        tasks = load_tasks()
        self.assertEqual(len(tasks), 11)
        self.assertEqual(tasks.loc[tasks["key"].eq("kpneumoniae_5plus"), "pool"].item(), 60)
        self.assertEqual(tasks.loc[tasks["key"].eq("kpneumoniae_6plus"), "pool"].item(), 60)
        jobs = umap_jobs(cfg)
        self.assertEqual(len(jobs), 27)
        self.assertEqual(jobs[-1]["species"], "kpneumoniae")
        self.assertEqual(jobs[-1]["n_neighbors"], 50)
        self.assertEqual(jobs[-1]["min_dist"], 0.5)
        self.assertEqual(cfg["umap"]["metric"], "euclidean")
        self.assertEqual(cfg["umap"]["descriptor_group_k"], {
            "ecoli": 6, "paeruginosa": 6, "kpneumoniae": 3,
        })
        self.assertEqual(cfg["shap"]["bootstrap_replicates"], 200)

    def test_quantile_subset_is_deterministic(self):
        y = np.linspace(0, 1, 100)
        first = quantile_subset_indices(y, 30, 42)
        second = quantile_subset_indices(y, 30, 42)
        self.assertEqual(len(first), 30)
        np.testing.assert_array_equal(first, second)

    def test_metrics(self):
        values = np.array([1.0, 2.0, 3.0])
        metrics = regression_metrics(values, values)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(jaccard({1, 2}, {2, 3}), 1 / 3)

    def test_knn_indices_returns_first_nonself_neighbors(self):
        values = np.array([[0.0], [1.0], [10.0], [20.0]])
        observed = knn_indices(values, k=1)
        expected = np.array([[1], [0], [1], [2]])
        np.testing.assert_array_equal(observed, expected)

    def test_knn_overlap_is_intersection_over_k_not_jaccard(self):
        first = np.array([[1, 2, 3], [0, 2, 3]])
        second = np.array([[1, 2, 4], [0, 4, 5]])
        self.assertAlmostEqual(mean_knn_overlap(first, second), 0.5)

    def test_completed_output_requires_matching_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "table.csv"
            begin_output(directory, "fingerprint")
            self.assertFalse(completed_output(directory, "fingerprint", ["table.csv"]))
            output.write_text("x\n1\n", encoding="utf-8")
            write_completed_output(directory, "fingerprint", ["table.csv"])
            self.assertTrue(completed_output(directory, "fingerprint", ["table.csv"]))
            output.write_text("x\n2\n", encoding="utf-8")
            self.assertFalse(completed_output(directory, "fingerprint", ["table.csv"]))


if __name__ == "__main__":
    unittest.main()
