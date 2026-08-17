from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "network_analysis", ROOT / "scripts" / "run_network_analysis.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NetworkAnalysisTests(unittest.TestCase):
    def test_task_registry_is_canonical(self):
        tasks = MODULE.load_tasks(ROOT / "config" / "tasks.csv")
        self.assertEqual(len(tasks), 11)
        kp = tasks.loc[tasks["species"].eq("kpneumoniae")].set_index("charge")["pool"].to_dict()
        self.assertEqual(kp, {4: 10, 5: 60, 6: 60})

    def test_threshold_sweep_counts_edges(self):
        i = np.array([0, 0, 1])
        j = np.array([1, 2, 2])
        sim = np.array([0.9, 0.6, 0.4])
        out = MODULE.threshold_sweep_table(3, i, j, sim, np.array([0.5, 0.8]))
        self.assertEqual(out.loc[0, "n_edges"], 2)
        self.assertEqual(out.loc[1, "n_edges"], 1)
        self.assertEqual(out.loc[0, "n_nodes_nonisolated"], 3)
        self.assertEqual(out.loc[1, "n_nodes_nonisolated"], 2)

    def test_candidate_rule(self):
        nodes = pd.DataFrame(
            {
                "ID": ["a", "b", "c", "d", "e", "f"],
                "modularity_class": [0, 0, 1, 1, 2, 2],
                "pagerank": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
                "predicted": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0],
            }
        )
        selected = MODULE.select_candidates(nodes)
        # Best two of three modules are retained; best half of each module by PageRank.
        self.assertEqual(set(selected["ID"]), {"a", "c"})
        self.assertEqual(selected["ID"].tolist(), ["a", "c"])


if __name__ == "__main__":
    unittest.main()
