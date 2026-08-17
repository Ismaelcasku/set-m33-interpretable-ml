from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import load_config  # noqa: E402
from leakage_common import (  # noqa: E402
    audit_task_from_id,
    features_for_pool,
    load_audit_config,
    parse_cv_attribute_frequency,
    prepare_positive_dataset,
    total_audit_tasks,
)


class LeakageAuditTests(unittest.TestCase):
    def test_audit_task_mapping(self) -> None:
        root = Path(__file__).resolve().parents[1]
        project = load_config(root)
        audit = load_audit_config(root)
        self.assertEqual(total_audit_tasks(project, audit), 1100)
        expected = {
            0: ("legacy_global", "ecoli_4plus", 1, 1),
            549: ("legacy_global", "kpneumoniae_6plus", 10, 5),
            550: ("nested_matched", "ecoli_4plus", 1, 1),
            1099: ("nested_matched", "kpneumoniae_6plus", 10, 5),
        }
        for task_id, values in expected.items():
            spec = audit_task_from_id(project, audit, task_id)
            self.assertEqual(
                (spec.branch, spec.base.dataset_name, spec.base.seed, spec.base.fold),
                values,
            )

    def test_weka_cv_frequency_parser_and_strict_pool_rule(self) -> None:
        output = """
=== Attribute Selection on all input data ===
Selected attributes: 1,3 : 2

=== Attribute selection 5 fold cross-validation (stratified), seed: 1 ===

 number of folds (%)  attribute
           5(100 %)     1 descriptor_a
           1( 20 %)     2 descriptor_b
           2( 40 %)     3 descriptor_c
           0(  0 %)     4 __target__
"""
        names = ["descriptor_a", "descriptor_b", "descriptor_c"]
        frequency = parse_cv_attribute_frequency(output, names, expected_folds=5)
        self.assertEqual(
            frequency.set_index("feature")["selected_in_fits"].to_dict(),
            {"descriptor_a": 5, "descriptor_c": 2, "descriptor_b": 1},
        )
        self.assertEqual(features_for_pool(frequency, 0, names), names)
        self.assertEqual(features_for_pool(frequency, 20, names), ["descriptor_a", "descriptor_c"])
        self.assertEqual(features_for_pool(frequency, 90, names), ["descriptor_a"])

    def test_positive_filter_precedes_log_transform(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frame = pd.DataFrame(
                {
                    "ID": ["a", "b", "c", "d", "e", "f"],
                    "MIC": [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0],
                    "x1": [10, 20, 30, 40, 50, 60],
                    "x2": [1, 2, 3, 4, 5, 6],
                }
            )
            path = tmp_path / "data.csv"
            frame.to_csv(path, index=False)
            config = {
                "target_transform": "none",
                "missing_target_policy": "exclude",
                "target_column": "MIC",
                "id_column": "ID",
                "csv_separator": ",",
                "n_splits": 2,
            }
            data = prepare_positive_dataset("synthetic", path, config)
            self.assertEqual(data["n_rows_excluded_nonpositive"], 2)
            self.assertEqual(data["ids"].tolist(), ["c", "d", "e", "f"])
            self.assertEqual(data["source_pandas_rows"].tolist(), [2, 3, 4, 5])
            self.assertTrue(np.allclose(data["y_raw"], [0.5, 1.0, 2.0, 4.0]))
            self.assertTrue(np.allclose(data["y_log10"], np.log10([0.5, 1.0, 2.0, 4.0])))


if __name__ == "__main__":
    unittest.main()
