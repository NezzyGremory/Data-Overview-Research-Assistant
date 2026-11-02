import unittest
import numpy as np
import pandas as pd
from app.analyzer.data_quality import DataQualityDiagnoser, QualityDiagnosisItem
from app.reports.generator import ReportGenerator


class TestDataQualityDiagnoser(unittest.TestCase):

    def setUp(self):
        self.diagnoser = DataQualityDiagnoser()

    def test_clean_dataset_no_issues(self):
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": ["A", "B", "C", "D", "E"],
        })
        issues = self.diagnoser.diagnose(df)
        # No duplicates, no missing, no constant, etc.
        missing_or_dup = [i for i in issues if i.category in ("missing_values", "duplicates")]
        self.assertEqual(len(missing_or_dup), 0)

    def test_duplicate_diagnosis(self):
        df = pd.DataFrame({
            "a": [1, 2, 2, 3],
            "b": ["x", "y", "y", "z"],
        })
        issues = self.diagnoser.diagnose(df)
        dup_issue = next((i for i in issues if i.category == "duplicates"), None)
        self.assertIsNotNone(dup_issue)
        self.assertIn("1 baris", dup_issue.magnitude)
        self.assertTrue(len(dup_issue.options) > 0)
        self.assertTrue("Apa Dampaknya" not in dup_issue.impact)  # Impact text is meaningful
        self.assertIn("data leakage", dup_issue.impact.lower())

    def test_missing_values_diagnosis(self):
        df = pd.DataFrame({
            "col1": [1, None, None, 4],
            "col2": ["a", "b", "c", "d"],
        })
        issues = self.diagnoser.diagnose(df)
        miss_issue = next((i for i in issues if i.category == "missing_values"), None)
        self.assertIsNotNone(miss_issue)
        self.assertIn("col1", miss_issue.problem)
        self.assertIn("Imputasi", miss_issue.options[0])

    def test_outlier_diagnosis(self):
        df = pd.DataFrame({
            "val": [10, 11, 12, 11, 10, 12, 11, 1000],  # 1000 is extreme outlier
        })
        issues = self.diagnoser.diagnose(df)
        outlier_issue = next((i for i in issues if i.category == "outliers"), None)
        self.assertIsNotNone(outlier_issue)
        self.assertIn("val", outlier_issue.affected_columns)
        self.assertIn("RobustScaler", outlier_issue.options[0])

    def test_constant_column(self):
        df = pd.DataFrame({
            "constant_col": [99, 99, 99, 99],
            "valid_col": [1, 2, 3, 4],
        })
        issues = self.diagnoser.diagnose(df)
        const_issue = next((i for i in issues if i.category == "constant_columns"), None)
        self.assertIsNotNone(const_issue)
        self.assertIn("constant_col", const_issue.affected_columns)

    def test_non_finite_values_are_distinct_from_missing(self):
        df = pd.DataFrame({
            "measurement": [1.0, np.inf, -np.inf, np.nan, 5.0],
            "group": ["a", "b", "c", "d", "e"],
        })

        issues = self.diagnoser.diagnose(df)
        invalid = next(item for item in issues if item.category == "invalid_values")
        missing = next(item for item in issues if item.category == "missing_values")

        self.assertEqual(invalid.metrics["non_finite_count"], 2)
        self.assertEqual(invalid.severity, "critical")
        self.assertEqual(invalid.affected_columns, ["measurement"])
        self.assertEqual(missing.metrics["total_missing"], 1)

    def test_non_finite_diagnosis_appears_in_report(self):
        df = pd.DataFrame({"measurement": [1.0, np.inf, 2.0]})
        diagnoses = self.diagnoser.diagnose(df)

        html = ReportGenerator().generate(
            dataset_name="Non-finite data",
            analysis_data={"data_quality": diagnoses},
        )

        self.assertIn("Nilai Numerik Tidak Terhingga", html)
        self.assertIn("nilai +Inf/-Inf", html)


if __name__ == "__main__":
    unittest.main()
