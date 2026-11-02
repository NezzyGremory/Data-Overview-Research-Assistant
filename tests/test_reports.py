"""Unit tests for app.reports.generator.ReportGenerator."""

import unittest
from pathlib import Path
from app.reports.generator import ReportGenerator


def _make_analysis_data():
    """Minimal analysis_result dict simulating MainWindow.analysis_result."""
    return {
        "profile": {
            "total_rows": 1000,
            "total_columns": 12,
            "memory_usage_str": "45.2 KB",
            "dtypes": {
                "Age": "int64",
                "Fare": "float64",
                "Sex": "object",
                "Survived": "int64",
            },
        },
        "missing_values": [
            {"column": "Age", "missing_count": 50, "missing_percentage": 5.0},
            {"column": "Fare", "missing_count": 10, "missing_percentage": 1.0},
        ],
        "duplicates": {"duplicate_count": 15},
        "data_quality": [],
    }


def _make_ml_data():
    """Minimal ml_data dict simulating MLIntelligencePage.last_intelligence_result."""
    return {
        "status": "ML_INTELLIGENCE",
        "task_detection": {
            "primary_task": {"task": "binary_classification", "target": "Survived"},
        },
        "primary_task": {"task": "binary_classification", "target": "Survived"},
        "evaluation": {
            "status": "COMPLETED",
            "results": [
                {
                    "method_id": "random_forest",
                    "method": "Random Forest",
                    "metrics": {"accuracy": 0.8750, "f1_macro": 0.8600, "cv_std": 0.0210},
                    "folds": 5,
                    "validation": "stratified_kfold",
                },
                {
                    "method_id": "logistic_regression",
                    "method": "Logistic Regression",
                    "metrics": {"accuracy": 0.8200, "f1_macro": 0.8100, "cv_std": 0.0340},
                    "folds": 5,
                    "validation": "stratified_kfold",
                },
            ],
        },
        "best_method": {
            "method_id": "random_forest",
            "method": "Random Forest",
            "metrics": {"accuracy": 0.8750, "f1_macro": 0.8600},
            "folds": 5,
            "status": "EMPIRICAL_BEST",
        },
        "recommendation": {
            "status": "EMPIRICAL_RECOMMENDATION",
            "recommendations": [
                {
                    "method_id": "random_forest",
                    "strengths": ["Tahan terhadap overfitting", "Robust terhadap outlier"],
                    "limitations": ["Lambat pada data besar", "Sulit diinterpretasi"],
                },
            ],
        },
    }


def _make_research_data():
    """Minimal research_data dict simulating ResearchPage.research_result."""
    return {
        "domain": {"domain": "Healthcare / Medical AI"},
        "keywords": {"keywords": ["survival prediction", "titanic", "classification"]},
        "papers": [
            {
                "title": "Machine Learning for Survival Prediction",
                "authors": ["A. Smith", "B. Jones"],
                "year": 2023,
                "relevance_score": 0.92,
                "citation_count": 45,
                "abstract": "A study of ML methods for survival prediction.",
                "url": "https://example.com/paper1",
            },
            {
                "title": "Deep Learning Approaches in Healthcare",
                "authors": ["C. Wang"],
                "year": 2024,
                "relevance_score": 0.78,
                "citation_count": 12,
                "abstract": "Survey of deep learning in healthcare.",
            },
        ],
        "gaps": {
            "methodology": ["Kurangnya validasi external dataset"],
            "data": ["Dataset kecil dan tidak representatif"],
        },
    }


class TestReportGenerator(unittest.TestCase):
    """Tests for ReportGenerator."""

    def setUp(self):
        self.generator = ReportGenerator()

    def test_generate_empty_inputs(self):
        """Generator should produce valid HTML even with no data."""
        html = self.generator.generate(dataset_name="Empty Dataset")
        self.assertIn("<!DOCTYPE html>", html.lower().replace("<!doctype html>", "<!DOCTYPE html>"))
        self.assertIn("Empty Dataset", html)

    def test_generate_with_analysis_only(self):
        """Generator should work with only analysis_data."""
        html = self.generator.generate(
            dataset_name="Test Dataset",
            analysis_data=_make_analysis_data(),
        )
        self.assertIn("Test Dataset", html)
        self.assertIn("1,000", html)  # total_rows formatted
        self.assertIn("45.2 KB", html)  # memory_usage

    def test_generate_with_ml_data(self):
        """Generator should include ML benchmark sections."""
        html = self.generator.generate(
            dataset_name="ML Test",
            analysis_data=_make_analysis_data(),
            ml_data=_make_ml_data(),
        )
        self.assertIn("Random Forest", html)
        self.assertIn("Logistic Regression", html)
        self.assertIn("0.8750", html)  # accuracy metric
        self.assertIn("Best Empirical Performer", html)

    def test_report_explains_manual_task_selection(self):
        ml_data = _make_ml_data()
        ml_data["task_detection"]["selection_mode"] = "manual"
        ml_data["primary_task"]["selection_mode"] = "manual"
        ml_data["primary_task"]["reasons"] = ["Target dikonfirmasi pengguna."]

        html = self.generator.generate(
            dataset_name="Manual Classification",
            analysis_data=_make_analysis_data(),
            ml_data=ml_data,
        )

        self.assertIn("Dikonfirmasi pengguna", html)
        self.assertIn("Target dikonfirmasi pengguna", html)
        self.assertIn("Klasifikasi Biner", html)

    def test_generate_with_research_data(self):
        """Generator should include research and gap sections."""
        html = self.generator.generate(
            dataset_name="Research Test",
            research_data=_make_research_data(),
        )
        self.assertIn("Machine Learning for Survival Prediction", html)
        self.assertIn("Healthcare", html)
        self.assertIn("survival prediction", html)
        self.assertIn("Research Gap", html.replace("Research Gap", "Research Gap"))  # gaps section header

    def test_generate_full_report(self):
        """Generator should produce complete HTML with all sections."""
        html = self.generator.generate(
            dataset_name="Full Report",
            analysis_data=_make_analysis_data(),
            ml_data=_make_ml_data(),
            research_data=_make_research_data(),
            ai_synthesis="AI Summary: dataset ini cocok untuk klasifikasi biner.",
        )
        # All major sections present
        self.assertIn("Full Report", html)
        self.assertIn("Random Forest", html)
        self.assertIn("Machine Learning for Survival Prediction", html)
        self.assertIn("AI Summary", html)
        self.assertIn("Diagnostik Kualitas", html)

    def test_export_to_file(self, ):
        """export_to_file should write valid HTML to disk."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.html"
            result = self.generator.export_to_file(
                filepath,
                dataset_name="Export Test",
                analysis_data=_make_analysis_data(),
                ml_data=_make_ml_data(),
            )
            self.assertTrue(result.exists())
            content = result.read_text(encoding="utf-8")
            self.assertIn("Export Test", content)
            self.assertIn("Random Forest", content)

    def test_readiness_score_calculation(self):
        """Readiness score should decrease with quality issues."""
        ctx_clean = self.generator._build_context(
            dataset_name="Clean",
            analysis_data={"profile": {"total_rows": 100, "total_columns": 5, "dtypes": {}}, "data_quality": []},
            ml_data={},
            research_data={},
            ai_synthesis="",
            metadata={},
        )
        ctx_dirty = self.generator._build_context(
            dataset_name="Dirty",
            analysis_data={
                "profile": {"total_rows": 100, "total_columns": 5, "dtypes": {}},
                "missing_values": [{"column": "a", "missing_count": 80, "missing_percentage": 80.0}],
                "duplicates": {"duplicate_count": 30},
                "data_quality": [
                    type("Q", (), {"category": "missing", "title": "Missing", "severity": "critical",
                                   "problem": "p", "magnitude": "m", "impact": "i", "options": []})(),
                ],
            },
            ml_data={},
            research_data={},
            ai_synthesis="",
            metadata={},
        )
        self.assertGreater(ctx_clean["readiness_score"], ctx_dirty["readiness_score"])

    def test_context_paper_formatting(self):
        """Papers should be formatted with authors, year and relevance."""
        ctx = self.generator._build_context(
            dataset_name="Papers",
            analysis_data={},
            ml_data={},
            research_data=_make_research_data(),
            ai_synthesis="",
            metadata={},
        )
        papers = ctx["papers_list"]
        self.assertEqual(len(papers), 2)
        self.assertIn("A. Smith", papers[0]["authors_display"])
        self.assertEqual(papers[0]["year"], 2023)
        self.assertIn("92.0%", papers[0]["relevance_display"])

    def test_candidate_models_ordering(self):
        """Candidate models should preserve evaluation order."""
        ctx = self.generator._build_context(
            dataset_name="Order",
            analysis_data={},
            ml_data=_make_ml_data(),
            research_data={},
            ai_synthesis="",
            metadata={},
        )
        models = ctx["candidate_models"]
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["method"], "Random Forest")
        self.assertEqual(models[1]["method"], "Logistic Regression")

    def test_inferential_results_are_normalized_and_included_in_report(self):
        result = {
            "status": "COMPLETED",
            "test_id": "welch_t",
            "test_name": "Welch independent t-test",
            "variables": ["measurement", "group"],
            "sample_size": 24,
            "statistic": 2.5,
            "degrees_of_freedom": 20.4,
            "p_value": 0.02,
            "decision": "TOLAK_H0",
            "hypotheses": {"null": "Rata-rata sama.", "alternative": "Rata-rata berbeda."},
            "confidence_interval": [0.2, 1.8],
            "effect_size": {"name": "Hedges g", "value": 0.7},
            "interpretation": "P-value tidak dikoreksi untuk pengujian hipotesis lain.",
            "assumptions": [{"name": "Independensi", "status": "USER_ASSUMED", "detail": "Unit dianggap independen."}],
        }
        analysis = {"inferential_statistics": [result]}
        context = self.generator._build_context(
            dataset_name="Inference",
            analysis_data=analysis,
            ml_data={},
            research_data={},
            ai_synthesis="",
            metadata={},
        )

        self.assertEqual(len(context["inferential_results"]), 1)
        self.assertEqual(context["inferential_results"][0]["variables"], "measurement, group")
        self.assertEqual(context["inferential_results"][0]["confidence_interval"], "[0.2, 1.8]")
        report = self.generator.generate(dataset_name="Inference", analysis_data=analysis)
        self.assertIn("Statistik Inferensial", report)
        self.assertIn("Welch independent t-test", report)
        self.assertIn("Unit dianggap independen.", report)

    def test_fallback_report_includes_inferential_assumptions(self):
        context = self.generator._build_context(
            dataset_name="Inference fallback",
            analysis_data={
                "inferential_statistics": [{
                    "test_name": "Pearson correlation",
                    "variables": ["x", "y"],
                    "sample_size": 10,
                    "statistic": 0.9,
                    "p_value": 0.01,
                    "decision": "TOLAK_H0",
                    "assumptions": [{"name": "Independensi", "detail": "Periksa rancangan sampling."}],
                }],
            },
            ml_data={},
            research_data={},
            ai_synthesis="",
            metadata={},
        )
        report = self.generator._generate_fallback_html(context)
        self.assertIn("Statistik Inferensial", report)
        self.assertIn("Periksa rancangan sampling.", report)


if __name__ == "__main__":
    unittest.main()

