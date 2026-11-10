from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.ai.local_explainer import LocalAcademicExplainer
from app.ai.gemini_explainer import (
    GeminiDatasetExplainer,
    get_saved_api_key,
    save_api_key,
    _EXPLANATION_CACHE,
)


class TestExplainerSystem(unittest.TestCase):
    def setUp(self):
        self.sample_analysis = {
            "profile": {
                "rows": 891,
                "columns": 12,
                "numeric_columns": 7,
                "categorical_columns": 5,
                "memory_usage": 121800,
            },
            "statistics": [
                {
                    "column": "Age",
                    "dtype": "float64",
                    "missing_count": 177,
                    "missing_percentage": 19.86,
                    "unique_count": 88,
                    "mean": 29.69,
                    "min": 0.42,
                    "max": 80.0,
                },
                {
                    "column": "Survived",
                    "dtype": "int64",
                    "missing_count": 0,
                    "missing_percentage": 0.0,
                    "unique_count": 2,
                    "mean": 0.38,
                }
            ],
            "missing_values": [
                {"column": "Cabin", "missing_count": 687, "missing_percentage": 77.1},
                {"column": "Age", "missing_count": 177, "missing_percentage": 19.86},
            ],
            "duplicates": {
                "duplicate_count": 5,
                "duplicate_percentage": 0.56,
            },
            "outliers": [
                {
                    "column": "Fare",
                    "outlier_count": 116,
                    "outlier_percentage": 13.0,
                    "lower_bound": -26.7,
                    "upper_bound": 65.6,
                }
            ],
            "correlations": None,
            "fingerprint": {
                "fingerprint": "test_fp_hash_1234567890abcdef",
                "representation": {
                    "target_candidates": ["Survived"],
                    "column_signature": [
                        {
                            "name": "Survived",
                            "unique_count": 2,
                            "target_score": 85,
                        }
                    ],
                },
            },
        }

    def test_local_explainer_generates_comprehensive_narrative(self):
        explainer = LocalAcademicExplainer()
        text = explainer.explain(self.sample_analysis)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 200)

        # Check key scientific aspects are covered
        self.assertIn("891", text)
        self.assertIn("12", text)
        self.assertIn("Survived", text)
        self.assertIn("Klasifikasi Biner", text)
        self.assertIn("Cabin", text)
        self.assertIn("Fare", text)

    def test_hybrid_explainer_falls_back_to_local_when_no_key(self):
        # Clear cache for this test fingerprint
        _EXPLANATION_CACHE.pop("test_fp_hash_1234567890abcdef", None)

        explainer = GeminiDatasetExplainer(api_key=None)
        # Force no API key
        explainer.api_keys = []
        text, source = explainer.explain_with_source(self.sample_analysis)

        self.assertEqual(source, "local")
        self.assertIn("891", text)
        self.assertIn("Survived", text)

    def test_hybrid_explainer_caching_by_fingerprint(self):
        fp = "test_caching_fp_999"
        analysis_with_fp = dict(self.sample_analysis)
        analysis_with_fp["fingerprint"] = {"fingerprint": fp}

        _EXPLANATION_CACHE.pop(fp, None)

        explainer = GeminiDatasetExplainer(api_key=None)
        explainer.api_keys = []

        # First call generates and caches
        text1, src1 = explainer.explain_with_source(analysis_with_fp)
        self.assertIn(fp, _EXPLANATION_CACHE)

        # Second call returns from cache
        text2, src2 = explainer.explain_with_source(analysis_with_fp)
        self.assertEqual(text1, text2)
        self.assertEqual(src1, src2)

    def test_rejected_gemini_key_gives_actionable_auth_key_guidance(self):
        fp = "test_rejected_key_fp"
        analysis = dict(self.sample_analysis)
        analysis["fingerprint"] = {"fingerprint": fp}
        _EXPLANATION_CACHE.pop(fp, None)
        explainer = GeminiDatasetExplainer(api_key="secret-test-key")
        response = Mock(
            status_code=403,
            text="PERMISSION_DENIED",
            json=Mock(return_value={
                "error": {
                    "message": "Requests from referer blocked for secret-test-key"
                }
            }),
        )

        with patch("app.ai.gemini_explainer.httpx.post", return_value=response):
            _text, source = explainer.explain_with_source(analysis)

        self.assertEqual(source, "local")
        self.assertIn("Auth key baru", explainer.last_error)
        self.assertIn("batasi untuk Gemini API", explainer.last_error)
        self.assertIn("Detail Google:", explainer.last_error)
        self.assertIn("[key disamarkan]", explainer.last_error)
        self.assertNotIn("secret-test-key", explainer.last_error)


if __name__ == "__main__":
    unittest.main()
