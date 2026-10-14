from __future__ import annotations

import pandas as pd
import pytest

from app.analyzer.ml_readiness import (
    ML_PIPELINE_STAGES,
    ML_READINESS_LABELS,
    analyze_ml_readiness,
    build_pipeline_checklist,
    class_balance_analysis,
    data_leakage_check,
    data_quality_summary,
    detect_column_types,
    distribution_analysis,
    feature_selection_hints,
    preprocessing_recommendation,
    train_test_split_recommendation,
)


@pytest.fixture
def mixed_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "numeric_text": ["1", "2", "3", "4", "5", "6"],
            "category": ["a", "a", "b", "c", "b", "a"],
            "binary": [0, 0, 0, 0, 1, 1],
            "record_id": [1001, 1002, 1003, 1004, 1005, 1006],
            "constant": [7, 7, 7, 7, 7, 7],
            "target": [0, 0, 0, 0, 1, 1],
        }
    )


def _existing_results(dataframe: pd.DataFrame) -> dict:
    return {
        "missing_values": [
            {"column": "numeric", "missing_count": 1, "missing_percentage": 16.7},
        ],
        "duplicates": {"duplicate_count": 1},
        "outliers": [{"column": "numeric", "outlier_count": 2}],
    }


def test_detect_column_types_handles_numeric_text_ids_constants_and_edges(mixed_dataset):
    original = mixed_dataset.copy(deep=True)

    result = detect_column_types(mixed_dataset)
    types = {item["column"]: item for item in result}

    assert types["numeric"]["detected_type"] == "Numeric"
    assert types["numeric_text"]["numeric_as_text"] is True
    assert types["record_id"]["detected_type"] == "ID-like"
    assert types["constant"]["detected_type"] == "Constant"
    assert types["category"]["detected_type"] == "Categorical"
    assert types["binary"]["detected_type"] == "Binary"
    assert {item["column"]: item["detected_type"] for item in detect_column_types(
        pd.DataFrame({"record_key": [11, 12, 13]})
    )}["record_key"] == "ID-like"
    pd.testing.assert_frame_equal(mixed_dataset, original)
    assert detect_column_types(pd.DataFrame()) == []


def test_distribution_analysis_covers_numeric_categorical_and_empty(mixed_dataset):
    types = detect_column_types(mixed_dataset)
    result = distribution_analysis(mixed_dataset, types)

    assert result["numeric"]
    assert any(item["column"] == "numeric_text" for item in result["numeric"])
    assert result["categorical"]
    assert result["sampled"] is False
    assert distribution_analysis(pd.DataFrame())["numeric"] == []


def test_class_balance_detects_imbalance_and_handles_regression_and_missing_target(mixed_dataset):
    balanced = class_balance_analysis(mixed_dataset, "target", "binary_classification")
    imbalanced_df = pd.DataFrame({"label": ["a"] * 8 + ["b"] * 2})
    imbalanced = class_balance_analysis(imbalanced_df, "label", "binary_classification")
    regression = class_balance_analysis(mixed_dataset, "numeric", "regression")
    missing = class_balance_analysis(mixed_dataset, None, None)

    assert balanced["imbalance_ratio"] == 2.0
    assert balanced["status"] == "Warning"
    assert imbalanced["status"] == "Issue"
    assert regression["summary"] == "Not applicable for this dataset"
    assert missing["summary"] == "Target not detected"


def test_data_quality_summary_uses_existing_analyzer_counts(mixed_dataset):
    existing = _existing_results(mixed_dataset)
    result = data_quality_summary(
        mixed_dataset,
        existing["missing_values"],
        existing["duplicates"],
        existing["outliers"],
        column_types=detect_column_types(mixed_dataset),
    )

    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 1
    assert result["outlier_values"] == 2
    assert "constant" in result["constant_columns"]
    assert "record_id" in result["id_like_columns"]


def test_data_leakage_finds_copied_target_and_id_and_handles_no_target(mixed_dataset):
    dataframe = mixed_dataset.assign(target_copy=mixed_dataset["target"])
    result = data_leakage_check(dataframe, "target")
    no_target = data_leakage_check(dataframe, None)

    findings = {item["column"]: item["reasons"] for item in result["findings"]}
    assert "Identical to target" in findings["target_copy"]
    assert "ID-like column" in findings["record_id"]
    assert result["status"] == "Warning"
    assert no_target["summary"].startswith("Target not detected")


def test_train_test_split_recommendation_handles_small_and_empty_data(mixed_dataset):
    result = train_test_split_recommendation(mixed_dataset, "target", "binary_classification")
    tiny = train_test_split_recommendation(mixed_dataset.iloc[:2], "target", "regression")
    empty = train_test_split_recommendation(pd.DataFrame(), None, None)

    assert result["stratify"] is True
    assert result["train_rows"] + result["test_rows"] == len(mixed_dataset)
    assert tiny["cross_validation"] is True
    assert tiny["test_size"] == 0.3
    assert empty["status"] == "Issue"


def test_preprocessing_recommendation_returns_encoding_scaling_and_missing_guidance(mixed_dataset):
    types = detect_column_types(mixed_dataset)
    distributions = distribution_analysis(mixed_dataset, types)
    balance = class_balance_analysis(mixed_dataset, "target", "binary_classification")
    result = preprocessing_recommendation(
        mixed_dataset,
        types,
        distributions,
        balance,
        [{"column": "numeric", "missing_percentage": 8.0}],
        [{"column": "numeric", "outlier_count": 1}],
        [{"method": "SVM", "scaling_required": True}],
    )

    assert result["encoding"][0]["method"] == "One-Hot"
    assert result["scaling"]
    assert result["scaling_models"] == ["SVM"]
    assert result["missing_strategy"][0]["method"] == "Mean"
    assert result["fit_note"] == "Fit preprocessing on the training set only to avoid data leakage."


def test_feature_selection_hints_and_targetless_case(mixed_dataset):
    types = detect_column_types(mixed_dataset)
    result = feature_selection_hints(mixed_dataset, "target", "binary_classification", types)
    targetless = feature_selection_hints(mixed_dataset, None, None, types)

    assert "constant" in result["constant_columns"]
    assert result["importance"]
    assert len(result["importance"]) <= 10
    assert targetless["importance"] == []
    assert targetless["importance_note"] == "Target not detected"


def test_pipeline_checklist_uses_evidence_and_has_15_stages(mixed_dataset):
    results = _existing_results(mixed_dataset)
    original = mixed_dataset.copy(deep=True)
    readiness = analyze_ml_readiness(
        mixed_dataset,
        missing_values=results["missing_values"],
        duplicates=results["duplicates"],
        outliers=results["outliers"],
        target="target",
        task="binary_classification",
        recommended_models=[{"method": "Random Forest"}],
        ml_result={"evaluation": {"results": [{"method": "Random Forest"}]}},
    )
    statuses = {item["stage"]: item["status"] for item in readiness["pipeline_checklist"]}

    assert len(readiness["pipeline_checklist"]) == 15
    assert [item["stage"] for item in readiness["pipeline_checklist"]] == list(ML_PIPELINE_STAGES)
    assert statuses["Problem Definition"] == "Done"
    assert statuses["Data Loading"] == "Done"
    assert statuses["Model Selection"] == "Done"
    assert statuses["Model Training"] == "Done"
    assert statuses["Evaluation and Cross-Validation"] == "Done"
    assert statuses["Hyperparameter Tuning"] == "Not Available Yet"
    assert statuses["Model Saving and Deployment"] == "Not Available Yet"
    assert readiness["target"] == "target"
    pd.testing.assert_frame_equal(mixed_dataset, original)


def test_large_dataframe_reports_sampling_note(mixed_dataset):
    large = pd.concat([mixed_dataset] * 17_000, ignore_index=True)

    result = analyze_ml_readiness(large)

    assert result["distributions"]["sampled"] is True
    assert result["distributions"]["sample_count"] == 10_000
    assert "Computed on a sample of 10,000 rows" in result["summary_note"]


def test_empty_single_column_constant_and_numeric_regression_edges():
    empty = analyze_ml_readiness(pd.DataFrame())
    single = analyze_ml_readiness(pd.DataFrame({"only": [1]}), target="missing")
    constant = analyze_ml_readiness(pd.DataFrame({"x": [3, 3, 3], "y": [1.0, 2.0, 3.0]}))
    regression = analyze_ml_readiness(
        pd.DataFrame({"feature": [1, 2, 3, 4, 5], "outcome": [2., 4., 6., 8., 10.]}),
        target="outcome",
        task="regression",
    )

    assert empty["data_types"]["status"] == "Issue"
    assert single["split"]["test_rows"] == 1
    assert constant["data_types"]["columns"][0]["detected_type"] == "Constant"
    assert regression["metric_recommendation"] == ["MAE", "RMSE", "R2"]
    assert regression["class_balance"]["applicable"] is False


def test_new_label_constants_are_english_and_pipeline_titles_are_stable():
    assert ML_PIPELINE_STAGES[0] == "Problem Definition"
    assert ML_PIPELINE_STAGES[-1] == "Monitoring"
    all_labels = list(ML_READINESS_LABELS["sections"]) + list(ML_READINESS_LABELS["statuses"])
    all_labels += list(ML_READINESS_LABELS["checklist_statuses"]) + list(ML_PIPELINE_STAGES)
    assert all(label.isascii() for label in all_labels)
    lower_labels = " ".join(all_labels).casefold()
    assert not any(
        marker in lower_labels
        for marker in ("kolom", "baris", "nilai", "saran", "peringatan", "pemeriksaan", "tahap")
    )