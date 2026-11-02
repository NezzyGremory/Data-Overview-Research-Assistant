import pandas as pd

from app.ml.intelligence import MLIntelligenceEngine


def create_classification_data():
    return pd.DataFrame(
        {
            "Age": [
                20, 25, 30, 35, 40,
                45, 50, 55, 28, 32,
                22, 38, 42, 48, 27,
                31, 36, 44, 52, 29,
            ],
            "Fare": [
                10, 15, 20, 25, 30,
                35, 40, 45, 18, 22,
                12, 28, 33, 38, 17,
                21, 26, 32, 42, 19,
            ],
            "Sex": [
                "male", "female", "male", "female", "male",
                "male", "female", "male", "female", "female",
                "male", "female", "male", "male", "female",
                "male", "female", "male", "female", "male",
            ],
            "Survived": [
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 1,
                0, 1, 0, 0, 1,
                0, 1, 0, 1, 0,
            ],
        }
    )


def create_fingerprint(dataframe):
    """
    Fingerprint minimal yang kompatibel dengan
    MLTaskDetector.
    """

    return {
        "representation": {
            "column_signature": [
                {
                    "name": "Age",
                    "normalized_name": "age",
                    "dtype": "int64",
                    "semantic_type": "numeric",
                    "potential_id": False,
                },
                {
                    "name": "Fare",
                    "normalized_name": "fare",
                    "dtype": "int64",
                    "semantic_type": "numeric",
                    "potential_id": False,
                },
                {
                    "name": "Sex",
                    "normalized_name": "sex",
                    "dtype": "object",
                    "semantic_type": "categorical",
                    "potential_id": False,
                },
                {
                    "name": "Survived",
                    "normalized_name": "survived",
                    "dtype": "int64",
                    "semantic_type": "numeric",
                    "potential_id": False,
                    "unique_count": 2,
                },
            ],
            "target_candidates": [
                {
                    "name": "Survived",
                    "score": 100,
                }
            ],
            "characteristics": {
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
            },
        }
    }


def test_full_ml_intelligence():

    dataframe = create_classification_data()
    fingerprint = create_fingerprint(dataframe)

    engine = MLIntelligenceEngine()

    result = engine.analyze(
        dataframe=dataframe,
        fingerprint=fingerprint,
        evaluate_models=True,
    )

    assert result["status"] == "ML_INTELLIGENCE"

    # Task Detection
    assert result["primary_task"]["task"] == (
        "binary_classification"
    )

    assert result["primary_task"]["target"] == "Survived"

    # Recommendation
    recommendation = result["recommendation"]

    assert recommendation["recommendation_count"] > 0

    first_recommendation = recommendation[
        "recommendations"
    ][0]

    assert "knowledge_base" in first_recommendation
    assert "description" in first_recommendation
    assert "strengths" in first_recommendation
    assert "limitations" in first_recommendation

    # Evaluation
    evaluation = result["evaluation"]

    assert evaluation["status"] == "EVALUATION"
    assert evaluation["result_count"] > 0

    # Best method
    assert result["best_method"] is not None
    assert result["best_method"]["method_id"]

    # Summary
    summary = result["summary"]

    assert summary["task"] == "Binary Classification"
    assert summary["recommended_method"] is not None
    assert summary["best_empirical_method"] is not None

    # Transparency
    assert result["transparency"]["task_detection"] == (
        "ESTIMATION"
    )

    assert result["transparency"]["method_recommendation"] == (
        "RECOMMENDATION"
    )

    assert result["transparency"]["model_evaluation"] == (
        "EMPIRICAL"
    )

    print("Task:", summary["task"])
    print(
        "Recommended:",
        summary["recommended_method"],
        summary["recommended_method_score"],
    )
    print(
        "Best empirical:",
        summary["best_empirical_method"],
    )

    print("\nEvaluation ranking:")

    for item in evaluation["results"]:
        print(
            item["method_id"],
            item["metrics"],
        )


def test_evaluation_can_be_disabled():

    dataframe = create_classification_data()
    fingerprint = create_fingerprint(dataframe)

    engine = MLIntelligenceEngine()

    result = engine.analyze(
        dataframe=dataframe,
        fingerprint=fingerprint,
        evaluate_models=False,
    )

    assert result["status"] == "ML_INTELLIGENCE"
    assert result["evaluation"]["status"] == "SKIPPED"

    print("Evaluation disabled: PASSED")


def test_dataset_info_counts_string_categorical_dtypes_explicitly():
    dataframe = pd.DataFrame({
        "object_text": pd.Series(["a", "b"], dtype=object),
        "string_text": pd.Series(["x", "y"], dtype="string"),
        "category": pd.Series(pd.Categorical(["low", "high"])),
        "flag": [True, False],
        "value": [1, 2],
    })

    info = MLIntelligenceEngine._extract_dataset_info(dataframe, {})

    assert info["categorical_features"] == 4


if __name__ == "__main__":

    print("=== ML INTELLIGENCE ENGINE ===")

    test_full_ml_intelligence()
    test_evaluation_can_be_disabled()

    print()
    print("================================")
    print("Semua ML Intelligence test berhasil.")
    print("================================")
