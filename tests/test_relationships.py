import numpy as np
import pandas as pd
import pytest

from app.analyzer.relationships import RelationshipAnalyzer


def test_covariance_uses_finite_numeric_columns_and_excludes_boolean_columns():
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.inf, 4.0],
            "y": [3.0, 5.0, 7.0, 9.0],
            "flag": [True, False, True, False],
            "label": ["a", "b", "a", "b"],
        }
    )

    covariance = RelationshipAnalyzer().covariance(frame)

    expected = frame.loc[[0, 1, 3], ["x", "y"]].cov().loc["x", "y"]
    assert list(covariance.columns) == ["x", "y"]
    assert covariance.loc["x", "y"] == pytest.approx(expected)
    assert np.isfinite(covariance.to_numpy()).all()


def test_covariance_returns_empty_matrix_when_fewer_than_two_numeric_features():
    frame = pd.DataFrame({"value": [1.0, 2.0], "flag": [True, False]})

    assert RelationshipAnalyzer().covariance(frame).empty


def test_covariance_bounds_wide_data_to_first_thirty_numeric_columns():
    frame = pd.DataFrame({f"feature_{index}": [1.0, 2.0, 3.0] for index in range(35)})

    covariance = RelationshipAnalyzer().covariance(frame)

    assert covariance.shape == (30, 30)
    assert "feature_29" in covariance.columns
    assert "feature_30" not in covariance.columns


def test_covariance_handles_nullable_numeric_dtypes():
    frame = pd.DataFrame(
        {
            "left": pd.Series([1, pd.NA, 3, 4], dtype="Int64"),
            "right": pd.Series([2.0, 3.0, 6.0, 8.0], dtype="Float64"),
        }
    )

    covariance = RelationshipAnalyzer().covariance(frame)

    assert np.isfinite(covariance.to_numpy(dtype=float, na_value=np.nan)).any()


def test_categorical_associations_identifies_perfect_and_independent_pairs():
    labels = np.repeat(["a", "b", "c"], 20)
    independent = np.tile(np.repeat(["left", "right"], 10), 3)
    frame = pd.DataFrame(
        {
            "label": labels,
            "copy": labels,
            "independent": independent,
            "numeric_code": np.arange(len(labels)) % 3,
            "id": [f"id-{index}" for index in range(len(labels))],
        }
    )

    matrix = RelationshipAnalyzer().categorical_associations(frame)

    assert matrix.loc["label", "copy"] == pytest.approx(1.0)
    assert matrix.loc["label", "independent"] == pytest.approx(0.0, abs=1e-12)
    assert "numeric_code" not in matrix.index
    assert "id" not in matrix.index
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T, equal_nan=True)

