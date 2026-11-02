import numpy as np
import pandas as pd

from app.analyzer.statistics import DatasetStatistics


def test_descriptive_statistics_use_only_finite_numeric_values():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, np.inf, -np.inf, np.nan]})

    result = DatasetStatistics().analyze(frame)[0]

    assert result["missing_count"] == 1
    assert result["non_finite_count"] == 2
    assert result["finite_count"] == 3
    assert result["mean"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0


def test_undefined_std_and_skewness_are_not_reported_as_nan():
    result = DatasetStatistics().analyze(pd.DataFrame({"value": [7.0]}))[0]

    assert result["mean"] == 7.0
    assert result["std"] is None
    assert result["skewness"] is None


def test_pandas_string_columns_have_categorical_summary():
    frame = pd.DataFrame({"category": pd.Series(["A", "B", "A"], dtype="str")})

    result = DatasetStatistics().analyze(frame)[0]

    assert result["category_count"] == 2
    assert result["top_values"][0] == {"value": "A", "frequency": 2}
