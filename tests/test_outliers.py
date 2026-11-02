import numpy as np
import pandas as pd

from app.analyzer.outliers import OutlierAnalyzer


def test_infinite_values_are_not_counted_as_statistical_outliers():
    frame = pd.DataFrame({"value": [10.0, 10.0, 11.0, 11.0, 12.0, np.inf]})

    result = OutlierAnalyzer().analyze(frame)

    assert len(result) == 1
    assert result[0]["outlier_count"] == 0
    assert np.isfinite(result[0]["lower_bound"])
    assert np.isfinite(result[0]["upper_bound"])
