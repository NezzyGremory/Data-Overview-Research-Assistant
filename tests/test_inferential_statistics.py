import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.analyzer.inferential_statistics import InferentialStatisticsAnalyzer


@pytest.fixture
def analyzer():
    return InferentialStatisticsAnalyzer()


def test_shapiro_wilk_matches_scipy_and_documents_hypotheses(analyzer):
    values = np.array([-1.2, -0.5, 0.1, 0.3, 0.6, 1.4, 1.8, -0.2])
    result = analyzer.analyze(pd.DataFrame({"x": values}), "shapiro", column="x")

    expected = stats.shapiro(values)
    assert result["status"] == "COMPLETED"
    assert result["statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["hypotheses"]["null"]
    assert result["assumptions"]
    assert result["sample_size"] == len(values)


@pytest.mark.parametrize("test", ["welch_t", "mann_whitney", "anova", "welch_anova", "kruskal", "levene", "bartlett"])
def test_group_test_methods_return_finite_measures(analyzer, test):
    frame = pd.DataFrame({
        "value": [1.0, 2.0, 1.5, 2.2, 5.0, 5.3, 4.8, 5.2, 8.0, 8.2, 7.7, 8.4],
        "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
    })
    if test in {"welch_t", "mann_whitney"}:
        frame = frame[frame.group != "c"]
    result = analyzer.analyze(frame, test, column="value", group_column="group")

    assert result["status"] == "COMPLETED"
    assert np.isfinite(result["statistic"])
    assert 0 <= result["p_value"] <= 1
    assert result["decision"] in {"TOLAK_H0", "GAGAL_MENOLAK_H0"}
    assert result["assumptions"]


@pytest.mark.parametrize("test", ["paired_t", "wilcoxon"])
def test_paired_test_keeps_pairing_and_ci_when_available(analyzer, test):
    frame = pd.DataFrame({"before": [8, 10, 9, 12, 11, 10], "after": [7, 9, 8, 10, 11, 8]})
    result = analyzer.analyze(frame, test, column="before", second_column="after")

    assert result["status"] == "COMPLETED"
    assert result["sample_size"] == 6
    assert result["effect_size"]["value"] is not None
    assert (result["confidence_interval"] is not None) == (test == "paired_t")
    assert result["assumptions"][0]["status"] == "WARNING"


@pytest.mark.parametrize("test", ["chi_square", "fisher"])
def test_categorical_tests_match_reference(analyzer, test):
    table = np.array([[8, 2], [1, 9]])
    data = []
    for row, left in enumerate(["A", "B"]):
        for col, right in enumerate(["X", "Y"]):
            data.extend([{"left": left, "right": right}] * int(table[row, col]))
    result = analyzer.analyze(pd.DataFrame(data), test, column="left", second_column="right")

    if test == "chi_square":
        expected = stats.chi2_contingency(table, correction=False)
        assert result["statistic"] == pytest.approx(expected.statistic)
        assert result["p_value"] == pytest.approx(expected.pvalue)
    else:
        expected = stats.fisher_exact(table)
        assert result["statistic"] == pytest.approx(expected.statistic)
        assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["sample_size"] == int(table.sum())


@pytest.mark.parametrize("test", ["pearson", "spearman", "kendall"])
def test_correlations_match_reference_and_pearson_has_ci(analyzer, test):
    x = np.array([1, 2, 3, 4, 5, 7], dtype=float)
    y = np.array([2, 1, 4, 3, 7, 5], dtype=float)
    result = analyzer.analyze(pd.DataFrame({"x": x, "y": y}), test, column="x", second_column="y")
    reference = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[test](x, y)

    assert result["statistic"] == pytest.approx(reference.statistic)
    assert result["p_value"] == pytest.approx(reference.pvalue)
    assert (result["confidence_interval"] is not None) == (test == "pearson")
    assert any("sebab-akibat" in item["detail"] for item in result["assumptions"])


@pytest.mark.parametrize("test,kwargs", [
    ("welch_t", {"column": "x", "group_column": "group"}),
    ("paired_t", {"column": "x", "second_column": "y"}),
    ("pearson", {"column": "constant", "second_column": "y"}),
    ("fisher", {"column": "cat1", "second_column": "cat2"}),
])
def test_invalid_designs_fail_with_actionable_message(analyzer, test, kwargs):
    frame = pd.DataFrame({
        "x": [1, 2, 3], "y": [1, 2, 3], "constant": [1, 1, 1],
        "group": ["only", "only", "only"], "cat1": ["a", "b", "c"],
        "cat2": ["x", "y", "z"],
    })
    result = analyzer.analyze(frame, test, **kwargs)

    assert result["status"] in {"ERROR", "NOT_EVALUABLE"}
    assert result.get("message") or result.get("interpretation")


def test_anderson_darling_reports_critical_values_not_a_fake_p_value(analyzer):
    result = analyzer.analyze(pd.DataFrame({"x": np.arange(1, 20)}), "anderson", column="x")

    assert result["status"] == "COMPLETED"
    assert result["p_value"] is None
    assert "5%" in result["critical_values"]
    assert result["decision_alpha"] == 0.05
