from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from app.analyzer.cleaner import (
    drop_missing,
    export_cleaned_csv,
    normalize_missing,
    preview_drop,
)
from app.analyzer.loader import DatasetLoader
from app.analyzer.missing_values import MissingValueAnalyzer


@pytest.fixture
def dirty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [1.0, 0.0, 2.0, float("inf"), float("-inf"), float("nan"), 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "text": ["valid", "0", "0.0", "0.00", "", "   ", "NA", "N/A", "#N/A", "#NA", "nan", "null", "none", "nil", "-"],
            "other_tokens": ["ok", "--", "?", "missing", "unknown", "not available", "safe", "safe", "safe", "safe", "safe", "safe", "safe", "safe", "safe"],
            "flag": pd.Series([True, False, True] * 5, dtype="boolean"),
        }
    )


def test_normalize_recognizes_missing_forms_and_preserves_input(dirty_dataframe):
    original = dirty_dataframe.copy(deep=True)

    normalized = normalize_missing(dirty_dataframe)

    assert normalized.loc[4:14, "text"].isna().all()
    assert pd.isna(normalized.loc[1, "number"]) is False
    assert normalized.loc[1, "text"] == "0"
    assert normalized.loc[2, "text"] == "0.0"
    assert normalized.loc[3, "text"] == "0.00"
    assert normalized["number"].iloc[3:5].isna().all()
    assert normalized["other_tokens"].iloc[1:6].isna().all()
    assert normalized["flag"].equals(dirty_dataframe["flag"])
    pd.testing.assert_frame_equal(dirty_dataframe, original)


def test_drop_any_removes_missing_rows_and_keeps_original_unchanged(dirty_dataframe):
    original = dirty_dataframe.copy(deep=True)

    result = drop_missing(dirty_dataframe, zero_mode="off")

    assert result.rows_before == len(dirty_dataframe)
    assert result.rows_after == 1
    assert result.rows_dropped == len(dirty_dataframe) - 1
    assert not result.cleaned_df.isna().any().any()
    pd.testing.assert_frame_equal(dirty_dataframe, original)


def test_how_any_and_all_have_expected_row_counts():
    dataframe = pd.DataFrame({
        "left": [None, 0, None, 2],
        "right": [None, "ok", "ok", None],
    })

    any_result = drop_missing(dataframe, how="any")
    all_result = drop_missing(dataframe, how="all")

    assert any_result.rows_after == 1
    assert any_result.rows_dropped == 3
    assert all_result.rows_after == 3
    assert all_result.rows_dropped == 1
    assert all_result.cleaned_df.isna().any().any()
    assert any("Mode 'all'" in warning for warning in all_result.warnings)


def test_zero_modes_off_selective_and_strict():
    dataframe = pd.DataFrame({
        "numeric": [0, 1, 2],
        "text_number": ["1", "0", "2"],
        "keep_zero": [0, 1, 2],
        "flag": [True, False, True],
    })

    off = drop_missing(dataframe, zero_mode="off")
    selective = drop_missing(
        dataframe,
        zero_mode="selective",
        zero_columns=["numeric"],
    )
    selective_none = drop_missing(dataframe, zero_mode="selective")
    strict = drop_missing(dataframe, zero_mode="strict")

    assert off.rows_after == 3
    assert selective.rows_after == 2
    assert selective_none.rows_after == 3
    assert selective.cleaned_df["text_number"].tolist() == [0.0, 2.0]
    assert selective.cleaned_df["keep_zero"].tolist() == [1, 2]
    assert strict.rows_after == 1
    assert strict.cleaned_df["text_number"].tolist() == [2.0]
    assert strict.cleaned_df["flag"].tolist() == [True]
    pd.testing.assert_series_equal(dataframe["flag"], pd.Series([True, False, True], name="flag"))


def test_target_binary_and_large_drop_warnings():
    dataframe = pd.DataFrame({
        "feature": [0, 1, 0, 1, 0],
        "Survived": [0, 1, 0, 1, 0],
    })

    result = drop_missing(dataframe, zero_mode="strict")

    assert any("target 'Survived'" in warning for warning in result.warnings)
    assert any("biner 'feature'" in warning for warning in result.warnings)
    assert any("biner 'Survived'" in warning for warning in result.warnings)
    assert any("30%" in warning for warning in result.warnings)


def test_no_missing_empty_and_all_rows_removed_edges():
    complete = pd.DataFrame({"value": [1, 2]})
    no_op = drop_missing(complete)
    empty = drop_missing(pd.DataFrame({"value": pd.Series(dtype="float64")}))
    all_removed = drop_missing(pd.DataFrame({"value": [0, 0]}), zero_mode="strict")

    assert no_op.rows_dropped == 0
    assert "Tidak ada yang dihapus." in no_op.warnings
    assert empty.rows_before == empty.rows_after == 0
    assert all_removed.rows_after == 0
    assert any("0 baris" in warning for warning in all_removed.warnings)


def test_preview_does_not_change_dataframe(dirty_dataframe):
    original = dirty_dataframe.copy(deep=True)

    preview = preview_drop(dirty_dataframe)

    assert preview.rows_dropped > 0
    pd.testing.assert_frame_equal(dirty_dataframe, original)


def test_drop_report_includes_auditable_dropped_rows_and_resets_index():
    dataframe = pd.DataFrame({"value": [1, None, 3]}, index=[10, 20, 30])

    result = drop_missing(dataframe, zero_as_missing=False)

    assert result.cleaned_df.index.tolist() == [0, 1]
    assert result.dropped_df["value"].isna().tolist() == [True]
    assert result.report["rows_dropped"] == 1
    assert "missing_before_pct" in result.report["columns"]["value"]


def test_export_writes_selected_path_and_rejects_original(tmp_path):
    original_path = tmp_path / "source.csv"
    original_path.write_text("value\n1\n", encoding="utf-8")
    original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
    dataframe = pd.DataFrame({"value": [2, 3]})

    exported = export_cleaned_csv(dataframe, tmp_path / "cleaned", original_path)

    assert exported == tmp_path / "cleaned.csv"
    assert pd.read_csv(exported)["value"].tolist() == [2, 3]
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == original_hash
    with pytest.raises(ValueError, match="sama dengan CSV asli"):
        export_cleaned_csv(dataframe, original_path, original_path)


def test_export_writes_dropped_rows_to_a_sibling_audit_file(tmp_path):
    result = drop_missing(pd.DataFrame({"value": [1, None, 3]}, index=[5, 8, 13]), zero_as_missing=False)

    exported = export_cleaned_csv(
        result.cleaned_df,
        tmp_path / "cleaned.csv",
        dropped_rows=result.dropped_df,
    )

    audit = pd.read_csv(exported.with_name("cleaned_dropped_rows.csv"))
    assert audit.columns.tolist() == ["source_index", "value"]
    assert audit["source_index"].tolist() == [8]


def test_analysis_missing_count_matches_any_drop_rows():
    dataframe = pd.DataFrame({
        "first": ["ok", "NA", "ok", "ok"],
        "second": [1, 2, None, 4],
    })
    result = drop_missing(dataframe, how="any")
    analyzed_missing = sum(
        int(item["missing_count"])
        for item in MissingValueAnalyzer().analyze(dataframe)
    )

    assert analyzed_missing == result.rows_dropped


def test_loader_and_drop_leave_original_csv_bytes_unchanged(tmp_path):
    source = tmp_path / "original.csv"
    source.write_text(
        "value,label\n1,ok\n0,NA\n2,valid\ninf,unknown\n",
        encoding="utf-8",
    )
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    dataframe = DatasetLoader().load_csv(str(source))
    result = drop_missing(dataframe)

    assert result.rows_after == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash


def test_analyzer_reports_zero_counts_separately_from_missing():
    dataframe = pd.DataFrame({
        "numeric": [0, 1, None],
        "text": ["0.0", "ok", "NA"],
        "flag": [False, True, False],
    })

    result = MissingValueAnalyzer().analyze(dataframe)

    assert result[0]["missing_count"] == 1
    assert result[0]["zero_count"] == 1
    assert result[1]["missing_count"] == 1
    assert result[1]["zero_count"] == 1
    assert result[2]["zero_count"] == 0


def test_placeholder_normalization_and_user_extensions_cover_all_columns():
    dataframe = pd.DataFrame({
        "a": [None, "  ", "tidak tersedia", "valid"],
        "b": ["#VALUE!", "tanpa data", "nil", "keep"],
    })

    normalized = normalize_missing(dataframe, extra_placeholders=["redacted"])
    custom = normalize_missing(pd.DataFrame({"value": ["redacted", "ok"]}), extra_placeholders=["redacted"])

    assert normalized.isna().sum().to_dict() == {"a": 3, "b": 3}
    assert custom["value"].isna().sum() == 1
    assert normalized.loc[3].tolist() == ["valid", "keep"]


def test_localized_and_international_numeric_columns_are_inferred():
    dataframe = pd.DataFrame({
        "indonesia": ["2.286,09", "1.000,00", "0,00"],
        "international": ["2,286.09", "1,000.00", "0.00"],
        "currency": ["Rp 1.000", "Rp 2.000", "Rp 3.000"],
        "percentage": ["12%", "15%", "20%"],
        "negative": ["(500)", "(1.250)", "100"],
    })

    normalized = normalize_missing(dataframe)

    assert normalized["indonesia"].tolist() == [2286.09, 1000.0, 0.0]
    assert normalized["international"].tolist() == [2286.09, 1000.0, 0.0]
    assert normalized["currency"].tolist() == [1000.0, 2000.0, 3000.0]
    assert normalized["percentage"].tolist() == [12.0, 15.0, 20.0]
    assert normalized["negative"].tolist() == [-500.0, -1250.0, 100.0]


def test_auto_zero_detection_warns_and_explicit_zero_cols_are_supported():
    dataframe = pd.DataFrame({
        "measurement": ["0,00"] * 6 + ["100,00", "120,00", "150,00", "200,00"],
        "valid_zero": [0, 1] * 5,
    })

    automatic = drop_missing(dataframe)
    explicit = drop_missing(dataframe, zero_as_missing=False, zero_cols=["valid_zero"])

    assert automatic.rows_dropped == 6
    assert "measurement" in automatic.zero_columns_used
    assert any("nilai 0 diperlakukan" in warning for warning in automatic.warnings)
    assert explicit.rows_dropped == 5
    assert "valid_zero" in explicit.zero_columns_used


def test_identifier_and_code_columns_remain_text():
    dataframe = pd.DataFrame({
        "postal_code": ["00123", "04567", "09001"],
        "phone_number": ["081234567890", "081234567891", "081234567892"],
        "account_id": ["00001", "00002", "00003"],
        "amount": ["1.000,00", "2.000,00", "3.000,00"],
    })

    normalized = normalize_missing(dataframe)

    assert normalized["postal_code"].tolist() == ["00123", "04567", "09001"]
    assert normalized["phone_number"].tolist()[0] == "081234567890"
    assert normalized["account_id"].tolist() == ["00001", "00002", "00003"]
    assert normalized["amount"].tolist() == [1000.0, 2000.0, 3000.0]


def test_report_drop_columns_impute_and_group_methods():
    dataframe = pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "value": [1.0, None, 10.0, None],
        "mostly_missing": [None, None, 2, 3],
    })
    original = dataframe.copy(deep=True)

    report = drop_missing(dataframe, strategy="report", zero_as_missing=False)
    dropped_cols = drop_missing(dataframe, strategy="drop_cols", col_threshold=0.49, zero_as_missing=False)
    grouped = drop_missing(dataframe, strategy="impute", method="mean", group_by=["group"], zero_as_missing=False)

    assert report.cleaned_df.equals(original)
    assert dropped_cols.report["columns_dropped_names"] == ["value", "mostly_missing"]
    assert grouped.cleaned_df["value"].tolist() == [1.0, 1.0, 10.0, 10.0]
    assert grouped.report["columns_imputed"] == ["value", "mostly_missing"]


def test_synthetic_finance_and_sensor_data_are_handled_generically():
    finance = pd.DataFrame({
        "account_code": ["0001", "0002", "0003"],
        "balance": ["Rp 2.286,09", "-", "Rp 5.000,00"],
        "rate": ["2,5%", "3,0%", "4,0%"],
    })
    sensor = pd.DataFrame({
        "device_id": ["D-01", "D-02", "D-03"],
        "temperature": ["20.5", "NaN", "22.0"],
        "observed_at": ["2025-01-01", "2025-01-02", "2025-01-03"],
    })

    finance_result = drop_missing(finance, zero_as_missing=False)
    sensor_normalized = normalize_missing(sensor)

    assert finance_result.cleaned_df["balance"].tolist() == [2286.09, 5000.0]
    assert finance_result.dropped_df["account_code"].tolist() == ["0002"]
    assert sensor_normalized["device_id"].tolist() == ["D-01", "D-02", "D-03"]
    assert sensor_normalized["temperature"].isna().sum() == 1
    assert pd.api.types.is_datetime64_any_dtype(sensor_normalized["observed_at"])
