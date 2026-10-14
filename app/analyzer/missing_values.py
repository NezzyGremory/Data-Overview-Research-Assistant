import pandas as pd

from app.analyzer.cleaner import MISSING_TOKENS, drop_missing, normalize_missing


MISSING_TEXT_VALUES = MISSING_TOKENS


def normalize_missing_values(
    dataframe: pd.DataFrame,
    extra_placeholders: list[str] | None = None,
    zero_as_missing: bool | str = False,
) -> pd.DataFrame:
    """Compat wrapper for the shared missing-value normalizer."""
    return normalize_missing(
        dataframe,
        extra_placeholders=extra_placeholders,
        zero_as_missing=zero_as_missing,
    )


def drop_missing_rows(
    dataframe: pd.DataFrame,
    zero_as_missing: bool | str = "auto",
    zero_cols: list[str] | None = None,
    extra_placeholders: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop every row containing an actual or recognized textual missing value."""
    if dataframe.empty or dataframe.shape[1] == 0:
        raise ValueError("Dataset tidak memiliki baris atau kolom untuk dibersihkan.")

    result = drop_missing(
        dataframe,
        zero_as_missing=zero_as_missing,
        zero_cols=zero_cols,
        extra_placeholders=extra_placeholders,
        strategy="drop_rows",
    )
    if int(result.cleaned_df.isna().sum().sum()):
        raise ValueError(
            f"Drop missing belum tuntas: masih ada {int(result.cleaned_df.isna().sum().sum()):,} nilai kosong."
        )
    if result.cleaned_df.empty:
        raise ValueError(
            "Tidak ada baris yang seluruh kolomnya terisi. Dataset tidak diubah."
        )
    report = dict(result.report)
    report["rows_with_missing"] = result.rows_dropped
    report["how"] = "any"
    return result.cleaned_df, report


def impute_missing_values(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Impute usable columns and remove columns with no observed values."""
    if dataframe.empty or dataframe.shape[1] == 0:
        raise ValueError("Dataset tidak memiliki baris atau kolom untuk dibersihkan.")

    cleaned = normalize_missing_values(dataframe)
    missing_before = int(cleaned.isna().sum().sum())
    empty_columns = [column for column in cleaned.columns if cleaned[column].isna().all()]
    if len(empty_columns) == len(cleaned.columns):
        raise ValueError(
            "Semua kolom berisi nilai kosong. Tidak ada nilai yang bisa digunakan "
            "untuk menghitung imputasi."
        )

    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)

    imputed_columns = []
    for column in cleaned.columns:
        series = cleaned[column]
        if not series.isna().any():
            continue

        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        ):
            fill_value = series.median(skipna=True)
            if pd.isna(fill_value):
                raise ValueError(
                    f"Nilai pengganti untuk kolom numerik '{column}' tidak dapat dihitung."
                )

            if pd.api.types.is_integer_dtype(series.dtype) and float(fill_value) % 1:
                cleaned[column] = series.astype("Float64").fillna(fill_value)
            else:
                cleaned[column] = series.fillna(fill_value)
            method = "median"
        else:
            modes = series.mode(dropna=True)
            if modes.empty:
                raise ValueError(
                    f"Nilai pengganti untuk kolom '{column}' tidak dapat dihitung."
                )
            fill_value = modes.iloc[0]
            if pd.api.types.is_bool_dtype(series.dtype):
                fill_value = bool(fill_value)
            cleaned[column] = series.fillna(fill_value)
            method = "mode"

        imputed_columns.append(
            {"column": str(column), "method": method, "fill_value": str(fill_value)}
        )

    remaining_missing = int(cleaned.isna().sum().sum())
    if remaining_missing:
        raise ValueError(
            f"Pembersihan belum tuntas: masih ada {remaining_missing:,} nilai kosong."
        )

    return cleaned, {
        "missing_before": missing_before,
        "missing_after": remaining_missing,
        "imputed_columns": imputed_columns,
        "dropped_empty_columns": [str(column) for column in empty_columns],
    }


class MissingValueAnalyzer:
    """Analyze missing values in a dataset."""

    def analyze(self, dataframe: pd.DataFrame) -> list[dict]:
        normalized = normalize_missing(dataframe)
        total_rows = len(normalized)
        results = []

        for column in normalized.columns:
            missing_count = int(normalized[column].isna().sum())
            if pd.api.types.is_bool_dtype(normalized[column].dtype):
                zero_count = 0
            else:
                numeric_values = pd.to_numeric(
                    normalized[column], errors="coerce"
                )
                zero_count = int(numeric_values.eq(0).fillna(False).sum())

            results.append(
                {
                    "column": str(column),
                    "missing_count": missing_count,
                    "missing_percentage": (
                        missing_count / total_rows * 100
                        if total_rows > 0
                        else 0
                    ),
                    "zero_count": zero_count,
                    "zero_percentage": (
                        zero_count / total_rows * 100
                        if total_rows > 0
                        else 0
                    ),
                }
            )

        return results
