"""Normalisasi, analisis, dan pembersihan missing value secara umum."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.ml.target_detector import MLTargetDetector

logger = logging.getLogger(__name__)

MISSING_TOKENS = {
    "", "-", "--", "?", "na", "n/a", "nan", "null", "none", "nil",
    "kosong", "tanpa data", "tidak ada", "tidak tersedia", "#n/a", "#value!",
    # Markers used by existing imports and analysis reports.
    "#na", "missing", "unknown", "not available", "n.a.", "undefined",
    "missing value", "inf", "+inf", "-inf", "infinity", "+infinity",
    "-infinity",
}
INVISIBLE_WHITESPACE = "\u200b\u200c\u200d\ufeff"
IDENTIFIER_NAME = re.compile(
    r"(^|[^a-z0-9])(id|identifier|kode|code|zip|postal|postcode|phone|telephone|tel|nik|ssn|account)([^a-z0-9]|$)",
    re.IGNORECASE,
)
CURRENCY_PREFIX = re.compile(
    r"^(?:rp\.?\s*|idr\s*|usd\s*|eur\s*|\$\s*|\u20ac\s*|\u00a3\s*)",
    re.IGNORECASE,
)


@dataclass
class CleanResult:
    """Hasil bersih beserta data audit dan laporan terstruktur."""

    cleaned_df: pd.DataFrame
    rows_before: int
    rows_after: int
    rows_dropped: int
    percent_dropped: float
    dropped_per_column: dict[str, int]
    zero_columns_used: list[str]
    warnings: list[str] = field(default_factory=list)
    dropped_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    report: dict[str, Any] = field(default_factory=dict)
    formatted_report: str = ""


def _selected_columns(dataframe: pd.DataFrame, columns: Sequence[str] | None) -> list[Any]:
    selected = list(dataframe.columns) if columns is None else list(columns)
    missing_columns = [column for column in selected if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Kolom tidak ditemukan: {', '.join(map(str, missing_columns))}")
    return selected


def _clean_string(value: Any) -> Any:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    text = str(value)
    for character in INVISIBLE_WHITESPACE:
        text = text.replace(character, "")
    return text.strip()


def _is_identifier_column(name: Any, series: pd.Series) -> bool:
    normalized_name = re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")
    if IDENTIFIER_NAME.search(str(name)) or normalized_name.endswith(
        ("id", "identifier", "idnumber", "code", "zipcode", "postalcode", "phone", "telephone", "tel", "nik", "ssn")
    ):
        return True
    present = series.dropna().astype(str).str.strip()
    if len(present) < 3:
        return False
    # Long digit-only strings and consistently zero-padded values are usually
    # identifiers. Keep them as text even when every value parses as a number.
    digit_strings = present.str.fullmatch(r"\+?\d+")
    if digit_strings.mean() >= 0.9 and present[digit_strings].str.len().ge(9).mean() >= 0.8:
        return True
    return bool(digit_strings.mean() >= 0.9 and present.str.match(r"0\d+").mean() >= 0.5)


def _parse_number(value: Any) -> float | None:
    """Parse common localized number strings without choosing a global locale."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        return number if np.isfinite(number) else None
    if not pd.api.types.is_scalar(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    text = _clean_string(value)
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = CURRENCY_PREFIX.sub("", text)
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    text = re.sub(r"(?i)\b(?:idr|usd|eur|rp)\b", "", text)
    text = re.sub(r"[\s\u00a0'\u2019]", "", text)
    text = re.sub(r"[^0-9,\.\-+]", "", text)
    if not text or text in {"-", "+", ".", ","}:
        return None

    comma_positions = [m.start() for m in re.finditer(",", text)]
    dot_positions = [m.start() for m in re.finditer(r"\.", text)]
    if comma_positions and dot_positions:
        decimal_separator = "," if comma_positions[-1] > dot_positions[-1] else "."
        grouping_separator = "." if decimal_separator == "," else ","
        text = text.replace(grouping_separator, "")
        if decimal_separator == ",":
            text = text.replace(",", ".")
    elif comma_positions or dot_positions:
        separator = "," if comma_positions else "."
        pieces = text.split(separator)
        if len(pieces) > 2:
            tail = pieces[-1]
            if len(tail) in {1, 2}:
                text = "".join(pieces[:-1]) + "." + tail
            else:
                text = "".join(pieces)
        else:
            tail = pieces[-1]
            # Three trailing digits normally mean a thousands group (2,286).
            if len(tail) == 3 and pieces[0].lstrip("+-"):
                text = "".join(pieces)
            else:
                text = pieces[0] + "." + tail
    try:
        parsed = float(text)
    except ValueError:
        return None
    if negative:
        parsed = -abs(parsed)
    return parsed if np.isfinite(parsed) else None


def _numeric_candidate(series: pd.Series) -> tuple[pd.Series, float]:
    present = series.notna()
    if not present.any():
        return pd.Series(np.nan, index=series.index, dtype="float64"), 0.0
    parsed = series.map(_parse_number).astype("float64")
    ratio = float(parsed[present].notna().mean())
    return parsed, ratio


def _auto_zero_columns(dataframe: pd.DataFrame, candidates: Sequence[Any]) -> list[Any]:
    suspicious: list[Any] = []
    for column in candidates:
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if len(values) < 5:
            continue
        zero_ratio = float(values.eq(0).mean())
        nonzero = values[values.ne(0)].abs()
        if nonzero.empty:
            continue
        median_nonzero = float(nonzero.median())
        if zero_ratio >= 0.2 and median_nonzero >= 10:
            suspicious.append(column)
    return suspicious


def _normalize_frame(
    dataframe: pd.DataFrame,
    *,
    extra_placeholders: Sequence[str] | None = None,
    numeric_threshold: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str], dict[str, int]]:
    if not 0 <= numeric_threshold <= 1:
        raise ValueError("numeric_threshold harus berada di antara 0 dan 1.")
    normalized = dataframe.copy(deep=True)
    placeholders = {str(token).strip().casefold() for token in MISSING_TOKENS}
    placeholders.update(str(token).strip().casefold() for token in (extra_placeholders or []))
    placeholder_counts: dict[str, int] = {}
    detected_types: dict[str, str] = {}

    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_bool_dtype(series.dtype):
            detected_types[str(column)] = "boolean"
            continue
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            detected_types[str(column)] = "datetime"
            continue

        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            if isinstance(series.dtype, pd.CategoricalDtype):
                cleaned = series.astype("string").map(_clean_string)
            else:
                cleaned = series.map(
                    lambda value: _clean_string(value) if isinstance(value, str) else value
                )
            token_mask = cleaned.map(
                lambda value: isinstance(value, str) and value.casefold() in placeholders
            )
            placeholder_counts[str(column)] = int(token_mask.sum())
            cleaned = cleaned.mask(token_mask)
            if pd.api.types.is_numeric_dtype(series.dtype):
                cleaned = pd.to_numeric(cleaned, errors="coerce")
            parsed, ratio = _numeric_candidate(cleaned)
            if ratio >= numeric_threshold and not _is_identifier_column(column, cleaned):
                normalized[column] = parsed
                detected_types[str(column)] = "numeric"
                continue

            present = cleaned.dropna()
            if len(present) >= 3:
                date_values = pd.to_datetime(present, errors="coerce", format="mixed", dayfirst=True)
                date_ratio = float(date_values.notna().mean()) if len(present) else 0.0
            else:
                date_values = pd.Series(dtype="datetime64[ns]")
                date_ratio = 0.0
            if date_ratio >= numeric_threshold and not _is_identifier_column(column, cleaned):
                converted = pd.to_datetime(cleaned, errors="coerce", format="mixed", dayfirst=True)
                normalized[column] = converted
                detected_types[str(column)] = "datetime"
            else:
                normalized[column] = cleaned
                distinct_ratio = present.nunique(dropna=True) / max(len(present), 1)
                detected_types[str(column)] = "categorical" if distinct_ratio <= 0.5 else "text"
            continue

        if pd.api.types.is_numeric_dtype(series.dtype):
            normalized[column] = series.replace([np.inf, -np.inf], np.nan)
            detected_types[str(column)] = "numeric"
        else:
            detected_types[str(column)] = "categorical"

    missing_counts = {str(column): int(normalized[column].isna().sum()) for column in normalized.columns}
    return normalized, placeholder_counts, detected_types, missing_counts


def normalize_missing(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    zero_as_missing: bool | str = False,
    zero_columns: Sequence[str] | None = None,
    extra_placeholders: Sequence[str] | None = None,
    numeric_threshold: float = 0.9,
) -> pd.DataFrame:
    """Kembalikan salinan DataFrame dengan placeholder dan tipe angka dinormalisasi."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")
    selected = _selected_columns(df, columns)
    normalized, _, _, _ = _normalize_frame(
        df, extra_placeholders=extra_placeholders, numeric_threshold=numeric_threshold
    )
    if columns is not None:
        for column in df.columns:
            if column not in selected:
                normalized[column] = df[column].copy(deep=True)
    if zero_as_missing not in {False, True, "auto"}:
        raise ValueError("zero_as_missing harus False, True, atau 'auto'.")
    requested_zeros = list(zero_columns or [])
    if zero_as_missing is True:
        requested_zeros = list(dict.fromkeys([*requested_zeros, *selected]))
    elif zero_as_missing == "auto":
        requested_zeros = list(dict.fromkeys([*requested_zeros, *_auto_zero_columns(normalized, selected)]))
    for column in requested_zeros:
        if column not in selected:
            continue
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = normalized[column].mask(numeric.eq(0).fillna(False))
    return normalized


def _zero_columns_for_mode(
    dataframe: pd.DataFrame,
    selected: Sequence[str],
    zero_mode: str | None,
    zero_columns: Sequence[str] | None,
    zero_as_missing: bool | str,
) -> list[str]:
    missing_zero_columns = [column for column in (zero_columns or []) if column not in dataframe.columns]
    if missing_zero_columns:
        raise KeyError(f"Kolom zero_cols tidak ditemukan: {', '.join(map(str, missing_zero_columns))}")
    if zero_mode is not None:
        if zero_mode not in {"off", "selective", "strict"}:
            raise ValueError("zero_mode harus 'off', 'selective', atau 'strict'.")
        if zero_mode == "off":
            return []
        if zero_mode == "selective":
            candidates = [column for column in selected if column in set(zero_columns or [])]
        else:
            candidates = list(selected)
    elif zero_as_missing is True:
        candidates = list(selected)
    elif zero_as_missing == "auto":
        # Auto detection is performed after placeholder and numeric parsing.
        candidates = []
    elif zero_as_missing is False:
        candidates = []
    else:
        raise ValueError("zero_as_missing harus False, True, atau 'auto'.")
    explicit = [column for column in (zero_columns or []) if column in selected]
    candidates = list(dict.fromkeys([*candidates, *explicit]))
    for column in candidates:
        if column not in dataframe.columns:
            raise KeyError(f"Kolom zero_cols tidak ditemukan: {column}")
    return [column for column in candidates if not pd.api.types.is_bool_dtype(dataframe[column].dtype)]


def _build_warnings(
    dataframe: pd.DataFrame,
    zero_columns_used: Sequence[str],
    percent_dropped: float,
) -> list[str]:
    warnings: list[str] = []
    if zero_columns_used:
        try:
            candidates = MLTargetDetector().detect(dataframe=dataframe)
            target_columns = {str(item.get("name")) for item in candidates if item.get("name") is not None}
        except Exception:
            target_columns = set()
        for column in zero_columns_used:
            if str(column) in target_columns:
                warnings.append(f"Kolom target '{column}' ikut menganggap nilai 0 sebagai missing.")
            warnings.append(f"Kolom '{column}': nilai 0 diperlakukan sebagai missing; periksa hasil pratinjau.")
            values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            if not values.empty and set(values.unique().tolist()).issubset({0, 1}):
                warnings.append(f"Kolom biner '{column}' dapat kehilangan salah satu kelas karena nilai 0 dianggap missing.")
    if percent_dropped > 30:
        warnings.append(f"Sebanyak {percent_dropped:.1f}% baris akan dihapus, melebihi ambang 30%.")
    return warnings


def _impute(
    dataframe: pd.DataFrame,
    method: str,
    fill_value: Any,
    group_by: Sequence[str] | None,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = dataframe.copy(deep=True)
    method = method.casefold().replace(" ", "_")
    allowed = {"mean", "median", "mode", "ffill", "bfill", "interpolate", "constant"}
    if method not in allowed:
        raise ValueError(f"method harus salah satu dari: {', '.join(sorted(allowed))}.")
    grouping = list(group_by or [])
    for column in grouping:
        if column not in result.columns:
            raise KeyError(f"Kolom group_by tidak ditemukan: {column}")
    changed: list[str] = []
    for column in columns:
        series = result[column]
        if not series.isna().any() or column in grouping:
            continue
        numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)
        selected_method = method
        if method == "mean" and not numeric:
            selected_method = "mode"
        if method == "median" and not numeric:
            selected_method = "mode"

        if grouping and selected_method in {"mean", "median", "mode"}:
            group_values = result.groupby(grouping, dropna=False, sort=False)[column]
            if selected_method == "mode":
                fills = group_values.transform(
                    lambda values: values.mode(dropna=True).iloc[0] if not values.mode(dropna=True).empty else np.nan
                )
            else:
                fills = group_values.transform(selected_method)
            result[column] = series.fillna(fills)
            if result[column].isna().any():
                fallback = _fill_value(series, selected_method, fill_value)
                result[column] = result[column].fillna(fallback)
        elif selected_method in {"mean", "median", "mode"}:
            result[column] = series.fillna(_fill_value(series, selected_method, fill_value))
        elif selected_method == "constant":
            if fill_value is None:
                raise ValueError("fill_value wajib diisi untuk method='constant'.")
            result[column] = series.fillna(fill_value)
        elif selected_method == "ffill":
            result[column] = series.ffill().bfill()
        elif selected_method == "bfill":
            result[column] = series.bfill().ffill()
        elif selected_method == "interpolate":
            if not numeric:
                raise ValueError(f"interpolate hanya mendukung kolom numerik: {column}")
            result[column] = series.interpolate().ffill().bfill()
        changed.append(str(column))
    still_missing = [column for column in columns if result[column].isna().any()]
    if still_missing:
        raise ValueError(f"Missing value tidak dapat diimputasi pada kolom: {', '.join(map(str, still_missing))}")
    return result, changed


def _fill_value(series: pd.Series, method: str, constant: Any) -> Any:
    if method == "constant":
        if constant is None:
            raise ValueError("fill_value wajib diisi untuk method='constant'.")
        return constant
    if method == "mode":
        modes = series.mode(dropna=True)
        if modes.empty:
            raise ValueError("Mode tidak dapat dihitung karena kolom tidak memiliki nilai teramati.")
        return modes.iloc[0]
    value = getattr(series, method)(skipna=True)
    if pd.isna(value):
        raise ValueError(f"{method} tidak dapat dihitung untuk kolom tanpa nilai teramati.")
    return value


def format_clean_report(report: dict[str, Any]) -> str:
    """Format ringkasan pembersihan untuk log, dialog, atau CLI."""
    lines = [
        f"Strategi: {report.get('strategy', 'report')}",
        f"Baris: {report.get('rows_before', 0):,} → {report.get('rows_after', 0):,}",
        f"Baris dihapus: {report.get('rows_dropped', 0):,}",
        f"Kolom dihapus: {report.get('columns_dropped', 0):,}",
        f"Missing sebelum/sesudah: {report.get('missing_before', 0):,} / {report.get('missing_after', 0):,}",
        f"Placeholder dinormalisasi: {report.get('placeholders_converted', 0):,}",
        f"Nilai 0 dinormalisasi: {report.get('zeros_as_missing', 0):,}",
    ]
    per_column = report.get("columns", {})
    if per_column:
        lines.append("Rincian kolom:")
        for name, values in per_column.items():
            lines.append(
                f"- {name}: missing {values.get('missing_before', 0):,} ({values.get('missing_before_pct', 0):.1f}%) → "
                f"{values.get('missing_after', 0):,} ({values.get('missing_after_pct', 0):.1f}%); "
                f"placeholder {values.get('placeholders_converted', 0):,}; 0 {values.get('zeros_as_missing', 0):,}"
            )
    return "\n".join(lines)


def drop_missing(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    how: str = "any",
    zero_mode: str | None = None,
    zero_columns: Sequence[str] | None = None,
    *,
    extra_placeholders: Sequence[str] | None = None,
    zero_as_missing: bool | str = "auto",
    zero_cols: Sequence[str] | None = None,
    strategy: str = "drop_rows",
    col_threshold: float = 0.5,
    thresh: int | None = None,
    method: str = "median",
    fill_value: Any = None,
    group_by: Sequence[str] | None = None,
    numeric_threshold: float = 0.9,
    subset: Sequence[str] | None = None,
) -> CleanResult:
    """Bersihkan DataFrame tanpa mengubah input; API lama tetap didukung."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")
    if how not in {"any", "all"}:
        raise ValueError("how harus 'any' atau 'all'.")
    if strategy not in {"drop_rows", "drop_cols", "impute", "report"}:
        raise ValueError("strategy harus 'drop_rows', 'drop_cols', 'impute', atau 'report'.")
    if not 0 <= col_threshold <= 1:
        raise ValueError("col_threshold harus berada di antara 0 dan 1.")
    if columns is not None and subset is not None:
        raise ValueError("Gunakan columns atau subset, jangan keduanya sekaligus.")
    selected = _selected_columns(df, subset if subset is not None else columns)
    selected_zero_columns = list(dict.fromkeys([*(zero_columns or []), *(zero_cols or [])]))
    zero_used = _zero_columns_for_mode(df, selected, zero_mode, selected_zero_columns, zero_as_missing)
    normalized, placeholder_counts, detected_types, _ = _normalize_frame(
        df, extra_placeholders=extra_placeholders, numeric_threshold=numeric_threshold
    )
    if zero_mode is None and zero_as_missing == "auto":
        zero_used = [
            column for column in _auto_zero_columns(normalized, selected)
            if not _is_identifier_column(column, normalized[column])
        ] + [column for column in selected_zero_columns if column in selected]
    # Infer numeric values before treating zeros as missing, but preserve ID/text columns.
    if zero_used:
        usable_zero_columns: list[str] = []
        for column in zero_used:
            parsed, ratio = _numeric_candidate(normalized[column])
            if ratio >= numeric_threshold and not _is_identifier_column(column, normalized[column]):
                normalized[column] = parsed
                usable_zero_columns.append(column)
            else:
                logger.warning("Kolom %s tidak diproses sebagai numerik untuk deteksi nilai 0.", column)
        zero_used = usable_zero_columns

    before_missing = normalized.isna().sum()
    pre_zero_frame = normalized.copy(deep=True)
    zero_counts = {
        str(column): int(pd.to_numeric(normalized[column], errors="coerce").eq(0).fillna(False).sum())
        for column in zero_used
    }
    for column in zero_used:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = normalized[column].mask(numeric.eq(0).fillna(False))
    analysis_frame = normalized.copy(deep=True)

    rows_before, columns_before = df.shape
    dropped_rows = df.iloc[0:0].copy()
    dropped_columns: list[str] = []
    imputed_columns: list[str] = []
    working = normalized.copy(deep=True)
    if strategy == "drop_rows":
        if not selected:
            raise ValueError("Pilih setidaknya satu kolom untuk drop_rows.")
        if thresh is not None:
            if thresh < 0 or thresh > len(selected):
                raise ValueError("thresh harus berada di antara 0 dan jumlah kolom yang dipilih.")
            row_mask = working.loc[:, selected].notna().sum(axis=1) < thresh
        else:
            row_mask = working.loc[:, selected].isna().any(axis=1) if how == "any" else working.loc[:, selected].isna().all(axis=1)
        dropped_rows = df.loc[row_mask].copy(deep=True)
        working = working.loc[~row_mask].copy(deep=True)
        working = working.reset_index(drop=True)
    elif strategy == "drop_cols":
        denominator = max(len(working), 1)
        dropped_columns = [column for column in selected if working[column].isna().sum() / denominator > col_threshold]
        working = working.drop(columns=dropped_columns).copy()
        if working.shape[1] == 0:
            raise ValueError("Drop cols akan menghapus semua kolom. Turunkan col_threshold atau tinjau pilihan.")
    elif strategy == "impute":
        working, imputed_columns = _impute(working, method, fill_value, group_by, selected)
    # report strategy intentionally leaves original values and shape untouched.
    if strategy == "report":
        working = df.copy(deep=True)
        dropped_rows = df.iloc[0:0].copy()

    report_frame = analysis_frame if strategy == "report" else working
    after_missing = report_frame.isna().sum()
    rows_after = len(working)
    rows_dropped = rows_before - rows_after if strategy == "drop_rows" else 0
    columns_after = len(working.columns)
    percent_dropped = rows_dropped / rows_before * 100 if rows_before else 0.0
    zeros_total = sum(zero_counts.values())
    columns_report: dict[str, dict[str, Any]] = {}
    for column in df.columns:
        name = str(column)
        before_count = int(before_missing[column])
        after_count = int(after_missing[column]) if column in after_missing.index else 0
        columns_report[name] = {
            "type": detected_types.get(name, str(df[column].dtype)),
            "missing_before": before_count,
            "missing_before_pct": before_count / rows_before * 100 if rows_before else 0.0,
            "missing_after": after_count,
            "missing_after_pct": after_count / rows_after * 100 if rows_after else 0.0,
            "placeholders_converted": int(placeholder_counts.get(name, 0)),
            "zeros_as_missing": int(zero_counts.get(name, 0)),
        }
    report = {
        "strategy": strategy,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_dropped,
        "columns_before": columns_before,
        "columns_after": columns_after,
        "columns_dropped": len(dropped_columns),
        "columns_dropped_names": [str(column) for column in dropped_columns],
        "columns_imputed": imputed_columns,
        "columns_imputed_count": len(imputed_columns),
        "cells_imputed": int(sum(
            int(before_missing[column]) + zero_counts.get(str(column), 0)
            for column in selected
            if str(column) in imputed_columns
        )),
        "missing_before": int(before_missing.sum()),
        "missing_after": int(after_missing.sum()),
        "placeholders_converted": sum(placeholder_counts.values()),
        "zeros_as_missing": zeros_total,
        "zero_columns": [str(column) for column in zero_used],
        "columns": columns_report,
        "extra_placeholders": list(extra_placeholders or []),
    }
    warnings = _build_warnings(pre_zero_frame, zero_used, percent_dropped)
    if df.empty or df.shape[1] == 0:
        warnings.append("Dataset kosong atau tidak memiliki kolom; tidak ada pembersihan yang dilakukan.")
    if strategy == "drop_rows" and rows_dropped == 0:
        warnings.append("Tidak ada yang dihapus.")
    if strategy == "report":
        warnings.append("Mode report: DataFrame tidak diubah; tinjau laporan sebelum mengonfirmasi strategi.")
    if strategy == "drop_rows" and rows_after == 0 and rows_before:
        warnings.append("Hasil drop berisi 0 baris; tinjau hasil sebelum disimpan.")
    if strategy == "drop_rows" and how == "all":
        warnings.append("Mode 'all' hanya menghapus baris jika semua kolom terpilih missing.")
    if strategy == "drop_rows" and how == "any" and working.loc[:, selected].isna().any().any():
        raise RuntimeError("Verifikasi drop gagal: masih ada missing pada kolom terpilih.")
    report["warnings"] = warnings
    formatted_report = format_clean_report(report)
    logger.info("Pembersihan missing selesai: strategi=%s baris=%s→%s", strategy, rows_before, rows_after)
    return CleanResult(
        cleaned_df=working,
        rows_before=rows_before,
        rows_after=rows_after,
        rows_dropped=rows_dropped,
        percent_dropped=percent_dropped,
        dropped_per_column={str(column): int(before_missing[column]) for column in df.columns},
        zero_columns_used=[str(column) for column in zero_used],
        warnings=warnings,
        dropped_df=dropped_rows.copy(deep=True),
        report=report,
        formatted_report=formatted_report,
    )


def preview_drop(df: pd.DataFrame, columns: Sequence[str] | None = None, how: str = "any", zero_mode: str | None = None, zero_columns: Sequence[str] | None = None, strategy: str = "drop_rows", **options: Any) -> CleanResult:
    """Hitung pratinjau drop tanpa memutasi DataFrame input."""
    return drop_missing(df, columns=columns, how=how, zero_mode=zero_mode, zero_columns=zero_columns, strategy=strategy, **options)


def export_cleaned_csv(
    df: pd.DataFrame,
    path: str | Path,
    original_path: str | Path | None = None,
    dropped_rows: pd.DataFrame | None = None,
) -> Path:
    """Ekspor hasil dan file *_dropped_rows.csv untuk audit jika tersedia."""
    target = Path(path).expanduser()
    if not target.suffix:
        target = target.with_suffix(".csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    original = Path(original_path).expanduser() if original_path is not None else None
    if original is not None and target.resolve() == original.resolve():
        raise ValueError("File tujuan sama dengan CSV asli. Pilih nama atau folder lain.")
    audit_path = target.with_name(f"{target.stem}_dropped_rows.csv")
    if original is not None and audit_path.resolve() == original.resolve():
        raise ValueError("File audit akan menimpa file asli. Pilih nama atau folder lain.")
    try:
        df.to_csv(target, index=False, encoding="utf-8-sig")
        if dropped_rows is not None and not dropped_rows.empty:
            dropped_rows.to_csv(audit_path, index=True, index_label="source_index", encoding="utf-8-sig")
    except PermissionError as error:
        raise OSError(f"Tidak dapat menulis '{target}'. File mungkin sedang terbuka atau tidak memiliki izin tulis.") from error
    except OSError as error:
        raise OSError(f"Gagal mengekspor dataset ke '{target}': {error}") from error
    return target
