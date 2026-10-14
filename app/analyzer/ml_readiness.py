"""Read-only pre-modeling diagnostics and recommendations."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.analyzer.cleaner import normalize_missing
from app.ml.method_recommender import MethodRecommender
from app.ml.target_detector import MLTargetDetector


ML_PIPELINE_STAGES = (
    "Problem Definition",
    "Data Loading",
    "EDA",
    "Data Cleaning",
    "Train/Test Split",
    "Preprocessing",
    "Feature Engineering and Feature Selection",
    "Model Selection",
    "Model Training",
    "Evaluation and Cross-Validation",
    "Hyperparameter Tuning",
    "Final Test",
    "Interpretation",
    "Model Saving and Deployment",
    "Monitoring",
)
ML_READINESS_LABELS = {
    "sections": (
        "Problem Definition",
        "Data Types Overview",
        "Distribution Analysis",
        "Class Balance",
        "Data Quality Summary",
        "Data Leakage Check",
        "Train/Test Split Recommendation",
        "Preprocessing Recommendation",
        "Feature Selection Hints",
        "ML Pipeline Checklist",
    ),
    "statuses": ("OK", "Warning", "Issue"),
    "checklist_statuses": ("Done", "Partial", "Recommended", "Not Available Yet"),
}
SAMPLE_THRESHOLD = 100_000
ANALYSIS_SAMPLE_SIZE = 10_000
IMPORTANCE_SAMPLE_SIZE = 5_000


def _selected_target(
    dataframe: pd.DataFrame,
    target: str | None = None,
    task: str | None = None,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    candidates = MLTargetDetector().detect(dataframe=dataframe)
    valid_columns = set(dataframe.columns)
    if target not in valid_columns:
        target = None
    if target is None and candidates:
        target = str(candidates[0].get("name") or "") or None
    candidate = next((item for item in candidates if item.get("name") == target), None)
    inferred_task = task
    if inferred_task is None and candidate:
        inferred_task = candidate.get("suggested_task")
    if inferred_task not in {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }:
        inferred_task = None
    return target, inferred_task, candidates


def _as_string(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def detect_column_types(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Classify columns and flag numeric text, identifiers, and constants."""
    source = dataframe.copy(deep=True)
    normalized = normalize_missing(source)
    row_count = len(source)
    results: list[dict[str, Any]] = []

    for column in source.columns:
        raw = source[column]
        series = normalized[column]
        observed = series.dropna()
        unique_count = int(observed.nunique(dropna=True))
        unique_ratio = unique_count / row_count if row_count else 0.0
        numeric_text = False
        normalized_name = str(column).casefold().replace(" ", "_")
        name_suggests_id = any(
            token in normalized_name
            for token in ("_id", "id_", "_key", "key_", "identifier", "record_id", "uuid")
        )

        if observed.empty or unique_count <= 1:
            detected_type = "Constant"
        elif pd.api.types.is_datetime64_any_dtype(raw.dtype):
            detected_type = "Datetime"
        elif pd.api.types.is_bool_dtype(raw.dtype):
            detected_type = "Binary"
        elif name_suggests_id or (unique_ratio >= 0.98 and row_count >= 10):
            detected_type = "ID-like"
        elif pd.api.types.is_numeric_dtype(raw.dtype):
            detected_type = "Binary" if unique_count == 2 else "Numeric"
        else:
            numeric_values = pd.to_numeric(observed, errors="coerce")
            numeric_text = bool(numeric_values.notna().all())
            if numeric_text:
                detected_type = "Binary" if unique_count == 2 else "Numeric"
            elif unique_count == 2:
                detected_type = "Binary"
            else:
                if unique_ratio >= 0.98 and (row_count >= 10 or name_suggests_id):
                    detected_type = "ID-like"
                else:
                    average_length = float(observed.astype("string").str.len().mean())
                    detected_type = (
                        "Text"
                        if unique_count > 50 or average_length >= 40
                        else "Categorical"
                    )

        if detected_type == "Numeric" and unique_ratio >= 0.98 and row_count >= 10:
            detected_type = "ID-like"

        sample_values = [_as_string(value) for value in observed.head(3).tolist()]
        results.append(
            {
                "column": str(column),
                "detected_type": detected_type,
                "dtype": str(raw.dtype),
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "sample_values": sample_values,
                "numeric_as_text": numeric_text,
                "recommendation": (
                    "Exclude from model features" if detected_type == "ID-like"
                    else "Review constant feature" if detected_type == "Constant"
                    else "Convert to numeric before modeling" if numeric_text
                    else ""
                ),
            }
        )
    return results


def distribution_analysis(
    dataframe: pd.DataFrame,
    column_types: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize numeric distribution shape and categorical cardinality."""
    source = dataframe.copy(deep=True)
    sampled = len(source) > SAMPLE_THRESHOLD
    if sampled:
        source = source.sample(ANALYSIS_SAMPLE_SIZE, random_state=42)
    normalized = normalize_missing(source)
    type_map = {item["column"]: item for item in (column_types or detect_column_types(source))}
    numeric_rows = []
    categorical_rows = []

    for column in source.columns:
        info = type_map.get(str(column), {})
        series = normalized[column]
        detected_type = info.get("detected_type")
        if detected_type == "Numeric" or info.get("numeric_as_text"):
            values = pd.to_numeric(series, errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            skewness = float(values.skew()) if len(values) >= 3 else None
            kurtosis = float(values.kurt()) if len(values) >= 4 else None
            shape = (
                "Not applicable for this dataset" if skewness is None
                else "Symmetric" if abs(skewness) < 0.5
                else "Right-skewed" if skewness > 0
                else "Left-skewed"
            )
            numeric_rows.append(
                {
                    "column": str(column),
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "shape": shape,
                    "sample_count": len(values),
                }
            )
        elif detected_type in {"Categorical", "Binary", "Text"}:
            values = series.dropna().astype("string")
            count = int(values.nunique())
            cardinality = "Low" if count <= 10 else "Medium" if count <= 50 else "High"
            top = values.value_counts().head(5)
            categorical_rows.append(
                {
                    "column": str(column),
                    "cardinality": cardinality,
                    "unique_count": count,
                    "top_categories": [
                        {"value": str(value), "count": int(frequency)}
                        for value, frequency in top.items()
                    ],
                }
            )

    count = min(len(dataframe), ANALYSIS_SAMPLE_SIZE) if sampled else len(dataframe)
    return {
        "summary": (
            f"Summarized {len(numeric_rows)} numeric and "
            f"{len(categorical_rows)} categorical columns."
        ),
        "numeric": numeric_rows,
        "categorical": categorical_rows,
        "sampled": sampled,
        "sample_count": count,
        "sample_note": f"Computed on a sample of {count:,} rows" if sampled else "",
    }


def class_balance_analysis(
    dataframe: pd.DataFrame,
    target: str | None,
    task: str | None,
) -> dict[str, Any]:
    """Measure class counts and imbalance for a detected classification target."""
    if not target or target not in dataframe.columns:
        return {
            "applicable": False,
            "summary": "Target not detected",
            "status": "OK",
            "classes": [],
            "imbalance_ratio": None,
        }
    if not task or "classification" not in task:
        return {
            "applicable": False,
            "summary": "Not applicable for this dataset",
            "status": "OK",
            "classes": [],
            "imbalance_ratio": None,
        }

    normalized = normalize_missing(dataframe[[target]])
    counts = normalized[target].value_counts(dropna=True)
    if counts.empty:
        return {
            "applicable": True,
            "summary": "Target not detected",
            "status": "Warning",
            "classes": [],
            "imbalance_ratio": None,
        }
    minimum = int(counts.min())
    ratio = float(counts.max() / minimum) if minimum else float("inf")
    status = "Issue" if ratio > 3 else "Warning" if ratio > 1.5 else "OK"
    total = int(counts.sum())
    return {
        "applicable": True,
        "summary": f"{len(counts)} classes detected for target '{target}'.",
        "status": status,
        "target": str(target),
        "classes": [
            {
                "class": _as_string(value),
                "count": int(count),
                "percentage": float(count / total * 100),
            }
            for value, count in counts.items()
        ],
        "imbalance_ratio": ratio,
    }


def data_quality_summary(
    dataframe: pd.DataFrame,
    missing_values: Sequence[dict[str, Any]] | None,
    duplicates: dict[str, Any] | None,
    outliers: Sequence[dict[str, Any]] | None,
    quality_issues: Sequence[Any] | None = None,
    column_types: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate existing analyzer outputs and type diagnostics without re-counting them."""
    issue_categories: dict[str, list[Any]] = {}
    for item in quality_issues or []:
        if isinstance(item, dict):
            category = item.get("category")
        else:
            category = getattr(item, "category", None)
        if category:
            issue_categories.setdefault(str(category), []).append(item)

    missing_count = sum(
        int(item.get("missing_count", 0))
        for item in (missing_values or [])
        if isinstance(item, dict)
    )
    duplicate_count = int((duplicates or {}).get("duplicate_count", 0))
    outlier_count = sum(
        int(item.get("outlier_count", 0))
        for item in (outliers or [])
        if isinstance(item, dict)
    )
    types = list(column_types or detect_column_types(dataframe))
    constants = [item["column"] for item in types if item["detected_type"] == "Constant"]
    id_like = [item["column"] for item in types if item["detected_type"] == "ID-like"]
    mixed_types = []
    mixed_source = (
        dataframe.sample(ANALYSIS_SAMPLE_SIZE, random_state=42)
        if len(dataframe) > SAMPLE_THRESHOLD else dataframe
    )
    normalized = normalize_missing(mixed_source)
    for column in mixed_source.columns:
        source = mixed_source[column].dropna()
        if not (pd.api.types.is_object_dtype(source.dtype) or pd.api.types.is_string_dtype(source.dtype)):
            continue
        values = normalized[column].dropna()
        numeric_mask = pd.to_numeric(values, errors="coerce").notna()
        if numeric_mask.any() and (~numeric_mask).any():
            mixed_types.append(str(column))

    # Prefer the existing quality diagnoses for constants and identifiers when available.
    if issue_categories.get("constant_columns"):
        constants = sorted({
            str(column)
            for item in issue_categories["constant_columns"]
            for column in (
                item.get("affected_columns", []) if isinstance(item, dict)
                else getattr(item, "affected_columns", [])
            )
        }) or constants
    if issue_categories.get("identifiers"):
        id_like = sorted({
            str(column)
            for item in issue_categories["identifiers"]
            for column in (
                item.get("affected_columns", []) if isinstance(item, dict)
                else getattr(item, "affected_columns", [])
            )
        }) or id_like

    return {
        "summary": (
            f"{missing_count:,} missing cells, {duplicate_count:,} duplicate rows, "
            f"{outlier_count:,} outlier values."
        ),
        "status": "Issue" if missing_count or duplicate_count or outlier_count else "OK",
        "missing_cells": missing_count,
        "duplicate_rows": duplicate_count,
        "outlier_values": outlier_count,
        "constant_columns": constants,
        "id_like_columns": id_like,
        "mixed_type_columns": mixed_types,
    }


def data_leakage_check(
    dataframe: pd.DataFrame,
    target: str | None,
    column_types: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Flag target copies, near-perfect numeric correlation, and ID-like columns."""
    if not target or target not in dataframe.columns:
        return {
            "summary": "Target not detected; target leakage checks are limited.",
            "status": "Warning",
            "findings": [],
        }

    source = dataframe.copy(deep=True)
    normalized = normalize_missing(source)
    types = list(column_types or detect_column_types(source))
    id_columns = {
        item["column"] for item in types if item["detected_type"] == "ID-like"
    }
    findings = []
    target_series = normalized[target]
    target_numeric = pd.to_numeric(target_series, errors="coerce")

    for column in source.columns:
        if column == target:
            continue
        reasons = []
        feature = normalized[column]
        aligned = pd.DataFrame({"feature": feature, "target": target_series}).dropna()
        if not aligned.empty and aligned["feature"].astype("string").equals(
            aligned["target"].astype("string")
        ):
            reasons.append("Identical to target")
        feature_numeric = pd.to_numeric(normalized[column], errors="coerce")
        pair = pd.concat(
            [feature_numeric.rename("feature"), target_numeric.rename("target")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) >= 3 and pair["feature"].nunique() > 1 and pair["target"].nunique() > 1:
            correlation = float(pair["feature"].corr(pair["target"]))
            if abs(correlation) > 0.95:
                reasons.append(f"Target correlation |r| = {abs(correlation):.3f}")
        if str(column) in id_columns:
            reasons.append("ID-like column")
        if reasons:
            findings.append({"column": str(column), "reasons": reasons})
    return {
        "summary": f"{len(findings)} suspicious feature(s) detected.",
        "status": "Warning" if findings else "OK",
        "findings": findings,
    }


def train_test_split_recommendation(
    dataframe: pd.DataFrame,
    target: str | None,
    task: str | None,
) -> dict[str, Any]:
    """Recommend holdout sizing and stratification without splitting the data."""
    rows = len(dataframe)
    if rows == 0:
        return {
            "summary": "Not applicable for this dataset",
            "status": "Issue",
            "test_size": None,
            "train_rows": 0,
            "test_rows": 0,
            "stratify": False,
            "recommendation": "Load rows before planning a split.",
        }
    fraction = 0.3 if rows < 500 else 0.2
    test_rows = min(rows, max(1, int(round(rows * fraction))))
    train_rows = rows - test_rows
    is_classification = bool(task and "classification" in task)
    too_small = rows < 500
    return {
        "summary": f"Recommended {int((1 - fraction) * 100)}/{int(fraction * 100)} split: {train_rows:,} train, {test_rows:,} test rows.",
        "status": "Warning" if too_small else "OK",
        "test_size": fraction,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "stratify": is_classification and target in dataframe.columns,
        "cross_validation": too_small,
        "random_state": 42,
        "recommendation": (
            "Use cross-validation because this dataset has fewer than 500 rows."
            if too_small else "Use a fixed random state for reproducibility."
        ),
    }


def preprocessing_recommendation(
    dataframe: pd.DataFrame,
    column_types: Sequence[dict[str, Any]],
    distributions: dict[str, Any],
    class_balance: dict[str, Any],
    missing_values: Sequence[dict[str, Any]] | None,
    outliers: Sequence[dict[str, Any]] | None,
    recommended_models: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recommend encoding, scaling, imbalance, and missing-value strategies."""
    type_map = {item["column"]: item for item in column_types}
    missing_map = {
        str(item.get("column")): float(item.get("missing_percentage", 0))
        for item in (missing_values or []) if isinstance(item, dict)
    }
    distribution_map = {
        item["column"]: item for item in distributions.get("numeric", [])
    }
    outlier_columns = {
        str(item.get("column")) for item in (outliers or [])
        if isinstance(item, dict) and int(item.get("outlier_count", 0)) > 0
    }

    encoding = []
    scaling = []
    missing_strategy = []
    for column in dataframe.columns:
        name = str(column)
        info = type_map.get(name, {})
        detected_type = info.get("detected_type")
        if detected_type in {"Categorical", "Binary", "Text"}:
            count = int(info.get("unique_count", 0))
            method = "One-Hot" if count <= 10 else "Ordinal" if count <= 50 else "Target Encoding"
            encoding.append({"column": name, "cardinality": info.get("unique_count", 0), "method": method})
        if detected_type == "Numeric" or info.get("numeric_as_text"):
            distribution = distribution_map.get(name, {})
            skew = distribution.get("skewness")
            if name in outlier_columns or (skew is not None and abs(skew) >= 1):
                method = "Robust Scaling"
            elif dataframe[name].dropna().between(0, 1).all() if pd.api.types.is_numeric_dtype(dataframe[name].dtype) else False:
                method = "Min-Max Scaling"
            else:
                method = "Standardization"
            scaling.append({"columns": [name], "method": method, "skewness": skew})

        missing_pct = missing_map.get(name, 0.0)
        if missing_pct > 0:
            if missing_pct >= 40:
                method = "Drop column or compare with row dropping"
            elif detected_type == "Numeric" or info.get("numeric_as_text"):
                skew = distribution_map.get(name, {}).get("skewness")
                method = "Mean" if skew is not None and abs(skew) < 0.5 else "Median"
            else:
                method = "Mode"
            missing_strategy.append({"column": name, "missing_percentage": missing_pct, "method": method})

    ratio = class_balance.get("imbalance_ratio")
    if ratio is None or not class_balance.get("applicable") or ratio <= 1.5:
        imbalance_method = "None"
    elif ratio <= 3:
        imbalance_method = "Class Weight"
    else:
        imbalance_method = "SMOTE or Class Weight"

    scaling_models = [
        str(item.get("method") or item.get("method_id"))
        for item in (recommended_models or [])
        if item.get("scaling_required")
    ]
    return {
        "summary": (
            f"{len(encoding)} categorical column(s) need encoding; "
            f"{len(scaling)} numeric group(s) have scaling guidance."
        ),
        "status": "Warning" if missing_strategy or imbalance_method != "None" else "OK",
        "encoding": encoding,
        "scaling": scaling,
        "scaling_models": scaling_models,
        "imbalance": imbalance_method,
        "missing_strategy": missing_strategy,
        "fit_note": "Fit preprocessing on the training set only to avoid data leakage.",
        "recommendation": "Recommendations are analytical suggestions, not decisions.",
    }


def feature_selection_hints(
    dataframe: pd.DataFrame,
    target: str | None,
    task: str | None,
    column_types: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Find constant/near-constant and correlated columns plus quick importance."""
    source = dataframe.copy(deep=True)
    normalized = normalize_missing(source)
    target_exists = bool(target and target in source.columns)
    feature_columns = [column for column in source.columns if column != target]
    constant_columns = [
        str(column) for column in feature_columns
        if normalized[column].nunique(dropna=False) <= 1
    ]
    near_constant_columns = []
    for column in feature_columns:
        values = normalized[column].dropna()
        if len(values) < 2:
            continue
        if pd.api.types.is_numeric_dtype(values.dtype):
            variance = float(values.var())
            if np.isfinite(variance) and variance <= 1e-8:
                near_constant_columns.append(str(column))

    numeric = source.select_dtypes(include=["number"]).drop(
        columns=[target] if target in source.columns else [], errors="ignore"
    )
    correlations = []
    if numeric.shape[1] >= 2:
        matrix = numeric.corr().abs().fillna(0)
        names = list(matrix.columns)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                value = float(matrix.loc[left, right])
                if value > 0.9:
                    correlations.append({"left": str(left), "right": str(right), "correlation": value})
    correlations.sort(key=lambda item: item["correlation"], reverse=True)

    importance = []
    importance_note = "Target not detected"
    if target_exists and task in {"binary_classification", "multiclass_classification", "regression"}:
        working = normalized.dropna(subset=[target]).copy()
        if len(working) > IMPORTANCE_SAMPLE_SIZE:
            working = working.sample(IMPORTANCE_SAMPLE_SIZE, random_state=42)
        y = working[target]
        features = working.drop(columns=[target])
        usable = [
            column for column in features.columns
            if column_types and next(
                (item.get("detected_type") != "ID-like" and item.get("detected_type") != "Constant"
                 for item in column_types if item.get("column") == str(column)),
                True,
            )
        ]
        features = features.loc[:, usable]
        if len(working) >= 4 and not features.empty and y.nunique(dropna=True) >= 2:
            encoded = pd.get_dummies(features, dummy_na=True)
            encoded = encoded.replace([np.inf, -np.inf], np.nan)
            encoded = encoded.select_dtypes(include=["number", "bool"]).fillna(0)
            encoded = encoded.loc[:, list(encoded.columns[:80])]
            try:
                if not encoded.empty:
                    if task == "regression":
                        target_values = pd.to_numeric(y, errors="coerce")
                        valid = target_values.notna()
                        encoded = encoded.loc[valid]
                        target_values = target_values.loc[valid]
                        estimator = RandomForestRegressor(
                            n_estimators=40, max_depth=8, min_samples_leaf=2,
                            random_state=42, n_jobs=1,
                        )
                    else:
                        target_values = y.astype(str).to_numpy()
                        estimator = RandomForestClassifier(
                            n_estimators=40, max_depth=8, min_samples_leaf=2,
                            random_state=42, n_jobs=1,
                        )
                    if len(target_values) >= 4 and pd.Series(target_values).nunique() >= 2:
                        estimator.fit(encoded, target_values)
                        ranked = sorted(
                            zip(encoded.columns, estimator.feature_importances_),
                            key=lambda pair: float(pair[1]),
                            reverse=True,
                        )[:10]
                        importance = [
                            {"feature": str(name), "importance": float(value)}
                            for name, value in ranked
                        ]
                        importance_note = (
                            f"Quick Random Forest importance computed on a sample of {len(working):,} rows"
                            if len(source) > IMPORTANCE_SAMPLE_SIZE
                            else "Quick Random Forest importance"
                        )
            except (TypeError, ValueError, RuntimeError):
                importance_note = "Quick Feature Importance not available for this dataset"

    return {
        "summary": (
            f"{len(constant_columns)} constant, {len(near_constant_columns)} near-constant, "
            f"and {len(correlations)} highly correlated feature pair(s)."
        ),
        "status": "Warning" if constant_columns or near_constant_columns or correlations else "OK",
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
        "correlated_pairs": correlations,
        "importance": importance,
        "importance_note": importance_note,
        "recommendation": "Review redundant and ID-like features before model fitting.",
    }


def build_pipeline_checklist(
    dataframe: pd.DataFrame,
    target: str | None,
    task: str | None,
    distributions: dict[str, Any],
    quality_summary: dict[str, Any],
    split: dict[str, Any],
    preprocessing: dict[str, Any],
    feature_hints: dict[str, Any],
    recommended_models: Sequence[dict[str, Any]] | None = None,
    ml_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build data-driven statuses for the 15 ML workflow stages."""
    has_data = not dataframe.empty and len(dataframe.columns) > 0
    has_target = bool(target and target in dataframe.columns and task)
    evaluation = (ml_result or {}).get("evaluation", {})
    has_evaluation = bool(evaluation.get("results"))
    recommended = bool(recommended_models)
    stages = [
        ("Problem Definition", "Done" if has_target else "Partial", "Target, task, and metric recommendation."),
        ("Data Loading", "Done" if has_data else "Not Available Yet", "Rows, columns, and schema analysis."),
        ("EDA", "Done" if has_data and distributions else "Not Available Yet", "Distribution and class summaries."),
        ("Data Cleaning", "Done" if has_data and quality_summary else "Not Available Yet", "Existing missing, duplicate, outlier, and quality outputs."),
        ("Train/Test Split", "Recommended" if split.get("test_size") is not None else "Not Available Yet", "Suggested split; no data is split here."),
        ("Preprocessing", "Recommended" if has_data and preprocessing else "Not Available Yet", "Encoding, scaling, missing, and imbalance guidance."),
        ("Feature Engineering and Feature Selection", "Partial" if has_data and feature_hints else "Not Available Yet", "Feature review hints; no transformation is applied."),
        ("Model Selection", "Done" if recommended else "Recommended" if task else "Not Available Yet", "Existing ML Intelligence recommendations."),
        ("Model Training", "Done" if has_evaluation else "Recommended", "Available in ML Intelligence; no model is trained by this analysis."),
        ("Evaluation and Cross-Validation", "Done" if has_evaluation else "Recommended", "Available in ML Intelligence; results appear after evaluation."),
        ("Hyperparameter Tuning", "Not Available Yet", "No tuning workflow is available."),
        ("Final Test", "Not Available Yet", "No independent post-selection final test is available."),
        ("Interpretation", "Partial" if has_evaluation else "Recommended", "Model interpretation is limited to current method metadata."),
        ("Model Saving and Deployment", "Not Available Yet", "No model artifact or deployment workflow is available."),
        ("Monitoring", "Not Available Yet", "No post-deployment monitoring workflow is available."),
    ]
    return [
        {"stage": name, "status": status, "details": details}
        for name, status, details in stages
    ]


def analyze_ml_readiness(
    dataframe: pd.DataFrame,
    *,
    missing_values: Sequence[dict[str, Any]] | None = None,
    duplicates: dict[str, Any] | None = None,
    outliers: Sequence[dict[str, Any]] | None = None,
    quality_issues: Sequence[Any] | None = None,
    fingerprint: dict[str, Any] | None = None,
    target: str | None = None,
    task: str | None = None,
    recommended_models: Sequence[dict[str, Any]] | None = None,
    ml_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute structured, read-only pre-modeling diagnostics for one dataset."""
    if dataframe is None:
        dataframe = pd.DataFrame()
    original = dataframe.copy(deep=True)
    sampled = len(original) > SAMPLE_THRESHOLD
    analysis_df = (
        original.sample(ANALYSIS_SAMPLE_SIZE, random_state=42)
        if sampled else original.copy(deep=True)
    )
    target, task, candidates = _selected_target(analysis_df, target, task)
    column_types = detect_column_types(analysis_df)
    distributions = distribution_analysis(analysis_df, column_types)
    distributions["sampled"] = sampled
    distributions["sample_count"] = len(analysis_df)
    distributions["sample_note"] = (
        f"Computed on a sample of {len(analysis_df):,} rows" if sampled else ""
    )
    class_balance = class_balance_analysis(analysis_df, target, task)
    quality = data_quality_summary(
        original,
        missing_values,
        duplicates,
        outliers,
        quality_issues,
        column_types,
    )
    if sampled:
        quality["sample_note"] = f"Computed on a sample of {len(analysis_df):,} rows"
    leakage = data_leakage_check(analysis_df, target, column_types)
    split = train_test_split_recommendation(original, target, task)

    if recommended_models is None and task:
        detector_task = {
            "primary_task": {"task": task, "target": target},
            "dataset": {
                "rows": len(original),
                "columns": len(original.columns),
                "numeric_features": len(original.select_dtypes(include="number").columns),
            },
        }
        recommended_models = MethodRecommender().recommend(detector_task).get("recommendations", [])
    recommended_models = list(recommended_models or [])

    preprocessing = preprocessing_recommendation(
        analysis_df,
        column_types,
        distributions,
        class_balance,
        missing_values,
        outliers,
        recommended_models,
    )
    feature_hints = feature_selection_hints(analysis_df, target, task, column_types)
    checklist = build_pipeline_checklist(
        original,
        target,
        task,
        distributions,
        quality,
        split,
        preprocessing,
        feature_hints,
        recommended_models,
        ml_result,
    )
    if sampled:
        summary_note = f"Computed on a sample of {len(analysis_df):,} rows"
    else:
        summary_note = "Recommendations are analytical suggestions, not decisions."

    return {
        "target": target,
        "task": task,
        "target_candidates": candidates,
        "metric_recommendation": (
            ["Accuracy", "Precision", "Recall", "F1"]
            if task and "classification" in task
            else ["MAE", "RMSE", "R2"] if task == "regression" else []
        ),
            "problem_definition": {
                "summary": (
                    f"{task.replace('_', ' ').title()} target '{target}' detected."
                    if target and task
                    else "Target not detected"
                ),
                "status": "OK" if target and task else "Warning",
                "target": target,
                "task": task,
                "metrics": (
                    ["Accuracy", "Precision", "Recall", "F1"]
                    if task and "classification" in task
                    else ["MAE", "RMSE", "R2"] if task == "regression" else []
                ),
                "target_score": next(
                    (
                        candidate.get("score")
                        for candidate in candidates
                        if candidate.get("name") == target
                    ),
                    None,
                ),
            },
        "summary_note": summary_note,
        "data_types": {
            "summary": f"Reviewed {len(column_types)} columns across {len(original):,} rows.",
            "status": "OK" if column_types else "Issue",
            "columns": column_types,
            "sample_note": distributions["sample_note"],
        },
        "distributions": distributions,
        "class_balance": class_balance,
        "quality_summary": quality,
        "leakage": leakage,
        "split": split,
        "preprocessing": preprocessing,
        "feature_selection": feature_hints,
        "pipeline_checklist": checklist,
    }