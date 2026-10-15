"""Dataset profiling, automated ML task detection, dynamic graph template generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Profile tabular dataset: shapes, column types, missing rates, target cardinality, outliers."""
    row_count = len(df)
    col_count = len(df.columns)

    column_types: Dict[str, str] = {}
    missing_percent: Dict[str, float] = {}
    total_missing = int(df.isna().sum().sum())

    datetime_cols: List[str] = []
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        s = df[col]
        missing_percent[col] = round(float(s.isna().mean() * 100), 2)

        # Detect datetime
        if pd.api.types.is_datetime64_any_dtype(s):
            column_types[col] = "datetime"
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            # Check if high unique ID-like sequential integer
            if s.nunique() == row_count and s.is_monotonic_increasing:
                column_types[col] = "id_like"
            else:
                column_types[col] = "numeric"
                numeric_cols.append(col)
        else:
            # Check if parseable as datetime
            if s.dropna().astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8:
                column_types[col] = "datetime"
                datetime_cols.append(col)
            else:
                column_types[col] = "categorical"
                categorical_cols.append(col)

    # Outlier detection (IQR method on numeric columns)
    outlier_count = 0
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 10:
            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                outliers = ((s < (q25 - 1.5 * iqr)) | (s > (q75 + 1.5 * iqr))).sum()
                outlier_count += int(outliers)

    target_info: Dict[str, Any] = {}
    if target_column and target_column in df.columns:
        t_series = df[target_column].dropna()
        n_unique = t_series.nunique()
        counts = t_series.value_counts()
        is_num = pd.api.types.is_numeric_dtype(t_series)

        if n_unique == 2:
            target_type = "binary"
            ratio = float(counts.min() / counts.max()) if counts.max() > 0 else 1.0
        elif n_unique < 20 or not is_num:
            target_type = "multiclass"
            ratio = float(counts.min() / counts.max()) if counts.max() > 0 else 1.0
        else:
            target_type = "continuous"
            ratio = 1.0

        target_info = {
            "target_column": target_column,
            "unique_count": n_unique,
            "target_type": target_type,
            "is_imbalanced": (target_type in ("binary", "multiclass")) and (ratio < 0.35),
            "imbalance_ratio": round(ratio, 3),
            "class_counts": counts.head(10).to_dict(),
        }

    return {
        "row_count": row_count,
        "col_count": col_count,
        "column_types": column_types,
        "missing_percent": missing_percent,
        "total_missing": total_missing,
        "has_missing_values": total_missing > 0,
        "outlier_count": outlier_count,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "target_info": target_info,
    }


def detect_task(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Automatically deduce ML task type from dataset profile and target column."""
    profile = profile_dataset(df, target_column)

    if not target_column:
        return {
            "task_type": "clustering",
            "target_column": None,
            "confidence": 0.90,
            "reason": "Tidak ada target yang ditentukan; task otomatis diarahkan ke Unsupervised Clustering.",
            "profile": profile,
        }

    target_info = profile["target_info"]
    has_datetime = len(profile["datetime_cols"]) > 0

    if has_datetime and target_info.get("target_type") == "continuous":
        return {
            "task_type": "time_series",
            "target_column": target_column,
            "confidence": 0.85,
            "reason": f"Kolom datetime ditemukan ({profile['datetime_cols'][0]}) dengan target numerik kontinu '{target_column}'.",
            "profile": profile,
        }

    if target_info.get("target_type") == "binary":
        return {
            "task_type": "binary_classification",
            "target_column": target_column,
            "confidence": 0.98,
            "reason": f"Target '{target_column}' memiliki 2 nilai unik (Klasifikasi Biner).",
            "profile": profile,
        }

    if target_info.get("target_type") == "multiclass":
        return {
            "task_type": "multiclass_classification",
            "target_column": target_column,
            "confidence": 0.95,
            "reason": f"Target '{target_column}' memiliki {target_info['unique_count']} kelas unik (Klasifikasi Multikelas).",
            "profile": profile,
        }

    return {
        "task_type": "regression",
        "target_column": target_column,
        "confidence": 0.92,
        "reason": f"Target '{target_column}' merupakan nilai numerik kontinu dengan kardinalitas tinggi (Regresi).",
        "profile": profile,
    }


def generate_template_graphs(df: pd.DataFrame, target_column: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate 3 to 5 candidate ready-to-run template graphs customized dynamically
    to dataset characteristics (missing values, class imbalance, feature counts).
    """
    detection = detect_task(df, target_column)
    task_type = detection["task_type"]
    profile = detection["profile"]
    has_missing = profile["has_missing_values"]
    is_imbalanced = profile.get("target_info", {}).get("is_imbalanced", False)
    many_features = len(profile["numeric_cols"]) > 30

    templates: List[Dict[str, Any]] = []

    if task_type in ("binary_classification", "multiclass_classification"):
        # Template 1: Fast Baseline
        templates.append({
            "id": "template_classification_baseline",
            "name": "Baseline Cepat (Decision Tree)",
            "tag": "baseline",
            "description": "Pipeline pohon keputusan ringan untuk evaluasi cepat tanpa normalisasi skala.",
            "task_type": task_type,
            "graph": {
                "nodes": [
                    {"id": "load", "spec_id": "load_dataset", "position": (50, 100)},
                    {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}, "position": (250, 100)},
                    {"id": "model", "spec_id": "model:decision_tree_classifier", "parameters": {"max_depth": 8, "random_state": 42}, "position": (250, 250)},
                    {"id": "train", "spec_id": "train", "parameters": {"target": target_column, "task": "classification"}, "position": (450, 100)},
                    {"id": "eval", "spec_id": "evaluate", "parameters": {"target": target_column, "task": "classification"}, "position": (650, 100)},
                ],
                "connections": [
                    ["load", "table", "split", "table"],
                    ["split", "train", "train", "train"],
                    ["model", "estimator", "train", "estimator"],
                    ["train", "model", "eval", "model"],
                    ["split", "test", "eval", "test"],
                ],
            },
        })

        # Template 2: Standard Scaled Random Forest
        prep_params = {"scaler": "standard", "target": target_column}
        if has_missing:
            prep_params["imputer"] = "median"

        templates.append({
            "id": "template_classification_standard",
            "name": "Standar Robust (Scaled Random Forest)",
            "tag": "recommended",
            "description": "Pipeline preprocessing lengkap dengan StandardScaler dan Random Forest.",
            "task_type": task_type,
            "graph": {
                "nodes": [
                    {"id": "load", "spec_id": "load_dataset", "position": (50, 100)},
                    {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}, "position": (250, 100)},
                    {"id": "prep", "spec_id": "preprocess", "parameters": prep_params, "position": (450, 100)},
                    {"id": "model", "spec_id": "model:random_forest_classifier", "parameters": {"n_estimators": 100, "random_state": 42}, "position": (450, 250)},
                    {"id": "train", "spec_id": "train", "parameters": {"target": target_column, "task": "classification"}, "position": (650, 100)},
                    {"id": "eval", "spec_id": "evaluate", "parameters": {"target": target_column, "task": "classification"}, "position": (850, 100)},
                ],
                "connections": [
                    ["load", "table", "split", "table"],
                    ["split", "train", "prep", "train"],
                    ["split", "test", "prep", "test"],
                    ["prep", "train", "train", "train"],
                    ["model", "estimator", "train", "estimator"],
                    ["train", "model", "eval", "model"],
                    ["prep", "test", "eval", "test"],
                ],
            },
        })

        # Template 3: Advanced Ensemble / Imbalanced-Aware
        if is_imbalanced:
            templates.append({
                "id": "template_classification_imbalanced",
                "name": "Imbalanced-Aware (SMOTE + Gradient Boosting)",
                "tag": "advanced",
                "description": "Penanganan data tidak seimbang dengan oversampling SMOTE dan Gradient Boosting.",
                "task_type": task_type,
                "graph": {
                    "nodes": [
                        {"id": "load", "spec_id": "load_dataset", "position": (50, 100)},
                        {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}, "position": (250, 100)},
                        {"id": "smote", "spec_id": "smote", "parameters": {"target": target_column, "random_state": 42}, "position": (450, 100)},
                        {"id": "model", "spec_id": "model:gradient_boosting_classifier", "parameters": {"n_estimators": 100, "random_state": 42}, "position": (450, 250)},
                        {"id": "train", "spec_id": "train", "parameters": {"target": target_column, "task": "classification"}, "position": (650, 100)},
                        {"id": "eval", "spec_id": "evaluate", "parameters": {"target": target_column, "task": "classification"}, "position": (850, 100)},
                    ],
                    "connections": [
                        ["load", "table", "split", "table"],
                        ["split", "train", "smote", "train"],
                        ["smote", "train", "train", "train"],
                        ["model", "estimator", "train", "estimator"],
                        ["train", "model", "eval", "model"],
                        ["split", "test", "eval", "test"],
                    ],
                },
            })
        else:
            templates.append({
                "id": "template_classification_gradient_boosting",
                "name": "Model Lanjut (Hist Gradient Boosting)",
                "tag": "advanced",
                "description": "Model gradient boosting cepat untuk akurasi tinggi pada dataset tabular.",
                "task_type": task_type,
                "graph": {
                    "nodes": [
                        {"id": "load", "spec_id": "load_dataset", "position": (50, 100)},
                        {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}, "position": (250, 100)},
                        {"id": "prep", "spec_id": "preprocess", "parameters": {"scaler": "robust", "target": target_column}, "position": (450, 100)},
                        {"id": "model", "spec_id": "model:hist_gradient_boosting_classifier", "parameters": {"max_iter": 100, "random_state": 42}, "position": (450, 250)},
                        {"id": "train", "spec_id": "train", "parameters": {"target": target_column, "task": "classification"}, "position": (650, 100)},
                        {"id": "eval", "spec_id": "evaluate", "parameters": {"target": target_column, "task": "classification"}, "position": (850, 100)},
                    ],
                    "connections": [
                        ["load", "table", "split", "table"],
                        ["split", "train", "prep", "train"],
                        ["split", "test", "prep", "test"],
                        ["prep", "train", "train", "train"],
                        ["model", "estimator", "train", "estimator"],
                        ["train", "model", "eval", "model"],
                        ["prep", "test", "eval", "test"],
                    ],
                },
            })

    elif task_type == "regression":
        # Regression Template 1: Ridge Baseline
        templates.append({
            "id": "template_regression_baseline",
            "name": "Baseline Cepat (Ridge Regression)",
            "tag": "baseline",
            "description": "Model regresi linier teratur dengan StandardScaler.",
            "task_type": "regression",
            "graph": {
                "nodes": [
                    {"id": "load", "spec_id": "load_dataset", "position": (50, 100)},
                    {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}, "position": (250, 100)},
                    {"id": "prep", "spec_id": "preprocess", "parameters": {"scaler": "standard", "target": target_column}, "position": (450, 100)},
                    {"id": "model", "spec_id": "model:ridge", "parameters": {"alpha": 1.0}, "position": (450, 250)},
                    {"id": "train", "spec_id": "train", "parameters": {"target": target_column, "task": "regression"}, "position": (650, 100)},
                    {"id": "eval", "spec_id": "evaluate", "parameters": {"target": target_column, "task": "regression"}, "position": (850, 100)},
                ],
                "connections": [
                    ["load", "table", "split", "table"],
                    ["split", "train", "prep", "train"],
                    ["split", "test", "prep", "test"],
                    ["prep", "train", "train", "train"],
                    ["model", "estimator", "train", "estimator"],
                    ["train", "model", "eval", "model"],
                    ["prep", "test", "eval", "test"],
                ],
            },
        })

        # Regression Template 2: Random Forest Regressor
        templates.append({
            "id": "template_regression_rf",
            "name": "Standar Robust (Random Forest Regressor)",
            "tag": "recommended",
            "description": "Non-linear ensemble regressor dengan 100 decision trees.",
            "task_type": "regression",
            "graph": {
                "nodes": [
                    {"id": "load", "spec_id": "load_dataset", "position": (50, 100)},
                    {"id": "split", "spec_id": "split", "parameters": {"test_size": 0.25, "random_state": 42}, "position": (250, 100)},
                    {"id": "model", "spec_id": "model:random_forest_regressor", "parameters": {"n_estimators": 100, "random_state": 42}, "position": (250, 250)},
                    {"id": "train", "spec_id": "train", "parameters": {"target": target_column, "task": "regression"}, "position": (450, 100)},
                    {"id": "eval", "spec_id": "evaluate", "parameters": {"target": target_column, "task": "regression"}, "position": (650, 100)},
                ],
                "connections": [
                    ["load", "table", "split", "table"],
                    ["split", "train", "train", "train"],
                    ["model", "estimator", "train", "estimator"],
                    ["train", "model", "eval", "model"],
                    ["split", "test", "eval", "test"],
                ],
            },
        })

    else:
        # Clustering Templates (No target required)
        templates.append({
            "id": "template_clustering_kmeans",
            "name": "K-Means Clustering",
            "tag": "recommended",
            "description": "StandardScaler diikuti K-Means untuk pengelompokan data tanpa label.",
            "task_type": "clustering",
            "graph": {
                "nodes": [
                    {"id": "load", "spec_id": "load_dataset", "position": (50, 100)},
                    {"id": "model", "spec_id": "model:kmeans", "parameters": {"n_clusters": 3, "random_state": 42}, "position": (250, 250)},
                    {"id": "train", "spec_id": "train", "parameters": {"task": "clustering"}, "position": (450, 100)},
                    {"id": "eval", "spec_id": "evaluate", "parameters": {"task": "clustering"}, "position": (650, 100)},
                ],
                "connections": [
                    ["load", "table", "train", "train"],
                    ["model", "estimator", "train", "estimator"],
                    ["train", "model", "eval", "model"],
                    ["load", "table", "eval", "test"],
                ],
            },
        })

    return templates


# ---------------------------------------------------------------------
# CUSTOM TEMPLATES PERSISTENCE
# ---------------------------------------------------------------------

_CUSTOM_TEMPLATES_FILE = Path("data/ml_graph_custom_templates.json")


def save_custom_template(name: str, graph_def: Dict[str, Any], description: str = "") -> Dict[str, Any]:
    """Save user-defined graph as a reusable custom template."""
    _CUSTOM_TEMPLATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = list_custom_templates()

    template_id = f"custom_{name.lower().replace(' ', '_')}"
    new_item = {
        "id": template_id,
        "name": name,
        "description": description,
        "tag": "custom",
        "graph": graph_def,
    }
    # Update if already exists, else append
    existing = [t for t in existing if t["id"] != template_id]
    existing.append(new_item)

    _CUSTOM_TEMPLATES_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return new_item


def list_custom_templates() -> List[Dict[str, Any]]:
    """Load all saved custom templates."""
    if not _CUSTOM_TEMPLATES_FILE.exists():
        return []
    try:
        return json.loads(_CUSTOM_TEMPLATES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def delete_custom_template(template_id: str) -> bool:
    """Delete a custom template by ID."""
    existing = list_custom_templates()
    filtered = [t for t in existing if t["id"] != template_id]
    if len(filtered) < len(existing):
        _CUSTOM_TEMPLATES_FILE.write_text(json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    return False
