from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from app.ml.evaluation import MLEvaluator
from app.ml.method_info import get_method_info
from app.ml.method_recommender import MethodRecommender
from app.ml.task_detector import MLTaskDetector


class MLIntelligenceEngine:
    """Dataset-driven ML Intelligence.

    The recommendation shown to the user is produced only after real model
    training/evaluation on the current dataset. The legacy method recommender
    is used only for human-readable method descriptions, not for ranking.

    Pipeline:
        DataFrame + fingerprint
            -> dataframe-aware task/target detection
            -> K-Fold model training/validation
            -> empirical metric ranking
            -> recommended methods
    """

    def __init__(
        self,
        evaluator: Optional[MLEvaluator] = None,
        recommender: Optional[MethodRecommender] = None,
        task_detector: Optional[MLTaskDetector] = None,
    ) -> None:
        self.task_detector = task_detector or MLTaskDetector()
        self.recommender = recommender or MethodRecommender()
        self.evaluator = evaluator or MLEvaluator()

    def analyze(
        self,
        dataframe: pd.DataFrame,
        fingerprint: Dict[str, Any],
        evaluate_models: bool = True,
        task_override: Optional[str] = None,
        target_override: Optional[str] = None,
        time_column: Optional[str] = None,
        requested_methods: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if dataframe is None or dataframe.empty:
            return self._error("Dataset kosong.")

        # IMPORTANT:
        # The task detector must see the real DataFrame, not only the
        # fingerprint. This lets it inspect actual dtypes, cardinality,
        # target-like columns, and missing values.
        try:
            task_result = self.task_detector.detect(
                fingerprint,
                dataframe=dataframe,
            )
        except TypeError:
            # Compatibility with older detectors that only accept fingerprint.
            try:
                task_result = self.task_detector.detect(fingerprint)
            except Exception as exc:
                return self._error(
                    f"Gagal melakukan ML task detection: {exc}"
                )
        except Exception as exc:
            return self._error(
                f"Gagal melakukan ML task detection: {exc}"
            )

        primary_task = task_result.get("primary_task", {}) or {}
        task = primary_task.get("task", "unknown_task")
        target = primary_task.get("target")
        if task_override:
            from app.ml.task_detector import MLTaskDetector

            task = task_override
            target = target_override
            primary_task = {
                "task": task,
                "label": MLTaskDetector.TASK_LABELS.get(task, task.replace("_", " ").title()),
                "target": target,
                "score": 100,
                "reasons": ["Task dipilih manual oleh pengguna."],
            }
            task_result = dict(task_result)
            task_result["primary_task"] = primary_task
            task_result["selection_mode"] = "manual"
        elif task == "time_series_forecasting":
            target = target or next(
                (column for column in dataframe.select_dtypes(include=["number"]).columns),
                None,
            )
            time_column = time_column or next(
                (
                    column for column in dataframe.columns
                    if pd.api.types.is_datetime64_any_dtype(dataframe[column])
                    or self.task_detector._is_datetime_series(dataframe[column])
                ),
                None,
            )
            primary_task = dict(primary_task)
            primary_task.update({"target": target, "time_column": time_column})
            task_result = dict(task_result)
            task_result["primary_task"] = primary_task

        dataset_info = self._extract_dataset_info(
            dataframe,
            fingerprint,
        )

        # ---------------------------------------------------------
        # Knowledge base is NOT the ranking mechanism.
        # It only supplies descriptions/strengths/limitations for cards.
        # ---------------------------------------------------------
        try:
            knowledge_result = self.recommender.recommend(
                {
                    "primary_task": primary_task,
                    "dataset": dataset_info,
                }
            )
        except Exception:
            knowledge_result = {
                "status": "KNOWLEDGE_ONLY",
                "recommendations": [],
                "recommendation_count": 0,
            }

        evaluation = {
            "status": "SKIPPED",
            "results": [],
            "result_count": 0,
            "message": "Evaluation tidak dijalankan.",
        }

        supported_tasks = {
            "binary_classification",
            "multiclass_classification",
            "regression",
            "clustering",
            "anomaly_detection",
            "time_series_forecasting",
        }

        # ---------------------------------------------------------
        # REAL TRAINING FIRST
        # ---------------------------------------------------------
        unsupervised_tasks = {"clustering", "anomaly_detection"}

        if evaluate_models and task in supported_tasks:
            is_unsupervised = task in unsupervised_tasks
            can_evaluate = is_unsupervised or (target is not None)

            if can_evaluate:
                try:
                    eval_kwargs: Dict[str, Any] = {
                        "dataframe": dataframe,
                        "task": str(task),
                        "methods": requested_methods,
                    }
                    if not is_unsupervised:
                        eval_kwargs["target"] = str(target)
                    if task == "time_series_forecasting":
                        eval_kwargs["time_column"] = time_column

                    evaluation = self.evaluator.evaluate(**eval_kwargs)
                except Exception as exc:
                    evaluation = {
                        "status": "ERROR",
                        "results": [],
                        "result_count": 0,
                        "message": f"Evaluation gagal: {exc}",
                    }

        # The recommendation list is created FROM empirical evaluation.
        empirical_recommendations = self._build_empirical_recommendation(
            evaluation=evaluation,
            knowledge_result=knowledge_result,
        )

        best = empirical_recommendations[0] if empirical_recommendations else None

        return {
            "status": "ML_INTELLIGENCE",
            "task_detection": task_result,
            "primary_task": primary_task,
            "dataset": dataset_info,

            # UI compatibility: this remains the recommendation payload,
            # but it is now empirically ranked.
            "recommendation": {
                "status": "EMPIRICAL_RECOMMENDATION",
                "recommendations": empirical_recommendations,
                "recommendation_count": len(empirical_recommendations),
                "message": self._recommendation_message(evaluation),
                "selection_basis": (
                    evaluation.get("validation")
                    or "empirical_evaluation"
                ),
            },

            # Keep knowledge output separate so it is never mistaken for
            # the empirical ranking.
            "static_recommendation": knowledge_result,
            "evaluation": evaluation,
            "best_method": (
                {
                    "method_id": best.get("method_id"),
                    "method": best.get("method"),
                    "metrics": best.get("metrics", {}),
                    "validation": best.get("validation"),
                    "folds": best.get("folds"),
                    "status": "EMPIRICAL_BEST",
                    "selection_basis": "empirical_evaluation",
                }
                if best
                else None
            ),

            "summary": self._build_summary(
                primary_task,
                empirical_recommendations,
                evaluation,
                best,
            ),

            "transparency": {
                "task_detection": "ESTIMATION",
                "method_knowledge": "FACT",
                "method_recommendation": "RECOMMENDATION",
                "model_training": (
                    "EMPIRICAL"
                    if evaluation.get("result_count", 0) > 0
                    else "NOT_AVAILABLE"
                ),
                "model_evaluation": (
                    "EMPIRICAL"
                    if evaluation.get("result_count", 0) > 0
                    else "NOT_AVAILABLE"
                ),
                "model_validation": evaluation.get(
                    "validation",
                    "NOT_AVAILABLE",
                ),
                "final_ranking": (
                    "EMPIRICAL"
                    if empirical_recommendations
                    else "NOT_AVAILABLE"
                ),
                "best_method": (
                    "EMPIRICAL" if best else "NOT_AVAILABLE"
                ),
            },
        }

    # =========================================================
    # EMPIRICAL -> UI RECOMMENDATION
    # =========================================================

    def _build_empirical_recommendation(
        self,
        evaluation: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        results = evaluation.get("results", []) or []

        knowledge_map = {
            item.get("method_id"): item
            for item in knowledge_result.get("recommendations", [])
            if isinstance(item, dict)
        }

        output: list[Dict[str, Any]] = []

        for rank, result in enumerate(results, start=1):
            method_id = result.get("method_id")
            if not method_id:
                continue

            info = knowledge_map.get(method_id) or get_method_info(method_id) or {}
            metrics = result.get("metrics", {}) or {}

            if "f1_score" in metrics:
                score_type = "f1_score"
                score = float(metrics.get("f1_score", 0.0))
            elif "r2_score" in metrics:
                score_type = "r2_score"
                score = float(metrics.get("r2_score", 0.0))
            elif "cluster_score" in metrics:
                score_type = "cluster_score"
                score = float(metrics.get("cluster_score", 0.0))
            elif "silhouette_score" in metrics:
                score_type = "silhouette_score"
                score = float(metrics.get("silhouette_score", 0.0))
            elif "score_spread" in metrics:
                score_type = "score_spread"
                score = float(metrics.get("score_spread", 0.0))
            elif "mae" in metrics:
                score_type = "mae"
                score = float(metrics.get("mae", 0.0))
            else:
                score_type = evaluation.get(
                    "ranking_metric",
                    "empirical",
                )
                score = 0.0

            validation = result.get(
                "validation",
                evaluation.get("validation", "empirical"),
            )

            folds = result.get(
                "folds",
                evaluation.get("folds"),
            )

            reasons = [
                "Model benar-benar dilatih pada dataset ini.",
                f"Metric utama: {self._format_metric_name(score_type)}.",
                f"Validasi: {self._format_validation(validation, folds)}.",
            ]

            if result.get("stability") is not None:
                reasons.append(
                    "Variasi metric antar fold: "
                    f"{result.get('stability')}."
                )

            # Build knowledge_base nested dict from knowledge info
            knowledge_base = {
                "description": info.get("description", ""),
                "strengths": info.get("strengths", []),
                "limitations": info.get("limitations", []),
                "preprocessing": info.get("preprocessing", []),
                "interpretability": info.get(
                    "interpretability",
                    "unknown",
                ),
                "scaling_required": bool(
                    info.get("scaling_required", False)
                ),
                "nonlinear": bool(
                    info.get("nonlinear", False)
                ),
                "small_data": bool(
                    info.get("small_data", False)
                ),
                "large_data": bool(
                    info.get("large_data", False)
                ),
                "score": info.get("score"),
            }

            output.append(
                {
                    "rank": rank,
                    "method_id": method_id,
                    "method": info.get(
                        "method",
                        result.get("method") or self._pretty_method(method_id),
                    ),
                    "category": info.get(
                        "category",
                        self._category(method_id),
                    ),
                    "score": round(score, 4),
                    "score_type": score_type,
                    "status": "EMPIRICAL",
                    "metrics": metrics,
                    "validation": validation,
                    "folds": folds,
                    "stability": result.get("stability"),
                    "description": info.get("description", ""),
                    "strengths": info.get("strengths", []),
                    "limitations": info.get("limitations", []),
                    "preprocessing": info.get("preprocessing", []),
                    "interpretability": info.get(
                        "interpretability",
                        "unknown",
                    ),
                    "scaling_required": bool(
                        info.get("scaling_required", False)
                    ),
                    "nonlinear": bool(
                        info.get("nonlinear", False)
                    ),
                    "small_data": bool(
                        info.get("small_data", False)
                    ),
                    "large_data": bool(
                        info.get("large_data", False)
                    ),
                    "reasons": reasons,
                    # Knowledge base as nested dict for test compatibility
                    "knowledge_base": knowledge_base,
                    # Preserved only as background knowledge.
                    "knowledge_base_score": info.get("score"),
                    "empirical": True,
                }
            )

        return output

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _extract_dataset_info(
        dataframe: pd.DataFrame,
        fingerprint: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "numeric_features": int(
                dataframe.select_dtypes(include=["number"]).shape[1]
            ),
            "categorical_features": int(
                dataframe.select_dtypes(
                    include=["object", "category", "bool"]
                ).shape[1]
            ),
            "datetime_features": int(
                dataframe.select_dtypes(include=["datetime"]).shape[1]
            ),
            "missing_cells": int(
                dataframe.isna().sum().sum()
            ),
            "memory_usage": int(
                dataframe.memory_usage(deep=True).sum()
            ),
        }

    @staticmethod
    def _recommendation_message(
        evaluation: Dict[str, Any],
    ) -> str:
        if evaluation.get("result_count", 0) <= 0:
            return (
                "Model belum direkomendasikan karena training/validation "
                "belum menghasilkan evaluasi yang valid."
            )

        validation = evaluation.get(
            "validation",
            "empirical_evaluation",
        )
        folds = evaluation.get("folds")

        if folds:
            validation_label = f"{validation} ({folds} fold)"
        else:
            validation_label = str(validation)

        return (
            "Urutan metode dihasilkan setelah model dilatih dan divalidasi "
            f"pada dataset ini menggunakan {validation_label}."
        )

    @staticmethod
    def _format_metric_name(value: str) -> str:
        names = {
            "f1_score": "F1 Score",
            "r2_score": "R²",
            "accuracy": "Accuracy",
            "cluster_score": "Cluster Score",
            "silhouette_score": "Silhouette Score",
            "calinski_harabasz": "Calinski-Harabasz",
            "davies_bouldin": "Davies-Bouldin",
            "score_spread": "Score Spread",
            "anomaly_ratio": "Anomaly Ratio",
            "mae": "Mean Absolute Error (lower is better)",
        }
        return names.get(value, str(value).replace("_", " ").title())

    @staticmethod
    def _format_validation(
        validation: str,
        folds: Optional[int],
    ) -> str:
        if folds:
            return f"{validation} ({folds} fold)"
        return str(validation)

    @staticmethod
    def _pretty_method(method_id: str) -> str:
        return str(method_id or "Unknown").replace(
            "_", " "
        ).title()

    @staticmethod
    def _category(method_id: str) -> str:
        method_id = str(method_id)
        clustering_ids = {
            "kmeans", "minibatch_kmeans", "agglomerative_clustering",
            "dbscan", "hdbscan", "optics", "gaussian_mixture",
            "spectral_clustering", "birch", "mean_shift",
            "affinity_propagation",
        }
        anomaly_ids = {
            "isolation_forest", "local_outlier_factor",
            "one_class_svm", "elliptic_envelope",
        }
        if method_id in clustering_ids:
            return "Clustering"
        if method_id in anomaly_ids:
            return "Anomaly Detection"
        if method_id in {"naive", "seasonal_naive", "arima", "exponential_smoothing", "sarimax"}:
            return "Time-Series Forecasting"
        if "regressor" in method_id or "regression" in method_id:
            return "Regression"
        return "Classification"

    @staticmethod
    def _build_summary(
        primary_task: Dict[str, Any],
        empirical: list[Dict[str, Any]],
        evaluation: Dict[str, Any],
        best: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "task": primary_task.get(
                "label",
                "Unknown Task",
            ),
            "recommended_method": (
                empirical[0]["method_id"]
                if empirical
                else None
            ),
            "recommended_method_score": (
                empirical[0]["score"]
                if empirical
                else None
            ),
            "evaluated_methods": evaluation.get(
                "result_count",
                0,
            ),
            "best_empirical_method": (
                best["method_id"]
                if best
                else None
            ),
            "selection_basis": (
                "empirical_evaluation"
                if empirical
                else "not_available"
            ),
        }

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "message": message,
            "task_detection": {},
            "primary_task": {},
            "dataset": {},
            "recommendation": {},
            "evaluation": {},
            "best_method": None,
            "summary": {},
            "transparency": {},
        }
