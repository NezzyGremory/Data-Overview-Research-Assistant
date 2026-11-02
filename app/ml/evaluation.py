from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.ml.registry import ModelRegistry, CandidateModelSelector


def _numeric_feature_columns(frame: pd.DataFrame) -> list:
    return frame.select_dtypes(include=["number"]).columns.tolist()


def _categorical_feature_columns(frame: pd.DataFrame) -> list:
    return [
        column for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(frame[column])
    ]


class _FoldFeaturePreparation(BaseEstimator, TransformerMixin):
    """Learn feature inclusion and date expansion only from a training fold."""

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature preparation membutuhkan DataFrame.")
        self.date_columns_ = []
        for column in X.columns:
            if MLEvaluator._looks_datetime(X[column]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(X[column], errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    self.date_columns_.append(column)
        prepared = MLEvaluator._prepare_features(X)
        self.feature_columns_ = prepared.columns.tolist()
        if not self.feature_columns_:
            raise ValueError("Tidak ada fitur yang dapat digunakan pada training fold.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature preparation membutuhkan DataFrame.")
        transformed = X.copy()
        for column in self.date_columns_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(transformed[column], errors="coerce")
            transformed[f"{column}_year"] = parsed.dt.year
            transformed[f"{column}_month"] = parsed.dt.month
            transformed[f"{column}_day"] = parsed.dt.day
            transformed = transformed.drop(columns=[column])
        numeric_columns = transformed.select_dtypes(include=["number"]).columns
        if len(numeric_columns):
            transformed.loc[:, numeric_columns] = transformed.loc[:, numeric_columns].replace(
                [np.inf, -np.inf], np.nan
            )
        return transformed.loc[:, self.feature_columns_]


class MLEvaluator:
    """
    Empirical Machine Learning Evaluator.

    Trains, validates, and compares actual ML models on the user's dataset.
    Supported Tasks:
    - Binary Classification
    - Multiclass Classification
    - Regression
    - Clustering (Unsupervised)
    - Anomaly Detection (Unsupervised)

    All evaluations use leakage-safe ColumnTransformer + Pipeline.
    """

    CLASSIFICATION_METHODS = [
        "random_forest_classifier",
        "gradient_boosting_classifier",
        "logistic_regression",
        "hist_gradient_boosting_classifier",
        "decision_tree_classifier",
        "svm_classifier",
        "knn_classifier",
        "extra_trees_classifier",
    ]

    REGRESSION_METHODS = [
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "hist_gradient_boosting_regressor",
        "decision_tree_regressor",
        "extra_trees_regressor",
    ]

    CLUSTERING_METHODS = [
        "kmeans",
        "agglomerative_clustering",
        "dbscan",
        "gaussian_mixture",
        "minibatch_kmeans",
        "birch",
    ]

    ANOMALY_METHODS = [
        "isolation_forest",
        "local_outlier_factor",
        "one_class_svm",
        "elliptic_envelope",
    ]

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        cv: int = 3,
        max_cv_rows: int = 20_000,
        n_jobs: int = 1,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.max_cv_rows = max_cv_rows
        self.n_jobs = n_jobs

    # =========================================================
    # PUBLIC API
    # =========================================================

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        target: Optional[str] = None,
        task: str = "binary_classification",
        methods: Optional[List[str]] = None,
        time_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        if dataframe is None or dataframe.empty:
            return self._error("Dataset kosong.")

        supported = {
            "binary_classification",
            "multiclass_classification",
            "regression",
            "clustering",
            "anomaly_detection",
            "time_series_forecasting",
        }
        if task not in supported:
            return self._error(f"Task '{task}' belum didukung evaluator.")

        if task == "time_series_forecasting":
            return self._evaluate_forecasting(dataframe, target, time_column, methods)

        # -----------------------------------------------------
        # UNSUPERVISED TASKS (No target required)
        # -----------------------------------------------------
        if task == "clustering":
            data = dataframe.copy()
            if target and target in data.columns:
                data = data.drop(columns=[target])
            X = self._prepare_features(data)
            if X.shape[1] < 2:
                return self._error("Clustering membutuhkan minimal 2 fitur numerik.")
            requested = CandidateModelSelector.select_candidates(
                task="clustering",
                n_rows=len(X),
                n_cols=X.shape[1],
                requested_methods=methods,
            )
            if methods and not requested:
                return self._error("Algoritma clustering yang dipilih tidak tersedia.")
            requested = requested or self.CLUSTERING_METHODS
            return self._evaluate_clustering(X, requested)

        if task == "anomaly_detection":
            data = dataframe.copy()
            if target and target in data.columns:
                data = data.drop(columns=[target])
            X = self._prepare_features(data)
            if X.shape[1] < 1:
                return self._error("Deteksi anomali membutuhkan minimal 1 fitur yang dapat digunakan.")
            requested = CandidateModelSelector.select_candidates(
                task="anomaly_detection",
                n_rows=len(X),
                n_cols=X.shape[1],
                requested_methods=methods,
            )
            if methods and not requested:
                return self._error("Algoritma deteksi anomali yang dipilih tidak tersedia.")
            requested = requested or self.ANOMALY_METHODS
            return self._evaluate_anomaly_detection(X, requested)

        # -----------------------------------------------------
        # SUPERVISED TASKS (Target required)
        # -----------------------------------------------------
        if not target or target not in dataframe.columns:
            return self._error(f"Target '{target}' tidak ditemukan dalam dataset.")

        data = dataframe.copy()
        data = data.dropna(subset=[target]).reset_index(drop=True)

        if len(data) < 10:
            return self._error("Dataset terlalu kecil untuk evaluation yang stabil.")

        X = data.drop(columns=[target])
        y = data[target]

        if X.shape[1] == 0:
            return self._error("Tidak terdapat feature yang dapat digunakan.")

        target_copies = [column for column in X.columns if X[column].equals(y)]
        if target_copies:
            X = X.drop(columns=target_copies)
        if X.shape[1] == 0:
            return self._error("Semua fitur identik dengan target; training dihentikan untuk mencegah kebocoran target.")

        if task == "regression":
            if not pd.api.types.is_numeric_dtype(y):
                return self._error("Target regression harus berupa nilai numerik.")
            finite_target = np.isfinite(y.to_numpy(dtype=float))
            if not finite_target.all():
                X = X.iloc[finite_target].reset_index(drop=True)
                y = y.iloc[finite_target].reset_index(drop=True)
            if len(X) < 10:
                return self._error("Target regression memiliki kurang dari 10 nilai numerik finite.")

            requested = CandidateModelSelector.select_candidates(
                task="regression",
                n_rows=len(X),
                n_cols=X.shape[1],
                requested_methods=methods,
            )
            if methods and not requested:
                return self._error("Algoritma regresi yang dipilih tidak tersedia.")
            requested = requested or self.REGRESSION_METHODS
            result = self._evaluate_supervised(X, y, task, requested)
            result["excluded_target_copies"] = [str(column) for column in target_copies]
            if target_copies:
                result["message"] += " Fitur yang identik dengan target dikeluarkan: " + ", ".join(map(str, target_copies)) + "."
            return result

        class_count = int(y.nunique(dropna=True))
        if class_count < 2:
            return self._error("Target classification harus memiliki minimal 2 kelas.")

        if task == "binary_classification" and class_count != 2:
            return self._error("Binary classification membutuhkan tepat 2 kelas.")

        if task == "multiclass_classification" and class_count < 3:
            return self._error("Multiclass classification membutuhkan minimal 3 kelas.")

        requested = CandidateModelSelector.select_candidates(
            task=task,
            n_rows=len(X),
            n_cols=X.shape[1],
            requested_methods=methods,
        )
        if methods and not requested:
            return self._error("Algoritma klasifikasi yang dipilih tidak tersedia.")
        requested = requested or self.CLASSIFICATION_METHODS
        result = self._evaluate_supervised(X, y, task, requested)
        result["excluded_target_copies"] = [str(column) for column in target_copies]
        if target_copies:
            result["message"] += " Fitur yang identik dengan target dikeluarkan: " + ", ".join(map(str, target_copies)) + "."
        return result

    def _evaluate_supervised(
        self, X: pd.DataFrame, y: pd.Series, task: str, methods: List[str]
    ) -> Dict[str, Any]:
        """Rank on development data, then evaluate the winner on an untouched test set."""
        source_rows = len(X)
        if len(X) > self.max_cv_rows:
            if task == "regression":
                positions = np.random.RandomState(self.random_state).choice(
                    len(X), size=self.max_cv_rows, replace=False
                )
                X = X.iloc[positions].reset_index(drop=True)
                y = y.iloc[positions].reset_index(drop=True)
            else:
                X, y = self._stratified_sample(X, y, self.max_cv_rows)
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)

        is_classification = task != "regression"
        holdout_reason = None
        if len(X) < 50:
            holdout_reason = "Minimal 50 baris diperlukan untuk uji akhir terpisah."
        elif is_classification and int(y.value_counts(dropna=False).min()) < 5:
            holdout_reason = "Setiap kelas memerlukan minimal 5 baris untuk uji akhir terpisah."

        X_train, y_train = X, y
        X_test = y_test = None
        if holdout_reason is None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state,
                    stratify=y if is_classification else None,
                )
            except ValueError as exc:
                holdout_reason = f"Pemisahan uji akhir tidak memenuhi syarat: {exc}"

        if is_classification:
            result = self._evaluate_classification(X_train, y_train, task, methods)
        else:
            result = self._evaluate_regression(X_train, y_train, methods)

        result["dataset"] = {
            "rows": source_rows, "evaluated_rows": len(X),
            "selection_rows": len(X_train), "test_rows": len(X_test) if X_test is not None else 0,
            "features": X.shape[1],
        }
        result["selection_basis"] = result.get("validation")
        result["final_test"] = {
            "status": "NOT_AVAILABLE", "reason": holdout_reason or "Tidak ada model yang berhasil dievaluasi."
        }
        if source_rows > len(X):
            result["message"] += f" Sampel acak {len(X):,} dari {source_rows:,} baris digunakan."
        if X_test is None or not result.get("results"):
            result["message"] += f" Uji akhir belum tersedia: {result['final_test']['reason']}"
            return result

        best = result["results"][0]
        method_id = best["method_id"]
        try:
            model = self._get_model_instance(method_id)
            pipe = self._pipeline(X_train, model, method_id)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)
                predictions = pipe.predict(X_test)
            if is_classification:
                metrics = self._classification_metrics(y_test, predictions)
            else:
                metrics = {
                    "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
                    "r2_score": round(float(r2_score(y_test, predictions)), 4),
                }
            result["final_test"] = {
                "status": "EVALUATED", "method_id": method_id, "method": best["method"],
                "metrics": metrics, "validation": "independent_holdout",
                "train_rows": len(X_train), "test_rows": len(X_test),
                "test_size": self.test_size, "random_state": self.random_state,
                "selection_basis": result["selection_basis"],
            }
            result["message"] += " Model teratas diuji sekali pada data akhir yang tidak dipakai memilih model."
        except Exception as exc:
            result["final_test"] = {
                "status": "FAILED", "method_id": method_id,
                "reason": f"Uji akhir gagal: {exc}",
            }
            result["message"] += f" {result['final_test']['reason']}"
        return result

    # =========================================================
    # TIME-SERIES FORECASTING
    # =========================================================
    def _evaluate_forecasting(
        self,
        dataframe: pd.DataFrame,
        target: Optional[str],
        time_column: Optional[str],
        methods: Optional[List[str]],
    ) -> Dict[str, Any]:
        if not target or target not in dataframe.columns:
            return self._error("Pilih kolom target numerik untuk forecasting.")
        if not pd.api.types.is_numeric_dtype(dataframe[target]):
            return self._error("Target forecasting harus berupa nilai numerik.")
        if not time_column or time_column not in dataframe.columns:
            time_column = next(
                (column for column in dataframe.columns if self._looks_datetime(dataframe[column])),
                None,
            )
        if not time_column:
            return self._error("Pilih kolom tanggal atau waktu untuk forecasting.")
        if time_column == target:
            return self._error("Kolom target dan kolom waktu harus berbeda.")

        frame = dataframe[[time_column, target]].copy()
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        frame = frame.dropna().sort_values(time_column, kind="stable")
        if len(frame) < 10:
            return self._error("Forecasting membutuhkan minimal 10 baris bertanggal dan bernilai numerik.")

        holdout = max(1, int(np.ceil(len(frame) * self.test_size)))
        split_at = len(frame) - holdout
        if split_at < 3:
            return self._error("Data training forecasting terlalu sedikit.")
        train = frame[target].iloc[:split_at].to_numpy(dtype=float)
        actual = frame[target].iloc[split_at:].to_numpy(dtype=float)
        requested = methods or ["naive", "seasonal_naive"]
        labels = {
            "naive": "Naive Forecast",
            "seasonal_naive": "Seasonal Naive",
            "arima": "ARIMA",
            "exponential_smoothing": "Exponential Smoothing",
            "sarimax": "SARIMAX",
        }
        results = []
        failed = []
        for method in requested:
            if method not in labels:
                failed.append({"method_id": method, "error": "Metode forecasting tidak dikenal."})
                continue
            try:
                if method == "naive":
                    predictions = np.full(len(actual), train[-1], dtype=float)
                elif method == "seasonal_naive":
                    period = min(12, len(train))
                    predictions = np.asarray(
                        [train[-period + index % period] for index in range(len(actual))],
                        dtype=float,
                    )
                elif method == "arima":
                    from statsmodels.tsa.arima.model import ARIMA
                    predictions = np.asarray(ARIMA(train, order=(1, 1, 1)).fit().forecast(len(actual)))
                elif method == "exponential_smoothing":
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    predictions = np.asarray(ExponentialSmoothing(train).fit().forecast(len(actual)))
                else:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    predictions = np.asarray(
                    SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 0, max(2, min(12, len(train) // 2)))).fit(disp=False).forecast(len(actual))
                    )
                error = actual - predictions
                nonzero = np.abs(actual) > np.finfo(float).eps
                metrics = {
                    "mae": round(float(np.mean(np.abs(error))), 4),
                    "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 4),
                    "mape": round(float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100), 2)
                    if nonzero.any() else None,
                }
                results.append({
                    "method_id": method,
                    "method": labels[method],
                    "status": "EVALUATED",
                    "metrics": metrics,
                    "validation": "chronological_holdout",
                    "folds": None,
                })
            except Exception as exc:
                failed.append({"method_id": method, "error": str(exc)})

        results.sort(key=lambda item: (item["metrics"]["mae"], item["metrics"]["rmse"]))
        return {
            "status": "EVALUATION" if results else "FAILED",
            "task": "time_series_forecasting",
            "target": target,
            "time_column": time_column,
            "dataset": {"rows": len(frame), "features": 1},
            "validation": "chronological_holdout",
            "folds": None,
            "metrics_used": ["mae", "rmse", "mape"],
            "ranking_metric": "mae",
            "results": results,
            "failed_methods": failed,
            "result_count": len(results),
            "message": "Model forecasting diuji pada bagian waktu terbaru tanpa mengacak urutan data.",
        }


    # =========================================================
    # CLASSIFICATION
    # =========================================================

    def _evaluate_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
        methods: List[str],
    ) -> Dict[str, Any]:
        data_note = ""

        if len(X) > self.max_cv_rows:
            X, y = self._stratified_sample(X, y, self.max_cv_rows)
            data_note = f" Evaluasi menggunakan sampel {len(X):,} baris karena dataset sangat besar."

        min_class = int(y.value_counts(dropna=False).min())
        folds = min(self.cv, min_class)
        use_cv = folds >= 2

        splitter = (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            if use_cv
            else None
        )

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
        }

        results: List[Dict[str, Any]] = []

        for method_id in methods:
            model = self._get_model_instance(method_id)
            if model is None:
                continue

            spec = ModelRegistry.get_spec(method_id)
            method_name = spec.name if spec else self._pretty_name(method_id)

            try:
                pipe = self._pipeline(X, model, method_id)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if use_cv:
                        scores = cross_validate(
                            pipe,
                            X,
                            y,
                            cv=splitter,
                            scoring=scoring,
                            n_jobs=self.n_jobs,
                            error_score="raise",
                        )

                        metrics = {
                            "accuracy": self._mean_pct(scores["test_accuracy"]),
                            "precision": self._mean_pct(scores["test_precision"]),
                            "recall": self._mean_pct(scores["test_recall"]),
                            "f1_score": self._mean_pct(scores["test_f1"]),
                        }
                        stability_values = scores["test_f1"] * 100.0
                        cv_std = round(float(np.std(stability_values)), 2)
                        cv_mean = metrics["f1_score"]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=self.test_size,
                            random_state=self.random_state,
                            stratify=y if min_class >= 2 else None,
                        )
                        pipe.fit(X_train, y_train)
                        predictions = pipe.predict(X_test)
                        metrics = self._classification_metrics(y_test, predictions)
                        stability_values = np.array([metrics["f1_score"]], dtype=float)
                        cv_std = 0.0
                        cv_mean = metrics["f1_score"]

                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "EVALUATED",
                    "metrics": metrics,
                    "validation": "stratified_k_fold" if use_cv else "holdout",
                    "folds": folds if use_cv else 1,
                    "stability": cv_std,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                })

            except Exception as exc:
                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "FAILED",
                    "metrics": {},
                    "error": str(exc),
                })

        evaluated = [r for r in results if r.get("status") == "EVALUATED"]
        evaluated.sort(
            key=lambda r: (r["metrics"]["f1_score"], r["metrics"]["accuracy"]),
            reverse=True,
        )

        failed = [
            {"method_id": r.get("method_id"), "error": r.get("error", "Unknown error")}
            for r in results
            if r.get("status") == "FAILED"
        ]

        return {
            "status": "EVALUATION" if evaluated else "FAILED",
            "task": task,
            "target": str(y.name),
            "dataset": {"rows": len(X), "features": X.shape[1]},
            "validation": "stratified_k_fold" if use_cv else "holdout",
            "folds": folds if use_cv else 1,
            "metrics_used": ["accuracy", "precision", "recall", "f1_score"],
            "ranking_metric": "f1_score",
            "results": evaluated,
            "failed_methods": failed,
            "result_count": len(evaluated),
            "message": (
                "Model benar-benar dilatih dan dibandingkan menggunakan "
                f"{'Stratified K-Fold Cross Validation' if use_cv else 'holdout evaluation'}."
                f"{data_note}"
            ),
        }

    # =========================================================
    # REGRESSION
    # =========================================================

    def _evaluate_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str],
    ) -> Dict[str, Any]:
        data_note = ""

        if len(X) > self.max_cv_rows:
            X = X.sample(self.max_cv_rows, random_state=self.random_state)
            y = y.loc[X.index]
            data_note = f" Evaluasi menggunakan sampel {len(X):,} baris karena dataset sangat besar."

        folds = min(self.cv, len(X))
        use_cv = folds >= 2

        splitter = (
            KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
            if use_cv
            else None
        )

        scoring = {
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        }

        results: List[Dict[str, Any]] = []

        for method_id in methods:
            model = self._get_model_instance(method_id)
            if model is None:
                continue

            spec = ModelRegistry.get_spec(method_id)
            method_name = spec.name if spec else self._pretty_name(method_id)

            try:
                pipe = self._pipeline(X, model, method_id)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if use_cv:
                        scores = cross_validate(
                            pipe,
                            X,
                            y,
                            cv=splitter,
                            scoring=scoring,
                            n_jobs=self.n_jobs,
                            error_score="raise",
                        )

                        mae_values = -scores["test_mae"]
                        mse_values = -scores["test_mse"]
                        r2_values = scores["test_r2"]

                        metrics = {
                            "mae": round(float(np.mean(mae_values)), 4),
                            "rmse": round(float(np.sqrt(np.mean(mse_values))), 4),
                            "r2_score": round(float(np.mean(r2_values)), 4),
                        }
                        stability_values = r2_values
                        cv_std = round(float(np.std(stability_values)), 4)
                        cv_mean = metrics["r2_score"]
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=self.test_size,
                            random_state=self.random_state,
                        )
                        pipe.fit(X_train, y_train)
                        predictions = pipe.predict(X_test)
                        mse = mean_squared_error(y_test, predictions)
                        metrics = {
                            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
                            "rmse": round(float(np.sqrt(mse)), 4),
                            "r2_score": round(float(r2_score(y_test, predictions)), 4),
                        }
                        cv_std = 0.0
                        cv_mean = metrics["r2_score"]

                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "EVALUATED",
                    "metrics": metrics,
                    "validation": "k_fold" if use_cv else "holdout",
                    "folds": folds if use_cv else 1,
                    "stability": cv_std,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                })

            except Exception as exc:
                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "FAILED",
                    "metrics": {},
                    "error": str(exc),
                })

        evaluated = [r for r in results if r.get("status") == "EVALUATED"]
        evaluated.sort(
            key=lambda r: (r["metrics"]["r2_score"], -r["metrics"]["rmse"]),
            reverse=True,
        )

        failed = [
            {"method_id": r.get("method_id"), "error": r.get("error", "Unknown error")}
            for r in results
            if r.get("status") == "FAILED"
        ]

        return {
            "status": "EVALUATION" if evaluated else "FAILED",
            "task": "regression",
            "target": str(y.name),
            "dataset": {"rows": len(X), "features": X.shape[1]},
            "validation": "k_fold" if use_cv else "holdout",
            "folds": folds if use_cv else 1,
            "metrics_used": ["mae", "rmse", "r2_score"],
            "ranking_metric": "r2_score",
            "results": evaluated,
            "failed_methods": failed,
            "result_count": len(evaluated),
            "message": (
                "Model benar-benar dilatih dan dibandingkan menggunakan "
                f"{'K-Fold Cross Validation' if use_cv else 'holdout evaluation'}."
                f"{data_note}"
            ),
        }

    # =========================================================
    # CLUSTERING (Unsupervised)
    # =========================================================

    def _evaluate_clustering(
        self,
        X: pd.DataFrame,
        methods: List[str],
    ) -> Dict[str, Any]:
        preprocessor = self._build_preprocessor(X, requires_scaling=True)

        try:
            X_trans = preprocessor.fit_transform(X)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
        except Exception as exc:
            return self._error(f"Gagal melakukan preprocessing data clustering: {exc}")

        results: List[Dict[str, Any]] = []

        for method_id in methods:
            model = self._get_model_instance(method_id)
            if model is None:
                continue

            spec = ModelRegistry.get_spec(method_id)
            method_name = spec.name if spec else self._pretty_name(method_id)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if hasattr(model, "fit_predict"):
                        labels = model.fit_predict(X_trans)
                    elif hasattr(model, "predict"):
                        model.fit(X_trans)
                        labels = model.predict(X_trans)
                    else:
                        model.fit(X_trans)
                        labels = getattr(model, "labels_", None)

                if labels is None:
                    continue

                labels = np.asarray(labels)
                if len(labels) != len(X_trans):
                    raise ValueError("Jumlah label cluster tidak sama dengan jumlah baris data.")
                # DBSCAN uses -1 for noise. Noise is not a cluster and must not
                # be passed as one to internal validation metrics.
                valid_mask = labels != -1
                clustered_labels = labels[valid_mask]
                clustered_rows = X_trans[valid_mask]
                n_clusters = len(np.unique(clustered_labels))
                noise_count = int((~valid_mask).sum())

                if not (2 <= n_clusters < len(clustered_rows)):
                    results.append({
                        "method_id": method_id,
                        "method": method_name,
                        "status": "NOT_EVALUABLE",
                        "metrics": {
                            "n_clusters": n_clusters,
                            "noise_count": noise_count,
                            "evaluated_rows": len(clustered_rows),
                        },
                        "error": "Metrik internal membutuhkan sedikitnya 2 cluster dan lebih banyak baris daripada cluster setelah noise dikeluarkan.",
                    })
                    continue

                sil = float(silhouette_score(clustered_rows, clustered_labels))
                ch = float(calinski_harabasz_score(clustered_rows, clustered_labels))
                db = float(davies_bouldin_score(clustered_rows, clustered_labels))
                if not np.isfinite([sil, ch, db]).all():
                    raise ValueError("Metrik clustering menghasilkan nilai non-finite.")

                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "EVALUATED",
                    "metrics": {
                        "n_clusters": n_clusters,
                        "noise_count": noise_count,
                        "evaluated_rows": len(clustered_rows),
                        "silhouette_score": round(sil, 4),
                        "calinski_harabasz": round(ch, 2),
                        "davies_bouldin": round(db, 4),
                    },
                    "validation": "internal_cluster_validation",
                    "folds": 1,
                    "stability": None,
                    "cv_mean": None,
                    "cv_std": None,
                })

            except Exception as exc:
                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "FAILED",
                    "metrics": {},
                    "error": str(exc),
                })

        evaluated = [r for r in results if r.get("status") == "EVALUATED"]
        evaluated.sort(
            key=lambda r: r["metrics"]["silhouette_score"],
            reverse=True,
        )

        failed = [
            {"method_id": r.get("method_id"), "error": r.get("error", "Unknown error")}
            for r in results
            if r.get("status") != "EVALUATED"
        ]

        return {
            "status": "EVALUATION" if evaluated else "FAILED",
            "task": "clustering",
            "target": None,
            "dataset": {"rows": len(X), "features": X.shape[1]},
            "validation": "internal_cluster_validation",
            "folds": 1,
            "metrics_used": ["silhouette_score", "calinski_harabasz", "davies_bouldin"],
            "ranking_metric": "silhouette_score",
            "results": evaluated,
            "failed_methods": failed,
            "result_count": len(evaluated),
            "message": "Metrik clustering dihitung pada baris yang memiliki label cluster; noise dikeluarkan. Metode tanpa sedikitnya dua cluster tidak dirangking.",
        }

    # =========================================================
    # ANOMALY DETECTION (Unsupervised)
    # =========================================================

    def _evaluate_anomaly_detection(
        self,
        X: pd.DataFrame,
        methods: List[str],
    ) -> Dict[str, Any]:
        preprocessor = self._build_preprocessor(X, requires_scaling=True)

        try:
            X_trans = preprocessor.fit_transform(X)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
        except Exception as exc:
            return self._error(f"Gagal melakukan preprocessing data anomali: {exc}")

        results: List[Dict[str, Any]] = []

        for method_id in methods:
            model = self._get_model_instance(method_id)
            if model is None:
                continue

            spec = ModelRegistry.get_spec(method_id)
            method_name = spec.name if spec else self._pretty_name(method_id)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if hasattr(model, "fit_predict"):
                        preds = model.fit_predict(X_trans)
                    else:
                        model.fit(X_trans)
                        preds = model.predict(X_trans)

                n_anomalies = int(np.sum(preds == -1))
                n_total = len(X_trans)
                anomaly_ratio = round(float(n_anomalies / max(n_total, 1)) * 100.0, 2)

                score_spread = 0.0
                if hasattr(model, "decision_function"):
                    try:
                        scores = model.decision_function(X_trans)
                        score_spread = round(float(np.std(scores)), 4)
                    except Exception:
                        pass
                elif hasattr(model, "score_samples"):
                    try:
                        scores = model.score_samples(X_trans)
                        score_spread = round(float(np.std(scores)), 4)
                    except Exception:
                        pass

                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "EVALUATED",
                    "metrics": {
                        "anomalies_detected": n_anomalies,
                        "anomaly_ratio": anomaly_ratio,
                        "score_spread": score_spread,
                        "inliers_detected": int(np.sum(preds == 1)),
                    },
                    "validation": "outlier_separation_analysis",
                    "folds": 1,
                    "stability": score_spread,
                    "cv_mean": anomaly_ratio,
                    "cv_std": 0.0,
                })

            except Exception as exc:
                results.append({
                    "method_id": method_id,
                    "method": method_name,
                    "status": "FAILED",
                    "metrics": {},
                    "error": str(exc),
                })

        evaluated = [r for r in results if r.get("status") == "EVALUATED"]
        evaluated.sort(
            key=lambda r: (r["metrics"]["score_spread"], -abs(r["metrics"]["anomaly_ratio"] - 5.0)),
            reverse=True,
        )

        failed = [
            {"method_id": r.get("method_id"), "error": r.get("error", "Unknown error")}
            for r in results
            if r.get("status") == "FAILED"
        ]

        return {
            "status": "EVALUATION" if evaluated else "FAILED",
            "task": "anomaly_detection",
            "target": None,
            "dataset": {"rows": len(X), "features": X.shape[1]},
            "validation": "outlier_separation_analysis",
            "folds": 1,
            "metrics_used": ["anomalies_detected", "anomaly_ratio", "score_spread"],
            "ranking_metric": "score_spread",
            "results": evaluated,
            "failed_methods": failed,
            "result_count": len(evaluated),
            "message": "Model anomaly detection dilatih untuk memisahkan data reguler dan outlier aktual.",
        }

    # =========================================================
    # PREPROCESSING & PIPELINE (Leakage-Safe)
    # =========================================================

    def _pipeline(self, X: pd.DataFrame, model, method_id: str) -> Pipeline:
        spec = ModelRegistry.get_spec(method_id)
        requires_scaling = spec.requires_scaling if spec else False
        preprocessor = self._build_preprocessor(
            X, requires_scaling=requires_scaling, method_id=method_id, dynamic_columns=True
        )

        return Pipeline([
            ("feature_preparation", _FoldFeaturePreparation()),
            ("preprocessor", preprocessor),
            ("model", model),
        ])

    def _build_preprocessor(
        self,
        X: pd.DataFrame,
        requires_scaling: bool = False,
        method_id: str = "",
        dynamic_columns: bool = False,
    ) -> ColumnTransformer:
        numeric = _numeric_feature_columns if dynamic_columns else _numeric_feature_columns(X)
        categorical = _categorical_feature_columns if dynamic_columns else _categorical_feature_columns(X)

        transformers = []

        if dynamic_columns or numeric:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if requires_scaling or not method_id.startswith(("random_forest", "extra_trees", "decision_tree")):
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("numeric", Pipeline(num_steps), numeric))

        if dynamic_columns or categorical:
            if str(method_id).startswith("gradient_boosting") or str(method_id).startswith("hist_gradient"):
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]
            transformers.append(("categorical", Pipeline(cat_steps), categorical))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    @staticmethod
    def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Date parsing
        for col in list(X.columns):
            if MLEvaluator._looks_datetime(X[col]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(X[col], errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        X[f"{col}_year"] = parsed.dt.year
                        X[f"{col}_month"] = parsed.dt.month
                        X[f"{col}_day"] = parsed.dt.day
                        X = X.drop(columns=[col])

        # Drop identifiers and degenerate columns
        drop_columns = []
        for col in X.columns:
            series = X[col]
            normalized = str(col).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

            if normalized in {"id", "identifier", "uuid", "index", "recordid", "passengerid", "rowid"}:
                drop_columns.append(col)
                continue

            if series.nunique(dropna=True) <= 1:
                drop_columns.append(col)
                continue

            # Free text with very high cardinality
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                sample = series.dropna().astype(str).head(500)
                if not sample.empty:
                    avg_len = float(sample.str.len().mean())
                    unique_ratio = float(sample.nunique() / max(len(sample), 1))
                    if avg_len > 60 and unique_ratio > 0.5:
                        drop_columns.append(col)

        return X.drop(columns=drop_columns, errors="ignore")

    @staticmethod
    def _looks_datetime(series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False
        sample = series.dropna().head(50)
        if sample.empty:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) >= 0.8

    # =========================================================
    # SAMPLING
    # =========================================================

    def _stratified_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if len(X) <= n:
            return X, y

        selected_parts = []
        for _, indices in y.groupby(y).groups.items():
            count = max(1, int(round(n * len(indices) / len(y))))
            count = min(count, len(indices))
            selected_parts.append(
                pd.Series(indices).sample(count, random_state=self.random_state).to_numpy()
            )

        selected = np.concatenate(selected_parts)
        if len(selected) > n:
            selected = pd.Series(selected).sample(n, random_state=self.random_state).to_numpy()

        return X.loc[selected], y.loc[selected]

    # =========================================================
    # MODEL FACTORY HELPER
    # =========================================================

    def _get_model_instance(self, method_id: str):
        return ModelRegistry.instantiate(method_id, random_state=self.random_state)

    # Legacy compatibility methods
    def _classification_model(self, method_id: str):
        return self._get_model_instance(method_id)

    def _regression_model(self, method_id: str):
        return self._get_model_instance(method_id)

    # =========================================================
    # METRICS & HELPERS
    # =========================================================

    @staticmethod
    def _mean_pct(values) -> float:
        return round(float(np.mean(values) * 100.0), 2)

    @staticmethod
    def _classification_metrics(y_true, predictions) -> Dict[str, float]:
        return {
            "accuracy": round(float(accuracy_score(y_true, predictions) * 100.0), 2),
            "precision": round(
                float(precision_score(y_true, predictions, average="weighted", zero_division=0) * 100.0),
                2,
            ),
            "recall": round(
                float(recall_score(y_true, predictions, average="weighted", zero_division=0) * 100.0),
                2,
            ),
            "f1_score": round(
                float(f1_score(y_true, predictions, average="weighted", zero_division=0) * 100.0),
                2,
            ),
        }

    @staticmethod
    def _pretty_name(method_id: str) -> str:
        return str(method_id).replace("_", " ").title()

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "results": [],
            "failed_methods": [],
            "result_count": 0,
            "message": message,
        }
