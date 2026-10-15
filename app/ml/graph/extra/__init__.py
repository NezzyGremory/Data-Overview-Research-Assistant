"""Optional and extended graph node catalog."""

from __future__ import annotations

import importlib.util

import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from app.ml.graph.engine import GraphError, NodeSpec, PortSpec


def _paired_transform(inputs, params, context):
    train, test = inputs["train"].copy(), inputs["test"].copy()
    target = params.get("target")
    y_train = train.pop(target) if target and target in train else None
    y_test = test.pop(target) if target and target in test else None
    kind = params["operation"]
    if kind == "scale":
        transformer = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[params.get("method", "standard")]()
    elif kind == "pca":
        transformer = PCA(n_components=params.get("components", .95))
    elif kind == "variance":
        transformer = VarianceThreshold(params.get("threshold", 0.0))
    elif kind == "select_k":
        if y_train is None:
            raise ValueError("Seleksi fitur membutuhkan target pada data train.")
        transformer = SelectKBest(f_classif, k=min(int(params.get("k", 5)), train.shape[1]))
    elif kind == "outlier":
        lo, hi = train.quantile(.25) - 1.5 * (train.quantile(.75) - train.quantile(.25)), train.quantile(.75) + 1.5 * (train.quantile(.75) - train.quantile(.25))
        return {"train": train.clip(lo, hi, axis=1).assign(**({target: y_train} if y_train is not None else {})), "test": test.clip(lo, hi, axis=1).assign(**({target: y_test} if y_test is not None else {}))}
    else:
        raise ValueError("Operasi fitur tidak dikenal.")
    a = transformer.fit_transform(train, y_train)
    b = transformer.transform(test)
    columns = [f"fitur_{i}" for i in range(a.shape[1])]
    a, b = pd.DataFrame(a, index=train.index, columns=columns), pd.DataFrame(b, index=test.index, columns=columns)
    if y_train is not None:
        a[target], b[target] = y_train, y_test
    return {"train": a, "test": b}


def _smote(inputs, params, context):
    if importlib.util.find_spec("imblearn") is None:
        raise GraphError("Library 'imbalanced-learn' belum terpasang")
    from imblearn.over_sampling import SMOTE
    frame = inputs["train"].copy()
    target = params.get("target")
    if not target or target not in frame:
        raise ValueError("SMOTE membutuhkan target di data train.")
    min_samples = frame[target].value_counts().min()
    k_neighbors = min(int(params.get("k_neighbors", 5)), max(1, min_samples - 1))
    X, y = SMOTE(k_neighbors=k_neighbors, random_state=int(params.get("random_state", 42))).fit_resample(frame.drop(columns=[target]), frame[target])
    result = pd.DataFrame(X, columns=frame.drop(columns=[target]).columns)
    result[target] = y
    return {"train": result}


def _cross_validate(inputs, params, context):
    from sklearn.model_selection import cross_validate
    frame, estimator = inputs["table"], inputs["estimator"]
    target = params.get("target")
    if target not in frame:
        raise ValueError("Target tidak ditemukan.")
    scoring = params.get("scoring", "accuracy")
    result = cross_validate(clone(estimator), frame.drop(columns=[target]), frame[target], cv=int(params.get("cv", 3)), scoring=scoring, return_train_score=False)
    return {"metrics": {"model": str(estimator), "metrics": {"mean_score": float(result["test_score"].mean()), "std_score": float(result["test_score"].std())}}}


def _search(inputs, params, context):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    frame, estimator = inputs["table"], inputs["estimator"]
    target = params.get("target")
    if target not in frame:
        raise ValueError("Target tidak ditemukan.")
    grid = params.get("param_grid") or {}
    if not grid:
        available = estimator.get_params(deep=False)
        key = next((name for name in ("C", "n_estimators", "max_depth", "alpha", "n_neighbors") if name in available), None)
        if key is None:
            raise ValueError("Estimator ini tidak memiliki parameter bawaan untuk tuning.")
        grid = {key: ([0.1, 1, 10] if key in {"C", "alpha"} else [10, 50, 100] if key in {"n_estimators", "max_depth"} else [3, 5, 7])}
    cls = GridSearchCV if params.get("method") == "grid" else RandomizedSearchCV
    options = {"estimator": clone(estimator), "param_distributions" if cls is RandomizedSearchCV else "param_grid": grid, "cv": int(params.get("cv", 3)), "n_iter": int(params.get("n_iter", 10)), "error_score": "raise"} if cls is RandomizedSearchCV else {"estimator": clone(estimator), "param_grid": grid, "cv": int(params.get("cv", 3)), "error_score": "raise"}
    search = cls(**options).fit(frame.drop(columns=[target]), frame[target])
    return {"estimator": search.best_estimator_, "metrics": {"model": str(search.best_estimator_), "metrics": {"best_score": float(search.best_score_)}}}


def _optuna(inputs, params, context):
    if importlib.util.find_spec("optuna") is None:
        raise GraphError("Library 'optuna' belum terpasang")
    import optuna
    from sklearn.model_selection import cross_val_score
    frame, estimator = inputs["table"], inputs["estimator"]
    target = params.get("target")
    if target not in frame:
        raise ValueError("Target tidak ditemukan.")
    X, y = frame.drop(columns=[target]), frame[target]
    available = estimator.get_params(deep=False)
    parameter = params.get("parameter") or next((name for name in ("C", "n_estimators", "max_depth", "alpha", "n_neighbors") if name in available), "")
    if not parameter:
        raise ValueError("Estimator ini tidak memiliki parameter bawaan untuk tuning.")
    def objective(trial):
        candidate = clone(estimator)
        if parameter in {"C", "alpha", "learning_rate"}:
            value = trial.suggest_float(parameter, .001, 100.0, log=True)
        elif parameter in {"n_estimators", "max_depth", "n_neighbors"}:
            value = trial.suggest_int(parameter, 2, 200)
        else:
            raise ValueError(f"Parameter '{parameter}' belum didukung tuner.")
        candidate.set_params(**{parameter: value})
        return float(cross_val_score(candidate, X, y, cv=int(params.get("cv", 3)), scoring=params.get("scoring", "accuracy")).mean())
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(params.get("trials", 20)))
    best = clone(estimator).set_params(**study.best_params)
    return {"estimator": best, "metrics": {"model": str(best), "metrics": {"best_score": float(study.best_value)}}}


def _time_split(inputs, params, context):
    frame = inputs["table"].sort_values(params.get("time_column"), kind="stable")
    cutoff = max(1, int(len(frame) * (1 - float(params.get("test_size", .2)))))
    return {"train": frame.iloc[:cutoff].copy(), "test": frame.iloc[cutoff:].copy()}


def _forecast(inputs, params, context):
    train, test = inputs["train"], inputs["test"]
    target = params.get("target")
    y, actual = train[target].astype(float), test[target].astype(float)
    method = params.get("method", "naive")
    if method == "naive":
        pred = [float(y.iloc[-1])] * len(test)
    elif method == "seasonal_naive":
        period = max(1, int(params.get("period", 12)))
        pred = [float(y.iloc[-period + (i % period)]) for i in range(len(test))]
    else:
        if importlib.util.find_spec("statsmodels") is None:
            raise GraphError("Library 'statsmodels' belum terpasang")
        if method == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            pred = ARIMA(y, order=tuple(params.get("order", [1, 1, 1]))).fit().forecast(len(test))
        elif method == "exponential_smoothing":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            pred = ExponentialSmoothing(y).fit().forecast(len(test))
        else:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            pred = SARIMAX(y, order=tuple(params.get("order", [1, 1, 1])), seasonal_order=tuple(params.get("seasonal_order", [1, 0, 0, 12]))).fit(disp=False).forecast(len(test))
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    pred = pd.Series(pred, index=actual.index)
    return {"metrics": {"model": method, "metrics": {"mae": float(mean_absolute_error(actual, pred)), "rmse": float(np.sqrt(mean_squared_error(actual, pred))), "mape": float((abs((actual - pred) / actual.replace(0, float("nan")))).mean() * 100)}}}


def _make_extra_estimator(inputs, params, context):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, BayesianRidge, TheilSenRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.neighbors import NearestCentroid
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.cluster import OPTICS, MeanShift, AffinityPropagation
    import sklearn.cluster as cluster_module
    factory = {"ridge_classifier": RidgeClassifier, "passive_aggressive_classifier": PassiveAggressiveClassifier, "multinomial_nb": MultinomialNB, "complement_nb": ComplementNB, "nearest_centroid": NearestCentroid, "calibrated_classifier": lambda: CalibratedClassifierCV(estimator=LogisticRegression(max_iter=1000)), "bayesian_ridge": BayesianRidge, "theil_sen": TheilSenRegressor, "passive_aggressive_regressor": PassiveAggressiveRegressor, "kernel_ridge": KernelRidge, "optics": OPTICS, "mean_shift": MeanShift, "affinity_propagation": AffinityPropagation}
    if hasattr(cluster_module, "HDBSCAN"):
        factory["hdbscan"] = cluster_module.HDBSCAN
    if params["factory"] not in factory:
        raise GraphError("HDBSCAN tidak tersedia pada versi scikit-learn ini.")
    factory = factory[params["factory"]]
    return {"estimator": factory()}


def register_extra(graph):
    table = PortSpec("table", "table")
    pair_inputs = [PortSpec("train", "table"), PortSpec("test", "table")]
    pair_outputs = [PortSpec("train", "table"), PortSpec("test", "table")]
    for key, label, operation, defaults in [("scale", "Scaler", "scale", {"method": "standard"}), ("pca", "PCA", "pca", {"components": .95}), ("variance", "Seleksi Variance", "variance", {"threshold": 0.0}), ("select_k", "Seleksi Fitur KBest", "select_k", {"k": 5}), ("outlier", "Penanganan Outlier IQR", "outlier", {})]:
        graph.register(NodeSpec(f"feature:{key}", label, "Preprocessing", ["all"], pair_inputs, pair_outputs, _paired_transform, {"operation": operation, **defaults, "target": ""}))
    graph.register(NodeSpec("smote", "SMOTE", "Data Tidak Seimbang", ["classification"], [PortSpec("train", "table")], [PortSpec("train", "table")], _smote, {"target": "", "random_state": 42}, "imblearn"))
    graph.register(NodeSpec("cross_validate", "Cross-Validation", "Validasi", ["classification", "regression"], [table, PortSpec("estimator", "estimator")], [PortSpec("metrics", "metrics")], _cross_validate, {"target": "", "cv": 3}))
    for key, label, method in [("grid_search", "Grid Search", "grid"), ("random_search", "Random Search", "random")]:
        graph.register(NodeSpec(key, label, "Tuning", ["classification", "regression"], [table, PortSpec("estimator", "estimator")], [PortSpec("estimator", "estimator"), PortSpec("metrics", "metrics")], _search, {"target": "", "method": method, "cv": 3}))
    graph.register(NodeSpec("optuna", "Optuna Tuner", "Tuning", ["classification", "regression"], [table, PortSpec("estimator", "estimator")], [PortSpec("estimator", "estimator"), PortSpec("metrics", "metrics")], _optuna, {}, "optuna"))
    groups = {"classification": [("ridge_classifier", "Ridge Classifier"), ("passive_aggressive_classifier", "Passive Aggressive Classifier"), ("multinomial_nb", "Multinomial Naive Bayes"), ("complement_nb", "Complement Naive Bayes"), ("nearest_centroid", "Nearest Centroid"), ("calibrated_classifier", "Calibrated Classifier")], "regression": [("bayesian_ridge", "Bayesian Ridge"), ("theil_sen", "Theil-Sen Regressor"), ("passive_aggressive_regressor", "Passive Aggressive Regressor"), ("kernel_ridge", "Kernel Ridge")], "clustering": [("optics", "OPTICS"), ("mean_shift", "Mean Shift"), ("affinity_propagation", "Affinity Propagation"), ("hdbscan", "HDBSCAN")]}
    import sklearn.cluster as cluster_module
    if not hasattr(cluster_module, "HDBSCAN"):
        groups["clustering"] = [item for item in groups["clustering"] if item[0] != "hdbscan"]
    for task, methods in groups.items():
        for key, label in methods:
            graph.register(NodeSpec(f"extra_model:{key}", label, task.title(), [task], [], [PortSpec("estimator", "estimator")], _make_extra_estimator, {"factory": key}, max_recommended_rows=10_000 if key in {"theil_sen", "mean_shift", "affinity_propagation"} else 100_000))
    graph.register(NodeSpec("time_split", "Split Waktu", "Forecasting", ["time_series"], [table], pair_outputs, _time_split, {"time_column": "", "test_size": .2}))
    graph.register(NodeSpec("forecast", "Forecasting", "Forecasting", ["time_series"], pair_inputs, [PortSpec("metrics", "metrics")], _forecast, {"target": "", "method": "naive"}))
