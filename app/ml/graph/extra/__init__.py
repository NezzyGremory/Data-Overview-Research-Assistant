"""Optional and extended graph node catalog."""

from __future__ import annotations

import importlib.util
import json
import math

import pandas as pd
from sklearn.base import clone, is_classifier
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


_DEFAULT_GRID_VALUES = {
    "C": [0.1, 1.0, 10.0],
    "alpha": [0.0001, 0.01, 1.0],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "n_neighbors": [3, 5, 9, 15],
    "max_features": ["sqrt", "log2", None],
    "kernel": ["linear", "rbf"],
    "class_weight": [None, "balanced"],
}
_TUNING_PARAMETER_ORDER = (
    "C", "n_estimators", "max_depth", "learning_rate", "alpha",
    "n_neighbors", "min_samples_split", "min_samples_leaf", "max_features",
    "kernel", "class_weight",
)
_DEFAULT_OPTUNA_SPACE = {
    "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
    "alpha": {"type": "float", "low": 0.0001, "high": 10.0, "log": True},
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "n_estimators": {"type": "int", "low": 50, "high": 500},
    "max_depth": {"type": "int", "low": 2, "high": 30},
    "min_samples_split": {"type": "int", "low": 2, "high": 20},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
    "n_neighbors": {"type": "int", "low": 1, "high": 50},
    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    "kernel": {"type": "categorical", "choices": ["linear", "rbf"]},
    "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
}


def _tuning_data(inputs, params, estimator):
    frame = inputs.get("table")
    target = params.get("target")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input tuning harus berupa tabel dataset.")
    if not target or target not in frame.columns:
        raise ValueError("Target belum dipilih atau tidak ditemukan pada tabel.")
    if len(frame) < 2:
        raise ValueError("Dataset membutuhkan minimal dua baris untuk tuning.")
    if frame[target].isna().any():
        raise ValueError("Kolom target masih memiliki nilai kosong. Bersihkan atau hapus baris tersebut sebelum tuning.")
    X, y = frame.drop(columns=[target]), frame[target]
    if X.shape[1] == 0:
        raise ValueError("Dataset tidak memiliki kolom fitur selain target.")

    task = params.get("task", "auto")
    task = task if task in {"classification", "regression"} else (
        "classification" if is_classifier(estimator) else "regression"
    )
    folds = int(params.get("cv", 3))
    if folds < 2 or len(frame) < folds:
        raise ValueError("Jumlah fold CV harus minimal 2 dan tidak boleh melebihi jumlah baris dataset.")
    if task == "classification":
        class_counts = y.value_counts(dropna=False)
        if len(class_counts) < 2:
            raise ValueError("Target klasifikasi harus memiliki sedikitnya dua kelas.")
        if int(class_counts.min()) < folds:
            raise ValueError(
                f"Jumlah fold CV ({folds}) melebihi jumlah data pada kelas terkecil ({int(class_counts.min())})."
            )
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=int(params.get("random_state", 42))
        )
        ratio = float(class_counts.min() / class_counts.max()) if class_counts.max() else 1.0
        imbalanced = ratio < 0.35
        default_scoring = "f1_macro" if imbalanced else "accuracy"
    else:
        from sklearn.model_selection import KFold
        splitter = KFold(
            n_splits=folds, shuffle=True, random_state=int(params.get("random_state", 42))
        )
        ratio, imbalanced, default_scoring = 1.0, False, "r2"

    scoring = params.get("scoring")
    if not scoring or scoring == "auto":
        scoring = default_scoring
    from sklearn.metrics import get_scorer
    try:
        get_scorer(scoring)
    except Exception as exc:
        raise ValueError(f"Metrik scoring '{scoring}' tidak dikenali scikit-learn.") from exc

    # Keep preprocessing inside every CV fold. This prevents validation leakage
    # and lets tuning work with raw numeric, categorical, and missing features.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    wrapped = Pipeline([("preprocessor", preprocessor), ("estimator", clone(estimator))])
    return X, y, task, splitter, scoring, wrapped, imbalanced, ratio


def _normalize_parameter_name(name, estimator, wrapped):
    name = str(name).strip()
    if name.startswith("model__"):
        name = "estimator__" + name.removeprefix("model__")
    if name.startswith("estimator__") or name.startswith("preprocessor__"):
        normalized = name
    elif name in estimator.get_params(deep=True):
        normalized = f"estimator__{name}"
    elif name in wrapped.get_params(deep=True):
        normalized = name
    else:
        raise ValueError(f"Parameter '{name}' tidak tersedia pada estimator yang dipilih.")
    if normalized not in wrapped.get_params(deep=True):
        raise ValueError(f"Parameter '{name}' tidak tersedia pada estimator yang dipilih.")
    return normalized


def _default_parameter_grid(estimator, wrapped, row_count):
    available = estimator.get_params(deep=True)
    grid = {}
    for name in _TUNING_PARAMETER_ORDER:
        if name not in available or name not in _DEFAULT_GRID_VALUES:
            continue
        values = list(_DEFAULT_GRID_VALUES[name])
        if name == "n_neighbors":
            values = [value for value in values if value < row_count]
        if values:
            grid[f"estimator__{name}"] = values
        if len(grid) >= 3:
            break
    if not grid:
        raise ValueError(
            "Estimator ini belum memiliki ruang parameter bawaan. Isi param_grid secara manual dengan format JSON."
        )
    return grid


def _parse_parameter_grid(raw_grid, estimator, wrapped, row_count, random_search=False):
    if not isinstance(raw_grid, dict):
        raise ValueError("param_grid harus berupa objek JSON berisi nama parameter dan daftar nilai.")
    if not raw_grid:
        return _default_parameter_grid(estimator, wrapped, row_count)
    from scipy.stats import loguniform, randint, uniform
    normalized = {}
    for key, raw_values in raw_grid.items():
        parameter = _normalize_parameter_name(key, estimator, wrapped)
        if isinstance(raw_values, dict):
            if not random_search:
                raise ValueError("Grid Search menerima daftar nilai. Gunakan Random Search untuk rentang distribusi.")
            kind = str(raw_values.get("type", "")).lower()
            if kind in {"int", "integer"}:
                low, high = int(raw_values["low"]), int(raw_values["high"])
                if high < low:
                    raise ValueError(f"Batas rentang '{key}' tidak valid.")
                normalized[parameter] = randint(low, high + 1)
            elif kind in {"float", "continuous"}:
                low, high = float(raw_values["low"]), float(raw_values["high"])
                if not math.isfinite(low) or not math.isfinite(high) or high <= low:
                    raise ValueError(f"Batas rentang '{key}' tidak valid.")
                if raw_values.get("log") and low <= 0:
                    raise ValueError(f"Batas bawah distribusi log '{key}' harus lebih besar dari 0.")
                normalized[parameter] = (
                    loguniform(low, high) if raw_values.get("log") else uniform(low, high - low)
                )
            elif kind in {"categorical", "category"}:
                choices = raw_values.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError(f"Parameter kategori '{key}' membutuhkan daftar choices.")
                normalized[parameter] = choices
            else:
                raise ValueError(f"Jenis distribusi '{kind}' untuk '{key}' tidak didukung.")
        elif isinstance(raw_values, (list, tuple)) and raw_values:
            normalized[parameter] = list(raw_values)
        else:
            raise ValueError(f"Nilai parameter '{key}' harus berupa daftar yang tidak kosong.")
    return normalized


def _score_metric(scoring, score):
    metric_names = {
        "accuracy": ("accuracy", False),
        "balanced_accuracy": ("balanced_accuracy", False),
        "f1": ("f1", False),
        "f1_macro": ("f1_macro", False),
        "f1_weighted": ("f1_weighted", False),
        "precision_macro": ("precision", False),
        "recall_macro": ("recall", False),
        "roc_auc_ovr_weighted": ("roc_auc", False),
        "r2": ("r2", False),
        "explained_variance": ("explained_variance", False),
        "neg_mean_absolute_error": ("mae", True),
        "neg_root_mean_squared_error": ("rmse", True),
        "neg_mean_squared_error": ("mse", True),
        "neg_median_absolute_error": ("median_ae", True),
    }
    name, negate = metric_names.get(scoring, ("cv_score", False))
    value = -float(score) if negate else float(score)
    return name, value


def _tuning_table(cv_results, scoring):
    frame = pd.DataFrame(cv_results)
    ordered = frame.sort_values("rank_test_score", kind="stable")
    records = []
    metric_name = _score_metric(scoring, 0.0)[0]
    percent_score = metric_name in {
        "accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted", "precision", "recall", "r2"
    }
    for _, row in ordered.head(200).iterrows():
        mean_score = _score_metric(scoring, row["mean_test_score"])[1]
        record = {
            "rank": int(row["rank_test_score"]),
            "mean_cv_score": mean_score,
            "std_cv_score": float(row["std_test_score"]),
            "mean_fit_time_sec": float(row.get("mean_fit_time", 0.0)),
            "params": json.dumps(
                {key.removeprefix("estimator__"): value for key, value in row["params"].items()},
                ensure_ascii=False,
                default=str,
            ),
        }
        if percent_score:
            record["mean_score_percent"] = f"{mean_score * 100:.2f}%"
        for extra in ("iter", "n_resources"):
            if extra in row.index:
                record[extra] = int(row[extra])
        records.append(record)
    return pd.DataFrame(records)


def _search(inputs, params, context):
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import (
        GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV,
    )

    estimator = inputs["estimator"]
    X, y, task, splitter, scoring, wrapped, imbalanced, imbalance_ratio = _tuning_data(inputs, params, estimator)
    method = str(params.get("method", "grid"))
    is_random = method in {"random", "halving_random"}
    parameter_values = _parse_parameter_grid(
        params.get("param_grid") or {}, estimator, wrapped, len(X), random_search=is_random
    )
    jobs = int(params.get("n_jobs", 1))
    common = {
        "estimator": wrapped,
        "cv": splitter,
        "scoring": scoring,
        "n_jobs": jobs,
        "error_score": "raise",
        "return_train_score": False,
    }
    if method == "grid":
        search = GridSearchCV(param_grid=parameter_values, **common)
    elif method == "random":
        n_iter = int(params.get("n_iter", 20))
        if all(isinstance(values, list) for values in parameter_values.values()):
            combinations = math.prod(len(values) for values in parameter_values.values())
            n_iter = min(n_iter, combinations)
        search = RandomizedSearchCV(
            param_distributions=parameter_values,
            n_iter=max(1, n_iter),
            random_state=int(params.get("random_state", 42)),
            **common,
        )
    elif method == "halving_grid":
        search = HalvingGridSearchCV(
            param_grid=parameter_values,
            factor=int(params.get("factor", 3)),
            min_resources=params.get("min_resources", "smallest"),
            max_resources=params.get("max_resources", "auto"),
            resource=params.get("resource", "n_samples"),
            aggressive_elimination=bool(params.get("aggressive_elimination", False)),
            **common,
        )
    elif method == "halving_random":
        search = HalvingRandomSearchCV(
            param_distributions=parameter_values,
            n_candidates=int(params.get("n_iter", 20)),
            factor=int(params.get("factor", 3)),
            min_resources=params.get("min_resources", "smallest"),
            max_resources=params.get("max_resources", "auto"),
            resource=params.get("resource", "n_samples"),
            aggressive_elimination=bool(params.get("aggressive_elimination", False)),
            random_state=int(params.get("random_state", 42)),
            **common,
        )
    else:
        raise ValueError(f"Metode tuning '{method}' tidak dikenali.")

    search.fit(X, y)
    best_params = {
        key.removeprefix("estimator__"): value
        for key, value in search.best_params_.items()
    }
    metric_name, best_metric = _score_metric(scoring, search.best_score_)
    metrics = {
        "best_score": float(search.best_score_),
        metric_name: float(best_metric),
        "scoring": scoring,
        "best_params": json.dumps(best_params, ensure_ascii=False, default=str),
        "search_method": method,
        "cv": int(params.get("cv", 3)),
        "candidate_count": len(search.cv_results_["params"]),
        "task": task,
        "is_imbalanced": bool(imbalanced),
        "imbalance_ratio": float(imbalance_ratio),
    }
    return {
        "estimator": search.best_estimator_,
        "table": _tuning_table(search.cv_results_, scoring),
        "metrics": {"model": str(search.best_estimator_.named_steps["estimator"]), "metrics": metrics},
    }


def _default_optuna_space(estimator, wrapped, row_count):
    available = estimator.get_params(deep=True)
    spaces = {}
    for name in _TUNING_PARAMETER_ORDER:
        if name not in available or name not in _DEFAULT_OPTUNA_SPACE:
            continue
        space = dict(_DEFAULT_OPTUNA_SPACE[name])
        if name == "n_neighbors":
            space["high"] = min(space["high"], max(1, row_count - 1))
        if name in {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "n_neighbors"}:
            space["high"] = max(space["low"], space["high"])
        spaces[f"estimator__{name}"] = space
        if len(spaces) >= 5:
            break
    if not spaces:
        raise ValueError(
            "Estimator ini belum memiliki ruang parameter bawaan. Isi search_space secara manual dengan format JSON."
        )
    return spaces


def _suggest_optuna_value(trial, name, spec):
    if not isinstance(spec, dict):
        if isinstance(spec, list) and spec:
            return trial.suggest_categorical(name, spec)
        raise ValueError(f"Ruang parameter '{name}' harus berupa objek atau daftar kategori.")
    kind = str(spec.get("type", "")).lower()
    if kind in {"float", "continuous"}:
        low, high = float(spec["low"]), float(spec["high"])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError(f"Batas rentang '{name}' tidak valid.")
        if spec.get("log") and low <= 0:
            raise ValueError(f"Batas bawah distribusi log '{name}' harus lebih besar dari 0.")
        return trial.suggest_float(name, low, high, log=bool(spec.get("log", False)))
    if kind in {"int", "integer"}:
        low, high = int(spec["low"]), int(spec["high"])
        if high < low:
            raise ValueError(f"Batas rentang '{name}' tidak valid.")
        return trial.suggest_int(name, low, high, log=bool(spec.get("log", False)))
    if kind in {"categorical", "category", "bool", "boolean"}:
        choices = spec.get("choices", [True, False] if kind in {"bool", "boolean"} else [])
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Ruang kategori '{name}' membutuhkan daftar choices.")
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Jenis ruang parameter '{kind}' untuk '{name}' tidak didukung.")


def _optuna(inputs, params, context):
    if importlib.util.find_spec("optuna") is None:
        raise GraphError("Library 'optuna' belum terpasang")
    import optuna
    from sklearn.model_selection import cross_val_score

    estimator = inputs["estimator"]
    X, y, task, splitter, scoring, wrapped, imbalanced, imbalance_ratio = _tuning_data(inputs, params, estimator)
    search_space = params.get("search_space") or {}
    if not isinstance(search_space, dict):
        raise ValueError("search_space harus berupa objek JSON.")
    if not search_space:
        search_space = _default_optuna_space(estimator, wrapped, len(X))
    normalized_space = {
        _normalize_parameter_name(name, estimator, wrapped): spec
        for name, spec in search_space.items()
    }

    sampler_name = str(params.get("sampler", "tpe")).lower()
    seed = int(params.get("random_state", 42))
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError("Sampler Optuna harus TPE atau Random.")
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        values = {
            name: _suggest_optuna_value(trial, name, spec)
            for name, spec in normalized_space.items()
        }
        candidate = clone(wrapped).set_params(**values)
        return float(cross_val_score(
            candidate, X, y, cv=splitter, scoring=scoring, n_jobs=1, error_score="raise"
        ).mean())

    timeout = int(params.get("timeout", 0))
    study.optimize(
        objective,
        n_trials=int(params.get("trials", 30)),
        timeout=timeout or None,
        n_jobs=1,
        show_progress_bar=False,
    )
    if not study.best_trials:
        raise ValueError("Optuna tidak menyelesaikan percobaan yang valid sebelum batas waktu.")

    best_params = {
        key.removeprefix("estimator__"): value
        for key, value in study.best_params.items()
    }
    metric_name, best_metric = _score_metric(scoring, study.best_value)
    records = []
    for trial in study.trials:
        record = {
            "trial": trial.number + 1,
            "status": trial.state.name,
            "score": _score_metric(scoring, trial.value)[1] if trial.value is not None else None,
            "duration_sec": trial.duration.total_seconds() if trial.duration else None,
            "params": json.dumps(
                {key.removeprefix("estimator__"): value for key, value in trial.params.items()},
                ensure_ascii=False,
                default=str,
            ),
        }
        if trial.value is not None and metric_name in {
            "accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted", "precision", "recall", "r2"
        }:
            record["score_percent"] = f"{_score_metric(scoring, trial.value)[1] * 100:.2f}%"
        records.append(record)

    metrics = {
        "best_score": float(study.best_value),
        metric_name: float(best_metric),
        "scoring": scoring,
        "best_params": json.dumps(best_params, ensure_ascii=False, default=str),
        "sampler": sampler_name,
        "trials_completed": len(study.trials),
        "cv": int(params.get("cv", 3)),
        "task": task,
        "is_imbalanced": bool(imbalanced),
        "imbalance_ratio": float(imbalance_ratio),
    }
    best_estimator = clone(wrapped).set_params(**study.best_params).fit(X, y)
    return {
        "estimator": best_estimator,
        "table": pd.DataFrame(records),
        "metrics": {"model": str(estimator), "metrics": metrics},
    }


def _cross_validate(inputs, params, context):
    from sklearn.model_selection import cross_validate

    estimator = inputs["estimator"]
    X, y, task, splitter, scoring, wrapped, imbalanced, imbalance_ratio = _tuning_data(inputs, params, estimator)
    result = cross_validate(
        wrapped, X, y, cv=splitter, scoring=scoring, return_train_score=False,
        n_jobs=int(params.get("n_jobs", 1)),
    )
    scores = result["test_score"]
    metric_name, mean_metric = _score_metric(scoring, scores.mean())
    fold_rows = [
        {"fold": index + 1, "score": _score_metric(scoring, score)[1], "fit_time_sec": float(result["fit_time"][index])}
        for index, score in enumerate(scores)
    ]
    metrics = {
        "mean_score": float(scores.mean()),
        metric_name: float(mean_metric),
        "std_score": float(scores.std()),
        "scoring": scoring,
        "task": task,
        "is_imbalanced": bool(imbalanced),
        "imbalance_ratio": float(imbalance_ratio),
    }
    return {
        "table": pd.DataFrame(fold_rows),
        "metrics": {"model": str(estimator), "metrics": metrics},
    }


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
    graph.register(NodeSpec(
        "cross_validate", "Cross-Validation", "Validasi", ["classification", "regression"],
        [table, PortSpec("estimator", "estimator")], [PortSpec("metrics", "metrics")], _cross_validate,
        {"target": "", "task": "auto", "scoring": "auto", "cv": 5, "random_state": 42, "n_jobs": 1},
    ))
    search_inputs = [table, PortSpec("estimator", "estimator")]
    search_outputs = [PortSpec("estimator", "estimator"), PortSpec("table", "table"), PortSpec("metrics", "metrics")]
    common_search = {
        "target": "", "task": "auto", "scoring": "auto", "cv": 5,
        "random_state": 42, "n_jobs": 1, "param_grid": {},
    }
    search_nodes = [
        ("grid_search", "Grid Search", "grid", dict(common_search)),
        ("random_search", "Random Search", "random", {**common_search, "n_iter": 20}),
        ("halving_grid_search", "Successive Halving Grid", "halving_grid", {
            **common_search, "factor": 3, "min_resources": "smallest", "max_resources": "auto",
            "resource": "n_samples", "aggressive_elimination": False,
        }),
        ("halving_random_search", "Successive Halving Random", "halving_random", {
            **common_search, "n_iter": 30, "factor": 3, "min_resources": "smallest",
            "max_resources": "auto", "resource": "n_samples", "aggressive_elimination": False,
        }),
    ]
    for key, label, method, defaults in search_nodes:
        graph.register(NodeSpec(
            key, label, "Tuning", ["classification", "regression"], search_inputs,
            search_outputs, _search, {**defaults, "method": method},
        ))
    graph.register(NodeSpec(
        "optuna", "Optuna Tuner", "Tuning", ["classification", "regression"],
        search_inputs, search_outputs, _optuna,
        {
            "target": "", "task": "auto", "scoring": "auto", "cv": 5,
            "trials": 30, "timeout": 0, "sampler": "tpe", "random_state": 42,
            "search_space": {},
        },
        "optuna",
    ))
    groups = {"classification": [("ridge_classifier", "Ridge Classifier"), ("passive_aggressive_classifier", "Passive Aggressive Classifier"), ("multinomial_nb", "Multinomial Naive Bayes"), ("complement_nb", "Complement Naive Bayes"), ("nearest_centroid", "Nearest Centroid"), ("calibrated_classifier", "Calibrated Classifier")], "regression": [("bayesian_ridge", "Bayesian Ridge"), ("theil_sen", "Theil-Sen Regressor"), ("passive_aggressive_regressor", "Passive Aggressive Regressor"), ("kernel_ridge", "Kernel Ridge")], "clustering": [("optics", "OPTICS"), ("mean_shift", "Mean Shift"), ("affinity_propagation", "Affinity Propagation"), ("hdbscan", "HDBSCAN")]}
    import sklearn.cluster as cluster_module
    if not hasattr(cluster_module, "HDBSCAN"):
        groups["clustering"] = [item for item in groups["clustering"] if item[0] != "hdbscan"]
    for task, methods in groups.items():
        for key, label in methods:
            graph.register(NodeSpec(f"extra_model:{key}", label, task.title(), [task], [], [PortSpec("estimator", "estimator")], _make_extra_estimator, {"factory": key}, max_recommended_rows=10_000 if key in {"theil_sen", "mean_shift", "affinity_propagation"} else 100_000))
    graph.register(NodeSpec("time_split", "Split Waktu", "Forecasting", ["time_series"], [table], pair_outputs, _time_split, {"time_column": "", "test_size": .2}))
    graph.register(NodeSpec("forecast", "Forecasting", "Forecasting", ["time_series"], pair_inputs, [PortSpec("metrics", "metrics")], _forecast, {"target": "", "method": "naive"}))
