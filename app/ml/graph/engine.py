from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.ml.registry import ModelRegistry


class GraphError(ValueError):
    """Invalid graph structure or execution request."""


@dataclass(frozen=True)
class PortSpec:
    name: str
    type: str
    multiple: bool = False


@dataclass
class NodeSpec:
    id: str
    name: str
    category: str
    task_types: list[str]
    inputs: list[PortSpec]
    outputs: list[PortSpec]
    factory: Callable[..., dict[str, Any]]
    parameters: dict[str, Any] = field(default_factory=dict)
    check_package: str | None = None
    requires_scaling: bool = False
    max_recommended_rows: int = 100_000
    requires: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Node:
    id: str
    spec_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)
    status: str = "menunggu"


class GraphEngine:
    """Typed DAG engine. Node factories receive ``inputs`` and parameters."""

    def __init__(self) -> None:
        self.specs: dict[str, NodeSpec] = {}
        self.nodes: dict[str, Node] = {}
        self.connections: list[tuple[str, str, str, str]] = []
        self._cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self.register_builtins()

    def register(self, spec: NodeSpec) -> None:
        self.specs[spec.id] = spec

    def register_builtins(self) -> None:
        self.register(NodeSpec("load_dataset", "Muat Dataset", "Data", ["all"], [], [PortSpec("table", "table")], lambda inputs, p, context: {"table": context.get("dataframe")}))
        self.register(NodeSpec("select_target", "Pilih Target", "Data", ["classification", "regression"], [PortSpec("table", "table")], [PortSpec("table", "table")], _select_target, {"target": ""}))
        self.register(NodeSpec("split", "Split Train/Test", "Validasi", ["classification", "regression", "clustering", "anomaly"], [PortSpec("table", "table")], [PortSpec("train", "table"), PortSpec("test", "table")], _split, {"target": "", "test_size": 0.2, "random_state": 42}))
        self.register(NodeSpec("preprocess", "Preprocessing", "Preprocessing", ["all"], [PortSpec("train", "table"), PortSpec("test", "table")], [PortSpec("train", "table"), PortSpec("test", "table")], _preprocess, {"imputer": "median", "scaler": "standard", "encoder": "onehot"}))
        self.register(NodeSpec("train", "Train", "Model", ["all"], [PortSpec("train", "table"), PortSpec("estimator", "estimator")], [PortSpec("model", "model")], _train, {"target": "", "task": "classification"}))
        self.register(NodeSpec("evaluate", "Evaluasi", "Evaluasi", ["all"], [PortSpec("model", "model"), PortSpec("test", "table")], [PortSpec("metrics", "metrics")], _evaluate, {"task": "classification", "target": ""}))
        self.register(NodeSpec("compare", "Bandingkan", "Evaluasi", ["all"], [PortSpec("metrics", "metrics", True)], [PortSpec("table", "table")], _compare))
        for method_id in ModelRegistry.get_models_for_task("binary_classification", available_only=False):
            spec = method_id
            self.register(NodeSpec("model:" + spec.id, spec.name, spec.category, spec.task_types, [], [PortSpec("estimator", "estimator")], lambda inputs, p, context, mid=spec.id: _make_estimator(mid, p), {}, spec.check_package, spec.requires_scaling, spec.max_recommended_rows))
        for task in ("regression", "clustering", "anomaly_detection"):
            for model_spec in ModelRegistry.get_models_for_task(task, available_only=False):
                if "model:" + model_spec.id in self.specs:
                    continue
                self.register(NodeSpec("model:" + model_spec.id, model_spec.name, model_spec.category, model_spec.task_types, [], [PortSpec("estimator", "estimator")], lambda inputs, p, context, mid=model_spec.id: _make_estimator(mid, p), {}, model_spec.check_package, model_spec.requires_scaling, model_spec.max_recommended_rows))
        self.register(NodeSpec("ensemble:voting", "Voting", "Ensemble", ["classification", "regression"], [PortSpec("estimators", "estimator", True)], [PortSpec("estimator", "estimator")], _ensemble, {"kind": "voting", "voting": "hard"}))
        self.register(NodeSpec("ensemble:stacking", "Stacking", "Ensemble", ["classification", "regression"], [PortSpec("estimators", "estimator", True)], [PortSpec("estimator", "estimator")], _ensemble, {"kind": "stacking"}))
        self.register(NodeSpec("ensemble:bagging", "Bagging", "Ensemble", ["classification", "regression"], [PortSpec("estimators", "estimator", True)], [PortSpec("estimator", "estimator")], _ensemble, {"kind": "bagging"}))
        from app.ml.graph.extra import register_extra
        register_extra(self)
        try:
            from app.ml.graph.node_registry import get_node_registry
            for spec_id, entry in get_node_registry().items():
                if spec_id not in self.specs:
                    self.register(entry.to_node_spec())
        except Exception as exc:
            pass

    def add_node(self, spec_id: str, node_id: str | None = None, parameters: dict[str, Any] | None = None, position=(0.0, 0.0)) -> Node:
        if spec_id not in self.specs:
            raise GraphError(f"Node tidak dikenal: {spec_id}")
        node_id = node_id or f"n{len(self.nodes) + 1}"
        if node_id in self.nodes:
            raise GraphError(f"ID node sudah digunakan: {node_id}")
        spec = self.specs[spec_id]
        merged = dict(spec.parameters)
        if spec_id.startswith("model:"):
            estimator = ModelRegistry.instantiate(spec_id.split(":", 1)[1])
            if estimator is not None:
                supported = estimator.get_params(deep=False)
                for key in ("n_estimators", "max_depth", "C", "kernel", "alpha", "n_neighbors", "n_clusters", "eps", "min_samples", "contamination", "random_state"):
                    if key in supported:
                        merged[key] = supported[key]
                if "class_weight" in supported:
                    merged["class_weight"] = "balanced"
        merged.update(parameters or {})
        node = Node(node_id, spec_id, merged, tuple(position))
        self.nodes[node_id] = node
        return node

    def connect(self, source: str, output: str, target: str, input_name: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise GraphError("Node sumber atau tujuan tidak ditemukan.")
        source_port = next((p for p in self.specs[self.nodes[source].spec_id].outputs if p.name == output), None)
        target_port = next((p for p in self.specs[self.nodes[target].spec_id].inputs if p.name == input_name), None)
        if not source_port or not target_port:
            raise GraphError("Port sumber atau tujuan tidak ditemukan.")
        if source_port.type != target_port.type:
            raise GraphError(f"Tipe port tidak cocok: {source_port.type} → {target_port.type}.")
        existing = [c for c in self.connections if c[2] == target and c[3] == input_name]
        if existing and not target_port.multiple:
            raise GraphError(f"Port '{input_name}' hanya menerima satu koneksi.")
        edge = (source, output, target, input_name)
        if edge in self.connections:
            return
        self.connections.append(edge)
        try:
            self.topological_order()
        except GraphError:
            self.connections.remove(edge)
            raise

    def topological_order(self) -> list[str]:
        indegree = {node_id: 0 for node_id in self.nodes}
        downstream = {node_id: [] for node_id in self.nodes}
        for source, _, target, _ in self.connections:
            indegree[target] += 1
            downstream[source].append(target)
        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        result = []
        while queue:
            current = queue.pop(0)
            result.append(current)
            for target in downstream[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(result) != len(self.nodes):
            raise GraphError("Koneksi membentuk loop.")
        return result

    def run(self, dataframe: pd.DataFrame | None = None, progress: Callable[[str, str], None] | None = None) -> dict[str, dict[str, Any]]:
        self._validate_preprocessing_order()
        results: dict[str, dict[str, Any]] = {}
        for node_id in self.topological_order():
            node = self.nodes[node_id]
            spec = self.specs[node.spec_id]
            if spec.check_package and importlib.util.find_spec(spec.check_package) is None:
                package_name = {"imblearn": "imbalanced-learn"}.get(spec.check_package, spec.check_package)
                raise GraphError(f"Library '{package_name}' belum terpasang")
            inputs: dict[str, list[Any]] = {}
            upstream_hashes = []
            for source, output, target, input_name in self.connections:
                if target == node_id:
                    value = results[source][output]
                    inputs.setdefault(input_name, []).append(value)
                    upstream_hashes.append(_stable_hash(value))
            normalized = {k: (v if any(p.name == k and p.multiple for p in spec.inputs) else v[0]) for k, v in inputs.items()}
            context = {"dataframe": dataframe, "engine": self}
            external_hash = _stable_hash(dataframe) if spec.id == "load_dataset" else ""
            key = hashlib.sha256((json.dumps(node.parameters, sort_keys=True, default=str) + "|" + external_hash + "|" + "|".join(upstream_hashes)).encode()).hexdigest()
            if node_id in self._cache and self._cache[node_id][0] == key:
                results[node_id] = self._cache[node_id][1]
                node.status = "selesai"
                continue
            node.status = "berjalan"
            if progress:
                progress(node_id, node.status)
            try:
                output_values = spec.factory(normalized, node.parameters, context)
                self._cache[node_id] = (key, output_values)
                results[node_id] = output_values
                node.status = "selesai"
            except Exception as exc:
                node.status = "error"
                raise GraphError(f"{spec.name}: {exc}") from exc
            if progress:
                progress(node_id, node.status)
        return results

    def detect_leakage(self) -> list[dict[str, Any]]:
        """Detect data leakage where preprocessing or balancing occurs before train/test split."""
        leakages = []
        has_split = any(n.spec_id in {"split", "time_split"} for n in self.nodes.values())
        for source, output, target, input_name in self.connections:
            target_node = self.nodes.get(target)
            source_node = self.nodes.get(source)
            if not target_node or not source_node:
                continue
            target_spec = self.specs.get(target_node.spec_id)
            if not target_spec:
                continue

            if source_node.spec_id in {"load_dataset", "select_target"}:
                if target_spec.category == "Preprocessing":
                    leakages.append({
                        "node_id": target,
                        "spec_id": target_node.spec_id,
                        "name": target_spec.name,
                        "category": target_spec.category,
                        "message": (
                            f"Node preprocessing '{target}' harus menerima data dari node Split Train/Test. "
                            f"Data leakage terdeteksi: node '{target}' ({target_spec.name}) dipasang sebelum Split."
                        ),
                        "suggested_fix": "Pindahkan node setelah Split Train/Test (hubungkan ke port 'train').",
                    })
                elif has_split and (
                    target_spec.category in {"Transformer", "Data Tidak Seimbang"}
                    or any(t in target_spec.id for t in ("scale", "smote", "pca", "variance", "select_k", "outlier"))
                ):
                    leakages.append({
                        "node_id": target,
                        "spec_id": target_node.spec_id,
                        "name": target_spec.name,
                        "category": target_spec.category,
                        "message": (
                            f"Data leakage terdeteksi: node '{target}' ({target_spec.name}) dipasang sebelum Split Train/Test. "
                            "Preprocessing dan balancing data harus menerima data dari node Split agar informasi data uji tidak bocor."
                        ),
                        "suggested_fix": "Pindahkan node setelah Split Train/Test (hubungkan ke port 'train').",
                    })
        return leakages

    def auto_fix_leakage(self) -> int:
        """Auto-fix data leakage by rewiring preprocessing/balancing nodes to receive data from split."""
        leakages = self.detect_leakage()
        if not leakages:
            return 0
        split_node_id = next((nid for nid, n in self.nodes.items() if n.spec_id in {"split", "time_split"}), None)
        if not split_node_id:
            return 0
        fixes_applied = 0
        for item in leakages:
            target_id = item["node_id"]
            self.connections = [
                c for c in self.connections
                if not (c[2] == target_id and self.nodes[c[0]].spec_id in {"load_dataset", "select_target"})
            ]
            target_spec = self.specs[self.nodes[target_id].spec_id]
            input_names = [p.name for p in target_spec.inputs]
            if "train" in input_names:
                self.connect(split_node_id, "train", target_id, "train")
            elif input_names:
                self.connect(split_node_id, "train", target_id, input_names[0])

            if "test" in input_names:
                self.connect(split_node_id, "test", target_id, "test")

            # Redirect downstream nodes connecting to split to now connect to target_id
            target_out_names = [p.name for p in target_spec.outputs]
            if "train" in target_out_names:
                new_conns = []
                for c in self.connections:
                    if c[0] == split_node_id and c[1] == "train" and c[2] != target_id:
                        new_conns.append([target_id, "train", c[2], c[3]])
                    elif c[0] == split_node_id and c[1] == "test" and c[2] != target_id and "test" in target_out_names:
                        new_conns.append([target_id, "test", c[2], c[3]])
                    else:
                        new_conns.append(c)
                self.connections = new_conns

            target_node = self.nodes[target_id]
            if "target" not in target_node.parameters:
                for other_n in self.nodes.values():
                    if other_n.parameters.get("target"):
                        target_node.parameters["target"] = other_n.parameters["target"]
                        break

            fixes_applied += 1
        return fixes_applied

    def validate_graph(self) -> dict[str, Any]:
        """Validate graph topology, port types, and data leakage."""
        errors = []
        warnings = []
        try:
            self.topological_order()
        except GraphError as ge:
            errors.append(str(ge))

        leakages = self.detect_leakage()
        for leak in leakages:
            warnings.append(leak["message"])

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "leakages": leakages,
        }

    def _validate_preprocessing_order(self) -> None:
        leakages = self.detect_leakage()
        self.last_warnings = [item["message"] for item in leakages]
        for leak in leakages:
            if leak.get("category") == "Preprocessing" or getattr(self, "strict_leakage", False):
                raise GraphError(leak["message"])

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "nodes": [{"id": n.id, "spec_id": n.spec_id, "parameters": n.parameters, "position": n.position} for n in self.nodes.values()], "connections": self.connections}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GraphEngine":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls()
        for raw in data.get("nodes", []):
            graph.add_node(raw["spec_id"], raw["id"], raw.get("parameters"), tuple(raw.get("position", (0, 0))))
        for edge in data.get("connections", []):
            graph.connect(*edge)
        return graph


def _stable_hash(value: Any) -> str:
    try:
        if isinstance(value, pd.DataFrame):
            return hashlib.sha256(pd.util.hash_pandas_object(value, index=True).values.tobytes() + str(tuple(value.columns)).encode()).hexdigest()
        return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
    except Exception:
        return hashlib.sha256(repr(value).encode()).hexdigest()


def _select_target(inputs, params, context):
    frame = inputs["table"]
    if params.get("target") not in frame.columns:
        raise ValueError(f"Target '{params.get('target')}' tidak ditemukan.")
    return {"table": frame}


def _split(inputs, params, context):
    from sklearn.model_selection import train_test_split
    frame = inputs["table"]
    target = params.get("target")
    if target and target in frame.columns:
        stratify = frame[target] if str(params.get("task", "classification")).endswith("classification") and frame[target].nunique() < 20 else None
        train, test = train_test_split(frame, test_size=float(params.get("test_size", .2)), random_state=int(params.get("random_state", 42)), stratify=stratify)
    else:
        train, test = train_test_split(frame, test_size=float(params.get("test_size", .2)), random_state=int(params.get("random_state", 42)))
    return {"train": train, "test": test}


def _preprocess(inputs, params, context):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
    train = inputs["train"].copy()
    test = inputs.get("test")
    if test is not None:
        test = test.copy()
    target = params.get("target")
    if not target and context and "engine" in context:
        for other in context["engine"].nodes.values():
            if other.parameters.get("target"):
                target = other.parameters.get("target")
                break
    y_train = train.pop(target) if target and target in train.columns else None
    y_test = test.pop(target) if test is not None and target and target in test.columns else None
    numeric = train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in train.columns if c not in numeric]
    scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}.get(params.get("scaler", "standard"), StandardScaler)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy=params.get("imputer", "median"))), ("scaler", scaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformer = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])
    a = transformer.fit_transform(train)
    cols = [f"fitur_{i}" for i in range(a.shape[1])]
    train_out = pd.DataFrame(a, columns=cols, index=train.index)
    if y_train is not None:
        train_out[target] = y_train

    if test is not None:
        b = transformer.transform(test)
        test_out = pd.DataFrame(b, columns=cols, index=test.index)
        if y_test is not None:
            test_out[target] = y_test
    else:
        test_out = None
    return {"train": train_out, "test": test_out}


def _make_estimator(method_id, params):
    estimator = ModelRegistry.instantiate(method_id, int(params.get("random_state", 42)))
    if estimator is None:
        raise ValueError(f"Model '{method_id}' tidak tersedia.")
    for key, value in params.items():
        if key != "random_state" and key in estimator.get_params(deep=True):
            estimator.set_params(**{key: value})
    return {"estimator": estimator}


def _train(inputs, params, context):
    from sklearn.base import clone
    frame, estimator = inputs["train"], inputs["estimator"]
    task = params.get("task", "classification")
    target = params.get("target")
    if task in {"clustering", "anomaly", "anomaly_detection"}:
        X = frame.drop(columns=[target]) if target and target in frame else frame
        model = clone(estimator)
        model.fit(X)
        return {"model": {"estimator": model, "target": target or None, "task": task}}
    target = target or frame.columns[-1]
    if target not in frame.columns:
        raise ValueError(f"Target '{target}' tidak ditemukan pada data train.")
    model = clone(estimator)
    model.fit(frame.drop(columns=[target]), frame[target])
    return {"model": {"estimator": model, "target": target, "task": task}}


def _evaluate(inputs, params, context):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        mean_absolute_error, mean_squared_error, r2_score,
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
        balanced_accuracy_score, confusion_matrix,
    )
    model, frame = inputs["model"], inputs["test"]
    target = params.get("target") or model.get("target")
    task = params.get("task", model.get("task", "classification"))
    random_state = int(params.get("random_state", 42))

    if task in {"clustering", "anomaly", "anomaly_detection"}:
        X = frame.drop(columns=[target]) if target and target in frame else frame
        estimator = model["estimator"]
        labels = estimator.predict(X) if hasattr(estimator, "predict") else estimator.fit_predict(X)
        unique = set(labels)
        metrics = {
            "cluster_count": len(unique),
            "noise_count": int(sum(label == -1 for label in labels)),
            "random_state": random_state,
        }
        if 1 < len(unique) < len(labels):
            metrics.update({
                "silhouette": float(silhouette_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
            })
        return {"metrics": {"model": str(estimator), "metrics": metrics}}

    if not target or target not in frame:
        raise ValueError(f"Target '{target}' tidak ditemukan pada data test.")
    X, y = frame.drop(columns=[target]), frame[target]
    pred = model["estimator"].predict(X)

    if task == "regression":
        metrics = {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "random_state": random_state,
        }
    else:
        # Detect class imbalance
        y_counts = y.value_counts()
        imbalance_ratio = float(y_counts.min() / y_counts.max()) if len(y_counts) > 1 and y_counts.max() > 0 else 1.0
        is_imbalanced = imbalance_ratio < 0.35

        f1_m = float(f1_score(y, pred, average="macro", zero_division=0))
        f1_w = float(f1_score(y, pred, average="weighted", zero_division=0))
        acc = float(accuracy_score(y, pred))
        bal_acc = float(balanced_accuracy_score(y, pred))

        try:
            cm = confusion_matrix(y, pred).tolist()
        except Exception:
            cm = []

        metrics = {
            "accuracy": acc,
            "f1": f1_m if is_imbalanced else f1_w,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "balanced_accuracy": bal_acc,
            "precision": float(precision_score(y, pred, average="macro" if is_imbalanced else "weighted", zero_division=0)),
            "recall": float(recall_score(y, pred, average="macro" if is_imbalanced else "weighted", zero_division=0)),
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "confusion_matrix": cm,
            "random_state": random_state,
        }

    return {"metrics": {"model": str(model["estimator"]), "metrics": metrics}}


def _ensemble(inputs, params, context):
    from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, BaggingClassifier, BaggingRegressor
    from sklearn.base import is_regressor
    estimators = inputs.get("estimators", [])
    if len(estimators) < 2:
        raise ValueError("Ensemble membutuhkan minimal dua estimator.")
    named = [(f"model_{i}", e) for i, e in enumerate(estimators)]
    kind = params.get("kind", "voting")
    regression = all(is_regressor(e) for e in estimators)
    if kind == "voting":
        model = VotingRegressor(named) if regression else VotingClassifier(named, voting=params.get("voting", "hard"))
    elif kind == "stacking":
        model = StackingRegressor(named) if regression else StackingClassifier(named)
    else:
        model = BaggingRegressor(estimator=estimators[0]) if regression else BaggingClassifier(estimator=estimators[0])
    return {"estimator": model}


def _compare(inputs, params, context):
    rows = []
    for item in inputs.get("metrics", []):
        rows.append({"model": item.get("model", "Model"), **item.get("metrics", {})})
    return {"table": pd.DataFrame(rows)}
