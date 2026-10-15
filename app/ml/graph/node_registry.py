"""Automated Node Registry for Machine Learning Graph Engine.

Discovers estimators and transformers dynamically from scikit-learn via
sklearn.utils.all_estimators() and integrates external plugins (XGBoost,
LightGBM, imbalanced-learn) without hardcoding them in the core engine.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.utils import all_estimators

from app.ml.graph.engine import GraphError, NodeSpec, PortSpec
from app.ml.graph.plugins import list_plugins


@dataclass
class ParameterDef:
    """Detailed metadata for a node parameter."""
    name: str
    type: str  # "int", "float", "str", "bool", "list", "dict", "any"
    default: Any
    options: List[Any] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "options": self.options,
            "description": self.description,
        }


@dataclass
class NodeRegistryEntry:
    """Entry in the master node registry."""
    id: str
    name: str
    category: str  # "Classification", "Regression", "Clustering", "Transformer", "Data Tidak Seimbang"
    task_types: List[str]
    params: Dict[str, Dict[str, Any]]  # param_name -> ParameterDef dict
    default_params: Dict[str, Any]  # param_name -> default_value
    requires: List[Dict[str, str]]  # list of {"package": ..., "version": ...}
    inputs: List[PortSpec]
    outputs: List[PortSpec]
    estimator_cls: Optional[Type] = None
    factory: Optional[Callable] = None
    check_package: Optional[str] = None
    requires_scaling: bool = False
    max_recommended_rows: int = 100_000
    description: str = ""
    is_plugin: bool = False
    plugin_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "task_types": self.task_types,
            "params": self.params,
            "default_params": self.default_params,
            "requires": self.requires,
            "inputs": [{"name": p.name, "type": p.type, "multiple": p.multiple} for p in self.inputs],
            "outputs": [{"name": p.name, "type": p.type, "multiple": p.multiple} for p in self.outputs],
            "description": self.description,
            "is_plugin": self.is_plugin,
            "plugin_name": self.plugin_name,
            "requires_scaling": self.requires_scaling,
            "max_recommended_rows": self.max_recommended_rows,
        }

    def to_node_spec(self) -> NodeSpec:
        """Convert this registry entry into a GraphEngine NodeSpec."""
        return NodeSpec(
            id=self.id,
            name=self.name,
            category=self.category,
            task_types=self.task_types,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            factory=self.factory,
            parameters=dict(self.default_params),
            check_package=self.check_package,
            requires_scaling=self.requires_scaling,
            max_recommended_rows=self.max_recommended_rows,
            requires=list(self.requires),
        )


# =====================================================================
# PARAMETER EXTRACTION UTILITIES
# =====================================================================

def _infer_type(name: str, default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    if isinstance(default, (list, tuple)):
        return "list"
    if isinstance(default, dict):
        return "dict"

    # Name-based heuristic when default is None
    name_lower = name.lower()
    if any(k in name_lower for k in ("n_estimators", "max_depth", "n_clusters", "n_neighbors", "n_iter", "max_iter", "random_state", "min_samples", "degree", "cv")):
        return "int"
    if any(k in name_lower for k in ("alpha", "learning_rate", "eps", "gamma", "tol", "subsample", "threshold", "c")):
        return "float"
    if any(k in name_lower for k in ("kernel", "criterion", "loss", "metric", "algorithm", "penalty", "scoring", "strategy")):
        return "str"
    return "any"


def _format_class_name(class_name: str) -> str:
    """Format CamelCase to human readable title (e.g. RandomForestClassifier -> Random Forest Classifier)."""
    # Handle acronyms (e.g. SGD, PCA, MLP, SVR, SVC, GMM)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', class_name)
    spaced = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', spaced)
    return spaced.strip()


def _extract_parameters(cls: Type) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Extract parameters, types, and defaults from an estimator class __init__."""
    params_meta: Dict[str, Dict[str, Any]] = {}
    default_params: Dict[str, Any] = {}

    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return params_meta, default_params

    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        default_val = None if param.default is inspect.Parameter.empty else param.default
        param_type = _infer_type(param.name, default_val)

        meta = ParameterDef(
            name=param.name,
            type=param_type,
            default=default_val,
        ).to_dict()

        params_meta[param.name] = meta
        default_params[param.name] = default_val

    return params_meta, default_params


# =====================================================================
# FACTORY BUILDERS
# =====================================================================

def _make_model_factory(cls: Type, default_params: Dict[str, Any]) -> Callable:
    """Create a GraphEngine factory for an estimator node."""
    def factory(inputs: Dict[str, Any], params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(default_params)
        merged.update(params)

        valid_kwargs = {}
        try:
            sig = inspect.signature(cls.__init__)
            valid_names = set(sig.parameters.keys())
        except Exception:
            valid_names = set()

        for k, v in merged.items():
            if valid_names and k not in valid_names:
                continue
            if v is not None:
                valid_kwargs[k] = v

        try:
            instance = cls(**valid_kwargs)
        except Exception as exc:
            # Fallback to no-argument init if parameter mismatch occurs
            try:
                instance = cls()
            except Exception:
                raise GraphError(f"Gagal menginisialisasi {cls.__name__}: {exc}") from exc

        return {"estimator": instance}

    return factory


def _make_transformer_factory(cls: Type, default_params: Dict[str, Any]) -> Callable:
    """Create a GraphEngine factory for a transformer node (train/test split aware)."""
    def factory(inputs: Dict[str, Any], params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        train = inputs.get("train")
        test = inputs.get("test")
        if train is None:
            raise GraphError(f"{cls.__name__} memerlukan input 'train'.")

        train_copy = train.copy()
        test_copy = test.copy() if test is not None else None

        target = params.get("target")
        y_train = train_copy.pop(target) if target and target in train_copy else None
        y_test = test_copy.pop(target) if target and test_copy is not None and target in test_copy else None

        merged = dict(default_params)
        merged.update(params)

        valid_kwargs = {}
        try:
            sig = inspect.signature(cls.__init__)
            valid_names = set(sig.parameters.keys())
        except Exception:
            valid_names = set()

        for k, v in merged.items():
            if k in ("target", "test_size", "random_state") and (not valid_names or k not in valid_names):
                continue
            if valid_names and k not in valid_names:
                continue
            if v is not None:
                valid_kwargs[k] = v

        try:
            transformer = cls(**valid_kwargs)
            # Only transform numeric columns if transformer is not text/categorical specialized
            num_cols = train_copy.select_dtypes(include="number").columns.tolist()
            if num_cols and not hasattr(transformer, "sparse_output"):
                transformed_train = transformer.fit_transform(train_copy[num_cols], y_train)
                train_out = train_copy.copy()
                cols = [f"{cls.__name__.lower()}_{i}" for i in range(transformed_train.shape[1])]
                train_out = pd.DataFrame(transformed_train, index=train_copy.index, columns=cols)

                if test_copy is not None:
                    transformed_test = transformer.transform(test_copy[num_cols])
                    test_out = pd.DataFrame(transformed_test, index=test_copy.index, columns=cols)
                else:
                    test_out = None
            else:
                transformed_train = transformer.fit_transform(train_copy, y_train)
                train_out = pd.DataFrame(transformed_train, index=train_copy.index)
                test_out = pd.DataFrame(transformer.transform(test_copy), index=test_copy.index) if test_copy is not None else None

            if y_train is not None:
                train_out[target] = y_train
            if y_test is not None and test_out is not None:
                test_out[target] = y_test

            return {"train": train_out, "test": test_out if test_out is not None else train_out}
        except Exception as exc:
            raise GraphError(f"Transformasi {cls.__name__} gagal: {exc}") from exc

    return factory


# =====================================================================
# REGISTRY CACHE & BUILDER
# =====================================================================

_CACHED_REGISTRY: Optional[Dict[str, NodeRegistryEntry]] = None

# Blacklist of problematic or purely internal meta-estimators
_EXCLUDED_ESTIMATORS = {
    # Meta estimators requiring base estimators in __init__ without default
    "ClassifierChain", "RegressorChain", "MultiOutputClassifier", "MultiOutputRegressor",
    "OneVsOneClassifier", "OneVsRestClassifier", "OutputCodeClassifier", "VotingClassifier",
    "VotingRegressor", "StackingClassifier", "StackingRegressor", "GridSearchCV",
    "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV",
    "RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelector",
    "SelfTrainingClassifier", "CalibratedClassifierCV", "FixedThresholdClassifier",
    "FrozenEstimator", "ColumnTransformer", "FeatureUnion", "Pipeline", "FeatureUnion",
    "IterativeImputer", "SparseCoder", "TargetEncoder",
}


def build_node_registry() -> Dict[str, NodeRegistryEntry]:
    """Dynamically build the node registry from sklearn and registered plugins."""
    registry: Dict[str, NodeRegistryEntry] = {}

    # 1. Scikit-Learn Auto Discovery
    for name, cls in all_estimators():
        if name in _EXCLUDED_ESTIMATORS or name.startswith("_"):
            continue

        # Check for required positional parameters without defaults
        try:
            sig = inspect.signature(cls.__init__)
            req = [
                p.name for p in sig.parameters.values()
                if p.name != "self"
                and p.default == inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            if req:
                continue
        except Exception:
            continue

        # Determine Category and Task Types
        category = "Model"
        task_types = ["all"]
        is_transformer = issubclass(cls, TransformerMixin) and not issubclass(cls, (ClassifierMixin, RegressorMixin, ClusterMixin))

        if issubclass(cls, ClassifierMixin):
            category = "Classification"
            task_types = ["classification", "binary_classification", "multiclass_classification"]
        elif issubclass(cls, RegressorMixin):
            category = "Regression"
            task_types = ["regression"]
        elif issubclass(cls, ClusterMixin):
            category = "Clustering"
            task_types = ["clustering"]
        elif is_transformer:
            category = "Transformer"
            task_types = ["all"]
        else:
            # Check for outlier/anomaly estimators
            if any(term in name.lower() for term in ("outlier", "isolation", "envelope", "novelty")):
                category = "Anomaly Detection"
                task_types = ["anomaly", "anomaly_detection"]
            else:
                continue

        params_meta, default_params = _extract_parameters(cls)

        spec_id = f"sklearn:{cls.__name__.lower()}"
        display_name = _format_class_name(name)

        if is_transformer:
            inputs = [PortSpec("train", "table"), PortSpec("test", "table")]
            outputs = [PortSpec("train", "table"), PortSpec("test", "table")]
            factory = _make_transformer_factory(cls, default_params)
        else:
            inputs = []
            outputs = [PortSpec("estimator", "estimator")]
            factory = _make_model_factory(cls, default_params)

        entry = NodeRegistryEntry(
            id=spec_id,
            name=display_name,
            category=category,
            task_types=task_types,
            params=params_meta,
            default_params=default_params,
            requires=[{"package": "scikit-learn", "version": ">=1.0.0"}],
            inputs=inputs,
            outputs=outputs,
            estimator_cls=cls,
            factory=factory,
            check_package="sklearn",
            description=f"Auto-generated scikit-learn {name} node.",
            is_plugin=False,
            plugin_name="",
        )
        registry[spec_id] = entry

    # 2. Integrate Registered External Plugins
    for plugin in list_plugins():
        for est_def in plugin.estimators:
            # Build requires field
            requires = [{"package": plugin.package, "version": plugin.min_version}]

            # Determine ports
            if est_def.category == "Data Tidak Seimbang":
                inputs = [PortSpec("train", "table")]
                outputs = [PortSpec("train", "table")]
                def make_resampler_factory(est=est_def, pkg=plugin):
                    def factory(inputs: Dict[str, Any], params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                        if not pkg.is_installed():
                            raise GraphError(f"Library '{pkg.package}' belum terpasang.")
                        import importlib
                        mod = importlib.import_module(est.module_path)
                        sampler_cls = getattr(mod, est.class_name)
                        frame = inputs["train"].copy()
                        target = params.get("target")
                        if not target or target not in frame:
                            raise ValueError(f"{est.name} membutuhkan target pada data train.")

                        # Dynamic k_neighbors / n_neighbors for safety on small data
                        kwargs = dict(est.default_params)
                        kwargs.update(params)
                        kwargs.pop("target", None)

                        min_samples = frame[target].value_counts().min()
                        for neighbor_key in ("k_neighbors", "n_neighbors"):
                            if neighbor_key in kwargs:
                                kwargs[neighbor_key] = min(int(kwargs[neighbor_key]), max(1, min_samples - 1))

                        sampler = sampler_cls(**kwargs)
                        X, y = sampler.fit_resample(frame.drop(columns=[target]), frame[target])
                        res = pd.DataFrame(X, columns=frame.drop(columns=[target]).columns)
                        res[target] = y
                        return {"train": res}
                    return factory
                node_factory = make_resampler_factory()
            else:
                inputs = []
                outputs = [PortSpec("estimator", "estimator")]
                def make_plugin_estimator_factory(est=est_def, pkg=plugin):
                    def factory(inputs: Dict[str, Any], params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                        if not pkg.is_installed():
                            raise GraphError(f"Library '{pkg.package}' belum terpasang.")
                        import importlib
                        mod = importlib.import_module(est.module_path)
                        est_cls = getattr(mod, est.class_name)
                        merged = dict(est.default_params)
                        merged.update(params)
                        instance = est_cls(**merged)
                        return {"estimator": instance}
                    return factory
                node_factory = make_plugin_estimator_factory()

            # Formulate params structure
            params_meta = {
                k: ParameterDef(name=k, type=est_def.param_types.get(k, "any"), default=v).to_dict()
                for k, v in est_def.default_params.items()
            }

            entry = NodeRegistryEntry(
                id=est_def.id,
                name=est_def.name,
                category=est_def.category,
                task_types=est_def.task_types,
                params=params_meta,
                default_params=dict(est_def.default_params),
                requires=requires,
                inputs=inputs,
                outputs=outputs,
                factory=node_factory,
                check_package=plugin.import_name,
                description=est_def.description,
                is_plugin=True,
                plugin_name=plugin.name,
            )
            registry[est_def.id] = entry

    return registry


def get_node_registry(force_reload: bool = False) -> Dict[str, NodeRegistryEntry]:
    """Retrieve the cached node registry or build it."""
    global _CACHED_REGISTRY
    if _CACHED_REGISTRY is None or force_reload:
        _CACHED_REGISTRY = build_node_registry()
    return _CACHED_REGISTRY


def get_node_entry(spec_id: str) -> Optional[NodeRegistryEntry]:
    """Retrieve a single registry entry by its spec ID."""
    return get_node_registry().get(spec_id)


def get_registry_summary() -> List[Dict[str, Any]]:
    """Return JSON-serializable list of all registry entries for API/frontend."""
    return [entry.to_dict() for entry in get_node_registry().values()]
