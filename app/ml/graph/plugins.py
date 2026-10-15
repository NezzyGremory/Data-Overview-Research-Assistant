"""Plugin mechanism for external ML libraries (XGBoost, LightGBM, imbalanced-learn, etc.)."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class PluginEstimatorDef:
    """Definition of an estimator provided by a plugin."""
    id: str
    name: str
    class_name: str
    module_path: str
    category: str
    task_types: List[str]
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_types: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class MLPlugin:
    """Plugin definition for an external library."""
    name: str
    package: str  # PyPI package name for pip
    min_version: str
    import_name: str  # python import name
    description: str = ""
    estimators: List[PluginEstimatorDef] = field(default_factory=list)

    def is_installed(self) -> bool:
        return importlib.util.find_spec(self.import_name) is not None

    def get_version(self) -> Optional[str]:
        if not self.is_installed():
            return None
        try:
            mod = importlib.import_module(self.import_name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "unknown"


# Central plugin registry
_PLUGIN_REGISTRY: Dict[str, MLPlugin] = {}


def register_plugin(plugin: MLPlugin) -> None:
    """Register an external ML plugin."""
    _PLUGIN_REGISTRY[plugin.name] = plugin


def get_plugin(name: str) -> Optional[MLPlugin]:
    return _PLUGIN_REGISTRY.get(name)


def list_plugins() -> List[MLPlugin]:
    return list(_PLUGIN_REGISTRY.values())


# =====================================================================
# DEFAULT BUILT-IN PLUGINS
# =====================================================================

def _register_default_plugins() -> None:
    # 1. XGBoost Plugin
    register_plugin(MLPlugin(
        name="xgboost",
        package="xgboost",
        min_version=">=1.7.0",
        import_name="xgboost",
        description="Scalable, Portable and Distributed Gradient Boosting",
        estimators=[
            PluginEstimatorDef(
                id="plugin:xgboost:classifier",
                name="XGBoost Classifier",
                class_name="XGBClassifier",
                module_path="xgboost",
                category="Classification",
                task_types=["binary_classification", "multiclass_classification", "classification"],
                default_params={
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "random_state": 42,
                },
                param_types={
                    "n_estimators": "int",
                    "max_depth": "int",
                    "learning_rate": "float",
                    "subsample": "float",
                    "random_state": "int",
                },
                description="Extreme Gradient Boosting decision tree classifier.",
            ),
            PluginEstimatorDef(
                id="plugin:xgboost:regressor",
                name="XGBoost Regressor",
                class_name="XGBRegressor",
                module_path="xgboost",
                category="Regression",
                task_types=["regression"],
                default_params={
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "random_state": 42,
                },
                param_types={
                    "n_estimators": "int",
                    "max_depth": "int",
                    "learning_rate": "float",
                    "subsample": "float",
                    "random_state": "int",
                },
                description="Extreme Gradient Boosting decision tree regressor.",
            ),
        ],
    ))

    # 2. LightGBM Plugin
    register_plugin(MLPlugin(
        name="lightgbm",
        package="lightgbm",
        min_version=">=3.3.0",
        import_name="lightgbm",
        description="Fast, highly efficient gradient boosting framework",
        estimators=[
            PluginEstimatorDef(
                id="plugin:lightgbm:classifier",
                name="LightGBM Classifier",
                class_name="LGBMClassifier",
                module_path="lightgbm",
                category="Classification",
                task_types=["binary_classification", "multiclass_classification", "classification"],
                default_params={
                    "n_estimators": 100,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "verbose": -1,
                },
                param_types={
                    "n_estimators": "int",
                    "max_depth": "int",
                    "num_leaves": "int",
                    "learning_rate": "float",
                    "random_state": "int",
                    "verbose": "int",
                },
                description="LightGBM tree-based classification algorithm.",
            ),
            PluginEstimatorDef(
                id="plugin:lightgbm:regressor",
                name="LightGBM Regressor",
                class_name="LGBMRegressor",
                module_path="lightgbm",
                category="Regression",
                task_types=["regression"],
                default_params={
                    "n_estimators": 100,
                    "max_depth": -1,
                    "num_leaves": 31,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "verbose": -1,
                },
                param_types={
                    "n_estimators": "int",
                    "max_depth": "int",
                    "num_leaves": "int",
                    "learning_rate": "float",
                    "random_state": "int",
                    "verbose": "int",
                },
                description="LightGBM tree-based regression algorithm.",
            ),
        ],
    ))

    # 3. Imbalanced-learn Plugin
    register_plugin(MLPlugin(
        name="imbalanced-learn",
        package="imbalanced-learn",
        min_version=">=0.10.0",
        import_name="imblearn",
        description="Dataset balancing and re-sampling tools for imbalanced classification",
        estimators=[
            PluginEstimatorDef(
                id="plugin:imblearn:smote",
                name="SMOTE",
                class_name="SMOTE",
                module_path="imblearn.over_sampling",
                category="Data Tidak Seimbang",
                task_types=["classification", "binary_classification", "multiclass_classification"],
                default_params={
                    "k_neighbors": 5,
                    "random_state": 42,
                },
                param_types={
                    "k_neighbors": "int",
                    "random_state": "int",
                },
                description="Synthetic Minority Over-sampling Technique.",
            ),
            PluginEstimatorDef(
                id="plugin:imblearn:random_under_sampler",
                name="Random Under Sampler",
                class_name="RandomUnderSampler",
                module_path="imblearn.under_sampling",
                category="Data Tidak Seimbang",
                task_types=["classification", "binary_classification", "multiclass_classification"],
                default_params={
                    "random_state": 42,
                },
                param_types={
                    "random_state": "int",
                },
                description="Under-sample the majority class(es) randomly.",
            ),
            PluginEstimatorDef(
                id="plugin:imblearn:adasyn",
                name="ADASYN",
                class_name="ADASYN",
                module_path="imblearn.over_sampling",
                category="Data Tidak Seimbang",
                task_types=["classification", "binary_classification", "multiclass_classification"],
                default_params={
                    "n_neighbors": 5,
                    "random_state": 42,
                },
                param_types={
                    "n_neighbors": "int",
                    "random_state": "int",
                },
                description="Adaptive Synthetic sampling approach for imbalanced datasets.",
            ),
        ],
    ))


# Initialize default plugins on import
_register_default_plugins()

