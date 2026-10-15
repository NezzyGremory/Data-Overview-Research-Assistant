import pytest

from app.ml.graph import GraphEngine
from app.ml.graph.node_registry import (
    get_node_entry,
    get_node_registry,
    get_registry_summary,
)
from app.ml.graph.plugins import (
    MLPlugin,
    PluginEstimatorDef,
    get_plugin,
    list_plugins,
    register_plugin,
)


def test_auto_discovery_finds_many_estimators():
    registry = get_node_registry()
    assert len(registry) >= 30
    categories = {entry.category for entry in registry.values()}
    assert "Classification" in categories
    assert "Regression" in categories
    assert "Clustering" in categories
    assert "Transformer" in categories


def test_params_and_types_extracted_correctly():
    # Test random forest classifier
    entry = get_node_entry("sklearn:randomforestclassifier")
    assert entry is not None
    assert entry.name == "Random Forest Classifier"
    assert entry.category == "Classification"
    assert "classification" in entry.task_types

    # Check params
    assert "n_estimators" in entry.params
    assert entry.params["n_estimators"]["type"] == "int"
    assert entry.params["n_estimators"]["default"] == 100

    assert "criterion" in entry.params
    assert entry.params["criterion"]["type"] == "str"

    assert "max_depth" in entry.params
    assert entry.default_params["n_estimators"] == 100

    # Test requires field
    assert len(entry.requires) >= 1
    assert entry.requires[0]["package"] == "scikit-learn"
    assert "version" in entry.requires[0]


def test_plugins_registered_and_configured():
    plugins = list_plugins()
    plugin_names = {p.name for p in plugins}
    assert "xgboost" in plugin_names
    assert "lightgbm" in plugin_names
    assert "imbalanced-learn" in plugin_names

    xgb = get_plugin("xgboost")
    assert xgb is not None
    assert xgb.package == "xgboost"
    assert len(xgb.estimators) == 2  # classifier and regressor

    # Plugin entries exist in master node registry
    registry = get_node_registry()
    assert "plugin:xgboost:classifier" in registry
    xgb_node = registry["plugin:xgboost:classifier"]
    assert xgb_node.is_plugin is True
    assert xgb_node.plugin_name == "xgboost"
    assert xgb_node.requires[0]["package"] == "xgboost"
    assert "n_estimators" in xgb_node.default_params


def test_custom_plugin_registration():
    custom_plugin = MLPlugin(
        name="test_plugin",
        package="test-ml-pkg",
        min_version=">=1.0.0",
        import_name="test_ml_pkg",
        estimators=[
            PluginEstimatorDef(
                id="plugin:test:dummy",
                name="Dummy Custom Classifier",
                class_name="DummyClassifier",
                module_path="sklearn.dummy",
                category="Classification",
                task_types=["classification"],
                default_params={"strategy": "prior"},
                param_types={"strategy": "str"},
            )
        ],
    )
    register_plugin(custom_plugin)
    registry = get_node_registry(force_reload=True)
    assert "plugin:test:dummy" in registry
    dummy_node = registry["plugin:test:dummy"]
    assert dummy_node.requires[0]["package"] == "test-ml-pkg"
    assert dummy_node.name == "Dummy Custom Classifier"


def test_graph_engine_integration_with_registry():
    engine = GraphEngine()
    # Auto-discovered nodes should be in engine specs
    assert "sklearn:randomforestclassifier" in engine.specs
    spec = engine.specs["sklearn:randomforestclassifier"]
    assert spec.category == "Classification"
    assert spec.parameters.get("n_estimators") == 100

    # Add node to graph and verify parameters
    node = engine.add_node("sklearn:randomforestclassifier", "rf_test", {"n_estimators": 50})
    assert node.parameters["n_estimators"] == 50

    # Test factory instantiation
    factory_out = spec.factory({}, node.parameters, {})
    assert "estimator" in factory_out
    assert factory_out["estimator"].n_estimators == 50


def test_registry_summary_serializable():
    summary = get_registry_summary()
    assert isinstance(summary, list)
    assert len(summary) >= 30
    first = summary[0]
    assert "id" in first
    assert "name" in first
    assert "category" in first
    assert "params" in first
    assert "requires" in first
