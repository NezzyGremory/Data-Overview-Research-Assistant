import pandas as pd
import pytest

from app.ml.graph import GraphEngine, GraphError


def test_feature_node_fits_train_only_and_split_time_is_ordered():
    graph = GraphEngine()
    graph.add_node("load_dataset", "load")
    graph.nodes["load"].parameters["dataframe"] = None
    graph.add_node("time_split", "split", {"time_column": "time", "test_size": .25})
    graph.connect("load", "table", "split", "table")
    result = graph.run(pd.DataFrame({"time": [4, 1, 3, 2], "y": [4., 1., 3., 2.]}))
    assert list(result["split"]["train"]["time"]) == [1, 2, 3]
    assert list(result["split"]["test"]["time"]) == [4]

    transform = GraphEngine()
    transform.add_node("load_dataset", "load")
    transform.add_node("split", "split")
    transform.add_node("feature:scale", "scale", {"target": "y"})
    transform.connect("load", "table", "split", "table")
    transform.connect("split", "train", "scale", "train")
    transform.connect("split", "test", "scale", "test")
    spec = transform.specs["feature:scale"]
    result = spec.factory({"train": pd.DataFrame({"x": [0., 1., 2.], "y": [0, 0, 1]}), "test": pd.DataFrame({"x": [100.], "y": [1]})}, {"operation": "scale", "method": "standard", "target": "y"}, {})
    assert abs(result["train"]["fitur_0"].mean()) < 1e-7
    assert result["test"]["fitur_0"].iloc[0] > 10


def test_validation_tuning_and_forecast_nodes_run():
    frame = pd.DataFrame({"x": range(30), "y": [0] * 15 + [1] * 15})
    graph = GraphEngine()
    graph.add_node("load_dataset", "load")
    graph.add_node("cross_validate", "cv", {"target": "y", "cv": 2})
    graph.add_node("model:logistic_regression", "model")
    graph.connect("load", "table", "cv", "table")
    graph.connect("model", "estimator", "cv", "estimator")
    assert "mean_score" in graph.run(frame)["cv"]["metrics"]["metrics"]

    tuner = GraphEngine()
    tuner.add_node("load_dataset", "load")
    tuner.add_node("model:logistic_regression", "model")
    tuner.add_node("grid_search", "search", {"target": "y", "param_grid": {"C": [.1, 1.]}, "cv": 2})
    tuner.connect("load", "table", "search", "table")
    tuner.connect("model", "estimator", "search", "estimator")
    assert "best_score" in tuner.run(frame)["search"]["metrics"]["metrics"]

    forecast = GraphEngine()
    forecast.add_node("load_dataset", "load")
    forecast.add_node("time_split", "split", {"time_column": "time", "test_size": .2})
    forecast.add_node("forecast", "forecast", {"target": "y", "method": "naive"})
    forecast.connect("load", "table", "split", "table")
    forecast.connect("split", "train", "forecast", "train")
    forecast.connect("split", "test", "forecast", "test")
    scores = forecast.run(pd.DataFrame({"time": range(20), "y": [float(i) for i in range(20)]}))
    assert set(scores["forecast"]["metrics"]["metrics"]) == {"mae", "rmse", "mape"}


def test_optional_package_gives_clear_error(monkeypatch):
    graph = GraphEngine()
    graph.add_node("optuna", "tuner")
    monkeypatch.setattr("app.ml.graph.engine.importlib.util.find_spec", lambda _: None)
    with pytest.raises(GraphError, match="Library 'optuna' belum terpasang"):
        graph.run()


def test_smote_input_contract_is_train_only():
    spec = GraphEngine().specs["smote"]
    assert [port.name for port in spec.inputs] == ["train"]
    graph = GraphEngine()
    graph.add_node("load_dataset", "load")
    graph.add_node("smote", "smote", {"target": "y"})
    graph.connect("load", "table", "smote", "train")
    frame = pd.DataFrame({"x": range(20), "y": [0] * 16 + [1] * 4})
    try:
        import imblearn  # noqa: F401
    except ImportError:
        with pytest.raises(GraphError, match="Library 'imbalanced-learn' belum terpasang"):
            graph.run(frame)
    else:
        assert len(graph.run(frame)["smote"]["train"]) >= len(frame)


def test_additional_sklearn_model_node_runs():
    graph = GraphEngine()
    graph.add_node("extra_model:ridge_classifier", "ridge")
    assert graph.run()["ridge"]["estimator"].__class__.__name__ == "RidgeClassifier"

