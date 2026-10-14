import pandas as pd
import pytest

from app.ml.graph import GraphEngine, GraphError, NodeSpec, PortSpec


def test_port_types_loop_and_multiple_inputs():
    graph = GraphEngine()
    graph.add_node("load_dataset", "source")
    graph.add_node("train", "train")
    with pytest.raises(GraphError, match="Tipe port"):
        graph.connect("source", "table", "train", "estimator")
    graph.add_node("ensemble:voting", "vote")
    models = [n for n in graph.specs if n.startswith("model:")][:2]
    for i, spec in enumerate(models):
        graph.add_node(spec, f"m{i}")
        graph.connect(f"m{i}", "estimator", "vote", "estimators")
    assert len([e for e in graph.connections if e[2:] == ("vote", "estimators")]) == 2
    graph.register(NodeSpec("test:cycle", "Cycle", "Test", ["all"], [PortSpec("table", "table", True)], [PortSpec("table", "table")], lambda i, p, c: {"table": None}))
    graph.add_node("test:cycle", "target")
    graph.add_node("test:cycle", "target2")
    graph.connect("target", "table", "target2", "table")
    with pytest.raises(GraphError, match="loop"):
        graph.connect("target2", "table", "target", "table")


def test_execution_order_cache_and_json(tmp_path):
    graph = GraphEngine()
    calls = []
    graph.register(NodeSpec("test:step", "Step", "Test", ["all"], [PortSpec("table", "table")], [PortSpec("table", "table")], lambda i, p, c: (calls.append(p["v"]) or {"table": i["table"] + p["v"]}), {"v": 1}))
    graph.add_node("load_dataset", "load")
    graph.add_node("test:step", "a")
    graph.add_node("test:step", "b")
    graph.nodes["b"].parameters["v"] = 2
    graph.connect("load", "table", "a", "table")
    graph.connect("a", "table", "b", "table")
    data = pd.DataFrame({"x": [1]})
    assert graph.topological_order() == ["load", "a", "b"]
    graph.run(data)
    graph.run(data)
    assert calls == [1, 2]
    graph.nodes["b"].parameters["v"] = 3
    graph.run(data)
    assert calls == [1, 2, 3]
    path = tmp_path / "graph.json"
    serializable = GraphEngine()
    serializable.add_node("load_dataset", "load")
    serializable.add_node("select_target", "target", {"target": "y"})
    serializable.connect("load", "table", "target", "table")
    serializable.save(path)
    restored = GraphEngine.load(path)
    assert restored.to_dict() == serializable.to_dict()


def test_preprocessing_must_follow_split():
    graph = GraphEngine()
    graph.add_node("load_dataset", "load")
    graph.add_node("preprocess", "prep")
    graph.connect("load", "table", "prep", "train")
    with pytest.raises(GraphError, match="harus menerima data dari node Split"):
        graph.run(pd.DataFrame({"x": [1]}))


def test_voting_ensemble_runs_on_synthetic_data():
    graph = GraphEngine()
    graph.add_node("model:logistic_regression", "lr")
    graph.add_node("model:decision_tree_classifier", "tree")
    graph.add_node("ensemble:voting", "vote")
    graph.connect("lr", "estimator", "vote", "estimators")
    graph.connect("tree", "estimator", "vote", "estimators")
    result = graph.run()
    model = result["vote"]["estimator"]
    frame = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    model.fit(frame, [0, 0, 0, 1, 1, 1])
    assert list(model.predict(frame)) == [0, 0, 0, 1, 1, 1]
