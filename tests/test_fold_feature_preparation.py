"""Feature decisions used by supervised CV must be fitted per train fold."""

import pandas as pd

from app.ml.evaluation import MLEvaluator, _FoldFeaturePreparation


def test_feature_inclusion_is_learned_only_from_train_rows():
    train = pd.DataFrame({
        "record_id": [1, 2, 3, 4],
        "signal": [1.0, 2.0, 3.0, 4.0],
        "constant_in_train": [7, 7, 7, 7],
    })
    test = pd.DataFrame({
        "record_id": [5, 6],
        "signal": [5.0, 6.0],
        "constant_in_train": [8, 9],
    })

    transformer = _FoldFeaturePreparation().fit(train)
    transformed = transformer.transform(test)

    assert list(transformed.columns) == ["signal"]
    assert transformed["signal"].tolist() == [5.0, 6.0]


def test_cross_validation_fits_feature_preparation_on_each_training_fold(monkeypatch):
    frame = pd.DataFrame({
        "record_id": list(range(18)),
        "signal": [float(index) for index in range(18)],
        "group": ["A" if index % 2 else "B" for index in range(18)],
        "target": [index % 2 for index in range(18)],
    })
    fitted_indices = []
    original_fit = _FoldFeaturePreparation.fit

    def record_fit(self, X, y=None):
        fitted_indices.append(set(X.index))
        return original_fit(self, X, y)

    monkeypatch.setattr(_FoldFeaturePreparation, "fit", record_fit)
    result = MLEvaluator(cv=3).evaluate(
        frame,
        target="target",
        task="binary_classification",
        methods=["logistic_regression"],
    )

    assert result["status"] == "EVALUATION"
    assert len(fitted_indices) == 3
    assert all(len(indices) == 12 for indices in fitted_indices)
    assert len(set.union(*fitted_indices)) == 18
    assert all(indices != set(frame.index) for indices in fitted_indices)
