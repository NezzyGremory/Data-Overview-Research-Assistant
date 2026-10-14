import pandas as pd

from app.reports.generator import ReportGenerator


def test_report_renders_new_ml_readiness_sections():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "group": ["A", "A", "B", "C", "B", "A"],
            "target": [0, 0, 0, 0, 1, 1],
        }
    )

    html = ReportGenerator().generate(
        dataset_name="Readiness Test",
        ml_data={
            "primary_task": {
                "task": "binary_classification",
                "target": "target",
            }
        },
        metadata={"dataframe": dataframe},
    )

    for title in (
        "Data Quality Summary",
        "Class Balance",
        "Preprocessing Recommendation",
        "Feature Selection Hints",
        "ML Pipeline Checklist",
    ):
        assert f"<h2>{title}</h2>" in html
    assert "Fit preprocessing on the training set only to avoid data leakage." in html
    assert "Problem Definition" in html
    assert "Model Saving and Deployment" in html


def test_report_skips_new_sections_without_readiness_data_or_dataframe():
    html = ReportGenerator().generate(dataset_name="Empty")

    assert "Data Quality Summary" not in html
    assert "ML Pipeline Checklist" not in html