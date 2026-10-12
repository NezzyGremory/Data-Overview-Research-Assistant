"""Capture current DORA screens with deterministic, local demo data."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["TELEMETRY_API_URL"] = ""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tools.screenshot_ui import _analysis_preview_result, _preview_dataframe
from app.dataset_search.huggingface import DatasetFile, DatasetSearchResult
from app.ui.main_window import MainWindow
from app.ui.views.credits_overlay import show_credits_overlay
from app.reports import ReportGenerator

# app.core.config loads .env during imports; explicitly disable telemetry again
# before MainWindow creates the telemetry client for this offline preview run.
os.environ["TELEMETRY_API_URL"] = ""

OUTPUT_DIR = PROJECT_ROOT / "screenshots" / "readme"
WIDTH = 1920
HEIGHT = 1080


def _research_result() -> dict:
    papers = [
        {
            "title": "Passenger Survival Prediction with Interpretable Machine Learning",
            "authors": ["A. Researcher", "B. Analyst"],
            "year": 2024,
            "venue": "Journal of Applied Data Science",
            "issn": ["2754-1234"],
            "doi": "10.0000/demo.2024.001",
            "source": "OpenAlex, Crossref",
            "relevance_score": 94.2,
            "dataset_usage_confidence": "HIGH",
            "abstract": "A demonstration record used only for README screenshots.",
            "keywords": ["passenger survival", "classification"],
        },
        {
            "title": "Comparing Ensemble Models for Passenger Risk Analysis",
            "authors": ["C. Scientist"],
            "year": 2023,
            "venue": "Data Mining Review",
            "issn": ["2049-3630"],
            "doi": "10.0000/demo.2023.002",
            "source": "OpenAlex",
            "relevance_score": 87.6,
            "dataset_usage_confidence": "MEDIUM",
            "abstract": "A demonstration record used only for README screenshots.",
            "keywords": ["ensemble learning", "survival prediction"],
        },
        {
            "title": "Missing Data Strategies in Historical Passenger Records",
            "authors": ["D. Scholar"],
            "year": 2022,
            "venue": "Research Methods Quarterly",
            "issn": ["2049-3630"],
            "doi": "10.0000/demo.2022.003",
            "source": "Crossref",
            "relevance_score": 78.3,
            "dataset_usage_confidence": "LOW",
            "abstract": "A demonstration record used only for README screenshots.",
            "keywords": ["missing values", "data quality"],
        },
    ]
    return {
        "status": "SUCCESS",
        "summary": {
            "papers_found": len(papers),
            "top_relevance_score": papers[0]["relevance_score"],
            "potential_gap_count": 2,
        },
        "keywords": {"keywords": ["passenger survival", "classification", "data quality"]},
        "domain": {"primary_domain": "Transportation"},
        "queries": ["passenger survival prediction machine learning"],
        "papers": papers,
        "pipeline": {
            "academic_search": {
                "sources": {"OpenAlex": 4, "Crossref": 3},
                "provider_status": {"OpenAlex": "ENABLED", "Crossref": "ENABLED"},
            }
        },
        "landscape": {
            "paper_count": 3,
            "summary": {
                "papers_analyzed": 3,
                "latest_publication_year": 2024,
                "dominant_method": "Random Forest",
            },
            "methods": [
                {"name": "Random Forest", "percentage": 67},
                {"name": "Gradient Boosting", "percentage": 45},
                {"name": "Logistic Regression", "percentage": 32},
            ],
            "top_venues": [
                {"name": "Journal of Applied Data Science", "percentage": 67},
                {"name": "Data Mining Review", "percentage": 33},
            ],
            "publication_years": {"2022": 1, "2023": 1, "2024": 1},
        },
        "trend": {
            "publication_trend": [
                {"year": 2022, "count": 1},
                {"year": 2023, "count": 1},
                {"year": 2024, "count": 1},
            ],
            "recent_direction": "STABLE",
            "peak_year": 2024,
            "peak_paper_count": 1,
            "year_range": {"start": 2022, "end": 2024},
            "growth": {"rate": 0},
        },
        "gaps": {
            "summary": {"gap_count": 2},
            "gaps": [
                {
                    "title": "Evaluasi strategi nilai kosong pada data penumpang",
                    "description": "Bandingkan dampak beberapa strategi imputasi terhadap performa klasifikasi.",
                    "evidence": "Contoh kandidat demo berdasarkan tiga rekaman contoh.",
                },
                {
                    "title": "Perbandingan interpretabilitas model ensemble",
                    "description": "Uji penjelasan fitur pada model dengan kinerja yang sebanding.",
                    "evidence": "Contoh kandidat demo berdasarkan tiga rekaman contoh.",
                },
            ],
        },
        "gap_explanation": {
            "text": "Kandidat berikut hanya contoh tampilan. Validasi topik dan kebaruan dengan meninjau publikasi asli.",
            "source": "local",
        },
    }


def _save(window: MainWindow, name: str) -> None:
    QApplication.processEvents()
    QTest.qWait(120)
    QApplication.processEvents()
    target = OUTPUT_DIR / f"{name}.png"
    if not window.grab().save(str(target)):
        raise RuntimeError(f"Tidak bisa menyimpan {target}")
    print(f"Screenshot saved: {target.relative_to(PROJECT_ROOT)}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(["dora-readme-screenshots"])
    window = MainWindow()
    window.resize(WIDTH, HEIGHT)
    window.show()
    QApplication.processEvents()

    dataframe = _preview_dataframe()
    analysis = _analysis_preview_result(window, dataframe)
    research = _research_result()
    ml_task = {
        "status": "ESTIMATION",
        "primary_task": {
            "label": "Binary Classification",
            "task": "binary classification",
            "confidence": 80,
            "target": "Survived",
        },
        "tasks": [],
        "dataset": {"rows": len(dataframe), "columns": len(dataframe.columns), "numeric_features": 7},
    }
    ml_methods = {
        "recommendations": [
            {"method": "Random Forest", "score": 82.8, "score_type": "accuracy"},
            {"method": "Gradient Boosting", "score": 82.1, "score_type": "accuracy"},
            {"method": "Logistic Regression", "score": 80.7, "score_type": "accuracy"},
        ]
    }
    ml_full = {
        "status": "SUCCESS",
        "primary_task": ml_task["primary_task"],
        "task_detection": ml_task,
        "recommendation": ml_methods,
        "evaluation": {
            "results": [
                {"method": "Random Forest", "metrics": {"accuracy": 0.828}},
                {"method": "Gradient Boosting", "metrics": {"accuracy": 0.821}},
            ]
        },
        "best_method": {"method": "Random Forest", "metrics": {"accuracy": 0.828}},
    }

    # CSV upload screen.
    window.current_dataset = None
    window.current_file_path = None
    window.open_upload_page()
    _save(window, "dataset-upload")

    # Loaded data preview screen.
    window.current_dataset = dataframe
    window.current_file_path = Path("Titanic-dataset.csv")
    window.current_project_id = None
    window.analysis_result = analysis
    window.dataset_preview_page.update_dataset(dataframe, "Titanic-dataset.csv")
    window.dataset_preview_page.file_details.setText("60.3 KB · Dataset demo")
    window.datasets_tabs.setCurrentWidget(window.dataset_preview_page)
    window._show_page("dataset")
    for tab_index, screenshot_name in enumerate((
        "dataset-preview",
        "dataset-data-quality",
        "dataset-columns",
        "dataset-missing-values",
        "dataset-outliers",
        "dataset-correlation",
    )):
        window.dataset_preview_page._select_tab(tab_index)
        _save(window, screenshot_name)

    # Online search uses illustrative results, not a live API request.
    window.open_dataset_search_page()
    search_page = window.dataset_search_page
    search_page.search_input.setText("titanic")
    search_page.source_combo.setCurrentIndex(search_page.source_combo.findData("all"))
    preview_file = DatasetFile(filename="titanic.csv", size_bytes=61_747)
    search_page._on_search_finished([
        DatasetSearchResult(
            dataset_id="kaggle:demo/titanic-passenger-survival",
            author="DORA demo",
            description="Example passenger survival dataset for the screenshot gallery.",
            downloads=12000,
            likes=340,
            tags=["tabular", "classification"],
            files=[preview_file],
            match_reason="Example result for the documentation preview.",
        ),
    ])
    search_page.status_label.setText("Example search results · demo records only")
    _save(window, "dataset-search")

    # Analysis and notebook-style editor.
    window.current_dataset = dataframe
    window.current_file_path = Path("Titanic-dataset.csv")
    window.analysis_result = analysis
    window.open_analysis_page()
    _save(window, "analysis")
    window.open_editor_analysis_page()
    _save(window, "editor-analysis")

    # ML model recommendations.
    window.ml_page.last_task_result = ml_task
    window.ml_page.last_method_result = ml_methods
    window.ml_page.last_intelligence_result = ml_full
    window.ml_page.display_result(ml_task, ml_methods)
    window.open_ml_page()
    _save(window, "ml-intelligence")

    # Overview workflow summary.
    dataframe_overview = dataframe.copy()
    dataframe_overview["Name"] = [f"Passenger {index + 1:04d}" for index in range(len(dataframe_overview))]
    dataframe_overview["Ticket"] = [f"TKT-{index + 1:07d}" for index in range(len(dataframe_overview))]
    dataframe_overview["Cabin"] = [f"C{index:03d}" for index in range(len(dataframe_overview))]
    dataframe_overview.loc[7:689, "Cabin"] = pd.NA
    rng = np.random.default_rng(2026)
    dataframe_overview["Age"] = rng.integers(18, 66, len(dataframe_overview)).astype(float)
    dataframe_overview.loc[7:182, "Age"] = pd.NA
    dataframe_overview.loc[10, "Fare"] = 512.3
    window.current_dataset = dataframe_overview
    window.analysis_result = _analysis_preview_result(window, dataframe_overview)
    window.ml_page.last_task_result = ml_task
    window.ml_page.last_method_result = ml_methods
    window.ml_page.last_intelligence_result = ml_full
    window.research_page.research_result = research
    window.dashboard._filename = "Titanic-dataset.csv"
    window.dashboard._preview_file_size = "60.3 KB"
    window.dashboard._preview_readiness_label = "Readiness 82/100"
    window.dashboard.update_dataset(dataframe_overview, "Titanic-dataset.csv")
    window.dashboard.update_research_result(research)
    window.dashboard.update_workflow_state(True, True, True, True)
    window.open_dashboard()
    _save(window, "overview")

    # Academic Research and its linked research tools use clearly labelled demo records.
    window.current_dataset = dataframe
    window.analysis_result = analysis
    window.ml_page.last_task_result = ml_task
    window.ml_page.last_method_result = ml_methods
    window.ml_page.last_intelligence_result = ml_full
    window.research_page.research_result = research
    window.research_page._update_status(dataframe, analysis)
    window.research_page._populate_result(research, {"primary_task": "Binary Classification"})
    window.open_research_page()
    _save(window, "academic-research")
    window.papers_tool_page.update_from_result(research)
    window.open_papers_tool()
    _save(window, "papers")
    window.landscape_tool_page.update_from_result(research)
    window.open_landscape_tool()
    _save(window, "research-landscape")
    window.gap_tool_page._attempted_explanation_this_session = True
    window.gap_tool_page.update_from_result(research)
    window.open_gap_tool()
    window.gap_tool_page._attempted_explanation_this_session = True
    _save(window, "research-gap")

    # Render a local demo report without starting the app's background worker.
    window.report_tool_page.research_result = research
    window._show_page("report")
    report_page = window.report_tool_page
    report_page._chart_tempdir = tempfile.TemporaryDirectory(prefix="dora_readme_report_")
    report_page.chart_dir = Path(report_page._chart_tempdir.name)
    report_page.current_html = ReportGenerator().generate(
        dataset_name="Titanic-dataset",
        analysis_data=analysis,
        ml_data=ml_full,
        research_data=research,
        metadata={"dataframe": dataframe, "chart_dir": str(report_page.chart_dir)},
    )
    report_page.preview.setSearchPaths([str(report_page.chart_dir)])
    report_page.preview.setHtml(report_page.current_html)
    report_page.status.setText("Pratinjau laporan · data demo lokal")
    report_page.export_button.setEnabled(True)
    _save(window, "research-report")

    window.open_help()
    _save(window, "help")
    window.open_about()
    _save(window, "about")

    overlay = show_credits_overlay(window)
    QApplication.processEvents()
    if overlay._scroll_animation is not None:
        overlay._scroll_animation.pause()
        overlay._scroll_animation.setCurrentTime(overlay._scroll_animation.duration() // 2)
    if overlay._fade_animation is not None:
        overlay._fade_animation.stop()
        overlay._opacity.setOpacity(1.0)
    _save(window, "copyright-credits")
    overlay.close_overlay()
    QApplication.processEvents()

    window._session_data_cleaned = True
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
