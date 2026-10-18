from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html import escape as html_escape
from html.parser import HTMLParser
from pathlib import Path
import re
import json
import os
import pickle
import subprocess
import textwrap
from types import SimpleNamespace
import sys
import tempfile
import time
from typing import Any, Dict
from urllib.parse import quote_plus, urlparse
import pandas as pd
import httpx
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from shiboken6 import isValid as is_qt_object_valid

from PySide6.QtCore import Qt, QUrl, QObject, QThread, QTimer, Signal, Slot, QMarginsF, QSize, QPoint, QRect, QRectF, QEvent, QEventLoop
from PySide6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QDesktopServices, QTextDocument, QFont, QFontDatabase, QColor, QPalette, QPageSize, QPageLayout, QPainter, QPen, QRegion
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QBoxLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTextBrowser,
    QLineEdit,
    QProgressBar,
    QHeaderView,
    QInputDialog,
    QComboBox,
    QCheckBox,
    QStyle,
    QStyleOptionButton,
    QToolButton,
)

from app.analyzer.loader import DatasetLoader
from app.analyzer.profiler import DatasetProfiler
from app.analyzer.statistics import DatasetStatistics
from app.analyzer.missing_values import MissingValueAnalyzer
from app.analyzer.missing_values import (
    impute_missing_values,
    normalize_missing_values,
)
from app.analyzer.cleaner import (
    CleanResult,
    drop_missing,
    export_cleaned_csv,
    preview_drop,
)
from app.analyzer.duplicates import DuplicateAnalyzer
from app.analyzer.outliers import OutlierAnalyzer
from app.analyzer.correlations import CorrelationAnalyzer
from app.analyzer.fingerprint import DatasetFingerprint
from app.analyzer.data_quality import DataQualityDiagnoser
from app.ai.gemini_explainer import GeminiDatasetExplainer, get_saved_api_key, save_api_key
from app.ai.local_explainer import LocalAcademicExplainer
from app.ai.research_gap_explainer import ResearchGapExplainer
from app.ml.task_detector import MLTaskDetector
from app.ml.method_recommender import MethodRecommender
from app.ui.dataset_search_page import DatasetSearchPage
from app.ui.dataset_preview_page import DatasetPreviewPage
from app.ui.upload_loaded_page import LoadedUploadPage
from app.ui.missing_values_dialogs import (
    DropMissingOptionsDialog,
    DropMissingPreviewDialog,
)
from app.ui.style_utils import apply_card_shadow, render_icon_pixmap
from app.ui.scale import SCALE_PRESETS, ScaleManager, build_stylesheet, logical_width, scale_stylesheet, ui_px
from app.storage import Database, ProjectRepository, DatasetVersionManager
from app.reports import ReportGenerator
from app.core.config import get_data_dir
from app.telemetry import get_telemetry


# Set to True to restore the Node Editor page and its navigation entry.
NODE_EDITOR_UI_ENABLED = True


def _resource_path(relative_path: str | Path) -> Path:
    """Return a resource path that works in development and PyInstaller builds."""
    relative = Path(relative_path)

    if getattr(sys, "frozen", False):
        # PyInstaller --onefile extracts bundled data under _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "app" / "ui" / relative

    return Path(__file__).resolve().parent / relative


def _sinta_ui_text(value: Any) -> str:
    """Avoid exposing raw transport wording in SINTA-facing UI text."""
    text = str(value or "")
    text = re.sub(r"\bkegagalan\s+pencarian\b", "pemeriksaan yang belum tuntas", text, flags=re.IGNORECASE)
    return re.sub(r"\bgagal\b", "belum berhasil", text, flags=re.IGNORECASE).strip()


def _sinta_ui_note(screening: Dict[str, Any] | None) -> str:
    """Turn internal SINTA lookup details into calm, user-facing guidance."""
    screening = screening if isinstance(screening, dict) else {}
    lookup_status = str(screening.get("lookup_status") or "").upper()
    if lookup_status == "FAILED":
        message = "Direktori SINTA belum dapat diperiksa. Periksa koneksi internet, lalu coba lagi."
    else:
        message = _sinta_ui_text(screening.get("notes"))
    assistive_note = _sinta_ui_text(screening.get("assistive_note"))
    return " ".join(part for part in (message, assistive_note) if part)


class GeminiExplainWorker(QObject):
    """Run Gemini / Hybrid academic explanation off the Qt UI thread."""

    finished = Signal(str, str, str)  # (text, source, optional cloud status)
    failed = Signal(str)

    def __init__(self, analysis: dict[str, Any]):
        super().__init__()
        self.analysis = analysis

    def run(self):
        try:
            if getattr(sys, "frozen", False):
                # Isolate the optional HTTPS client during development runs. Frozen
                # builds may not include a Python interpreter for the helper process.
                raise RuntimeError("Gunakan penjelasan lokal pada aplikasi terkemas.")
            helper = Path(__file__).with_name("gemini_subprocess.py")
            child_env = os.environ.copy()
            child_env["GEMINI_TIMEOUT"] = "12"
            command = [sys.executable, str(helper)]
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            completed = subprocess.run(
                command,
                input=pickle.dumps(self.analysis, protocol=pickle.HIGHEST_PROTOCOL),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=45,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=child_env,
                creationflags=creation_flags,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError("Proses Gemini gagal.")
            result = pickle.loads(completed.stdout)
            text, source = result[:2]
            detail = str(result[2] or "") if len(result) > 2 else ""
            self.finished.emit(text, source, detail)
        except Exception:
            try:
                local_text = LocalAcademicExplainer().explain(self.analysis)
                self.finished.emit(local_text, "local", "Penjelasan lokal digunakan karena proses Gemini tidak dapat dijalankan.")
            except Exception as local_exc:
                self.failed.emit(str(local_exc))


class DatasetAnalysisWorker(QObject):
    """Run the complete dataset analysis away from the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, analyzers: dict, quality_diagnoser, dataframe, ml_result=None, recommended_models=None):
        super().__init__()
        self.analyzers = analyzers
        self.quality_diagnoser = quality_diagnoser
        self.dataframe = dataframe
        self.ml_result = ml_result or {}
        self.recommended_models = recommended_models or []

    def run(self):
        try:
            # This analyzer loads scikit-learn, which is comparatively slow to
            # import. Defer it until a dataset analysis is actually requested
            # so opening the desktop UI does not pay that cost up front.
            from app.analyzer.ml_readiness import analyze_ml_readiness

            with ThreadPoolExecutor(max_workers=len(self.analyzers)) as executor:
                futures = {
                    name: executor.submit(analyzer, self.dataframe)
                    for name, analyzer in self.analyzers.items()
                }
                result = {}
                for name, future in futures.items():
                    try:
                        result[name] = future.result()
                    except Exception as error:
                        raise RuntimeError(f"{name} analysis failed: {error}") from error

            # Profiling already scans the frame for duplicate rows. Reuse that
            # count instead of running the same full-frame duplicate check again.
            duplicate_count = int((result.get("profile") or {}).get("duplicate_rows", 0))
            row_count = len(self.dataframe)
            result["duplicates"] = {
                "duplicate_count": duplicate_count,
                "duplicate_percentage": (
                    duplicate_count / row_count * 100 if row_count > 0 else 0
                ),
            }

            try:
                result["data_quality"] = self.quality_diagnoser.diagnose(
                    dataframe=self.dataframe,
                    profile=result.get("profile"),
                    missing_values=result.get("missing_values"),
                    duplicates=result.get("duplicates"),
                    outliers=result.get("outliers"),
                    fingerprint=result.get("fingerprint"),
                )
            except Exception as error:
                print(f"Warning: data quality diagnosis failed: {error}")
                result["data_quality"] = []

            result["ml_readiness"] = analyze_ml_readiness(
                self.dataframe,
                missing_values=result.get("missing_values"),
                duplicates=result.get("duplicates"),
                outliers=result.get("outliers"),
                quality_issues=result.get("data_quality"),
                fingerprint=result.get("fingerprint"),
                recommended_models=self.recommended_models or None,
                ml_result=self.ml_result,
            )

            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class DropMissingPreviewWorker(QObject):
    """Calculate a drop preview away from the Qt UI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, dataframe: pd.DataFrame, options: dict):
        super().__init__()
        self.dataframe = dataframe
        self.options = options

    def run(self):
        try:
            self.finished.emit(preview_drop(self.dataframe, **self.options))
        except Exception as error:
            self.failed.emit(str(error))


class CreateCleanedDatasetWorker(QObject):
    """Recompute and persist a confirmed cleaned dataset version."""

    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, dataframe, options, version_manager, project_id):
        super().__init__()
        self.dataframe = dataframe
        self.options = options
        self.version_manager = version_manager
        self.project_id = project_id

    def run(self):
        try:
            result = drop_missing(self.dataframe, **self.options)
            if (
                result.report.get("strategy") == "report"
                or result.cleaned_df.empty
                or result.cleaned_df.equals(self.dataframe)
            ):
                self.finished.emit(result, None)
                return

            report = {
                "rows_before": result.rows_before,
                "rows_after": result.rows_after,
                "rows_dropped": result.rows_dropped,
                "percent_dropped": result.percent_dropped,
                "dropped_per_column": result.dropped_per_column,
                "zero_columns_used": result.zero_columns_used,
                "warnings": result.warnings,
                **self.options,
            }
            version = self.version_manager.create_version(
                project_id=self.project_id,
                dataframe=result.cleaned_df,
                operation=f"missing_values_{self.options.get('strategy', 'drop_rows')}",
                parameters=report,
                impact=report,
                description=result.formatted_report,
                label="Cleaned Dataset",
            )
            audit_dir = (
                self.version_manager.data_dir
                / "projects"
                / str(self.project_id)
                / "versions"
            )
            audit_dir.mkdir(parents=True, exist_ok=True)
            if not result.dropped_df.empty:
                result.dropped_df.to_csv(
                    audit_dir / f"v{version.version}_dropped_rows.csv",
                    index=True,
                    index_label="source_index",
                    encoding="utf-8-sig",
                )
            self.finished.emit(result, version)
        except Exception as error:
            self.failed.emit(str(error))


class ExportCleanedDatasetWorker(QObject):
    """Load a stored cleaned version and export it without blocking the UI."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, version_manager, project_id, version, path, original_path):
        super().__init__()
        self.version_manager = version_manager
        self.project_id = project_id
        self.version = version
        self.path = path
        self.original_path = original_path

    def run(self):
        try:
            dataframe = self.version_manager.load_dataframe(
                self.project_id, self.version
            )
            audit_path = (
                self.version_manager.data_dir
                / "projects"
                / str(self.project_id)
                / "versions"
                / f"v{self.version}_dropped_rows.csv"
            )
            dropped_rows = pd.read_csv(audit_path, index_col=0) if audit_path.exists() else None
            exported_path = export_cleaned_csv(
                dataframe,
                self.path,
                original_path=self.original_path,
                dropped_rows=dropped_rows,
            )
            self.finished.emit(exported_path)
        except Exception as error:
            self.failed.emit(str(error))


class MLTrainingWorker(QObject):
    """Run the configured full ML evaluation without blocking the GUI."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, intelligence_engine, dataframe, fingerprint):
        super().__init__()
        self.intelligence_engine = intelligence_engine
        self.dataframe = dataframe
        self.fingerprint = fingerprint

    def run(self):
        try:
            # Keep the full evaluation path and all configured estimators.
            result = self.intelligence_engine.analyze(
                dataframe=self.dataframe,
                fingerprint=self.fingerprint,
                evaluate_models=True,
            )
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class AcademicResearchWorker(QObject):
    """Run literature retrieval and research synthesis outside the GUI thread."""

    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, engine, task_detector, dataframe, fingerprint):
        super().__init__()
        self.engine = engine
        self.task_detector = task_detector
        self.dataframe = dataframe
        self.fingerprint = fingerprint

    def run(self):
        try:
            try:
                ml_result = dict(self.task_detector.detect(self.fingerprint))
            except Exception:
                ml_result = {
                    "status": "ESTIMATION", "primary_task": None,
                    "tasks": [], "task_count": 0,
                }
            ml_result["dataset"] = {
                "rows": int(self.dataframe.shape[0]),
                "columns": int(self.dataframe.shape[1]),
                "numeric_features": int(
                    self.dataframe.select_dtypes(include="number").shape[1]
                ),
            }
            result = self.engine.analyze(
                dataframe=self.dataframe,
                fingerprint=self.fingerprint,
                ml_result=ml_result,
                search_limit=20,
                max_queries=5,
                include_insights=False,
            )
            self.finished.emit(result, ml_result)
        except Exception as error:
            self.failed.emit(str(error))


class AcademicResearchInsightsWorker(QObject):
    """Calculate the optional landscape/trend/gap views without searching again."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, engine, result, ml_result):
        super().__init__()
        self.engine = engine
        self.result = dict(result or {})
        self.result["pipeline"] = dict(self.result.get("pipeline") or {})
        self.ml_result = dict(ml_result or {})

    def run(self):
        try:
            result = self.engine.analyze_insights(self.result, self.ml_result)
            self.finished.emit(result)
        except Exception as error:
            self.failed.emit(str(error))


class SintaLookupWorker(QObject):
    """Worker background untuk verifikasi jurnal di direktori resmi SINTA tanpa memblokir GUI."""

    finished = Signal(int, str, object)
    failed = Signal(str)

    def __init__(
        self,
        query: str,
        issn_list: list[str] | str,
        journal_name: str,
        paper_year: Any,
        paper_index: int,
        paper_identity: str,
        article_doi: str = "",
    ):
        super().__init__()
        self.query = query
        self.issn_list = issn_list if isinstance(issn_list, list) else ([issn_list] if issn_list else [])
        self.journal_name = journal_name
        from app.research.sinta_verifier import SintaVerifier
        self.paper_year = SintaVerifier.normalize_year(paper_year)
        self.paper_index = paper_index
        self.paper_identity = paper_identity
        self.article_doi = str(article_doi or "").strip()

    def run(self):
        verifier = None
        try:
            from app.research.sinta_verifier import SintaVerifier
            verifier = SintaVerifier(timeout=12.0)
            result = verifier.verify(
                issn_list=self.issn_list,
                journal_name=self.journal_name,
                article_year=self.paper_year,
                article_doi=self.article_doi,
            )
            self.finished.emit(self.paper_index, self.paper_identity, result.to_dict())
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if verifier is not None:
                verifier.close()


class SintaBatchLookupWorker(QObject):
    """Verify several paper journals without blocking the interface."""

    progress = Signal(int, int, int, str, object)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from app.research.sinta_verifier import SintaVerifier

        total = len(self.tasks)

        # Group journal lookups by any shared normalized ISSN. Fall back to
        # normalized journal name only when the records have no ISSN.
        groups = []
        for task in self.tasks:
            task_issns = {
                SintaVerifier.normalize_issn(value)
                for value in (task.get("issn") or [])
                if SintaVerifier.normalize_issn(value)
            }
            task_name = SintaVerifier.normalize_name(task.get("journal"))
            matching = []
            for index, group in enumerate(groups):
                if task_issns and task_issns.intersection(group["issns"]):
                    matching.append(index)
                elif (
                    task_name
                    and task_name == group["name"]
                    and not task_issns
                    and not group["issns"]
                ):
                    matching.append(index)

            if not matching:
                groups.append({
                    "tasks": [task],
                    "issns": set(task_issns),
                    "name": task_name,
                    "journal": str(task.get("journal") or "").strip(),
                })
                continue

            group = groups[matching[0]]
            group["tasks"].append(task)
            group["issns"].update(task_issns)
            if not group["journal"] and task.get("journal"):
                group["journal"] = str(task["journal"]).strip()
            for duplicate_index in reversed(matching[1:]):
                duplicate = groups.pop(duplicate_index)
                group["tasks"].extend(duplicate["tasks"])
                group["issns"].update(duplicate["issns"])

        def lookup_group(group):
            verifier = SintaVerifier(timeout=8.0)
            try:
                result = verifier.verify(
                    issn_list=sorted(group["issns"]),
                    journal_name=group["journal"],
                    article_year=None,
                ).to_dict()
            except Exception as error:
                result = {
                    "found": False,
                    "status": "UNVERIFIED",
                    "lookup_status": "FAILED",
                    "journal_name": group["journal"],
                    "notes": f"Gagal memeriksa jurnal: {error}",
                    "source_url": "https://sinta.kemdiktisaintek.go.id/journals",
                }
            return group, result, verifier

        current = 0
        worker_count = min(3, len(groups)) or 1
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(lookup_group, group) for group in groups]
            for future in as_completed(futures):
                group, base_result, verifier = future.result()
                try:
                    for task in group["tasks"]:
                        try:
                            if task.get("doi"):
                                result = verifier.evaluate_article_doi(
                                    base_result,
                                    task.get("doi"),
                                    task.get("year"),
                                ).to_dict()
                            else:
                                result = SintaVerifier.evaluate_for_year(
                                    base_result, task.get("year")
                                ).to_dict()
                        except Exception as error:
                            result = {
                                **base_result,
                                "status": "UNVERIFIED",
                                "lookup_status": "FAILED",
                                "checked_for_year": task.get("year"),
                                "notes": f"Evaluasi artikel dari profil resmi gagal: {error}",
                            }
                        current += 1
                        self.progress.emit(
                            current,
                            total,
                            task["paper_index"],
                            task["identity"],
                            result,
                        )
                finally:
                    verifier.close()
        self.finished.emit(total)


class JournalIndexBatchLookupWorker(QObject):
    """Check public/entitled non-SINTA journal index sources in the background."""

    progress = Signal(int, int, int, str, object)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from app.research.journal_index_verifier import JournalIndexVerifier

        groups = {}
        for task in self.tasks:
            normalized_issns = tuple(sorted({
                JournalIndexVerifier.normalize_issn(value)
                for value in task.get("issn", [])
                if JournalIndexVerifier.normalize_issn(value)
            }))
            groups.setdefault(normalized_issns, []).append(task)

        verifier = JournalIndexVerifier(timeout=8.0)
        total = len(self.tasks)
        completed = 0
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(verifier.verify, list(key)): (key, group)
                    for key, group in groups.items()
                    if key
                }
                for key, group in groups.items():
                    if key:
                        continue
                    result = verifier.verify([])
                    for task in group:
                        completed += 1
                        self.progress.emit(
                            completed,
                            total,
                            task["paper_index"],
                            task["identity"],
                            result,
                        )
                for future in as_completed(futures):
                    _key, group = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {
                            "status": "FAILED",
                            "indexes": [],
                            "sources": {},
                            "message": f"Pemeriksaan indeks belum berhasil: {error}",
                        }
                    for task in group:
                        completed += 1
                        self.progress.emit(
                            completed,
                            total,
                            task["paper_index"],
                            task["identity"],
                            result,
                        )
            self.finished.emit(total)
        except Exception as error:
            self.failed.emit(str(error))


class NonSintaBatchLookupWorker(QObject):
    """Worker background untuk verifikasi akreditasi jurnal non-SINTA (HK-dir / ABDC) tanpa memblokir GUI."""

    progress = Signal(int, int, int, str, object)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks

    def run(self):
        from app.research.accreditation_verifier import AccreditationVerifier
        from app.research.journal_index_verifier import JournalIndexVerifier

        verifier = AccreditationVerifier(timeout=8.0)
        index_verifier = JournalIndexVerifier(timeout=5.0)
        total = len(self.tasks)
        try:
            all_issns = [
                value
                for task in self.tasks
                for value in (task.get("issn") or [])
            ]
            assessment_by_issn = index_verifier.verify_official_assessments(all_issns)

            for completed, task in enumerate(self.tasks, start=1):
                try:
                    result = verifier.verify(
                        issn_list=task.get("issn", []),
                        journal_name=task.get("journal"),
                        article_year=task.get("year"),
                    ).to_dict()
                except Exception as error:
                    result = {
                        "found": False,
                        "status": "UNVERIFIED",
                        "lookup_status": "FAILED",
                        "journal_name": task.get("journal"),
                        "notes": f"Pemeriksaan direktori HK-dir / ABDC belum berhasil: {error}",
                    }

                normalized_issns = {
                    index_verifier.normalize_issn(value)
                    for value in (task.get("issn") or [])
                    if index_verifier.normalize_issn(value)
                }
                assessments = []
                assessment_states = []
                seen_lists = set()
                for issn in normalized_issns:
                    lookup = assessment_by_issn.get(issn, {})
                    assessment_states.append(str(lookup.get("status") or "FAILED"))
                    for assessment in lookup.get("assessments", []):
                        if not isinstance(assessment, dict):
                            continue
                        list_id = str(assessment.get("list_id") or "")
                        if list_id and list_id not in seen_lists:
                            seen_lists.add(list_id)
                            assessments.append(dict(assessment))

                if assessments:
                    result["found"] = True

                if not assessment_states:
                    assessment_status = "NO_METADATA"
                elif all(state == "CHECKED" for state in assessment_states):
                    assessment_status = "CHECKED"
                elif any(state == "CHECKED" for state in assessment_states):
                    assessment_status = "PARTIAL"
                else:
                    assessment_status = "FAILED"
                result["official_assessments"] = assessments
                result["official_assessments_status"] = assessment_status
                result["official_assessments_source"] = "OpenAlex source lists"
                self.progress.emit(
                    completed,
                    total,
                    task["paper_index"],
                    task["identity"],
                    result,
                )
            self.finished.emit(total)
        except Exception as error:
            self.failed.emit(str(error))


class _SintaTextParser(HTMLParser):
    """Extract visible directory text into short lines for conservative matching."""

    _BLOCK_TAGS = {"br", "p", "div", "article", "li", "h1", "h2", "h3", "h4"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def lines(self):
        return [line.strip() for line in " ".join(self.parts).splitlines() if line.strip()]


class ResearchGapExplanationWorker(QObject):
    finished = Signal(str, str, str)
    failed = Signal(str)

    def __init__(self, research):
        super().__init__()
        self.research = research

    def run(self):
        try:
            explanation, source, reason = ResearchGapExplainer().explain(self.research)
            self.finished.emit(explanation, source, reason)
        except Exception as error:
            try:
                explanation, source, reason = ResearchGapExplainer().explain_local(
                    self.research,
                    f"Penjelasan lokal digunakan setelah proses Gemini mengalami kendala ({type(error).__name__}).",
                )
                self.finished.emit(explanation, source, reason)
            except Exception as local_error:
                self.failed.emit(str(local_error))


class ReportGenerationWorker(QObject):
    """Build report markup and chart images without blocking the Qt UI."""

    finished = Signal(str, str)
    failed = Signal(str)

    def __init__(self, dataset_name, analysis_data, ml_data, research_data, dataframe, chart_dir):
        super().__init__()
        self.dataset_name = dataset_name
        self.analysis_data = analysis_data
        self.ml_data = ml_data
        self.research_data = research_data
        self.dataframe = dataframe
        self.chart_dir = chart_dir

    def run(self):
        try:
            content = ReportGenerator().generate(
                dataset_name=self.dataset_name,
                analysis_data=self.analysis_data,
                ml_data=self.ml_data,
                research_data=self.research_data,
                metadata={"dataframe": self.dataframe, "chart_dir": self.chart_dir},
            )
            self.finished.emit(content, self.chart_dir)
        except Exception as error:
            self.failed.emit(str(error))


class ReportPdfWorker(QObject):
    """Render the prepared report to PDF away from the UI thread."""

    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, content, chart_dir, output_path):
        super().__init__()
        self.content = content
        self.chart_dir = chart_dir
        self.output_path = output_path

    def run(self):
        try:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(self.output_path)
            layout = QPageLayout(
                QPageSize(QPageSize.A4), QPageLayout.Portrait,
                QMarginsF(18, 18, 18, 18), QPageLayout.Millimeter,
            )
            printer.setPageLayout(layout)
            document = QTextDocument()
            document.setDefaultFont(QFont("Times New Roman", 11))
            if self.chart_dir:
                document.setBaseUrl(QUrl.fromLocalFile(Path(self.chart_dir).as_posix() + "/"))
            document.setHtml(self.content)
            document.print_(printer)
            output = Path(self.output_path)
            if not output.is_file() or output.stat().st_size == 0:
                raise OSError("File PDF tidak berhasil dibuat.")
            self.finished.emit(str(output))
        except Exception as error:
            self.failed.emit(str(error))


class LoadingCard(QFrame):
    """Visible animated progress state shared by long-running feature pages."""

    def __init__(self, title: str, description: str, parent=None):
        super().__init__(parent)
        self.setObjectName("contentCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(12)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")
        self.description_label = QLabel(description)
        self.description_label.setObjectName("cardDescription")
        self.description_label.setWordWrap(True)

        self.spinner_label = QLabel("◌")
        self.spinner_label.setObjectName("loadingSpinner")
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_label.setFixedWidth(32)
        self._spinner_frames = ("◌", "◔", "◑", "◕", "●", "◕", "◑", "◔")
        self._spinner_index = 0
        self._timer = QTimer(self)
        self._timer.setInterval(110)
        self._timer.timeout.connect(lambda: self._advance_spinner())
        self._timer.start()

        heading = QHBoxLayout()
        heading.addWidget(self.spinner_label)
        heading.addWidget(self.title_label, 1)
        layout.addLayout(heading)
        layout.addWidget(self.description_label)

        self.progress = QProgressBar()
        self.progress.setObjectName("loadingProgress")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        layout.addWidget(self.progress)

    def _advance_spinner(self):
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self.spinner_label.setText(self._spinner_frames[self._spinner_index])


class StatCard(QFrame):

    def __init__(
        self,
        title,
        value="—",
        description="",
    ):
        super().__init__()

        self.setObjectName(
            "statCard"
        )

        self.setMinimumWidth(
            105
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            14,
            11,
            14,
            11,
        )

        layout.setSpacing(
            1
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "statTitle"
        )

        self.value_label = QLabel(
            str(value)
        )

        self.value_label.setObjectName(
            "statValue"
        )

        description_label = QLabel(
            description
        )

        description_label.setObjectName(
            "statDescription"
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            self.value_label
        )

        layout.addWidget(
            description_label
        )

    def set_value(
        self,
        value,
    ):

        self.value_label.setText(
            str(value)
        )

class QuickActionCard(QFrame):

    def __init__(
        self,
        icon,
        title,
        description,
        callback,
        enabled=True,
    ):
        super().__init__()

        self.setObjectName(
            "quickActionCard"
        )

        self.setMinimumHeight(
            108
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            16,
            14,
            16,
            14,
        )

        layout.setSpacing(
            5
        )

        # -------------------------------------------------
        # TOP
        # -------------------------------------------------

        top_layout = QHBoxLayout()

        top_layout.setSpacing(
            10
        )

        icon_label = QLabel(
            icon
        )

        icon_label.setObjectName(
            "quickActionIcon"
        )

        icon_label.setFixedSize(
            32,
            32
        )

        icon_label.setAlignment(
            Qt.AlignCenter
        )

        top_layout.addWidget(
            icon_label
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "quickActionTitle"
        )

        top_layout.addWidget(
            title_label
        )

        top_layout.addStretch()

        layout.addLayout(
            top_layout
        )

        # -------------------------------------------------
        # DESCRIPTION
        # -------------------------------------------------

        description_label = QLabel(
            description
        )

        description_label.setObjectName(
            "quickActionDescription"
        )

        description_label.setWordWrap(
            True
        )

        layout.addWidget(
            description_label
        )

        layout.addStretch()

        # -------------------------------------------------
        # BUTTON
        # -------------------------------------------------

        action_button = QPushButton(
            "Open  →"
        )

        action_button.setObjectName(
            "quickActionButton"
        )

        action_button.setCursor(
            Qt.PointingHandCursor
        )

        action_button.setEnabled(
            enabled
        )

        if callback is not None:

            action_button.clicked.connect(
                callback
            )

        layout.addWidget(
            action_button
        )

class PipelineStage(QFrame):

    def __init__(
        self,
        number,
        title,
        status="WAITING",
    ):
        super().__init__()
        self.number = number
        self.setObjectName(
            "pipelineStageCard"
        )
        self.setMinimumHeight(88)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(
            self
        )
        layout.setContentsMargins(
            8,
            10,
            8,
            10,
        )
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignTop)

        self.marker_label = QLabel(str(number))
        self.marker_label.setObjectName("pipelineMarker")
        self.marker_label.setFixedSize(28, 28)
        self.marker_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.marker_label, 0, Qt.AlignHCenter)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("pipelineTitle")
        self.title_label.setAlignment(Qt.AlignHCenter)
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.status_label = QLabel(
            status
        )
        self.status_label.setObjectName(
            "pipelineStatus"
        )
        self.status_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.status_label)

    def set_status(
        self,
        status,
        state="waiting",
    ):

        self.status_label.setText(
            status
        )

        # Keep the step number visible in every state; the color and READY
        # label communicate completion without an emoji or decorative icon.
        self.marker_label.setText(str(self.number))

        self.setProperty(
            "state",
            state
        )

        self.status_label.setProperty(
            "state",
            state
        )
        self.marker_label.setProperty("state", state)

        style = self.style()

        style.unpolish(
            self
        )

        style.polish(
            self
        )
        self.marker_label.style().unpolish(self.marker_label)
        self.marker_label.style().polish(self.marker_label)

        self.update()

class MiniLineChart(QFrame):
    """Compact line chart used by the dashboard research snapshot."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("dashboardChartCard")
        self._title = title
        self._subtitle = subtitle
        self._labels = []
        self._values = []
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, labels, values):
        self._labels = [str(x) for x in labels]
        self._values = [float(x) for x in values]
        self.update()

    def clear_data(self):
        self._labels = []
        self._values = []
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
        from PySide6.QtCore import QRectF, QPointF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        card_rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor("#DCE6F3"), 1))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawRoundedRect(card_rect, 14, 14)

        painter.setPen(QColor("#17304E"))
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSize(12)
        painter.setFont(title_font)
        painter.drawText(16, 22, self._title)

        painter.setPen(QColor("#7C8BA0"))
        sub_font = QFont(self.font())
        sub_font.setPointSize(10)
        painter.setFont(sub_font)
        painter.drawText(16, 37, self._subtitle)

        chart = QRectF(18, 52, max(40, self.width() - 36), max(64, self.height() - 82))
        painter.setPen(QPen(QColor("#E8EEF6"), 1))
        for frac in (0.0, 0.5, 1.0):
            y = chart.top() + chart.height() * frac
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))

        if not self._values:
            painter.setPen(QColor("#A0ADBD"))
            painter.drawText(chart, Qt.AlignCenter, "Jalankan Academic Research untuk menampilkan grafik ini.")
            painter.end()
            return

        vmax = max(self._values) or 1.0
        n = len(self._values)
        points = []
        for i, value in enumerate(self._values):
            x = chart.left() if n == 1 else chart.left() + (chart.width() * i / (n - 1))
            y = chart.bottom() - (value / vmax) * chart.height()
            points.append(QPointF(x, y))

        pen = QPen(QColor("#4F8CFF"), 2)
        painter.setPen(pen)
        for a, b in zip(points, points[1:]):
            painter.drawLine(a, b)

        painter.setBrush(QBrush(QColor("#4F8CFF")))
        painter.setPen(Qt.NoPen)
        for point in points:
            painter.drawEllipse(point, 3.5, 3.5)

        painter.setPen(QColor("#7C8BA0"))
        painter.setFont(sub_font)
        # Keep enough horizontal space between year labels on compact cards.
        max_labels = max(2, min(4, int(chart.width() // 145)))
        if n <= max_labels:
            label_indexes = list(range(n))
        else:
            label_indexes = sorted({
                round(index * (n - 1) / (max_labels - 1))
                for index in range(max_labels)
            })

        for i in label_indexes:
            label = self._labels[i]
            if n == 1:
                x = chart.left()
            else:
                x = chart.left() + chart.width() * i / (n - 1)
            if n == 1:
                rect = QRectF(x - 30, chart.bottom() + 3, 60, 17)
                alignment = Qt.AlignHCenter
            elif i == 0:
                rect = QRectF(chart.left(), chart.bottom() + 3, 60, 17)
                alignment = Qt.AlignLeft
            elif i == n - 1:
                rect = QRectF(chart.right() - 60, chart.bottom() + 3, 60, 17)
                alignment = Qt.AlignRight
            else:
                rect = QRectF(x - 30, chart.bottom() + 3, 60, 17)
                alignment = Qt.AlignHCenter
            painter.drawText(rect, alignment | Qt.AlignTop, label)

        painter.end()


class MiniBarChart(QFrame):
    """Compact vertical bar chart used by the dashboard research snapshot."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("dashboardChartCard")
        self._title = title
        self._subtitle = subtitle
        self._labels = []
        self._values = []
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, labels, values):
        self._labels = [str(x) for x in labels]
        self._values = [float(x) for x in values]
        self.update()

    def clear_data(self):
        self._labels = []
        self._values = []
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QFontMetrics
        from PySide6.QtCore import QRectF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        card_rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor("#DCE6F3"), 1))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawRoundedRect(card_rect, 14, 14)

        painter.setPen(QColor("#17304E"))
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSize(12)
        painter.setFont(title_font)
        painter.drawText(16, 22, self._title)

        painter.setPen(QColor("#7C8BA0"))
        sub_font = QFont(self.font())
        sub_font.setPointSize(10)
        painter.setFont(sub_font)
        painter.drawText(16, 37, self._subtitle)

        chart = QRectF(16, 50, max(40, self.width() - 32), max(64, self.height() - 62))
        painter.setPen(QPen(QColor("#E8EEF6"), 1))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())

        if not self._values:
            painter.setPen(QColor("#A0ADBD"))
            painter.drawText(chart, Qt.AlignCenter, "Belum ada data riset")
            painter.end()
            return

        vmax = max(self._values) or 1.0
        count = len(self._values)

        # Horizontal rows keep provider names readable at every dashboard width.
        label_width = min(190.0, max(96.0, chart.width() * 0.40))
        value_width = 42.0
        bar_left = chart.left() + label_width + 8
        bar_right = chart.right() - value_width - 8
        bar_width = max(18.0, bar_right - bar_left)
        row_height = chart.height() / max(1, count)
        bar_height = min(10.0, row_height * 0.42)
        label_font_metrics = QFontMetrics(sub_font)
        for index, (label, value) in enumerate(zip(self._labels, self._values)):
            center_y = chart.top() + row_height * (index + 0.5)
            label_rect = QRectF(
                chart.left(), center_y - row_height / 2,
                label_width, row_height,
            )
            painter.setPen(QColor("#60728A"))
            painter.drawText(
                label_rect,
                Qt.AlignLeft | Qt.AlignVCenter,
                label_font_metrics.elidedText(
                    label, Qt.ElideRight, int(label_width - 4)
                ),
            )

            track_rect = QRectF(
                bar_left, center_y - bar_height / 2, bar_width, bar_height
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#EDF2FA")))
            painter.drawRoundedRect(track_rect, 4, 4)
            fill_width = bar_width * max(0.0, value) / vmax
            if fill_width > 0:
                painter.setBrush(QBrush(QColor("#77A8FF")))
                painter.drawRoundedRect(
                    QRectF(bar_left, center_y - bar_height / 2, fill_width, bar_height),
                    4, 4,
                )

            painter.setPen(QColor("#60728A"))
            value_rect = QRectF(
                chart.right() - value_width, center_y - row_height / 2,
                value_width, row_height,
            )
            value_text = f"{value:g}"
            painter.drawText(value_rect, Qt.AlignRight | Qt.AlignVCenter, value_text)

        painter.end()


from app.ui.overview_page import DashboardPage
from app.ui.about_page import AboutPage

class SwitchCheckBox(QCheckBox):
    """Checkbox painted as a pill switch while keeping normal Qt behavior."""

    def __init__(self, text, colors, parent=None):
        super().__init__(text, parent)
        self._surface = QColor(colors.get("surface", "#FFFFFF"))
        self._thumb = self._surface
        self._track_on = QColor(colors.get("blue-bright", "#0D99FF"))
        self._track_off = QColor(colors.get("switch-off", "#E7E7E7"))
        self._track_disabled = QColor(colors.get("border-solid", "#C7C6C6"))
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        indicator = self.style().subElementRect(
            QStyle.SE_CheckBoxIndicator, option, self
        )
        if not indicator.isValid() or indicator.height() <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        track_color = (
            self._track_disabled
            if not self.isEnabled()
            else self._track_on if self.isChecked() else self._track_off
        )
        painter.setBrush(self._surface)
        painter.drawRect(QRectF(indicator))
        painter.setBrush(track_color)
        track = QRectF(indicator).adjusted(5, 0, -5, 0)
        painter.drawRoundedRect(track, indicator.height() / 2, indicator.height() / 2)

        painter.setBrush(self._thumb)
        thumb_diameter = max(8, round(track.height()) - 8)
        thumb_left = (
            track.right() - thumb_diameter - 4
            if self.isChecked()
            else track.left() + 4
        )
        thumb = QRectF(
            thumb_left,
            track.top() + (track.height() - thumb_diameter) / 2,
            thumb_diameter,
            thumb_diameter,
        )
        painter.drawEllipse(thumb)


class UploadDatasetDialog(QDialog):
    """Figma-style upload confirmation; settings are display-only for now."""

    def __init__(self, main_window, file_path):
        super().__init__(main_window)
        self.main_window = main_window
        self.file_path = Path(file_path)
        try:
            self.columns = [str(column) for column in pd.read_csv(self.file_path, nrows=0).columns]
        except Exception:
            self.columns = []
        self.setObjectName("uploadDatasetDialog")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        screen = main_window.screen() or QApplication.primaryScreen()
        self.available_geometry = (
            screen.availableGeometry() if screen is not None else main_window.frameGeometry()
        )
        if (
            main_window.width() > self.available_geometry.width()
            or main_window.height() > self.available_geometry.height()
        ):
            self.available_geometry = QRect(
                0,
                0,
                max(main_window.width(), self.available_geometry.width()),
                max(main_window.height(), self.available_geometry.height()),
            )
        safe_width = min(
            max(1, self.available_geometry.width() - 32),
            max(1, main_window.width() - 32),
        )
        safe_height = min(
            max(1, self.available_geometry.height() - 32),
            max(1, main_window.height() - 32),
        )
        scale_manager = getattr(main_window, "scale_manager", None)
        self.ui_scale = max(
            0.01,
            float(scale_manager.get_scale()) if scale_manager is not None else 1.0,
        )
        # MainWindow applies the selected scale to the dialog subtree after it is
        # created, so keep these base dimensions in unscaled coordinates.
        self.dialog_width = max(1, round(min(1027, safe_width / self.ui_scale)))
        self.dialog_height = max(1, round(min(757, safe_height / self.ui_scale)))
        final_width = self.dialog_width * self.ui_scale
        final_height = self.dialog_height * self.ui_scale
        self.compact_layout = final_width < 900 or final_height < 740
        self.single_column_layout = final_width < 760
        self.setProperty("compact", self.compact_layout)

        self.surface = QFrame(self)
        self.surface.setObjectName("uploadDialogSurface")
        self.surface.setAttribute(Qt.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.surface)
        self._build_ui()
        apply_card_shadow(self.surface)
        self.setMinimumSize(min(520, self.dialog_width), min(400, self.dialog_height))
        self.setMaximumSize(self.dialog_width, self.dialog_height)
        self.resize(self.dialog_width, self.dialog_height)

    def _field(self, label, widget):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10 if self.compact_layout else 20)
        caption = QLabel(label)
        caption.setObjectName("uploadDialogLabel")
        layout.addWidget(caption)
        widget.setObjectName("uploadDialogInput")
        widget.setMinimumHeight(52 if self.compact_layout else 64)
        layout.addWidget(widget)
        return container

    def _build_ui(self):
        root = QVBoxLayout(self.surface)
        root.setContentsMargins(
            26 if self.compact_layout else 43,
            24 if self.compact_layout else 47,
            26 if self.compact_layout else 51,
            20 if self.compact_layout else 42,
        )
        root.setSpacing(0)

        content_scroll = QScrollArea(self.surface)
        content_scroll.setObjectName("uploadDialogContent")
        content_scroll.setFrameShape(QFrame.NoFrame)
        content_scroll.setWidgetResizable(True)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.viewport().setObjectName("uploadDialogViewport")
        content = QWidget()
        content.setObjectName("uploadDialogContentBody")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 12 if self.compact_layout else 0)
        content_layout.setSpacing(0)
        content_scroll.setWidget(content)
        root.addWidget(content_scroll, 1)

        title = QLabel("Upload Dataset")
        title.setObjectName("uploadDialogTitle")
        content_layout.addWidget(title)
        content_layout.addSpacing(6 if self.compact_layout else 6)
        subtitle = QLabel("Pilih file CSV dan konfirmasi cara DORA membacanya.")
        subtitle.setObjectName("uploadDialogSubtitle")
        subtitle.setWordWrap(True)
        content_layout.addWidget(subtitle)
        content_layout.addSpacing(18 if self.compact_layout else 25)

        file_card = QFrame()
        file_card.setObjectName("uploadDialogFileCard")
        file_card.setFixedHeight(108 if self.compact_layout else 133)
        file_layout = QHBoxLayout(file_card)
        if self.compact_layout:
            file_layout.setContentsMargins(16, 12, 16, 12)
        else:
            file_layout.setContentsMargins(25, 20, 26, 20)
        file_layout.setSpacing(16 if self.compact_layout else 26)
        file_icon = QLabel()
        file_icon.setObjectName("uploadDialogFileIcon")
        file_icon.setAlignment(Qt.AlignCenter)
        file_icon.setFixedSize(72 if self.compact_layout else 80, 72 if self.compact_layout else 80)
        file_icon.setPixmap(
            render_icon_pixmap(
                _resource_path("assets/icons/file-text.svg"),
                max(1, round(42 * self.ui_scale)),
                file_icon.devicePixelRatioF(),
            )
        )
        file_layout.addWidget(file_icon)
        file_text = QVBoxLayout()
        file_text.setSpacing(4)
        filename = QLabel(self.file_path.name)
        filename.setObjectName("uploadDialogFileName")
        size_kb = self.file_path.stat().st_size / 1024 if self.file_path.exists() else 0
        details_text = (
            f"{max(size_kb, 0.1):.1f} KB · {len(self.columns)} kolom terdeteksi"
            if self.columns
            else f"{max(size_kb, 0.1):.1f} KB · file CSV dipilih"
        )
        details = QLabel(details_text)
        details.setObjectName("uploadDialogFileDetails")
        file_text.addWidget(filename)
        file_text.addWidget(details)
        file_layout.addLayout(file_text, 1)
        content_layout.addWidget(file_card)
        content_layout.addSpacing(18 if self.compact_layout else 27)

        dataset_name = QLineEdit(self.file_path.name)
        delimiter = QComboBox()
        delimiter.addItems(["Koma (,)" , "Titik koma (;)" , "Tab"])
        target = QComboBox()
        target.addItems(self.columns or ["Pilih kolom target"])
        if "Survived" in self.columns:
            target.setCurrentText("Survived")
        encoding = QComboBox()
        encoding.addItems(["UTF-8", "UTF-8-SIG", "Latin-1"])

        fields = QGridLayout()
        fields.setContentsMargins(0, 0, 0 if self.compact_layout else 8, 0)
        fields.setHorizontalSpacing(10 if self.compact_layout else 15)
        fields.setVerticalSpacing(14 if self.compact_layout else 25)
        field_widgets = [
            self._field("NAMA DATASET", dataset_name),
            self._field("PEMISAH", delimiter),
        ]
        target_field = self._field("KOLOM TARGET", target)
        encoding_field = self._field("ENCODING", encoding)
        target_field.layout().setSpacing(17)
        encoding_field.layout().setSpacing(17)
        field_widgets.extend((target_field, encoding_field))
        columns = 1 if self.single_column_layout else 2
        for index, field in enumerate(field_widgets):
            fields.addWidget(field, index // columns, index % columns)
        for column in range(columns):
            fields.setColumnStretch(column, 1)
        content_layout.addLayout(fields)
        content_layout.addSpacing(18 if self.compact_layout else 26)

        first_row = SwitchCheckBox(
            "Baris pertama berisi nama kolom",
            getattr(self.main_window, "_theme_tokens", {}),
        )
        first_row.setObjectName("uploadDialogCheck")
        first_row.setChecked(True)
        content_layout.addWidget(first_row)
        content_layout.addStretch(1)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0 if self.compact_layout else 8, 0)
        actions.setSpacing(10 if self.compact_layout else 14)
        actions.addStretch(1)
        cancel = QPushButton("Batal")
        cancel.setObjectName("uploadDialogCancel")
        cancel.setFixedSize(126 if self.compact_layout else 153, 52 if self.compact_layout else 63)
        cancel.clicked.connect(lambda _checked=False: self.done(QDialog.Rejected))
        confirm = QPushButton("Unggah && Pratinjau")
        confirm.setObjectName("primaryButton")
        confirm.setFixedSize(230 if self.compact_layout else 291, 52 if self.compact_layout else 63)
        confirm.clicked.connect(lambda _checked=False: self.done(QDialog.Accepted))
        actions.addWidget(cancel)
        actions.addWidget(confirm)
        root.addLayout(actions)

    def exec_with_overlay(self):
        host = self.main_window.centralWidget()
        overlay = QWidget(host)
        overlay.setObjectName("dialogOverlay")
        overlay.setAttribute(Qt.WA_StyledBackground, True)
        overlay.setGeometry(host.rect())
        overlay.show()
        overlay.raise_()
        bounds = self.available_geometry
        parent_center = self.main_window.frameGeometry().center()
        left = parent_center.x() - self.width() // 2 + 1
        top = parent_center.y() - self.height() // 2 + 1
        left = max(bounds.left() + 16, min(left, bounds.right() - self.width() + 1 - 16))
        top = max(bounds.top() + 16, min(top, bounds.bottom() - self.height() + 1 - 16))
        self.move(left, top)
        result = self.exec()
        overlay.hide()
        overlay.deleteLater()
        return result


class UploadPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.setObjectName("uploadPage")

        self.main_window = main_window

        self.build_ui()

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.viewport().setObjectName("pageViewport")
        self.scroll.viewport().setAttribute(Qt.WA_StyledBackground, True)

        container = QWidget()
        container.setObjectName("uploadContainer")
        self.main_window._apply_canvas_background(container)
        layout = QVBoxLayout(container)
        self.upload_root_layout = layout
        layout.setContentsMargins(36, 20, 25, 36)
        layout.setSpacing(7)

        header_container = QWidget()
        self.upload_header_container = header_container
        header_container.setFixedHeight(81)
        header = QHBoxLayout(header_container)
        self.header_layout = header
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        title_block = QVBoxLayout()
        self.upload_title_block = title_block
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(4)
        title = QLabel("Dataset")
        title.setObjectName("pageTitle")
        title.setFixedHeight(42)
        title.setContentsMargins(0, 1, 0, 0)
        subtitle = QLabel("Unggah, pratinjau, dan bersihkan file CSV sebelum analisis.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setFixedHeight(35)
        subtitle.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        subtitle.setContentsMargins(0, -7, 0, 0)
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)

        self.upload_button = QPushButton("Upload Dataset")
        self.upload_button.setText("")
        self.upload_button.setAccessibleName("Upload Dataset")
        self.upload_button.setObjectName("datasetPreviewUploadButton")
        self.upload_button.setFixedSize(253, 61)
        upload_button_layout = QHBoxLayout(self.upload_button)
        upload_button_layout.setContentsMargins(19, 0, 14, 0)
        upload_button_layout.setSpacing(16)
        upload_icon = QLabel()
        upload_icon.setObjectName("datasetPreviewUploadIcon")
        upload_icon.setFixedSize(32, 24)
        upload_icon.setPixmap(
            render_icon_pixmap(
                _resource_path("assets/icons/upload.svg"),
                QSize(32, 24),
                upload_icon.devicePixelRatioF(),
            )
        )
        upload_button_layout.addWidget(upload_icon)
        upload_text = QLabel("Upload Dataset")
        upload_text.setObjectName("datasetPreviewUploadText")
        upload_button_layout.addWidget(upload_text)
        upload_button_layout.addStretch(1)
        self.upload_button.clicked.connect(self.select_dataset)
        upload_button_container = QWidget()
        upload_button_container.setFixedSize(253, 71)
        upload_button_layout_outer = QVBoxLayout(upload_button_container)
        upload_button_layout_outer.setContentsMargins(0, 2, 0, 0)
        upload_button_layout_outer.setSpacing(0)
        upload_button_layout_outer.addWidget(self.upload_button, 0, Qt.AlignTop)
        header.addWidget(upload_button_container, 0, Qt.AlignTop)
        layout.addWidget(header_container)

        content_row = QHBoxLayout()
        self.content_row = content_row
        content_row.setSpacing(22)
        upload_card = QFrame()
        self.upload_card = upload_card
        upload_card.setObjectName("uploadCard")
        upload_card.setFixedHeight(781)
        upload_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        apply_card_shadow(upload_card)
        upload_layout = QVBoxLayout(upload_card)
        upload_layout.setContentsMargins(17, 18, 24, 28)
        upload_layout.setSpacing(40)

        upload_title = QLabel("Unggah dataset")
        upload_title.setObjectName("sectionTitle")
        upload_title.setObjectName("uploadPageSectionTitle")
        upload_title.setContentsMargins(18, 10, 0, 0)
        upload_title.setFixedHeight(44)
        upload_layout.addWidget(upload_title)

        dropzone = QFrame()
        self.upload_dropzone = dropzone
        dropzone.setObjectName("uploadDropzone")
        dropzone.setCursor(Qt.PointingHandCursor)
        dropzone.setMaximumWidth(895)
        dropzone.setFixedHeight(325)
        drop_layout = QVBoxLayout(dropzone)
        drop_layout.setContentsMargins(17, 19, 15, 17)
        drop_layout.setSpacing(0)
        drop_layout.addStretch(1)
        icon_label = QLabel()
        icon_label.setObjectName("uploadDropIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedSize(70, 70)
        icon_label.setPixmap(
            render_icon_pixmap(
                _resource_path("assets/icons/file-arrow-up.svg"),
                70,
                icon_label.devicePixelRatioF(),
            )
        )
        drop_layout.addWidget(icon_label, 0, Qt.AlignHCenter)
        drop_layout.addSpacing(14)

        drop_title = QLabel("Seret dan lepas file CSV di sini")
        drop_title.setObjectName("uploadDropTitle")
        drop_title.setAlignment(Qt.AlignCenter)
        drop_title.setWordWrap(True)
        drop_layout.addWidget(drop_title)

        description_row = QGridLayout()
        self.drop_description_layout = description_row
        description_row.setContentsMargins(0, 0, 0, 0)
        description_row.setAlignment(Qt.AlignHCenter)
        description_row.setSpacing(2)
        prefix = QLabel("atau")
        prefix.setObjectName("uploadDropDescription")
        prefix.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        browse_link = QPushButton("pilih dari komputer.")
        self.upload_browse_link = browse_link
        browse_link.setObjectName("uploadBrowseLink")
        browse_link.setFlat(True)
        browse_link.setCursor(Qt.PointingHandCursor)
        browse_link.clicked.connect(self.select_dataset)
        description_row.addWidget(browse_link, 0, 1, Qt.AlignLeft | Qt.AlignVCenter)
        suffix = QLabel("CSV hingga 200 MB")
        suffix.setObjectName("uploadDropDescription")
        suffix.setObjectName("uploadDropSizeLimit")
        self.drop_size_label = suffix
        description_row.addWidget(prefix, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        description_row.addWidget(suffix, 0, 2, Qt.AlignLeft | Qt.AlignVCenter)
        drop_layout.addLayout(description_row)
        drop_layout.addStretch(1)
        drop_layout.setStretch(0, 5)
        drop_layout.setStretch(drop_layout.count() - 1, 9)
        self._install_dropzone_interaction(dropzone)
        dropzone_wrap = QWidget()
        dropzone_wrap_layout = QHBoxLayout(dropzone_wrap)
        dropzone_wrap_layout.setContentsMargins(5, 0, 0, 0)
        dropzone_wrap_layout.setSpacing(0)
        dropzone_wrap_layout.addWidget(dropzone)
        dropzone_wrap.setFixedHeight(326)
        upload_layout.addWidget(dropzone_wrap)

        steps_wrap = QWidget()
        steps_row = QGridLayout(steps_wrap)
        self.steps_layout = steps_row
        self.upload_steps = []
        steps_row.setContentsMargins(0, 0, 0, 0)
        steps_row.setContentsMargins(0, 1, 0, 0)
        steps_row.setHorizontalSpacing(4)
        steps_row.setVerticalSpacing(5)
        for column, stretch in enumerate((197, 230, 253, 234)):
            steps_row.setColumnStretch(column, stretch)
        for number, heading, detail in (
            ("1", "Unggah CSV", "Pilih file Anda"),
            ("2", "Pratinjau", "Periksa beberapa baris pertama"),
            ("3", "Pemeriksaan kualitas", "Periksa Missing Values & Outliers"),
            ("4", "Analysis", "Lanjutkan ke Analysis"),
        ):
            step = QFrame()
            step.setObjectName("uploadStep")
            step.setMinimumHeight(65)
            step.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            step_layout = QHBoxLayout(step)
            step_layout.setContentsMargins(9, 6, 3, 9)
            step_layout.setSpacing(15)
            badge = QLabel(number)
            badge.setObjectName("uploadStepBadge")
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(40, 38)
            badge.setMask(QRegion(0, 0, 40, 38, QRegion.Ellipse))
            text_layout = QVBoxLayout()
            text_layout.setSpacing(2)
            step_title = QLabel(heading)
            step_title.setObjectName("uploadStepTitle")
            step_detail = QLabel(detail)
            step_detail.setObjectName("uploadStepDetail")
            step_detail.setWordWrap(True)
            text_layout.addWidget(step_title)
            text_layout.addWidget(step_detail)
            step_layout.addWidget(badge)
            step_layout.addLayout(text_layout, 1)
            self.upload_steps.append(step)
            steps_row.addWidget(step, 0, len(self.upload_steps) - 1)
        upload_layout.addWidget(steps_wrap)
        upload_layout.addSpacing(2)
        upload_layout.addStretch(1)
        content_row.addWidget(upload_card, 215)

        side_column = QVBoxLayout()
        self.upload_side_column = side_column
        side_column.setSpacing(21)
        search_card = QFrame()
        self.upload_search_card = search_card
        search_card.setObjectName("uploadSearchCard")
        search_card.setFixedHeight(245)
        apply_card_shadow(search_card)
        search_layout = QVBoxLayout(search_card)
        search_layout.setContentsMargins(22, 20, 22, 22)
        search_layout.setSpacing(0)
        search_icon = QLabel()
        search_icon.setObjectName("uploadSearchIcon")
        search_icon.setAlignment(Qt.AlignCenter)
        search_icon.setFixedSize(42, 42)
        search_icon.setPixmap(
            render_icon_pixmap(
                _resource_path("assets/icons/search.svg"),
                28,
                search_icon.devicePixelRatioF(),
            )
        )
        search_layout.addWidget(search_icon, 0, Qt.AlignLeft)
        search_layout.addSpacing(9)
        search_title = QLabel("Belum ada data")
        search_title.setObjectName("cardTitle")
        search_title.setFixedSize(152, 24)
        search_layout.addWidget(search_title)
        search_layout.addSpacing(7)
        search_description = QLabel(
            "Cari dataset publik di Kaggle dan Hugging Face, lalu impor langsung ke DORA."
        )
        search_description.setObjectName("cardDescription")
        search_description.setWordWrap(True)
        search_description.setFixedSize(392, 38)
        search_layout.addWidget(search_description)
        search_layout.addSpacing(17)
        search_button = QPushButton("Buka Dataset Search")
        search_button.setObjectName("uploadSearchButton")
        search_button.setFixedSize(253, 63)
        search_button.clicked.connect(self.main_window.open_dataset_search_page)
        search_layout.addWidget(search_button, 0, Qt.AlignLeft)
        side_column.addWidget(search_card)

        tips_card = QFrame()
        self.upload_tips_card = tips_card
        tips_card.setObjectName("uploadTipsCard")
        tips_card.setFixedHeight(515)
        apply_card_shadow(tips_card)
        tips_layout = QVBoxLayout(tips_card)
        tips_layout.setContentsMargins(22, 20, 22, 22)
        tips_layout.setSpacing(0)
        tips_title = QLabel("Tips unggah CSV")
        tips_title.setObjectName("cardTitle")
        tips_title.setFixedSize(217, 24)
        tips_layout.addWidget(tips_title)
        tips_layout.addSpacing(24)
        tips = QLabel(
            "1. Baris pertama berisi nama kolom\n"
            "2. Gunakan satu tabel per file dengan pemisah koma\n"
            "3. Pilih kolom target pada langkah berikutnya\n"
            "4. Jangan sertakan data sensitif dalam file yang dibagikan"
        )
        tips.setObjectName("uploadTipsText")
        tips.setWordWrap(True)
        tips.setFixedSize(355, 76)
        tips_layout.addWidget(tips)
        tips_layout.addStretch(1)
        side_column.addWidget(tips_card, 1)
        content_row.addLayout(side_column, 100)
        layout.addLayout(content_row, 1)

        self.info_label = QLabel("Belum ada dataset yang dipilih.")
        self.info_label.setObjectName("cardDescription")
        self.scroll.setWidget(container)
        self.state_stack = QStackedWidget()
        self.state_stack.addWidget(self.scroll)
        self.loaded_page = LoadedUploadPage(self)
        self.state_stack.addWidget(self.loaded_page)
        root.addWidget(self.state_stack)

    def update_loaded_state(self, dataframe, filename=""):
        self.loaded_page.update_dataset(dataframe, filename)
        self.state_stack.setCurrentWidget(
            self.loaded_page if dataframe is not None else self.scroll
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = event.size().width()
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 700 else QBoxLayout.LeftToRight
        )
        self.upload_header_container.setFixedHeight(
            ui_px(152 if logical_width(width) < 700 else 81)
        )
        self.content_row.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 1450 else QBoxLayout.LeftToRight
        )

        description_columns = 1 if logical_width(width) < 480 else 3
        self.drop_description_layout.removeWidget(self.drop_size_label)
        if description_columns == 1:
            self.drop_description_layout.addWidget(
                self.drop_size_label, 1, 0, 1, 2, Qt.AlignHCenter
            )
        else:
            self.drop_description_layout.addWidget(
                self.drop_size_label, 0, 2, Qt.AlignLeft | Qt.AlignVCenter
            )

        card_width = self.upload_card.width()
        columns = 4 if logical_width(card_width) >= 900 else 2 if logical_width(card_width) >= 500 else 1
        for step in self.upload_steps:
            self.steps_layout.removeWidget(step)
        for index, step in enumerate(self.upload_steps):
            self.steps_layout.addWidget(step, index // columns, index % columns)

    # =====================================================
    # SELECT DATASET
    # =====================================================

    def _install_dropzone_interaction(self, dropzone):
        self._dropzone_targets = [dropzone, *dropzone.findChildren(QWidget)]
        for target in self._dropzone_targets:
            target.setAcceptDrops(True)
            target.installEventFilter(self)
            if target is not self.upload_browse_link:
                target.setCursor(Qt.PointingHandCursor)

    @staticmethod
    def _csv_paths_from_drop(event):
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return []
        return [Path(url.toLocalFile()) for url in mime_data.urls() if url.isLocalFile()]

    def eventFilter(self, watched, event):
        if watched in getattr(self, "_dropzone_targets", ()):
            event_type = event.type()
            if event_type in (QEvent.DragEnter, QEvent.DragMove):
                paths = self._csv_paths_from_drop(event)
                if paths and all(path.suffix.lower() == ".csv" for path in paths):
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
            if event_type == QEvent.Drop:
                paths = self._csv_paths_from_drop(event)
                if len(paths) != 1 or paths[0].suffix.lower() != ".csv":
                    QMessageBox.information(
                        self,
                        "Choose one CSV file",
                        "Drop one CSV dataset at a time.",
                    )
                else:
                    self.select_dataset(str(paths[0]))
                event.acceptProposedAction()
                return True
            if (
                event_type == QEvent.MouseButtonRelease
                and event.button() == Qt.LeftButton
                and watched is not self.upload_browse_link
            ):
                self.select_dataset()
                return True
        return super().eventFilter(watched, event)

    def select_dataset(self, file_path=None):
        if not isinstance(file_path, (str, Path)) or not str(file_path):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Pilih Dataset",
                "",
                "Datasets (*.csv *.tsv *.xlsx *.xls *.json *.parquet);;CSV/TSV Files (*.csv *.tsv);;Excel Files (*.xlsx *.xls);;JSON Files (*.json);;Parquet Files (*.parquet)",
            )

        if not file_path:
            return

        file_path = str(file_path)
        if Path(file_path).suffix.lower() not in self.main_window.loader.SUPPORTED_SUFFIXES:
            QMessageBox.information(
                self,
                "Unsupported file",
                "Please choose a CSV/TSV, Excel, JSON, or Parquet dataset.",
            )
            return

        upload_dialog = UploadDatasetDialog(self.main_window, file_path)
        if upload_dialog.exec_with_overlay() != QDialog.DialogCode.Accepted:
            return

        try:

            dataframe = self.main_window.loader.load(file_path)

            self.main_window.current_dataset = (
                dataframe
            )

            self.main_window.current_file_path = (
                file_path
            )

            filename = Path(
                file_path
            ).name

            # Inisialisasi Project & Dataset Versioning (v0)
            try:
                self.main_window.current_project_id = (
                    self.main_window.repository.create_project(
                        name=filename,
                        dataset_name=filename,
                        dataset_path=str(file_path),
                    )
                )
                self.main_window.version_manager.create_initial_version(
                    self.main_window.current_project_id,
                    file_path,
                )
            except Exception as ver_err:
                print(f"Warning: create_initial_version: {ver_err}")

            file_size = (
                Path(file_path).stat().st_size
                / 1024
            )


            self.info_label.setText(
                f"<b>Nama file:</b> {filename}<br>"
                f"<b>Ukuran:</b> {file_size:.2f} KB<br>"
                f"<b>Baris:</b> {len(dataframe):,}<br>"
                f"<b>Kolom:</b> "
                f"{len(dataframe.columns):,}"
            )

            self.main_window.dashboard.update_dataset(
                dataframe,
                filename
            )

            self.main_window.update_dataset_state()

            self.main_window.open_analysis_page()

        except Exception as error:

            QMessageBox.critical(
                self,
                "Gagal mengunggah dataset",
                "Dataset gagal dimuat.\n\n"
                f"{error}"
            )

class AnalysisPage(QWidget):

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self._gemini_thread: QThread | None = None
        self._gemini_worker: GeminiExplainWorker | None = None
        self._gemini_label: QLabel | None = None
        self._gemini_status_label: QLabel | None = None
        self._visualization_dataframe: pd.DataFrame | None = None
        self._visualization_correlations = None
        self._visualization_missing_values = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker: DatasetAnalysisWorker | None = None
        self._analysis_started_at = 0.0
        self._rendered_result_signature = None
        self._deferred_analysis_result = None
        self._detailed_analysis_prompt = None
        self._detailed_analysis_loaded = False
        self._clean_preview_thread = None
        self._clean_preview_worker = None
        self._clean_version_thread = None
        self._clean_version_worker = None
        self._export_thread = None
        self._export_worker = None
        self._drop_project_id = None
        self._drop_dataframe = None

        self.build_ui()

    @staticmethod
    def _analysis_icon_pixmap(
        name: str, size: int, color: str, device_pixel_ratio: float = 1.0
    ) -> QPixmap:
        """Draw the small line icons used by the analysis frame."""
        ratio = max(1.0, float(device_pixel_ratio or 1.0))
        pixel_size = max(1, round(size * ratio))
        pixmap = QPixmap(pixel_size, pixel_size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = painter.pen()
        pen.setColor(QColor(color))
        pen.setWidthF(max(1.2, size / 12))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        scale = pixel_size / 24
        painter.scale(scale, scale)
        if name == "file-spreadsheet":
            painter.drawRoundedRect(QRectF(4, 2.5, 16, 19), 2, 2)
            painter.drawLine(9, 2.5, 9, 21.5)
            painter.drawLine(9, 9, 20, 9)
            painter.drawLine(9, 15, 20, 15)
            painter.drawLine(14.5, 9, 14.5, 21.5)
        elif name == "hash":
            painter.drawLine(9, 3, 6, 21)
            painter.drawLine(17, 3, 14, 21)
            painter.drawLine(4, 9, 20, 9)
            painter.drawLine(3, 15, 19, 15)
        elif name == "copy":
            painter.drawRoundedRect(QRectF(7, 3.5, 13.5, 16), 1.6, 1.6)
            painter.drawRoundedRect(QRectF(3.5, 7, 13.5, 15), 1.6, 1.6)
        elif name == "circle-alert":
            painter.drawEllipse(QRectF(3, 3, 18, 18))
            painter.drawLine(12, 7.5, 12, 13.5)
            painter.drawPoint(12, 17)
        elif name == "triangle-alert":
            painter.drawLine(12, 3, 21, 20)
            painter.drawLine(21, 20, 3, 20)
            painter.drawLine(3, 20, 12, 3)
            painter.drawLine(12, 8, 12, 13.5)
            painter.drawPoint(12, 17)

        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        return pixmap

    def _apply_analysis_frame_layout(self):
        """Match the 1920 px design frame and keep smaller windows usable."""
        scale = QApplication.instance().property("uiScale") or 1.0
        window_height_getter = getattr(self.main_window, "height", None)
        window_height = (
            window_height_getter() if callable(window_height_getter) else self.height()
        ) / max(float(scale), 0.01)
        available_page_width = self.width() / max(float(scale), 0.01)
        available_window_height = window_height
        available_page_height = self.height() / max(float(scale), 0.01)
        # The frame can be used whenever the page itself has room for its
        # 1,683 px reference width; the sidebar and native window chrome are
        # outside that page area and should not force a different composition.
        target_frame = (
            callable(window_height_getter)
            and available_page_width >= 1683
            and available_window_height >= 900
        )

        if target_frame:
            bottom_margin = 104 if available_page_height >= 1024 else 16
            margins = (43, 40, 60, bottom_margin)
            header_height = 84
            header_top = 8
            spacing = 28
        else:
            margins = (28, 24, 28, 16)
            header_height = 84
            header_top = 0
            spacing = 24

        frame_scale = float(scale) if target_frame else None
        self._target_frame_scale = frame_scale
        px = (
            (lambda value: round(value * frame_scale))
            if target_frame
            else ui_px
        )
        self._target_frame_layout = target_frame
        self._root_layout.setContentsMargins(*(px(value) for value in margins))
        self._header_widget.setFixedHeight(px(header_height))
        self._header_layout.setContentsMargins(0, px(header_top), 0, 0)
        if target_frame:
            self._title_label.setFixedHeight(49)
            self._subtitle_label.setFixedHeight(25)
            self.analyze_button.setFixedSize(220, 52)
            self._analyze_button_container.setFixedSize(220, 52)
            self.dataset_label.setFixedHeight(92)
            self.dataset_icon_container.setFixedSize(56, 56)
            self.dataset_metrics_pill.setFixedSize(164, 22)
            if getattr(self, "chart_type", None) is not None:
                self.chart_type.setFixedSize(244, 53)
                self.chart_column.setFixedSize(240, 53)
                control_labels = self.findChildren(QLabel, "analysisControlLabel")
                if len(control_labels) >= 2:
                    control_labels[0].setFixedWidth(98)
                    control_labels[1].setFixedWidth(49)
        for index in (1, 3):
            item = self._root_layout.itemAt(index)
            spacer = item.spacerItem() if item is not None else None
            if spacer is not None:
                spacer.changeSize(
                    0,
                    px(spacing),
                    QSizePolicy.Minimum,
                    QSizePolicy.Fixed,
                )
        self._root_layout.invalidate()
        self._root_layout.activate()

    def build_ui(self):

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self.page_scroll = QScrollArea(self)
        self.page_scroll.setObjectName("analysisPageScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setMinimumWidth(0)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("analysisPageContent")
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        root_layout = QVBoxLayout(content)
        self._root_layout = root_layout
        root_layout.setSizeConstraint(QLayout.SetNoConstraint)

        root_layout.setContentsMargins(ui_px(43), ui_px(40), ui_px(60), ui_px(104))
        root_layout.setSpacing(0)

        # =================================================
        # HEADER
        # =================================================

        header_widget = QWidget()
        self._header_widget = header_widget
        self._header_widget.setFixedHeight(ui_px(84))
        header_layout = QHBoxLayout(header_widget)
        self._header_layout = header_layout
        header_layout.setContentsMargins(0, ui_px(8), 0, 0)
        header_layout.setSpacing(0)

        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(ui_px(3))

        title = QLabel(
            "Dataset Analysis"
        )
        self._title_label = title

        title.setObjectName("analysisPageTitle")
        title.setFixedHeight(ui_px(49))

        subtitle = QLabel(
            "Pahami struktur, kualitas, statistik, dan hubungan antarvariabel dalam dataset Anda."
        )
        self._subtitle_label = subtitle

        subtitle.setObjectName("analysisSubtitle")

        subtitle.setWordWrap(True)
        subtitle.setFixedHeight(ui_px(25))

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(title_layout, 1)

        self.analyze_button = QPushButton("Jalankan Analysis")

        self.analyze_button.setObjectName(
            "primaryButton"
        )
        self.analyze_button.setProperty("analysisRun", True)

        self.analyze_button.setFixedSize(ui_px(220), ui_px(52))
        run_icon = _resource_path("assets/icons/run.svg")
        if run_icon.exists():
            self.analyze_button.setIcon(QIcon(str(run_icon)))
            self.analyze_button.setIconSize(QSize(ui_px(18), ui_px(18)))

        self.analyze_button.clicked.connect(
            self.run_analysis
        )

        analyze_button_container = QWidget()
        self._analyze_button_container = analyze_button_container
        analyze_button_container.setFixedSize(ui_px(220), ui_px(52))
        analyze_button_layout = QVBoxLayout(analyze_button_container)
        analyze_button_layout.setContentsMargins(0, 0, 0, 0)
        analyze_button_layout.setSpacing(0)
        analyze_button_layout.addWidget(self.analyze_button)
        header_layout.addWidget(analyze_button_container, 0, Qt.AlignTop)

        root_layout.addWidget(header_widget)
        root_layout.addSpacing(ui_px(28))

        # =================================================
        # DATASET READINESS SUMMARY
        # =================================================

        self.readiness_card = QFrame()
        self.readiness_card.setObjectName("readinessCard")
        readiness_layout = QVBoxLayout(self.readiness_card)
        readiness_layout.setContentsMargins(18, 16, 18, 16)
        readiness_layout.setSpacing(10)

        readiness_header = QHBoxLayout()
        self.readiness_header_layout = readiness_header
        readiness_title = QLabel("Kesiapan dataset")
        readiness_title.setObjectName("readinessTitle")
        readiness_header.addWidget(readiness_title)
        readiness_header.addStretch()
        self.readiness_status = QLabel("Belum dianalisis")
        self.readiness_status.setObjectName("readinessStatus")
        self.readiness_status.setTextFormat(Qt.PlainText)
        self.readiness_status.setProperty("state", "waiting")
        self.readiness_status.setWordWrap(True)
        readiness_header.addWidget(self.readiness_status)
        readiness_layout.addLayout(readiness_header)

        readiness_metrics = QHBoxLayout()
        readiness_metrics.setSpacing(12)
        self.readiness_rows = self._readiness_metric("Ukuran dataset")
        self.readiness_missing = self._readiness_metric("Nilai kosong")
        self.readiness_target = self._readiness_metric("Kandidat target")
        readiness_metrics.addWidget(self.readiness_rows[0], 1)
        readiness_metrics.addWidget(self.readiness_missing[0], 1)
        readiness_metrics.addWidget(self.readiness_target[0], 1)
        readiness_layout.addLayout(readiness_metrics)

        self.readiness_recommendation = QLabel(
            "Jalankan analisis untuk memeriksa kualitas data dan kandidat target."
        )
        self.readiness_recommendation.setObjectName("readinessRecommendation")
        self.readiness_recommendation.setTextFormat(Qt.PlainText)
        self.readiness_recommendation.setWordWrap(True)
        readiness_layout.addWidget(self.readiness_recommendation)
        # Readiness values continue to update from analyzer results; the Figma
        # page presents the dataset summary in its banner instead.
        self.readiness_card.hide()

        self._update_readiness()

        # =================================================
        # DATASET STATUS
        # =================================================

        self.dataset_label = QFrame()
        self.dataset_label.setObjectName(
            "analysisDatasetBanner"
        )
        self.dataset_label.setFixedHeight(ui_px(92))
        dataset_banner_layout = QHBoxLayout(self.dataset_label)
        self.dataset_banner_layout = dataset_banner_layout
        dataset_banner_layout.setContentsMargins(ui_px(22), 0, ui_px(24), 0)
        dataset_banner_layout.setSpacing(ui_px(22))

        self.dataset_icon_container = QFrame()
        self.dataset_icon_container.setObjectName("analysisDatasetIconContainer")
        self.dataset_icon_container.setFixedSize(ui_px(56), ui_px(56))
        dataset_icon_layout = QVBoxLayout(self.dataset_icon_container)
        dataset_icon_layout.setContentsMargins(0, 0, 0, 0)
        dataset_icon_layout.setAlignment(Qt.AlignCenter)
        self.dataset_icon = QLabel()
        self.dataset_icon.setObjectName("analysisDatasetIcon")
        self.dataset_icon.setPixmap(
            self._analysis_icon_pixmap(
                "file-spreadsheet", 24, "#3278F2", self.dataset_icon.devicePixelRatioF()
            )
        )
        dataset_icon_layout.addWidget(self.dataset_icon)

        self.dataset_name_label = QLabel("Belum ada dataset yang dimuat")
        self.dataset_name_label.setObjectName("analysisDatasetName")
        self.dataset_name_label.setWordWrap(False)
        self.dataset_name_label.setMinimumWidth(0)

        self.dataset_metrics_pill = QFrame()
        self.dataset_metrics_pill.setObjectName("analysisDatasetMetricsPill")
        self.dataset_metrics_pill.setFixedSize(ui_px(164), ui_px(22))
        metrics_layout = QHBoxLayout(self.dataset_metrics_pill)
        metrics_layout.setContentsMargins(ui_px(12), 0, ui_px(12), 0)
        metrics_layout.setSpacing(0)
        self.dataset_metrics_label = QLabel("")
        self.dataset_metrics_label.setObjectName("analysisDatasetMetrics")
        self.dataset_metrics_label.setWordWrap(False)
        metrics_layout.addWidget(self.dataset_metrics_label, 1, Qt.AlignCenter)
        dataset_banner_layout.addWidget(self.dataset_icon_container)
        dataset_banner_layout.addWidget(self.dataset_name_label)
        dataset_banner_layout.addStretch(1)
        dataset_banner_layout.addWidget(self.dataset_metrics_pill)
        apply_card_shadow(self.dataset_label)

        root_layout.addWidget(self.dataset_label)
        root_layout.addSpacing(ui_px(28))

        # =================================================
        # RESULT AREA
        # =================================================

        scroll = QScrollArea()
        scroll.setObjectName("analysisResultScroll")
        scroll.setMinimumWidth(0)

        scroll.setWidgetResizable(
            True
        )
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        container = QWidget()
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.result_layout = QVBoxLayout(
            container
        )
        self.result_layout.setContentsMargins(0, 0, 0, 0)

        self.result_layout.setSpacing(
            15
        )

        scroll.setWidget(
            container
        )

        root_layout.addWidget(
            scroll
        )

        self.page_scroll.setWidget(content)
        page_layout.addWidget(self.page_scroll)

        self.result_scroll = scroll
        self._apply_analysis_frame_layout()

        self.show_empty_state()

    # =====================================================
    # EMPTY STATE
    # =====================================================

    def show_empty_state(self):

        self.clear_results()
        self._rendered_result_signature = None

        label = QLabel(
            "Belum ada hasil analisis.\n\n"
            "Unggah dataset, lalu pilih "
            "\"Jalankan Analysis\"."
        )

        label.setObjectName(
            "emptyState"
        )

        label.setWordWrap(True)

        self.result_layout.addWidget(
            label
        )

        self.result_layout.addStretch()

    def _apply_responsive_layout(self, width):
        compact = logical_width(width) < 760
        self._title_label.setMinimumHeight(self._title_label.sizeHint().height())
        self._subtitle_label.setMinimumWidth(0)
        self._subtitle_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        frame_scale = getattr(self, "_target_frame_scale", None)
        px = (
            (lambda value: round(value * frame_scale))
            if frame_scale is not None
            else ui_px
        )
        horizontal_margin = px(32 if compact else 61)
        button_width = 0 if compact else self.analyze_button.width()
        subtitle_width = max(1, width - horizontal_margin - button_width)
        subtitle_height = self._subtitle_label.heightForWidth(subtitle_width)
        self._subtitle_label.setFixedHeight(max(px(25 if self._target_frame_layout else 28), subtitle_height))
        if compact:
            self._header_widget.setMinimumHeight(
                self._title_label.minimumHeight()
                + self._subtitle_label.minimumHeight()
                + self.analyze_button.minimumHeight()
                + px(18)
            )
        else:
            self._header_widget.setMinimumHeight(0)
        self._root_layout.setContentsMargins(
            ui_px(16 if compact else 36),
            ui_px(16 if compact else 29),
            ui_px(16 if compact else 25),
            ui_px(12 if compact else 16),
        )
        self._header_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
        )
        self.readiness_header_layout.setDirection(
            QBoxLayout.TopToBottom
            if logical_width(width) < 560
            else QBoxLayout.LeftToRight
        )
        self.dataset_banner_layout.setDirection(
            QBoxLayout.TopToBottom
            if logical_width(width) < 720
            else QBoxLayout.LeftToRight
        )
        self.dataset_label.setFixedHeight(
            ui_px(112 if logical_width(width) < 720 else 86)
        )
        chart_controls = getattr(self, "chart_controls", None)
        if chart_controls is not None:
            try:
                if logical_width(width) < 900:
                    chart_controls.setDirection(QBoxLayout.TopToBottom)
                    self.chart_type.setFixedWidth(max(ui_px(180), width - ui_px(80)))
                    self.chart_column.setMinimumWidth(0)
                else:
                    chart_controls.setDirection(QBoxLayout.LeftToRight)
                    self.chart_type.setFixedWidth(ui_px(338))
                    self.chart_column.setMinimumWidth(ui_px(200))
            except RuntimeError:
                # Result cards are replaced with deleteLater(); a resize callback
                # can run before Qt has cleared this Python reference.
                self.chart_controls = None

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_analysis_layout(self.width())

    @staticmethod
    def _readiness_metric(label):
        panel = QFrame()
        panel.setObjectName("readinessMetric")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        value = QLabel("—")
        value.setObjectName("readinessMetricValue")
        value.setTextFormat(Qt.PlainText)
        value.setWordWrap(True)
        caption = QLabel(label)
        caption.setObjectName("readinessMetricLabel")
        layout.addWidget(value)
        layout.addWidget(caption)
        return panel, value

    def _update_readiness(self, result=None, analyzing=False, failed=False):
        dataframe = getattr(self.main_window, "current_dataset", None)
        if dataframe is None:
            state, status = "waiting", "Dataset belum dimuat"
            rows_value = missing_value = target_value = "—"
            recommendation = "Unggah dataset untuk melihat ringkasan kesiapan."
        elif analyzing:
            state, status = "active", "Analisis berjalan"
            rows_value = f"{len(dataframe):,} baris · {len(dataframe.columns):,} kolom"
            missing_value = target_value = "Sedang diperiksa"
            recommendation = "Ringkasan akan diperbarui setelah analisis selesai."
        elif failed:
            state, status = "error", "Analisis gagal"
            rows_value = f"{len(dataframe):,} baris · {len(dataframe.columns):,} kolom"
            missing_value = target_value = "Belum tersedia"
            recommendation = "Periksa pesan error di bawah, lalu jalankan analisis kembali."
        elif not isinstance(result, dict) or not result:
            state, status = "waiting", "Belum dianalisis"
            rows_value = f"{len(dataframe):,} baris · {len(dataframe.columns):,} kolom"
            missing_value = target_value = "Belum diperiksa"
            recommendation = "Jalankan analisis untuk memeriksa kualitas data dan kandidat target."
        else:
            profile = result.get("profile") or {}
            rows_value = (
                f"{int(profile.get('rows', len(dataframe))):,} baris · "
                f"{int(profile.get('columns', len(dataframe.columns))):,} kolom"
            )
            missing_items = result.get("missing_values") or []
            missing_count = sum(
                int(item.get("missing_count", 0))
                for item in missing_items if isinstance(item, dict)
            )
            total_cells = max(1, len(dataframe) * len(dataframe.columns))
            missing_value = f"{missing_count:,} ({missing_count / total_cells:.1%})"

            fingerprint = result.get("fingerprint") or {}
            representation = fingerprint.get("representation", {}) if isinstance(fingerprint, dict) else {}
            candidates = representation.get("target_candidates", []) if isinstance(representation, dict) else []
            target_value = str(candidates[0]) if candidates else "Belum terdeteksi"

            issues = result.get("data_quality") or []
            def issue_value(issue, key, default=None):
                if isinstance(issue, dict):
                    return issue.get(key, default)
                return getattr(issue, key, default)

            priority = {"critical": 0, "warning": 1, "info": 2}
            issues = sorted(
                issues,
                key=lambda item: priority.get(str(issue_value(item, "severity", "info")).lower(), 3),
            )
            actionable = next(
                (item for item in issues if str(issue_value(item, "severity", "info")).lower() in {"critical", "warning"}),
                None,
            )
            if actionable is not None:
                severity = str(issue_value(actionable, "severity", "warning")).lower()
                state = "error" if severity == "critical" else "attention"
                status = "Perlu perhatian" if severity == "critical" else "Perlu ditinjau"
                options = issue_value(actionable, "options", []) or []
                recommendation = str(options[0]) if options else str(issue_value(actionable, "problem", "Tinjau temuan kualitas data."))
            else:
                state, status = "success", "Tidak ada temuan prioritas"
                recommendation = "Dataset dapat ditinjau lebih lanjut; periksa statistik sebelum menentukan metode analisis."

        self.readiness_rows[1].setText(rows_value)
        self.readiness_missing[1].setText(missing_value)
        self.readiness_target[1].setText(target_value)
        self.readiness_recommendation.setText(recommendation)
        self.readiness_status.setText(status)
        self.readiness_status.setProperty("state", state)
        style = self.readiness_status.style()
        style.unpolish(self.readiness_status)
        style.polish(self.readiness_status)
        self.readiness_status.update()

    # =====================================================
    # CLEAR
    # =====================================================

    def clear_results(self):

        # Result cards are destroyed asynchronously by deleteLater(). Drop
        # their references before a resize/show callback can use old layouts.
        self._deferred_analysis_result = None
        self._detailed_analysis_prompt = None
        self._detailed_analysis_loaded = False
        for attribute in (
            "chart_controls",
            "chart_type",
            "chart_column",
            "chart_message",
            "chart_canvas",
            "chart_figure",
            "visualization_card",
            "_overview_layout",
            "_overview_widget",
            "_overview_visualization_host",
            "_overview_side_column",
        ):
            setattr(self, attribute, None)
        getattr(self, "_ml_readiness_cards", {}).clear()

        while self.result_layout.count():

            item = self.result_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

    # =====================================================
    # RUN ANALYSIS
    # =====================================================

    def run_analysis(self):
        dataframe = self.main_window.current_dataset
        if dataframe is None:
            self.show_empty_state()
            return
        if self._analysis_thread is not None:
            return

        try:
            self.main_window.prepare_dataset_analysis()
        except Exception as error:
            self._show_analysis_error(error)
            return

        self._analysis_started_at = time.perf_counter()
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("Analyzing...")
        self._update_readiness(analyzing=True)
        self._show_loading_state()

        analyzers = {
            "profile": self.main_window.profiler.profile,
            "statistics": self.main_window.statistics.analyze,
            "missing_values": self.main_window.missing_analyzer.analyze,
            "outliers": self.main_window.outlier_analyzer.analyze,
            "correlations": self.main_window.correlation_analyzer.analyze,
            "fingerprint": self.main_window.fingerprint_analyzer.generate,
        }
        self._analysis_thread = QThread(self)
        self._analysis_worker = DatasetAnalysisWorker(
            analyzers,
            self.main_window.quality_diagnoser,
            dataframe,
            getattr(self.main_window.ml_page, "last_intelligence_result", {}),
            getattr(self.main_window.ml_page, "last_method_result", {}).get("recommendations", []),
        )
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.failed.connect(self._on_analysis_failed)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.failed.connect(self._analysis_thread.quit)
        self._analysis_thread.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._cleanup_analysis_worker)
        self._analysis_thread.start()

    def _show_loading_state(self):
        self.clear_results()
        self.result_layout.addWidget(LoadingCard(
            "Analisis sedang berjalan",
            "Menghitung profil, statistik, kualitas data, korelasi, dan fingerprint dataset.",
            self,
        ))
        self.result_layout.addStretch()

    def _show_analysis_error(self, error):
        self._update_readiness(failed=True)
        self.clear_results()
        label = QLabel(f"Analisis gagal.\n\n{error}")
        label.setObjectName("errorState")
        label.setWordWrap(True)
        self.result_layout.addWidget(label)
        self.result_layout.addStretch()

    def _on_analysis_finished(self, result):
        duration_ms = int((time.perf_counter() - self._analysis_started_at) * 1000)
        try:
            self.main_window.complete_dataset_analysis(result, duration_ms)
            self.display_results(result)
        except Exception as error:
            self.main_window.fail_dataset_analysis(duration_ms)
            self._show_analysis_error(error)
        finally:
            self.analyze_button.setEnabled(True)
            self.analyze_button.setText("Jalankan Analysis")

    def _on_analysis_failed(self, error):
        duration_ms = int((time.perf_counter() - self._analysis_started_at) * 1000)
        self.main_window.fail_dataset_analysis(duration_ms)
        self._show_analysis_error(error)
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("Jalankan Analysis")

    def _cleanup_analysis_worker(self):
        self._analysis_worker = None
        self._analysis_thread = None

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    def display_results(self, result):

        self._update_readiness(result)

        self.clear_results()

        profile = result.get("profile") or {}
        statistics = result.get("statistics") or []
        missing_values = result.get("missing_values") or []
        duplicates = result.get("duplicates") or {}
        outliers = result.get("outliers") or []
        correlations = result.get("correlations")
        fingerprint = result.get("fingerprint") or {}

        self._visualization_dataframe = self.main_window.current_dataset
        self._visualization_correlations = correlations
        self._visualization_missing_values = missing_values
        if not hasattr(self, "_ml_readiness_cards"):
            self._ml_readiness_cards = {}
        ml_readiness = result.get("ml_readiness") or {}

        # The first viewport mirrors the Figma analysis dashboard. Existing
        # detailed result cards follow below it in the scroll area.
        self._build_visualization_panel()
        overview = QFrame()
        overview.setObjectName("analysisOverview")
        overview.setMinimumHeight(ui_px(648))
        self._overview_widget = overview
        overview_layout = QHBoxLayout(overview)
        self._overview_layout = overview_layout
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(ui_px(20))
        visualization_host = QWidget()
        visualization_host.setObjectName("analysisVisualizationHost")
        visualization_host_layout = QVBoxLayout(visualization_host)
        visualization_host_layout.setContentsMargins(0, ui_px(11), 0, 0)
        visualization_host_layout.setSpacing(0)
        visualization_host_layout.addWidget(self.visualization_card, 0, Qt.AlignTop)
        self._overview_visualization_host = visualization_host
        overview_layout.addWidget(visualization_host, 0, Qt.AlignTop)

        side_panel = QWidget()
        side_column = QVBoxLayout(side_panel)
        side_column.setContentsMargins(0, 0, 0, 0)
        side_column.setSpacing(ui_px(24))
        side_column.addWidget(self._build_fingerprint_card(fingerprint))
        problem_definition = ml_readiness.get("problem_definition") or {}
        problem_card = self._build_readiness_summary_card(
            "Definisi Masalah",
            problem_definition.get("summary", "Target belum terdeteksi"),
            problem_definition.get("status", "Warning"),
            [
                ("Target", problem_definition.get("target") or "Target belum terdeteksi"),
                ("ML Task", problem_definition.get("task") or "Belum terdeteksi"),
                ("Metrik yang Disarankan", problem_definition.get("metrics") or "Tidak tersedia"),
                (
                    "Target Confidence",
                    f"{problem_definition['target_score']}/100"
                    if problem_definition.get("target_score") is not None
                    else "Tidak tersedia",
                ),
            ],
            "Validate the target and metrics against the research objective."
            if problem_definition.get("target")
            else "Pilih atau identifikasi target sebelum menggunakan supervised modeling.",
        )
        self._ml_readiness_cards["problem_definition"] = problem_card

        quality = ml_readiness.get("quality_summary") or {}
        quality_card = self._build_quality_summary_card(
            quality,
            result.get("data_quality") or [],
            missing_values,
            outliers,
        )
        self._ml_readiness_cards["quality_summary"] = quality_card
        side_column.addWidget(quality_card, 1)
        overview_layout.addWidget(side_panel, 0, Qt.AlignLeft | Qt.AlignTop)
        self._overview_side_column = side_panel
        self.result_layout.addWidget(overview)
        self.result_layout.addSpacing(28)
        self.result_layout.addWidget(problem_card)

        # Detailed tables, version history, and generated explanations can
        # create many widgets. Keep the first Analysis view quick and build
        # those sections only when the user asks to inspect them.
        self._deferred_analysis_result = result
        self._detailed_analysis_loaded = False
        self._add_detailed_analysis_prompt()
        self._rendered_result_signature = (
            id(result),
            id(self.main_window.current_dataset),
        )
        self._refresh_analysis_layout(self.width())
        QTimer.singleShot(0, lambda: self._refresh_analysis_layout(self.width()))

    def _add_detailed_analysis_prompt(self):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        title = QLabel("Detail analisis lanjutan")
        title.setObjectName("sectionTitle")
        description = QLabel(
            "Profil kolom, tabel kualitas data, riwayat versi, dan detail ML "
            "akan dimuat saat dibutuhkan."
        )
        description.setObjectName("cardDescription")
        description.setWordWrap(True)
        button = QPushButton("Tampilkan detail analisis")
        button.setObjectName("secondaryButton")
        button.clicked.connect(self._load_detailed_analysis)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(button, 0, Qt.AlignLeft)
        self._detailed_analysis_prompt = card
        self.result_layout.addWidget(card)

    def _load_detailed_analysis(self):
        if self._detailed_analysis_loaded:
            return
        result = self._deferred_analysis_result
        if not isinstance(result, dict):
            return

        self._detailed_analysis_loaded = True
        prompt = self._detailed_analysis_prompt
        if prompt is not None:
            self.result_layout.removeWidget(prompt)
            prompt.deleteLater()
        self._detailed_analysis_prompt = None

        self.add_cell_profiling(
            result.get("profile") or {},
            result.get("fingerprint") or {},
            result.get("statistics") or [],
        )
        self.add_cell_data_quality(result.get("data_quality") or [])
        self.add_cell_missing_values(result.get("missing_values") or [])
        self.add_cell_outliers(result.get("outliers") or [])
        self.add_versioning_section()
        self.add_cell_gemini(result)
        self.add_ml_readiness_sections(result.get("ml_readiness") or {})
        self.result_layout.addStretch()
        self._refresh_analysis_layout(self.width())

    def _append_ml_readiness_section(
        self,
        section_key: str,
        title: str,
        summary: str,
        status: str,
        headers: list[str],
        rows: list[list[Any]],
        recommendation: str = "",
        empty_state: bool = False,
    ):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        header.addWidget(title_label)
        header.addStretch(1)
        status_badge = QLabel(status)
        status_badge.setObjectName("mlReadinessStatusBadge")
        status_badge.setProperty("state", status.casefold())
        header.addWidget(status_badge)
        layout.addLayout(header)
        summary_label = QLabel(summary or "Tidak berlaku untuk dataset ini.")
        summary_label.setObjectName("cardDescription")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)
        card.setObjectName("mlReadinessSection")
        if not hasattr(self, "_ml_readiness_cards"):
            self._ml_readiness_cards = {}
        self._ml_readiness_cards[section_key] = card

        if rows:
            table = QTableWidget()
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setRowCount(len(rows))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setWordWrap(True)
            table.setAlternatingRowColors(True)
            for row_index, values in enumerate(rows):
                for column_index, value in enumerate(values):
                    if isinstance(value, (list, tuple, set)):
                        text = ", ".join(str(item) for item in value)
                    elif isinstance(value, float):
                        text = f"{value:.3f}"
                    elif value is None:
                        text = "—"
                    else:
                        text = str(value)
                    item = QTableWidgetItem(text)
                    table.setItem(row_index, column_index, item)
            if section_key == "pipeline_checklist":
                table.cellClicked.connect(self._on_ml_checklist_row_clicked)
            layout.addWidget(table)
        elif empty_state:
            empty_label = QLabel("Tidak berlaku untuk dataset ini.")
            empty_label.setObjectName("cardDescription")
            layout.addWidget(empty_label)

        if recommendation:
            recommendation_label = QLabel(f"Rekomendasi: {recommendation}")
            recommendation_label.setObjectName("cardDescription")
            recommendation_label.setWordWrap(True)
            layout.addWidget(recommendation_label)

        self.result_layout.addWidget(card)

    def add_ml_readiness_sections(self, readiness: dict[str, Any]):
        if not readiness:
            return

        overview_cards = {
            key: widget
            for key, widget in getattr(self, "_ml_readiness_cards", {}).items()
            if key in {"problem_definition", "quality_summary"}
        }
        self._ml_readiness_cards = overview_cards
        type_data = readiness.get("data_types") or {}
        type_rows = [
            [
                item.get("column"),
                item.get("detected_type"),
                item.get("unique_count"),
                item.get("sample_values", []),
                item.get("recommendation") or "—",
            ]
            for item in type_data.get("columns", [])
        ]
        type_warnings = any(item.get("numeric_as_text") for item in type_data.get("columns", []))
        type_summary = type_data.get("summary", "Tidak berlaku untuk dataset ini.")
        if type_data.get("sample_note"):
            type_summary += f" {type_data['sample_note']}."
        if type_warnings:
            type_summary += " Teks numerik perlu ditinjau."
        self._append_ml_readiness_section(
            "data_types",
            "Data Types Overview",
            type_summary,
            "Warning" if type_warnings else type_data.get("status", "OK"),
            ["Kolom", "Tipe Terdeteksi", "Nilai Unik", "Contoh Nilai", "Rekomendasi"],
            type_rows,
            "Tinjau teks numerik dan kecualikan kolom yang menyerupai ID dari Feature model.",
            empty_state=not type_rows,
        )

        distributions = readiness.get("distributions") or {}
        distribution_rows = []
        for item in distributions.get("numeric", []):
            distribution_rows.append([
                item.get("column"),
                "Numeric Distribution",
                item.get("shape"),
                f"Skewness: {item.get('skewness')}; Kurtosis: {item.get('kurtosis')}",
            ])
        for item in distributions.get("categorical", []):
            top_values = item.get("top_categories", [])
            top_text = ", ".join(
                f"{value['value']} ({value['count']})" for value in top_values
            )
            distribution_rows.append([
                item.get("column"),
                "Categorical Distribution",
                f"{item.get('cardinality')} cardinality · {item.get('unique_count')} unique",
                top_text or "Tidak berlaku untuk dataset ini.",
            ])
        distribution_summary = distributions.get("summary", "Tidak berlaku untuk dataset ini.")
        if distributions.get("sample_note"):
            distribution_summary += f" {distributions['sample_note']}."
        self._append_ml_readiness_section(
            "distribution",
            "Distribution Analysis",
            distribution_summary,
            "OK" if distribution_rows else "Warning",
            ["Kolom", "Analysis", "Hasil", "Rincian"],
            distribution_rows,
            "Tinjau distribusi yang menceng sebelum memilih transformasi atau model yang sensitif terhadap skala.",
            empty_state=not distribution_rows,
        )

        balance = readiness.get("class_balance") or {}
        balance_summary = balance.get("summary", "Target belum terdeteksi")
        if distributions.get("sample_note"):
            balance_summary += f" {distributions['sample_note']}."
        if balance.get("imbalance_ratio") is not None:
            balance_summary += f" Imbalance ratio: {balance['imbalance_ratio']:.2f}:1."
        balance_rows = [
            [item.get("class"), item.get("count"), f"{item.get('percentage', 0):.1f}%"]
            for item in balance.get("classes", [])
        ]
        self._append_ml_readiness_section(
            "class_balance",
            "Class Balance",
            balance_summary,
            balance.get("status", "OK"),
            ["Class", "Count", "Percentage"],
            balance_rows,
            "Tinjau stratified split dan metrik yang peka terhadap kelas jika data tidak seimbang."
            if balance.get("applicable") else "Tidak berlaku untuk dataset ini.",
            empty_state=not balance_rows,
        )

        leakage = readiness.get("leakage") or {}
        leakage_summary = leakage.get("summary", "Target belum terdeteksi")
        if distributions.get("sample_note"):
            leakage_summary += f" {distributions['sample_note']}."
        leakage_rows = [
            [item.get("column"), item.get("reasons", [])]
            for item in leakage.get("findings", [])
        ]
        self._append_ml_readiness_section(
            "leakage",
            "Data Leakage Check",
            leakage_summary,
            leakage.get("status", "Warning"),
            ["Feature", "Alasan"],
            leakage_rows,
            "Tinjau Feature yang ditandai dan kecualikan identifier atau kolom turunan target bila diperlukan.",
            empty_state=not leakage_rows,
        )

        split = readiness.get("split") or {}
        split_rows = [
            ["Suggested Split", f"{int((1 - split['test_size']) * 100)}/{int(split['test_size'] * 100)}" if split.get("test_size") is not None else "—"],
            ["Training Rows", split.get("train_rows", 0)],
            ["Test Rows", split.get("test_rows", 0)],
            ["Stratify", "Recommended" if split.get("stratify") else "No"],
            ["Cross-Validation", "Recommended" if split.get("cross_validation") else "Optional"],
            ["Random State", split.get("random_state", 42)],
        ]
        self._append_ml_readiness_section(
            "split",
            "Train/Test Split Recommendation",
            split.get("summary", "Tidak berlaku untuk dataset ini."),
            split.get("status", "Warning"),
            ["Setting", "Recommendation"],
            split_rows,
            split.get("recommendation", "Tinjau strategi split sebelum melatih model."),
        )

        preprocessing = readiness.get("preprocessing") or {}
        preprocessing_summary = preprocessing.get("summary", "Tidak berlaku untuk dataset ini.")
        if distributions.get("sample_note"):
            preprocessing_summary += f" {distributions['sample_note']}."
        preprocessing_rows = []
        for item in preprocessing.get("encoding", []):
            preprocessing_rows.append(["Encoding", item.get("column"), item.get("method")])
        for item in preprocessing.get("scaling", []):
            preprocessing_rows.append(["Scaling", item.get("columns", []), item.get("method")])
        if preprocessing.get("scaling_models"):
            preprocessing_rows.append([
                "Models Requiring Scaling", preprocessing["scaling_models"], "Scaling required",
            ])
        preprocessing_rows.append(["Imbalance Handling", "Target", preprocessing.get("imbalance", "None")])
        for item in preprocessing.get("missing_strategy", []):
            preprocessing_rows.append([
                "Missing Value Strategy",
                item.get("column"),
                f"{item.get('method')} ({item.get('missing_percentage', 0):.1f}% missing)",
            ])
        preprocessing_rows.append([
            "Leakage Control", "Training Pipeline", preprocessing.get("fit_note", "")
        ])
        preprocessing_recommendation_text = " ".join(filter(None, [
            preprocessing.get("recommendation"),
            "Use recommended strategies as guidance only.",
        ]))
        self._append_ml_readiness_section(
            "preprocessing",
            "Preprocessing Recommendation",
            preprocessing_summary,
            preprocessing.get("status", "OK"),
            ["Area", "Columns or Scope", "Recommendation"],
            preprocessing_rows,
            preprocessing_recommendation_text,
            empty_state=not preprocessing_rows,
        )

        selection = readiness.get("feature_selection") or {}
        selection_rows = []
        selection_rows.extend([["Constant", column, "Remove or review"] for column in selection.get("constant_columns", [])])
        selection_rows.extend([["Near-constant", column, "Tinjau variance yang rendah"] for column in selection.get("near_constant_columns", [])])
        selection_rows.extend([
            ["High Correlation", f"{item.get('left')} / {item.get('right')}", f"|r| = {item.get('correlation', 0):.3f}"]
            for item in selection.get("correlated_pairs", [])
        ])
        selection_rows.extend([
            ["Quick Feature Importance", item.get("feature"), f"{item.get('importance', 0):.4f}"]
            for item in selection.get("importance", [])
        ])
        selection_summary = selection.get("summary", "Tidak berlaku untuk dataset ini.")
        if selection.get("importance_note"):
            selection_summary += f" {selection['importance_note']}."
        self._append_ml_readiness_section(
            "feature_selection",
            "Feature Selection Hints",
            selection_summary,
            selection.get("status", "OK"),
            ["Hint", "Feature", "Result"],
            selection_rows,
            selection.get("recommendation", "Tinjau Feature sebelum melatih model."),
            empty_state=not selection_rows,
        )

        checklist = readiness.get("pipeline_checklist") or []
        done_count = sum(item.get("status") == "Done" for item in checklist)
        self._append_ml_readiness_section(
            "pipeline_checklist",
            "Checklist ML Pipeline",
            f"{done_count} of {len(checklist)} stages done; remaining stages are guidance or not available yet.",
            "OK" if checklist and done_count == len(checklist) else "Warning" if checklist else "Issue",
            ["Tahap", "Status", "Rincian"],
            [[item.get("stage"), item.get("status"), item.get("details")] for item in checklist],
            "Recommendations are analytical suggestions, not decisions.",
            empty_state=not checklist,
        )

    def _on_ml_checklist_row_clicked(self, row: int, _column: int):
        table = self.sender()
        if not isinstance(table, QTableWidget):
            return
        stage_item = table.item(row, 0)
        if stage_item is None:
            return
        destination = {
            "Problem Definition": "data_types",
            "Data Loading": "data_types",
            "EDA": "distribution",
            "Data Cleaning": "quality_summary",
            "Train/Test Split": "split",
            "Preprocessing": "preprocessing",
            "Feature Engineering and Feature Selection": "feature_selection",
            "Model Selection": "pipeline_checklist",
            "Model Training": "pipeline_checklist",
            "Evaluation and Cross-Validation": "pipeline_checklist",
        }.get(stage_item.text())
        target_widget = getattr(self, "_ml_readiness_cards", {}).get(destination)
        if target_widget is not None:
            self.page_scroll.ensureWidgetVisible(target_widget)

    def _apply_overview_layout(self, width):
        overview_layout = getattr(self, "_overview_layout", None)
        if overview_layout is None:
            return
        if not is_qt_object_valid(overview_layout):
            self._overview_layout = None
            self._overview_widget = None
            self._overview_visualization_host = None
            self._overview_side_column = None
            self.visualization_card = None
            return
        frame_scale = getattr(self, "_target_frame_scale", None)
        px = (
            (lambda value: round(value * frame_scale))
            if frame_scale is not None
            else ui_px
        )
        stacked = (
            not getattr(self, "_target_frame_layout", False)
            and logical_width(width) < 1680
        )
        overview_layout.setDirection(
            QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight
        )
        overview = getattr(self, "_overview_widget", None)
        visualization = getattr(self, "visualization_card", None)
        visualization_host = getattr(self, "_overview_visualization_host", None)
        side_panel = getattr(self, "_overview_side_column", None)
        if not is_qt_object_valid(overview):
            overview = None
        if not is_qt_object_valid(visualization):
            visualization = None
        if not is_qt_object_valid(visualization_host):
            visualization_host = None
        if not is_qt_object_valid(side_panel):
            side_panel = None
        if stacked:
            overview_layout.setStretch(0, 1)
            overview_layout.setStretch(1, 0)
            overview_layout.setContentsMargins(0, 0, 0, 0)
            overview_layout.setSpacing(px(16))
            if overview is not None:
                overview.setMinimumHeight(0)
                overview.setMaximumHeight(16777215)
            if visualization is not None:
                visualization.setMinimumWidth(0)
                visualization.setMaximumWidth(16777215)
                visualization.setMinimumHeight(px(596))
                visualization.setMaximumHeight(16777215)
            if visualization_host is not None:
                visualization_host.setMinimumWidth(0)
                visualization_host.setMaximumWidth(16777215)
                visualization_host.setMinimumHeight(0)
                visualization_host.setMaximumHeight(16777215)
            if side_panel is not None:
                side_panel.setMinimumWidth(0)
                side_panel.setMaximumWidth(16777215)
                side_panel.setMinimumHeight(0)
                side_panel.setMaximumHeight(16777215)
        else:
            overview_layout.setStretch(0, 0)
            overview_layout.setStretch(1, 0)
            overview_layout.setContentsMargins(
                px(4), 0, px(72), 0
            )
            overview_layout.setSpacing(px(20))
            if overview is not None:
                overview.setFixedHeight(px(648))
            if visualization is not None:
                visualization.setFixedSize(px(1028), px(596))
            if visualization_host is not None:
                visualization_host.setFixedSize(px(1028), px(607))
            if side_panel is not None:
                side_panel.setFixedSize(px(456), px(648))
        overview_layout.invalidate()
        overview_layout.activate()

    def _refresh_analysis_layout(self, width=None):
        """Apply frame sizing before responsive overrides on the Analysis page."""
        width = self.width() if width is None else width
        self._apply_analysis_frame_layout()
        self._apply_responsive_layout(width)
        if getattr(self, "_target_frame_layout", False):
            # Restore the exact frame dimensions after the responsive pass.
            self._apply_analysis_frame_layout()
        self._apply_overview_layout(width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_analysis_layout(event.size().width())

    def _build_fingerprint_card(self, fingerprint):
        card = QFrame()
        card.setObjectName("analysisFingerprintCard")
        card.setFixedHeight(ui_px(172))
        layout = QVBoxLayout(card)
        layout.setContentsMargins(
            ui_px(26), ui_px(22), ui_px(26), ui_px(26)
        )
        layout.setSpacing(0)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(ui_px(12))
        icon = QLabel()
        icon.setObjectName("analysisFingerprintIcon")
        icon.setFixedSize(ui_px(20), ui_px(20))
        icon.setPixmap(
            self._analysis_icon_pixmap(
                "hash", ui_px(20), "#172235", icon.devicePixelRatioF()
            )
        )
        title_row.addWidget(icon)

        title = QLabel("Dataset Fingerprint")
        title.setObjectName("analysisPanelTitle")
        title.setFixedHeight(ui_px(29))
        title_row.addWidget(title)
        title_row.addStretch(1)
        layout.addLayout(title_row)
        layout.addSpacing(ui_px(15))

        algorithm = QLabel("SHA-256")
        algorithm.setObjectName("analysisPanelSubtitle")
        algorithm.setFixedHeight(ui_px(20))
        layout.addWidget(algorithm)
        layout.addSpacing(ui_px(12))

        hash_value = fingerprint.get("fingerprint") if isinstance(fingerprint, dict) else ""
        value_frame = QFrame()
        value_frame.setObjectName("analysisFingerprintValueContainer")
        value_frame.setFixedHeight(ui_px(44))
        value_layout = QHBoxLayout(value_frame)
        value_layout.setContentsMargins(ui_px(14), 0, ui_px(14), 0)
        value_layout.setSpacing(0)
        value = QLabel(f"{hash_value[:32]}..." if hash_value else "Tidak tersedia")
        value.setObjectName("analysisFingerprintValue")
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value.setToolTip(str(hash_value or ""))
        value_layout.addWidget(value)
        layout.addWidget(value_frame)
        return card

    def _build_readiness_summary_card(
        self,
        title: str,
        summary: str,
        status: str,
        rows: list[tuple[str, Any]],
        recommendation: str,
    ):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        header = QHBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        header.addWidget(heading, 1)
        badge = QLabel(status)
        badge.setObjectName("mlReadinessStatusBadge")
        badge.setProperty("state", status.casefold())
        header.addWidget(badge)
        layout.addLayout(header)

        summary_label = QLabel(summary)
        summary_label.setObjectName("cardDescription")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        for label_text, value in rows:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setObjectName("cardLabel")
            value_label = QLabel(
                ", ".join(map(str, value))
                if isinstance(value, (list, tuple, set))
                else str(value)
            )
            value_label.setObjectName("cardDescription")
            value_label.setWordWrap(True)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(label, 1)
            row.addWidget(value_label, 2)
            layout.addLayout(row)

        note = QLabel(f"Rekomendasi: {recommendation}")
        note.setObjectName("cardDescription")
        note.setWordWrap(True)
        layout.addWidget(note)
        apply_card_shadow(card)
        return card

    def _build_quality_summary_card(
        self, quality, quality_issues, missing_values=None, outliers=None
    ):
        """Render real analyzer counts in the quality summary from the frame."""
        card = QFrame()
        card.setObjectName("analysisQualitySummaryCard")
        card.setFixedHeight(ui_px(452))
        layout = QVBoxLayout(card)
        layout.setContentsMargins(
            ui_px(22), ui_px(18), ui_px(22), ui_px(50)
        )
        layout.setSpacing(0)

        heading = QLabel("Kualitas data")
        heading.setObjectName("analysisQualityTitle")
        heading.setFixedHeight(ui_px(29))
        layout.addWidget(heading)
        layout.addSpacing(ui_px(11))

        summary = QLabel(
            "Ringkasan Missing Values, Outliers, dan baris duplikat."
        )
        summary.setObjectName("analysisQualitySubtitle")
        summary.setWordWrap(True)
        summary.setFixedHeight(ui_px(42))
        layout.addWidget(summary)
        layout.addSpacing(ui_px(22))

        issues_by_category = {}
        for item in quality_issues or []:
            if isinstance(item, dict):
                category = item.get("category")
                severity = item.get("severity", "info")
            else:
                category = getattr(item, "category", None)
                severity = getattr(item, "severity", "info")
            if category:
                issues_by_category.setdefault(category, []).append(
                    str(severity).casefold()
                )

        def state_for(category, count):
            if not count:
                return "ok"
            severities = issues_by_category.get(category, [])
            if "critical" in severities or category == "missing_values":
                return "issue"
            return "warning"

        missing_values = missing_values or []
        missing_columns = [
            str(item.get("column"))
            for item in missing_values
            if isinstance(item, dict) and int(item.get("missing_count") or 0) > 0
        ]
        missing_count = int(
            quality.get("missing_cells")
            if quality.get("missing_cells") is not None
            else sum(
                int(item.get("missing_count") or 0)
                for item in missing_values
                if isinstance(item, dict)
            )
        )
        missing_detail = (
            f"{missing_count:,} cells across {len(missing_columns)} columns "
            f"({', '.join(missing_columns[:3])})"
            if missing_count and missing_columns
            else f"{missing_count:,} empty cells"
            if missing_count
            else "No missing values"
        )

        outliers = outliers or []
        outlier_items = [
            item for item in outliers
            if isinstance(item, dict) and int(item.get("outlier_count") or 0) > 0
        ]
        outlier_count = int(
            quality.get("outlier_values")
            if quality.get("outlier_values") is not None
            else sum(int(item.get("outlier_count") or 0) for item in outlier_items)
        )
        if outlier_count and len(outlier_items) == 1:
            item = outlier_items[0]
            outlier_detail = (
                f"{item.get('column')}: {int(item.get('outlier_count') or 0):,} rows "
                f"({float(item.get('outlier_percentage') or 0):.1f}%)"
            )
        elif outlier_count and outlier_items:
            outlier_columns = ", ".join(
                str(item.get("column")) for item in outlier_items[:3]
            )
            outlier_detail = (
                f"{outlier_count:,} values across {len(outlier_items)} columns "
                f"({outlier_columns})"
            )
        elif outlier_count:
            outlier_detail = f"{outlier_count:,} outlier values"
        else:
            outlier_detail = "No outliers found"

        duplicate_count = int(quality.get("duplicate_rows") or 0)
        duplicate_detail = f"{duplicate_count:,} duplicate rows"

        definitions = (
            ("Missing Values", missing_detail, "missing_values", missing_count, "circle-alert"),
            ("Outliers", outlier_detail, "outliers", outlier_count, "triangle-alert"),
            ("Duplicates", duplicate_detail, "duplicates", duplicate_count, "copy"),
        )
        for index, (title, detail, category, count, icon_name) in enumerate(definitions):
            row = self._build_quality_insight_row(
                title, detail, state_for(category, count), icon_name
            )
            layout.addWidget(row)
            if index < len(definitions) - 1:
                layout.addSpacing(ui_px(12))

        return card

    def _build_quality_insight_row(self, title, description, state, icon_name):
        row = QFrame()
        row.setObjectName("analysisQualityInsight")
        row.setProperty("state", state)
        row.setFixedHeight(ui_px(84))

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(ui_px(14), 0, ui_px(14), 0)
        row_layout.setSpacing(ui_px(20))

        icon_container = QFrame()
        icon_container.setObjectName("analysisQualityIcon")
        icon_container.setProperty("state", state)
        icon_container.setFixedSize(ui_px(40), ui_px(40))
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setAlignment(Qt.AlignCenter)
        icon = QLabel()
        icon.setObjectName("analysisQualityIconGlyph")
        icon.setPixmap(
            self._analysis_icon_pixmap(
                icon_name,
                ui_px(16),
                {
                    "issue": "#D94C5D",
                    "warning": "#D8892B",
                    "ok": "#26885A",
                }.get(state, "#75808F"),
                icon.devicePixelRatioF(),
            )
        )
        icon_layout.addWidget(icon)
        row_layout.addWidget(icon_container)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)
        label = QLabel(title)
        label.setObjectName("analysisQualityInsightTitle")
        label.setFixedHeight(ui_px(22))
        text_layout.addWidget(label)
        detail = QLabel(description)
        detail.setObjectName("analysisQualityInsightDescription")
        detail.setWordWrap(True)
        detail.setMaximumHeight(ui_px(38))
        text_layout.addWidget(detail)
        row_layout.addLayout(text_layout, 1)
        return row

    # =====================================================
    # VISUALIZATION
    # =====================================================

    MAX_PLOT_ROWS = 20_000
    MAX_CATEGORIES = 20

    def _build_visualization_panel(self):
        card = QFrame()
        card.setObjectName("analysisVisualizationCard")
        card.setMinimumHeight(ui_px(596))
        layout = QVBoxLayout(card)
        layout.setContentsMargins(ui_px(20), ui_px(20), ui_px(20), ui_px(20))
        layout.setSpacing(0)

        title = QLabel("Data Visualization")
        title.setObjectName("analysisVisualizationTitle")
        title.setFixedHeight(ui_px(30))
        layout.addWidget(title)
        layout.addSpacing(ui_px(18))

        controls = QHBoxLayout()
        self.chart_controls = controls
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(0)
        self.chart_type = QComboBox()
        self.chart_type.setObjectName("analysisControlSelect")
        self.chart_type.addItem("Histogram", "histogram")
        self.chart_type.addItem("Bar Chart", "bar")
        self.chart_type.addItem("Box Plot", "box")
        self.chart_type.addItem("Correlation Heatmap", "correlation")
        self.chart_type.addItem("Missing Values", "missing")
        self.chart_type.currentIndexChanged.connect(self._update_visualization_options)

        self.chart_column = QComboBox()
        self.chart_column.setObjectName("analysisControlSelect")
        self.chart_column.currentIndexChanged.connect(self._draw_visualization)
        self.chart_type.currentIndexChanged.connect(self._draw_visualization)

        chart_type_label = QLabel("Chart Type")
        chart_type_label.setObjectName("analysisControlLabel")
        chart_type_label.setFixedWidth(ui_px(98))
        chart_type_label.setFixedHeight(ui_px(24))
        controls.addWidget(chart_type_label)
        controls.addSpacing(ui_px(14))
        self.chart_type.setFixedSize(ui_px(244), ui_px(53))
        controls.addWidget(self.chart_type)
        controls.addSpacing(ui_px(26))
        column_label = QLabel("Column")
        column_label.setObjectName("analysisControlLabel")
        column_label.setFixedWidth(ui_px(49))
        column_label.setFixedHeight(ui_px(24))
        controls.addWidget(column_label)
        controls.addSpacing(ui_px(14))
        self.chart_column.setFixedSize(ui_px(240), ui_px(53))
        controls.addWidget(self.chart_column)
        layout.addLayout(controls)
        layout.addSpacing(ui_px(10))

        self.chart_message = QLabel("Jalankan analisis untuk melihat visualisasi.")
        self.chart_message.setObjectName("analysisChartMessage")
        self.chart_message.setAlignment(Qt.AlignCenter)
        self.chart_message.setMinimumHeight(ui_px(360))
        layout.addWidget(self.chart_message)

        self.chart_figure = Figure(figsize=(8, 4))
        self.chart_figure.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.16)
        self.chart_figure.patch.set_facecolor("white")
        self.chart_canvas = FigureCanvasQTAgg(self.chart_figure)
        self.chart_canvas.setObjectName("chartCanvas")
        self.chart_canvas.setMinimumHeight(ui_px(360))
        self.chart_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chart_canvas.hide()
        layout.addWidget(self.chart_canvas)

        apply_card_shadow(card)
        self.visualization_card = card
        self._update_visualization_options()

    @staticmethod
    def _chart_columns(dataframe, chart_type):
        if dataframe is None:
            return []
        if chart_type in ("histogram", "box"):
            # Do not scan every row just to populate the chart selector.
            # The selected, bounded plot sample checks for constant columns.
            return list(dataframe.select_dtypes(include=["number"]).columns)
        if chart_type == "bar":
            sample = AnalysisPage._plot_sample(dataframe)
            return [
                column for column in dataframe.columns
                if 1 < sample[column].nunique(dropna=True) <= 100
            ]
        return []

    @staticmethod
    def _plot_sample(dataframe):
        row_count = len(dataframe)
        if row_count <= AnalysisPage.MAX_PLOT_ROWS:
            return dataframe
        # Evenly spaced sampling avoids allocating random indices proportional
        # to a very large dataset while keeping the chart input bounded.
        step = max(
            1,
            (row_count + AnalysisPage.MAX_PLOT_ROWS - 1)
            // AnalysisPage.MAX_PLOT_ROWS,
        )
        return dataframe.iloc[::step].head(AnalysisPage.MAX_PLOT_ROWS)

    def _update_visualization_options(self, *_):
        if getattr(self, "chart_type", None) is None or getattr(self, "chart_column", None) is None:
            return
        chart_type = self.chart_type.currentData()
        columns = [str(column) for column in self._chart_columns(self._visualization_dataframe, chart_type)]
        current_column = self.chart_column.currentText()
        self.chart_column.blockSignals(True)
        self.chart_column.clear()
        self.chart_column.addItems(columns)
        preferred_column = current_column
        if preferred_column not in columns:
            preferred_column = next(
                (column for column in columns if column.casefold() == "age"),
                columns[0] if columns else "",
            )
        if preferred_column:
            self.chart_column.setCurrentText(preferred_column)
        self.chart_column.setEnabled(bool(columns))
        self.chart_column.blockSignals(False)
        self._draw_visualization()

    def _draw_visualization(self, *_):
        if getattr(self, "chart_figure", None) is None:
            return
        self.chart_figure.clear()
        dataframe = self._visualization_dataframe
        if dataframe is None:
            self._show_chart_message("Belum ada dataset yang dimuat untuk divisualisasikan.")
            return

        chart_type = self.chart_type.currentData()
        axis = self.chart_figure.add_subplot(111)
        sample = self._plot_sample(dataframe)

        if chart_type in ("histogram", "box", "bar"):
            if self.chart_column.currentIndex() < 0:
                self.chart_figure.clear()
                self._show_chart_message(self._empty_chart_message(chart_type))
                return
            candidates = self._chart_columns(dataframe, chart_type)
            column = candidates[self.chart_column.currentIndex()]
            column_name = str(column)
            values = sample[column].dropna()
            if chart_type == "histogram":
                if len(values) < 2 or values.nunique() < 2:
                    self._show_chart_message("This column does not have enough variation for a histogram.")
                    return
                chart_blue = "#3278F2"
                chart_grid = "#E8EDF5"
                chart_text = "#75808F"
                axis.set_facecolor("white")
                axis.hist(values, bins="auto", color=chart_blue, edgecolor="white", linewidth=0.7)
                axis.set_title(
                    f"Distribution of {column_name}", fontsize=17.6, fontweight="bold",
                    color="#172235", pad=16,
                )
                axis.set_xlabel("")
                axis.set_ylabel("")
                axis.set_axisbelow(True)
                axis.grid(axis="y", color=chart_grid, linewidth=1.2)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.tick_params(axis="both", colors=chart_text, labelsize=11, length=0, pad=8)
            elif chart_type == "box":
                if values.empty:
                    self._show_chart_message("Tidak ada nilai yang dapat ditampilkan pada box plot ini.")
                    return
                axis.boxplot(values, vert=False, patch_artist=True,
                             boxprops={"facecolor": "#DCEAFF", "color": "#4F8CFF"},
                             medianprops={"color": "#172033"})
                axis.set(title=f"Distribution of {column_name}", xlabel=column_name, yticks=[])
            else:
                counts = sample.iloc[:, dataframe.columns.get_loc(column)].dropna()
                counts = counts.astype("string").value_counts().head(self.MAX_CATEGORIES)
                if counts.empty:
                    self._show_chart_message("Tidak ada kategori yang dapat ditampilkan pada grafik ini.")
                    return
                axis.bar(counts.index.astype(str), counts.values, color="#3278F2")
                axis.set(title=f"Count by {column_name}", xlabel=column_name, ylabel="Count")
                axis.tick_params(axis="x", labelrotation=35)
                axis.margins(x=0.05)
        elif chart_type == "correlation":
            correlations = self._visualization_correlations
            if not hasattr(correlations, "empty") or correlations.empty or len(correlations.columns) < 2:
                self._show_chart_message("Heatmap memerlukan sedikitnya dua kolom numerik yang dapat dibandingkan.")
                return
            matrix = correlations.fillna(0).to_numpy()
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            names = [str(name) for name in correlations.columns]
            axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
            axis.set_yticks(range(len(names)), names)
            axis.set_title("Correlation between numeric columns")
            self.chart_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        else:
            missing = self._visualization_missing_values or []
            entries = [item for item in missing if item.get("missing_count", 0) > 0]
            if not entries:
                self._show_chart_message("Tidak ada nilai kosong pada dataset.")
                return
            names = [str(item["column"]) for item in entries[:self.MAX_CATEGORIES]]
            counts = [int(item["missing_count"]) for item in entries[:self.MAX_CATEGORIES]]
            axis.bar(names, counts, color="#3278F2")
            axis.set(title="Missing values by column", xlabel="Column", ylabel="Count")
            axis.tick_params(axis="x", labelrotation=35)

        self.chart_message.hide()
        self.chart_canvas.show()
        self.chart_canvas.draw_idle()

    @staticmethod
    def _empty_chart_message(chart_type):
        if chart_type in ("histogram", "box"):
            return "Tidak ada kolom numerik dengan variasi yang sesuai."
        if chart_type == "bar":
            return "Tidak ada kolom kategorikal yang sesuai untuk ditampilkan pada bar chart."
        if chart_type == "correlation":
            return "Correlation heatmap tidak tersedia untuk dataset ini."
        return "Tidak ada nilai kosong pada dataset."

    def _show_chart_message(self, message):
        self.chart_canvas.hide()
        self.chart_message.setText(message)
        self.chart_message.show()
        self.chart_canvas.draw_idle()

    # =====================================================
    # HUMAN-READABLE ANALYSIS
    # =====================================================

    @staticmethod
    def _number(value, default=0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _fmt_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}".rstrip("0").rstrip(".")

    @classmethod
    def format_profile(cls, profile):
        if not isinstance(profile, dict):
            return "Informasi dataset tidak tersedia."

        rows = profile.get("rows", profile.get("row_count", 0))
        columns = profile.get("columns", profile.get("column_count", 0))
        numeric = profile.get("numeric_columns", profile.get("numeric_features", 0))
        categorical = profile.get("categorical_columns", profile.get("categorical_features", 0))
        datetime = profile.get("datetime_columns", profile.get("datetime_features", 0))
        missing = profile.get("missing_values", profile.get("missing_count", 0))
        duplicates = profile.get("duplicate_rows", profile.get("duplicate_count", 0))

        lines = [
            f"{cls._fmt_number(rows)} baris dan {cls._fmt_number(columns)} kolom.",
            f"{cls._fmt_number(numeric)} kolom numerik dan {cls._fmt_number(categorical)} kolom kategorikal."
        ]

        if cls._number(datetime) > 0:
            lines.append(f"{cls._fmt_number(datetime)} kolom bertipe tanggal/waktu.")

        missing_n = cls._number(missing)
        duplicate_n = cls._number(duplicates)

        lines.append(
            f"{cls._fmt_number(missing_n)} nilai kosong ditemukan."
            if missing_n > 0
            else "Tidak ada nilai kosong."
        )
        lines.append(
            f"{cls._fmt_number(duplicate_n)} baris duplikat ditemukan."
            if duplicate_n > 0
            else "Tidak ada baris duplikat."
        )
        return "\n".join(lines)

    @classmethod
    def format_statistics(cls, statistics):
        if not isinstance(statistics, (list, tuple)) or not statistics:
            return "Statistik kolom belum tersedia."

        lines = []
        for item in statistics:
            if not isinstance(item, dict):
                continue

            name = item.get("column", item.get("name", "Kolom"))
            dtype = str(item.get("dtype", item.get("semantic_type", ""))).lower()
            missing = cls._number(item.get("missing_count", 0))
            unique = item.get("unique_count")
            numeric = item.get("numeric_statistics") or {}

            parts = [str(name)]

            if "numeric" in dtype or numeric:
                parts.append("numerik")
                if "min" in numeric and "max" in numeric:
                    parts.append(
                        f"rentang {cls._fmt_number(numeric['min'])}–{cls._fmt_number(numeric['max'])}"
                    )
                if "mean" in numeric:
                    parts.append(f"rata-rata {cls._fmt_number(numeric['mean'])}")
            else:
                if dtype:
                    parts.append(dtype)

            if unique is not None:
                parts.append(f"{cls._fmt_number(unique)} nilai unik")

            if missing > 0:
                percentage = cls._number(item.get("missing_percentage", 0))
                parts.append(
                    f"{cls._fmt_number(missing)} kosong ({percentage:.1f}%)"
                )

            lines.append(" — ".join(parts[:1]) + (" — " + " | ".join(parts[1:]) if len(parts) > 1 else ""))

        return "\n".join(lines) if lines else "Statistik kolom belum tersedia."

    @classmethod
    def format_missing(cls, missing_values):
        if not isinstance(missing_values, (list, tuple)):
            return "Informasi nilai kosong tidak tersedia."

        items = []
        for item in missing_values:
            if not isinstance(item, dict):
                continue
            count = cls._number(item.get("missing_count", item.get("count", 0)))
            if count <= 0:
                continue
            name = item.get("column", item.get("name", "Kolom"))
            percentage = cls._number(item.get("missing_percentage", item.get("percentage", 0)))
            items.append(f"{name} — {cls._fmt_number(count)} kosong ({percentage:.1f}%)")

        if not items:
            return "Tidak ada nilai kosong."
        return "\n".join(items)

    @classmethod
    def format_duplicates(cls, duplicates):
        if not isinstance(duplicates, dict):
            return "Informasi duplikat tidak tersedia."
        count = cls._number(duplicates.get("duplicate_count", duplicates.get("count", 0)))
        percentage = cls._number(duplicates.get("duplicate_percentage", duplicates.get("percentage", 0)))
        if count <= 0:
            return "Tidak ada baris duplikat (0 baris)."
        return f"{cls._fmt_number(count)} baris duplikat ({percentage:.1f}%)."

    @classmethod
    def format_outliers(cls, outliers):
        if not isinstance(outliers, (list, tuple)):
            return "Informasi outlier tidak tersedia."

        items = []
        for item in outliers:
            if not isinstance(item, dict):
                continue
            count = cls._number(item.get("outlier_count", item.get("count", 0)))
            if count <= 0:
                continue
            name = item.get("column", item.get("name", "Kolom"))
            percentage = cls._number(item.get("outlier_percentage", item.get("percentage", 0)))
            items.append(f"{name} — {cls._fmt_number(count)} outlier ({percentage:.1f}%)")

        if not items:
            return "Tidak ditemukan outlier (0)."
        return "\n".join(items)

    @classmethod
    def format_correlations(cls, correlations):
        if correlations is None:
            return "Analisis korelasi tidak tersedia."

        # Matrix pandas/DataFrame-like
        if hasattr(correlations, "columns") and hasattr(correlations, "iloc"):
            try:
                pairs = []
                columns = list(correlations.columns)
                for i, left in enumerate(columns):
                    for j in range(i + 1, len(columns)):
                        right = columns[j]
                        value = float(correlations.iloc[i, j])
                        if value == value:
                            pairs.append((abs(value), left, right, value))
                pairs.sort(reverse=True)
                if not pairs:
                    return "Tidak ada hubungan numerik yang dapat dibandingkan."
                lines = [
                    f"{left} ↔ {right}: {value:.2f}"
                    for _, left, right, value in pairs[:5]
                ]
                return "Hubungan paling kuat:\n" + "\n".join(lines)
            except Exception:
                pass

        # Nested mapping
        if isinstance(correlations, dict):
            pairs = []
            for left, values in correlations.items():
                if not isinstance(values, dict):
                    continue
                for right, value in values.items():
                    if str(left) >= str(right):
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    pairs.append((abs(numeric), left, right, numeric))
            pairs.sort(reverse=True)
            if pairs:
                return "Hubungan paling kuat:\n" + "\n".join(
                    f"{left} ↔ {right}: {value:.2f}"
                    for _, left, right, value in pairs[:5]
                )

        return "Data korelasi tersedia, tetapi belum dapat diringkas otomatis."

    def add_section(
        self,
        title,
        content
    ):

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            22, 20, 22, 20
        )

        layout.setSpacing(
            10
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "sectionTitle"
        )

        content_label = QLabel(
            str(content)
        )

        content_label.setObjectName(
            "analysisContent"
        )

        content_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            content_label
        )

        self.result_layout.addWidget(
            card
        )

        return content_label

    def _start_gemini_explanation(self, analysis: dict):
        """Generate the natural-language explanation without blocking the UI."""
        self._stop_gemini_explanation()

        self._gemini_thread = QThread(self)
        self._gemini_worker = GeminiExplainWorker(analysis)
        self._gemini_worker.moveToThread(self._gemini_thread)

        self._gemini_thread.started.connect(self._gemini_worker.run)
        self._gemini_worker.finished.connect(self._on_gemini_finished)
        self._gemini_worker.failed.connect(self._on_gemini_failed)
        self._gemini_worker.finished.connect(self._gemini_thread.quit)
        self._gemini_worker.failed.connect(self._gemini_thread.quit)
        self._gemini_worker.finished.connect(self._gemini_worker.deleteLater)
        self._gemini_worker.failed.connect(self._gemini_worker.deleteLater)
        self._gemini_thread.finished.connect(self._gemini_thread.deleteLater)
        self._gemini_thread.finished.connect(self._clear_gemini_refs)
        self._gemini_thread.start()

    def _stop_gemini_explanation(self):
        if self._gemini_thread is not None and self._gemini_thread.isRunning():
            self._gemini_thread.requestInterruption()
            self._gemini_thread.quit()

    def _clear_gemini_refs(self):
        self._gemini_thread = None
        self._gemini_worker = None

    def _on_gemini_finished(self, text: str, source: str = "local", detail: str = ""):
        if self._gemini_label is not None:
            self._gemini_label.setText(text)
        if self._gemini_status_label is not None:
            self._gemini_status_label.setText(
                "Sumber penjelasan: Gemini"
                if source == "gemini"
                else (
                    "Gemini tidak tersedia; ditampilkan penjelasan lokal. "
                    + (detail if detail else "Periksa API key dan koneksi internet.")
                )
            )

    def _on_gemini_failed(self, message: str):
        if self._gemini_label is not None:
            self._gemini_label.setText(
                "Gagal memuat interpretasi cloud. Sistem menggunakan interpretasi lokal."
            )
        if self._gemini_status_label is not None:
            self._gemini_status_label.setText(
                "Penjelasan lokal juga belum berhasil dibuat. Coba jalankan analisis kembali."
            )

    # =====================================================
    # FORMAT COLLECTION
    # =====================================================

    @staticmethod
    def format_collection(
        collection
    ):

        if collection is None:
            return "No data."

        if isinstance(
            collection,
            (list, tuple)
        ):

            if not collection:
                return "No data."

            return "\n\n".join(
                str(item)
                for item in collection
            )

        return str(collection)

    # =====================================================
    # UPDATE DATASET
    # =====================================================

    def update_dataset(
        self,
        dataframe,
        filename
    ):

        self.dataset_name_label.setText(str(filename))
        self.dataset_metrics_label.setText(
            f"{len(dataframe):,} rows × {len(dataframe.columns):,} columns"
        )

        result = getattr(self.main_window, "analysis_result", {})
        if result:
            self._update_readiness(result)
            signature = (id(result), id(dataframe))
            if signature == self._rendered_result_signature:
                return
            self.display_results(result)
        else:
            self._update_readiness()
            self.show_empty_state()

    # =========================================================
    # ANALYSIS SECTIONS
    # =========================================================

    def _create_cell_frame(
        self,
        title: str,
        subtitle: str = "",
        badge_text: str = "",
        badge_style: str = "info",
    ) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        # Header row
        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        header_row.addWidget(title_label)

        header_row.addStretch()

        if badge_text:
            colors = {
                "success": ("#EAF7F0", "#2C9B68", "#D3EFDF"),
                "warning": ("#FFF8EB", "#C27D1A", "#F5DDA6"),
                "critical": ("#FFF0F0", "#C23030", "#F5CACA"),
                "info": ("#EEF5FF", "#3D78D8", "#D7E8FF"),
            }
            bg, text_c, border_c = colors.get(badge_style, colors["info"])
            status_badge = QLabel(badge_text)
            status_badge.setStyleSheet(
                f"background: {bg}; color: {text_c}; border: 1px solid {border_c}; "
                f"border-radius: 10px; font-size: 10px; font-weight: 700; padding: 2px 10px;"
            )
            card.status_badge = status_badge
        else:
            card.status_badge = None

        layout.addLayout(header_row)

        if subtitle:
            sub_label = QLabel(subtitle)
            sub_label.setObjectName("cardDescription")
            sub_label.setWordWrap(True)
            layout.addWidget(sub_label)

        return card, layout

    # ---------------------------------------------------------
    # DATASET PROFILING
    # ---------------------------------------------------------

    def add_cell_profiling(self, profile, _fingerprint, _statistics):
        card, layout = self._create_cell_frame(
            "Ringkasan dataset",
            "Ukuran dataset, jenis kolom, dan cuplikan data.",
        )

        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)
        num_cols = profile.get("numeric_columns", 0)
        cat_cols = profile.get("categorical_columns", 0)
        mem_bytes = profile.get("memory_usage", 0)
        mem_str = f"{mem_bytes / 1024:.1f} KB" if mem_bytes < 1024 * 1024 else f"{mem_bytes / (1024 * 1024):.2f} MB"

        # Metric Chips Row
        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)
        chips_data = [
            ("Baris", f"{rows:,}"),
            ("Kolom", f"{cols:,}"),
            ("Numerik", f"{num_cols:,}"),
            ("Kategorikal", f"{cat_cols:,}"),
            ("Memori", mem_str),
        ]
        for label, val in chips_data:
            chip = QLabel(f"<b>{val}</b> <span style='color: #8492A5;'>{label}</span>")
            chip.setStyleSheet(
                "background: #F2F7FD; border: 1px solid #DCE8F7; "
                "border-radius: 6px; padding: 5px 12px; font-size: 11px;"
            )
            chips_row.addWidget(chip)
        chips_row.addStretch()
        layout.addLayout(chips_row)

        # Cuplikan data (5 baris pertama)
        df = getattr(self.main_window, "current_dataset", None)
        if df is not None and not df.empty:
            preview_title = QLabel("<b>Cuplikan data (5 baris pertama)</b>")
            preview_title.setStyleSheet("font-size: 12px; color: #1C2B40; margin-top: 6px;")
            layout.addWidget(preview_title)

            preview_head = df.head(5)
            p_table = QTableWidget()
            p_table.setColumnCount(len(preview_head.columns))
            p_table.setRowCount(len(preview_head))
            p_table.setHorizontalHeaderLabels([str(c) for c in preview_head.columns])
            p_table.verticalHeader().setVisible(True)
            p_table.setEditTriggers(QTableWidget.NoEditTriggers)

            for r in range(len(preview_head)):
                for c in range(len(preview_head.columns)):
                    val = preview_head.iloc[r, c]
                    val_str = "" if pd.isna(val) else str(val)
                    p_table.setItem(r, c, QTableWidgetItem(val_str))

            p_table.setFixedHeight(175)
            layout.addWidget(p_table)

        self.result_layout.addWidget(card)

    # ---------------------------------------------------------
    # DATA QUALITY
    # ---------------------------------------------------------

    def add_cell_data_quality(self, quality_issues):
        # Workflow data written by older versions may contain serialized issue
        # strings; newer versions persist structured dataclass dictionaries.
        if isinstance(quality_issues, (str, dict)):
            quality_issues = [quality_issues]
        normalized_issues = []
        for issue in quality_issues or []:
            if isinstance(issue, dict):
                normalized_issues.append(SimpleNamespace(
                    category=issue.get("category", ""),
                    title=issue.get("title", "Temuan kualitas data"),
                    severity=issue.get("severity", "info"),
                    problem=issue.get("problem", ""),
                    magnitude=issue.get("magnitude", ""),
                    impact=issue.get("impact", ""),
                    options=issue.get("options") or [],
                ))
            elif isinstance(issue, str):
                normalized_issues.append(SimpleNamespace(
                    category="legacy", title="Temuan kualitas data tersimpan",
                    severity="info", problem=issue, magnitude="", impact="", options=[],
                ))
            else:
                normalized_issues.append(issue)
        quality_issues = normalized_issues

        has_crit = any(i.severity == "critical" for i in quality_issues)
        has_warn = any(i.severity == "warning" for i in quality_issues)
        b_text = "Isu Kritis" if has_crit else ("Peringatan Kualitas" if has_warn else "Data Bersih")
        b_style = "critical" if has_crit else ("warning" if has_warn else "success")

        card, layout = self._create_cell_frame(
            "Kualitas data",
            "Temuan pada nilai kosong, duplikasi, dan nilai ekstrem.",
            badge_text=b_text,
            badge_style=b_style,
        )

        if not quality_issues:
            good_lbl = QLabel("<b>Kondisi Data Prima:</b> Tidak ditemukan masalah duplikat, nilai kosong signifikan, maupun outlier ekstrem.")
            good_lbl.setStyleSheet("color: #2C9B68; font-size: 12px; padding: 6px 0;")
            layout.addWidget(good_lbl)
            self.result_layout.addWidget(card)
            return

        for issue in quality_issues:
            issue_box = QFrame()
            border_c = "#F5CACA" if issue.severity == "critical" else ("#F5DDA6" if issue.severity == "warning" else "#DCE8F7")
            bg_c = "#FFF8F8" if issue.severity == "critical" else ("#FFFBF0" if issue.severity == "warning" else "#F8FAFD")
            issue_box.setStyleSheet(
                f"background: {bg_c}; border: 1px solid {border_c}; border-radius: 8px; padding: 12px 14px;"
            )
            i_layout = QVBoxLayout(issue_box)
            i_layout.setContentsMargins(0, 0, 0, 0)
            i_layout.setSpacing(6)

            sev_tag = "[Kritis]" if issue.severity == "critical" else ("[Peringatan]" if issue.severity == "warning" else "[Info]")
            sev_color = "#C23030" if issue.severity == "critical" else ("#C27D1A" if issue.severity == "warning" else "#3D78D8")
            header = QLabel(f"<span style='color: {sev_color}; font-weight: 700;'>{sev_tag}</span> <b>{issue.title}</b>")
            header.setStyleSheet("font-size: 13px; color: #172033;")
            i_layout.addWidget(header)

            p1 = QLabel(f"<b>1. Apa Masalahnya?</b> {issue.problem}")
            p1.setWordWrap(True)
            p1.setStyleSheet("font-size: 11px; color: #344158;")
            i_layout.addWidget(p1)

            p2 = QLabel(f"<b>2. Seberapa Besar Masalahnya?</b> {issue.magnitude}")
            p2.setWordWrap(True)
            p2.setStyleSheet("font-size: 11px; color: #344158;")
            i_layout.addWidget(p2)

            p3 = QLabel(f"<b>3. Apa Dampaknya?</b> {issue.impact}")
            p3.setWordWrap(True)
            p3.setStyleSheet("font-size: 11px; color: #344158;")
            i_layout.addWidget(p3)

            opt_lines = "<br>".join([f"&nbsp;&bull;&nbsp; {opt}" for opt in issue.options])
            p4 = QLabel(f"<b>4. Apa Opsi Penanganannya?</b><br>{opt_lines}")
            p4.setWordWrap(True)
            p4.setStyleSheet("font-size: 11px; color: #344158;")
            i_layout.addWidget(p4)

            # Direct Remediation Actions (Stage 5 Data Prep with Versioning)
            if issue.category == "duplicates":
                btn_row = QHBoxLayout()
                btn_row.setSpacing(8)
                dup_action_btn = QPushButton("Hapus Duplikat Sekarang (Buat Versi Baru)")
                dup_action_btn.setObjectName("secondaryButton")
                dup_action_btn.clicked.connect(self._handle_remove_duplicates)
                btn_row.addWidget(dup_action_btn)
                btn_row.addStretch()
                i_layout.addLayout(btn_row)
            elif issue.category == "missing_values":
                btn_row = QHBoxLayout()
                btn_row.setSpacing(8)
                imp_btn = QPushButton("Terapkan Imputasi (Median / Modus)")
                imp_btn.setObjectName("secondaryButton")
                imp_btn.clicked.connect(self._handle_impute_missing)
                drop_btn = QPushButton("Hapus baris dengan Missing Values")
                drop_btn.setObjectName("secondaryButton")
                drop_btn.clicked.connect(self._handle_drop_missing)
                btn_row.addWidget(imp_btn)
                btn_row.addWidget(drop_btn)
                btn_row.addStretch()
                i_layout.addLayout(btn_row)

            layout.addWidget(issue_box)

        self.result_layout.addWidget(card)

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    def add_cell_missing_values(self, missing_values):
        active_missing = [
            m for m in missing_values
            if isinstance(m, dict) and self._number(m.get("missing_count", 0)) > 0
        ]
        zero_total = sum(
            int(self._number(item.get("zero_count", 0)))
            for item in missing_values
            if isinstance(item, dict)
        )
        b_text = (
            f"{len(active_missing)} Kolom Kosong · {zero_total:,} Nilai 0"
            if active_missing or zero_total
            else "0 Nilai Kosong"
        )
        b_style = "warning" if active_missing else "success"

        card, layout = self._create_cell_frame(
            "Nilai kosong",
            "Missing value dan nilai 0 ditampilkan terpisah; nilai 0 tidak otomatis dianggap missing.",
            badge_text=b_text,
            badge_style=b_style,
        )

        self._add_dataset_version_controls(layout)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Kolom", "Missing", "Missing %", "Nilai 0", "0 %"])
        table.setRowCount(len(missing_values))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row_idx, item in enumerate(missing_values):
            if not isinstance(item, dict):
                continue
            col_name = str(item.get("column", "—"))
            count = int(self._number(item.get("missing_count", 0)))
            pct = float(self._number(item.get("missing_percentage", 0)))
            zero_count = int(self._number(item.get("zero_count", 0)))
            zero_pct = float(self._number(item.get("zero_percentage", 0)))

            table.setItem(row_idx, 0, QTableWidgetItem(col_name))

            c_item = QTableWidgetItem(f"{count:,}")
            c_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 1, c_item)

            p_item = QTableWidgetItem(f"{pct:.1f}%")
            p_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 2, p_item)

            zero_item = QTableWidgetItem(f"{zero_count:,}")
            zero_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 3, zero_item)

            zero_percent_item = QTableWidgetItem(f"{zero_pct:.1f}%")
            zero_percent_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 4, zero_percent_item)

        if missing_values:
            table.setFixedHeight(min(280, 38 + len(missing_values) * 30))
            layout.addWidget(table)
        else:
            no_columns = QLabel("Tidak ada kolom yang dapat ditampilkan.")
            layout.addWidget(no_columns)

        # Action Buttons (Stage 5 Data Prep with Versioning!)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        if active_missing:
            impute_btn = QPushButton("Terapkan Imputasi (Median / Modus)")
            impute_btn.setObjectName("secondaryButton")
            impute_btn.clicked.connect(self._handle_impute_missing)
            action_row.addWidget(impute_btn)

            drop_btn = QPushButton("Hapus baris dengan Missing Values")
            drop_btn.setObjectName("secondaryButton")
            drop_btn.clicked.connect(self._handle_drop_missing)
            action_row.addWidget(drop_btn)

        drop_button = QPushButton("Hapus Missing Values")
        drop_button.setObjectName("primaryButton")
        drop_button.clicked.connect(self._start_drop_missing_preview)
        action_row.addWidget(drop_button)

        action_row.addStretch()
        layout.addLayout(action_row)

        export_button = QPushButton("Ekspor dataset yang sudah dibersihkan")
        export_button.setObjectName("secondaryButton")
        export_button.clicked.connect(self._export_latest_cleaned_dataset)
        latest_cleaned = self._latest_cleaned_version()
        export_button.setEnabled(latest_cleaned is not None)
        layout.addWidget(export_button, 0, Qt.AlignLeft)

        self.result_layout.addWidget(card)

    def _latest_cleaned_version(self):
        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)
        if vm is None or project_id is None:
            return None
        try:
            cleaned = [
                version
                for version in vm.list_versions(project_id)
                if version.label == "Cleaned Dataset"
            ]
        except Exception:
            return None
        return cleaned[-1] if cleaned else None

    def _add_dataset_version_controls(self, parent_layout):
        row = QHBoxLayout()
        row.addWidget(QLabel("Dataset aktif"))
        version_combo = QComboBox()
        version_combo.setMinimumWidth(190)
        status = QLabel("Original")
        status.setObjectName("analysisActiveDatasetLabel")

        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)
        versions = []
        if vm is not None and project_id is not None:
            try:
                versions = vm.list_versions(project_id)
                current = vm.get_current_version(project_id)
            except Exception:
                current = None
        else:
            current = None

        version_combo.blockSignals(True)
        for version in versions:
            label = (
                "Original (v0)"
                if version.version == 0
                else f"Cleaned (v{version.version}) · {version.label}"
            )
            version_combo.addItem(label, version.version)
            if version.is_current:
                version_combo.setCurrentIndex(version_combo.count() - 1)
        if not versions:
            version_combo.addItem("Original", 0)
            version_combo.setEnabled(False)
        version_combo.blockSignals(False)

        active_version = current.version if current is not None else 0
        status.setText(
            "Original"
            if active_version == 0
            else f"Cleaned · v{active_version}"
        )
        version_combo.currentIndexChanged.connect(
            self._on_dataset_version_selected
        )
        row.addWidget(version_combo)
        row.addWidget(status)
        row.addStretch(1)
        parent_layout.addLayout(row)

    def _on_dataset_version_selected(self, index):
        combo = self.sender()
        if not isinstance(combo, QComboBox) or index < 0:
            return
        version = combo.itemData(index)
        if version is None:
            return
        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)
        current = vm.get_current_version(project_id) if vm and project_id else None
        if current is not None and current.version == int(version):
            return
        try:
            self.main_window.activate_dataset_version(int(version))
        except Exception as error:
            QMessageBox.warning(self, "Versi Dataset", str(error))

    def _start_drop_missing_preview(self):
        if (
            self._clean_preview_thread is not None
            or self._clean_version_thread is not None
        ):
            return
        dataframe = getattr(self.main_window, "current_dataset", None)
        if dataframe is None:
            QMessageBox.information(self, "Hapus Missing Values", "Muat dataset terlebih dahulu.")
            return

        dialog = DropMissingOptionsDialog(dataframe, self)
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            options = dialog.options() if accepted else None
        finally:
            # exec() is modal but the dialog is parented to the page. Schedule
            # its C++ object for deletion so its combo signals cannot outlive
            # the Python wrapper after this method returns.
            dialog.deleteLater()
        if not accepted:
            return
        if not options["columns"]:
            QMessageBox.information(
                self, "Hapus Missing Values", "Pilih setidaknya satu kolom untuk diperiksa."
            )
            return

        try:
            project_id = self.main_window._ensure_current_project()
        except Exception as error:
            QMessageBox.warning(self, "Workspace Dataset", str(error))
            return
        if project_id is None:
            QMessageBox.warning(self, "Workspace Dataset", "Workspace dataset tidak tersedia.")
            return

        self._drop_dataframe = dataframe
        self._drop_options = options
        self._drop_project_id = project_id
        self.main_window.statusBar().showMessage("Menghitung pratinjau drop missing values...")
        worker = DropMissingPreviewWorker(dataframe, options)
        self._start_background_worker(
            worker,
            "_clean_preview_thread",
            "_clean_preview_worker",
            self._on_drop_preview_ready,
        )

    def _start_background_worker(
        self,
        worker,
        thread_attribute: str,
        worker_attribute: str,
        finished_slot,
    ):
        thread = QThread(self)
        setattr(self, thread_attribute, thread)
        setattr(self, worker_attribute, worker)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(finished_slot)
        worker.failed.connect(self._on_clean_worker_failed)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda ta=thread_attribute, wa=worker_attribute: self._clear_worker_refs(ta, wa)
        )
        thread.start()

    def _clear_worker_refs(self, thread_attribute: str, worker_attribute: str):
        setattr(self, thread_attribute, None)
        setattr(self, worker_attribute, None)

    def _on_clean_worker_failed(self, message: str):
        self.main_window.statusBar().clearMessage()
        QMessageBox.warning(self, "Hapus Missing Values", message)

    def _on_drop_preview_ready(self, result: CleanResult):
        self.main_window.statusBar().clearMessage()
        strategy = result.report.get("strategy", "drop_rows")
        if strategy == "drop_rows" and result.rows_dropped == 0:
            QMessageBox.information(
                self, "Hapus Missing Values", "Tidak ada yang dihapus. Dataset baru tidak dibuat."
            )
            return

        preview = DropMissingPreviewDialog(result, self)
        if preview.exec() != QDialog.DialogCode.Accepted:
            return
        if strategy == "report":
            return
        if strategy == "drop_rows" and result.rows_after == 0:
            QMessageBox.warning(
                self,
                "Dataset Bersih Tidak Dibuat",
                "Hasil drop berisi 0 baris. Tidak ada dataset baru yang disimpan.",
            )
            return

        worker = CreateCleanedDatasetWorker(
            self._drop_dataframe,
            self._drop_options,
            self.main_window.version_manager,
            self._drop_project_id,
        )
        self.main_window.statusBar().showMessage("Menyimpan dataset bersih ke workspace...")
        self._start_background_worker(
            worker,
            "_clean_version_thread",
            "_clean_version_worker",
            self._on_cleaned_dataset_created,
        )

    def _on_cleaned_dataset_created(self, result: CleanResult, version):
        self.main_window.statusBar().clearMessage()
        if version is None:
            QMessageBox.warning(
                self,
                "Dataset Bersih Tidak Dibuat",
                "Tidak ada perubahan yang dapat disimpan atau hasilnya kosong. Dataset baru tidak dibuat.",
            )
            return
        if self.main_window.current_project_id != self._drop_project_id:
            QMessageBox.information(
                self,
                "Dataset Bersih Tersimpan",
                f"Dataset bersih tersimpan sebagai versi v{version.version}, "
                "tetapi dataset aktif telah berubah sehingga versi tidak diaktifkan.",
            )
            return

        self.main_window.current_dataset = result.cleaned_df
        self.main_window.update_dataset_state()
        self.run_analysis()

        message = QMessageBox(self)
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle("Dataset Bersih")
        message.setText("Dataset bersih berhasil dibuat. Dataset asli tidak diubah.")
        message.setInformativeText(
            f"Versi v{version.version}: {result.rows_after:,} baris tersimpan di workspace."
        )
        export_now = message.addButton("Ekspor CSV sekarang", QMessageBox.AcceptRole)
        message.addButton("Nanti Saja", QMessageBox.ActionRole)
        message.addButton("Batal", QMessageBox.RejectRole)
        message.exec()
        if message.clickedButton() is export_now:
            self._choose_cleaned_export_path(version.version)

    def _export_latest_cleaned_dataset(self):
        version = self._latest_cleaned_version()
        if version is None:
            QMessageBox.information(
                self, "Ekspor dataset bersih", "Belum ada dataset bersih di ruang kerja."
            )
            return
        self._choose_cleaned_export_path(version.version)

    def _choose_cleaned_export_path(self, version: int):
        if self._export_thread is not None:
            return
        original_path = self.main_window.current_file_path
        if not original_path:
            project = self.main_window.repository.get_project(
                self.main_window.current_project_id
            )
            original_path = project.get("dataset_path") if project else None
        original = Path(original_path) if original_path and original_path != "in-memory" else None
        dataset_name = self.main_window._current_filename()
        filename = f"{Path(dataset_name).stem}_cleaned.csv"
        initial_folder = original.parent if original is not None else Path.home()
        suggested_path = str(initial_folder / filename)

        while True:
            selected_path, _ = QFileDialog.getSaveFileName(
                self,
                "Ekspor dataset bersih",
                suggested_path,
                "CSV Files (*.csv)",
            )
            if not selected_path:
                return
            destination = Path(selected_path)
            if not destination.suffix:
                destination = destination.with_suffix(".csv")
            if original is not None and destination.resolve() == original.resolve():
                QMessageBox.warning(
                    self,
                    "File Asli Dilindungi",
                    "File tujuan sama dengan CSV asli. Pilih nama atau folder lain.",
                )
                suggested_path = selected_path
                continue
            if destination.exists():
                overwrite = QMessageBox.question(
                    self,
                    "Konfirmasi Timpa File",
                    f"File berikut sudah ada:\n{destination}\n\nTimpa file ini?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if overwrite != QMessageBox.Yes:
                    suggested_path = selected_path
                    continue
            selected_path = str(destination)
            break

        project_id = self.main_window.current_project_id
        if project_id is None:
            QMessageBox.warning(self, "Ekspor dataset bersih", "Project dataset tidak tersedia.")
            return
        worker = ExportCleanedDatasetWorker(
            self.main_window.version_manager,
            project_id,
            version,
            selected_path,
            original_path,
        )
        self.main_window.statusBar().showMessage("Mengekspor dataset bersih...")
        self._start_background_worker(
            worker,
            "_export_thread",
            "_export_worker",
            self._on_cleaned_dataset_exported,
        )

    def _on_cleaned_dataset_exported(self, path):
        self.main_window.statusBar().clearMessage()
        QMessageBox.information(
            self,
            "Export Berhasil",
            f"Dataset bersih berhasil diekspor ke:\n{path}",
        )

    # ---------------------------------------------------------
    # OUTLIERS
    # ---------------------------------------------------------

    def add_cell_outliers(self, outliers):
        active_outliers = [
            o for o in outliers
            if isinstance(o, dict) and self._number(o.get("outlier_count", 0)) > 0
        ]
        b_text = f"{len(active_outliers)} Kolom Outlier" if active_outliers else "0 Outlier"
        b_style = "info" if active_outliers else "success"

        card, layout = self._create_cell_frame(
            "Nilai ekstrem",
            "Penanda dihitung dengan metode rentang interkuartil (IQR).",
            badge_text=b_text,
            badge_style=b_style,
        )

        if not active_outliers:
            no_out = QLabel("Tidak terdeteksi nilai ekstrem di luar ambang 1.5×IQR pada seluruh fitur numerik.")
            no_out.setStyleSheet("color: #2C9B68; font-size: 12px; padding: 4px 0;")
            layout.addWidget(no_out)
            self.result_layout.addWidget(card)
            return

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Kolom", "Batas Bawah", "Batas Atas", "Jumlah Outlier", "Persentase"])
        table.setRowCount(len(active_outliers))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row_idx, item in enumerate(active_outliers):
            col_name = str(item.get("column", "—"))
            low = float(self._number(item.get("lower_bound", 0)))
            high = float(self._number(item.get("upper_bound", 0)))
            cnt = int(self._number(item.get("outlier_count", 0)))
            pct = float(self._number(item.get("outlier_percentage", 0)))

            table.setItem(row_idx, 0, QTableWidgetItem(col_name))
            table.setItem(row_idx, 1, QTableWidgetItem(self._fmt_number(low)))
            table.setItem(row_idx, 2, QTableWidgetItem(self._fmt_number(high)))
            c_item = QTableWidgetItem(f"{cnt:,}")
            c_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 3, c_item)
            p_item = QTableWidgetItem(f"{pct:.1f}%")
            p_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 4, p_item)

        table.setFixedHeight(min(160, 36 + len(active_outliers) * 30))
        layout.addWidget(table)

        tip = QLabel(
            "<b>Catatan Metodologis:</b> Outlier pada dataset penelitian ilmiah tidak selalu berupa noise. "
            "Untuk model regresi/KNN gunakan <b>RobustScaler</b>; untuk model <b>Random Forest / Gradient Boosting</b>, "
            "outlier tidak mendistorsi pembagian pohon keputusan."
        )
        tip.setStyleSheet("font-size: 11px; color: #52657D; background: #F2F7FD; border: 1px solid #DCE8F7; border-radius: 6px; padding: 8px 12px;")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        self.result_layout.addWidget(card)

    # ---------------------------------------------------------
    # CORRELATIONS
    # ---------------------------------------------------------

    def add_cell_correlations(self, correlations):
        card, layout = self._create_cell_frame(
            "Korelasi",
            "Hubungan antar kolom numerik.",
        )

        pairs = []
        if hasattr(correlations, "columns") and hasattr(correlations, "iloc"):
            try:
                cols = list(correlations.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        val = float(correlations.iloc[i, j])
                        if val == val:
                            pairs.append((abs(val), cols[i], cols[j], val))
                pairs.sort(reverse=True)
            except Exception:
                pass

        if not pairs:
            no_corr = QLabel("Tidak ada hubungan numerik yang dapat dibandingkan (kurang dari 2 kolom numerik).")
            no_corr.setStyleSheet("color: #64748B; font-size: 12px;")
            layout.addWidget(no_corr)
            self.result_layout.addWidget(card)
            return

        top_pairs = pairs[:5]
        p_lines = []
        high_corr_found = False
        for _, left, right, val in top_pairs:
            dir_str = "positif" if val > 0 else "negatif"
            strength = "kuat" if abs(val) >= 0.7 else ("sedang" if abs(val) >= 0.3 else "lemah")
            if abs(val) > 0.85:
                high_corr_found = True
            p_lines.append(f"&bull; <b>{left}</b> \u2194 <b>{right}</b>: <code style='font-weight: bold;'>{val:+.2f}</code> (Korelasi {dir_str} {strength})")

        p_label = QLabel("<br>".join(p_lines))
        p_label.setStyleSheet("font-size: 12px; color: #1E293B; line-height: 1.5;")
        layout.addWidget(p_label)

        if high_corr_found:
            mc_box = QLabel(
                "<b>Peringatan Multikolinearitas:</b> Ditemukan korelasi sangat kuat (|r| > 0.85). "
                "Fitur dengan redundansi tinggi dapat mendistorsi koefisien regresi. "
                "Pertimbangkan Feature Selection saat tahap Feature Engineering."
            )
            mc_box.setStyleSheet("background: #FFFBF0; border: 1px solid #F5DDA6; border-radius: 6px; padding: 8px 12px; font-size: 11px; color: #C27D1A;")
            mc_box.setWordWrap(True)
            layout.addWidget(mc_box)
        else:
            mc_box = QLabel("<b>Pemeriksaan korelasi:</b> Tidak ditemukan pasangan kolom dengan korelasi sangat kuat (|r| > 0.85).")
            mc_box.setStyleSheet("background: #EAF7F0; border: 1px solid #D3EFDF; border-radius: 6px; padding: 8px 12px; font-size: 11px; color: #2C9B68;")
            mc_box.setWordWrap(True)
            layout.addWidget(mc_box)

        self.result_layout.addWidget(card)

    # ---------------------------------------------------------
    # DATASET VERSIONS
    # ---------------------------------------------------------

    def add_versioning_section(self):
        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)

        if vm is None or project_id is None:
            return

        versions = vm.list_versions(project_id)
        if not versions:
            return

        current_ver = vm.get_current_version(project_id)
        trail = vm.get_trail(project_id)

        card, layout = self._create_cell_frame(
            "Riwayat dataset",
            "Versi dan perubahan yang tersimpan.",
        )

        curr_num = current_ver.version if current_ver else 0
        curr_label = current_ver.label if current_ver else "Raw Dataset"
        curr_rows = current_ver.row_count if (current_ver and current_ver.row_count is not None) else len(self.main_window.current_dataset)
        curr_cols = current_ver.column_count if (current_ver and current_ver.column_count is not None) else len(self.main_window.current_dataset.columns)

        status_badge = QLabel(
            f"<b>Versi Aktif:</b> <span style='color: #2C9B68; font-weight: bold;'>v{curr_num} — {curr_label}</span> "
            f"({curr_rows:,} baris × {curr_cols} kolom) &nbsp;&bull;&nbsp; "
            f"<span style='color: #4F8CFF; font-weight: 600;'>Original (v0) Immutable</span>"
        )
        status_badge.setObjectName("cardDescription")
        layout.addWidget(status_badge)

        # Action Buttons (Quick Data Prep)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        clean_dup_btn = QPushButton("Hapus Duplikat (Buat Versi Baru)")
        clean_dup_btn.setObjectName("secondaryButton")
        clean_dup_btn.clicked.connect(self._handle_remove_duplicates)
        action_row.addWidget(clean_dup_btn)

        export_btn = QPushButton("Ekspor Dataset Aktif (CSV)")
        export_btn.setObjectName("secondaryButton")
        export_btn.clicked.connect(self._export_active_dataset)
        action_row.addWidget(export_btn)

        action_row.addStretch()
        layout.addLayout(action_row)

        # Versions Table
        table_title = QLabel("Riwayat Versi Dataset:")
        table_title.setStyleSheet("font-weight: bold; margin-top: 4px; color: #1C2B40;")
        layout.addWidget(table_title)

        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Versi", "Label", "Baris", "Kolom", "Status", "Aksi"
        ])
        table.setRowCount(len(versions))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row_idx, ver in enumerate(versions):
            v_item = QTableWidgetItem(f"v{ver.version}")
            v_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 0, v_item)

            table.setItem(row_idx, 1, QTableWidgetItem(str(ver.label)))

            r_item = QTableWidgetItem(f"{ver.row_count:,}" if ver.row_count is not None else "—")
            r_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 2, r_item)

            c_item = QTableWidgetItem(f"{ver.column_count}" if ver.column_count is not None else "—")
            c_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 3, c_item)

            status_str = "Aktif" if ver.is_current else "Arsip"
            s_item = QTableWidgetItem(status_str)
            s_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 4, s_item)

            if not ver.is_current:
                target_v = ver.version
                rollback_btn = QPushButton(f"Rollback ke v{target_v}")
                rollback_btn.setObjectName("secondaryButton")
                rollback_btn.clicked.connect(lambda _, v=target_v: self._handle_rollback(v))
                table.setCellWidget(row_idx, 5, rollback_btn)
            else:
                curr_item = QTableWidgetItem("Versi Saat Ini")
                curr_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, 5, curr_item)

        table.setFixedHeight(min(170, 38 + len(versions) * 32))
        layout.addWidget(table)

        # Research Trail Timeline Log
        if trail:
            trail_title = QLabel("Research Trail (Audit Log Transformasi):")
            trail_title.setStyleSheet("font-weight: bold; margin-top: 8px; color: #1C2B40;")
            layout.addWidget(trail_title)

            trail_box = QTextEdit()
            trail_box.setReadOnly(True)
            trail_box.setFixedHeight(min(140, 35 + len(trail) * 32))

            lines = []
            for t in trail:
                impact_parts = [f"{k}: {v}" for k, v in t.impact.items()] if t.impact else []
                impact_str = ", ".join(impact_parts) if impact_parts else "ok"
                desc_str = f" | Catatan: {t.description}" if t.description else ""
                lines.append(
                    f"&bull; v{t.from_version} \u2192 v{t.to_version} | Operasi: {t.operation.upper()} | Dampak: [{impact_str}]{desc_str}"
                )
            trail_box.setText("\n".join(lines))
            layout.addWidget(trail_box)

        self.result_layout.addWidget(card)

    # ---------------------------------------------------------
    # AUTOMATED INTERPRETATION
    # ---------------------------------------------------------

    def add_cell_gemini(self, result):
        card, layout = self._create_cell_frame(
            "Interpretasi AI",
            "Ringkasan otomatis berdasarkan hasil analisis dataset.",
        )
        self._gemini_label = QLabel("Sedang menyusun interpretasi akademis berdasarkan fakta dataset...")
        self._gemini_label.setObjectName("analysisContent")
        self._gemini_label.setWordWrap(True)
        layout.addWidget(self._gemini_label)

        # Action row for API key configuration (optional)
        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(10)

        key_btn = QPushButton("Atur API Key (Opsional)")
        key_btn.setObjectName("secondaryButton")
        key_btn.setStyleSheet("font-size: 10px; padding: 2px 10px; min-height: 26px;")
        key_btn.clicked.connect(self._handle_configure_gemini_key)
        cfg_row.addWidget(key_btn)

        status_note = QLabel("Menghubungkan ke Gemini; jika tidak tersedia, dipakai penjelasan lokal.")
        status_note.setObjectName("cardDescription")
        status_note.setWordWrap(True)
        self._gemini_status_label = status_note
        cfg_row.addWidget(status_note)
        cfg_row.addStretch()
        layout.addLayout(cfg_row)

        telemetry = get_telemetry()
        delay_ms = int((telemetry.timeout + 0.5) * 1000) if telemetry.enabled else 0
        QTimer.singleShot(delay_ms, lambda analysis=result: self._start_gemini_explanation(analysis))
        self.result_layout.addWidget(card)

    def _handle_configure_gemini_key(self):
        current_key = get_saved_api_key() or ""
        mask_key = current_key[:8] + "..." if len(current_key) > 8 else (current_key or "(Belum diatur)")
        new_key, ok = QInputDialog.getText(
            self,
            "Pengaturan Gemini API Key",
            f"Status API Key saat ini: {mask_key}\n\n"
            "Masukkan Gemini API Key baru (opsional, kosongkan untuk menggunakan Local Engine):",
            QLineEdit.Normal,
            "",
        )
        if ok:
            save_api_key(new_key.strip())
            QMessageBox.information(
                self,
                "Pengaturan Disimpan",
                "API Key berhasil disimpan!\n\n"
                "Aplikasi akan menggunakannya untuk analisis berikutnya dengan auto-fallback lokal jika terkena limit."
            )

    # ---------------------------------------------------------
    # QUICK DATA PREP HANDLERS (STAGE 5 & VERSIONING)
    # ---------------------------------------------------------

    def _handle_impute_missing(self):
        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)
        df = self.main_window.current_dataset

        if vm is None or project_id is None or df is None:
            return

        df = normalize_missing_values(df)
        missing_total = int(df.isna().sum().sum())
        if missing_total == 0:
            QMessageBox.information(
                self,
                "Missing Values",
                "Dataset pada versi aktif saat ini sudah lengkap (0 nilai kosong)."
            )
            return

        confirm = QMessageBox.question(
            self,
            "Konfirmasi Imputasi Missing Values",
            f"Ditemukan {missing_total:,} nilai kosong pada dataset.\n\n"
            f"Sistem akan menerapkan:\n"
            f"• Median untuk kolom numerik\n"
            f"• Modus (kategori terbanyak) untuk kolom non-numerik\n\n"
            f"Versi baru akan dibuat di Research Trail, dan versi original (v0) tetap aman.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            df_imputed, cleaning_report = impute_missing_values(df)
        except ValueError as error:
            QMessageBox.warning(self, "Pembersihan Tidak Dapat Dilakukan", str(error))
            return

        new_ver = vm.create_version(
            project_id=project_id,
            dataframe=df_imputed,
            operation="impute_missing",
            parameters={"method": "median_numeric_mode_other", **cleaning_report},
            impact=cleaning_report,
            description=(
                f"Imputasi {missing_total:,} nilai kosong; "
                f"menghapus {len(cleaning_report['dropped_empty_columns'])} kolom kosong total"
            ),
            label="Missing Values Cleaned",
        )

        self.main_window.current_dataset = df_imputed
        self.main_window.dashboard.update_dataset(
            df_imputed,
            self.main_window._current_filename(),
        )
        self.run_analysis()

        export_now = QMessageBox.question(
            self,
            "Sukses Imputasi",
            f"Versi baru v{new_ver.version} ({new_ver.label}) berhasil dibuat!\n"
            f"Nilai kosong sebelum: {cleaning_report['missing_before']:,}; "
            f"sesudah: {cleaning_report['missing_after']:,}.\n"
            f"Kolom kosong total yang dihapus: {len(cleaning_report['dropped_empty_columns'])}.\n\n"
            "Ekspor dataset hasil imputasi ke file CSV sekarang?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if export_now == QMessageBox.Yes:
            self._export_active_dataset()

    def _handle_drop_missing(self):
        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)
        df = self.main_window.current_dataset

        if vm is None or project_id is None or df is None:
            return

        try:
            clean_result = drop_missing(df, strategy="drop_rows", zero_as_missing="auto")
            df_dropped = clean_result.cleaned_df
            cleaning_report = dict(clean_result.report)
            cleaning_report["rows_with_missing"] = clean_result.rows_dropped
            cleaning_report["how"] = "any"
        except ValueError as error:
            QMessageBox.warning(self, "Drop Missing Tidak Dapat Dilakukan", str(error))
            return

        rows_with_na = cleaning_report["rows_dropped"]
        if rows_with_na == 0:
            QMessageBox.information(
                self,
                "Missing Values",
                "Dataset pada versi aktif sudah lengkap. Tidak ada baris yang dihapus."
            )
            return

        warning_text = "\n".join(cleaning_report.get("warnings", []))
        if warning_text:
            warning_text = f"Peringatan:\n{warning_text}\n\n"

        confirm = QMessageBox.question(
            self,
            "Konfirmasi Drop Baris Missing",
            f"Ditemukan {cleaning_report['missing_before']:,} nilai kosong "
            f"pada {cleaning_report['rows_with_missing']:,} baris.\n\n"
            f"Nilai 0 yang dianggap missing: {cleaning_report.get('zeros_as_missing', 0):,}.\n"
            f"{warning_text}"
            f"Apakah Anda ingin menghapus baris-baris tersebut?\n"
            f"(Ukuran dataset akan berkurang dari {cleaning_report['rows_before']:,} "
            f"menjadi {cleaning_report['rows_after']:,} baris)\n\n"
            f"Versi original (v0) tetap aman dan tidak akan berubah.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        # Final guard immediately before persistence: never activate a version
        # that still contains missing cells.
        remaining_missing = int(df_dropped.isna().sum().sum())
        if remaining_missing:
            QMessageBox.critical(
                self,
                "Verifikasi Gagal",
                f"Dataset hasil drop masih berisi {remaining_missing:,} nilai kosong. "
                "Versi baru tidak dibuat.",
            )
            return

        new_ver = vm.create_version(
            project_id=project_id,
            dataframe=df_dropped,
            operation="drop_missing_rows",
            parameters=cleaning_report,
            impact=cleaning_report,
            description=f"Drop {rows_with_na:,} baris yang memiliki missing value",
            label="Dropped Missing Rows",
        )
        audit_dir = vm.data_dir / "projects" / str(project_id) / "versions"
        audit_dir.mkdir(parents=True, exist_ok=True)
        if not clean_result.dropped_df.empty:
            clean_result.dropped_df.to_csv(
                audit_dir / f"v{new_ver.version}_dropped_rows.csv",
                index=True,
                index_label="source_index",
                encoding="utf-8-sig",
            )

        self.main_window.current_dataset = df_dropped
        self.main_window.dashboard.update_dataset(
            df_dropped,
            self.main_window._current_filename(),
        )
        self.run_analysis()

        export_now = QMessageBox.question(
            self,
            "Sukses Bersihkan Missing Values",
            f"Versi baru v{new_ver.version} ({new_ver.label}) berhasil dibuat!\n"
            f"{rows_with_na:,} dari {cleaning_report['rows_before']:,} baris telah dihapus.\n"
            f"Nilai kosong: {cleaning_report['missing_before']:,} sebelum, "
            f"{remaining_missing:,} sesudah.\n\n"
            "Ekspor dataset bersih ke file CSV sekarang?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if export_now == QMessageBox.Yes:
            self._export_active_dataset()

    def _export_active_dataset(self):
        dataframe = getattr(self.main_window, "current_dataset", None)
        if dataframe is None or dataframe.empty:
            QMessageBox.information(
                self,
                "Ekspor Dataset",
                "Tidak ada dataset aktif yang dapat diekspor.",
            )
            return

        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)
        current_version = (
            vm.get_current_version(project_id)
            if vm is not None and project_id is not None
            else None
        )
        version_number = current_version.version if current_version else 0

        filename = Path(self.main_window._current_filename()).stem or "dataset"
        default_name = f"{filename}_v{version_number}_cleaned.csv"
        export_dir = get_data_dir() / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Ekspor Dataset Aktif",
            str(export_dir / default_name),
            "CSV Files (*.csv)",
        )
        if not path:
            return

        export_path = Path(path)
        if export_path.suffix.lower() != ".csv":
            export_path = export_path.with_suffix(".csv")

        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path = (
                vm.data_dir / "projects" / str(project_id) / "versions"
                / f"v{version_number}_dropped_rows.csv"
            ) if vm is not None and project_id is not None else None
            dropped_rows = pd.read_csv(audit_path, index_col=0) if audit_path is not None and audit_path.exists() else None
            export_cleaned_csv(
                dataframe,
                export_path,
                original_path=getattr(self.main_window, "current_file_path", None),
                dropped_rows=dropped_rows,
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Ekspor Gagal",
                f"Dataset tidak dapat diekspor.\n\n{error}",
            )
            return

        QMessageBox.information(
            self,
            "Ekspor Berhasil",
            f"Dataset aktif berhasil diekspor:\n{export_path}",
        )

    def _handle_remove_duplicates(self):
        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)
        df = self.main_window.current_dataset

        if vm is None or project_id is None or df is None:
            return

        dup_count = int(df.duplicated().sum())
        if dup_count == 0:
            QMessageBox.information(
                self,
                "Duplikat",
                "Dataset pada versi aktif saat ini sudah bersih dari duplikat (0 baris duplikat)."
            )
            return

        confirm = QMessageBox.question(
            self,
            "Konfirmasi Hapus Duplikat",
            f"Ditemukan {dup_count:,} baris duplikat.\n\n"
            f"Apakah Anda ingin menghapus duplikat dan membuat versi baru di Research Trail?\n"
            f"(Dataset original v0 tetap aman dan tidak akan berubah)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            return

        df_cleaned = df.drop_duplicates()
        new_ver = vm.create_version(
            project_id=project_id,
            dataframe=df_cleaned,
            operation="remove_duplicates",
            parameters={"keep": "first"},
            impact={
                "rows_before": len(df),
                "rows_after": len(df_cleaned),
                "rows_removed": dup_count,
            },
            description=f"Menghapus {dup_count:,} baris duplikat via UI",
        )

        self.main_window.current_dataset = df_cleaned
        self.main_window.dashboard.update_dataset(
            df_cleaned,
            self.main_window._current_filename(),
        )
        self.run_analysis()

        QMessageBox.information(
            self,
            "Sukses",
            f"Versi baru v{new_ver.version} ({new_ver.label}) berhasil dibuat!\n"
            f"{dup_count:,} baris duplikat telah dibersihkan."
        )

    def _handle_rollback(self, target_version: int):
        vm = getattr(self.main_window, "version_manager", None)
        project_id = getattr(self.main_window, "current_project_id", None)

        if vm is None or project_id is None:
            return

        confirm = QMessageBox.question(
            self,
            "Konfirmasi Rollback",
            f"Apakah Anda ingin rollback versi aktif ke v{target_version}?\n\n"
            f"Versi lain tidak akan dihapus dan history Research Trail tetap tersimpan.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            return

        rolled = vm.rollback_to(project_id, target_version)
        restored_df = vm.load_dataframe(project_id, target_version)

        self.main_window.current_dataset = restored_df
        self.main_window.dashboard.update_dataset(
            restored_df,
            self.main_window._current_filename(),
        )
        self.run_analysis()

        QMessageBox.information(
            self,
            "Rollback Berhasil",
            f"Dataset berhasil dikembalikan ke versi aktif: v{rolled.version} ({rolled.label})."
        )


class InfoCard(QFrame):

    def __init__(
        self,
        title: str,
        value: str = "-",
        subtitle: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName("mlInfoCard")

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.setMinimumHeight(112)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(16)

        icon_name, icon_kind = {
            "Rows": ("figma-overview-upload.svg", "blue"),
            "Columns": ("figma-overview-dataset.svg", "blue"),
            "Numeric Features": ("figma-overview-analysis.svg", "purple"),
            "Primary ML Task": ("figma-overview-ml.svg", "purple"),
            "Target": ("figma-overview-check.svg", "green"),
            "Confidence": ("figma-overview-shield.svg", "orange"),
        }.get(title, ("figma-overview-dataset.svg", "blue"))
        icon_tile = QFrame()
        icon_tile.setObjectName("mlMetricIconTile")
        icon_tile.setProperty("kind", icon_kind)
        icon_tile.setFixedSize(48, 48)
        icon_layout = QHBoxLayout(icon_tile)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon = QLabel()
        icon.setAlignment(Qt.AlignCenter)
        icon_path = _resource_path(Path("assets/icons") / icon_name)
        icon.setPixmap(
            render_icon_pixmap(icon_path, 22, icon.devicePixelRatioF())
        )
        icon_layout.addWidget(icon)
        layout.addWidget(icon_tile)

        value_layout = QVBoxLayout()
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.setSpacing(1)

        title_label = QLabel(
            title.upper()
        )

        title_label.setObjectName("mlMetricLabel")

        self.value_label = QLabel(
            value
        )

        self.value_label.setObjectName("mlMetricValue")

        self.value_label.setWordWrap(
            True
        )

        self.subtitle_label = QLabel(
            subtitle
        )

        self.subtitle_label.setObjectName("mlMetricDescription")

        self.subtitle_label.setWordWrap(
            True
        )

        value_layout.addWidget(title_label)
        value_layout.addWidget(self.value_label)
        value_layout.addWidget(self.subtitle_label)
        layout.addLayout(value_layout, 1)

    def set_value(
        self,
        value: str,
    ):
        self.value_label.setText(
            value
        )

    def set_subtitle(
        self,
        text: str,
    ):
        self.subtitle_label.setText(
            text
        )

class TaskCard(QFrame):

    def __init__(
        self,
        task: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(
            10
        )

        header = QHBoxLayout()

        label = (
            task.get("label")
            or task.get("task")
            or "Unknown Task"
        )

        title = QLabel(
            str(label)
        )

        title.setObjectName(
            "sectionTitle"
        )

        score = task.get(
            "confidence",
            task.get("score", 0),
        )

        confidence = QLabel(
            f"{score}% confidence"
        )

        confidence.setObjectName(
            "scoreBadge"
        )

        header.addWidget(
            title
        )

        header.addStretch()

        header.addWidget(
            confidence
        )

        layout.addLayout(
            header
        )

        target = task.get(
            "target"
        )

        if target:

            target_label = QLabel(
                f"Target: {target}"
            )

            target_label.setObjectName(
                "cardDescription"
            )

            layout.addWidget(
                target_label
            )

        status = task.get(
            "status",
            "ESTIMATION",
        )

        status_label = QLabel(
            f"Status: {status}"
        )

        status_label.setObjectName(
            "statusLabel"
        )

        layout.addWidget(
            status_label
        )

        reasons = task.get(
            "reasons",
            [],
        )

        if reasons:

            reasons_title = QLabel(
                "Mengapa task ini dipilih?"
            )

            reasons_title.setObjectName(
                "cardLabel"
            )

            layout.addWidget(
                reasons_title
            )

            for reason in reasons:

                reason_label = QLabel(
                    f"• {reason}"
                )

                reason_label.setObjectName(
                    "cardDescription"
                )

                reason_label.setWordWrap(
                    True
                )

                layout.addWidget(
                    reason_label
                )

class MethodCard(QFrame):

    def __init__(
        self,
        method: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.method = method

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(
            8
        )

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        header = QHBoxLayout()

        title = QLabel(
            str(
                method.get(
                    "method",
                    "Metode tidak diketahui",
                )
            )
        )

        title.setObjectName(
            "sectionTitle"
        )

        score = method.get(
            "score",
            0,
        )

        score_type = method.get(
            "score_type",
        )

        if score_type == "f1_score":
            score_text = f"F1 {float(score):.2f}%"
        elif score_type == "r2_score":
            score_text = f"R² {float(score):.4f}"
        elif score_type == "accuracy":
            score_text = f"Acc {float(score):.2f}%"
        elif score_type == "cluster_score":
            score_text = f"Cluster Score {float(score):.1f}"
        elif score_type == "silhouette_score":
            score_text = f"Silhouette {float(score):.3f}"
        elif score_type == "score_spread":
            score_text = f"Spread {float(score):.4f}"
        else:
            score_text = f"{score}% match"

        score_label = QLabel(
            score_text
        )

        score_label.setObjectName(
            "scoreBadge"
        )

        header.addWidget(
            title
        )

        header.addStretch()

        header.addWidget(
            score_label
        )

        layout.addLayout(
            header
        )

        # --------------------------------------------------
        # CATEGORY
        # --------------------------------------------------

        category = QLabel(
            f"Category: "
            f"{method.get('category', '-')}"
        )

        category.setObjectName(
            "cardDescription"
        )

        layout.addWidget(
            category
        )

        # --------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------

        description = QLabel(
            str(
                method.get(
                    "description",
                    "",
                )
            )
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            description
        )

        # --------------------------------------------------
        # REASONS
        # --------------------------------------------------

        reasons = method.get(
            "reasons",
            [],
        )

        if reasons:

            reasons_title = QLabel(
                "Mengapa model ini dipilih?"
            )

            reasons_title.setObjectName(
                "cardLabel"
            )

            layout.addWidget(
                reasons_title
            )

            for reason in reasons[:5]:

                reason_label = QLabel(
                    f"• {reason}"
                )

                reason_label.setObjectName(
                    "cardDescription"
                )

                reason_label.setWordWrap(
                    True
                )

                layout.addWidget(
                    reason_label
                )

        # --------------------------------------------------
        # QUICK INFO
        # --------------------------------------------------

        info_layout = QHBoxLayout()

        interpretability = method.get(
            "interpretability",
            "-",
        )

        scaling = method.get(
            "scaling_required",
            False,
        )

        nonlinear = method.get(
            "nonlinear",
            False,
        )

        info_layout.addWidget(
            QLabel(
                f"Interpretability: "
                f"{interpretability}"
            )
        )

        info_layout.addWidget(
            QLabel(
                f"Feature Scaling: "
                f"{'Wajib' if scaling else 'Tidak wajib'}"
            )
        )

        info_layout.addWidget(
            QLabel(
                f"Nonlinear: "
                f"{'Ya' if nonlinear else 'Tidak'}"
            )
        )

        info_layout.addStretch()

        layout.addLayout(
            info_layout
        )

class MethodRankingRow(QFrame):
    """Compact empirical result row for the three strongest evaluated methods."""

    def __init__(self, rank: int, method: dict[str, Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("mlMethodRankingRow")
        self.setMinimumHeight(72)

        score_type = str(method.get("score_type") or "")
        try:
            score = float(method.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        display_score = (
            score * 100
            if score_type in {"accuracy", "f1_score"} and abs(score) <= 1
            else score
        )

        if score_type == "f1_score":
            score_text = f"F1 {display_score:.2f}%"
        elif score_type == "r2_score":
            score_text = f"R² {score:.4f}"
        elif score_type == "accuracy":
            score_text = f"Acc {display_score:.2f}%"
        elif score_type == "cluster_score":
            score_text = f"Cluster Score {score:.1f}"
        elif score_type == "silhouette_score":
            score_text = f"Silhouette {score:.3f}"
        elif score_type == "score_spread":
            score_text = f"Spread {score:.4f}"
        else:
            score_text = f"{score:g} match"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(14)

        rank_label = QLabel(f"#{rank}")
        rank_label.setObjectName("mlMethodRank")
        rank_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(rank_label)

        identity = QVBoxLayout()
        identity.setSpacing(2)
        title = QLabel(str(method.get("method", "Metode tidak diketahui")))
        title.setObjectName("mlMethodName")
        category = QLabel(str(method.get("category", "")))
        category.setObjectName("mlMethodCategory")
        identity.addWidget(title)
        identity.addWidget(category)
        layout.addLayout(identity, 0)

        score_bar = QProgressBar()
        score_bar.setObjectName("mlMethodScoreBar")
        score_bar.setRange(0, 100)
        score_bar.setTextVisible(False)
        if score_type in {"accuracy", "f1_score"}:
            score_bar.setValue(max(0, min(100, round(display_score))))
            score_bar.setToolTip("Hasil validasi model dalam persen.")
        else:
            score_bar.hide()
        layout.addWidget(score_bar, 1)

        score_label = QLabel(score_text)
        score_label.setObjectName("mlMethodScore")
        layout.addWidget(score_label)


class MethodDetailCard(QFrame):

    def __init__(
        self,
        method: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            24,
            22,
            24,
            22,
        )

        layout.setSpacing(
            12
        )

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        title = QLabel(
            method.get(
                "method",
                "Detail Metode",
            )
        )

        title.setObjectName(
            "sectionTitle"
        )

        layout.addWidget(
            title
        )

        # --------------------------------------------------
        # EMPIRICAL METRICS (if evaluated)
        # --------------------------------------------------
        metrics = method.get("metrics", {})
        if metrics:
            metrics_title = QLabel(
                "Evaluation Metrics"
            )
            metrics_title.setObjectName(
                "cardLabel"
            )
            layout.addWidget(
                metrics_title
            )

            for key, val in metrics.items():
                label_text = str(key).replace("_", " ").title()
                if isinstance(val, float):
                    val_str = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
                else:
                    val_str = str(val)
                lbl = QLabel(
                    f"• {label_text}: {val_str}"
                )
                lbl.setObjectName(
                    "cardDescription"
                )
                layout.addWidget(
                    lbl
                )

        # --------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------

        description_title = QLabel(
            "Description"
        )

        description_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            description_title
        )

        description = QLabel(
            method.get(
                "description",
                "-",
            )
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            description
        )

        # --------------------------------------------------
        # STRENGTHS
        # --------------------------------------------------

        strengths_title = QLabel(
            "Strengths"
        )

        strengths_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            strengths_title
        )

        for item in method.get(
            "strengths",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # LIMITATIONS
        # --------------------------------------------------

        limitations_title = QLabel(
            "Limitations"
        )

        limitations_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            limitations_title
        )

        for item in method.get(
            "limitations",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------

        preprocessing_title = QLabel(
            "Preprocessing"
        )

        preprocessing_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            preprocessing_title
        )

        for item in method.get(
            "preprocessing",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # META
        # --------------------------------------------------

        meta = QLabel(
            "Interpretability: "
            f"{method.get('interpretability', '-')}"
            "\n"
            "Feature Scaling: "
            f"{'Wajib' if method.get('scaling_required') else 'Tidak wajib'}"
            "\n"
            "Nonlinear: "
            f"{'Ya' if method.get('nonlinear') else 'Tidak'}"
            "\n"
            "Small Data: "
            f"{'Sesuai' if method.get('small_data') else 'Terbatas'}"
            "\n"
            "Large Data: "
            f"{'Sesuai' if method.get('large_data') else 'Terbatas'}"
        )

        meta.setObjectName(
            "cardDescription"
        )

        meta.setWordWrap(
            True
        )

        layout.addWidget(
            meta
        )


class MLIntelligencePage(QWidget):


    """
    Halaman ML Intelligence.

    Pipeline:

        Dataset
            ↓
        Fingerprint
            ↓
        ML Task Detection
            ↓
        Method Recommendation

    Semua hasil tetap bersifat estimasi/rekomendasi.
    """

    def __init__(
        self,
        main_window,
    ):
        super().__init__()

        self.main_window = main_window

        self.detector = (
            MLTaskDetector()
        )

        self.recommender = (
            MethodRecommender()
        )

        # Empirical engine: actual model training + validation happens here.
        self.intelligence_engine = None
        self._ml_thread: QThread | None = None
        self._ml_worker: MLTrainingWorker | None = None

        self.last_task_result = {}
        self.last_method_result = {}
        self.last_intelligence_result = {}
        self._rendered_result_signature = None

        self.build_ui()

    # ======================================================
    # UI
    # ======================================================

    def build_ui(self):

        root_layout = QVBoxLayout(
            self
        )
        self._root_layout = root_layout

        root_layout.setContentsMargins(
            64,
            25,
            20,
            25,
        )

        root_layout.setSpacing(
            18
        )

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        header = QHBoxLayout()
        self.header_layout = header
        header.setContentsMargins(0, 0, 22, 0)

        title_layout = QVBoxLayout()

        title_layout.setSpacing(
            4
        )

        title = QLabel(
            "ML Intelligence"
        )

        title.setObjectName("mlPageTitle")

        description = QLabel(
            "Deteksi task Machine Learning "
            "dan rekomendasi metode berdasarkan "
            "struktur serta karakteristik dataset."
        )

        description.setObjectName("mlPageDescription")

        description.setWordWrap(
            True
        )
        description.setMinimumWidth(0)
        description.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        title_layout.addWidget(
            title
        )

        title_layout.addWidget(
            description
        )

        header.addLayout(
            title_layout,
            1,
        )

        header.addStretch()

        self.refresh_button = QPushButton("Jalankan ML")

        self.refresh_button.setObjectName(
            "primaryButton"
        )
        self.refresh_button.setProperty("mlRun", "true")

        self.refresh_button.clicked.connect(
            self.run_detection
        )

        header.addWidget(
            self.refresh_button
        )

        root_layout.addLayout(
            header
        )

        # --------------------------------------------------
        # SCROLL
        # --------------------------------------------------

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        self.content = QWidget()

        self.content_layout = QVBoxLayout(
            self.content
        )

        self.content_layout.setContentsMargins(
            0,
            30,
            0,
            20,
        )

        self.content_layout.setSpacing(
            16
        )

        scroll.setWidget(
            self.content
        )

        root_layout.addWidget(
            scroll
        )

        self.show_empty_state()

    def _apply_responsive_layout(self, width):
        compact = logical_width(width) < 760
        self._root_layout.setContentsMargins(
            16 if compact else 64,
            16 if compact else 25,
            16 if compact else 10,
            16 if compact else 25,
        )
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout(self.width())

    # ======================================================
    # EMPTY STATE
    # ======================================================

    def show_empty_state(self):

        self.last_task_result = {}
        self.last_method_result = {}
        self.last_intelligence_result = {}
        self._rendered_result_signature = None
        self.refresh_button.setText("Jalankan ML")

        self.clear_content()

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            30,
            40,
            30,
            40,
        )

        layout.setSpacing(
            10
        )

        title = QLabel(
            "ML Intelligence belum tersedia"
        )

        title.setObjectName(
            "sectionTitle"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        description = QLabel(
            "Unggah dan analisis dataset terlebih dahulu "
            "untuk mendapatkan estimasi task Machine Learning "
            "dan rekomendasi metode."
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setAlignment(
            Qt.AlignCenter
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        self.content_layout.addWidget(
            card
        )

        self.content_layout.addStretch()

    # ======================================================
    # DETECTION
    # ======================================================

    def run_detection(self):
        if self._ml_thread is not None:
            return
        analysis_result = getattr(self.main_window, "analysis_result", {})
        fingerprint = analysis_result.get("fingerprint") if analysis_result else None
        dataframe = getattr(self.main_window, "current_dataset", None)
        if not fingerprint or dataframe is None:
            self.show_empty_state()
            return

        if self.intelligence_engine is None:
            from app.ml.intelligence import MLIntelligenceEngine

            self.intelligence_engine = MLIntelligenceEngine(
                recommender=self.recommender,
                task_detector=self.detector,
            )

        self.refresh_button.setEnabled(False)
        self.refresh_button.setText("Training models...")
        self.clear_content()
        self.content_layout.addWidget(LoadingCard(
            "Training and comparing models",
            "Running the full configured model training and validation. This may take a while.",
            self.content,
        ))
        self.content_layout.addStretch()

        self._ml_thread = QThread(self)
        self._ml_worker = MLTrainingWorker(self.intelligence_engine, dataframe, fingerprint)
        self._ml_worker.moveToThread(self._ml_thread)
        self._ml_thread.started.connect(self._ml_worker.run)
        self._ml_worker.finished.connect(self._on_ml_finished)
        self._ml_worker.failed.connect(self._on_ml_failed)
        self._ml_worker.finished.connect(self._ml_thread.quit)
        self._ml_worker.failed.connect(self._ml_thread.quit)
        self._ml_thread.finished.connect(self._ml_worker.deleteLater)
        self._ml_thread.finished.connect(self._cleanup_ml_worker)
        self._ml_thread.start()

    def _on_ml_finished(self, intelligence_result):
        try:
            self._process_ml_result(intelligence_result)
        except Exception as error:
            self.show_error(f"Gagal menampilkan hasil ML Intelligence:\n{error}")
            self._finish_ml_run()

    def _process_ml_result(self, intelligence_result):
        self.last_intelligence_result = intelligence_result
        if intelligence_result.get("status") == "ERROR":
            self.show_error(
                "Gagal melakukan ML Intelligence:\n"
                f"{intelligence_result.get('message', 'Unknown error')}"
            )
            self._finish_ml_run()
            return

        analysis_result = getattr(self.main_window, "analysis_result", {})
        fingerprint = analysis_result.get("fingerprint", {})
        task_result = dict(intelligence_result.get("task_detection", {}))
        task_result["dataset"] = intelligence_result.get(
            "dataset", self.extract_dataset_info(analysis_result, fingerprint)
        )
        evaluation = intelligence_result.get("evaluation", {})
        if evaluation.get("message"):
            task_result["message"] = evaluation["message"]
        method_result = intelligence_result.get(
            "recommendation", {"recommendations": [], "recommendation_count": 0}
        )
        self.last_task_result = task_result
        self.last_method_result = method_result
        self.display_result(task_result, method_result)

        project_id = getattr(self.main_window, "current_project_id", None)
        if project_id is not None:
            try:
                self.main_window.repository.save_workflow_section(project_id, "ml_task", task_result)
                saved_methods = dict(method_result)
                saved_methods["_intelligence_result"] = intelligence_result
                self.main_window.repository.save_workflow_section(project_id, "ml_methods", saved_methods)
            except Exception as error:
                print(f"Warning: ML workflow could not be saved: {error}")
        self.main_window.dashboard.update_workflow_state(
            has_dataset=True, analysis_done=True, ml_done=True,
        )
        self._finish_ml_run()

    def _on_ml_failed(self, error):
        self.show_error(f"Gagal melakukan ML Intelligence:\n{error}")
        self._finish_ml_run()

    def _finish_ml_run(self):
        self.refresh_button.setEnabled(True)
        self.refresh_button.setText("Jalankan Ulang ML" if self.last_task_result else "Jalankan ML")

    def _cleanup_ml_worker(self):
        self._ml_worker = None
        self._ml_thread = None

    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    def extract_dataset_info(
        self,
        analysis_result: dict[str, Any],
        fingerprint: dict[str, Any],
    ) -> dict[str, Any]:

        dataset = {}

        dataframe = getattr(
            self.main_window,
            "current_dataset",
            None,
        )

        # --------------------------------------------------
        # PRIMARY SOURCE: DATAFRAME
        # --------------------------------------------------

        if dataframe is not None:

            try:

                dataset["rows"] = int(
                    dataframe.shape[0]
                )

                dataset["columns"] = int(
                    dataframe.shape[1]
                )

                numeric_features = len(
                    dataframe.select_dtypes(
                        include="number"
                    ).columns
                )

                dataset[
                    "numeric_features"
                ] = numeric_features

            except Exception:
                pass

        # --------------------------------------------------
        # FALLBACK: FINGERPRINT
        # --------------------------------------------------

        representation = fingerprint.get(
            "representation",
            fingerprint,
        )

        characteristics = (
            representation.get(
                "characteristics",
                {},
            )
        )

        if not dataset.get("rows"):

            dataset["rows"] = int(
                representation.get(
                    "rows",
                    fingerprint.get(
                        "rows",
                        0,
                    ),
                )
                or 0
            )

        if not dataset.get("columns"):

            dataset["columns"] = int(
                representation.get(
                    "columns",
                    fingerprint.get(
                        "columns",
                        0,
                    ),
                )
                or 0
            )

        if not dataset.get(
            "numeric_features"
        ):

            dataset[
                "numeric_features"
            ] = int(
                characteristics.get(
                    "numeric_count",
                    0,
                )
                or 0
            )

        return dataset

    # ======================================================
    # DISPLAY RESULT
    # ======================================================

    def display_result(
        self,
        result: dict[str, Any],
        method_result: dict[str, Any],
    ):

        self.clear_content()

        status = result.get(
            "status",
            "ESTIMATION",
        )

        primary = result.get(
            "primary_task"
        )

        tasks = result.get(
            "tasks",
            [],
        )

        dataset = result.get(
            "dataset",
            {},
        )

        recommendations = method_result.get(
            "recommendations",
            [],
        )

        # ==================================================
        # DATASET SUMMARY
        # ==================================================

        dataset_title = QLabel(
            "Dataset Intelligence"
        )

        dataset_title.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            dataset_title
        )

        dataset_cards = QHBoxLayout()

        dataset_cards.setSpacing(
            14
        )

        dataset_cards.addWidget(
            InfoCard(
                "Rows",
                str(
                    dataset.get(
                        "rows",
                        0,
                    )
                ),
                "Jumlah observasi dalam dataset.",
            )
        )

        dataset_cards.addWidget(
            InfoCard(
                "Columns",
                str(
                    dataset.get(
                        "columns",
                        0,
                    )
                ),
                "Jumlah kolom yang dianalisis.",
            )
        )

        dataset_cards.addWidget(
            InfoCard(
                "Numeric Features",
                str(
                    dataset.get(
                        "numeric_features",
                        0,
                    )
                ),
                "Jumlah fitur numerik.",
            )
        )

        self.content_layout.addLayout(
            dataset_cards
        )

        # ==================================================
        # PRIMARY TASK
        # ==================================================

        if primary:

            primary_label = primary.get(
                "label",
                primary.get(
                    "task",
                    "-",
                ),
            )

            confidence = primary.get(
                "confidence",
                primary.get(
                    "score",
                    0,
                ),
            )

            target = primary.get(
                "target"
            )

            cards_layout = QHBoxLayout()

            cards_layout.setSpacing(
                14
            )

            cards_layout.addWidget(
                InfoCard(
                    "Primary ML Task",
                    str(
                        primary_label
                    ),
                    f"Confidence: {confidence}%",
                )
            )

            cards_layout.addWidget(
                InfoCard(
                    "Target",
                    str(target)
                    if target
                    else "None",
                    "Target candidate yang digunakan detector.",
                )
            )

            cards_layout.addWidget(
                InfoCard(
                    "Confidence",
                    f"{confidence}%",
                    "Nilai ini merupakan estimasi heuristik.",
                )
            )

            self.content_layout.addLayout(
                cards_layout
            )

        # ==================================================
        # STATUS
        # ==================================================

        status_card = QFrame()

        status_card.setObjectName("mlDetectionStatusCard")
        status_card.setMinimumHeight(150)

        status_layout = QVBoxLayout(
            status_card
        )

        status_layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        status_layout.setSpacing(
            8
        )

        status_title = QLabel(
            "Status Deteksi"
        )

        status_title.setObjectName(
            "cardLabel"
        )

        status_value = QLabel(
            str(status)
        )

        status_value.setObjectName(
            "statusLabel"
        )

        message = QLabel(
            result.get(
                "message",
                "Task merupakan estimasi berdasarkan dataset.",
            )
        )

        message.setObjectName(
            "cardDescription"
        )

        message.setWordWrap(
            True
        )

        status_layout.addWidget(
            status_title
        )

        status_layout.addWidget(
            status_value
        )

        status_layout.addWidget(
            message
        )

        primary_status_row = QHBoxLayout()
        primary_status_row.setSpacing(18)
        primary_status_row.setContentsMargins(0, 0, 15, 0)

        if primary:
            primary_summary = QFrame()
            primary_summary.setObjectName("mlPrimaryTaskSummary")
            primary_summary.setMinimumHeight(150)
            summary_layout = QVBoxLayout(primary_summary)
            summary_layout.setContentsMargins(20, 16, 20, 16)
            summary_layout.setSpacing(8)

            summary_header = QHBoxLayout()
            summary_title = QLabel("PRIMARY TASK ANALYSIS")
            summary_title.setObjectName("mlSummaryEyebrow")
            summary_header.addWidget(summary_title)
            summary_header.addStretch(1)
            confidence_value = primary.get("confidence", primary.get("score", 0))
            confidence_badge = QLabel(f"{confidence_value}% confidence")
            confidence_badge.setObjectName("mlSummaryConfidence")
            summary_header.addWidget(confidence_badge)
            summary_layout.addLayout(summary_header)

            summary_task = QLabel(str(primary.get("label", primary.get("task", "-"))))
            summary_task.setObjectName("mlSummaryTaskTitle")
            summary_layout.addWidget(summary_task)

            summary_target = QLabel(f"Target: {primary.get('target') or 'None'}")
            summary_target.setObjectName("mlSummaryTarget")
            summary_layout.addWidget(summary_target)

            confidence_bar = QProgressBar()
            confidence_bar.setObjectName("mlSummaryConfidenceBar")
            confidence_bar.setRange(0, 100)
            try:
                confidence_bar.setValue(max(0, min(100, int(float(confidence_value)))))
            except (TypeError, ValueError):
                confidence_bar.setValue(0)
            confidence_bar.setTextVisible(False)
            summary_layout.addWidget(confidence_bar)

            primary_status_row.addWidget(primary_summary, 7)

        primary_status_row.addWidget(status_card, 3 if primary else 1)
        self.content_layout.addLayout(primary_status_row)

        # ==================================================
        # METHOD RECOMMENDATION
        # ==================================================

        recommendation_title = QLabel(
            "Evaluated ML Methods"
        )

        recommendation_title.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            recommendation_title
        )

        self.content_layout.addSpacing(8)

        recommendation_description = QLabel(
            "Ranking berikut berdasarkan hasil training dan validation "
            "pada dataset ini. Tidak menggunakan recommendation score statis."
        )

        recommendation_description.setObjectName(
            "cardDescription"
        )

        recommendation_description.setWordWrap(
            True
        )

        self.content_layout.addWidget(
            recommendation_description
        )

        # --------------------------------------------------
        # METHOD CARDS
        # --------------------------------------------------

        if recommendations:

            for index, method in enumerate(recommendations[:3], start=1):
                self.content_layout.addWidget(MethodRankingRow(index, method))

            detail_heading = QLabel("Rincian Evaluasi")
            detail_heading.setObjectName("sectionTitle")
            self.content_layout.addWidget(detail_heading)

            for method in recommendations:
                self.content_layout.addWidget(MethodCard(method))

            # ------------------------------------------------
            # TOP METHOD DETAIL
            # ------------------------------------------------

            best_method = recommendations[0]

            detail_title = QLabel(
                "Best Empirical Method"
            )

            detail_title.setObjectName(
                "sectionTitle"
            )

            self.content_layout.addWidget(
                detail_title
            )

            self.content_layout.addWidget(
                MethodDetailCard(
                    best_method
                )
            )

        else:

            evaluation_message = (
                getattr(
                    self,
                    "last_intelligence_result",
                    {},
                ).get(
                    "evaluation",
                    {},
                ).get(
                    "message",
                    "Model belum dievaluasi pada dataset ini.",
                )
            )

            empty_methods = QLabel(
                str(evaluation_message)
            )

            empty_methods.setObjectName(
                "cardDescription"
            )

            empty_methods.setWordWrap(
                True
            )

            self.content_layout.addWidget(
                empty_methods
            )

        if primary:
            primary_detail_title = QLabel("Rincian Task Utama")
            primary_detail_title.setObjectName("sectionTitle")
            self.content_layout.addWidget(primary_detail_title)
            self.content_layout.addWidget(TaskCard(primary))

        # ==================================================
        # OTHER POSSIBLE TASKS
        # ==================================================

        tasks_section = QLabel(
            f"Possible ML Tasks ({len(tasks)})"
        )

        tasks_section.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            tasks_section
        )

        if tasks:

            for task in tasks:

                self.content_layout.addWidget(
                    TaskCard(
                        task
                    )
                )

        else:

            empty = QLabel(
                "Tidak ada task ML yang berhasil dideteksi."
            )

            empty.setObjectName(
                "cardDescription"
            )

            self.content_layout.addWidget(
                empty
            )

        # ==================================================
        # DISCLAIMER
        # ==================================================

        disclaimer = QFrame()

        disclaimer.setObjectName(
            "contentCard"
        )

        disclaimer_layout = QVBoxLayout(
            disclaimer
        )

        disclaimer_layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        disclaimer_title = QLabel(
            "Catatan Riset"
        )

        disclaimer_title.setObjectName(
            "cardLabel"
        )

        disclaimer_text = QLabel(
            "Deteksi task ML dan rekomendasi metode merupakan "
            "estimasi berbasis heuristik dan knowledge base. "
            "Performa aktual metode harus divalidasi melalui "
            "eksperimen, cross-validation, dan metrik evaluasi yang sesuai."
        )

        disclaimer_text.setObjectName(
            "cardDescription"
        )

        disclaimer_text.setWordWrap(
            True
        )

        disclaimer_layout.addWidget(
            disclaimer_title
        )

        disclaimer_layout.addWidget(
            disclaimer_text
        )

        self.content_layout.addWidget(
            disclaimer
        )

        self.content_layout.addStretch()
        self._rendered_result_signature = (
            id(result),
            id(method_result),
            id(self.main_window.current_dataset),
        )

    # ======================================================
    # ERROR
    # ======================================================

    def show_error(
        self,
        message: str,
    ):

        self.clear_content()

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            25,
            25,
            25,
            25,
        )

        layout.setSpacing(
            10
        )

        title = QLabel(
            "Kesalahan ML Intelligence"
        )

        title.setObjectName(
            "sectionTitle"
        )

        error = QLabel(
            message
        )

        error.setObjectName(
            "cardDescription"
        )

        error.setWordWrap(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            error
        )

        self.content_layout.addWidget(
            card
        )

        self.content_layout.addStretch()

    # ======================================================
    # UPDATE DATASET
    # ======================================================

    def update_dataset(self):
        if self.last_task_result and self.last_method_result:
            return
        if getattr(self.main_window, "analysis_result", {}):
            self.show_ready_state()
        else:
            self.show_empty_state()

    def is_result_displayed(self) -> bool:
        return self._rendered_result_signature == (
            id(self.last_task_result),
            id(self.last_method_result),
            id(self.main_window.current_dataset),
        ) and bool(self.last_task_result and self.last_method_result)

    def restore_saved_result(self) -> bool:
        """Restore completed ML output for the active local dataset project."""
        project_id = getattr(self.main_window, "current_project_id", None)
        if project_id is None:
            return False
        try:
            state = self.main_window.repository.get_workflow_state(project_id) or {}
        except Exception as error:
            print(f"Warning: saved ML workflow could not be loaded: {error}")
            return False

        task_result = state.get("ml_task") or {}
        saved_methods = state.get("ml_methods") or {}
        if not task_result or not saved_methods:
            return False
        self.last_task_result = task_result
        self.last_method_result = {
            key: value for key, value in saved_methods.items()
            if key != "_intelligence_result"
        }
        self.last_intelligence_result = saved_methods.get("_intelligence_result") or {}
        self.display_result(self.last_task_result, self.last_method_result)
        self.refresh_button.setText("Jalankan Ulang ML")
        return True

    def show_ready_state(self):
        self.clear_content()
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(30, 40, 30, 40)
        layout.setSpacing(10)
        title = QLabel("Analisis dataset siap dijalankan")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        description = QLabel(
            "Pilih Jalankan ML untuk melatih dan membandingkan seluruh metode "
            "yang dikonfigurasi. Proses dapat memerlukan waktu."
        )
        description.setObjectName("cardDescription")
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)
        self.content_layout.addWidget(card)
        self.content_layout.addStretch()

    # ======================================================
    # CLEAR CONTENT
    # ======================================================

    def clear_content(self):

        while self.content_layout.count():

            item = (
                self.content_layout.takeAt(0)
            )

            widget = item.widget()

            if widget is not None:

                widget.deleteLater()

            elif item.layout() is not None:

                self.clear_layout(
                    item.layout()
                )

    # ======================================================
    # CLEAR LAYOUT
    # ======================================================

    def clear_layout(
        self,
        layout,
    ):

        while layout.count():

            item = layout.takeAt(0)

            widget = item.widget()

            if widget is not None:

                widget.deleteLater()

            elif item.layout() is not None:

                self.clear_layout(
                    item.layout()
                )

class _ResearchPaperTableItem(QTableWidgetItem):
    """Table item with native numeric sorting for rank, year, and relevance."""

    def __init__(self, text, sort_key=None):
        super().__init__(str(text))
        self._sort_key = str(text).casefold() if sort_key is None else sort_key

    def __lt__(self, other):
        other_key = getattr(other, "_sort_key", other.text().casefold())
        try:
            return self._sort_key < other_key
        except TypeError:
            return str(self._sort_key).casefold() < str(other_key).casefold()


class ResearchPage(QWidget):
    """
    Academic Research workspace.

    Menampilkan:
    - Ranked Papers and journal screening
    """

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.engine = None
        self.task_detector = MLTaskDetector()

        self.research_result: Dict[str, Any] = {}
        self._last_ml_result: Dict[str, Any] = {}
        self._all_papers = []
        self._all_paper_result_indices = []
        self.filtered_paper_indices = []
        self._research_thread: QThread | None = None
        self._research_worker: AcademicResearchWorker | None = None
        self._insights_thread: QThread | None = None
        self._insights_worker: AcademicResearchInsightsWorker | None = None
        self._sinta_thread: QThread | None = None
        self._sinta_worker: SintaLookupWorker | None = None
        self._sinta_lookup_cache = {}
        self._journal_index_thread: QThread | None = None
        self._journal_index_worker: JournalIndexBatchLookupWorker | None = None
        self._non_sinta_thread: QThread | None = None
        self._non_sinta_worker: NonSintaBatchLookupWorker | None = None
        self._sinta_table_refresh_timer = QTimer(self)
        self._sinta_table_refresh_timer.setSingleShot(True)
        self._sinta_table_refresh_timer.setInterval(140)
        self._sinta_table_refresh_timer.timeout.connect(self._render_filtered_papers)

        self._build_ui()

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self):

        root = QVBoxLayout(self)
        self.root_layout = root

        root.setContentsMargins(
            30,
            25,
            30,
            25,
        )

        root.setSpacing(18)

        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------

        header_widget = QWidget()
        self.header_widget = header_widget
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout = header_layout

        title_layout = QVBoxLayout()

        title = QLabel(
            "Academic Research"
        )
        self.title_label = title

        title.setObjectName(
            "pageTitle"
        )

        subtitle = QLabel(
            "Temukan paper yang relevan, telaah detail artikel, dan periksa status jurnal dalam satu workspace."
        )
        self.subtitle_label = subtitle

        subtitle.setObjectName(
            "pageSubtitle"
        )

        title_layout.addWidget(title)
        subtitle.setWordWrap(True)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(
            title_layout,
            1,
        )

        self.run_button = QPushButton("Jalankan Academic Research")

        self.run_button.setObjectName(
            "primaryButton"
        )

        self.run_button.setMinimumHeight(
            42
        )

        self.run_button.setCursor(
            Qt.PointingHandCursor
        )

        self.run_button.clicked.connect(
            self.run_research
        )

        header_layout.addWidget(
            self.run_button
        )

        root.addWidget(header_widget)

        # -----------------------------------------------------
        # STATUS
        # -----------------------------------------------------

        self.status_frame = QFrame()

        self.status_frame.setObjectName(
            "researchStatus"
        )

        status_layout = QVBoxLayout(
            self.status_frame
        )

        status_layout.setContentsMargins(
            18,
            14,
            18,
            14,
        )

        self.dataset_label = QLabel(
            "Dataset: belum dimuat"
        )

        self.dataset_label.setObjectName(
            "statusLabel"
        )

        self.analysis_label = QLabel(
            "Analysis: belum tersedia"
        )

        self.analysis_label.setObjectName(
            "statusLabel"
        )

        status_layout.addWidget(
            self.dataset_label
        )

        status_layout.addWidget(
            self.analysis_label
        )

        root.addWidget(
            self.status_frame
        )

        # -----------------------------------------------------
        # PROGRESS
        # -----------------------------------------------------

        self.progress = QProgressBar()

        self.progress.setRange(
            0,
            0,
        )

        self.progress.setVisible(
            False
        )

        root.addWidget(self.progress)
        self.loading_card = LoadingCard(
            "Academic research is running",
            "Searching academic sources and preparing ranked paper results.",
            self,
        )
        self.loading_card.setVisible(False)
        root.addWidget(self.loading_card)

        # -----------------------------------------------------
        # TABS
        # -----------------------------------------------------

        self.tabs = QTabWidget()

        self.tabs.setDocumentMode(
            True
        )
        self.tabs.tabBar().setVisible(False)

        self.papers_tab = self._create_papers_tab()
        self.tabs.addTab(self.papers_tab, "Papers")

        root.addWidget(
            self.tabs,
            1,
        )

        # Initial state
        self.show_empty_state()

    def _apply_responsive_layout(self, width):
        compact = logical_width(width) < 760
        horizontal_margin = ui_px(16 if compact else 30)
        vertical_margin = ui_px(16 if compact else 25)
        self.root_layout.setContentsMargins(
            horizontal_margin,
            vertical_margin,
            horizontal_margin,
            vertical_margin,
        )
        self.title_label.setMinimumHeight(self.title_label.sizeHint().height())
        self.subtitle_label.setMinimumWidth(0)
        self.subtitle_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
        )
        subtitle_width = width - 2 * horizontal_margin
        if not compact:
            subtitle_width -= self.run_button.sizeHint().width() + self.header_layout.spacing()
        subtitle_height = self.subtitle_label.heightForWidth(max(1, subtitle_width))
        self.subtitle_label.setMinimumHeight(max(0, subtitle_height))
        if compact:
            self.header_widget.setMinimumHeight(
                self.title_label.minimumHeight()
                + self.subtitle_label.minimumHeight()
                + self.run_button.minimumHeight()
                + ui_px(18)
            )
        else:
            self.header_widget.setMinimumHeight(0)
        compact = logical_width(width) < 1100
        for layout_name in (
            "_paper_toolbar_layout",
            "_provider_row_layout",
            "_sinta_controls_layout",
        ):
            child_layout = getattr(self, layout_name, None)
            if child_layout is not None:
                child_layout.setDirection(
                    QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
                )
        selection_layout = getattr(self, "_paper_selection_layout", None)
        if selection_layout is not None:
            selection_layout.setDirection(
                QBoxLayout.TopToBottom if logical_width(width) < 1250 else QBoxLayout.LeftToRight
            )
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout(self.width())

    # =========================================================
    # TAB FACTORIES
    # =========================================================

    def _create_papers_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(7)

        title = QLabel("Ranked Papers")

        title.setObjectName(
            "sectionTitle"
        )

        description = QLabel(
            "Daftar literatur yang ditemukan dan diurutkan berdasarkan relevansi terhadap topik dataset. "
            "Pilih satu baris untuk membaca metadata, membuka sumber, dan memeriksa status jurnal."
        )
        description.setObjectName("cardDescription")
        description.setWordWrap(True)

        guide = QFrame()
        guide.setObjectName("paperResultsGuide")
        guide_layout = QVBoxLayout(guide)
        guide_layout.setContentsMargins(14, 10, 14, 10)
        guide_layout.setSpacing(3)
        guide_title = QLabel("Panduan membaca hasil")
        guide_title.setObjectName("paperResultsGuideTitle")
        guide_text = QLabel(
            "Skor relevansi adalah penyaringan heuristik topik, bukan akurasi model atau bukti bahwa paper "
            "memakai dataset yang sama. Klik judul kolom untuk mengurutkan tabel; klik dua kali baris "
            "untuk membuka sumber artikel."
        )
        guide_text.setObjectName("paperResultsGuideText")
        guide_text.setWordWrap(True)
        guide_layout.addWidget(guide_title)
        guide_layout.addWidget(guide_text)
        self.paper_run_summary = QLabel("Belum ada hasil. Jalankan Academic Research untuk mencari paper.")
        self.paper_run_summary.setObjectName("topbarStatus")
        self.paper_run_summary.setWordWrap(True)
        guide_layout.addWidget(self.paper_run_summary)

        filter_panel = QFrame()
        filter_panel.setObjectName("paperScreeningToolbar")
        toolbar = QHBoxLayout(filter_panel)
        self._paper_toolbar_layout = toolbar
        toolbar.setContentsMargins(10, 8, 10, 8)
        toolbar.setSpacing(8)

        self.paper_search_input = QLineEdit()
        self.paper_search_input.setPlaceholderText(
            "Cari judul, penulis, jurnal, ISSN, atau DOI"
        )
        self.paper_search_input.setMinimumHeight(34)
        self.paper_search_input.textChanged.connect(self._apply_paper_filter)
        toolbar.addWidget(self.paper_search_input, 1)

        filter_label = QLabel("Status jurnal")
        filter_label.setObjectName("cardDescription")
        toolbar.addWidget(filter_label)

        self.sinta_filter = QComboBox()
        self.sinta_filter.setMinimumHeight(34)
        self.sinta_filter.setMinimumWidth(185)
        self.sinta_filter.addItem("Semua Status Jurnal", "ALL")
        for rank in range(1, 7):
            self.sinta_filter.addItem(f"SINTA {rank}", f"S{rank}")
        self.sinta_filter.addItem("Tidak Terakreditasi", "NOT_ACCREDITED")
        self.sinta_filter.addItem("Belum Diperiksa", "UNVERIFIED")
        self.sinta_filter.addItem("Perlu Verifikasi Tahun", "PERIOD_UNVERIFIED")
        self.sinta_filter.addItem("Kandidat Ambigu", "AMBIGUOUS")
        self.sinta_filter.addItem("Jurnal Belum Cocok", "NO_MATCH")
        self.sinta_filter.addItem("Metadata Jurnal Tidak Ada", "NO_METADATA")
        self.sinta_filter.addItem("Peringkat Tidak Terbaca", "RANK_UNAVAILABLE")
        self.sinta_filter.addItem("Tidak Ditemukan", "NOT_FOUND")
        self.sinta_filter.addItem("Perlu Dicoba Lagi", "FAILED")
        self.sinta_filter.insertSeparator(self.sinta_filter.count())
        self.sinta_filter.addItem("Terindeks DOAJ", "INDEX_DOAJ")
        self.sinta_filter.addItem("Terindeks Scopus", "INDEX_SCOPUS")
        self.sinta_filter.addItem("Indeks lain belum diperiksa", "INDEX_UNVERIFIED")
        self.sinta_filter.addItem("Tidak ditemukan di indeks yang dicek", "INDEX_NOT_FOUND")
        self.sinta_filter.addItem("Pemeriksaan indeks parsial", "INDEX_PARTIAL")
        self.sinta_filter.addItem("Metadata ISSN tidak tersedia", "INDEX_NO_METADATA")
        self.sinta_filter.currentIndexChanged.connect(self._apply_paper_filter)
        toolbar.addWidget(self.sinta_filter)

        accred_label = QLabel("Akreditasi Non-SINTA")
        accred_label.setObjectName("cardDescription")
        toolbar.addWidget(accred_label)

        self.accreditation_filter = QComboBox()
        self.accreditation_filter.setMinimumHeight(34)
        self.accreditation_filter.setMinimumWidth(190)
        self.accreditation_filter.addItem("Semua Akreditasi Non-SINTA", "ALL")
        self.accreditation_filter.addItem("Terakreditasi (Semua Sumber)", "ACCREDITED")
        self.accreditation_filter.addItem("HK-dir Level 2 (Unggulan)", "HKDIR_L2")
        self.accreditation_filter.addItem("HK-dir Level 1 (Akademik)", "HKDIR_L1")
        self.accreditation_filter.addItem("ABDC Rating A* / A", "ABDC_TOP")
        self.accreditation_filter.addItem("ABDC Rating (A*-C)", "ABDC_ACCREDITED")
        self.accreditation_filter.addItem("Tidak Terakreditasi (Level 0)", "NOT_ACCREDITED")
        self.accreditation_filter.addItem("Belum Diperiksa", "UNVERIFIED")
        self.accreditation_filter.addItem("Periode Perlu Verifikasi", "PERIOD_UNVERIFIED")
        self.accreditation_filter.addItem("Kandidat Ambigu", "AMBIGUOUS")
        self.accreditation_filter.addItem("Tidak Ditemukan di Direktori Resmi", "NOT_FOUND")
        self.accreditation_filter.addItem("Metadata ISSN Tidak Ada", "NO_METADATA")
        self.accreditation_filter.currentIndexChanged.connect(self._apply_paper_filter)
        toolbar.addWidget(self.accreditation_filter)

        self.sinta_batch_button = QPushButton("Periksa jurnal belum dicek")
        self.sinta_batch_button.setObjectName("secondaryButton")
        self.sinta_batch_button.setMinimumHeight(34)
        self.sinta_batch_button.setToolTip(
            "Periksa semua jurnal yang belum memiliki hasil direktori SINTA untuk tahun artikelnya."
        )
        self.sinta_batch_button.clicked.connect(self._check_pending_sinta_batch)
        toolbar.addWidget(self.sinta_batch_button)

        self.non_sinta_batch_button = QPushButton("Periksa Akreditasi Non-SINTA")
        self.non_sinta_batch_button.setObjectName("secondaryButton")
        self.non_sinta_batch_button.setMinimumHeight(34)
        self.non_sinta_batch_button.setToolTip(
            "Periksa akreditasi resmi dari direktori HK-dir (Kanalregisteret) dan ABDC Journal Quality List."
        )
        self.non_sinta_batch_button.setEnabled(False)
        self.non_sinta_batch_button.clicked.connect(self._check_pending_non_sinta_batch)
        toolbar.addWidget(self.non_sinta_batch_button)

        self.other_index_button = QPushButton("Periksa Indeks Lain")
        self.other_index_button.setObjectName("secondaryButton")
        self.other_index_button.setMinimumHeight(34)
        self.other_index_button.setToolTip(
            "Cek ISSN otomatis di DOAJ. Scopus dicek bila API Elsevier sudah memiliki akses."
        )
        self.other_index_button.setEnabled(False)
        self.other_index_button.clicked.connect(self._check_pending_other_indexes)
        toolbar.addWidget(self.other_index_button)

        self.paper_count_label = QLabel("0 paper")
        self.paper_count_label.setObjectName("topbarStatus")
        toolbar.addWidget(self.paper_count_label)

        self.provider_status_label = QLabel(
            "Sumber: OpenAlex, Crossref, Europe PMC, OpenAIRE, dan Semantic Scholar. "
            "Scopus memerlukan API key/akses Elsevier; UCL Discovery membuka pencarian eksternal."
        )
        self.provider_status_label.setObjectName("cardDescription")
        self.provider_status_label.setWordWrap(True)

        self.ucl_discovery_button = QPushButton("Pencarian Eksternal UCL Discovery")
        self.ucl_discovery_button.setObjectName("secondaryButton")
        self.ucl_discovery_button.setToolTip(
            "Buka pencarian resmi repositori UCL Discovery di browser menggunakan kata kunci riset. "
            "Hasil UCL tidak ditarik otomatis ke dalam daftar Ranked Papers aplikasi."
        )
        self.ucl_discovery_button.setMinimumHeight(32)
        self.ucl_discovery_button.clicked.connect(
            self._open_ucl_discovery_search
        )
        self.ucl_search_query = ""

        provider_row = QHBoxLayout()
        self._provider_row_layout = provider_row
        provider_row.setContentsMargins(0, 0, 0, 0)
        provider_row.setSpacing(8)
        provider_row.addWidget(self.provider_status_label, 1)
        provider_row.addWidget(self.ucl_discovery_button)

        self.paper_table = QTableWidget()
        self.paper_table.setObjectName("rankedPapersTable")

        self.paper_table.setColumnCount(9)

        self.paper_table.setHorizontalHeaderLabels(
            [
                "#",
                "Judul Artikel",
                "Tahun",
                "Jurnal / Venue",
                "ISSN",
                "Sumber",
                "Relevansi",
                "SINTA / Indeks",
                "Akreditasi Non-SINTA",
            ]
        )

        self.paper_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        self.paper_table.setSelectionBehavior(
            QTableWidget.SelectRows
        )
        self.paper_table.setSelectionMode(QTableWidget.SingleSelection)

        self.paper_table.setAlternatingRowColors(
            True
        )
        self.paper_table.setShowGrid(True)
        self.paper_table.setSortingEnabled(True)

        header = (
            self.paper_table.horizontalHeader()
        )

        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(0, 38)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(2, 58)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(3, 185)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(4, 110)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(5, 100)
        header.setSectionResizeMode(6, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(6, 78)
        header.setSectionResizeMode(7, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(7, 132)
        header.setSectionResizeMode(8, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(8, 175)

        self.paper_table.setMinimumHeight(210)
        self.paper_table.setWordWrap(False)
        self.paper_table.setTextElideMode(Qt.ElideRight)
        self.paper_table.setShowGrid(True)
        self.paper_table.verticalHeader().setDefaultSectionSize(34)
        self.paper_table.verticalHeader().setVisible(False)

        self.paper_table.itemSelectionChanged.connect(
            self._show_selected_paper
        )
        self.paper_table.cellDoubleClicked.connect(
            self._open_paper_source_for_row
        )

        self.paper_detail = QTextBrowser()
        self.paper_detail.setObjectName("paperDetail")
        self.paper_detail.setOpenExternalLinks(True)
        self.paper_detail.setReadOnly(True)
        self.paper_detail.setFrameShape(QFrame.NoFrame)
        self.paper_detail.setMinimumHeight(145)
        self.paper_detail.document().setDefaultStyleSheet(
            "body { font-family: 'Segoe UI', sans-serif; font-size: 9pt; color: #20324A; line-height: 1.45; } "
            "h2 { font-size: 11pt; color: #0F172A; margin: 0 0 6px 0; font-weight: 600; } "
            "p { margin: 3px 0; } "
            "h3 { font-size: 9.5pt; color: #1E293B; margin: 8px 0 4px; font-weight: bold; } "
            "ul { margin: 2px 0 6px 16px; padding: 0; } "
            "li { margin: 2px 0; } "
            "a { color: #2563EB; text-decoration: none; font-weight: bold; } "
            "a:hover { text-decoration: underline; }"
        )

        self.sinta_screening_panel = QFrame()
        self.sinta_screening_panel.setObjectName("sintaScreeningPanel")
        screening_layout = QVBoxLayout(self.sinta_screening_panel)
        screening_layout.setContentsMargins(14, 10, 14, 10)
        screening_layout.setSpacing(6)

        screening_heading = QLabel("Verifikasi Akreditasi Jurnal (SINTA)")
        screening_heading.setObjectName("sectionTitle")
        screening_note = QLabel(
            "Pemeriksaan mencari jurnal dari ISSN atau nama, lalu memeriksa masa akreditasi untuk tahun terbit. Jika profil resmi mencantumkan DOI artikel yang sama, DOI itu dicocokkan langsung."
        )
        screening_note.setObjectName("cardDescription")
        screening_note.setWordWrap(True)
        screening_layout.addWidget(screening_heading)
        screening_layout.addWidget(screening_note)

        screening_controls = QHBoxLayout()
        self._sinta_controls_layout = screening_controls
        screening_controls.setSpacing(8)

        self.sinta_status_combo = QComboBox()
        self.sinta_status_combo.setToolTip(
            "Pilih peringkat SINTA atau status jurnal yang berlaku untuk tahun publikasi artikel."
        )
        self.sinta_status_combo.setMinimumWidth(160)
        self.sinta_status_combo.addItem("Belum Terverifikasi (Pilih status)", "UNVERIFIED")
        for rank in range(1, 7):
            self.sinta_status_combo.addItem(f"SINTA {rank}", f"S{rank}")
        self.sinta_status_combo.addItem("Tidak Terakreditasi", "NOT_ACCREDITED")
        self.sinta_status_combo.addItem("Tidak Terdaftar di SINTA", "NOT_IN_SINTA")
        self.sinta_status_combo.setMinimumHeight(32)
        self.sinta_status_combo.currentIndexChanged.connect(
            self._update_sinta_save_enabled
        )
        screening_controls.addWidget(self.sinta_status_combo, 1)

        self.check_sinta_button = QPushButton("Periksa di SINTA")
        self.check_sinta_button.setObjectName("secondaryButton")
        self.check_sinta_button.setMinimumHeight(32)
        self.check_sinta_button.setToolTip("Cari jurnal di direktori resmi SINTA berdasarkan ISSN dan nama jurnal")
        self.check_sinta_button.clicked.connect(self._check_sinta_in_app)
        screening_controls.addWidget(self.check_sinta_button)

        self.open_sinta_browser_button = QPushButton("Buka di Browser")
        self.open_sinta_browser_button.setObjectName("secondaryButton")
        self.open_sinta_browser_button.setMinimumHeight(32)
        self.open_sinta_browser_button.setToolTip("Buka halaman pencarian resmi SINTA di web browser")
        self.open_sinta_browser_button.clicked.connect(self._open_sinta_in_browser)
        screening_controls.addWidget(self.open_sinta_browser_button)

        self.save_sinta_button = QPushButton("Simpan Status")
        self.save_sinta_button.setObjectName("primaryButton")
        self.save_sinta_button.setMinimumHeight(32)
        self.save_sinta_button.setEnabled(False)
        self.save_sinta_button.setToolTip("Simpan status akreditasi yang telah dikonfirmasi ke database lokal")
        self.save_sinta_button.clicked.connect(self._save_sinta_screening)
        screening_controls.addWidget(self.save_sinta_button)
        screening_layout.addLayout(screening_controls)

        self.sinta_period_confirmed = QCheckBox(
            "Saya telah memverifikasi masa berlaku akreditasi jurnal untuk tahun terbit artikel ini"
        )
        self.sinta_period_confirmed.setObjectName("sintaPeriodConfirmed")
        self.sinta_period_confirmed.toggled.connect(self._update_sinta_save_enabled)
        screening_layout.addWidget(self.sinta_period_confirmed)

        self.sinta_lookup_progress = QProgressBar()
        self.sinta_lookup_progress.setRange(0, 0)
        self.sinta_lookup_progress.setTextVisible(False)
        self.sinta_lookup_progress.setMaximumHeight(5)
        self.sinta_lookup_progress.setVisible(False)
        screening_layout.addWidget(self.sinta_lookup_progress)

        self.sinta_lookup_message = QLabel(
            "Pilih artikel, lalu periksa jurnal pada direktori resmi SINTA."
        )
        self.sinta_lookup_message.setObjectName("sintaLookupMessage")
        self.sinta_lookup_message.setWordWrap(True)
        self.sinta_lookup_message.setMinimumHeight(32)
        screening_layout.addWidget(self.sinta_lookup_message)

        # -----------------------------------------------------
        # NON-SINTA ACCREDITATION SCREENING PANEL
        # -----------------------------------------------------
        self.non_sinta_screening_panel = QFrame()
        self.non_sinta_screening_panel.setObjectName("nonSintaScreeningPanel")
        ns_screening_layout = QVBoxLayout(self.non_sinta_screening_panel)
        ns_screening_layout.setContentsMargins(14, 10, 14, 10)
        ns_screening_layout.setSpacing(6)

        ns_screening_heading = QLabel("Verifikasi Akreditasi Non-SINTA (HK-dir / ABDC)")
        ns_screening_heading.setObjectName("sectionTitle")
        ns_screening_note = QLabel(
            "Pemeriksaan akreditasi resmi dari direktori statutory HK-dir (Kanalregisteret) dan ABDC. "
            "Pencocokan ketat menggunakan ISSN dan masa berlaku tahun artikel. Indeksasi (DOAJ/Scopus) bukan akreditasi."
        )
        ns_screening_note.setObjectName("cardDescription")
        ns_screening_note.setWordWrap(True)
        ns_screening_layout.addWidget(ns_screening_heading)
        ns_screening_layout.addWidget(ns_screening_note)

        ns_controls = QHBoxLayout()
        ns_controls.setSpacing(8)

        self.non_sinta_status_combo = QComboBox()
        self.non_sinta_status_combo.setToolTip("Pilih status atau tingkat akreditasi non-SINTA yang terkonfirmasi.")
        self.non_sinta_status_combo.setMinimumWidth(160)
        self.non_sinta_status_combo.addItem("Belum Terverifikasi (Pilih status)", "UNVERIFIED")
        self.non_sinta_status_combo.addItem("HK-dir Level 2 (Unggulan)", "HKDIR_L2")
        self.non_sinta_status_combo.addItem("HK-dir Level 1 (Akademik)", "HKDIR_L1")
        self.non_sinta_status_combo.addItem("ABDC Rating A*", "ABDC_A_STAR")
        self.non_sinta_status_combo.addItem("ABDC Rating A", "ABDC_A")
        self.non_sinta_status_combo.addItem("ABDC Rating B", "ABDC_B")
        self.non_sinta_status_combo.addItem("ABDC Rating C", "ABDC_C")
        self.non_sinta_status_combo.addItem("Tidak Terakreditasi (Level 0)", "NOT_ACCREDITED")
        self.non_sinta_status_combo.addItem("Tidak Terdaftar di Direktori Resmi", "NOT_FOUND")
        self.non_sinta_status_combo.setMinimumHeight(32)
        self.non_sinta_status_combo.currentIndexChanged.connect(self._update_non_sinta_save_enabled)
        ns_controls.addWidget(self.non_sinta_status_combo, 1)

        self.check_non_sinta_button = QPushButton("Periksa Akreditasi")
        self.check_non_sinta_button.setObjectName("secondaryButton")
        self.check_non_sinta_button.setMinimumHeight(32)
        self.check_non_sinta_button.setToolTip("Periksa akreditasi jurnal di direktori resmi HK-dir dan ABDC")
        self.check_non_sinta_button.clicked.connect(self._check_non_sinta_in_app)
        ns_controls.addWidget(self.check_non_sinta_button)

        self.open_non_sinta_browser_button = QPushButton("Buka di Browser")
        self.open_non_sinta_browser_button.setObjectName("secondaryButton")
        self.open_non_sinta_browser_button.setMinimumHeight(32)
        self.open_non_sinta_browser_button.setToolTip("Buka halaman portal resmi HK-dir Kanalregisteret di browser")
        self.open_non_sinta_browser_button.clicked.connect(self._open_non_sinta_in_browser)
        ns_controls.addWidget(self.open_non_sinta_browser_button)

        self.save_non_sinta_button = QPushButton("Simpan Status")
        self.save_non_sinta_button.setObjectName("primaryButton")
        self.save_non_sinta_button.setMinimumHeight(32)
        self.save_non_sinta_button.setEnabled(False)
        self.save_non_sinta_button.setToolTip("Simpan status akreditasi non-SINTA ke database lokal")
        self.save_non_sinta_button.clicked.connect(self._save_non_sinta_screening)
        ns_controls.addWidget(self.save_non_sinta_button)
        ns_screening_layout.addLayout(ns_controls)

        self.non_sinta_period_confirmed = QCheckBox(
            "Saya telah memverifikasi masa berlaku akreditasi untuk tahun terbit artikel ini"
        )
        self.non_sinta_period_confirmed.setObjectName("nonSintaPeriodConfirmed")
        self.non_sinta_period_confirmed.toggled.connect(self._update_non_sinta_save_enabled)
        ns_screening_layout.addWidget(self.non_sinta_period_confirmed)

        self.non_sinta_lookup_progress = QProgressBar()
        self.non_sinta_lookup_progress.setRange(0, 0)
        self.non_sinta_lookup_progress.setTextVisible(False)
        self.non_sinta_lookup_progress.setMaximumHeight(5)
        self.non_sinta_lookup_progress.setVisible(False)
        ns_screening_layout.addWidget(self.non_sinta_lookup_progress)

        self.non_sinta_lookup_message = QLabel(
            "Pilih artikel, lalu periksa akreditasi pada direktori resmi HK-dir / ABDC."
        )
        self.non_sinta_lookup_message.setObjectName("nonSintaLookupMessage")
        self.non_sinta_lookup_message.setWordWrap(True)
        self.non_sinta_lookup_message.setMinimumHeight(32)
        ns_screening_layout.addWidget(self.non_sinta_lookup_message)

        screening_box = QVBoxLayout()
        screening_box.setSpacing(8)
        screening_box.addWidget(self.sinta_screening_panel)
        screening_box.addWidget(self.non_sinta_screening_panel)

        selection_panel = QFrame()
        selection_panel.setObjectName("paperSelectionPanel")
        selection_layout = QHBoxLayout(selection_panel)
        self._paper_selection_layout = selection_layout
        selection_layout.setContentsMargins(10, 8, 10, 8)
        selection_layout.setSpacing(12)
        selection_layout.addWidget(self.paper_detail, 3)
        selection_layout.addLayout(screening_box, 3)
        selection_panel.setMinimumHeight(240)
        self.paper_selection_panel = selection_panel

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        layout.addWidget(guide)

        layout.addWidget(filter_panel)
        layout.addLayout(provider_row)

        layout.addWidget(
            self.paper_table,
            1,
        )

        layout.addWidget(selection_panel)

        return widget

    # ---------------------------------------------------------

    def show_empty_state(self):

        self.research_result = {}
        self._all_papers = []
        self._all_paper_result_indices = []
        self.filtered_paper_indices = []

        self.run_button.setEnabled(
            True
        )

        self.progress.setVisible(
            False
        )

        self.dataset_label.setText(
            "Dataset: belum dimuat"
        )

        self.analysis_label.setText(
            "Analysis: belum tersedia"
        )

        self.paper_table.setRowCount(
            0
        )
        self.paper_run_summary.setText(
            "Belum ada hasil. Jalankan Academic Research untuk mencari paper."
        )

        self.paper_detail.clear()
        self.paper_count_label.setText("0 paper | 0 perlu disaring")
        self.provider_status_label.setText(
            "Sumber: OpenAlex, Crossref, Semantic Scholar, Europe PMC, dan OpenAIRE. "
            "Scopus memerlukan API key atau akses Elsevier; UCL Discovery hanya tersedia melalui situs eksternal."
        )
        self.sinta_status_combo.setCurrentIndex(0)
        self.sinta_period_confirmed.setChecked(False)
        self.save_sinta_button.setEnabled(False)
        self.check_sinta_button.setEnabled(False)
        self.sinta_lookup_progress.setVisible(False)
        self.sinta_lookup_message.setText(
            "Pilih paper, lalu periksa jurnalnya menggunakan ISSN atau judul jurnal."
        )

        self.loading_card.setVisible(False)
        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.show_empty_state()

    def restore_result(self, result: dict[str, Any]) -> None:
        """Rebuild the research page from a result stored in the local project DB."""
        if not isinstance(result, dict) or not result:
            self.show_empty_state()
            return
        dataframe = getattr(self.main_window, "current_dataset", None)
        analysis_result = getattr(self.main_window, "analysis_result", {})
        fingerprint = analysis_result.get("fingerprint", {})
        if dataframe is None or not fingerprint:
            self.show_empty_state()
            return
        ml_result = self._build_ml_result(dataframe, fingerprint)
        self.research_result = result
        self._update_status(dataframe, analysis_result)
        self._populate_result(result, ml_result)

    # =========================================================
    # RUN RESEARCH
    # =========================================================

    def run_research(self):
        dataframe = self.main_window.current_dataset
        analysis_result = self.main_window.analysis_result
        if dataframe is None:
            QMessageBox.warning(self, "Dataset Required", "Load a dataset before running academic research.")
            return
        if not analysis_result:
            QMessageBox.warning(self, "Analysis Required", "Run Dataset Analysis first before starting Academic Research.")
            return
        fingerprint = analysis_result.get("fingerprint")
        if not fingerprint:
            QMessageBox.warning(self, "Fingerprint Required", "Dataset fingerprint is not available. Run Dataset Analysis again.")
            return
        if self._research_thread is not None:
            return

        if self.engine is None:
            from app.research.intelligence import ResearchIntelligenceEngine

            self.engine = ResearchIntelligenceEngine()

        self.run_button.setEnabled(False)
        self.run_button.setText("Academic Research sedang berjalan...")
        self.progress.setVisible(True)
        self.loading_card.setVisible(True)
        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.set_research_running(True)
        self._research_thread = QThread(self)
        self._research_worker = AcademicResearchWorker(
            self.engine, self.task_detector, dataframe, fingerprint
        )
        self._research_worker.moveToThread(self._research_thread)
        self._research_thread.started.connect(self._research_worker.run)
        self._research_worker.finished.connect(self._on_research_finished)
        self._research_worker.failed.connect(self._on_research_failed)
        self._research_worker.finished.connect(self._research_thread.quit)
        self._research_worker.failed.connect(self._research_thread.quit)
        self._research_thread.finished.connect(self._research_worker.deleteLater)
        self._research_thread.finished.connect(self._cleanup_research_worker)
        self._research_thread.start()

    def _on_research_finished(self, result, ml_result):
        try:
            if result.get("status") not in {"SUCCESS", "PARTIAL"}:
                QMessageBox.critical(
                    self,
                    "Academic Research gagal",
                    str(result.get("error", "Terjadi kesalahan pada Academic Research.")),
                )
                return
            self.research_result = result
            self._last_ml_result = dict(ml_result or {})
            if result.get("status") == "PARTIAL":
                QMessageBox.warning(
                    self, "Academic Research Parsial",
                    "Tahap Academic Research sudah dijalankan, tetapi belum menemukan paper yang dapat dianalisis. "
                    "Hasil yang tersedia tetap disimpan dan dapat dimasukkan ke laporan.",
                )
            project_id = getattr(self.main_window, "current_project_id", None)
            if project_id is not None:
                try:
                    self.main_window.repository.save_workflow_section(project_id, "research", result)
                    self.main_window.repository.save_research_result(project_id, result)
                except Exception as error:
                    print(f"Warning: research workflow could not be saved: {error}")
            self._update_status(self.main_window.current_dataset, self.main_window.analysis_result)
            self._populate_result(result, ml_result)
            # External non-SINTA journal lists are checked off the UI thread;
            # their matches update the table as enrichment finishes.
            self._check_pending_non_sinta_batch()
            self.tabs.setCurrentWidget(self.papers_tab)
        except Exception as error:
            QMessageBox.critical(self, "Academic Research gagal", str(error))
        finally:
            self._finish_research()

    def _on_research_failed(self, error):
        QMessageBox.critical(self, "Academic Research gagal", str(error))
        self._finish_research()

    def _finish_research(self):
        self.progress.setVisible(False)
        self.loading_card.setVisible(False)
        self.run_button.setEnabled(True)
        self.run_button.setText("Jalankan Academic Research")
        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.set_research_running(False)

    def _cleanup_research_worker(self):
        self._research_worker = None
        self._research_thread = None

    def ensure_insights(self):
        """Start expensive landscape/trend/gap analysis only when its page is opened."""
        if not self.research_result or self._insights_thread is not None:
            return
        if self.research_result.get("insights_status") == "READY":
            return
        if self.research_result.get("insights_status") != "NOT_COMPUTED":
            if all(self.research_result.get(key) for key in ("landscape", "trend", "gaps")):
                return

        if self.engine is None:
            from app.research.intelligence import ResearchIntelligenceEngine

            self.engine = ResearchIntelligenceEngine()

        ml_result = self._last_ml_result
        if not ml_result:
            ml_page = getattr(self.main_window, "ml_page", None)
            saved_task = getattr(ml_page, "last_task_result", {}) if ml_page is not None else {}
            ml_result = dict(saved_task) if isinstance(saved_task, dict) else {}

        landscape_status = getattr(self.main_window.landscape_tool_page, "status_label", None)
        if landscape_status is not None:
            landscape_status.setText("Menghitung landscape dari hasil Papers...")
        gap_status = getattr(self.main_window.gap_tool_page, "status", None)
        if gap_status is not None:
            gap_status.setText("Menghitung kandidat research gap...")

        self._insights_thread = QThread(self)
        self._insights_worker = AcademicResearchInsightsWorker(
            self.engine, self.research_result, ml_result
        )
        self._insights_worker.moveToThread(self._insights_thread)
        self._insights_thread.started.connect(self._insights_worker.run)
        self._insights_worker.finished.connect(self._on_insights_finished)
        self._insights_worker.failed.connect(self._on_insights_failed)
        self._insights_worker.finished.connect(self._insights_thread.quit)
        self._insights_worker.failed.connect(self._insights_thread.quit)
        self._insights_thread.finished.connect(self._insights_worker.deleteLater)
        self._insights_thread.finished.connect(self._cleanup_insights_worker)
        self._insights_thread.start()

    @Slot(object)
    def _on_insights_finished(self, result):
        if not isinstance(result, dict):
            return
        self.research_result = result
        project_id = getattr(self.main_window, "current_project_id", None)
        if project_id is not None:
            try:
                self.main_window.repository.save_workflow_section(project_id, "research", result)
                self.main_window.repository.save_research_result(project_id, result)
            except Exception as error:
                print(f"Warning: hasil Insights tidak dapat disimpan: {error}")
        self.main_window.landscape_tool_page.update_from_result(result)
        self.main_window.gap_tool_page.update_from_result(result)
        insights_page = getattr(self.main_window, "research_insights_page", None)
        gap_is_active = (
            getattr(self.main_window, "_current_page_key", "") == "gap"
            or (
                insights_page is not None
                and insights_page.tabs.currentWidget()
                is self.main_window.gap_tool_page
            )
        )
        if gap_is_active:
            self.main_window.gap_tool_page.ensure_explanation()
        if getattr(self.main_window, "_current_page_key", "") == "report":
            self.main_window.report_tool_page.generate_preview(force=True)

    @Slot(str)
    def _on_insights_failed(self, error):
        message = f"Analisis Insights belum selesai: {error}"
        for page, attribute in (
            (self.main_window.landscape_tool_page, "status_label"),
            (self.main_window.gap_tool_page, "status"),
        ):
            label = getattr(page, attribute, None)
            if label is not None:
                label.setText(message)

    def _cleanup_insights_worker(self):
        self._insights_worker = None
        self._insights_thread = None

    # =========================================================
    # ML RESULT
    # =========================================================

    def _build_ml_result(
        self,
        dataframe,
        fingerprint,
    ):

        try:

            detected = (
                self.task_detector.detect(
                    fingerprint
                )
            )

        except Exception:

            detected = {
                "status": "ESTIMATION",
                "primary_task": None,
                "tasks": [],
                "task_count": 0,
            }

        result = dict(
            detected
        )

        result["dataset"] = {
            "rows": int(
                dataframe.shape[0]
            ),
            "columns": int(
                dataframe.shape[1]
            ),
            "numeric_features": int(
                dataframe.select_dtypes(
                    include="number"
                ).shape[1]
            ),
        }

        return result

    # =========================================================
    # UPDATE STATUS
    # =========================================================

    def _update_status(
        self,
        dataframe,
        analysis_result,
    ):

        filename = "Dataset"

        file_path = (
            self.main_window.current_file_path
        )

        if file_path:

            try:

                from pathlib import Path

                filename = Path(
                    file_path
                ).name

            except Exception:
                pass

        self.dataset_label.setText(
            f"Dataset: {filename} "
            f"({dataframe.shape[0]:,} baris × "
            f"{dataframe.shape[1]} kolom)"
        )

        self.analysis_label.setText(
            "Analysis: Dataset Fingerprint tersedia"
        )

    # =========================================================
    # POPULATE RESULT
    # =========================================================

    def _populate_result(self, result, ml_result):
        """Populate the paper workspace; analysis views are built on demand."""
        queries = result.get("queries") or []
        self.ucl_search_query = next(
            (str(query).strip() for query in queries if str(query).strip()),
            "",
        )
        papers = result.get("papers", [])
        if not isinstance(papers, list):
            papers = []
        summary = result.get("summary") or {}
        top_score = summary.get("top_relevance_score")
        summary_parts = [f"{len(papers):,} paper ditemukan"]
        if queries:
            summary_parts.append(f"{len(queries)} query pencarian")
        if isinstance(top_score, (int, float)):
            summary_parts.append(f"skor relevansi tertinggi {top_score:.1f}%")
        self.paper_run_summary.setText("  |  ".join(summary_parts))

        academic_search = (result.get("pipeline") or {}).get("academic_search") or {}
        provider_status = academic_search.get("provider_status") or {}
        provider_counts = academic_search.get("sources") or {}
        provider_parts = []
        for provider in (
            "OpenAlex",
            "Crossref",
            "Semantic Scholar",
            "Europe PMC",
            "OpenAIRE",
            "Scopus",
        ):
            state = provider_status.get(provider)
            count = provider_counts.get(provider)
            if state == "NOT_CONFIGURED":
                requirement = (
                    "memerlukan API key atau akses Elsevier"
                    if provider == "Scopus"
                    else "memerlukan key atau akses"
                )
                provider_parts.append(f"{provider}: {requirement}")
            elif state == "DISABLED":
                provider_parts.append(f"{provider}: nonaktif")
            elif state == "ERROR":
                provider_parts.append(f"{provider}: sementara tidak tersedia")
            elif state == "RATE_LIMITED":
                provider_parts.append(f"{provider}: batas permintaan tercapai")
            elif state == "ENABLED":
                provider_parts.append(f"{provider} {int(count or 0)}")
        self.provider_status_label.setText(
            "Kandidat sebelum penghapusan duplikat  |  "
            + ("  |  ".join(provider_parts) if provider_parts else "Status provider belum tersedia.")
        )
        self._populate_papers(papers)
        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.update_from_result(result)
    def _open_ucl_discovery_search(self):
        base_url = "https://discovery.ucl.ac.uk/cgi/search/simple"
        if self.ucl_search_query:
            url = f"{base_url}?q={quote_plus(self.ucl_search_query)}"
        else:
            url = "https://discovery.ucl.ac.uk/"
        QDesktopServices.openUrl(QUrl(url))

    def _populate_papers(
        self,
        papers,
    ):
        if not isinstance(papers, list):
            papers = []

        # Muat hasil screening akreditasi yang tersimpan di SQLite lokal
        repo = getattr(self.main_window, "repository", None)
        if repo is not None:
            try:
                saved_screenings = repo.list_journal_screenings()
                saved_non_sinta = repo.list_non_sinta_screenings()
                for paper in papers:
                    if not isinstance(paper, dict):
                        continue
                    doi = str(paper.get("doi") or "").strip().lower()
                    key = self._paper_identity(paper)
                    saved = None
                    if doi and f"doi:{doi}" in saved_screenings:
                        saved = saved_screenings[f"doi:{doi}"]
                    elif key and f"key:{key}" in saved_screenings:
                        saved = saved_screenings[f"key:{key}"]
                    if saved:
                        existing = paper.get("sinta_screening") or {}
                        expected_key = self._screening_metadata_key(paper)
                        saved_key = saved.get("metadata_key")
                        legacy_matches = (
                            not saved_key
                            and bool(saved.get("period_confirmed"))
                            and str(saved.get("checked_for_year") or "")
                            == str(paper.get("year") or "")
                        )
                        if saved_key == expected_key or legacy_matches:
                            if not existing.get("period_confirmed"):
                                paper["sinta_screening"] = {**existing, **saved}
                    existing = paper.get("sinta_screening")
                    if isinstance(existing, dict):
                        stored_key = existing.get("metadata_key")
                        # Screening results from old parser versions (and legacy
                        # results without a metadata key) must be rechecked. A
                        # prior result may belong to a different article DOI.
                        if stored_key != self._screening_metadata_key(paper):
                            paper.pop("sinta_screening", None)

                    # Muat juga screening akreditasi non-SINTA
                    saved_ns = None
                    if doi and f"doi:{doi}" in saved_non_sinta:
                        saved_ns = saved_non_sinta[f"doi:{doi}"]
                    elif key and f"key:{key}" in saved_non_sinta:
                        saved_ns = saved_non_sinta[f"key:{key}"]
                    if saved_ns:
                        existing_ns = paper.get("accreditation_screening") or {}
                        if not existing_ns.get("period_confirmed"):
                            paper["accreditation_screening"] = {**existing_ns, **saved_ns}
            except Exception as exc:
                print(f"Warning: gagal memuat riwayat screening dari database: {exc}")

        self._all_paper_result_indices = [
            index for index, paper in enumerate(papers) if isinstance(paper, dict)
        ]
        self._all_papers = [papers[index] for index in self._all_paper_result_indices]
        self.other_index_button.setEnabled(
            bool(self._all_papers) and self._journal_index_thread is None
        )
        if hasattr(self, "non_sinta_batch_button"):
            self.non_sinta_batch_button.setEnabled(
                bool(self._all_papers) and self._non_sinta_thread is None
            )
        self._apply_paper_filter()

    @staticmethod
    def _sinta_screening(paper):
        screening = paper.get("sinta_screening", {})
        if not isinstance(screening, dict):
            return {}
        screening = dict(screening)
        # Legacy states did not separate an explicit empty result from errors.
        if screening.get("status") == "NOT_IN_SINTA" and not screening.get("lookup_status"):
            screening["status"] = "UNVERIFIED"
            screening["lookup_status"] = "LEGACY_UNVERIFIED"
            screening["notes"] = (
                "Hasil versi lama tidak membedakan hasil kosong dari kegagalan pencarian. "
                "Periksa kembali direktori resmi SINTA."
            )
        if (
            screening.get("status") in {f"S{value}" for value in range(1, 7)}
            and not screening.get("lookup_status")
        ):
            # Older app versions stored a rank as if it were verified without
            # retaining the source or the accreditation-period check. Recheck it.
            screening["rank"] = screening.get("status")
            screening["status"] = "UNVERIFIED"
            screening["lookup_status"] = "LEGACY_UNVERIFIED"
            screening["period_status"] = "PERIOD_CHECK_REQUIRED"
            screening["period_confirmed"] = False
            screening["notes"] = (
                "Hasil versi lama tidak menyimpan bukti sumber dan masa berlaku SK. "
                "Jurnal perlu diperiksa ulang sebelum peringkat dianggap terverifikasi."
            )
        if (
            screening.get("status") == "NOT_ACCREDITED"
            and screening.get("rank")
            and not screening.get("lookup_status")
        ):
            screening["status"] = "UNVERIFIED"
            screening["lookup_status"] = "PERIOD_UNVERIFIED"
            screening["period_status"] = "EXPIRED"
            screening["notes"] = (
                "Versi lama menyimpulkan status ini dari periode yang tidak mencakup tahun artikel. "
                "Periksa SK historis sebelum menyimpulkan akreditasi."
            )
        return screening

    @staticmethod
    def _screening_metadata_key(paper):
        from app.research.sinta_verifier import SintaVerifier

        issn_values = paper.get("issn", [])
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        normalized_issns = sorted({
            SintaVerifier.normalize_issn(value)
            for value in issn_values
            if SintaVerifier.normalize_issn(value)
        })
        journal = SintaVerifier.normalize_name(str(paper.get("venue") or ""))
        year = SintaVerifier.normalize_year(paper.get("year"))
        doi = SintaVerifier.normalize_doi(str(paper.get("doi") or ""))
        return (
            f"sinta-parser:v4|issn:{','.join(normalized_issns)}"
            f"|journal:{journal}|year:{year or ''}|doi:{doi}"
        )

    @classmethod
    def _sinta_status_label(cls, paper):
        screening = cls._sinta_screening(paper)
        status = str(screening.get("status") or "UNVERIFIED")
        lookup_status = str(screening.get("lookup_status") or "UNVERIFIED")
        if lookup_status == "FAILED":
            return "Perlu Dicoba Lagi"
        if lookup_status == "NOT_FOUND" or status == "NOT_IN_SINTA":
            return "Tidak Ditemukan"
        if lookup_status == "AMBIGUOUS":
            return "Kandidat Ambigu"
        if lookup_status == "PERIOD_UNVERIFIED":
            rank = str(screening.get("rank") or "")
            if rank in {f"S{value}" for value in range(1, 7)}:
                return f"SINTA {rank[1:]}*"
            return "Perlu Verifikasi Tahun"
        if lookup_status == "NO_MATCH":
            return "Jurnal Belum Cocok"
        if lookup_status == "NO_METADATA":
            return "ISSN/Jurnal Tidak Ada"
        if lookup_status == "RANK_UNAVAILABLE":
            return "Peringkat Tidak Terbaca"
        if lookup_status == "LEGACY_UNVERIFIED":
            return "Perlu Diperiksa Ulang"
        labels = {
            "UNVERIFIED": "Belum Diperiksa",
            "NOT_ACCREDITED": "Tidak Terakreditasi",
            "NOT_IN_SINTA": "Tidak di SINTA",
        }
        if status in {f"S{rank}" for rank in range(1, 7)}:
            return f"SINTA {status[1:]}"
        return labels.get(status, "Belum Terverifikasi")

    @classmethod
    def _sinta_table_status_label(cls, paper):
        screening = cls._sinta_screening(paper)
        if str(screening.get("lookup_status") or "") == "PERIOD_UNVERIFIED":
            rank = str(screening.get("rank") or "")
            if rank in {f"S{value}" for value in range(1, 7)}:
                return f"SINTA {rank[1:]}"
        label = cls._sinta_status_label(paper)
        compact_labels = {
            "Perlu Dicoba Lagi": "Coba Lagi",
            "Tidak Ditemukan": "Tidak di SINTA",
            "Kandidat Ambigu": "Ambigu",
            "Perlu Verifikasi Tahun": "Cek Periode SK",
            "Jurnal Belum Cocok": "Belum Cocok",
            "ISSN/Jurnal Tidak Ada": "Data Kurang",
            "Peringkat Tidak Terbaca": "Rank Tak Ada",
            "Perlu Diperiksa Ulang": "Cek Ulang",
            "Belum Diperiksa": "Belum Dicek",
            "Tidak Terakreditasi": "Tanpa Akreditasi",
            "Belum Terverifikasi": "Belum Pasti",
        }
        base_label = compact_labels.get(label, label)
        other_indexes = screening.get("other_indexes", {})
        indexed_sources = [
            name
            for name, details in (other_indexes.get("sources", {}) or {}).items()
            if isinstance(details, dict) and details.get("status") == "INDEXED"
        ] if isinstance(other_indexes, dict) else []
        if indexed_sources:
            return f"{base_label} · {'/'.join(indexed_sources)}"
        if isinstance(other_indexes, dict):
            extra_labels = {
                "PARTIAL": "Indeks Parsial",
                "NOT_FOUND": "Indeks tak cocok",
                "NO_METADATA": "ISSN kosong",
                "FAILED": "Cek indeks gagal",
            }
            extra_label = extra_labels.get(other_indexes.get("status"))
            if extra_label:
                return f"{base_label} · {extra_label}"
        return base_label

    @staticmethod
    def _screening_journal_display_name(paper, screening):
        candidate = str(screening.get("journal_name") or "").strip()
        if (
            not candidate
            or candidate.startswith(("#", "["))
            or re.search(r"mynetwork|current accreditation|history accreditation", candidate, re.I)
        ):
            candidate = str(paper.get("venue") or "").strip()
        return candidate or "Nama jurnal belum tersedia"

    @staticmethod
    def _sinta_lookup_label(value):
        return {
            "VERIFIED": "Terverifikasi",
            "NOT_FOUND": "Tidak ada hasil pada direktori resmi",
            "NO_MATCH": "Hasil ditemukan, tetapi metadata belum cocok persis",
            "NO_METADATA": "Data ISSN dan nama jurnal tidak tersedia",
            "RANK_UNAVAILABLE": "Jurnal cocok, tetapi peringkat tidak terbaca dari sumber",
            "FAILED": "Gagal diperiksa; dapat dicoba ulang",
            "AMBIGUOUS": "Kandidat jurnal perlu ditinjau",
            "PERIOD_UNVERIFIED": "Periode SK belum terverifikasi",
            "MANUALLY_CONFIRMED": "Dikonfirmasi manual",
            "LEGACY_UNVERIFIED": "Hasil lama perlu diperiksa ulang",
            "UNVERIFIED": "Belum dapat dipastikan",
        }.get(str(value or "UNVERIFIED"), "Belum dapat dipastikan")

    @classmethod
    def _matches_sinta_filter(cls, paper, selected_status):
        if selected_status in {None, "ALL"}:
            return True
        screening = cls._sinta_screening(paper)
        other_indexes = screening.get("other_indexes", {})
        if selected_status in {"INDEX_DOAJ", "INDEX_SCOPUS"}:
            source_name = selected_status.removeprefix("INDEX_")
            sources = other_indexes.get("sources", {}) if isinstance(other_indexes, dict) else {}
            source = sources.get(source_name, {}) if isinstance(sources, dict) else {}
            return isinstance(source, dict) and source.get("status") == "INDEXED"
        if selected_status == "INDEX_UNVERIFIED":
            return not isinstance(other_indexes, dict) or not other_indexes.get("checked_at")
        other_status = other_indexes.get("status") if isinstance(other_indexes, dict) else None
        if selected_status == "INDEX_NOT_FOUND":
            return other_status == "NOT_FOUND"
        if selected_status == "INDEX_PARTIAL":
            return other_status in {"PARTIAL", "FAILED"}
        if selected_status == "INDEX_NO_METADATA":
            return other_status == "NO_METADATA"
        accreditation = paper.get("accreditation_screening", {}) if isinstance(paper, dict) else {}
        if not isinstance(accreditation, dict):
            accreditation = {}
        assessments = accreditation.get("official_assessments", [])
        assessments = [item for item in assessments if isinstance(item, dict)] if isinstance(assessments, list) else []
        accreditation_status = str(accreditation.get("status") or "")
        accreditation_system = str(accreditation.get("system") or "").upper()
        if selected_status == "EXT_ASSESSMENT":
            return bool(assessments) or accreditation_status in {
                "HKDIR_L2", "HKDIR_L1", "ABDC_A_STAR", "ABDC_A", "ABDC_B", "ABDC_C", "ACCREDITED",
            }
        if selected_status == "EXT_ACCREDITATION":
            return (
                any(item.get("kind") == "ACCREDITATION" for item in assessments)
                or accreditation_status in {"HKDIR_L2", "HKDIR_L1"}
                or (
                    accreditation_status == "ACCREDITED"
                    and accreditation_system not in {"ABDC", "OPENALEX"}
                )
            )
        if selected_status == "EXT_RATING":
            return (
                any(item.get("kind") == "RATING_OR_CLASSIFICATION" for item in assessments)
                or accreditation_system == "ABDC"
                or accreditation_status in {"ABDC_A_STAR", "ABDC_A", "ABDC_B", "ABDC_C"}
            )
        status = str(screening.get("status") or "UNVERIFIED")
        lookup_status = str(screening.get("lookup_status") or "UNVERIFIED")
        if selected_status == "UNVERIFIED":
            return status == "UNVERIFIED" and lookup_status in {"UNVERIFIED", "LEGACY_UNVERIFIED"}
        if selected_status in {f"S{rank}" for rank in range(1, 7)}:
            return str(screening.get("rank") or status) == selected_status
        if selected_status == "NOT_FOUND":
            return lookup_status == "NOT_FOUND"
        return status == selected_status or lookup_status == selected_status

    @classmethod
    def _matches_non_sinta_filter(cls, paper, selected_status):
        if selected_status in {None, "ALL"}:
            return True
        screening = paper.get("accreditation_screening", {}) if isinstance(paper, dict) else {}
        if not isinstance(screening, dict):
            screening = {}
        status = str(screening.get("status") or "UNVERIFIED")
        lookup_status = str(screening.get("lookup_status") or "UNVERIFIED")
        system = str(screening.get("system") or "").upper()
        rank = str(screening.get("rank") or "").upper()
        assessments = screening.get("official_assessments", [])
        assessments = assessments if isinstance(assessments, list) else []

        if selected_status == "ACCREDITED":
            return (
                any(item.get("kind") == "ACCREDITATION" for item in assessments if isinstance(item, dict))
                or (
                    status == "ACCREDITED"
                    and lookup_status in {"VERIFIED", "MANUALLY_CONFIRMED", "OPENALEX_LISTED"}
                    and system not in {"ABDC", "OPENALEX"}
                )
                or status in {"HKDIR_L2", "HKDIR_L1"}
            )
        if selected_status == "HKDIR_L2":
            return (system in {"HKDIR", "HK-DIR"} and "LEVEL 2" in rank and lookup_status in {"VERIFIED", "MANUALLY_CONFIRMED"}) or status == "HKDIR_L2"
        if selected_status == "HKDIR_L1":
            return (system in {"HKDIR", "HK-DIR"} and "LEVEL 1" in rank and lookup_status in {"VERIFIED", "MANUALLY_CONFIRMED"}) or status == "HKDIR_L1"
        if selected_status == "ABDC_TOP":
            return (system == "ABDC" and rank in {"A*", "A"} and lookup_status in {"VERIFIED", "MANUALLY_CONFIRMED"}) or status in {"ABDC_A_STAR", "ABDC_A"}
        if selected_status == "ABDC_ACCREDITED":
            return (system == "ABDC" and rank in {"A*", "A", "B", "C"} and lookup_status in {"VERIFIED", "MANUALLY_CONFIRMED"}) or status in {"ABDC_A_STAR", "ABDC_A", "ABDC_B", "ABDC_C"}
        if selected_status == "NOT_ACCREDITED":
            return status == "NOT_ACCREDITED" or "LEVEL 0" in rank
        if selected_status == "UNVERIFIED":
            return status == "UNVERIFIED" and lookup_status in {"UNVERIFIED", "UNKNOWN", ""}
        if selected_status == "PERIOD_UNVERIFIED":
            return lookup_status == "PERIOD_UNVERIFIED" or str(screening.get("period_status")) in {"EXPIRED", "PERIOD_CHECK_REQUIRED"}
        if selected_status == "AMBIGUOUS":
            return lookup_status == "AMBIGUOUS" or bool(screening.get("is_ambiguous"))
        if selected_status == "NO_METADATA":
            return lookup_status == "NO_METADATA"
        if selected_status == "NOT_FOUND":
            return lookup_status == "NOT_FOUND" or status == "NOT_FOUND"
        if selected_status == "FAILED":
            return lookup_status == "FAILED"
        return status == selected_status or lookup_status == selected_status

    @classmethod
    def _non_sinta_status_label(cls, paper):
        screening = paper.get("accreditation_screening", {}) if isinstance(paper, dict) else {}
        if not isinstance(screening, dict):
            return "Belum Diperiksa"
        status = str(screening.get("status") or "UNVERIFIED")
        lookup_status = str(screening.get("lookup_status") or "UNVERIFIED")
        rank = screening.get("rank")
        system = screening.get("system")
        assessments = screening.get("official_assessments", [])
        assessments = [item for item in assessments if isinstance(item, dict)] if isinstance(assessments, list) else []
        if assessments:
            kind_labels = {
                "ACCREDITATION": "Akreditasi",
                "RATING_OR_CLASSIFICATION": "Peringkat",
                "DIRECTORY_OR_INDEX": "Direktori/indeks",
            }
            labels = []
            for item in assessments:
                kind = str(item.get("kind") or "").upper()
                name = item.get("system_name") or item.get("rank") or item.get("list_id")
                labels.append(f"{kind_labels.get(kind, 'Penilaian')}: {name}")
            first = " · ".join(label for label in labels[:2] if label)
            if len(labels) > 2:
                first += f" +{len(labels) - 2}"
            return first or "Ada penilaian non-SINTA"

        if lookup_status == "AMBIGUOUS" or screening.get("is_ambiguous"):
            return f"{rank} (Kandidat Ambigu)" if rank else "Kandidat Ambigu"
        if lookup_status == "PERIOD_UNVERIFIED":
            return f"{rank} (Perlu Verifikasi Tahun)" if rank else "Periode Belum Terverifikasi"
        if lookup_status == "NO_METADATA":
            return "Metadata ISSN Tidak Ada"
        if lookup_status == "FAILED":
            return "Pemeriksaan Gagal"
        if lookup_status == "NOT_FOUND" or status == "NOT_FOUND":
            return "Tidak Terdaftar di Direktori Resmi"
        if status == "NOT_ACCREDITED":
            return f"Tidak Terakreditasi ({rank})" if rank else "Tidak Terakreditasi"
        if status == "ACCREDITED" and rank:
            if str(system or "").upper() == "ABDC":
                return f"Peringkat ABDC {rank}"
            return f"Akreditasi {system} {rank}" if system else f"Akreditasi {rank}"
        if screening.get("official_assessments_status") == "PARTIAL":
            return "Daftar penilaian diperiksa sebagian"
        if screening.get("official_assessments_status") == "CHECKED":
            return "Tidak ditemukan di daftar yang dicek"
        if screening.get("official_assessments_status") == "FAILED":
            return "Daftar penilaian belum dapat diperiksa"
        if screening.get("official_assessments_status") == "NO_METADATA":
            return "ISSN tidak tersedia untuk katalog eksternal"
        return "Belum Diperiksa"

    def _apply_paper_filter(self, *_):
        papers = getattr(self, "_all_papers", [])
        result_indices = getattr(self, "_all_paper_result_indices", [])
        query = self.paper_search_input.text().strip().casefold()
        selected_status = self.sinta_filter.currentData()
        selected_accred = self.accreditation_filter.currentData() if hasattr(self, "accreditation_filter") else "ALL"
        self.filtered_paper_indices = []

        for index, paper in zip(result_indices, papers):
            if not self._matches_sinta_filter(paper, selected_status):
                continue
            if hasattr(self, "accreditation_filter") and not self._matches_non_sinta_filter(paper, selected_accred):
                continue

            authors = paper.get("authors", [])
            keywords = paper.get("keywords", [])
            issn = paper.get("issn", [])
            searchable_parts = [
                paper.get("title", ""),
                paper.get("venue", ""),
                paper.get("doi", ""),
                paper.get("source", ""),
                paper.get("year", ""),
                *(authors if isinstance(authors, list) else [authors]),
                *(keywords if isinstance(keywords, list) else [keywords]),
                *(issn if isinstance(issn, list) else [issn]),
            ]
            searchable = " ".join(str(part or "") for part in searchable_parts).casefold()
            if query and query not in searchable:
                continue
            self.filtered_paper_indices.append(index)

        self._render_filtered_papers()

    def _render_filtered_papers(self):
        papers = self.research_result.get("papers", [])
        if not isinstance(papers, list):
            papers = []
        selected_paper_index = self._selected_paper_index()
        self.paper_table.setUpdatesEnabled(False)
        self.paper_table.setSortingEnabled(False)
        self.paper_table.setRowCount(len(self.filtered_paper_indices))

        for row, paper_index in enumerate(self.filtered_paper_indices):
            if paper_index >= len(papers) or not isinstance(papers[paper_index], dict):
                continue
            paper = papers[paper_index]

            title = str(paper.get("title") or "Untitled")
            year = str(paper.get("year") or "-")
            relevance = paper.get("relevance_score")

            venue = str(paper.get("venue") or paper.get("source") or "-")
            venue_display = venue
            if len(venue_display) > 38:
                venue_display = venue_display[:35].rstrip() + "..."

            issn_list = paper.get("issn", [])
            if isinstance(issn_list, str):
                issn_list = [issn_list]
            issn_clean = [str(i).strip() for i in issn_list if str(i).strip()]
            issn_str = issn_clean[0] if issn_clean else "-"
            issn_tooltip = ", ".join(issn_clean) if len(issn_clean) > 1 else (issn_str if issn_str != "-" else "ISSN tidak tersedia")

            sources = [
                source.strip()
                for source in str(paper.get("source") or "").split(",")
                if source.strip()
            ]
            source_label = sources[0] if sources else "-"
            if len(sources) > 1:
                source_label += f" +{len(sources) - 1}"

            sinta_label = self._sinta_table_status_label(paper)
            full_sinta_label = self._sinta_status_label(paper)
            non_sinta_label = self._non_sinta_status_label(paper)

            values = [
                str(row + 1),
                title,
                year,
                venue_display,
                issn_str,
                source_label,
                self._format_score(relevance),
                sinta_label,
                non_sinta_label,
            ]

            for column, value in enumerate(values):
                if column == 0:
                    sort_key = row + 1
                elif column == 2:
                    try:
                        sort_key = float(year)
                    except (TypeError, ValueError):
                        sort_key = float("-inf")
                elif column == 6:
                    try:
                        sort_key = float(relevance)
                    except (TypeError, ValueError):
                        sort_key = float("-inf")
                else:
                    sort_key = value.casefold()
                item = _ResearchPaperTableItem(value, sort_key)

                if column == 0:
                    item.setData(Qt.UserRole, paper_index)
                    item.setTextAlignment(Qt.AlignCenter)
                elif column == 1:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    item.setToolTip(title)
                elif column in {2, 4, 5, 6, 7, 8}:
                    item.setTextAlignment(Qt.AlignCenter)

                if column == 3:
                    item.setToolTip(venue)
                elif column == 4:
                    item.setToolTip(issn_tooltip)
                elif column == 5:
                    item.setToolTip(", ".join(sources) or "Sumber tidak tersedia")
                elif column == 7:
                    screening = self._sinta_screening(paper)
                    status = str(screening.get("status") or "UNVERIFIED")
                    external = screening.get("other_indexes", {})
                    if status in {f"S{rank}" for rank in range(1, 7)}:
                        item.setForeground(QColor("#0F766E"))
                        item.setBackground(QColor("#F0FDF4"))
                    elif status == "UNVERIFIED":
                        item.setForeground(QColor("#B45309"))
                        item.setBackground(QColor("#FFFBEB"))
                    elif status == "NOT_ACCREDITED":
                        item.setForeground(QColor("#475569"))
                        item.setBackground(QColor("#F8FAFC"))
                    else:
                        item.setForeground(QColor("#64748B"))
                        item.setBackground(QColor("#F1F5F9"))
                    item.setToolTip(
                        f"{full_sinta_label}. "
                        f"{_sinta_ui_note(screening) or screening.get('source_url') or 'Status SINTA belum diperiksa.'} "
                        f"{self._other_index_tooltip(external)}"
                    )
                elif column == 8:
                    accreditation = paper.get("accreditation_screening", {})
                    accreditation = accreditation if isinstance(accreditation, dict) else {}
                    lookup_status = str(accreditation.get("lookup_status") or "UNVERIFIED")
                    if lookup_status in {"VERIFIED", "MANUALLY_CONFIRMED"}:
                        item.setForeground(QColor("#0F766E"))
                        item.setBackground(QColor("#F0FDF4"))
                    elif lookup_status in {"FAILED", "AMBIGUOUS", "PERIOD_UNVERIFIED", "NO_METADATA"}:
                        item.setForeground(QColor("#B45309"))
                        item.setBackground(QColor("#FFFBEB"))
                    else:
                        item.setForeground(QColor("#64748B"))
                        item.setBackground(QColor("#F1F5F9"))
                    source = accreditation.get("verification_source") or accreditation.get("system") or "Sumber akreditasi non-SINTA belum diperiksa."
                    note = accreditation.get("notes") or accreditation.get("ambiguity_reason") or ""
                    item.setToolTip(f"{source}. {note}".strip())

                if not item.toolTip():
                    item.setToolTip(str(value))

                self.paper_table.setItem(row, column, item)

        self.paper_table.setSortingEnabled(True)
        self.paper_table.setUpdatesEnabled(True)
        checked_states = {
            "VERIFIED", "MANUALLY_CONFIRMED", "FAILED", "NOT_FOUND", "NO_MATCH",
            "NO_METADATA", "RANK_UNAVAILABLE", "AMBIGUOUS", "PERIOD_UNVERIFIED",
            "LEGACY_UNVERIFIED",
        }
        checked_count = sum(
            str(self._sinta_screening(paper).get("lookup_status") or "UNVERIFIED")
            in checked_states
            for paper in self._all_papers
        )
        other_index_checked = sum(
            bool(self._sinta_screening(paper).get("other_indexes", {}).get("checked_at"))
            for paper in self._all_papers
        )
        self.paper_count_label.setText(
            f"{len(self.filtered_paper_indices)} dari {len(self._all_papers)} paper | "
            f"SINTA {checked_count} diperiksa | indeks lain {other_index_checked} diperiksa"
        )

        if self.filtered_paper_indices:
            selected_row = 0
            if selected_paper_index is not None:
                for row in range(self.paper_table.rowCount()):
                    item = self.paper_table.item(row, 0)
                    if item is not None and item.data(Qt.UserRole) == selected_paper_index:
                        selected_row = row
                        break
            self.paper_table.selectRow(selected_row)
            # Rebuild the detail pane even when the same visual row stayed
            # selected while the filtered table contents changed.
            self._show_selected_paper()
        else:
            if not self._all_papers:
                self.paper_detail.setPlainText(
                    "Belum ada paper yang ditemukan untuk dataset ini.\n\n"
                    "Academic Research mencari literatur menggunakan topik riset dan metadata kolom dataset; "
                    "dataset pribadi tidak harus sudah dipublikasikan sebelumnya."
                )
            else:
                self.paper_detail.setPlainText("Tidak ada paper yang cocok dengan pencarian atau filter saat ini.")
            self.save_sinta_button.setEnabled(False)
            self.check_sinta_button.setEnabled(False)
            self.sinta_status_combo.setCurrentIndex(0)
            self.sinta_period_confirmed.setChecked(False)
            self.sinta_lookup_message.setText("Tidak ada artikel yang sesuai dengan filter saat ini.")
            self.sinta_lookup_message.setToolTip("")

        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.update_from_result(self.research_result)

    def _open_paper_source_for_row(self, row, _column=0):
        if row < 0 or row >= self.paper_table.rowCount():
            return
        item = self.paper_table.item(row, 0)
        paper_index = item.data(Qt.UserRole) if item is not None else None
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or not isinstance(papers, list):
            return
        if paper_index < 0 or paper_index >= len(papers):
            return

        paper = papers[paper_index]
        target_url = str(paper.get("url") or "").strip()
        doi = str(paper.get("doi") or "").strip()
        if not target_url and doi:
            if doi[:4].lower() == "doi:":
                doi = doi[4:].strip()
            target_url = f"https://doi.org/{doi}"
        if not target_url:
            QMessageBox.information(
                self,
                "Sumber tidak tersedia",
                "Artikel ini tidak menyediakan tautan sumber atau DOI.",
            )
            return
        if not QDesktopServices.openUrl(QUrl(target_url)):
            QMessageBox.warning(
                self,
                "Sumber tidak dapat dibuka",
                "Windows tidak dapat membuka tautan sumber artikel ini.",
            )

    # =========================================================
    # PAPER DETAIL
    # =========================================================

    @staticmethod
    def _other_index_tooltip(result):
        if not isinstance(result, dict) or not result.get("checked_at"):
            return "Indeks lain belum diperiksa."
        pieces = []
        for name, source in (result.get("sources", {}) or {}).items():
            if not isinstance(source, dict):
                continue
            status = source.get("status")
            if status == "INDEXED":
                pieces.append(f"{name}: terindeks")
            elif status == "NOT_INDEXED":
                pieces.append(f"{name}: tidak ditemukan")
            elif status == "NOT_CONFIGURED":
                pieces.append(f"{name}: API belum dikonfigurasi")
            elif status == "NO_METADATA":
                pieces.append(f"{name}: ISSN tidak tersedia")
            elif status in {"FAILED", "UNAVAILABLE"}:
                pieces.append(f"{name}: belum dapat dipastikan")
        return "Indeks lain — " + ("; ".join(pieces) or "belum ada hasil sumber.")

    def _show_selected_paper(self):
        rows = self.paper_table.selectionModel().selectedRows()
        if not rows:
            self.paper_detail.clear()
            self.save_sinta_button.setEnabled(False)
            self.check_sinta_button.setEnabled(False)
            self.sinta_status_combo.setCurrentIndex(0)
            self.sinta_period_confirmed.setChecked(False)
            self.sinta_lookup_message.setText("Pilih artikel untuk melihat detail dan status verifikasi SINTA.")
            self.sinta_lookup_message.setToolTip("")
            if hasattr(self, "save_non_sinta_button"):
                self.save_non_sinta_button.setEnabled(False)
            if hasattr(self, "check_non_sinta_button"):
                self.check_non_sinta_button.setEnabled(False)
            if hasattr(self, "non_sinta_status_combo"):
                self.non_sinta_status_combo.setCurrentIndex(0)
            if hasattr(self, "non_sinta_period_confirmed"):
                self.non_sinta_period_confirmed.setChecked(False)
            if hasattr(self, "non_sinta_lookup_message"):
                self.non_sinta_lookup_message.setText("Pilih artikel, lalu periksa akreditasi pada direktori resmi HK-dir / ABDC.")
                self.non_sinta_lookup_message.setToolTip("")
            return
        row = rows[0].row()
        paper_index_item = self.paper_table.item(row, 0)
        paper_index = paper_index_item.data(Qt.UserRole) if paper_index_item else None
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            return
        paper = papers[paper_index]
        screening = self._sinta_screening(paper)
        current_status = screening.get("status") or screening.get("rank") or "UNVERIFIED"
        status_index = self.sinta_status_combo.findData(current_status)
        self.sinta_status_combo.setCurrentIndex(max(0, status_index))
        self.sinta_period_confirmed.setChecked(
            bool(screening.get("period_confirmed"))
            and str(screening.get("checked_for_year") or "") == str(paper.get("year") or "")
        )
        self._update_sinta_save_enabled()

        ns_screening = paper.get("accreditation_screening") or {}
        if hasattr(self, "non_sinta_status_combo"):
            ns_status = ns_screening.get("status") or ns_screening.get("rank") or "UNVERIFIED"
            ns_status_index = self.non_sinta_status_combo.findData(ns_status)
            self.non_sinta_status_combo.setCurrentIndex(max(0, ns_status_index))
        if hasattr(self, "non_sinta_period_confirmed"):
            self.non_sinta_period_confirmed.setChecked(
                bool(ns_screening.get("period_confirmed"))
                and str(ns_screening.get("checked_for_year") or "") == str(paper.get("year") or "")
            )
        self._update_non_sinta_save_enabled()

        issn_values = paper.get("issn", [])
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        issn_clean = [str(i).strip() for i in issn_values if str(i).strip()]
        self.check_sinta_button.setEnabled(
            self._sinta_thread is None
            and bool(issn_clean or str(paper.get("venue") or "").strip())
        )
        if hasattr(self, "check_non_sinta_button"):
            self.check_non_sinta_button.setEnabled(
                self._non_sinta_thread is None
                and bool(issn_clean or str(paper.get("venue") or "").strip())
            )

        if screening.get("journal_name") or screening.get("lookup_status") not in {None, "UNVERIFIED"}:
            rank_hint = screening.get("rank") or "peringkat belum terdeteksi"
            status_label = self._sinta_status_label(paper)
            if screening.get("match_method") == "DOI" and screening.get("lookup_status") == "VERIFIED":
                message = f"DOI artikel cocok dengan catatan resmi SINTA: {rank_hint}."
            elif screening.get("lookup_status") == "PERIOD_UNVERIFIED" and screening.get("rank"):
                message = (
                    f"Peringkat jurnal saat ini {rank_hint}. Periksa masa berlaku SK "
                    f"untuk tahun {paper.get('year') or '-'} sebelum menetapkan status artikel."
                )
            elif screening.get("lookup_status") == "AMBIGUOUS":
                message = "Ada lebih dari satu kandidat jurnal. Cocokkan ISSN dan profil SINTA sebelum memilih status."
            else:
                message = _sinta_ui_note(screening) or status_label
            self.sinta_lookup_message.setText(message)
            self.sinta_lookup_message.setToolTip(_sinta_ui_note(screening) or message)
        else:
            self.sinta_lookup_message.setText(
                "Pilih artikel, lalu periksa jurnal pada direktori resmi SINTA."
            )
            self.sinta_lookup_message.setToolTip("")

        if hasattr(self, "non_sinta_lookup_message"):
            if ns_screening.get("journal_name") or ns_screening.get("lookup_status") not in {None, "UNVERIFIED"}:
                ns_rank_hint = ns_screening.get("rank") or "peringkat belum terdeteksi"
                ns_status_label = self._non_sinta_status_label(paper)
                if ns_screening.get("lookup_status") == "VERIFIED":
                    ns_msg = f"Akreditasi terverifikasi ({ns_screening.get('system')}): {ns_rank_hint}."
                elif ns_screening.get("lookup_status") == "PERIOD_UNVERIFIED":
                    ns_msg = (
                        f"Akreditasi {ns_rank_hint} terdaftar. Masa berlaku untuk tahun "
                        f"{paper.get('year') or '-'} perlu diverifikasi sebelum konfirmasi."
                    )
                elif ns_screening.get("lookup_status") == "AMBIGUOUS":
                    ns_msg = "Ada kandidat nama jurnal yang ambigu. Cocokkan ISSN dengan direktori resmi sebelum konfirmasi."
                elif ns_screening.get("lookup_status") == "NOT_FOUND":
                    ns_msg = "Jurnal tidak ditemukan pada direktori resmi HK-dir / ABDC yang diperiksa."
                elif ns_screening.get("lookup_status") == "NO_METADATA":
                    ns_msg = "ISSN tidak tersedia untuk verifikasi akreditasi non-SINTA."
                else:
                    ns_msg = ns_screening.get("notes") or ns_status_label
                self.non_sinta_lookup_message.setText(ns_msg)
                self.non_sinta_lookup_message.setToolTip(ns_screening.get("notes") or ns_msg)
            else:
                self.non_sinta_lookup_message.setText(
                    "Pilih artikel, lalu periksa akreditasi pada direktori resmi HK-dir / ABDC."
                )
                self.non_sinta_lookup_message.setToolTip("")

        def esc(value):
            if isinstance(value, (list, tuple)):
                value = ", ".join(map(str, value))
            return html_escape(str(value or "-"))

        doi = str(paper.get("doi") or "").strip()
        source_url = str(paper.get("url") or "").strip()
        if not source_url and doi:
            source_url = doi if doi.startswith(("http://", "https://")) else f"https://doi.org/{doi.removeprefix('doi:')}"

        links_html = []
        if source_url:
            links_html.append(f"<a href='{html_escape(source_url, quote=True)}'>Buka sumber artikel</a>")
        if doi:
            doi_url = doi if doi.startswith(("http://", "https://")) else f"https://doi.org/{doi.removeprefix('doi:')}"
            links_html.append(f"<a href='{html_escape(doi_url, quote=True)}'>Buka DOI</a>")
        sinta_url = screening.get("source_url") or self._sinta_search_url(paper)
        links_html.append(f"<a href='{html_escape(sinta_url, quote=True)}'>Buka pencarian SINTA</a>")
        candidate_url = str(screening.get("assistive_candidate_url") or "").strip()
        if candidate_url.startswith("https://sinta.kemdiktisaintek.go.id/journals/profile/"):
            links_html.append(
                f"<a href='{html_escape(candidate_url, quote=True)}'>Kandidat profil SINTA (Gemini)</a>"
            )
        ns_url = ns_screening.get("source_url") or self._non_sinta_search_url(paper)
        if ns_url:
            links_html.append(f"<a href='{html_escape(ns_url, quote=True)}'>Buka direktori HK-dir</a>")

        issn_display = ", ".join(issn_clean) if issn_clean else "Tidak tersedia"
        sinta_status = str(screening.get("status") or "UNVERIFIED")
        sinta_rank = screening.get("rank") or (sinta_status if sinta_status.startswith("S") else "-")
        lookup_status = str(screening.get("lookup_status") or "UNVERIFIED")
        match_method = {
            "ISSN": "ISSN persis",
            "JOURNAL_NAME": "Nama jurnal",
            "DOI": "DOI artikel persis",
            "MANUAL": "Konfirmasi pengguna",
            "NONE": "Belum ada kecocokan",
        }.get(str(screening.get("match_method") or "NONE"), "Belum diketahui")
        sinta_label = self._sinta_status_label(paper)
        period_start = screening.get("period_start")
        period_end = screening.get("period_end")
        period_str = (
            "Diverifikasi dari entri DOI artikel di SINTA"
            if screening.get("period_status") == "ARTICLE_RECORD"
            else f"{period_start} - {period_end}" if (period_start and period_end) else "Belum tersedia"
        )
        checked_year = screening.get("checked_for_year") or paper.get("year") or "-"
        verification_src = screening.get("verification_source") or "Direktori Resmi SINTA (Kemdiktisaintek RI)"
        checked_at = screening.get("verification_date") or screening.get("checked_at") or "-"
        display_journal_name = self._screening_journal_display_name(paper, screening)
        other_indexes = screening.get("other_indexes", {})
        other_index_lines = []
        if isinstance(other_indexes, dict) and other_indexes.get("checked_at"):
            for source_name, source_result in (other_indexes.get("sources", {}) or {}).items():
                if not isinstance(source_result, dict):
                    continue
                source_status = source_result.get("status")
                source_label = {
                    "INDEXED": "Terindeks",
                    "NOT_INDEXED": "Tidak ditemukan",
                    "NOT_CONFIGURED": "API belum dikonfigurasi",
                    "NO_METADATA": "ISSN tidak tersedia",
                    "FAILED": "Belum dapat diperiksa",
                    "UNAVAILABLE": "Akses tidak tersedia",
                }.get(source_status, "Belum diperiksa")
                source_journal = source_result.get("journal_name")
                source_line = f"<b>{esc(source_name)}:</b> {esc(source_label)}"
                if source_journal:
                    source_line += f" ({esc(source_journal)})"
                source_url = str(source_result.get("source_url") or "").strip()
                if source_url.startswith("https://"):
                    source_line += f" — <a href='{html_escape(source_url, quote=True)}'>sumber</a>"
                other_index_lines.append(source_line)
        if not other_index_lines:
            other_index_lines.append("Belum ada pemeriksaan indeks non-SINTA.")
        external_index_block = (
            "<div style='background:#F0F9FF;border:1px solid #BAE6FD;border-radius:6px;padding:8px 12px;margin:6px 0;'>"
            "<h3 style='margin:0 0 4px 0;color:#1E293B;'>Indeks jurnal lain</h3>"
            f"<p style='margin:2px 0;'>{'<br>'.join(other_index_lines)}</p>"
            "<p style='color:#64748B;font-size:8pt;margin-top:4px;'><i>Indeksasi bukan peringkat akreditasi. DOAJ adalah direktori jurnal akses terbuka; status Scopus membutuhkan akses API Elsevier.</i></p>"
            "</div>"
        )

        ambiguity_block = ""
        if screening.get("is_ambiguous"):
            ambiguity_block = (
                f"<div style='background:#FFFBEB;color:#92400E;padding:6px 10px;border-radius:6px;border:1px solid #FCD34D;margin:6px 0;'>"
                f"<b>Ambiguitas pencocokan:</b> {esc(_sinta_ui_text(screening.get('ambiguity_reason')) or 'Pencocokan menggunakan nama jurnal. Harap verifikasi manual.')}</div>"
            )

        notes_block = ""
        display_note = _sinta_ui_note(screening)
        if display_note:
            notes_block = f"<p style='color:#475569;margin:3px 0;'><b>Catatan:</b> {esc(display_note)}</p>"

        ns_label = self._non_sinta_status_label(paper)
        ns_system = ns_screening.get("system_name") or ns_screening.get("system") or ("Direktori Resmi HK-dir / ABDC" if ns_screening.get("found") else "HK-dir / ABDC")
        ns_rank = ns_screening.get("rank") or "-"
        ns_lookup_status = ns_screening.get("lookup_status") or "UNVERIFIED"
        ns_match_method = {
            "ISSN": "ISSN persis",
            "JOURNAL_NAME": "Nama jurnal",
            "MANUAL": "Konfirmasi pengguna",
            "NONE": "Belum ada kecocokan",
        }.get(str(ns_screening.get("match_method") or "NONE"), "Belum diketahui")
        ns_p_start = ns_screening.get("period_start")
        ns_p_end = ns_screening.get("period_end")
        if ns_p_start and ns_p_end:
            ns_period_str = f"{ns_p_start} - {ns_p_end}"
        elif ns_p_start:
            ns_period_str = f"Mulai {ns_p_start}"
        else:
            ns_period_str = "Belum tersedia / tidak dibatasi"
        ns_checked_year = ns_screening.get("checked_for_year") or paper.get("year") or "-"
        ns_checked_at = ns_screening.get("checked_at") or ns_screening.get("verification_date") or "-"
        ns_notes = ns_screening.get("notes") or ""

        ns_ambiguity_block = ""
        if ns_screening.get("is_ambiguous"):
            ns_ambiguity_block = (
                f"<div style='background:#FFFBEB;color:#92400E;padding:6px 10px;border-radius:6px;border:1px solid #FCD34D;margin:6px 0;'>"
                f"<b>Ambiguitas pencocokan:</b> {esc(ns_screening.get('ambiguity_reason') or 'Pencocokan menggunakan nama jurnal. Harap verifikasi manual.')}</div>"
            )

        ns_notes_block = ""
        if ns_notes:
            ns_notes_block = f"<p style='color:#475569;margin:3px 0;'><b>Catatan:</b> {esc(ns_notes)}</p>"

        non_sinta_accred_block = (
            "<div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:6px;padding:8px 12px;margin:6px 0;'>"
            f"<h3 style='margin:0 0 4px 0;color:#1E293B;'>Verifikasi Akreditasi Non-SINTA (HK-dir / ABDC): <span style='color:#0F766E;'>{esc(ns_label)}</span></h3>"
            f"<p style='margin:2px 0;'><b>Sistem / Direktori:</b> {esc(ns_system)}<br>"
            f"<b>Jurnal cocok:</b> {esc(ns_screening.get('journal_name') or paper.get('venue') or '-')}<br>"
            f"<b>ISSN artikel:</b> {esc(issn_display)} &nbsp;|&nbsp; <b>ISSN cocok:</b> {esc(ns_screening.get('issn_matched') or 'Tidak ada kecocokan ISSN')}<br>"
            f"<b>Metode pencocokan:</b> {esc(ns_match_method)} &nbsp;|&nbsp; <b>Peringkat / Tingkat:</b> {esc(ns_rank)}<br>"
            f"<b>Masa berlaku:</b> {esc(ns_period_str)} &nbsp;|&nbsp; <b>Status untuk tahun {esc(ns_checked_year)}:</b> {esc(ns_label)}<br>"
            f"<b>Status pemeriksaan:</b> {esc(ns_lookup_status)}<br>"
            f"<b>Sumber Pemeriksaan:</b> {esc(ns_screening.get('verification_source') or 'Direktori Resmi HK-dir / ABDC')} ({esc(ns_checked_at)})</p>"
            f"{ns_ambiguity_block}"
            f"{ns_notes_block}"
            "<p style='color:#64748B;font-size:8pt;margin-top:4px;'><i>Batasan: HK-dir (Kanalregisteret) dan ABDC Journal Quality List adalah sistem akreditasi/pemeringkatan resmi statutory/institusional. Indeksasi (DOAJ, Scopus) bukan akreditasi.</i></p>"
            "</div>"
        )

        rows_html = [
            f"<h2 style='color:#0F172A;margin-bottom:4px;'>{esc(paper.get('title') or 'Untitled')}</h2>",
            f"<p><b>Penulis:</b> {esc(paper.get('authors'))}<br>",
            f"<b>Tahun:</b> {esc(paper.get('year'))} &nbsp;|&nbsp; <b>Jurnal / Venue:</b> {esc(paper.get('venue'))}<br>",
            f"<b>ISSN:</b> {esc(issn_display)} &nbsp;|&nbsp; <b>DOI:</b> {esc(doi or '-')}<br>",
            f"<b>Sumber:</b> {esc(paper.get('source'))} &nbsp;|&nbsp; <b>Relevansi:</b> {esc(self._format_score(paper.get('relevance_score')))} &nbsp;|&nbsp; <b>Keyakinan Penggunaan Dataset:</b> {esc(paper.get('dataset_usage_confidence') or '-')}</p>",
            f"<p style='margin:6px 0;'>{' &nbsp;|&nbsp; '.join(links_html)}</p>",
            "<hr style='border:0;border-top:1px solid #E2E8F0;margin:6px 0;'>",
            "<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;padding:8px 12px;margin:6px 0;'>",
            f"<h3 style='margin:0 0 4px 0;color:#1E293B;'>Verifikasi Akreditasi SINTA: <span style='color:#2563EB;'>{esc(sinta_label)}</span></h3>",
            f"<p style='margin:2px 0;'><b>Jurnal:</b> {esc(display_journal_name)}<br>",
            f"<b>ISSN artikel:</b> {esc(issn_display)} &nbsp;|&nbsp; <b>ISSN cocok:</b> {esc(screening.get('issn_matched') or 'Tidak ada kecocokan ISSN')}<br>",
            f"<b>Metode pencocokan:</b> {esc(match_method)} &nbsp;|&nbsp; <b>Peringkat terkini:</b> {esc(sinta_rank)}<br>",
            f"<b>Masa berlaku SK:</b> {esc(period_str)} &nbsp;|&nbsp; <b>Status untuk tahun {esc(checked_year)}:</b> {esc(sinta_label)}<br>",
            f"<b>Status pemeriksaan:</b> {esc(self._sinta_lookup_label(lookup_status))}<br>",
            f"<b>Sumber Pemeriksaan:</b> {esc(verification_src)} ({esc(checked_at)})</p>",
            ambiguity_block,
            notes_block,
            "<p style='color:#64748B;font-size:8pt;margin-top:4px;'><i>Batasan: Keterindeksan di Scopus, Crossref, atau OpenAlex adalah indeksasi internasional dan BUKAN akreditasi SINTA. Akreditasi SINTA diterbitkan secara nasional oleh Kemdiktisaintek RI.</i></p>",
            "</div>",
            non_sinta_accred_block,
            external_index_block,
        ]

        abstract = paper.get("abstract")
        if abstract:
            rows_html.append(f"<h3 style='margin-top:8px;'>Abstrak</h3><p style='color:#334155;text-align:justify;'>{esc(abstract)}</p>")

        self.paper_detail.setHtml("".join(rows_html))

    def _update_sinta_save_enabled(self, *_):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        has_identifier = False
        if isinstance(paper_index, int) and paper_index < len(papers):
            paper = papers[paper_index]
            screening = self._sinta_screening(paper)
            has_identifier = bool(
                paper.get("venue") or paper.get("issn") or screening.get("source_url")
            )
        self.save_sinta_button.setEnabled(
            has_identifier
            and self.sinta_status_combo.currentData() != "UNVERIFIED"
            and self.sinta_period_confirmed.isChecked()
        )

    def _selected_paper_index(self):
        rows = self.paper_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.paper_table.item(rows[0].row(), 0)
        return item.data(Qt.UserRole) if item else None

    def _check_pending_sinta_batch(self):
        if self._sinta_thread is not None or self._journal_index_thread is not None:
            return

        tasks = []
        for paper_index, paper in zip(self._all_paper_result_indices, self._all_papers):
            screening = self._sinta_screening(paper)
            if str(screening.get("status") or "UNVERIFIED") != "UNVERIFIED":
                continue
            checked_year = str(screening.get("checked_for_year") or "")
            same_year = checked_year == str(paper.get("year") or "")
            reusable_lookup = str(screening.get("lookup_status") or "") in {
                "VERIFIED", "PERIOD_UNVERIFIED", "AMBIGUOUS"
            }
            if screening.get("found") and same_year and reusable_lookup:
                continue

            issn_values = paper.get("issn", [])
            if isinstance(issn_values, str):
                issn_values = [issn_values]
            issn_values = [str(value).strip() for value in issn_values if str(value).strip()]
            journal_name = str(paper.get("venue") or "").strip()
            if not issn_values and not journal_name:
                self._apply_sinta_result(paper, {
                    "found": False,
                    "status": "UNVERIFIED",
                    "lookup_status": "NO_METADATA",
                    "checked_for_year": paper.get("year"),
                    "notes": "Artikel ini tidak memiliki ISSN atau nama jurnal. Lengkapi metadata jurnal sebelum pemeriksaan SINTA.",
                    "source_url": self._sinta_search_url(paper),
                })
                self._persist_sinta_screening(paper)
                continue
            tasks.append({
                "paper_index": paper_index,
                "identity": self._paper_identity(paper),
                "issn": issn_values,
                "journal": journal_name,
                "year": paper.get("year"),
                "doi": paper.get("doi"),
            })

        if not tasks:
            self._render_filtered_papers()
            metadata_missing = sum(
                self._sinta_screening(paper).get("lookup_status") == "NO_METADATA"
                for paper in self._all_papers
            )
            if metadata_missing:
                message = (
                    f"Tidak ada jurnal lain untuk diperiksa. {metadata_missing} artikel "
                    "tidak memiliki ISSN atau nama jurnal; lengkapi metadata jika tersedia. "
                    "Hasil pemeriksaan sebelumnya tetap tersimpan."
                )
            else:
                message = (
                    "Tidak ada jurnal baru untuk diperiksa. Hasil yang sudah ada tetap tersimpan; "
                    "artikel dengan periode SK belum terverifikasi perlu konfirmasi manual."
                )
            self.sinta_lookup_message.setText(
                message
            )
            return

        self._sinta_batch_found = 0
        self._sinta_batch_manual = 0
        self.sinta_batch_button.setEnabled(False)
        self.other_index_button.setEnabled(False)
        self.sinta_batch_button.setText("Memeriksa jurnal...")
        self.check_sinta_button.setEnabled(False)
        self.sinta_lookup_progress.setRange(0, len(tasks))
        self.sinta_lookup_progress.setValue(0)
        self.sinta_lookup_progress.setVisible(True)
        self.sinta_lookup_message.setText(f"Memeriksa jurnal 0 dari {len(tasks)}...")

        self._sinta_thread = QThread(self)
        self._sinta_worker = SintaBatchLookupWorker(tasks)
        self._sinta_worker.moveToThread(self._sinta_thread)
        self._sinta_thread.started.connect(self._sinta_worker.run)
        self._sinta_worker.progress.connect(self._on_sinta_batch_progress)
        self._sinta_worker.finished.connect(self._finish_sinta_batch)
        self._sinta_worker.failed.connect(self._on_sinta_batch_failed)
        self._sinta_worker.finished.connect(self._sinta_thread.quit)
        self._sinta_worker.failed.connect(self._sinta_thread.quit)
        self._sinta_thread.finished.connect(self._sinta_worker.deleteLater)
        self._sinta_thread.finished.connect(self._cleanup_sinta_lookup)
        self._sinta_thread.start()

    def _check_pending_other_indexes(self):
        if self._journal_index_thread is not None or self._sinta_thread is not None:
            return
        papers = self.research_result.get("papers", [])
        if not isinstance(papers, list) or not papers:
            self.sinta_lookup_message.setText(
                "Jalankan Academic Research terlebih dahulu untuk memuat jurnal."
            )
            return

        tasks = []
        for paper_index, paper in enumerate(papers):
            if not isinstance(paper, dict):
                continue
            screening = self._sinta_screening(paper)
            existing = screening.get("other_indexes", {})
            if isinstance(existing, dict) and existing.get("checked_at"):
                source_statuses = [
                    source.get("status")
                    for source in (existing.get("sources", {}) or {}).values()
                    if isinstance(source, dict)
                ]
                if source_statuses and all(
                    status in {"INDEXED", "NOT_INDEXED", "NO_METADATA"}
                    for status in source_statuses
                ):
                    continue
            issn_values = paper.get("issn", [])
            if isinstance(issn_values, str):
                issn_values = [issn_values]
            tasks.append({
                "paper_index": paper_index,
                "identity": self._paper_identity(paper),
                "issn": [str(value).strip() for value in issn_values if str(value).strip()],
            })

        if not tasks:
            self.sinta_lookup_message.setText(
                "Indeks lain sudah diperiksa untuk semua artikel. Hasil yang ada tetap tersimpan."
            )
            return

        self.other_index_button.setEnabled(False)
        self.sinta_batch_button.setEnabled(False)
        self.check_sinta_button.setEnabled(False)
        self.other_index_button.setText("Memeriksa indeks...")
        self.sinta_lookup_progress.setRange(0, len(tasks))
        self.sinta_lookup_progress.setValue(0)
        self.sinta_lookup_progress.setVisible(True)
        self.sinta_lookup_message.setText(
            f"Memeriksa DOAJ dan Scopus (jika akses tersedia): 0 dari {len(tasks)}..."
        )

        self._journal_index_thread = QThread(self)
        self._journal_index_worker = JournalIndexBatchLookupWorker(tasks)
        self._journal_index_worker.moveToThread(self._journal_index_thread)
        self._journal_index_thread.started.connect(self._journal_index_worker.run)
        self._journal_index_worker.progress.connect(self._on_other_index_progress)
        self._journal_index_worker.finished.connect(self._finish_other_index_batch)
        self._journal_index_worker.failed.connect(self._on_other_index_failed)
        self._journal_index_worker.finished.connect(self._journal_index_thread.quit)
        self._journal_index_worker.failed.connect(self._journal_index_thread.quit)
        self._journal_index_thread.finished.connect(self._journal_index_worker.deleteLater)
        self._journal_index_thread.finished.connect(self._cleanup_other_index_batch)
        self._journal_index_thread.start()

    @Slot(int, int, int, str, object)
    def _on_other_index_progress(self, current, total, paper_index, identity, result):
        papers = self.research_result.get("papers", [])
        if (
            isinstance(paper_index, int)
            and 0 <= paper_index < len(papers)
            and self._paper_identity(papers[paper_index]) == identity
        ):
            paper = papers[paper_index]
            screening = self._sinta_screening(paper)
            screening["other_indexes"] = dict(result)
            paper["sinta_screening"] = screening
            self._persist_sinta_screening(paper)
            label = str(paper.get("venue") or "Jurnal")
        else:
            label = "Jurnal"
        self.sinta_lookup_progress.setValue(current)
        self.sinta_lookup_message.setText(
            f"Pemeriksaan indeks {current} dari {total} | {label}"
        )
        if not self._sinta_table_refresh_timer.isActive():
            self._sinta_table_refresh_timer.start()

    @Slot(int)
    def _finish_other_index_batch(self, total):
        self._sinta_table_refresh_timer.stop()
        papers = self.research_result.get("papers", [])
        indexed = sum(
            bool(self._sinta_screening(paper).get("other_indexes", {}).get("indexes"))
            for paper in papers if isinstance(paper, dict)
        )
        partial = sum(
            self._sinta_screening(paper).get("other_indexes", {}).get("status") == "PARTIAL"
            for paper in papers if isinstance(paper, dict)
        )
        self.sinta_lookup_message.setText(
            f"Pemeriksaan selesai: {indexed} artikel cocok dengan DOAJ/Scopus dari {total} yang diperiksa. "
            f"{partial} hasil parsial; akses Scopus membutuhkan API Elsevier."
        )
        project_id = getattr(self.main_window, "current_project_id", None)
        if project_id is not None:
            try:
                repository = self.main_window.repository
                repository.save_workflow_section(project_id, "research", self.research_result)
                repository.save_research_result(project_id, self.research_result)
            except Exception as error:
                print(f"Warning: hasil pemeriksaan indeks tidak dapat disimpan: {error}")
        self._render_filtered_papers()

    @Slot(str)
    def _on_other_index_failed(self, error):
        self.sinta_lookup_message.setText(
            f"Pemeriksaan indeks berhenti sebelum selesai: {error}"
        )

    def _cleanup_other_index_batch(self):
        self.sinta_lookup_progress.setVisible(False)
        self.other_index_button.setText("Periksa Indeks Lain")
        self.other_index_button.setEnabled(bool(self.research_result.get("papers")))
        self.sinta_batch_button.setEnabled(bool(self._all_papers))
        self.check_sinta_button.setEnabled(bool(self._all_papers))
        self._journal_index_worker = None
        self._journal_index_thread = None

    @Slot(int, int, int, str, object)
    def _on_sinta_batch_progress(self, current, total, paper_index, identity, result):
        papers = self.research_result.get("papers", [])
        if isinstance(paper_index, int) and 0 <= paper_index < len(papers):
            paper = papers[paper_index]
            if self._paper_identity(paper) == identity:
                self._apply_sinta_result(paper, result)
                self._sinta_lookup_cache[self._screening_metadata_key(paper)] = dict(result)
                self._persist_sinta_screening(paper)
                state = str(result.get("lookup_status") or "UNVERIFIED")
                if result.get("found") and state not in {"AMBIGUOUS", "FAILED"}:
                    self._sinta_batch_found += 1
                if state in {
                    "FAILED", "UNVERIFIED", "AMBIGUOUS", "PERIOD_UNVERIFIED",
                    "NO_MATCH", "NO_METADATA", "RANK_UNAVAILABLE",
                }:
                    self._sinta_batch_manual += 1
                journal = str(result.get("journal_name") or paper.get("venue") or "Jurnal")
            else:
                journal = "Jurnal"
        else:
            journal = "Jurnal"
        self.sinta_lookup_progress.setValue(current)
        self.sinta_lookup_message.setText(
            f"{current} dari {total} artikel | Memeriksa {journal} | "
            f"cocok {self._sinta_batch_found}, perlu ditinjau {self._sinta_batch_manual}"
        )
        refresh_timer = getattr(self, "_sinta_table_refresh_timer", None)
        if refresh_timer is None:
            self._render_filtered_papers()
        elif not refresh_timer.isActive():
            refresh_timer.start()

    @Slot(int)
    def _finish_sinta_batch(self, total):
        self._sinta_table_refresh_timer.stop()
        missing_metadata = sum(
            self._sinta_screening(paper).get("lookup_status") == "NO_METADATA"
            for paper in self._all_papers
        )
        missing_note = (
            f" {missing_metadata} artikel tidak memiliki ISSN/nama jurnal."
            if missing_metadata
            else ""
        )
        self.sinta_lookup_message.setText(
            f"Pemeriksaan selesai: jurnal cocok untuk {self._sinta_batch_found} artikel dari {total} yang diperiksa; "
            f"{self._sinta_batch_manual} perlu pemeriksaan manual. "
            f"Hasil yang tersedia sudah disimpan ke ruang kerja lokal.{missing_note}"
        )
        project_id = getattr(self.main_window, "current_project_id", None)
        if project_id is not None:
            try:
                repository = self.main_window.repository
                repository.save_workflow_section(project_id, "research", self.research_result)
                repository.save_research_result(project_id, self.research_result)
            except Exception as error:
                print(f"Warning: hasil batch SINTA tidak dapat disimpan: {error}")
        self._render_filtered_papers()

    def _persist_sinta_screening(self, paper):
        repository = getattr(self.main_window, "repository", None)
        if repository is None:
            return
        try:
            repository.save_journal_screening(
                paper_key=self._paper_identity(paper),
                screening_data=self._sinta_screening(paper),
                doi=str(paper.get("doi") or "").strip().lower(),
            )
        except Exception as error:
            print(f"Warning: hasil screening SINTA belum tersimpan: {error}")

    def _check_sinta_in_app(self):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            return
        if self._sinta_thread is not None or self._journal_index_thread is not None:
            return

        paper = papers[paper_index]
        issn_values = paper.get("issn", [])
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        issn_clean = [str(i).strip() for i in issn_values if str(i).strip()]
        venue = str(paper.get("venue") or "").strip()

        if not issn_clean and not venue:
            QMessageBox.information(
                self,
                "Informasi Tidak Lengkap",
                "Paper ini tidak memiliki ISSN maupun nama jurnal untuk diverifikasi di SINTA.",
            )
            return

        cache_key = self._screening_metadata_key(paper)
        cached_result = self._sinta_lookup_cache.get(cache_key)
        if cached_result is None:
            existing_screening = self._sinta_screening(paper)
            if (
                existing_screening.get("metadata_key") == cache_key
                and existing_screening.get("found")
            ):
                cached_result = existing_screening
        if cached_result is not None:
            self._on_sinta_lookup_result(
                paper_index, self._paper_identity(paper), cached_result
            )
            return

        self.check_sinta_button.setEnabled(False)
        self.sinta_batch_button.setEnabled(False)
        self.other_index_button.setEnabled(False)
        self.check_sinta_button.setText("Memeriksa SINTA...")
        self.sinta_lookup_progress.setVisible(True)
        self.sinta_lookup_message.setText(
            "Mencari jurnal di direktori resmi SINTA (memeriksa ISSN terlebih dahulu)..."
        )
        self._sinta_thread = QThread(self)
        paper_identity = self._paper_identity(paper)
        query = issn_clean[0] if issn_clean else venue
        self._sinta_worker = SintaLookupWorker(
            query=query,
            issn_list=issn_clean,
            journal_name=venue,
            paper_year=paper.get("year"),
            paper_index=paper_index,
            paper_identity=paper_identity,
            article_doi=str(paper.get("doi") or ""),
        )
        self._sinta_worker.moveToThread(self._sinta_thread)
        self._sinta_thread.started.connect(self._sinta_worker.run)
        self._sinta_worker.finished.connect(self._on_sinta_lookup_result)
        self._sinta_worker.failed.connect(self._on_sinta_lookup_failed)
        self._sinta_worker.finished.connect(self._sinta_thread.quit)
        self._sinta_worker.failed.connect(self._sinta_thread.quit)
        self._sinta_thread.finished.connect(self._sinta_worker.deleteLater)
        self._sinta_thread.finished.connect(self._cleanup_sinta_lookup)
        self._sinta_thread.start()

    @staticmethod
    def _paper_identity(paper):
        return "|".join((
            re.sub(r"\s+", " ", str(paper.get("title") or "").casefold()).strip(),
            re.sub(r"\s+", " ", str(paper.get("doi") or "").casefold()).strip(),
        ))

    @staticmethod
    def _apply_sinta_result(paper, result):
        previous = ResearchPage._sinta_screening(paper)
        paper["sinta_screening"] = {
            **previous,
            **result,
            "status": result.get("status") or "UNVERIFIED",
            "checked_at": time.strftime("%Y-%m-%d %H:%M"),
            "checked_for_year": result.get("checked_for_year") or paper.get("year"),
            "source_url": result.get("source_url") or previous.get("source_url"),
            "metadata_key": ResearchPage._screening_metadata_key(paper),
        }

    @Slot(int, str, object)
    def _on_sinta_lookup_result(self, paper_index, paper_identity, result):
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            return
        paper = papers[paper_index]
        if self._paper_identity(paper) != paper_identity:
            return

        issn_clean = [str(i).strip() for i in paper.get("issn", []) if str(i).strip()]
        venue = str(paper.get("venue") or "").strip()
        cache_key = self._screening_metadata_key(paper)
        self._sinta_lookup_cache[cache_key] = result
        self._apply_sinta_result(paper, result)
        self._persist_sinta_screening(paper)

        if not result.get("found"):
            self.sinta_lookup_message.setText(
                _sinta_ui_note(result) or "Hasil pemeriksaan belum dapat diverifikasi."
            )
            self._render_filtered_papers()
            self._show_selected_paper()
            return

        rank = result.get("rank")
        status = result.get("status")
        match_method = result.get("match_method")
        is_ambiguous = result.get("is_ambiguous")

        if is_ambiguous:
            self.sinta_lookup_message.setText(
                f"Kecocokan nama jurnal (ambigu): {result.get('journal_name')} ({rank or '-'}). "
                "Harap konfirmasi kesesuaian jurnal & masa berlaku SK secara manual."
            )
        elif rank and status == rank:
            if match_method == "DOI":
                message = f"DOI artikel cocok dengan entri resmi SINTA: {rank}."
            else:
                message = (
                    f"Kecocokan {match_method}: {result.get('journal_name')} ({rank}). "
                    f"Masa berlaku SK terverifikasi untuk tahun {paper.get('year')}."
                )
            self.sinta_lookup_message.setText(message)
        elif rank and status == "UNVERIFIED":
            self.sinta_lookup_message.setText(
                f"Jurnal terdaftar: {result.get('journal_name')} ({rank}). "
                f"Masa berlaku SK untuk tahun {paper.get('year') or '-'} perlu dikonfirmasi manual."
            )
        else:
            self.sinta_lookup_message.setText(
                _sinta_ui_note(result) or f"Pemeriksaan selesai: status {status}."
            )

        combo_val = status
        combo_index = self.sinta_status_combo.findData(combo_val)
        if combo_index >= 0:
            self.sinta_status_combo.setCurrentIndex(combo_index)

        self._render_filtered_papers()
        self._show_selected_paper()
        self._update_sinta_save_enabled()

    @Slot(str)
    def _on_sinta_lookup_failed(self, error):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        if isinstance(paper_index, int) and 0 <= paper_index < len(papers):
            paper = papers[paper_index]
            self._apply_sinta_result(paper, {
                "found": False,
                "status": "UNVERIFIED",
                "lookup_status": "FAILED",
                "checked_for_year": paper.get("year"),
                "verification_date": time.strftime("%Y-%m-%d %H:%M"),
                "notes": "Direktori SINTA belum dapat diperiksa. Periksa koneksi internet, lalu coba lagi.",
                "source_url": self._sinta_search_url(paper),
            })
        self.sinta_lookup_message.setText(
            "Direktori SINTA belum dapat diperiksa. Periksa koneksi internet, lalu coba lagi."
        )
        self._render_filtered_papers()

    @Slot(str)
    def _on_sinta_batch_failed(self, error):
        self.sinta_lookup_message.setText(
            "Pemeriksaan jurnal terhenti sebelum selesai. Jurnal yang belum diperiksa dapat dicoba kembali."
        )

    def _cleanup_sinta_lookup(self):
        self.sinta_lookup_progress.setVisible(False)
        self.check_sinta_button.setText("Periksa di SINTA")
        self.sinta_batch_button.setText("Periksa jurnal belum dicek")
        self.sinta_batch_button.setEnabled(bool(self._all_papers))
        self.other_index_button.setEnabled(bool(self._all_papers))
        self._sinta_worker = None
        self._sinta_thread = None
        self._show_selected_paper()

    def _open_sinta_in_browser(self):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            url = "https://sinta.kemdiktisaintek.go.id/journals"
        else:
            url = self._sinta_search_url(papers[paper_index])
        QDesktopServices.openUrl(QUrl(url))

    def _save_sinta_screening(self):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            return

        status = self.sinta_status_combo.currentData()
        if status == "UNVERIFIED":
            QMessageBox.information(
                self,
                "Status SINTA Belum Dipilih",
                "Silakan periksa direktori resmi SINTA dan pilih status akreditasi yang sesuai.",
            )
            return
        if not self.sinta_period_confirmed.isChecked():
            QMessageBox.information(
                self,
                "Konfirmasi Masa Berlaku Akreditasi",
                "Harap konfirmasi bahwa status akreditasi ini mencakup tahun publikasi artikel sebelum menyimpan.",
            )
            return

        paper = papers[paper_index]
        previous = self._sinta_screening(paper)
        updated_screening = {
            **previous,
            "status": status,
            "lookup_status": "MANUALLY_CONFIRMED",
            "rank": status if status.startswith("S") else previous.get("rank"),
            "checked_at": time.strftime("%Y-%m-%d %H:%M"),
            "checked_for_year": paper.get("year"),
            "period_confirmed": True,
            "period_status": "MANUALLY_CONFIRMED",
            "metadata_key": self._screening_metadata_key(paper),
            "verification_source": "Konfirmasi Pengguna terhadap Direktori Resmi SINTA",
            "source_url": previous.get("source_url") or self._sinta_search_url(paper),
        }
        paper["sinta_screening"] = updated_screening

        # Simpan secara permanen ke SQLite lokal
        repo = getattr(self.main_window, "repository", None)
        if repo is not None:
            doi = str(paper.get("doi") or "").strip().lower()
            key = self._paper_identity(paper)
            try:
                repo.save_journal_screening(
                    paper_key=key,
                    screening_data=updated_screening,
                    doi=doi,
                )
            except Exception as exc:
                print(f"Warning: gagal menyimpan status screening ke database: {exc}")

            project_id = getattr(self.main_window, "current_project_id", None)
            if project_id is not None:
                try:
                    repo.save_workflow_section(
                        project_id, "research", self.research_result
                    )
                    repo.save_research_result(
                        project_id, self.research_result
                    )
                except Exception as error:
                    print(f"Warning: workflow riset tidak dapat disimpan: {error}")

        self._apply_paper_filter()
        self._show_selected_paper()

    @staticmethod
    def _sinta_search_url(paper):
        issn_values = paper.get("issn", [])
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        query = next((str(value).strip() for value in issn_values if value), "")
        if not query:
            query = str(paper.get("venue") or "").strip()
        if not query:
            return "https://sinta.kemdiktisaintek.go.id/journals"
        return "https://sinta.kemdiktisaintek.go.id/journals?q=" + quote_plus(query)

    def _update_non_sinta_save_enabled(self, *_):
        if not hasattr(self, "save_non_sinta_button") or not hasattr(self, "non_sinta_status_combo"):
            return
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        has_identifier = False
        if isinstance(paper_index, int) and paper_index < len(papers):
            paper = papers[paper_index]
            has_identifier = bool(paper.get("venue") or paper.get("issn"))
        self.save_non_sinta_button.setEnabled(
            has_identifier
            and self.non_sinta_status_combo.currentData() != "UNVERIFIED"
            and self.non_sinta_period_confirmed.isChecked()
        )

    def _open_non_sinta_in_browser(self):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            url = "https://kanalregister.hkdir.no/publiseringskanaler/Kanalfaeste"
        else:
            ns_screening = papers[paper_index].get("accreditation_screening") or {}
            url = ns_screening.get("source_url") or self._non_sinta_search_url(papers[paper_index])
        QDesktopServices.openUrl(QUrl(url))

    def _save_non_sinta_screening(self):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            return

        status = self.non_sinta_status_combo.currentData()
        if status == "UNVERIFIED":
            QMessageBox.information(
                self,
                "Status Akreditasi Belum Dipilih",
                "Silakan pilih status akreditasi non-SINTA yang sesuai sebelum menyimpan.",
            )
            return
        if not self.non_sinta_period_confirmed.isChecked():
            QMessageBox.information(
                self,
                "Konfirmasi Masa Berlaku Akreditasi",
                "Harap konfirmasi bahwa status akreditasi ini mencakup tahun publikasi artikel sebelum menyimpan.",
            )
            return

        paper = papers[paper_index]
        previous = paper.get("accreditation_screening") or {}
        rank_val = status
        system_val = previous.get("system") or ("HKDIR" if status.startswith("HKDIR") else ("ABDC" if status.startswith("ABDC") else "Direktori Resmi"))
        system_name_val = previous.get("system_name") or ("Norwegian Register (HK-dir / Kanalregisteret)" if status.startswith("HKDIR") else ("ABDC Journal Quality List" if status.startswith("ABDC") else "Direktori Resmi"))
        updated_screening = {
            **previous,
            "found": status not in {"NOT_FOUND", "UNVERIFIED"},
            "status": status,
            "lookup_status": "MANUALLY_CONFIRMED",
            "rank": rank_val,
            "system": system_val,
            "system_name": system_name_val,
            "checked_at": time.strftime("%Y-%m-%d %H:%M"),
            "checked_for_year": paper.get("year"),
            "period_confirmed": True,
            "period_status": "MANUALLY_CONFIRMED",
            "verification_source": f"Konfirmasi Pengguna terhadap Direktori Resmi {system_val}",
            "source_url": previous.get("source_url") or self._non_sinta_search_url(paper),
        }
        paper["accreditation_screening"] = updated_screening
        self._persist_non_sinta_screening(paper)

        project_id = getattr(self.main_window, "current_project_id", None)
        repo = getattr(self.main_window, "repository", None)
        if repo is not None and project_id is not None:
            try:
                repo.save_workflow_section(project_id, "research", self.research_result)
                repo.save_research_result(project_id, self.research_result)
            except Exception as error:
                print(f"Warning: workflow riset tidak dapat disimpan: {error}")

        self._apply_paper_filter()
        self._show_selected_paper()

    def _persist_non_sinta_screening(self, paper):
        repository = getattr(self.main_window, "repository", None)
        if repository is None:
            return
        screening = paper.get("accreditation_screening")
        if not isinstance(screening, dict):
            return
        try:
            repository.save_non_sinta_screening(
                paper_key=self._paper_identity(paper),
                screening_data=screening,
                doi=str(paper.get("doi") or "").strip().lower(),
            )
        except Exception as error:
            print(f"Warning: hasil screening non-SINTA belum tersimpan: {error}")

    def _check_non_sinta_in_app(self):
        paper_index = self._selected_paper_index()
        papers = self.research_result.get("papers", [])
        if not isinstance(paper_index, int) or paper_index >= len(papers):
            return
        if self._non_sinta_thread is not None:
            return

        paper = papers[paper_index]
        issn_values = paper.get("issn", [])
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        issn_clean = [str(i).strip() for i in issn_values if str(i).strip()]
        venue = str(paper.get("venue") or "").strip()

        if not issn_clean and not venue:
            QMessageBox.information(
                self,
                "Informasi Tidak Lengkap",
                "Paper ini tidak memiliki ISSN maupun nama jurnal untuk diverifikasi.",
            )
            return

        from app.research.accreditation_verifier import AccreditationVerifier
        verifier = AccreditationVerifier()
        result = verifier.verify(
            issn_list=issn_clean,
            journal_name=venue,
            article_year=paper.get("year"),
        ).to_dict()

        paper["accreditation_screening"] = result
        self._persist_non_sinta_screening(paper)

        status = result.get("status") or "UNVERIFIED"
        combo_index = self.non_sinta_status_combo.findData(status)
        if combo_index >= 0:
            self.non_sinta_status_combo.setCurrentIndex(combo_index)

        self._render_filtered_papers()
        self._show_selected_paper()
        self._update_non_sinta_save_enabled()

    def _check_pending_non_sinta_batch(self):
        if self._non_sinta_thread is not None:
            return

        tasks = []
        for paper_index, paper in zip(self._all_paper_result_indices, self._all_papers):
            screening = paper.get("accreditation_screening") or {}
            checked_year = str(screening.get("checked_for_year") or "")
            same_year = checked_year == str(paper.get("year") or "")
            if same_year and screening.get("official_assessments_status") in {"CHECKED", "NO_METADATA"}:
                continue

            issn_values = paper.get("issn", [])
            if isinstance(issn_values, str):
                issn_values = [issn_values]
            issn_values = [str(value).strip() for value in issn_values if str(value).strip()]
            journal_name = str(paper.get("venue") or "").strip()
            if not issn_values and not journal_name:
                no_meta = {
                    "found": False,
                    "status": "UNVERIFIED",
                    "lookup_status": "NO_METADATA",
                    "checked_for_year": paper.get("year"),
                    "notes": "Artikel ini tidak memiliki ISSN atau nama jurnal untuk pemeriksaan akreditasi non-SINTA.",
                }
                paper["accreditation_screening"] = no_meta
                self._persist_non_sinta_screening(paper)
                continue
            tasks.append({
                "paper_index": paper_index,
                "identity": self._paper_identity(paper),
                "issn": issn_values,
                "journal": journal_name,
                "year": paper.get("year"),
                "doi": paper.get("doi"),
            })

        if not tasks:
            self._render_filtered_papers()
            self.non_sinta_lookup_message.setText(
                "Semua artikel sudah diperiksa pada direktori dan daftar penilaian yang didukung."
            )
            papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
            if papers_tool_page is not None:
                checked_count = sum(
                    1 for paper in self._all_papers
                    if (paper.get("accreditation_screening") or {}).get("official_assessments_status") == "CHECKED"
                )
                papers_tool_page.set_assessment_lookup_status(
                    checked_count, checked_count, finished=True
                )
            return

        self._non_sinta_batch_found = 0
        self._non_sinta_batch_manual = 0
        self.non_sinta_batch_button.setEnabled(False)
        self.check_non_sinta_button.setEnabled(False)
        self.non_sinta_batch_button.setText("Memeriksa akreditasi...")
        self.non_sinta_lookup_progress.setRange(0, len(tasks))
        self.non_sinta_lookup_progress.setValue(0)
        self.non_sinta_lookup_progress.setVisible(True)
        self.non_sinta_lookup_message.setText(f"Memeriksa akreditasi 0 dari {len(tasks)}...")
        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.set_assessment_lookup_status(0, len(tasks))

        self._non_sinta_thread = QThread(self)
        self._non_sinta_worker = NonSintaBatchLookupWorker(tasks)
        self._non_sinta_worker.moveToThread(self._non_sinta_thread)
        self._non_sinta_thread.started.connect(self._non_sinta_worker.run)
        self._non_sinta_worker.progress.connect(self._on_non_sinta_batch_progress)
        self._non_sinta_worker.finished.connect(self._finish_non_sinta_batch)
        self._non_sinta_worker.failed.connect(self._on_non_sinta_batch_failed)
        self._non_sinta_worker.finished.connect(self._non_sinta_thread.quit)
        self._non_sinta_worker.failed.connect(self._non_sinta_thread.quit)
        self._non_sinta_thread.finished.connect(self._non_sinta_worker.deleteLater)
        self._non_sinta_thread.finished.connect(self._cleanup_non_sinta_batch)
        self._non_sinta_thread.start()

    @Slot(int, int, int, str, object)
    def _on_non_sinta_batch_progress(self, current, total, paper_index, identity, result):
        papers = self.research_result.get("papers", [])
        if isinstance(paper_index, int) and 0 <= paper_index < len(papers):
            paper = papers[paper_index]
            if self._paper_identity(paper) == identity:
                paper["accreditation_screening"] = dict(result)
                self._persist_non_sinta_screening(paper)
                state = str(result.get("lookup_status") or "UNVERIFIED")
                if result.get("found") and state not in {"AMBIGUOUS", "FAILED"}:
                    self._non_sinta_batch_found += 1
                if state in {
                    "FAILED", "UNVERIFIED", "AMBIGUOUS", "PERIOD_UNVERIFIED",
                    "NO_MATCH", "NO_METADATA", "EXPIRED",
                }:
                    self._non_sinta_batch_manual += 1
                journal = str(result.get("journal_name") or paper.get("venue") or "Jurnal")
            else:
                journal = "Jurnal"
        else:
            journal = "Jurnal"
        self.non_sinta_lookup_progress.setValue(current)
        self.non_sinta_lookup_message.setText(
            f"{current} dari {total} artikel | Memeriksa {journal} | "
            f"cocok {self._non_sinta_batch_found}, perlu ditinjau {self._non_sinta_batch_manual}"
        )
        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.set_assessment_lookup_status(current, total)
        refresh_timer = getattr(self, "_sinta_table_refresh_timer", None)
        if refresh_timer is None:
            self._render_filtered_papers()
        elif not refresh_timer.isActive():
            refresh_timer.start()

    @Slot(int)
    def _finish_non_sinta_batch(self, total):
        if hasattr(self, "_sinta_table_refresh_timer"):
            self._sinta_table_refresh_timer.stop()
        self.non_sinta_lookup_message.setText(
            f"Pemeriksaan direktori/penilaian jurnal selesai: cocok {self._non_sinta_batch_found} dari {total} artikel; "
            f"{self._non_sinta_batch_manual} perlu tinjauan/konfirmasi manual."
        )
        papers_tool_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_tool_page is not None:
            papers_tool_page.set_assessment_lookup_status(total, total, finished=True)
        project_id = getattr(self.main_window, "current_project_id", None)
        if project_id is not None:
            try:
                repository = self.main_window.repository
                repository.save_workflow_section(project_id, "research", self.research_result)
                repository.save_research_result(project_id, self.research_result)
            except Exception as error:
                print(f"Warning: hasil batch non-SINTA tidak dapat disimpan: {error}")
        self._render_filtered_papers()

    @Slot(str)
    def _on_non_sinta_batch_failed(self, error):
        self.non_sinta_lookup_message.setText(
            f"Pemeriksaan akreditasi non-SINTA terhenti: {error}"
        )

    def _cleanup_non_sinta_batch(self):
        self.non_sinta_lookup_progress.setVisible(False)
        self.check_non_sinta_button.setEnabled(bool(self._all_papers))
        self.non_sinta_batch_button.setText("Periksa akreditasi non-SINTA")
        self.non_sinta_batch_button.setEnabled(bool(self._all_papers))
        self._non_sinta_worker = None
        self._non_sinta_thread = None
        self._show_selected_paper()

    @staticmethod
    def _non_sinta_search_url(paper):
        issn_values = paper.get("issn", [])
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        query = next((str(value).strip() for value in issn_values if value), "")
        if not query:
            query = str(paper.get("venue") or "").strip()
        if not query:
            return "https://kanalregister.hkdir.no/publiseringskanaler/Kanalfaeste"
        return "https://kanalregister.hkdir.no/publiseringskanaler/Kanalfaeste?query=" + quote_plus(query)

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _set_metric(
        card,
        value,
    ):

        if hasattr(
            card,
            "value_label"
        ):

            card.value_label.setText(
                str(value)
            )

    @staticmethod
    def _format_score(
        value
    ):

        if isinstance(
            value,
            (int, float)
        ):

            return f"{value:.1f}%"

        return "-"

    @staticmethod
    def _extract_domain(
        result
    ):

        if not isinstance(
            result,
            dict
        ):

            return ""

        return (
            result.get(
                "primary_domain"
            )
            or result.get(
                "domain"
            )
            or ""
        )

    @staticmethod
    def _extract_keywords(
        result
    ):

        if not isinstance(
            result,
            dict
        ):

            return []

        keywords = result.get(
            "keywords",
            []
        )

        if isinstance(
            keywords,
            dict
        ):

            keywords = keywords.get(
                "keywords",
                []
            )

        if isinstance(
            keywords,
            str
        ):

            return [keywords]

        if isinstance(
            keywords,
            list
        ):

            return keywords

        return []

    @staticmethod
    def _pretty_key(key):
        text = str(key).replace("_", " ").strip()
        return " ".join(word.capitalize() for word in text.split())

    @staticmethod
    def _format_value(value):
        if value is None or value == "":
            return "—"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value).strip()

    @staticmethod
    def _format_section(data, indent=0):
        """Render research data with consistent, readable labels."""
        if not data:
            return "Data belum tersedia."

        prefix = " " * indent
        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                label = ResearchPage._pretty_key(key)
                if isinstance(value, dict):
                    lines.append(f"{prefix}{label}")
                    lines.extend(ResearchPage._format_section(value, indent + 2).splitlines())
                elif isinstance(value, list):
                    lines.append(f"{prefix}{label}")
                    if value:
                        for item in value:
                            if isinstance(item, dict):
                                nested = ResearchPage._format_section(item, indent + 2)
                                lines.extend(f"{prefix}  • {line.strip()}" for line in nested.splitlines())
                            else:
                                lines.append(f"{prefix}  • {str(item).strip()}")
                    else:
                        lines.append(f"{prefix}  • —")
                else:
                    lines.append(f"{prefix}{label}: {ResearchPage._format_value(value)}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.extend(ResearchPage._format_section(item, indent).splitlines())
                else:
                    lines.append(f"{prefix}• {str(item).strip()}")
        else:
            lines.append(f"{prefix}{str(data).strip()}")

        return "\n".join(line.rstrip() for line in lines if line.strip())

    @staticmethod
    def _format_gap(gaps):
        if not gaps:
            return "No potential research gaps detected."

        if isinstance(gaps, dict):
            sections = []
            summary = gaps.get("summary")
            if summary:
                sections.append("SUMMARY\n" + ResearchPage._format_section(summary))

            gap_items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items")
            if isinstance(gap_items, list):
                items = []
                for item in gap_items:
                    if isinstance(item, dict):
                        title = item.get("title") or item.get("gap") or item.get("description") or "Potential research direction"
                        items.append(f"• {str(title).strip()}")
                    else:
                        items.append(f"• {str(item).strip()}")
                sections.append("POTENTIAL RESEARCH DIRECTIONS\n" + "\n".join(items))
            elif not summary:
                sections.append(ResearchPage._format_section(gaps))

            return "\n\n".join(section for section in sections if section.strip())

        return ResearchPage._format_section(gaps)


class PapersToolPage(QWidget):
    """
    Dedicated Research Tool: Papers.

    Uses the already-generated Academic Research result instead of
    running a second academic search. This keeps the tool lightweight
    and synchronized with the Academic Research workspace.
    """

    def __init__(self, main_window):
        super().__init__()
        self.setObjectName("papersToolPage")
        self.main_window = main_window
        self.papers = []
        self.filtered_papers = []
        self._result_signature = None
        self._sinta_batch_was_running = False
        self._sinta_last_progress_value = -1
        self._sinta_sync_timer = QTimer(self)
        self._sinta_sync_timer.setInterval(500)
        self._sinta_sync_timer.timeout.connect(self._sync_sinta_batch_state)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        self._root_layout = root
        root.setContentsMargins(30, 25, 30, 25)
        root.setSpacing(16)

        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------

        header = QHBoxLayout()

        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)

        title = QLabel("Papers")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Telusuri, filter, dan periksa paper yang ditemukan melalui Academic Research."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header.addLayout(title_layout)
        header.addStretch()

        self.count_label = QLabel("0 paper")
        self.count_label.setObjectName("topbarStatus")
        header.addWidget(self.count_label)

        self.accreditation_status_label = QLabel("Penilaian jurnal otomatis saat proses berjalan")
        self.accreditation_status_label.setObjectName("topbarStatus")
        self.accreditation_status_label.setWordWrap(True)
        header.addWidget(self.accreditation_status_label)

        self.run_research_button = QPushButton("Jalankan Academic Research")
        self.run_research_button.setObjectName("primaryButton")
        self.run_research_button.setMinimumHeight(38)
        self.run_research_button.clicked.connect(self._run_academic_research)
        header.addWidget(self.run_research_button)

        root.addLayout(header)

        self.research_progress = QProgressBar()
        self.research_progress.setRange(0, 0)
        self.research_progress.setMaximumHeight(6)
        self.research_progress.setTextVisible(False)
        self.research_progress.setVisible(False)
        root.addWidget(self.research_progress)

        # -----------------------------------------------------
        # TOOLBAR
        # -----------------------------------------------------

        toolbar = QBoxLayout(QBoxLayout.LeftToRight)
        self._paper_toolbar_layout = toolbar
        toolbar.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Cari judul, penulis, jurnal, ISSN, DOI, kata kunci..."
        )
        self.search_input.setMinimumHeight(38)
        self.search_input.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self.search_input, 1)

        self.sinta_filter = QComboBox()
        self.sinta_filter.setMinimumHeight(38)
        self.sinta_filter.setMinimumWidth(185)
        self.sinta_filter.addItem("Semua Status Jurnal", "ALL")
        for rank in range(1, 7):
            self.sinta_filter.addItem(f"SINTA {rank}", f"S{rank}")
        self.sinta_filter.addItem("Tidak Terakreditasi", "NOT_ACCREDITED")
        self.sinta_filter.addItem("Belum Terverifikasi", "UNVERIFIED")
        self.sinta_filter.addItem("Periode SK perlu dicek", "PERIOD_UNVERIFIED")
        self.sinta_filter.addItem("Kandidat ambigu", "AMBIGUOUS")
        self.sinta_filter.addItem("Jurnal belum cocok", "NO_MATCH")
        self.sinta_filter.addItem("ISSN/jurnal tidak tersedia", "NO_METADATA")
        self.sinta_filter.addItem("Peringkat tidak terbaca", "RANK_UNAVAILABLE")
        self.sinta_filter.addItem("Perlu Dicoba Lagi", "FAILED")
        self.sinta_filter.addItem("Tidak ada hasil resmi", "NOT_FOUND")
        self.sinta_filter.insertSeparator(self.sinta_filter.count())
        self.sinta_filter.addItem("Terindeks DOAJ", "INDEX_DOAJ")
        self.sinta_filter.addItem("Terindeks Scopus", "INDEX_SCOPUS")
        self.sinta_filter.addItem("Indeks lain belum diperiksa", "INDEX_UNVERIFIED")
        self.sinta_filter.addItem("Tidak ditemukan di indeks yang dicek", "INDEX_NOT_FOUND")
        self.sinta_filter.addItem("Pemeriksaan indeks parsial", "INDEX_PARTIAL")
        self.sinta_filter.addItem("Metadata ISSN tidak tersedia", "INDEX_NO_METADATA")
        self.sinta_filter.insertSeparator(self.sinta_filter.count())
        self.sinta_filter.addItem("Akreditasi eksternal ditemukan", "EXT_ACCREDITATION")
        self.sinta_filter.addItem("Peringkat/klasifikasi eksternal", "EXT_RATING")
        self.sinta_filter.addItem("Semua daftar penilaian eksternal", "EXT_ASSESSMENT")
        self.sinta_filter.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(self.sinta_filter)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("secondaryButton")
        self.refresh_button.setMinimumHeight(38)
        self.refresh_button.clicked.connect(self._refresh_from_research)
        toolbar.addWidget(self.refresh_button)

        self.sinta_batch_button = QPushButton("Periksa jurnal belum dicek")
        self.sinta_batch_button.setObjectName("primaryButton")
        self.sinta_batch_button.setMinimumHeight(38)
        self.sinta_batch_button.setToolTip(
            "Periksa jurnal pada hasil Academic Research dengan direktori SINTA."
        )
        self.sinta_batch_button.clicked.connect(self._run_sinta_batch)
        toolbar.addWidget(self.sinta_batch_button)

        self.other_index_button = QPushButton("Periksa Indeks Lain")
        self.other_index_button.setObjectName("secondaryButton")
        self.other_index_button.setMinimumHeight(38)
        self.other_index_button.setToolTip(
            "Cek ISSN otomatis di DOAJ. Scopus dicek bila API Elsevier sudah memiliki akses."
        )
        self.other_index_button.clicked.connect(self._run_other_index_batch)
        toolbar.addWidget(self.other_index_button)

        root.addLayout(toolbar)

        sinta_status_panel = QFrame()
        sinta_status_panel.setObjectName("researchCard")
        sinta_status_layout = QVBoxLayout(sinta_status_panel)
        sinta_status_layout.setContentsMargins(14, 10, 14, 10)
        sinta_status_layout.setSpacing(6)

        self.sinta_batch_status = QLabel(
            "Status SINTA dan indeks lain ditampilkan terpisah. Jurnal yang tidak ada di SINTA bisa terindeks di sumber lain."
        )
        self.sinta_batch_status.setObjectName("cardDescription")
        self.sinta_batch_status.setWordWrap(True)
        sinta_status_layout.addWidget(self.sinta_batch_status)

        self.sinta_scope_note = QLabel(
            "Daftar artikel berasal dari Academic Research. SINTA memeriksa akreditasi jurnal dan tahun berlakunya; "
            "DOI artikel dicocokkan langsung bila tercantum pada profil resmi. "
            "Skor relevansi adalah penyaringan heuristik topik, bukan akurasi model atau bukti penggunaan dataset yang sama. "
            "Status SINTA berbeda dari indeks jurnal. DOAJ diperiksa otomatis; Scopus membutuhkan API Elsevier. "
            "DOAJ adalah direktori, bukan akreditasi. Status WoS dan Garuda belum diperiksa otomatis. "
            "Status dan cakupan jurnal dapat berubah; hasil pemeriksaan adalah snapshot."
        )
        self.sinta_scope_note.setObjectName("cardDescription")
        self.sinta_scope_note.setWordWrap(True)
        sinta_status_layout.addWidget(self.sinta_scope_note)

        self.sinta_batch_progress = QProgressBar()
        self.sinta_batch_progress.setTextVisible(False)
        self.sinta_batch_progress.setMaximumHeight(7)
        self.sinta_batch_progress.setVisible(False)
        sinta_status_layout.addWidget(self.sinta_batch_progress)
        root.addWidget(sinta_status_panel)

        # -----------------------------------------------------
        # PAPER TABLE
        # -----------------------------------------------------

        table_card = QFrame()
        table_card.setObjectName("researchCard")

        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(12, 12, 12, 12)
        table_layout.setSpacing(8)

        self.paper_table = QTableWidget()
        self.paper_table.setObjectName("rankedPapersTable")
        self.paper_table.setColumnCount(9)
        self._paper_headers = [
            "#",
            "Judul Artikel",
            "Tahun",
            "Jurnal / Venue",
            "ISSN",
            "Sumber",
            "Relevansi",
            "SINTA / Indeks",
            "Akreditasi / Penilaian Lain",
        ]
        self._paper_compact_headers = [
            "#", "Artikel", "Thn", "Jurnal", "ISSN", "Sumber", "Skor", "SINTA/Indeks", "Akreditasi lain"
        ]
        self.paper_table.setHorizontalHeaderLabels(self._paper_headers)
        self.paper_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.paper_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.paper_table.setSelectionMode(QTableWidget.SingleSelection)
        self.paper_table.setAlternatingRowColors(True)

        table_header = self.paper_table.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(0, 42)
        table_header.setSectionResizeMode(1, QHeaderView.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(2, 64)
        table_header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(3, 175)
        table_header.setSectionResizeMode(4, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(4, 105)
        table_header.setSectionResizeMode(5, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(5, 110)
        table_header.setSectionResizeMode(6, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(6, 85)
        table_header.setSectionResizeMode(7, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(7, 145)
        table_header.setSectionResizeMode(8, QHeaderView.Fixed)
        self.paper_table.setColumnWidth(8, 210)

        self.paper_table.setMinimumHeight(280)
        self.paper_table.setShowGrid(True)
        self.paper_table.setSortingEnabled(True)
        table_header.setSectionsClickable(True)
        table_header.setSortIndicatorShown(True)
        self.paper_table.verticalHeader().setDefaultSectionSize(38)
        self.paper_table.verticalHeader().setVisible(False)

        self.paper_table.itemSelectionChanged.connect(
            self._show_selected_paper
        )
        self.paper_table.cellDoubleClicked.connect(
            self._open_paper_source_for_row
        )

        table_layout.addWidget(self.paper_table, 1)

        # -----------------------------------------------------
        # DETAIL
        # -----------------------------------------------------

        detail_card = QFrame()
        detail_card.setObjectName("researchCard")

        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(16, 14, 16, 14)
        detail_layout.setSpacing(8)

        detail_title = QLabel("Detail Paper")
        detail_title.setObjectName("sectionTitle")

        self.paper_detail = QTextEdit()
        self.paper_detail.setReadOnly(True)
        self.paper_detail.setMinimumHeight(165)
        self.paper_detail.setMaximumHeight(220)

        self.open_source_button = QPushButton("Buka sumber paper")
        self.open_source_button.setObjectName("secondaryButton")
        self.open_source_button.setEnabled(False)
        self.open_source_button.clicked.connect(self._open_selected_source)

        detail_actions = QHBoxLayout()
        detail_actions.addStretch()
        detail_actions.addWidget(self.open_source_button)

        detail_layout.addWidget(detail_title)
        detail_layout.addWidget(self.paper_detail)
        detail_layout.addLayout(detail_actions)

        paper_splitter = QSplitter(Qt.Vertical)
        paper_splitter.setObjectName("papersTableDetailSplitter")
        paper_splitter.setChildrenCollapsible(False)
        paper_splitter.addWidget(table_card)
        paper_splitter.addWidget(detail_card)
        paper_splitter.setStretchFactor(0, 3)
        paper_splitter.setStretchFactor(1, 1)
        root.addWidget(paper_splitter, 1)

        research_page = getattr(self.main_window, "research_page", None)
        selection_panel = getattr(research_page, "paper_selection_panel", None)
        if selection_panel is not None:
            selection_panel.hide()

        self.show_empty_state()

    def _run_academic_research(self):
        research_page = getattr(self.main_window, "research_page", None)
        if research_page is not None:
            research_page.run_research()

    def set_research_running(self, running):
        self.research_progress.setVisible(bool(running))
        self.run_research_button.setEnabled(not running)
        self.run_research_button.setText(
            "Academic Research sedang berjalan..." if running else "Jalankan Academic Research"
        )

    def set_assessment_lookup_status(self, current, total, finished=False):
        if finished:
            message = (
                f"Penilaian jurnal selesai · {total:,} diperiksa"
                if total else "Tidak ada ISSN jurnal yang dapat diperiksa"
            )
            self.accreditation_status_label.setText(message)
        elif total:
            self.accreditation_status_label.setText(
                f"Memeriksa penilaian jurnal · {current:,}/{total:,}"
            )

    # =====================================================
    # STATE
    # =====================================================

    def show_empty_state(self):
        self._result_signature = None
        self.papers = []
        self.filtered_papers = []
        self.paper_table.setSortingEnabled(False)
        self.paper_table.setRowCount(0)
        self.paper_table.setSortingEnabled(True)
        self.paper_detail.setPlainText(
            "Hasil Academic Research belum tersedia.\n\n"
            "Jalankan Academic Research untuk menampilkan paper di sini."
        )
        self.count_label.setText("0 paper")
        self.accreditation_status_label.setText("Penilaian jurnal otomatis saat proses berjalan")
        self.open_source_button.setEnabled(False)
        self.sinta_batch_button.setEnabled(False)
        self.other_index_button.setEnabled(False)
        self.sinta_batch_progress.setVisible(False)
        self.sinta_batch_status.setText(
            "Jalankan Academic Research terlebih dahulu untuk memeriksa jurnal di SINTA."
        )
        self.set_research_running(False)

    def update_from_result(self, result):
        if not isinstance(result, dict):
            self.show_empty_state()
            return

        papers = result.get("papers", [])
        if not isinstance(papers, list):
            papers = []

        paper_signatures = []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            screening = ResearchPage._sinta_screening(paper)
            paper_signatures.append((
                id(paper),
                str(paper.get("doi") or ""),
                str(screening.get("status") or ""),
                str(screening.get("lookup_status") or ""),
                str(screening.get("rank") or ""),
                str(screening.get("checked_at") or ""),
                str(screening.get("other_indexes", {})),
                str(screening.get("official_assessments_status") or ""),
                str(screening.get("official_assessments", [])),
            ))
        signature = (id(result), tuple(paper_signatures))
        if signature == self._result_signature:
            self._follow_sinta_batch()
            return
        self._result_signature = signature

        self.papers = [
            dict(paper)
            for paper in papers
            if isinstance(paper, dict)
        ]

        self.sinta_batch_button.setEnabled(bool(self.papers))
        self.other_index_button.setEnabled(bool(self.papers))
        self._apply_filter()
        self._follow_sinta_batch()

    def _refresh_from_research(self):
        result = getattr(
            self.main_window.research_page,
            "research_result",
            {},
        )
        self.update_from_result(result)

    def _follow_sinta_batch(self):
        research_page = getattr(self.main_window, "research_page", None)
        if research_page is None or (
            getattr(research_page, "_sinta_thread", None) is None
            and getattr(research_page, "_journal_index_thread", None) is None
        ):
            return
        if not self._sinta_sync_timer.isActive():
            self._sinta_last_progress_value = -1
        self._sinta_batch_was_running = True
        self.sinta_batch_button.setEnabled(False)
        self.other_index_button.setEnabled(False)
        if getattr(research_page, "_journal_index_thread", None) is not None:
            self.other_index_button.setText("Memeriksa indeks...")
        else:
            self.sinta_batch_button.setText("Memeriksa jurnal...")
        self._sinta_sync_timer.start()

    def _run_sinta_batch(self):
        research_page = getattr(self.main_window, "research_page", None)
        if research_page is None or not research_page.research_result:
            self.sinta_batch_status.setText(
                "Jalankan Academic Research terlebih dahulu untuk memuat daftar jurnal."
            )
            return

        if getattr(research_page, "_sinta_thread", None) is not None:
            self._sinta_batch_was_running = True
            self._sinta_last_progress_value = -1
            self._sinta_sync_timer.start()
            self._sync_sinta_batch_state()
            return

        if not getattr(research_page, "_all_papers", []):
            research_page._render_filtered_papers()
        if not getattr(research_page, "_all_papers", []):
            self.sinta_batch_status.setText(
                "Hasil Academic Research belum memuat artikel yang dapat diperiksa."
            )
            return

        self.sinta_batch_status.setText("Menyiapkan pemeriksaan jurnal SINTA...")
        research_page._check_pending_sinta_batch()
        if getattr(research_page, "_sinta_thread", None) is None:
            self.sinta_batch_status.setText(
                research_page.sinta_lookup_message.text()
            )
            self.update_from_result(research_page.research_result)
            return

        self._sinta_batch_was_running = True
        self._sinta_last_progress_value = -1
        self.sinta_batch_button.setEnabled(False)
        self._sinta_sync_timer.start()
        self._sync_sinta_batch_state()

    def _run_other_index_batch(self):
        research_page = getattr(self.main_window, "research_page", None)
        if research_page is None or not research_page.research_result:
            self.sinta_batch_status.setText(
                "Jalankan Academic Research terlebih dahulu untuk memuat daftar jurnal."
            )
            return
        if getattr(research_page, "_journal_index_thread", None) is None:
            research_page._check_pending_other_indexes()
        if getattr(research_page, "_journal_index_thread", None) is None:
            self.sinta_batch_status.setText(research_page.sinta_lookup_message.text())
            self.update_from_result(research_page.research_result)
            return
        self._sinta_batch_was_running = True
        self._sinta_last_progress_value = -1
        self.other_index_button.setEnabled(False)
        self._sinta_sync_timer.start()
        self._sync_sinta_batch_state()

    def _sync_sinta_batch_state(self):
        research_page = getattr(self.main_window, "research_page", None)
        if research_page is None:
            self._sinta_sync_timer.stop()
            return

        sinta_thread = getattr(research_page, "_sinta_thread", None)
        index_thread = getattr(research_page, "_journal_index_thread", None)
        thread = sinta_thread or index_thread
        if thread is not None:
            self._sinta_batch_was_running = True
            is_index_check = index_thread is not None
            self.sinta_batch_button.setEnabled(False)
            self.other_index_button.setEnabled(False)
            self.sinta_batch_button.setText(
                "Memeriksa jurnal..." if not is_index_check else "Periksa jurnal belum dicek"
            )
            self.other_index_button.setText(
                "Memeriksa indeks..." if is_index_check else "Periksa Indeks Lain"
            )
            message = getattr(research_page, "sinta_lookup_message", None)
            if message is not None:
                self.sinta_batch_status.setText(message.text())
            progress = getattr(research_page, "sinta_lookup_progress", None)
            if progress is not None:
                minimum, maximum = progress.minimum(), progress.maximum()
                self.sinta_batch_progress.setRange(minimum, maximum)
                if minimum != maximum:
                    self.sinta_batch_progress.setValue(progress.value())
                self.sinta_batch_progress.setVisible(True)
                if progress.value() != self._sinta_last_progress_value:
                    self._sinta_last_progress_value = progress.value()
                    self.update_from_result(
                        getattr(research_page, "research_result", {})
                    )
            return

        if self._sinta_batch_was_running:
            result = getattr(research_page, "research_result", {})
            self.update_from_result(result)
            message = getattr(research_page, "sinta_lookup_message", None)
            if message is not None and message.text():
                self.sinta_batch_status.setText(message.text())
            self._sinta_batch_was_running = False
            self._sinta_last_progress_value = -1

        self.sinta_batch_button.setText("Periksa jurnal belum dicek")
        self.sinta_batch_button.setEnabled(bool(self.papers))
        self.other_index_button.setText("Periksa Indeks Lain")
        self.other_index_button.setEnabled(bool(self.papers))
        self.sinta_batch_progress.setVisible(False)
        self._sinta_sync_timer.stop()

    # =====================================================
    # FILTER
    # =====================================================

    def _apply_filter(self):
        query = self.search_input.text().strip().lower()
        selected_sinta = self.sinta_filter.currentData() if hasattr(self, "sinta_filter") else "ALL"

        filtered = []
        for paper in self.papers:
            if not ResearchPage._matches_sinta_filter(paper, selected_sinta):
                continue

            authors = paper.get("authors", [])
            authors_text = " ".join(str(a) for a in authors) if isinstance(authors, list) else str(authors)
            keywords = paper.get("keywords", [])
            keywords_text = " ".join(str(k) for k in keywords) if isinstance(keywords, list) else str(keywords)
            issn = paper.get("issn", [])
            issn_text = " ".join(str(i) for i in issn) if isinstance(issn, list) else str(issn)

            searchable = " ".join(
                [
                    str(paper.get("title", "")),
                    authors_text,
                    str(paper.get("venue", "")),
                    str(paper.get("doi", "")),
                    keywords_text,
                    issn_text,
                    str(paper.get("source", "")),
                    str(paper.get("year", "")),
                ]
            ).lower()

            if not query or query in searchable:
                filtered.append(paper)

        self.filtered_papers = filtered
        self._populate_table()

    # =====================================================
    # TABLE
    # =====================================================

    def _populate_table(self):
        self.paper_table.setUpdatesEnabled(False)
        self.paper_table.setSortingEnabled(False)
        self.paper_table.setRowCount(len(self.filtered_papers))

        for row, paper in enumerate(self.filtered_papers):
            title = str(paper.get("title") or "Untitled")
            year = str(paper.get("year") or "-")
            relevance = paper.get("relevance_score")

            venue = str(paper.get("venue") or paper.get("source") or "-")
            venue_display = venue
            if len(venue_display) > 38:
                venue_display = venue_display[:35].rstrip() + "..."

            issn_list = paper.get("issn", [])
            if isinstance(issn_list, str):
                issn_list = [issn_list]
            issn_clean = [str(i).strip() for i in issn_list if str(i).strip()]
            issn_str = issn_clean[0] if issn_clean else "-"

            sources = [
                s.strip()
                for s in str(paper.get("source") or "").split(",")
                if s.strip()
            ]
            source_label = sources[0] if sources else "-"
            if len(sources) > 1:
                source_label += f" +{len(sources) - 1}"

            screening = ResearchPage._sinta_screening(paper)
            status = str(screening.get("status") or "UNVERIFIED")
            sinta_label = ResearchPage._sinta_table_status_label(paper)
            other_assessment_label = ResearchPage._non_sinta_status_label(paper)

            relevance_str = f"{relevance:.1f}%" if isinstance(relevance, (int, float)) else "-"

            values = [
                str(row + 1),
                title,
                year,
                venue_display,
                issn_str,
                source_label,
                relevance_str,
                sinta_label,
                other_assessment_label,
            ]

            for column, value in enumerate(values):
                if column == 0:
                    sort_key = row + 1
                elif column == 2:
                    try:
                        sort_key = float(year)
                    except (TypeError, ValueError):
                        sort_key = float("-inf")
                elif column == 6:
                    try:
                        sort_key = float(relevance)
                    except (TypeError, ValueError):
                        sort_key = float("-inf")
                else:
                    sort_key = value.casefold()
                item = _ResearchPaperTableItem(value, sort_key)
                if column == 0:
                    item.setData(Qt.UserRole, id(paper))

                if column in {0, 2, 4, 5, 6, 7, 8}:
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                if column == 1:
                    item.setToolTip(title)
                elif column == 3:
                    item.setToolTip(venue)
                elif column == 4:
                    item.setToolTip(", ".join(issn_clean) if len(issn_clean) > 1 else (issn_str if issn_str != "-" else "ISSN tidak tersedia"))
                elif column == 5:
                    item.setToolTip(", ".join(sources) or "Sumber tidak tersedia")
                elif column == 7:
                    if status in {f"S{rank}" for rank in range(1, 7)} or screening.get("rank"):
                        item.setForeground(QColor("#0F766E"))
                        item.setBackground(QColor("#F0FDF4"))
                    elif status == "UNVERIFIED":
                        item.setForeground(QColor("#B45309"))
                        item.setBackground(QColor("#FFFBEB"))
                    elif status == "NOT_ACCREDITED":
                        item.setForeground(QColor("#475569"))
                        item.setBackground(QColor("#F8FAFC"))
                    else:
                        item.setForeground(QColor("#64748B"))
                        item.setBackground(QColor("#F1F5F9"))
                    item.setToolTip(_sinta_ui_note(screening) or sinta_label)
                elif column == 8:
                    accreditation = paper.get("accreditation_screening", {})
                    accreditation = accreditation if isinstance(accreditation, dict) else {}
                    assessments = accreditation.get("official_assessments", [])
                    assessments = assessments if isinstance(assessments, list) else []
                    lookup_status = str(accreditation.get("official_assessments_status") or "")
                    if any(
                        isinstance(value, dict) and value.get("kind") == "ACCREDITATION"
                        for value in assessments
                    ):
                        item.setForeground(QColor("#0F766E"))
                        item.setBackground(QColor("#F0FDF4"))
                    elif assessments:
                        item.setForeground(QColor("#1D4ED8"))
                        item.setBackground(QColor("#EFF6FF"))
                    elif lookup_status == "FAILED":
                        item.setForeground(QColor("#B45309"))
                        item.setBackground(QColor("#FFFBEB"))
                    else:
                        item.setForeground(QColor("#64748B"))
                        item.setBackground(QColor("#F1F5F9"))
                    assessment_notes = [
                        f"{value.get('system_name') or value.get('list_id')}"
                        + (f" | edisi {value.get('list_version')}" if value.get("list_version") else "")
                        + (f" | {value.get('source_url')}" if value.get("source_url") else "")
                        for value in assessments if isinstance(value, dict)
                    ]
                    if not assessment_notes:
                        assessment_notes.append(
                            "Tidak ditemukan di daftar eksternal yang terhubung; ini tidak membuktikan jurnal tidak terakreditasi."
                            if lookup_status == "CHECKED"
                            else "Sebagian ISSN belum dapat diperiksa; hasilnya belum lengkap."
                            if lookup_status == "PARTIAL"
                            else "Pencarian daftar penilaian jurnal belum berhasil."
                            if lookup_status == "FAILED"
                            else "Pemeriksaan otomatis berjalan setelah Academic Research."
                        )
                    item.setToolTip("\n".join(assessment_notes))

                self.paper_table.setItem(row, column, item)

        self.paper_table.setSortingEnabled(True)
        self.paper_table.setUpdatesEnabled(True)
        self.count_label.setText(
            f"{len(self.filtered_papers)} dari {len(self.papers)} paper"
        )

        self.paper_detail.clear()
        self.open_source_button.setEnabled(False)

        if self.filtered_papers:
            self.paper_table.selectRow(0)
            self._show_selected_paper()
        else:
            self.paper_detail.setPlainText(
                "Tidak ada paper yang cocok dengan pencarian atau filter saat ini."
            )

    # =====================================================
    # DETAIL
    # =====================================================

    def _show_selected_paper(self):
        rows = self.paper_table.selectionModel().selectedRows()

        if not rows:
            self.paper_detail.clear()
            self.open_source_button.setEnabled(False)
            return

        paper = self._paper_for_row(rows[0].row())
        if paper is None:
            return

        authors = paper.get("authors", [])
        authors_text = ", ".join(str(a) for a in authors) if isinstance(authors, list) else str(authors)

        issn_list = paper.get("issn", [])
        if isinstance(issn_list, str):
            issn_list = [issn_list]
        issn_clean = [str(i).strip() for i in issn_list if str(i).strip()]
        issn_display = ", ".join(issn_clean) if issn_clean else "-"

        screening = paper.get("sinta_screening", {}) if isinstance(paper.get("sinta_screening"), dict) else {}
        status = str(screening.get("status") or "UNVERIFIED")
        labels = {
            "UNVERIFIED": "Belum Terverifikasi",
            "NOT_ACCREDITED": "Tidak Terakreditasi",
            "NOT_IN_SINTA": "Tidak di SINTA",
        }
        sinta_label = f"SINTA {status[1:]}" if status in {f"S{r}" for r in range(1, 7)} else labels.get(status, "Belum Terverifikasi")

        lines = [
            f"JUDUL ARTIKEL\n{paper.get('title') or '-'}",
            f"PENULIS\n{authors_text or '-'}",
            f"TAHUN PUBLIKASI\n{paper.get('year') or '-'}",
            f"JURNAL / VENUE\n{paper.get('venue') or '-'}",
            f"ISSN\n{issn_display}",
            f"DOI\n{paper.get('doi') or '-'}",
            f"SUMBER PENCARIAN\n{paper.get('source') or '-'}",
            (
                "SKOR RELEVANSI\n"
                + (
                    f"{paper.get('relevance_score'):.1f}%"
                    if isinstance(paper.get("relevance_score"), (int, float))
                    else "-"
                )
            ),
            f"STATUS AKREDITASI SINTA\n{sinta_label}",
        ]

        accreditation = paper.get("accreditation_screening", {})
        accreditation = accreditation if isinstance(accreditation, dict) else {}
        assessments = accreditation.get("official_assessments", [])
        assessments = assessments if isinstance(assessments, list) else []
        lines.append("PENILAIAN JURNAL DI LUAR SINTA")
        if assessments:
            for assessment in assessments:
                if not isinstance(assessment, dict):
                    continue
                system = assessment.get("system_name") or assessment.get("list_id") or "Daftar jurnal"
                maintainer = assessment.get("maintainer")
                edition = assessment.get("list_version")
                source_url = assessment.get("source_url")
                kind = str(assessment.get("kind") or "").upper()
                kind_label = {
                    "ACCREDITATION": "Akreditasi",
                    "RATING_OR_CLASSIFICATION": "Peringkat/klasifikasi",
                    "DIRECTORY_OR_INDEX": "Direktori/indeks",
                }.get(kind, "Penilaian eksternal")
                details = [f"{kind_label}: {system}"]
                if maintainer:
                    details.append(f"pengelola: {maintainer}")
                if edition:
                    details.append(f"edisi: {edition}")
                if source_url:
                    details.append(f"sumber: {source_url}")
                lines.append(" · ".join(details))
            lines.append(
                "Kecocokan ISSN berasal dari katalog OpenAlex yang memetakan daftar eksternal terbaru. Validasi status dan masa berlaku untuk tahun artikel pada sumber resmi."
            )
        elif accreditation.get("official_assessments_status") == "CHECKED":
            lines.append(
                "Tidak ditemukan pada daftar penilaian eksternal yang terhubung. Ini bukan bukti bahwa jurnal tidak terakreditasi di semua sistem."
            )
        elif accreditation.get("official_assessments_status") == "FAILED":
            lines.append("Pencarian daftar penilaian eksternal gagal; coba lagi saat koneksi tersedia.")
        elif accreditation.get("official_assessments_status") == "PARTIAL":
            lines.append("Sebagian ISSN belum dapat diperiksa; hasil penilaian belum lengkap.")
        else:
            lines.append("Pemeriksaan otomatis dimulai setelah Academic Research selesai.")

        if screening.get("journal_name"):
            lines.append(f"JURNAL TERDAFTAR DI SINTA\n{screening.get('journal_name')}")
        if screening.get("issn_matched"):
            lines.append(f"ISSN COCOK DI SINTA\n{screening.get('issn_matched')}")
        if screening.get("period_start") and screening.get("period_end"):
            lines.append(f"MASA BERLAKU SK AKREDITASI\n{screening.get('period_start')} - {screening.get('period_end')}")
        external = screening.get("other_indexes", {})
        if isinstance(external, dict) and external.get("checked_at"):
            lines.append("INDEKS JURNAL LAIN")
            for source_name, source_result in (external.get("sources", {}) or {}).items():
                if not isinstance(source_result, dict):
                    continue
                source_status = source_result.get("status")
                source_label = {
                    "INDEXED": "Terindeks",
                    "NOT_INDEXED": "Tidak ditemukan",
                    "NOT_CONFIGURED": "API belum dikonfigurasi",
                    "NO_METADATA": "ISSN tidak tersedia",
                    "FAILED": "Belum dapat diperiksa",
                    "UNAVAILABLE": "Akses tidak tersedia",
                }.get(source_status, "Belum diperiksa")
                lines.append(f"{source_name}\n{source_label}")
            lines.append(
                "DOAJ merupakan direktori jurnal akses terbuka. Status indeks tidak sama dengan akreditasi atau peringkat kuartil."
            )
        display_note = _sinta_ui_note(screening)
        if display_note:
            lines.append(f"CATATAN VERIFIKASI\n{display_note}")

        evidence = paper.get("dataset_usage_evidence", [])
        if evidence:
            lines.append("BUKTI PENGGUNAAN DATASET\n" + "\n".join(f"• {item}" for item in evidence))

        limitations = paper.get("dataset_usage_limitations", [])
        if limitations:
            lines.append("BATASAN PENGGUNAAN DATASET\n" + "\n".join(f"• {item}" for item in limitations))

        abstract = paper.get("abstract")
        if abstract:
            lines.append(f"ABSTRAK\n{abstract}")

        url = paper.get("url")
        doi = paper.get("doi")
        openable_url = url or (f"https://doi.org/{str(doi).strip().removeprefix('doi:')}" if doi else "")
        if openable_url:
            lines.append(f"TAUTAN SUMBER\n{openable_url}")

        self.paper_detail.setPlainText("\n\n".join(lines))
        self.open_source_button.setEnabled(bool(openable_url))

    def _selected_paper(self):
        rows = self.paper_table.selectionModel().selectedRows()

        if not rows:
            return None

        return self._paper_for_row(rows[0].row())

    def _paper_for_row(self, row):
        if row < 0 or row >= self.paper_table.rowCount():
            return None
        item = self.paper_table.item(row, 0)
        paper_identity = item.data(Qt.UserRole) if item is not None else None
        for paper in self.filtered_papers:
            if id(paper) == paper_identity:
                return paper
        return None


    def _open_selected_source(self):
        self._open_paper_source(self._selected_paper())

    def _open_paper_source_for_row(self, row, _column=0):
        self._open_paper_source(self._paper_for_row(row))

    def _open_paper_source(self, paper):
        if not paper:
            return

        target_url = str(paper.get("url") or "").strip()
        doi = str(paper.get("doi") or "").strip()

        if not target_url and doi:
            if doi[:4].lower() == "doi:":
                doi = doi[4:].strip()
            target_url = f"https://doi.org/{doi}"

        if not target_url:
            QMessageBox.information(
                self,
                "Source Unavailable",
                "This paper does not provide a source URL or DOI.",
            )
            return

        try:
            opened = QDesktopServices.openUrl(QUrl(target_url))
            if not opened:
                QMessageBox.warning(
                    self,
                    "Open Source Failed",
                    "Windows tidak dapat membuka tautan sumber artikel ini.",
                )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Open Source Failed",
                f"Unable to open the paper source.\n\n{exc}",
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = event.size().width() < 900
        self._paper_toolbar_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
        )
        self.paper_table.setHorizontalHeaderLabels(
            self._paper_compact_headers if compact else self._paper_headers
        )
        for column, full_label in enumerate(self._paper_headers):
            header_item = self.paper_table.horizontalHeaderItem(column)
            if header_item is not None:
                header_item.setToolTip(full_label)
        compact_widths = (34, 52, 110, 68, 78, 68, 100)
        standard_widths = (42, 64, 175, 105, 110, 85, 145, 210)
        compact_widths = (34, 52, 110, 68, 78, 68, 100, 145)
        for column, width in zip((0, 2, 3, 4, 5, 6, 7, 8), compact_widths if compact else standard_widths):
            self.paper_table.setColumnWidth(column, width)
        self._root_layout.setContentsMargins(
            16 if compact else 30,
            16 if compact else 25,
            16 if compact else 30,
            16 if compact else 25,
        )
        research_page = getattr(self.main_window, "research_page", None)
        if research_page is not None:
            research_page._apply_responsive_layout(event.size().width())


class ResearchLandscapeToolPage(QWidget):
    """Dedicated Research Tool: Research Landscape.

    Reads the already-generated Academic Research landscape result.
    No additional academic search is performed here.
    """

    def __init__(self, main_window):
        super().__init__()
        self.setObjectName("researchLandscapeToolPage")
        self.main_window = main_window
        self._rendered_result_id = None
        self._build_ui()

    def _build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self.page_scroll = QScrollArea(self)
        self.page_scroll.setObjectName("researchLandscapePageScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_layout.addWidget(self.page_scroll)

        page_content = QWidget()
        page_content.setObjectName("researchLandscapePageContent")
        root = QVBoxLayout(page_content)
        self.root_layout = root
        root.setContentsMargins(35, 21, 26, 16)
        root.setSpacing(0)

        header_widget = QWidget()
        self.header_widget = header_widget
        header = QHBoxLayout(header_widget)
        self.header_layout = header
        header.setContentsMargins(0, 0, 0, 0)
        header.setAlignment(Qt.AlignTop)
        title_layout = QVBoxLayout()
        self.title_layout = title_layout
        title = QLabel("Research Landscape")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Jelajahi metode, topik, dan pola publikasi dalam literatur yang dianalisis."
        )
        self.subtitle_label = subtitle
        self.title_label = title
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_layout.setContentsMargins(0, 9, 0, 0)
        title_layout.setSpacing(0)
        title_layout.addWidget(title)
        subtitle.setParent(header_widget)
        header.addLayout(title_layout, 1)
        header.setAlignment(title_layout, Qt.AlignTop)

        self.refresh_button = QPushButton("Perbarui Landscape")
        self.refresh_button.setObjectName("landscapeSearchButton")
        self.refresh_button.setFixedSize(299, 66)
        self.refresh_button.setCursor(Qt.PointingHandCursor)
        refresh_icon = _resource_path("assets/icons/refresh-white.svg")
        if refresh_icon.exists():
            self.refresh_button.setIcon(QIcon(str(refresh_icon)))
            self.refresh_button.setIconSize(QSize(32, 32))
        self.refresh_button.clicked.connect(self.refresh)
        header.addWidget(self.refresh_button, 0, Qt.AlignTop | Qt.AlignRight)
        root.addWidget(header_widget)
        root.addSpacing(4)

        self.status_label = QLabel("Hasil Academic Research belum tersedia.")
        self.status_label.setObjectName("landscapeStatus")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addSpacing(22)

        self.container = QWidget()
        self.container.setObjectName("researchLandscapeContainer")
        self.container.setAttribute(Qt.WA_StyledBackground, True)
        self.content = QVBoxLayout(self.container)
        self.content.setContentsMargins(0, 0, 0, 14)
        self.content.setSpacing(25)
        root.addWidget(self.container, 1)

        self.show_empty_state()
        self.page_scroll.setWidget(page_content)

    def _clear(self):
        while self.content.count():
            item = self.content.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def show_empty_state(self):
        self._clear()
        self._rendered_result_id = None
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 36, 28, 36)
        layout.setSpacing(10)

        title = QLabel("Research Landscape belum tersedia")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        desc = QLabel(
            "Jalankan Academic Research terlebih dahulu.\n"
            "Hasil Research Landscape akan tampil setelah analisis literatur selesai."
        )
        desc.setObjectName("cardDescription")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        title.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(desc)
        self.content.addWidget(card)
        self.content.addStretch()
        self.status_label.setText("Hasil Academic Research belum tersedia.")

    def refresh(self):
        result = getattr(self.main_window.research_page, "research_result", {})
        if not result:
            self.show_empty_state()
            return
        self._rendered_result_id = None
        self.update_from_result(result)

    @staticmethod
    def _safe_dict(value):
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _pick(mapping, *keys, default=None):
        for key in keys:
            value = mapping.get(key)
            if value not in (None, "", [], {}):
                return value
        return default

    @staticmethod
    def _listify(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, dict):
            return list(value.items())
        return [value]

    @staticmethod
    def _pretty(value):
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _metric_card(self, title, value, subtitle=""):
        card = QFrame()
        card.setObjectName("landscapeMetricCard")
        card.setMinimumHeight(154)
        apply_card_shadow(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 22, 18)
        layout.setSpacing(0)

        title_label = QLabel(title.upper())
        title_label.setObjectName("landscapeMetricTitle")
        title_label.setFixedHeight(20)
        title_label.setWordWrap(True)
        value_label = QLabel(self._pretty(value))
        value_label.setObjectName("landscapeMetricValue")
        value_label.setFixedHeight(44)
        value_label.setWordWrap(True)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("landscapeMetricSubtitle")
        subtitle_label.setFixedHeight(24)
        subtitle_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addSpacing(14)
        layout.addWidget(value_label)
        layout.addSpacing(5)
        layout.addWidget(subtitle_label)
        return card

    def _add_text_card(self, title, content):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        body = QLabel(content)
        body.setObjectName("cardDescription")
        body.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(body)
        self.content.addWidget(card)

    def _format_mapping(self, mapping):
        lines = []
        for key, value in mapping.items():
            if isinstance(value, (dict, list, tuple, set)):
                lines.append(f"{key}: {self._pretty(value)}")
            else:
                lines.append(f"{key}: {self._pretty(value)}")
        return "\n".join(lines)

    @staticmethod
    def _method_distribution_rows(value):
        normalized = {}
        for item in ResearchLandscapeToolPage._listify(value):
            if isinstance(item, tuple) and len(item) == 2:
                label, amount = item
            elif isinstance(item, dict):
                label = ResearchLandscapeToolPage._pick(
                    item, "method", "venue", "label", "name", "category", default=""
                )
                amount = ResearchLandscapeToolPage._pick(
                    item, "percentage", "percent", "score", "count", "paper_count", "value"
                )
            else:
                continue
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                continue
            key = re.sub(r"[^a-z0-9]", "", str(label).lower())
            normalized[key] = (str(label), amount)

        preferred = [
            ("Random Forest", ("randomforest",)),
            ("Desicion Tree", ("decisiontree", "desiciontree")),
            ("Logistic Regression", ("logisticregression",)),
            ("SVM", ("svm", "supportvectormachine", "supportvectormachines")),
            ("Gradient Boosting", ("gradientboosting",)),
            ("Linear Regression", ("linearregression",)),
        ]
        rows = []
        consumed = set()
        for display_name, aliases in preferred:
            for alias in aliases:
                if alias in normalized:
                    rows.append((display_name, normalized[alias][1]))
                    consumed.add(alias)
                    break
        for key, (label, amount) in normalized.items():
            if key not in consumed:
                rows.append((label, amount))
        return rows[:10]

    def _add_distribution_card(
        self, title, rows, show_values=True, right_padding=16, value_gap=12
    ):
        card = QFrame()
        card.setObjectName("landscapeDistributionCard")
        card.setFixedHeight(418)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        apply_card_shadow(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, right_padding, 24)
        layout.setSpacing(20)
        heading = QLabel(title)
        heading.setObjectName("landscapeDistributionTitle")
        heading.setFixedHeight(34)
        layout.addWidget(heading)

        if rows:
            maximum = max((amount for _, amount in rows), default=1) or 1
            for label, amount in rows:
                row = QHBoxLayout()
                row.setSpacing(0)
                name = QLabel(label)
                name.setObjectName("landscapeMethodLabel")
                name.setFixedWidth(259)
                name.setWordWrap(True)

                bar = QProgressBar()
                bar.setObjectName("landscapeMethodBar")
                bar.setRange(0, 1000)
                bar.setValue(round(max(0, amount) / maximum * 1000))
                bar.setTextVisible(False)
                bar.setFixedHeight(8)

                row.addWidget(name)
                row.addSpacing(12)
                row.addWidget(bar, 1)
                if show_values:
                    value_label = QLabel(f"{amount:.2f}".rstrip("0").rstrip(".").replace(".", ","))
                    value_label.setObjectName("landscapeMethodValue")
                    value_label.setFixedSize(54, 23)
                    value_label.setAttribute(Qt.WA_StyledBackground, True)
                    value_label.setAlignment(Qt.AlignCenter)
                    row.addSpacing(value_gap)
                    row.addWidget(value_label)
                layout.addLayout(row)
            layout.addStretch(1)
        else:
            empty = QLabel("Distribusi metode belum tersedia.")
            empty.setObjectName("landscapeMethodEmpty")
            layout.addWidget(empty)
        return card

    def update_from_result(self, result):
        if not isinstance(result, dict) or not result:
            self.show_empty_state()
            return
        if id(result) == self._rendered_result_id:
            return
        self._clear()

        landscape = self._safe_dict(result.get("landscape"))
        summary = self._safe_dict(landscape.get("summary"))
        papers = result.get("papers", [])
        if not isinstance(papers, list):
            papers = []

        paper_count = self._pick(
            summary,
            "paper_count", "papers_analyzed", "total_papers", "papers_found",
            default=len(papers),
        )
        methods = self._pick(
            landscape,
            "method_distribution", "methods", "method_counts", "top_methods", "dominant_methods",
            default={},
        )
        method_rows = self._method_distribution_rows(methods)
        venues = self._pick(
            landscape,
            "venue_distribution", "publication_venues", "top_venues", "venues",
            default={},
        )
        venue_rows = self._method_distribution_rows(venues)
        dominant_method = self._pick(
            summary,
            "dominant_method", "top_method", "most_common_method",
            default=(max(method_rows, key=lambda row: row[1])[0] if method_rows else "-"),
        )
        latest_year = self._pick(
            summary,
            "latest_publication_year", "latest_year", "max_year",
            default=max(
                (str(paper.get("year")) for paper in papers if paper.get("year")),
                default="-",
            ),
        )

        self.status_label.setText(
            f"Landscape dimuat dari Academic Research | {paper_count} paper dianalisis"
        )

        cards_widget = QWidget()
        cards_widget.setObjectName("landscapeMetricsRow")
        cards = QHBoxLayout(cards_widget)
        self._metrics_layout = cards
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(14)
        cards.addWidget(self._metric_card("Papers", paper_count, "Literatur dianalisis"))
        cards.addWidget(self._metric_card("Dominan Method", dominant_method, "Most represented method"))
        cards.addWidget(self._metric_card("Latest Publication", latest_year, "Newest publication year"))
        cards.setStretch(0, 445)
        cards.setStretch(1, 477)
        cards.setStretch(2, 489)
        self.content.addWidget(cards_widget)

        distribution_widget = QWidget()
        distribution_widget.setObjectName("landscapeDistributionRow")
        distribution_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        distribution = QHBoxLayout(distribution_widget)
        distribution.setContentsMargins(0, 0, 0, 0)
        distribution.setSpacing(18)
        distribution.addWidget(
            self._add_distribution_card(
                "Method Distribution", method_rows, show_values=True
            )
        )
        distribution.addWidget(
            self._add_distribution_card(
                "Publication Venues",
                venue_rows,
                show_values=True,
                right_padding=30,
                value_gap=28,
            )
        )
        distribution.setStretch(0, 805)
        distribution.setStretch(1, 616)
        self._distribution_layout = distribution
        self._distribution_widget = distribution_widget
        self._distribution_area = distribution_widget
        self.content.addWidget(distribution_widget)
        self.content.addStretch()
        self._rendered_result_id = id(result)
        self._apply_responsive_layout(self.width())

    def _apply_responsive_layout(self, width):
        compact = logical_width(width) < 760
        self.title_label.setFixedHeight(self.title_label.sizeHint().height())
        header = getattr(self, "header_layout", None)
        if header is not None:
            header.setDirection(
                QBoxLayout.TopToBottom
                if compact
                else QBoxLayout.LeftToRight
            )
            header.setSpacing(14 if compact else 6)
        self.subtitle_label.setMinimumWidth(0)
        self.subtitle_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Fixed
        )
        content_width = max(
            1,
            width - ui_px(32 if compact else 163),
        )
        self.subtitle_label.setWordWrap(logical_width(content_width) < 1000)
        subtitle_height = (
            self.subtitle_label.heightForWidth(content_width)
            if logical_width(content_width) < 1000
            else ui_px(25)
        )
        if subtitle_height > 0:
            self.subtitle_label.setFixedHeight(subtitle_height)
        if compact:
            subtitle_y = (
                self.title_layout.sizeHint().height()
                + self.header_layout.spacing()
                + self.refresh_button.height()
                + self.header_layout.spacing()
            )
        else:
            subtitle_y = ui_px(53)
        self.subtitle_label.setGeometry(
            0,
            subtitle_y,
            min(content_width, self.header_widget.width()),
            max(1, subtitle_height),
        )
        self.header_widget.setMinimumHeight(
            max(self.header_layout.sizeHint().height(), subtitle_y + subtitle_height)
        )
        self.root_layout.setContentsMargins(
            ui_px(16 if compact else 36),
            ui_px(16 if compact else 21),
            ui_px(16 if compact else 26),
            ui_px(12 if compact else 16),
        )
        metrics = getattr(self, "_metrics_layout", None)
        if metrics is not None:
            metrics.setDirection(
                QBoxLayout.TopToBottom
                if logical_width(width) < 820
                else QBoxLayout.LeftToRight
            )
        distribution = getattr(self, "_distribution_layout", None)
        if distribution is None:
            return
        distribution.setDirection(
            QBoxLayout.TopToBottom
            if logical_width(width) < 1100
            else QBoxLayout.LeftToRight
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout(self.width())



class ResearchGapToolPage(QWidget):
    """Explain candidate research gaps and recommend practical next steps."""

    def __init__(self, main_window):
        super().__init__()
        self.setObjectName("researchGapToolPage")
        self.main_window = main_window
        self.research_result: Dict[str, Any] = {}
        self._explanation_thread: QThread | None = None
        self._explanation_worker: ResearchGapExplanationWorker | None = None
        self._attempted_explanation_this_session = False
        self._explanation_result_identity = None
        self._rendered_result_signature = None
        self._build_ui()

    def _build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self.page_scroll = QScrollArea(self)
        self.page_scroll.setObjectName("researchGapPageScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_layout.addWidget(self.page_scroll)

        page_content = QWidget()
        page_content.setObjectName("researchGapPageContent")
        root = QVBoxLayout(page_content)
        self.root_layout = root
        root.setContentsMargins(30, 25, 30, 25)
        root.setSpacing(14)
        header_widget = QWidget()
        self.header_widget = header_widget
        header = QHBoxLayout(header_widget)
        self.header_layout = header
        header.setContentsMargins(0, 0, 0, 0)
        title_box = QVBoxLayout()
        title = QLabel("Research Gap")
        self.title_label = title
        title.setObjectName("pageTitle")
        subtitle = QLabel("Pahami celah riset yang terindikasi dan tentukan langkah penelitian berikutnya.")
        self.subtitle_label = subtitle
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setMinimumWidth(0)
        subtitle.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.explain_button = QPushButton("Buat penjelasan")
        self.explain_button.setObjectName("primaryButton")
        self.explain_button.setMinimumHeight(40)
        self.explain_button.clicked.connect(self.explain_gaps)
        header.addWidget(self.explain_button, 0, Qt.AlignTop)
        root.addWidget(header_widget)
        self.status = QLabel("Jalankan Academic Research untuk melihat kandidat celah riset.")
        self.status.setObjectName("statusLabel")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.container = QWidget()
        self.content = QVBoxLayout(self.container)
        self.content.setContentsMargins(0, 4, 8, 16)
        self.content.setSpacing(14)
        root.addWidget(self.container, 1)
        self.show_empty_state()
        self.page_scroll.setWidget(page_content)

    def _apply_responsive_layout(self, width):
        compact = logical_width(width) < 760
        self.root_layout.setContentsMargins(
            16 if compact else 30,
            16 if compact else 25,
            16 if compact else 30,
            16 if compact else 25,
        )
        self.title_label.setFixedHeight(self.title_label.sizeHint().height())
        self.subtitle_label.setMinimumWidth(0)
        self.subtitle_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Fixed
        )
        text_width = max(1, width - (32 if compact else 64))
        if not compact:
            text_width = max(
                1, text_width - self.explain_button.sizeHint().width() - 14
            )
        subtitle_height = self.subtitle_label.heightForWidth(text_width)
        if subtitle_height > 0:
            self.subtitle_label.setFixedHeight(subtitle_height)
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
        )
        self.header_layout.setSpacing(14 if compact else 6)
        self.header_layout.setAlignment(
            self.explain_button,
            Qt.AlignLeft if compact else Qt.AlignTop | Qt.AlignRight,
        )
        if compact:
            self.header_widget.setMinimumHeight(
                self.title_label.minimumHeight()
                + self.subtitle_label.minimumHeight()
                + self.explain_button.minimumHeight()
                + 34
            )
        else:
            self.header_widget.setMinimumHeight(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout(self.width())

    def clear_content(self):
        self._clear_layout_items(self.content)

    @staticmethod
    def _clear_layout_items(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.deleteLater()
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    ResearchGapToolPage._clear_layout_items(child_layout)
                    child_layout.deleteLater()

    def show_empty_state(self):
        self.research_result = {}
        self._attempted_explanation_this_session = False
        self._explanation_result_identity = None
        self._rendered_result_signature = None
        self.status.setText("Jalankan Academic Research terlebih dahulu untuk menemukan kandidat gap.")
        self.explain_button.setEnabled(False)
        self.clear_content()
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(30, 40, 30, 40)
        title = QLabel("Belum ada hasil Research Gap")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        description = QLabel("Setelah Academic Research menemukan paper, kandidat gap dan rekomendasi penelitian akan muncul di sini.")
        description.setObjectName("cardDescription")
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(description)
        self.content.addWidget(card)
        self.content.addStretch()

    def refresh(self):
        result = getattr(self.main_window.research_page, "research_result", {})
        if result:
            self.update_from_result(result)
        else:
            self.show_empty_state()

    def update_from_result(self, result):
        incoming_result = result if isinstance(result, dict) else {}
        result_identity = id(incoming_result)
        if result_identity != self._explanation_result_identity:
            self._attempted_explanation_this_session = False
            self._explanation_result_identity = result_identity
        self.research_result = incoming_result
        gaps = self.research_result.get("gaps", {})
        explanation = self.research_result.get("gap_explanation")
        signature = (
            id(self.research_result),
            id(gaps),
            id(explanation),
            self._explanation_thread is not None,
        )
        if signature == self._rendered_result_signature:
            return
        self._rendered_result_signature = signature
        summary = gaps.get("summary", {}) if isinstance(gaps, dict) else {}
        items = []
        if isinstance(gaps, dict):
            items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items") or []
        elif isinstance(gaps, list):
            items = gaps
        if not isinstance(items, list):
            items = []
        insights_ready = self.research_result.get("insights_status") == "READY"
        self.explain_button.setEnabled(
            bool(self.research_result) and insights_ready and self._explanation_thread is None
        )
        self.status.setText(f"{len(self.research_result.get('papers', []))} paper dianalisis; {len(items)} kandidat gap terdeteksi.")
        self.clear_content()

        metrics = QHBoxLayout()
        gap_count = summary.get("gap_count", len(items)) if isinstance(summary, dict) else len(items)
        metrics.addWidget(InfoCard("Kandidat gap", str(gap_count), "Perlu ditinjau, bukan klaim pasti"))
        metrics.addWidget(InfoCard("Paper dianalisis", str(len(self.research_result.get("papers", []))), "Literatur yang ditemukan"))
        explanation = self.research_result.get("gap_explanation") or {}
        explanation_source = explanation.get("source", "") if isinstance(explanation, dict) else ""
        explanation_status = (
            "Gemini" if explanation_source == "gemini"
            else ("Gagal" if explanation_source == "error" else ("Lokal" if explanation else "Belum dibuat"))
        )
        metrics.addWidget(InfoCard("Penjelasan", explanation_status, "Gemini opsional; otomatis beralih ke analisis lokal"))
        self.content.addLayout(metrics)

        if self._explanation_thread is not None:
            self.content.addWidget(LoadingCard(
                "Menyusun penjelasan Research Gap",
                "Menganalisis bukti dan batas hasil. Jika Gemini tidak tersedia atau kuotanya habis, penjelasan lokal tetap dibuat.",
                self.container,
            ))
        elif isinstance(explanation, dict) and explanation.get("text"):
            self._add_explanation_card(
                explanation.get("text", ""), explanation_source,
                explanation.get("reason", ""),
            )
        else:
            self._add_explanation_card(
                "Penjelasan akan disusun otomatis setelah analisis landscape dan kandidat gap selesai. Ringkasan ini memakai paper yang ditemukan dan tetap perlu divalidasi melalui penelusuran pustaka lanjutan.",
                "menunggu",
            )

        heading = QLabel("Kandidat yang ditemukan dari literatur")
        heading.setObjectName("sectionTitle")
        self.content.addWidget(heading)
        if not items:
            gap_status = str(gaps.get("status") or "").upper() if isinstance(gaps, dict) else ""
            if gap_status == "ERROR":
                empty_text = "Analisis kandidat gap belum selesai. Lihat penjelasan lokal untuk status dan langkah pemeriksaannya."
            elif gap_status == "SKIPPED":
                empty_text = "Analisis kandidat gap belum dijalankan. Buka tab Research Gap untuk menghitungnya."
            elif self.research_result.get("papers"):
                empty_text = "Tidak ada sinyal gap heuristik yang cukup kuat pada paper yang dianalisis. Ini belum membuktikan bahwa gap tidak ada; batas pencarian dijelaskan di atas."
            else:
                empty_text = "Belum ada paper yang dapat dibandingkan. Perluas kata kunci atau jalankan Academic Research terlebih dahulu."
            empty = QLabel(empty_text)
            empty.setObjectName("emptyState")
            empty.setWordWrap(True)
            self.content.addWidget(empty)
        else:
            for index, item in enumerate(items, 1):
                self.content.addWidget(self._gap_card(index, item))

        note = QLabel("Kandidat gap merupakan indikasi dari literatur yang berhasil ditemukan. Periksa paper terbaru dan validasi kebaruan sebelum menetapkan topik penelitian.")
        note.setObjectName("cardDescription")
        note.setWordWrap(True)
        self.content.addWidget(note)
        self.content.addStretch()

    def _add_explanation_card(self, text, source, reason=""):

        card = QFrame()
        card.setObjectName("contentCard")
        card.setProperty("researchGapExplanation", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        eyebrow = QLabel("RINGKASAN BERDASARKAN LITERATUR YANG TERSEDIA")
        eyebrow.setObjectName("gapExplanationEyebrow")
        title = QLabel("Penjelasan dan saran penelitian")
        title.setObjectName("sectionTitle")
        source_names = {
            "gemini": "Gemini menggunakan API key tersimpan",
            "local": "Penjelasan lokal",
            "error": "Penjelasan belum berhasil dibuat",
            "menunggu": "Belum dibuat",
        }
        source_label = QLabel(f"Sumber: {source_names.get(source, source)}")
        source_label.setObjectName("statusLabel")
        if reason:
            reason_label = QLabel(reason)
            reason_label.setObjectName("errorState" if source == "error" else "cardDescription")
            reason_label.setWordWrap(True)
        markdown_document = QTextDocument()
        markdown_document.setDefaultFont(self.font())
        markdown_document.setMarkdown(str(text))
        body = QLabel()
        body.setObjectName("gapExplanationBody")
        body.setTextFormat(Qt.RichText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setText(markdown_document.toHtml())
        body.setMinimumHeight(60)
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(source_label)
        if reason:
            layout.addWidget(reason_label)
        layout.addWidget(body)
        self.content.addWidget(card)

    @staticmethod
    def _gap_card(index, item):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        if isinstance(item, dict):
            title = item.get("title") or item.get("gap") or "Kandidat celah riset"
            description = item.get("description") or ""
            evidence = item.get("evidence")
        else:
            title, description, evidence = str(item), "", None
        heading = QLabel(f"{index:02d}  {title}")
        heading.setObjectName("sectionTitle")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        if description and description != title:
            detail = QLabel(description)
            detail.setObjectName("cardDescription")
            detail.setWordWrap(True)
            layout.addWidget(detail)
        if evidence:
            evidence_label = QLabel("Bukti ringkas: " + ResearchPage._format_value(evidence))
            evidence_label.setObjectName("cardDescription")
            evidence_label.setWordWrap(True)
            layout.addWidget(evidence_label)
        return card

    def ensure_explanation(self):
        # Do not explain the pre-insights result: the landscape/gap worker must
        # finish first so this request receives the completed evidence set.
        if self.research_result.get("insights_status") != "READY":
            return
        explanation = self.research_result.get("gap_explanation", {})
        needs_generation = not isinstance(explanation, dict) or not explanation.get("text")
        needs_generation = needs_generation or explanation.get("source") == "error"
        if self.research_result and needs_generation and not self._attempted_explanation_this_session:
            self.explain_gaps()

    def explain_gaps(self):
        if (
            not self.research_result
            or self.research_result.get("insights_status") != "READY"
            or self._explanation_thread is not None
        ):
            return
        self._attempted_explanation_this_session = True
        self.explain_button.setEnabled(False)
        self._explanation_thread = QThread(self)
        self._explanation_worker = ResearchGapExplanationWorker(self.research_result)
        self._explanation_worker.moveToThread(self._explanation_thread)
        self._explanation_thread.started.connect(self._explanation_worker.run)
        self._explanation_worker.finished.connect(self._on_explanation_finished)
        self._explanation_worker.failed.connect(self._on_explanation_failed)
        self._explanation_worker.finished.connect(self._explanation_thread.quit)
        self._explanation_worker.failed.connect(self._explanation_thread.quit)
        self._explanation_thread.finished.connect(self._explanation_worker.deleteLater)
        self._explanation_thread.finished.connect(self._cleanup_explanation_worker)
        self.update_from_result(self.research_result)
        self._explanation_thread.start()

    def _on_explanation_finished(self, text, source, reason):
        explanation = {"text": text, "source": source, "reason": reason}
        self.research_result["gap_explanation"] = explanation
        project_id = getattr(self.main_window, "current_project_id", None)
        if project_id is not None:
            try:
                self.main_window.repository.save_workflow_section(project_id, "research", self.research_result)
            except Exception as error:
                print(f"Warning: gap explanation could not be saved: {error}")
        self.update_from_result(self.research_result)

    def _on_explanation_failed(self, error):
        self.research_result["gap_explanation"] = {
            "text": f"Penjelasan AI gagal dibuat: {error}. Periksa koneksi internet dan konfigurasi Gemini, lalu coba lagi.",
            "source": "error",
            "reason": "Periksa koneksi internet dan konfigurasi Gemini, lalu coba lagi.",
        }
        self.update_from_result(self.research_result)

    def _cleanup_explanation_worker(self):
        self._explanation_worker = None
        self._explanation_thread = None
        self.explain_button.setEnabled(
            bool(self.research_result)
            and self.research_result.get("insights_status") == "READY"
        )
        if self.research_result:
            self.update_from_result(self.research_result)



class ResearchReportToolPage(QWidget):
    """Preview and direct PDF export for a complete research workflow."""

    def __init__(self, main_window):
        super().__init__()
        self.setObjectName("researchReportPage")
        self.main_window = main_window
        self.report_generator = ReportGenerator()
        self.research_result: Dict[str, Any] = {}
        self.current_html = ""
        self.last_exported_path = None
        self._chart_tempdir = None
        self.chart_dir = None
        self._report_thread = None
        self._report_worker = None
        self._report_operation = None
        self._pending_pdf_path = None
        self._report_project_id = None
        self._report_dataset_title = None
        self._preview_signature = None
        self._generating_report_signature = None
        self._workflow_source_objects = ()
        self.build_ui()

    def build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self.page_scroll = QScrollArea(self)
        self.page_scroll.setObjectName("researchReportPageScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_layout.addWidget(self.page_scroll)

        content = QWidget()
        content.setObjectName("researchReportContent")
        root = QVBoxLayout(content)
        self._content_layout = root
        root.setContentsMargins(30, 25, 30, 25)
        root.setSpacing(16)

        header_widget = QWidget()
        self.header_widget = header_widget
        header = QHBoxLayout(header_widget)
        self.header_layout = header
        header.setContentsMargins(0, 0, 0, 0)
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Laporan Penelitian")
        self.title_label = title
        title.setObjectName("pageTitle")
        subtitle = QLabel("Laporan gabungan analisis dataset, evaluasi machine learning, dan telaah literatur akademik.")
        self.subtitle_label = subtitle
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setMinimumWidth(0)
        subtitle.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        box.addWidget(title)
        box.addWidget(subtitle)
        header.addLayout(box, 1)

        self.header_actions = QHBoxLayout()
        self.header_actions.setSpacing(10)
        self.generate_button = QPushButton("Perbarui Pratinjau")
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(
            lambda _checked=False: self.generate_preview(force=True)
        )
        self.header_actions.addWidget(self.generate_button)

        self.export_button = QPushButton("Simpan sebagai PDF")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.export_pdf)
        self.export_button.setEnabled(False)
        self.header_actions.addWidget(self.export_button)
        header.addLayout(self.header_actions)
        root.addWidget(header_widget)

        self.status = QLabel("Dataset atau analisis belum dimuat.")
        self.status.setObjectName("statusLabel")
        root.addWidget(self.status)

        self.loading_card = LoadingCard(
            "Menyusun laporan",
            "Menyiapkan tata letak dan grafik dari hasil analisis.",
            self,
        )
        self.loading_card.setVisible(False)
        root.addWidget(self.loading_card)

        self.preview = QTextBrowser()
        self.preview.setObjectName("researchReportPreview")
        self.preview.setReadOnly(True)
        self.preview.setOpenLinks(False)
        self.preview.setOpenExternalLinks(False)
        root.addWidget(self.preview, 1)
        self.page_scroll.setWidget(content)
        self.show_empty_state()

    def _apply_responsive_layout(self, width):
        compact = logical_width(width) < 760
        self._content_layout.setContentsMargins(
            16 if compact else 30,
            16 if compact else 25,
            16 if compact else 30,
            16 if compact else 25,
        )
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
        )
        self.header_actions.setDirection(
            QBoxLayout.TopToBottom if logical_width(width) < 480 else QBoxLayout.LeftToRight
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout(self.width())

    def show_empty_state(self):
        self.research_result = {}
        self.current_html = ""
        self._preview_signature = None
        self._generating_report_signature = None
        self.status.setText("Jalankan semua tahap penelitian sebelum mengekspor laporan.")
        self.preview.setPlainText(
            "Laporan penelitian belum tersedia.\n\n"
            "Muat dataset, lalu selesaikan Analisis Dataset, ML Intelligence, dan Academic Research."
        )
        self.export_button.setEnabled(False)

    def update_from_result(self, result):
        self.research_result = result or {}
        self.generate_preview()

    def refresh_from_main(self):
        self.research_result = getattr(self.main_window.research_page, "research_result", {}) or {}

    def _workflow_signature(self):
        """Identify the exact in-memory workflow state used by the report."""
        main = self.main_window
        research = (
            getattr(main.research_page, "research_result", {})
            or self.research_result
            or {}
        )
        ml_page = main.ml_page
        papers = research.get("papers", []) if isinstance(research, dict) else []
        paper_signatures = []
        for paper in papers if isinstance(papers, list) else []:
            if not isinstance(paper, dict):
                continue
            screening = ResearchPage._sinta_screening(paper)
            authors = paper.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            elif not isinstance(authors, (list, tuple)):
                authors = [authors]
            paper_signatures.append((
                str(paper.get("doi") or ""),
                str(paper.get("title") or ""),
                str(paper.get("year") or ""),
                str(paper.get("venue") or ""),
                str(paper.get("abstract") or ""),
                str(paper.get("issn") or ""),
                tuple(str(author) for author in authors),
                str(screening.get("status") or ""),
                str(screening.get("rank") or ""),
                str(screening.get("checked_at") or ""),
            ))
        explanation = research.get("gap_explanation") or {}
        if isinstance(explanation, dict):
            explanation_signature = (
                str(explanation.get("source") or ""),
                str(explanation.get("text") or ""),
            )
        else:
            explanation_signature = ("", str(explanation or ""))
        dataset = getattr(main, "current_dataset", None)
        analysis = getattr(main, "analysis_result", None)
        task_result = getattr(ml_page, "last_task_result", None)
        method_result = getattr(ml_page, "last_method_result", None)
        ml_result = getattr(ml_page, "last_intelligence_result", None)
        sources = (
            dataset,
            analysis,
            task_result,
            method_result,
            ml_result,
            research,
            research.get("papers") if isinstance(research, dict) else None,
            research.get("landscape") if isinstance(research, dict) else None,
            research.get("trend") if isinstance(research, dict) else None,
            research.get("gaps") if isinstance(research, dict) else None,
        )
        signature = (
            getattr(main, "current_project_id", None),
            id(dataset),
            main._current_filename(),
            id(analysis),
            id(task_result),
            id(method_result),
            id(ml_result),
            id(research),
            tuple(paper_signatures),
            explanation_signature,
        )
        # Hold references so Python cannot recycle an old result object's id
        # and accidentally treat a new workflow as the cached report.
        self._workflow_source_objects = sources
        return signature

    def _workflow_missing(self):
        missing = []
        if getattr(self.main_window, "current_dataset", None) is None:
            missing.append("Dataset belum dimuat")
        if not getattr(self.main_window, "analysis_result", {}):
            missing.append("Analisis Dataset belum dijalankan")
        ml_page = self.main_window.ml_page
        ml_result = getattr(ml_page, "last_intelligence_result", {}) or {}
        if not (getattr(ml_page, "last_task_result", {}) and getattr(ml_page, "last_method_result", {}) and ml_result and ml_result.get("status") != "ERROR"):
            missing.append("ML Intelligence belum selesai")
        research = getattr(self.main_window.research_page, "research_result", {}) or self.research_result
        if not research or research.get("status") not in {"SUCCESS", "PARTIAL"}:
            missing.append("Academic Research belum selesai")
            return missing
        gaps = research.get("gaps") or {}
        if isinstance(gaps, dict):
            gap_items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items") or []
        else:
            gap_items = gaps if isinstance(gaps, list) else []
        explanation = research.get("gap_explanation") or {}
        if gap_items and not (isinstance(explanation, dict) and explanation.get("text")):
            missing.append("Penjelasan Research Gap belum dibuat")
        return missing

    def _release_chart_tempdir(self):
        tempdir = self._chart_tempdir
        self._chart_tempdir = None
        self.chart_dir = None
        if tempdir is not None:
            tempdir.cleanup()

    def generate_preview(self, force=False):
        if self._report_thread is not None:
            return
        if not self.research_result:
            self.refresh_from_main()

        filename = self.main_window._current_filename()
        has_dataset = self.main_window.current_dataset is not None
        analysis_data = getattr(self.main_window, "analysis_result", {}) or {}
        ml_data = getattr(self.main_window.ml_page, "last_intelligence_result", {}) or {}
        research_data = self.research_result or getattr(self.main_window.research_page, "research_result", {}) or {}

        if not has_dataset and not analysis_data and not ml_data and not research_data:
            self.show_empty_state()
            return

        missing = self._workflow_missing()
        if missing:
            self._release_chart_tempdir()
            self.current_html = ""
            self._preview_signature = None
            self._generating_report_signature = None
            self.status.setText("Selesaikan semua tahap penelitian untuk membuka ekspor PDF.")
            self.preview.setPlainText("Laporan belum siap. Selesaikan tahap berikut:\n\n" + "\n".join(f"- {item}" for item in missing))
            self.export_button.setEnabled(False)
            return

        workflow_signature = self._workflow_signature()
        if (
            not force
            and self.current_html
            and self._preview_signature == workflow_signature
        ):
            self.export_button.setEnabled(True)
            return

        try:
            dataset_title = Path(filename).stem or filename
            self._generating_report_signature = workflow_signature
            self._report_project_id = getattr(self.main_window, "current_project_id", None)
            self._report_dataset_title = dataset_title
            self._release_chart_tempdir()
            self._chart_tempdir = tempfile.TemporaryDirectory(prefix="dataset_research_report_")
            self.chart_dir = Path(self._chart_tempdir.name)
            self.current_html = ""
            self.preview.setPlainText("Laporan sedang disusun. Pratinjau akan tampil setelah grafik selesai dibuat.")
            self.generate_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.loading_card.title_label.setText("Menyusun laporan dan grafik")
            self.loading_card.description_label.setText(
                "Aplikasi sedang membuat histogram, korelasi, dan grafik ringkasan pada background."
            )
            self.loading_card.setVisible(True)
            worker = ReportGenerationWorker(
                dataset_title, analysis_data, ml_data, research_data,
                self.main_window.current_dataset, str(self.chart_dir),
            )
            self._start_report_worker(worker, "preview")
        except Exception as exc:
            self._finish_report_busy()
            self.status.setText(f"Gagal membuat laporan: {exc}")
            QMessageBox.warning(self, "Gagal membuat laporan", f"Terjadi kesalahan saat membuat laporan: {exc}")

    def _start_report_worker(self, worker, operation):
        self._report_operation = operation
        self._report_thread = QThread(self)
        self._report_worker = worker
        worker.moveToThread(self._report_thread)
        self._report_thread.started.connect(worker.run)
        if operation == "preview":
            worker.finished.connect(self._on_report_generated)
        else:
            worker.finished.connect(self._on_pdf_exported)
        worker.failed.connect(self._on_report_worker_failed)
        worker.finished.connect(self._report_thread.quit)
        worker.failed.connect(self._report_thread.quit)
        self._report_thread.finished.connect(worker.deleteLater)
        self._report_thread.finished.connect(self._cleanup_report_worker)
        self._report_thread.start()

    def _on_report_generated(self, content, chart_dir):
        current_project_id = getattr(self.main_window, "current_project_id", None)
        current_dataset_title = Path(self.main_window._current_filename()).stem
        if (
            current_project_id != self._report_project_id
            or current_dataset_title != self._report_dataset_title
            or self._generating_report_signature != self._workflow_signature()
        ):
            self.current_html = ""
            self._preview_signature = None
            self._generating_report_signature = None
            self._release_chart_tempdir()
            self.status.setText("Dataset berubah saat laporan disusun. Perbarui pratinjau untuk dataset aktif.")
            self._finish_report_busy()
            self.export_button.setEnabled(False)
            return
        self.current_html = content
        self._preview_signature = self._generating_report_signature
        self._generating_report_signature = None
        self.chart_dir = Path(chart_dir)
        self.preview.setSearchPaths([str(self.chart_dir)])
        self.preview.setHtml(self.current_html)
        self.status.setText(
            f"Pratinjau lengkap untuk dataset '{self._report_dataset_title}'."
        )
        self._finish_report_busy()
        self.export_button.setEnabled(bool(self.current_html) and not self._workflow_missing())

    def _on_pdf_exported(self, output_path):
        self.last_exported_path = Path(output_path)
        self.status.setText(f"PDF berhasil disimpan: {output_path}")
        self._finish_report_busy()
        self.export_button.setEnabled(bool(self.current_html) and not self._workflow_missing())
        QMessageBox.information(self, "PDF Tersimpan", f"Laporan berhasil disimpan ke:\n{output_path}")

    def _on_report_worker_failed(self, error):
        operation = self._report_operation
        if operation == "preview":
            self._preview_signature = None
            self._generating_report_signature = None
            self._release_chart_tempdir()
        self._finish_report_busy()
        self.export_button.setEnabled(bool(self.current_html) and not self._workflow_missing())
        self.status.setText("Gagal membuat pratinjau laporan." if operation == "preview" else "Gagal menyimpan PDF.")
        QMessageBox.critical(self, "Proses Laporan Gagal", str(error))

    def _finish_report_busy(self):
        self.loading_card.setVisible(False)
        self.generate_button.setEnabled(True)

    def _cleanup_report_worker(self):
        self._report_worker = None
        self._report_thread = None
        self._report_operation = None

    def export_pdf(self):
        if self._report_thread is not None:
            return
        if not self.current_html or self._workflow_missing():
            self.generate_preview()
        if not self.current_html:
            return

        filename = Path(self.main_window._current_filename()).stem or "dataset"
        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Laporan PDF", f"Laporan_Riset_{filename}.pdf", "Dokumen PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            self._pending_pdf_path = path
            self.generate_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.loading_card.title_label.setText("Menyimpan PDF")
            self.loading_card.description_label.setText(
                "Menyusun halaman dan menyematkan grafik ke dokumen PDF."
            )
            self.loading_card.setVisible(True)
            worker = ReportPdfWorker(self.current_html, str(self.chart_dir or ""), path)
            self._start_report_worker(worker, "pdf")
        except Exception as exc:
            self._finish_report_busy()
            QMessageBox.critical(self, "Gagal Mengekspor PDF", f"PDF tidak dapat disimpan:\n{exc}")

class ResearchInsightsPage(QWidget):
    """One workspace for literature landscape and potential research gaps."""

    def __init__(self, landscape_page, gap_page):
        super().__init__()
        self.setObjectName("researchInsightsPage")
        self.setAttribute(Qt.WA_StyledBackground, True)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("researchInsightsTabs")
        self.tabs.setAttribute(Qt.WA_StyledBackground, True)
        self.tabs.addTab(landscape_page, "Landscape")
        self.tabs.addTab(gap_page, "Research Gap")
        tab_pages = self.tabs.findChild(QStackedWidget)
        if tab_pages is not None:
            tab_pages.setObjectName("researchInsightsTabPages")
            tab_pages.setAttribute(Qt.WA_StyledBackground, True)
        self.tabs.tabBar().setVisible(False)
        root.addWidget(self.tabs)
        self.landscape_page = landscape_page
        self.gap_page = gap_page
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        if self.tabs.widget(index) is self.gap_page:
            self.gap_page.ensure_explanation()


def _load_credits_manifest() -> dict[str, Any]:
    """Build the visible contributor manifest from the existing project credits."""
    source_path = _resource_path(Path("data") / "credits.json")
    with source_path.open("r", encoding="utf-8") as source:
        credits = json.load(source)

    visible_roles = {
        "Academic Supervisor",
        "Software Engineering",
        "UI/UX Design",
        "Graphic Design & Web Developer",
        "Scientific Paper Writing",
    }
    contributors = [
        {"role": section.get("role", ""), "names": section.get("names", [])}
        for section in credits.get("sections", [])
        if section.get("role") in visible_roles
    ]
    return {
        "team": credits.get("team", "Metriks Studio"),
        "project": "Dataset Research",
        "tagline": "Built with curiosity. Designed to make research easier.",
        "sections": contributors,
    }


def _credits_manifest_html(manifest: dict[str, Any]) -> str:
    """Render formatted JSON with stable line-number alignment and token colors."""
    source = json.dumps(manifest, ensure_ascii=False, indent=2)
    token_pattern = re.compile(r'("(?:\\.|[^"\\])*")|([{}\[\],:])|(-?\d+(?:\.\d+)?|true|false|null)')
    rows = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        parts = []
        position = 0
        for match in token_pattern.finditer(line):
            parts.append(html_escape(line[position:match.start()]))
            token = match.group(0)
            if match.group(1):
                is_key = line[match.end():].lstrip().startswith(":")
                color = "#F2BD68" if is_key else "#A9D7FF"
            elif match.group(2):
                color = "#63D6E8"
            else:
                color = "#B6E379"
            parts.append(f'<span style="color:{color}">{html_escape(token)}</span>')
            position = match.end()
        parts.append(html_escape(line[position:]))
        code_line = "".join(parts) or "&nbsp;"
        rows.append(
            '<tr><td class="line-number">'
            f"{line_number:02d}"
            '</td><td class="code-line">'
            f"{code_line}"
            "</td></tr>"
        )
    return (
        '<html><head><meta charset="utf-8"><style>'
        'body{margin:0;background:#1B2029;color:#DCE5F1;}'
        'table{border-collapse:collapse;width:100%;table-layout:fixed;}'
        'td{vertical-align:top;padding:1px 0;}'
        '.line-number{width:42px;padding-right:14px;text-align:right;color:#B6E379;'
        'font-family:Consolas,"Cascadia Code",monospace;user-select:none;}'
        '.code-line{padding-right:18px;white-space:pre-wrap;word-wrap:break-word;'
        'font-family:Consolas,"Cascadia Code",monospace;}'
        '</style></head><body><table>'
        + "".join(rows)
        + "</table></body></html>"
    )


class CopyrightPage(QWidget):
    """Project credits and copyright information for DORA."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("copyrightPage")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        content = QVBoxLayout(container)
        content.setContentsMargins(28, 24, 28, 28)
        content.setSpacing(16)
        scroll.setWidget(container)
        root.addWidget(scroll)

        eyebrow = QLabel("PROJECT CONTRIBUTORS")
        eyebrow.setObjectName("creditsEyebrow")
        content.addWidget(eyebrow)

        heading = QLabel("Credits")
        heading.setObjectName("aboutPageTitle")
        content.addWidget(heading)

        subtitle = QLabel("Built with curiosity. Designed to make research easier.")
        subtitle.setObjectName("aboutPageSubtitle")
        subtitle.setWordWrap(True)
        content.addWidget(subtitle)

        manifest = _load_credits_manifest()
        code_panel = QFrame()
        code_panel.setObjectName("creditsCodePanel")
        panel_layout = QVBoxLayout(code_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        panel_header = QHBoxLayout()
        panel_header.setContentsMargins(18, 13, 18, 13)
        panel_header.setSpacing(8)
        panel_title = QLabel("team-manifest.json")
        panel_title.setObjectName("creditsPanelTitle")
        panel_header.addWidget(panel_title)
        panel_header.addStretch(1)
        panel_language = QLabel("JSON")
        panel_language.setObjectName("creditsLanguageBadge")
        panel_header.addWidget(panel_language)
        panel_layout.addLayout(panel_header)

        code = QTextBrowser()
        code.setObjectName("creditsManifestCode")
        code.setReadOnly(True)
        code.setOpenLinks(False)
        code.setOpenExternalLinks(False)
        code.setLineWrapMode(QTextEdit.WidgetWidth)
        code.setFrameShape(QFrame.NoFrame)
        code.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        code.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        code.setMinimumHeight(340)
        code.setHtml(_credits_manifest_html(manifest))
        panel_layout.addWidget(code, 1)
        content.addWidget(code_panel, 1)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        copyright_mark = QLabel("\u00a9 2026 Metriks Studio. All rights reserved.")
        copyright_mark.setObjectName("creditsCopyright")
        copyright_mark.setWordWrap(True)
        footer.addWidget(copyright_mark, 1)
        content.addLayout(footer)
        content.addStretch(1)


class HelpPage(QWidget):
    """Panduan penggunaan yang bisa dibuka dari menu Help."""

    def __init__(self, main_window):
        super().__init__()
        self.setObjectName("helpPage")
        self.main_window = main_window
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("helpScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.scroll)

        self.container = QWidget()
        self.container.setObjectName("helpContainer")
        self.content = QVBoxLayout(self.container)
        self.content.setContentsMargins(28, 24, 28, 28)
        self.content.setSpacing(16)
        self.scroll.setWidget(self.container)

        heading = QLabel("Bantuan")
        heading.setObjectName("helpPageTitle")
        self.content.addWidget(heading)

        subtitle = QLabel(
            "Panduan singkat untuk memakai DORA dari memuat dataset sampai menyimpan laporan."
        )
        subtitle.setObjectName("helpPageSubtitle")
        subtitle.setWordWrap(True)
        self.content.addWidget(subtitle)

        hero = QFrame()
        hero.setObjectName("helpHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(22, 20, 22, 20)
        hero_layout.setSpacing(6)
        hero_title = QLabel("Alur kerja yang disarankan")
        hero_title.setObjectName("helpHeroTitle")
        hero_body = QLabel(
            "Mulai dari satu file CSV. Setelah dataset dimuat, lanjutkan ke Analysis, "
            "lalu gunakan ML Intelligence dan Academic Research sesuai kebutuhan."
        )
        hero_body.setObjectName("helpHeroBody")
        hero_body.setWordWrap(True)
        hero_layout.addWidget(hero_title)
        hero_layout.addWidget(hero_body)
        self.content.addWidget(hero)

        self.cards_layout = QGridLayout()
        self.cards_layout.setHorizontalSpacing(14)
        self.cards_layout.setVerticalSpacing(14)
        self.cards = []
        self.content.addLayout(self.cards_layout)

        guides = (
            (
                "1. Muat dataset dari komputer",
                "Buka Dataset, pilih Upload Dataset, lalu pilih file CSV. Pada jendela konfirmasi, "
                "periksa nama dataset, pemisah kolom, encoding, dan kolom target. Pastikan baris "
                "pertama berisi nama kolom, lalu pilih Unggah & Pratinjau.",
                "Buka Dataset",
                self.main_window.open_datasets_page,
            ),
            (
                "2. Cari dataset online",
                "Buka Dataset Search, ketik kata kunci, pilih Kaggle dataset, Hugging Face dataset, "
                "atau Semua sumber, lalu pilih Cari. Pilih berkas CSV yang tersedia dan pilih "
                "Impor & Analisis untuk mengunduh dan membukanya.",
                "Buka Dataset Search",
                self.main_window.open_dataset_search_page,
            ),
            (
                "3. Pratinjau dan rapikan data",
                "Di halaman Dataset, periksa beberapa baris, daftar kolom, nilai kosong, dan duplikat. "
                "Gunakan pilihan pembersihan cepat yang tersedia bila diperlukan, lalu pilih "
                "Lanjutkan ke Analysis.",
                "Buka pratinjau Dataset",
                self.main_window.open_datasets_page,
            ),
            (
                "4. Jalankan analisis dataset",
                "Pada halaman Analysis, pilih Jalankan Analysis. Tinjau ringkasan dataset, statistik kolom, "
                "kualitas data, nilai kosong, duplikat, pencilan, dan hubungan antar kolom. "
                "Gunakan hasilnya untuk menentukan langkah berikutnya.",
                "Buka Analysis",
                self.main_window.open_analysis_page,
            ),
            (
                "5. Tinjau ML Intelligence",
                "Setelah dataset dimuat, buka ML Intelligence untuk melihat perkiraan jenis tugas "
                "Machine Learning dan metode yang disarankan. Periksa alasan dan keterbatasan "
                "rekomendasi sebelum memilih metode penelitian.",
                "Buka ML Intelligence",
                self.main_window.open_ml_page,
            ),
            (
                "6. Cari referensi penelitian",
                "Setelah analisis tersedia, buka Academic Research dan pilih Jalankan Academic Research. "
                "Tinjau kata kunci, query, dan hasil paper. Periksa sumber dan relevansi paper "
                "sebelum menggunakannya dalam tulisan.",
                "Buka Academic Research",
                self.main_window.open_research_page,
            ),
            (
                "7. Tinjau paper dan temuan riset",
                "Gunakan Papers untuk melihat kandidat paper. Research Landscape merangkum pola "
                "metode dan publikasi; Research Gap menampilkan kandidat celah riset yang bisa "
                "ditinjau lebih lanjut.",
                "Buka Papers",
                self.main_window.open_papers_tool,
            ),
            (
                "8. Buat laporan",
                "Buka Research Report, pilih Perbarui Pratinjau, lalu tekan Simpan sebagai PDF "
                "jika tombol ekspor sudah aktif. Laporan gabungan membutuhkan dataset, Analisis "
                "Dataset, ML Intelligence, dan Academic Research.",
                "Buka Research Report",
                self.main_window.open_report_tool,
            ),
        )
        for title, body, action, callback in guides:
            self._add_guide_card(title, body, action, callback)
        self.content.addStretch(1)
        self._card_columns = 0

    def _add_guide_card(self, title, body, action_text, callback):
        card = QFrame()
        card.setObjectName("helpInfoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(9)
        heading = QLabel(title)
        heading.setObjectName("helpCardTitle")
        description = QLabel(body)
        description.setObjectName("helpBody")
        description.setWordWrap(True)
        action = QPushButton(action_text)
        action.setObjectName("helpShortcut")
        action.setCursor(Qt.PointingHandCursor)
        action.clicked.connect(callback)
        layout.addWidget(heading)
        layout.addWidget(description, 1)
        layout.addWidget(action, 0, Qt.AlignLeft)
        self.cards.append(card)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow_cards(event.size().width())

    def _reflow_cards(self, width):
        scale = self.main_window.scale_manager.get_scale()
        columns = 2 if width / max(scale, 0.01) >= 1000 else 1
        if columns == self._card_columns:
            return
        for card in self.cards:
            self.cards_layout.removeWidget(card)
        for index, card in enumerate(self.cards):
            self.cards_layout.addWidget(card, index // columns, index % columns)
        self._card_columns = columns

    def apply_scale(self, _scale=None):
        self._reflow_cards(self.width())


class LatestProjectRestoreWorker(QThread):
    """Load the last project without blocking the startup UI thread."""

    result_ready = Signal(object)

    def __init__(self, repository, version_manager, preferred_project_id, parent=None):
        super().__init__(parent)
        self.repository = repository
        self.version_manager = version_manager
        self.preferred_project_id = preferred_project_id

    def run(self):
        try:
            projects = self.repository.list_projects()
            if self.preferred_project_id:
                projects.sort(
                    key=lambda project: project.get("id") != self.preferred_project_id
                )
            for project in projects:
                project_id = project.get("id")
                if project_id is None or not self.version_manager.has_versions(project_id):
                    continue
                try:
                    dataframe = self.version_manager.load_current_dataframe(project_id)
                    workflow_state = self.repository.get_workflow_state(project_id) or {}
                except Exception as error:
                    print(f"Warning: could not restore project {project_id}: {error}")
                    continue

                self.result_ready.emit({
                    "project": project,
                    "dataframe": dataframe,
                    "workflow_state": workflow_state,
                })
                return
            self.result_ready.emit(None)
        except Exception as error:
            self.result_ready.emit({"error": str(error)})


class MainWindow(QMainWindow):
    """Main application window for Dataset Research.

    The UI pages are intentionally kept in this single module so the
    desktop interface has one clear entry point while the analytical
    backend remains completely independent.
    """

    def __init__(self, startup_splash=None):
        super().__init__()
        self._startup_splash = startup_splash
        self._startup_checkpoint("Menyiapkan tampilan DORA…")

        application = QApplication.instance()
        self.scale_manager = ScaleManager()
        application.setProperty("uiScale", self.scale_manager.get_scale())
        application.installEventFilter(self)
        self._base_application_font = QFont(application.font())
        if self._base_application_font.pointSizeF() <= 0:
            self._base_application_font.setPointSizeF(10.0)
        self._widget_scale_baselines = {}
        self._layout_scale_baselines = {}
        self._icon_scale_baselines = {}
        self._font_scale_baselines = {}
        self._style_scale_baselines = {}
        self._scale_widget_flags = {}
        self._page_scale_hook_cache = {}
        self._scale_stylesheet_template_cache = {}
        self._scale_refresh_scheduled = False
        self._pending_scale_roots = []
        self._pending_ui_scale = None
        self._scale_apply_timer = QTimer(self)
        self._scale_apply_timer.setSingleShot(True)
        self._scale_apply_timer.setInterval(32)
        self._scale_apply_timer.timeout.connect(self._flush_ui_scale)
        self._page_chrome_sync_scheduled = False
        self.scale_manager.scaleChanged.connect(self._apply_ui_scale)
        self._sidebar_manual_state = None

        # Load the stylesheet before building the interface so every
        # widget is styled as soon as it is created.
        self._load_stylesheet()

        self.setWindowTitle("Data Overview Research & Assistant v6")
        app_icon = _resource_path("assets/dataset_research.ico")
        if app_icon.exists():
            self.setWindowIcon(QIcon(str(app_icon)))

        self._set_screen_aware_window_limits()

        self.current_dataset = None
        self.current_file_path = None
        self.analysis_result = {}
        self._shutdown_pending = False
        self._shutdown_idle_checks = 0
        self._session_data_saved = False
        self._startup_restore_complete = False
        self._project_restore_pending = False
        self._project_restore_thread = None
        self._dataset_session_changed = False
        self._restored_ml_task = {}
        self._restored_ml_methods = {}
        self._restored_research_result = {}
        self._pending_insights_tab = 0
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(50)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)

        self.loader = DatasetLoader()
        self.profiler = DatasetProfiler()
        self.statistics = DatasetStatistics()
        self.missing_analyzer = MissingValueAnalyzer()
        self.duplicate_analyzer = DuplicateAnalyzer()
        self.outlier_analyzer = OutlierAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.fingerprint_analyzer = DatasetFingerprint()
        self.quality_diagnoser = DataQualityDiagnoser()

        self.database = Database()
        self.repository = ProjectRepository(self.database)
        self.version_manager = DatasetVersionManager(
            self.database,
            data_dir=get_data_dir(),
        )
        self.current_project_id = None

        self._startup_checkpoint("Membangun halaman workspace…")
        self._build_window()
        self._build_view_menu()
        self._build_scale_status()
        # Only scale the visible chrome and first page during startup. Other
        # pages are scaled on first navigation, avoiding a full hidden-widget
        # walk while the application is opening.
        initial_page = self.pages.get(
            getattr(self, "_current_page_key", "dashboard"), self.dashboard
        )
        initial_host = self.page_hosts.get(id(initial_page), initial_page)
        initial_scale = self.scale_manager.get_scale()
        self._apply_widget_scale(
            initial_scale,
            [self.sidebar_container, self.topbar, initial_host],
        )
        self._apply_active_page_layout(initial_page, initial_scale)
        self._page_scale_values = {id(initial_page): initial_scale}
        self._initial_widget_scale_applied = True
        self._startup_checkpoint("Memulihkan dataset dan halaman terakhir…")
        self._restore_latest_project()
        self._restore_window_geometry()
        self._restore_last_page()
        self._startup_restore_complete = True
        get_telemetry().track("app_open", status="started")
        self._startup_checkpoint("Workspace siap dibuka…")

    def _startup_checkpoint(self, message: str):
        splash = getattr(self, "_startup_splash", None)
        if splash is None or not is_qt_object_valid(splash):
            return
        splash.set_message(message)
        application = QApplication.instance()
        if application is not None:
            application.processEvents(QEventLoop.ExcludeUserInputEvents)

    def _active_worker_threads(self):
        """Return running background workers that could still write session data."""
        worker_fields = {
            "dataset_search_page": ("_search_thread", "_download_thread"),
            "analysis_page": (
                "_analysis_thread",
                "_gemini_thread",
                "_clean_preview_thread",
                "_clean_version_thread",
                "_export_thread",
            ),
            "node_editor_page": ("thread",),
            "ml_page": ("_ml_thread",),
            "research_page": (
                "_research_thread", "_insights_thread", "_sinta_thread",
                "_journal_index_thread", "_non_sinta_thread",
            ),
            "gap_tool_page": ("_explanation_thread",),
            "report_tool_page": ("_report_thread",),
        }
        active = []
        for page_name, attributes in worker_fields.items():
            page = getattr(self, page_name, None)
            if page is None:
                continue
            for attribute in attributes:
                thread = getattr(page, attribute, None)
                if thread is not None and thread.isRunning():
                    active.append(thread)
        restore_thread = getattr(self, "_project_restore_thread", None)
        if restore_thread is not None and restore_thread.isRunning():
            active.append(restore_thread)
        return active

    def _poll_shutdown(self):
        if self._project_restore_pending or self._active_worker_threads():
            self._shutdown_idle_checks = 0
            self.statusBar().showMessage(
                "Menunggu proses selesai agar progres tersimpan…"
            )
            return

        # Give queued worker-finished slots a short chance to run before cleanup.
        self._shutdown_idle_checks += 1
        if self._shutdown_idle_checks < 2:
            return

        self._shutdown_timer.stop()
        try:
            self._save_application_progress()
        except Exception as error:
            self._shutdown_pending = False
            self._shutdown_idle_checks = 0
            self.setWindowTitle("Data Overview Research & Assistant v6")
            if self.centralWidget() is not None:
                self.centralWidget().setEnabled(True)
            self.statusBar().clearMessage()
            QMessageBox.warning(
                self,
                "Progres belum tersimpan",
                "Aplikasi tetap dibuka karena progres lokal belum berhasil disimpan. "
                f"Periksa penyimpanan lalu coba tutup kembali.\n\n{error}",
            )
            return

        self._session_data_saved = True
        self.close()

    def _save_application_progress(self):
        """Save recoverable application state before closing without deleting projects."""
        settings = self.scale_manager.settings
        settings.setValue("last_page_key", getattr(self, "_current_page_key", "dashboard"))
        settings.setValue("active_project_id", self.current_project_id)
        self._save_window_geometry()

        editor_page = getattr(self, "editor_analysis_page", None)
        if editor_page is not None and not editor_page._save_autosave_on_exit():
            raise OSError("Autosave Editor Analysis gagal ditulis ke penyimpanan lokal.")
        node_page = getattr(self, "node_editor_page", None)
        if node_page is not None and not node_page.save_session_graph():
            raise OSError("Autosave Node Editor gagal ditulis ke penyimpanan lokal.")

        if self.current_dataset is not None:
            project_id = self._ensure_current_project()
            settings.setValue("active_project_id", project_id)
            if project_id is not None:
                workflow_sections = {
                    "analysis": self.analysis_result,
                    "ml_task": getattr(self.ml_page, "last_task_result", {}),
                    "research": getattr(self.research_page, "research_result", {}),
                }
                methods = dict(getattr(self.ml_page, "last_method_result", {}) or {})
                intelligence = getattr(self.ml_page, "last_intelligence_result", {}) or {}
                if intelligence:
                    methods["_intelligence_result"] = intelligence
                if methods:
                    workflow_sections["ml_methods"] = methods
                for section, value in workflow_sections.items():
                    if value:
                        self.repository.save_workflow_section(project_id, section, value)

        settings.sync()

    def closeEvent(self, event):
        if self._session_data_saved:
            self._save_window_geometry()
            event.accept()
            return

        event.ignore()
        if self._shutdown_pending:
            return

        self._shutdown_pending = True
        self._shutdown_idle_checks = 0
        self.setWindowTitle("Data Overview Research & Assistant v6 — Menutup sesi")
        if self.centralWidget() is not None:
            self.centralWidget().setEnabled(False)
        self.statusBar().showMessage(
            "Menyelesaikan proses dan menyimpan progres…"
        )
        self._shutdown_timer.start()


    # =========================================================
    # STYLESHEET
    # =========================================================

    def _load_stylesheet(self):
        """Load the UI stylesheet located beside this module."""
        qss_path = _resource_path("app.qss")
        theme_path = _resource_path(Path("styles") / "theme.qss")

        if not qss_path.exists():
            print(f"Warning: stylesheet not found: {qss_path}")
            return

        try:
            application = QApplication.instance()
            if application is not None:
                # Keep Qt controls independent from the host OS appearance.
                # DORA's stylesheet and palette are designed for the light UI.
                application.setStyle("Fusion")
                application.styleHints().setColorScheme(Qt.ColorScheme.Light)

            assets_root = _resource_path("assets")
            for font_file in (
                assets_root / "fonts" / "Inter-Variable.ttf",
                assets_root / "fonts" / "PlusJakartaSans-Variable.ttf",
                assets_root / "fonts" / "IntelOneMono-Medium.ttf",
            ):
                if font_file.exists():
                    QFontDatabase.addApplicationFont(str(font_file))

            stylesheet = qss_path.read_text(encoding="utf-8-sig")
            self._base_stylesheet_template = stylesheet
            if theme_path.exists():
                theme = theme_path.read_text(encoding="utf-8")
                tokens = dict(re.findall(r"/\*\s*@([\w-]+):\s*(.*?)\s*\*/", theme))
                self._theme_tokens = tokens
                theme = re.sub(
                    r"var\(--([\w-]+)\)",
                    lambda match: tokens.get(match.group(1), match.group(0)),
                    theme,
                )
                self._theme_stylesheet_template = theme
                canvas = tokens.get("canvas", "#f1f7ff")
                surface = tokens.get("surface", "#ffffff")
                ink = tokens.get("ink", "#222222")
                body = tokens.get("body", "#564e4e")
                muted = tokens.get("muted", "#757575")
                blue = tokens.get("blue", "#006bb8")
                blue_active = tokens.get("blue-active", "#c9e0ff")
                # Start from Fusion's neutral palette, then apply DORA's light
                # colors to every state so disabled/inactive controls stay legible.
                palette = QPalette()
                role_colors = {
                    QPalette.Window: canvas,
                    QPalette.WindowText: ink,
                    QPalette.Base: surface,
                    QPalette.AlternateBase: canvas,
                    QPalette.ToolTipBase: surface,
                    QPalette.ToolTipText: ink,
                    QPalette.Text: ink,
                    QPalette.Button: surface,
                    QPalette.ButtonText: ink,
                    QPalette.BrightText: tokens.get("danger", "#e00000"),
                    QPalette.Highlight: blue_active,
                    QPalette.HighlightedText: ink,
                    QPalette.Link: blue,
                    QPalette.LinkVisited: blue,
                }
                for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
                    for role, color in role_colors.items():
                        palette.setColor(group, role, QColor(color))
                for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
                    palette.setColor(QPalette.Disabled, role, QColor(muted if role != QPalette.WindowText else body))
                self.setPalette(palette)
                if application is not None:
                    application.setPalette(palette)
            if application is not None:
                scale = self.scale_manager.get_scale()
                stylesheet = build_stylesheet(scale, qss_path)
                if getattr(self, "_theme_stylesheet_template", ""):
                    stylesheet += "\n\n" + scale_stylesheet(
                        self._theme_stylesheet_template, scale
                    )
                self._active_scale_stylesheet = stylesheet
                application.setStyleSheet(stylesheet)
                font = QFont(self._base_application_font)
                font.setPointSizeF(
                    self._base_application_font.pointSizeF() * scale
                )
                application.setFont(font)
        except OSError as error:
            print(f"Warning: failed to load stylesheet: {error}")

    def _apply_ui_scale(self, scale: float):
        """Coalesce rapid zoom events and apply only the newest scale."""
        self._pending_ui_scale = float(scale)
        self._scale_apply_timer.start()

    def _flush_ui_scale(self):
        """Apply the pending scale once the user pauses zoom input."""
        scale = self._pending_ui_scale
        self._pending_ui_scale = None
        if scale is None:
            return

        application = QApplication.instance()
        if application is None:
            return
        previous_scale = float(application.property("uiScale") or scale)
        if abs(previous_scale - scale) < 0.001:
            # Auto/manual presets may resolve to the same effective scale.
            self._update_scale_controls()
            return
        current_page = self.pages.get(
            getattr(self, "_current_page_key", "dashboard")
        )
        active_host = (
            self.page_hosts.get(id(current_page), current_page)
            if current_page is not None else None
        )
        scale_roots = [
            widget for widget in (
                getattr(self, "sidebar_container", None),
                getattr(self, "topbar", None),
                active_host,
            )
            if isinstance(widget, QWidget)
        ]
        application.setProperty("uiScale", scale)
        stylesheet = self._build_scaled_stylesheet(scale)
        self._active_scale_stylesheet = stylesheet
        shared_roots = [
            getattr(self, "sidebar_container", None),
            getattr(self, "topbar", None),
            self.menuBar(),
            self.statusBar(),
        ]
        self._apply_scale_stylesheet(stylesheet, (*shared_roots, active_host))
        self._apply_widget_scale(scale, scale_roots)
        if hasattr(self, "sidebar_container"):
            self._sync_page_chrome(self.width(), self.height())
        if current_page is not None:
            self._apply_active_page_layout(current_page, scale)
            self._page_scale_values[id(current_page)] = scale
        self._update_scale_controls()

    def _build_scaled_stylesheet(self, scale: float) -> str:
        qss_path = _resource_path("app.qss")
        stylesheet = build_stylesheet(scale, qss_path)
        theme = getattr(self, "_theme_stylesheet_template", "")
        if theme:
            stylesheet += "\n\n" + scale_stylesheet(theme, scale)
        return stylesheet

    def _stylesheet_template_for_root(self, root: QWidget) -> str:
        key = id(root)
        cached = self._scale_stylesheet_template_cache.get(key)
        if cached is not None:
            return cached

        names = {root.objectName()} if root.objectName() else set()
        classes = set()
        for widget in (root, *root.findChildren(QWidget)):
            if widget.objectName():
                names.add(widget.objectName())
            meta_object = widget.metaObject()
            while meta_object is not None:
                classes.add(meta_object.className())
                meta_object = meta_object.superClass()

        template = getattr(self, "_base_stylesheet_template", "")
        theme = getattr(self, "_theme_stylesheet_template", "")
        if theme:
            template += "\n\n" + theme
        template = re.sub(r"/\*.*?\*/", "", template, flags=re.DOTALL)
        filtered_rules = []
        for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", template):
            scaled_declarations = [
                declaration.strip()
                for declaration in match.group(2).split(";")
                if "@px(" in declaration
                or re.search(r"-?\d+(?:\.\d+)?px\b", declaration)
            ]
            if not scaled_declarations:
                continue
            selectors = []
            for selector in match.group(1).split(","):
                selector = selector.strip()
                if not selector:
                    continue
                object_names = re.findall(r"#([\w-]+)", selector)
                if object_names:
                    if any(name in names for name in object_names):
                        selectors.append(selector)
                    continue
                widget_classes = re.findall(r"\bQ[A-Za-z_][\w]*\b", selector)
                if widget_classes:
                    if any(name in classes for name in widget_classes):
                        selectors.append(selector)
                    continue
                if selector.startswith("*") or "[" in selector or "." in selector:
                    selectors.append(selector)
            if selectors:
                body = "; ".join(scaled_declarations)
                filtered_rules.append(",\n".join(selectors) + " {" + body + ";}")

        filtered = "\n".join(filtered_rules)
        self._scale_stylesheet_template_cache[key] = filtered
        return filtered

    def _apply_scale_stylesheet(self, stylesheet: str, roots):
        """Refresh relevant QSS on chrome and the active page only."""
        application = QApplication.instance()
        scale = float(
            application.property("uiScale") if application is not None
            else self.scale_manager.get_scale()
        )
        seen = set()
        for widget in roots:
            if (
                not isinstance(widget, QWidget)
                or not is_qt_object_valid(widget)
                or id(widget) in seen
            ):
                continue
            seen.add(id(widget))
            root_template = self._stylesheet_template_for_root(widget)
            root_stylesheet = scale_stylesheet(root_template, scale)
            if widget.styleSheet() != root_stylesheet:
                widget.setStyleSheet(root_stylesheet)

    def _apply_active_page_layout(self, page, scale: float):
        """Refresh scale-aware layout hooks on the active page only."""
        page_key = id(page)
        hooks = self._page_scale_hook_cache.get(page_key)
        if hooks is None:
            hooks = []
            for widget in (page, *page.findChildren(QWidget)):
                if callable(getattr(widget, "apply_scale", None)) or callable(
                    getattr(widget, "_apply_responsive_layout", None)
                ):
                    hooks.append(widget)
            self._page_scale_hook_cache[page_key] = hooks

        for widget in hooks:
            if not is_qt_object_valid(widget):
                continue
            if widget is not page and widget.isHidden():
                continue
            apply_scale = getattr(widget, "apply_scale", None)
            if callable(apply_scale):
                apply_scale(scale)
            else:
                responsive_layout = getattr(widget, "_apply_responsive_layout", None)
                if callable(responsive_layout):
                    responsive_layout(widget.width())

    def _apply_widget_scale(self, scale: float, roots=None):
        """Scale fixed widget hints, layout gaps, and explicit icon/font sizes."""
        max_hint = 16_777_215
        if roots is None:
            widgets = [self, *self.findChildren(QWidget)]
            layouts = self.findChildren(QLayout)
        else:
            widgets = []
            layouts = []
            for root in roots:
                if not isinstance(root, QWidget):
                    continue
                widgets.extend((root, *root.findChildren(QWidget)))
                if root.layout() is not None:
                    layouts.append(root.layout())
                layouts.extend(root.findChildren(QLayout))
            # A widget may be queued more than once while Qt constructs a subtree.
            widgets = list({id(widget): widget for widget in widgets}.values())
            layouts = list({id(layout): layout for layout in layouts}.values())
        for widget in widgets:
            key = id(widget)
            if key not in self._widget_scale_baselines:
                self._widget_scale_baselines[key] = (
                    widget.minimumWidth(), widget.maximumWidth(),
                    widget.minimumHeight(), widget.maximumHeight(),
                )
                icon_size_getter = getattr(widget, "iconSize", None)
                icon_size_setter = getattr(widget, "setIconSize", None)
                if callable(icon_size_getter) and callable(icon_size_setter):
                    self._icon_scale_baselines[key] = icon_size_getter()
                if not widget.property("scaleStylesheetExternally"):
                    self._style_scale_baselines[key] = widget.styleSheet()
                if widget.testAttribute(Qt.WA_SetFont):
                    base_font = QFont(widget.font())
                    if base_font.pointSizeF() > 0:
                        base_font.setPointSizeF(base_font.pointSizeF() / max(scale, 0.01))
                    elif base_font.pixelSize() > 0:
                        base_font.setPixelSize(
                            max(1, round(base_font.pixelSize() / max(scale, 0.01)))
                        )
                    self._font_scale_baselines[key] = base_font

            min_width, max_width, min_height, max_height = self._widget_scale_baselines[key]
            if key not in self._scale_widget_flags:
                icon_size = self._icon_scale_baselines.get(key)
                base_style = self._style_scale_baselines.get(key, "")
                self._scale_widget_flags[key] = bool(
                    min_width > 0 or max_width < max_hint
                    or min_height > 0 or max_height < max_hint
                    or (icon_size is not None and not icon_size.isEmpty())
                    or key in self._font_scale_baselines
                    or bool(base_style)
                    or isinstance(widget, QDialog)
                )
            if not self._scale_widget_flags[key]:
                continue

            if min_width == max_width and max_width < max_hint:
                target_width = max(0, round(min_width * scale))
                if (widget.minimumWidth(), widget.maximumWidth()) != (target_width, target_width):
                    widget.setFixedWidth(target_width)
            else:
                if min_width > 0:
                    target_min_width = max(0, round(min_width * scale))
                    if widget.minimumWidth() != target_min_width:
                        widget.setMinimumWidth(target_min_width)
                if max_width < max_hint:
                    target_max_width = max(0, round(max_width * scale))
                    if widget.maximumWidth() != target_max_width:
                        widget.setMaximumWidth(target_max_width)
            if min_height == max_height and max_height < max_hint:
                target_height = max(0, round(min_height * scale))
                if (widget.minimumHeight(), widget.maximumHeight()) != (target_height, target_height):
                    widget.setFixedHeight(target_height)
            else:
                if min_height > 0:
                    target_min_height = max(0, round(min_height * scale))
                    if widget.minimumHeight() != target_min_height:
                        widget.setMinimumHeight(target_min_height)
                if max_height < max_hint:
                    target_max_height = max(0, round(max_height * scale))
                    if widget.maximumHeight() != target_max_height:
                        widget.setMaximumHeight(target_max_height)

            icon_size = self._icon_scale_baselines.get(key)
            icon_size_setter = getattr(widget, "setIconSize", None)
            if icon_size is not None and not icon_size.isEmpty() and callable(icon_size_setter):
                target_icon_size = QSize(
                    round(icon_size.width() * scale), round(icon_size.height() * scale)
                )
                if widget.iconSize() != target_icon_size:
                    icon_size_setter(target_icon_size)
            base_font = self._font_scale_baselines.get(key)
            if base_font is not None:
                scaled_font = QFont(base_font)
                if base_font.pointSizeF() > 0:
                    scaled_font.setPointSizeF(base_font.pointSizeF() * scale)
                elif base_font.pixelSize() > 0:
                    scaled_font.setPixelSize(max(1, round(base_font.pixelSize() * scale)))
                if widget.font() != scaled_font:
                    widget.setFont(scaled_font)

            base_style = self._style_scale_baselines.get(key)
            if base_style:
                scaled_style = scale_stylesheet(base_style, scale)
                if widget.styleSheet() != scaled_style:
                    widget.setStyleSheet(scaled_style)

            if isinstance(widget, QDialog):
                screen = widget.screen() or QApplication.primaryScreen()
                if screen is not None:
                    available = screen.availableGeometry()
                    max_dialog_width = max(1, round(available.width() * 0.9))
                    max_dialog_height = max(1, round(available.height() * 0.9))
                    min_size = QSize(
                        min(widget.minimumWidth(), max_dialog_width),
                        min(widget.minimumHeight(), max_dialog_height),
                    )
                    max_size = QSize(
                        min(widget.maximumWidth(), max_dialog_width),
                        min(widget.maximumHeight(), max_dialog_height),
                    )
                    if widget.minimumSize() != min_size:
                        widget.setMinimumSize(min_size)
                    if widget.maximumSize() != max_size:
                        widget.setMaximumSize(max_size)
                    if widget.objectName() == "uploadDatasetDialog":
                        width = min(round(max_width * scale), max_dialog_width)
                        height = min(round(max_height * scale), max_dialog_height)
                        target_size = QSize(max(1, width), max(1, height))
                        if widget.size() != target_size:
                            widget.resize(target_size)

        for layout in layouts:
            key = id(layout)
            if key not in self._layout_scale_baselines:
                margins = layout.contentsMargins()
                self._layout_scale_baselines[key] = (
                    margins.left(), margins.top(), margins.right(), margins.bottom(),
                    layout.spacing(),
                )
            left, top, right, bottom, spacing = self._layout_scale_baselines[key]
            target_margins = tuple(
                value if value < 0 else round(value * scale)
                for value in (left, top, right, bottom)
            )
            current_margins = layout.contentsMargins()
            if tuple(
                (current_margins.left(), current_margins.top(),
                 current_margins.right(), current_margins.bottom())
            ) != target_margins:
                layout.setContentsMargins(*target_margins)
            if spacing >= 0:
                target_spacing = round(spacing * scale)
                if layout.spacing() != target_spacing:
                    layout.setSpacing(target_spacing)

    def _refresh_new_widget_scale(self):
        self._scale_refresh_scheduled = False
        roots, self._pending_scale_roots = self._pending_scale_roots, []
        candidates = [root for root in roots if is_qt_object_valid(root)]
        roots = []
        for candidate in candidates:
            if any(
                existing is candidate or existing.isAncestorOf(candidate)
                for existing in roots
            ):
                continue
            roots = [
                existing for existing in roots
                if not candidate.isAncestorOf(existing)
            ]
            roots.append(candidate)

        # A new child only changes the relevant page's scale hooks. Keep other
        # pages' cached hook lists intact instead of forcing full rescans.
        seen_pages = set()
        for page in getattr(self, "pages", {}).values():
            page_key = id(page)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            host = getattr(self, "page_hosts", {}).get(page_key, page)
            if any(
                root is page or page.isAncestorOf(root)
                or root is host or host.isAncestorOf(root)
                for root in roots
            ):
                self._page_scale_hook_cache.pop(page_key, None)

        scale = self.scale_manager.get_scale()
        self._apply_widget_scale(scale, roots)
        stylesheet = getattr(self, "_active_scale_stylesheet", "")
        if stylesheet:
            style_roots = [
                getattr(self, "sidebar_container", None),
                getattr(self, "topbar", None),
                self.menuBar(),
                self.statusBar(),
                *getattr(self, "page_hosts", {}).values(),
            ]
            affected_roots = []
            for style_root in style_roots:
                if not isinstance(style_root, QWidget):
                    continue
                if any(
                    root is style_root or style_root.isAncestorOf(root)
                    for root in roots
                ):
                    affected_roots.append(style_root)
                    self._scale_stylesheet_template_cache.pop(id(style_root), None)
            for root in roots:
                widget = root
                while isinstance(widget, QWidget) and widget is not self:
                    if isinstance(widget, QDialog):
                        affected_roots.append(widget)
                        self._scale_stylesheet_template_cache.pop(id(widget), None)
                        break
                    widget = widget.parentWidget()
            self._apply_scale_stylesheet(stylesheet, affected_roots)

    def _build_view_menu(self):
        view_menu = self.menuBar().addMenu("View")
        # Keep the sidebar control in the application chrome so it remains
        # available after the entire navigation rail has been hidden.
        self.menuBar().setCornerWidget(
            self.sidebar_toggle_button, Qt.TopLeftCorner
        )
        zoom_in = QAction("Zoom In", self)
        zoom_in.setShortcuts(["Ctrl+=", "Ctrl++"])
        zoom_in.triggered.connect(self.scale_manager.zoom_in)
        view_menu.addAction(zoom_in)

        zoom_out = QAction("Zoom Out", self)
        zoom_out.setShortcut("Ctrl+-")
        zoom_out.triggered.connect(self.scale_manager.zoom_out)
        view_menu.addAction(zoom_out)

        reset = QAction("Reset Zoom", self)
        reset.setShortcut("Ctrl+0")
        reset.triggered.connect(self.scale_manager.reset)
        view_menu.addAction(reset)
        view_menu.addSeparator()

        self.scale_menu = view_menu.addMenu("UI Scale")
        self._scale_actions = []
        group = QActionGroup(self)
        group.setExclusive(True)
        for label, value in SCALE_PRESETS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setData("auto" if value is None else value)
            group.addAction(action)
            self.scale_menu.addAction(action)
            action.triggered.connect(
                lambda checked=False, selected=action.data():
                self.scale_manager.set_scale(selected)
            )
            self._scale_actions.append(action)

        view_menu.addSeparator()
        self.sidebar_menu_action = QAction("Show Sidebar", self)
        self.sidebar_menu_action.setCheckable(True)
        self.sidebar_menu_action.setChecked(True)
        self.sidebar_menu_action.triggered.connect(
            self._set_sidebar_expanded
        )
        view_menu.addAction(self.sidebar_menu_action)

    def _build_scale_status(self):
        self.scale_status = QLabel()
        self.statusBar().addPermanentWidget(self.scale_status)
        self._update_scale_controls()

    def _update_scale_controls(self):
        if not hasattr(self, "scale_status"):
            return
        scale = self.scale_manager.get_scale()
        mode = f"Auto ({scale:.0%})" if self.scale_manager.is_auto else f"{scale:.0%}"
        self.scale_status.setText(f"UI Scale: {mode}")
        for action in getattr(self, "_scale_actions", []):
            selected = action.data()
            checked = (
                self.scale_manager.is_auto
                if selected == "auto"
                else not self.scale_manager.is_auto
                and abs(float(selected) - scale) < 0.001
            )
            action.setChecked(checked)

    def _set_screen_aware_window_limits(self, screen=None, set_initial=True):
        screen = screen or QApplication.primaryScreen()
        if screen is None:
            self._responsive_min_size = (320, 280)
            self._initial_window_size = (1100, 700)
            self.setMinimumSize(*self._responsive_min_size)
            self.resize(*self._initial_window_size)
            return
        available = screen.availableGeometry()
        max_width = max(1, round(available.width() * 0.9))
        max_height = max(1, round(available.height() * 0.9))
        min_width = min(max_width, max(1, round(available.width() * 0.45)))
        min_height = min(max_height, max(1, round(available.height() * 0.5)))
        self._responsive_min_size = (min_width, min_height)
        self._initial_window_size = (
            min(max_width, max(min_width, min(1500, max_width))),
            min(max_height, max(min_height, min(950, max_height))),
        )
        self.setMinimumSize(min_width, min_height)
        if id(self) in getattr(self, "_widget_scale_baselines", {}):
            baseline = self._widget_scale_baselines[id(self)]
            self._widget_scale_baselines[id(self)] = (
                min_width, baseline[1], min_height, baseline[3]
            )
        if set_initial:
            self.resize(*self._initial_window_size)

    def _restore_window_geometry(self):
        saved = self.scale_manager.settings.value("window_geometry")
        if saved:
            try:
                self.restoreGeometry(saved)
                if self.isMaximized():
                    self.showNormal()
            except (TypeError, ValueError):
                pass
        self._keep_window_on_screen()

    def _save_window_geometry(self):
        self.scale_manager.settings.setValue("window_geometry", self.saveGeometry())

    def _keep_window_on_screen(self):
        if not self.isWindow():
            return
        screen = QApplication.screenAt(self.frameGeometry().center())
        screen = screen or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        max_width = max(1, round(available.width() * 0.9))
        max_height = max(1, round(available.height() * 0.9))
        width = min(max_width, max(self.minimumWidth(), self.width()))
        height = min(max_height, max(self.minimumHeight(), self.height()))
        if (width, height) != (self.width(), self.height()):
            self.resize(width, height)
        x = min(max(self.x(), available.left()), available.right() - width + 1)
        y = min(max(self.y(), available.top()), available.bottom() - height + 1)
        if (x, y) != (self.x(), self.y()):
            self.move(x, y)

    def _set_sidebar_expanded(self, expanded: bool):
        self._sidebar_manual_state = not expanded
        self._sync_page_chrome(self.width(), self.height())

    def _toggle_sidebar(self):
        expanded = bool(
            getattr(self, "sidebar_toggle_button", None)
            and self.sidebar_toggle_button.isChecked()
        )
        self._set_sidebar_expanded(expanded)

    def showEvent(self, event):
        super().showEvent(event)
        handle = self.windowHandle()
        if handle is not None and not getattr(self, "_screen_signal_connected", False):
            handle.screenChanged.connect(self._on_screen_changed)
            self._screen_signal_connected = True

    def _on_screen_changed(self, screen):
        self._set_screen_aware_window_limits(screen, set_initial=False)
        self.scale_manager.update_screen(screen)
        self._keep_window_on_screen()

    def _apply_canvas_background(self, widget):
        color = QColor(self._theme_tokens.get("canvas", "#f1f7ff"))
        palette = widget.palette()
        palette.setColor(QPalette.Window, color)
        palette.setColor(QPalette.Base, color)
        widget.setPalette(palette)
        widget.setAutoFillBackground(True)

    # =========================================================
    # WINDOW
    # =========================================================

    def _build_window(self):
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        root_layout.setSizeConstraint(QLayout.SetNoConstraint)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_sidebar())

        content = QWidget()
        content.setObjectName("contentContainer")
        content_layout = QVBoxLayout(content)
        content_layout.setSizeConstraint(QLayout.SetNoConstraint)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.topbar = self._build_topbar()
        content_layout.addWidget(self.topbar)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        self.page_stack.setMinimumSize(0, 0)
        self.page_stack.setAttribute(Qt.WA_StyledBackground, True)
        content_layout.addWidget(self.page_stack, 1)

        root_layout.addWidget(content, 1)

        self._startup_checkpoint("Memuat Overview…")
        self.dashboard = DashboardPage(self)
        self._startup_checkpoint("Memuat alur dataset…")
        self.upload_page = UploadPage(self)
        self.dataset_search_page = DatasetSearchPage(self)
        self.dataset_preview_page = DatasetPreviewPage(self)
        self.datasets_tabs = QTabWidget()
        self.datasets_tabs.setObjectName("datasetsTabs")
        self.datasets_tabs.setAttribute(Qt.WA_StyledBackground, True)
        self.datasets_tabs.addTab(self.upload_page, "Unggah dari komputer")
        self.datasets_tabs.addTab(self.dataset_search_page, "Cari online")
        self.datasets_tabs.addTab(self.dataset_preview_page, "Pratinjau")
        self.datasets_tabs.tabBar().setVisible(False)
        self.datasets_tab_pages = self.datasets_tabs.findChild(QStackedWidget)
        if self.datasets_tab_pages is not None:
            self.datasets_tab_pages.setObjectName("datasetsTabPages")
            self.datasets_tab_pages.setAttribute(Qt.WA_StyledBackground, True)
        self.datasets_page = QWidget()
        self.datasets_page.setObjectName("datasetsPage")
        self.datasets_page.setAttribute(Qt.WA_StyledBackground, True)
        datasets_layout = QVBoxLayout(self.datasets_page)
        datasets_layout.setContentsMargins(0, 0, 0, 0)
        datasets_layout.setSpacing(0)
        datasets_layout.addWidget(self.datasets_tabs)

        self._startup_checkpoint("Menyiapkan analisis…")
        self.analysis_page = None
        self._analysis_build_scheduled = False
        self._analysis_placeholder = QWidget()
        self._analysis_placeholder.setObjectName("analysisLoadingPlaceholder")
        analysis_placeholder_layout = QVBoxLayout(self._analysis_placeholder)
        analysis_placeholder_layout.setContentsMargins(32, 32, 32, 32)
        analysis_placeholder_layout.addStretch(1)
        analysis_placeholder_message = QLabel(
            "Dataset Analysis akan disiapkan saat dibuka."
        )
        analysis_placeholder_message.setObjectName("emptyState")
        analysis_placeholder_message.setAlignment(Qt.AlignCenter)
        analysis_placeholder_layout.addWidget(analysis_placeholder_message)
        analysis_placeholder_layout.addStretch(1)
        self._startup_checkpoint("Analisis siap; menyiapkan halaman lain…")
        self.editor_analysis_page = None
        self._editor_analysis_build_scheduled = False
        self._editor_analysis_placeholder = QWidget()
        self._editor_analysis_placeholder.setObjectName(
            "editorAnalysisLoadingPlaceholder"
        )
        placeholder_layout = QVBoxLayout(self._editor_analysis_placeholder)
        placeholder_layout.setContentsMargins(32, 32, 32, 32)
        placeholder_layout.addStretch(1)
        placeholder_message = QLabel("Editor Analysis akan disiapkan saat dibuka.")
        placeholder_message.setObjectName("emptyState")
        placeholder_message.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(placeholder_message)
        placeholder_layout.addStretch(1)
        self.node_editor_page = None
        self._node_editor_build_scheduled = False
        if NODE_EDITOR_UI_ENABLED:
            self._node_editor_placeholder = QWidget()
            self._node_editor_placeholder.setObjectName(
                "nodeEditorLoadingPlaceholder"
            )
            node_placeholder_layout = QVBoxLayout(self._node_editor_placeholder)
            node_placeholder_layout.setContentsMargins(32, 32, 32, 32)
            node_placeholder_layout.addStretch(1)
            node_placeholder_message = QLabel(
                "Node Editor akan disiapkan saat dibuka."
            )
            node_placeholder_message.setObjectName("emptyState")
            node_placeholder_message.setAlignment(Qt.AlignCenter)
            node_placeholder_layout.addWidget(node_placeholder_message)
            node_placeholder_layout.addStretch(1)
        self._startup_checkpoint("Menyiapkan tools riset…")
        # These tools contain large tables, charts, and report widgets. Keep
        # lightweight placeholders in the stack and build each tool only when
        # the user opens it.
        self.ml_page = None
        self.research_page = None
        self.papers_tool_page = None
        self.landscape_tool_page = None
        self.gap_tool_page = None
        self.research_insights_page = None
        self.report_tool_page = None
        self._ml_page_build_scheduled = False
        self._research_page_build_scheduled = False
        self._insights_page_build_scheduled = False
        self._report_page_build_scheduled = False

        def deferred_page(object_name, message):
            page = QWidget()
            page.setObjectName(object_name)
            layout = QVBoxLayout(page)
            layout.setContentsMargins(32, 32, 32, 32)
            layout.addStretch(1)
            label = QLabel(message)
            label.setObjectName("emptyState")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            layout.addStretch(1)
            return page

        self._ml_placeholder = deferred_page(
            "mlLoadingPlaceholder", "ML Intelligence akan disiapkan saat dibuka."
        )
        self._research_placeholder = deferred_page(
            "researchLoadingPlaceholder", "Academic Research akan disiapkan saat dibuka."
        )
        self._insights_placeholder = deferred_page(
            "insightsLoadingPlaceholder", "Research Insights akan disiapkan saat dibuka."
        )
        self._report_placeholder = deferred_page(
            "reportLoadingPlaceholder", "Research Report akan disiapkan saat dibuka."
        )
        self._startup_checkpoint("Menyiapkan halaman bantuan…")
        self.about_page = AboutPage()
        self.copyright_page = CopyrightPage()
        self.help_page = HelpPage(self)

        self.pages = {
            "dashboard": self.dashboard,
            "dataset": self.datasets_page,
            "dataset_search": self.datasets_page,
            "analysis": self._analysis_placeholder,
            "editor_analysis": self._editor_analysis_placeholder,
            "ml": self._ml_placeholder,
            "research": self._research_placeholder,
            "papers": self._research_placeholder,
            "insights": self._insights_placeholder,
            # Landscape and gap are tabs inside Research Insights. Reusing
            # their parent page here keeps Qt from pulling the tab widgets
            # out into separate entries in the main page stack.
            "landscape": self._insights_placeholder,
            "gap": self._insights_placeholder,
            "report": self._report_placeholder,
            "about": self.about_page,
            "copyright": self.copyright_page,
            "help": self.help_page,
        }
        if NODE_EDITOR_UI_ENABLED:
            self.pages["node_editor"] = self._node_editor_placeholder

        self.page_hosts = {}
        for page in self.pages.values():
            if id(page) in self.page_hosts:
                continue
            host = page
            if not page.findChildren(QScrollArea):
                host = QScrollArea()
                host.setObjectName("responsivePageScroll")
                host.setWidgetResizable(True)
                host.setFrameShape(QFrame.NoFrame)
                host.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                host.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                host.setWidget(page)
            page.setMinimumWidth(0)
            page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            host.setProperty("pageWidget", True)
            self.page_hosts[id(page)] = host
            self.page_stack.addWidget(host)

        for scroll_area in self.findChildren(QScrollArea):
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for table in self.findChildren(QTableWidget):
            table.setTextElideMode(Qt.ElideRight)

        self._page_titles = {
            "dashboard": "Overview",
            "dataset": "Datasets",
            "dataset_search": "Dataset Search",
            "analysis": "Dataset Analysis",
            "editor_analysis": "Editor Analysis",
            "ml": "ML Intelligence",
            "research": "Academic Research",
            "papers": "Papers",
            "insights": "Research Insights",
            "landscape": "Research Landscape",
            "gap": "Research Gap",
            "report": "Research Report",
            "about": "Tentang DORA",
            "copyright": "Credits",
            "help": "Bantuan",
        }
        if NODE_EDITOR_UI_ENABLED:
            self._page_titles["node_editor"] = "Node Editor"

        for widget in (root, content, self.page_stack, self.datasets_page, self.datasets_tabs, self.datasets_tab_pages):
            if widget is None:
                continue
            self._apply_canvas_background(widget)
        seen_page_ids = set()
        for page in self.pages.values():
            if id(page) not in seen_page_ids:
                seen_page_ids.add(id(page))
                self._apply_canvas_background(page)
        self._startup_checkpoint("Merapikan navigasi…")
        for scroll_area in self.findChildren(QScrollArea):
            self._apply_canvas_background(scroll_area)
            self._apply_canvas_background(scroll_area.viewport())

        self._show_page("dashboard")

    # =========================================================
    # SIDEBAR
    # =========================================================

    def _build_sidebar(self):
        sidebar_container = QWidget()
        sidebar_container.setObjectName("sidebarContainer")
        container_layout = QVBoxLayout(sidebar_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        sidebar_container.setFixedWidth(418)
        self.sidebar_container = sidebar_container

        sidebar_scroll = QScrollArea()
        self.sidebar_scroll = sidebar_scroll
        sidebar_scroll.setObjectName("sidebarScroll")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        self.sidebar_layout = layout
        layout.setContentsMargins(11, 70, 14, 24)
        layout.setSpacing(0)

        brand = QHBoxLayout()
        self.sidebar_brand_layout = brand
        brand.setSpacing(18)
        brand.setContentsMargins(22, 0, 0, 0)
        logo = QLabel("D")
        self.sidebar_logo = logo
        logo.setObjectName("brandLogo")
        logo.setFixedSize(82, 82)
        logo.setAlignment(Qt.AlignCenter)
        logo_path = _resource_path("assets/logo.jpg")
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path)).copy(275, 130, 560, 650)
            if not pixmap.isNull():
                ratio = max(1.0, logo.devicePixelRatioF())
                logo_pixmap = pixmap.scaled(
                    round(82 * ratio),
                    round(82 * ratio),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                logo_pixmap.setDevicePixelRatio(ratio)
                logo.setPixmap(logo_pixmap)

        brand_text_container = QWidget()
        self.brand_text_container = brand_text_container
        brand_text = QVBoxLayout(brand_text_container)
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(2)
        title = QLabel("DORA")
        self.sidebar_brand_title = title
        title.setObjectName("brandTitle")
        subtitle = QLabel("Data Overview & Research Assistant")
        self.sidebar_brand_subtitle = subtitle
        subtitle.setObjectName("brandSubtitle")
        subtitle.setWordWrap(True)
        brand_text_container.setMaximumWidth(276)
        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)
        brand.addWidget(logo)
        brand.addWidget(brand_text_container, 1)
        layout.addLayout(brand)
        layout.addSpacing(45)

        self.nav_buttons = {}
        self.nav_text_widgets = []
        self.sidebar_section_labels = []
        self.sidebar_group_cards = []
        self.sidebar_group_layouts = []
        self._add_nav_section(layout, "RUANG KERJA")
        layout.itemAt(layout.count() - 1).widget().setFixedHeight(32)
        layout.addSpacing(2)
        self._add_nav_group(layout, 238, 15, 8, (
            ("dataset_search", "dataset-search.svg", "Dataset Search"),
            ("dataset", "dataset.svg", "Dataset"),
            ("analysis", "analysis.svg", "Analysis"),
            ("ml", "ml-intelligence.svg", "ML Intelligence"),
            ("dashboard", "overview.svg", "Overview"),
        ))
        layout.addSpacing(8)
        self._add_nav_section(layout, "RISET & PUBLIKASI")
        layout.itemAt(layout.count() - 1).widget().setFixedHeight(32)
        layout.addSpacing(15)
        self._add_nav_group(layout, 152, 6, 17, (
            ("research", "academic-research.svg", "Academic Research"),
            ("gap", "research-gap.svg", "Research Gap"),
            ("report", "research-report.svg", "Research Report"),
        ))
        layout.addSpacing(11)
        self._add_nav_section(layout, "ALAT TAMBAHAN")
        layout.itemAt(layout.count() - 1).widget().setFixedHeight(32)
        layout.addSpacing(2)
        optional_tools = [
            ("editor_analysis", "editor-analysis.svg", "Editor Analysis"),
        ]
        if NODE_EDITOR_UI_ENABLED:
            optional_tools.append(("node_editor", "node-editor.svg", "Node Editor"))
        optional_tools_height = 104 if NODE_EDITOR_UI_ENABLED else 62
        self._add_nav_group(layout, optional_tools_height, 8, 8, optional_tools)
        layout.addStretch()

        footer = QWidget()
        footer.setObjectName("overviewSidebarFooter")
        footer.setFixedHeight(136)
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(41, 0, 0, 0)
        footer_layout.setSpacing(6)
        self.sidebar_footer_layout = footer_layout

        self.sidebar_footer_buttons = {}
        for key, label, icon_name, width, callback in (
            ("about", "Tentang", "overview-about.svg", 116, self.open_about),
            ("copyright", "Credits", "copyright.svg", 116, self.open_copyright),
            ("help", "Bantuan", "help.svg", 104, self.open_help),
        ):
            button = QPushButton(label)
            button.setObjectName("overviewSidebarLink")
            button.setProperty("kind", key)
            button.setProperty("labelText", label)
            button.setProperty("preferredWidth", width)
            button.setToolTip(label)
            button.setIcon(QIcon(str(_resource_path(Path("assets") / "icons" / icon_name))))
            button.setIconSize(QSize(24, 24))
            button.setFixedSize(width, 40)
            button.setCursor(Qt.PointingHandCursor)
            apply_card_shadow(button)
            button.clicked.connect(callback)
            footer_layout.addWidget(button)
            self.sidebar_footer_buttons[key] = button

        self.sidebar_copyright_label = QLabel("© 2026 DORA\nBuilt by Meks Studio")
        self.sidebar_copyright_label.setObjectName("sidebarCopyrightLabel")
        self.sidebar_copyright_label.hide()
        footer_layout.addWidget(self.sidebar_copyright_label)

        self.sidebar_footer = footer

        sidebar_scroll.setWidget(sidebar)
        container_layout.addWidget(sidebar_scroll, 1)
        container_layout.addWidget(footer)
        return sidebar_container

    def _add_nav_group(self, parent_layout, height, top_margin, bottom_margin, entries):
        card = QFrame()
        card.setObjectName("sidebarMenuCard")
        card.setFixedHeight(height)
        group_layout = QVBoxLayout(card)
        group_layout.setContentsMargins(41, top_margin, 9, bottom_margin)
        group_layout.setSpacing(0)
        self.sidebar_group_cards.append(card)
        self.sidebar_group_layouts.append(group_layout)
        for key, icon, text in entries:
            button = self._add_nav_button(group_layout, key, icon, text)
            button.setFixedHeight(28)
        apply_card_shadow(card)
        parent_layout.addWidget(card)

    def _add_nav_section(self, layout, title):
        label = QLabel(title)
        label.setObjectName("sectionLabel")
        self.sidebar_section_labels.append(label)
        layout.addWidget(label)

    def _add_nav_button(self, layout, key, icon, text, enabled=True, tool_item=False):
        button = QPushButton()
        button.setObjectName("navButton")
        button.setEnabled(enabled)
        button.setProperty("toolItem", bool(tool_item))
        button.setCursor(
            Qt.PointingHandCursor if enabled else Qt.ArrowCursor
        )

        row = QHBoxLayout(button)
        row.setContentsMargins(5, 0, 0, 0)
        row.setSpacing(25)

        icon_label = QLabel()
        icon_label.setObjectName("navIcon")
        icon_label.setFixedSize(30, 30)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_path = _resource_path(Path("assets") / "icons" / icon)
        button.setProperty("iconPath", str(icon_path))
        if icon_path.exists():
            icon_size = {
                "dataset_search": (24, 24),
                "dataset": (30, 30),
                "analysis": (24, 24),
                "editor_analysis": (24, 24),
                "ml": (24, 24),
                "dashboard": (26, 26),
                "research": (30, 24),
                "papers": (30, 30),
                "landscape": (30, 30),
                "gap": (30, 30),
                "report": (30, 30),
            }.get(key, (24, 24))
            icon_label.setPixmap(
                render_icon_pixmap(
                    icon_path, QSize(*icon_size), icon_label.devicePixelRatioF()
                )
            )

        text_label = QLabel(text)
        text_label.setObjectName("navText")
        self.nav_text_widgets.append(text_label)
        button.setToolTip(text)

        row.addWidget(icon_label)
        row.addWidget(text_label)
        row.addStretch()

        layout.addWidget(button)
        self.nav_buttons[key] = button

        callbacks = {
            "dashboard": self.open_dashboard,
            "dataset": self.open_upload_page,
            "dataset_search": self.open_dataset_search_page,
            "analysis": self.open_analysis_page,
            "editor_analysis": self.open_editor_analysis_page,
            "node_editor": self.open_node_editor_page,
            "ml": self.open_ml_page,
            "research": self.open_research_page,
            "papers": self.open_papers_tool,
            "landscape": self.open_landscape_tool,
            "gap": self.open_gap_tool,
            "report": self.open_report_tool,
            "about": self.open_about,
        }

        if key in callbacks:
            button.clicked.connect(callbacks[key])
        return button

    # =========================================================
    # TOPBAR
    # =========================================================

    def _build_topbar(self):
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(124)
        overview_frame = QFrame(topbar)
        overview_frame.setObjectName("overviewTopbarFrame")
        overview_frame.hide()
        self.overview_topbar_frame = overview_frame
        topbar.installEventFilter(self)

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(34, 4, 127, 0)
        layout.setSpacing(13)
        self.topbar_layout = layout
        self.sidebar_toggle_button = QToolButton()
        menu_glyph = QPixmap(48, 48)
        menu_glyph.fill(Qt.transparent)
        painter = QPainter(menu_glyph)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#2563A6"), 4, Qt.SolidLine, Qt.RoundCap))
        for y in (12, 24, 36):
            painter.drawLine(9, y, 39, y)
        painter.end()
        self.sidebar_toggle_button.setIcon(QIcon(menu_glyph))
        self.sidebar_toggle_button.setIconSize(QSize(18, 18))
        self.sidebar_toggle_button.setObjectName("sidebarToggleButton")
        self.sidebar_toggle_button.setAccessibleName("Tampilkan atau sembunyikan sidebar")
        self.sidebar_toggle_button.setToolTip("Tampilkan atau sembunyikan sidebar")
        self.sidebar_toggle_button.setCheckable(True)
        self.sidebar_toggle_button.setChecked(True)
        self.sidebar_toggle_button.clicked.connect(self._toggle_sidebar)
        self.sidebar_toggle_button.setFixedSize(34, 34)
        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("topbarTitle")
        self.page_title.hide()
        layout.addStretch()
        version = QLabel("Metriks Studio")
        version.setObjectName("topbarVersion")
        self.topbar_version = version
        version_container = QWidget()
        self.topbar_version_container = version_container
        version_layout = QHBoxLayout(version_container)
        version_layout.setContentsMargins(0, 0, 0, 0)
        version_layout.setSpacing(0)
        version_layout.addWidget(version)
        layout.addWidget(version_container)
        return topbar

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sidebar = getattr(self, "sidebar_container", None)
        if sidebar is None:
            return
        self._pending_chrome_size = event.size()
        if not self._page_chrome_sync_scheduled:
            self._page_chrome_sync_scheduled = True
            QTimer.singleShot(0, self._flush_page_chrome_sync)

    def _flush_page_chrome_sync(self):
        self._page_chrome_sync_scheduled = False
        size = getattr(self, "_pending_chrome_size", None)
        if size is not None and hasattr(self, "sidebar_container"):
            self._sync_page_chrome(size.width(), size.height())

    def eventFilter(self, watched, event):
        watched_is_window_widget = isinstance(watched, QWidget) and (
            watched is self or self.isAncestorOf(watched)
        )
        if (
            event.type() == QEvent.ChildAdded
            and getattr(self, "_initial_widget_scale_applied", False)
            and isinstance(event.child(), QWidget)
            and isinstance(watched, QWidget)
            and watched_is_window_widget
        ):
            self._pending_scale_roots.append(event.child())
            if not self._scale_refresh_scheduled:
                self._scale_refresh_scheduled = True
                QTimer.singleShot(0, self._refresh_new_widget_scale)
        if (
            event.type() == QEvent.Wheel
            and event.modifiers() & Qt.ControlModifier
            and watched_is_window_widget
        ):
            if event.angleDelta().y() > 0:
                self.scale_manager.zoom_in()
            elif event.angleDelta().y() < 0:
                self.scale_manager.zoom_out()
            return True
        if (
            watched is getattr(self, "topbar", None)
            and event.type() == QEvent.Resize
            and getattr(self, "_current_page_key", "dashboard") == "dashboard"
        ):
            self.overview_topbar_frame.setGeometry(
                self.scale_manager.px(36), self.scale_manager.px(16),
                max(0, watched.width() - self.scale_manager.px(25)),
                self.scale_manager.px(130),
            )
        return super().eventFilter(watched, event)

    def _sync_page_chrome(self, window_width, window_height):
        page_key = getattr(self, "_current_page_key", "dashboard")
        overview_mode = page_key == "dashboard"
        about_mode = page_key == "about"
        analysis_mode = page_key == "analysis"
        # Keep one shared navigation shell across routes. Individual page content can
        # use its own responsive layout, but the sidebar should not change model.
        page4_shell_mode = True
        scale = self.scale_manager.get_scale()
        analysis_frame = analysis_mode and window_width >= 1850 and window_height >= 1000
        analysis_px = (lambda value: int(value)) if analysis_frame else self.scale_manager.px
        narrow_window = window_width / max(scale, 0.01) < 900
        manual_hidden = self._sidebar_manual_state is True
        compact = (
            narrow_window
            if self._sidebar_manual_state is None
            else False
        )
        sidebar_width = (
            self.scale_manager.px(190)
            if compact
            else self.scale_manager.px(178) if page4_shell_mode
            else analysis_px(237) if analysis_mode
            else 418 if about_mode and window_width >= self.scale_manager.px(1366)
            else min(self.scale_manager.px(418), max(self.scale_manager.px(240), round(window_width * 0.23)))
        )
        if manual_hidden:
            # Hiding the widget removes the complete navigation rail from the
            # root layout, allowing the page stack to reclaim its full width.
            self.sidebar_container.hide()
            if self.sidebar_container.width() != 0:
                self.sidebar_container.setFixedWidth(0)
        else:
            if self.sidebar_container.width() != sidebar_width:
                self.sidebar_container.setFixedWidth(sidebar_width)
            self.sidebar_container.show()
        if hasattr(self, "sidebar_menu_action"):
            self.sidebar_menu_action.blockSignals(True)
            self.sidebar_menu_action.setChecked(not manual_hidden)
            self.sidebar_menu_action.blockSignals(False)
        if hasattr(self, "sidebar_toggle_button"):
            self.sidebar_toggle_button.blockSignals(True)
            # In automatic compact mode, leave the hamburger unchecked so a
            # click expands the rail; an explicit hide also stays unchecked.
            self.sidebar_toggle_button.setChecked(not compact and not manual_hidden)
            self.sidebar_toggle_button.blockSignals(False)
            toggle_size = self.scale_manager.px(34)
            if self.sidebar_toggle_button.size() != QSize(toggle_size, toggle_size):
                self.sidebar_toggle_button.setFixedSize(toggle_size, toggle_size)
        if page4_shell_mode:
            sidebar_margins = (8 if compact else 12, 24, 8 if compact else 16, 12)
        elif analysis_mode:
            sidebar_margins = (8, 22, 8, 16)
        else:
            sidebar_margins = (
                8 if compact else 11,
                20 if compact else 70,
                8 if compact else 14,
                16 if compact else 44,
            )
        self.sidebar_layout.setContentsMargins(
            *((self.scale_manager.px(value) if page4_shell_mode else analysis_px(value)) for value in sidebar_margins)
        )
        self.sidebar_logo.setFixedSize(
            self.scale_manager.px(42 if compact else 58),
            self.scale_manager.px(42 if compact else 58),
        )
        self.sidebar_brand_layout.setContentsMargins(
            analysis_px(8 if analysis_mode and not compact else 22) if not page4_shell_mode else 0,
            0,
            0,
            0,
        )
        self.sidebar_brand_layout.setSpacing(
            analysis_px(10 if analysis_mode and not compact else 18) if not page4_shell_mode else self.scale_manager.px(9)
        )
        brand_spacer = self.sidebar_layout.itemAt(1)
        if brand_spacer is not None and brand_spacer.spacerItem() is not None:
                brand_spacer.spacerItem().changeSize(
                0,
                self.scale_manager.px(8 if page4_shell_mode else 16),
                QSizePolicy.Minimum,
                QSizePolicy.Fixed,
            )
        for group_index, (group, group_layout) in enumerate(zip(self.sidebar_group_cards, self.sidebar_group_layouts)):
            if page4_shell_mode:
                group_layout.setContentsMargins(0, 0, 0, 0)
                group_layout.setSpacing(self.scale_manager.px(3))
                group.setFixedHeight(
                    self.scale_manager.px((180, 92, 104)[min(group_index, 2)])
                )
            else:
                group_layout.setContentsMargins(
                    analysis_px(4 if compact else (8 if analysis_mode else 35 if about_mode else 41)),
                    analysis_px(10 if compact else (8 if analysis_mode else 15 if group_index == 0 else 6)),
                    analysis_px(4 if compact else (8 if analysis_mode else 15 if about_mode else 9)),
                    analysis_px(10 if compact else (8 if analysis_mode else 8 if group_index == 0 else 17)),
                )
                if not page4_shell_mode:
                    group.setFixedHeight(self.scale_manager.px((238, 152, 104)[min(group_index, 2)]))
            changed = False
            for name, value in (("compact", compact), ("figmaAnalysis", False), ("figmaPage4", page4_shell_mode)):
                if group.property(name) != value:
                    group.setProperty(name, value)
                    changed = True
            if changed:
                group.style().unpolish(group)
                group.style().polish(group)
        for button in getattr(self, "nav_buttons", {}).values():
            if page4_shell_mode:
                button.setFixedHeight(self.scale_manager.px(28))
            row = button.layout()
            if row is not None:
                row.setContentsMargins(
                    self.scale_manager.px(11) if page4_shell_mode else 3 if analysis_mode else 11 if about_mode else 5,
                    0,
                    0,
                    0,
                )
                row.setSpacing(self.scale_manager.px(8) if page4_shell_mode else analysis_px(8 if analysis_mode else 25))
                icon = button.findChild(QLabel, "navIcon")
                if icon is not None:
                    icon.setAttribute(Qt.WA_TransparentForMouseEvents, page4_shell_mode)
                    icon_size = self.scale_manager.px(20) if page4_shell_mode else analysis_px(30)
                    icon.setFixedSize(icon_size, icon_size)
                    icon_path = button.property("iconPath")
                    if icon_path and Path(icon_path).exists():
                        active_tint = "#0861EF" if button.property("active") else "#24344D"
                        if (
                            icon.property("renderedTint") != active_tint
                            or icon.property("renderedIconSize") != icon_size
                        ):
                            icon.setPixmap(
                                render_icon_pixmap(
                                    icon_path,
                                    icon_size - 1,
                                    icon.devicePixelRatioF(),
                                    tint=active_tint,
                                )
                            )
                            icon.setProperty("renderedTint", active_tint)
                            icon.setProperty("renderedIconSize", icon_size)
                text_label = button.findChild(QLabel, "navText")
                if text_label is not None:
                    text_label.setAttribute(Qt.WA_TransparentForMouseEvents, page4_shell_mode)
        if page4_shell_mode:
            for section_label in self.sidebar_section_labels:
                section_label.setFixedHeight(self.scale_manager.px(24))
        sidebar = self.sidebar_scroll.widget()
        sidebar_changed = False
        for name, value in (("figmaAbout", False), ("figmaAnalysis", False), ("figmaPage4", page4_shell_mode)):
            if sidebar.property(name) != value:
                sidebar.setProperty(name, value)
                sidebar_changed = True
        if sidebar_changed:
            sidebar.style().unpolish(sidebar)
            sidebar.style().polish(sidebar)
        for widget in (
            self.sidebar_brand_title,
            self.sidebar_brand_subtitle,
            self.sidebar_logo,
            *self.sidebar_section_labels,
            *self.nav_text_widgets,
            *self.nav_buttons.values(),
        ):
            changed = False
            for name, value in (("figmaAnalysis", False), ("figmaPage4", page4_shell_mode)):
                if widget.property(name) != value:
                    widget.setProperty(name, value)
                    changed = True
            if changed:
                widget.style().unpolish(widget)
                widget.style().polish(widget)
        if page4_shell_mode:
            logo_path = _resource_path("assets/logo.jpg")
            logo_key = (
                self.sidebar_logo.width(),
                self.sidebar_logo.height(),
                round(self.sidebar_logo.devicePixelRatioF(), 2),
            )
            if logo_path.exists() and getattr(self, "_sidebar_logo_render_key", None) != logo_key:
                pixmap = QPixmap(str(logo_path)).copy(275, 130, 560, 650)
                if not pixmap.isNull():
                    ratio = max(1.0, self.sidebar_logo.devicePixelRatioF())
                    logo_pixmap = pixmap.scaled(
                        round(self.sidebar_logo.width() * ratio),
                        round(self.sidebar_logo.height() * ratio),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    logo_pixmap.setDevicePixelRatio(ratio)
                    self.sidebar_logo.setPixmap(logo_pixmap)
                    self._sidebar_logo_render_key = logo_key
            section_names = ("RUANG KERJA", "RISET & PUBLIKASI", "ALAT TAMBAHAN")
            for label, text in zip(self.sidebar_section_labels, section_names):
                label.setText(text)
            self._reorder_page4_workspace_navigation(True)
        else:
            section_names = ("RUANG KERJA", "RISET & PUBLIKASI", "ALAT TAMBAHAN")
            for label, text in zip(self.sidebar_section_labels, section_names):
                label.setText(text)
            self._reorder_page4_workspace_navigation(False)
        self.brand_text_container.setVisible(True)
        for label in self.nav_text_widgets:
            label.setVisible(not compact)
        for label in self.sidebar_section_labels:
            label.setVisible(not compact)

        compact_topbar = window_width / max(scale, 0.01) < 850
        base_topbar_height = (
            146 if overview_mode else 124 if about_mode
            else 78 if window_height / max(scale, 0.01) < 620 else 124
        )
        self.topbar.setFixedHeight(self.scale_manager.px(base_topbar_height))
        # The menu bar and status bar stay in the same place across routes. The
        # page title and navigation controls live in the page/sidebar now.
        self.topbar.setVisible(False)
        self.menuBar().setVisible(True)
        self.statusBar().setVisible(True)
        self.topbar_layout.setContentsMargins(
            self.scale_manager.px(12 if compact_topbar else 34),
            0 if compact_topbar else self.scale_manager.px(4),
            self.scale_manager.px(12 if compact_topbar else 127),
            0,
        )
        self.topbar_version_container.setVisible(False)
        self.sidebar_footer.setVisible(True)
        self.sidebar_footer_buttons["about"].setVisible(True)
        self.sidebar_copyright_label.setVisible(False)
        if page4_shell_mode:
            help_button = self.sidebar_footer_buttons["help"]
            self.sidebar_footer_layout.removeWidget(help_button)
            self.sidebar_footer_layout.insertWidget(0, help_button)
            self.sidebar_footer_layout.setContentsMargins(self.scale_manager.px(31), 0, 0, 0)
        else:
            help_button = self.sidebar_footer_buttons["help"]
            self.sidebar_footer_layout.removeWidget(help_button)
            self.sidebar_footer_layout.insertWidget(2, help_button)
            self.sidebar_footer_layout.setContentsMargins(
                analysis_px(4 if compact else (16 if analysis_mode else 35 if about_mode else 41)),
                0,
                0,
                0,
            )
        for key, button in self.sidebar_footer_buttons.items():
            button.setProperty("figmaPage4", page4_shell_mode)
            button.setProperty("compact", compact)
            button.setText("" if compact else button.property("labelText"))
            button.setFixedSize(
                analysis_px(40 if compact else button.property("preferredWidth")),
                analysis_px(40),
            )
            button.setIconSize(
                QSize(analysis_px(20), analysis_px(20))
                if compact
                else QSize(analysis_px(24), analysis_px(24))
            )
            button.setProperty("active", page_key == key)
            if page4_shell_mode:
                button.setGraphicsEffect(None)
            else:
                apply_card_shadow(button)
            button.style().unpolish(button)
            button.style().polish(button)
        self.sidebar_footer_layout.setSpacing(analysis_px(4 if compact else 6))
        self.sidebar_footer.setFixedHeight(
            self.scale_manager.px(164) if page4_shell_mode
            else analysis_px(132 if compact else 136)
        )
        for key, button in self.sidebar_footer_buttons.items():
            if key == "help":
                button.setText("Bantuan")
        self.overview_topbar_frame.setVisible(False)
        self.topbar.setProperty("overviewMode", overview_mode)
        self.topbar.style().unpolish(self.topbar)
        self.topbar.style().polish(self.topbar)
        if overview_mode:
            self.overview_topbar_frame.setGeometry(
                self.scale_manager.px(36), self.scale_manager.px(16),
                max(0, self.topbar.width() - self.scale_manager.px(25)),
                self.scale_manager.px(130),
            )
        analysis_page = getattr(self, "analysis_page", None)
        if analysis_mode and analysis_page is not None:
            analysis_page._refresh_analysis_layout(analysis_page.width())
            QTimer.singleShot(0, self._apply_analysis_frame_geometry)

    def _reorder_page4_workspace_navigation(self, page4_mode: bool):
        """Keep workspace navigation in one shared order across routes."""
        if not getattr(self, "sidebar_group_layouts", None):
            return
        if getattr(self, "_workspace_navigation_mode", None) == page4_mode:
            return
        group_layout = self.sidebar_group_layouts[0]
        original = ("dataset_search", "dataset", "analysis", "ml", "dashboard")
        page4 = (
            "dashboard", "dataset_search", "dataset", "analysis",
            "ml",
        )
        ordered_keys = page4 if page4_mode else original
        buttons = [self.nav_buttons[key] for key in ordered_keys if key in self.nav_buttons]
        for button in buttons:
            group_layout.removeWidget(button)
        for position, button in enumerate(buttons):
            group_layout.insertWidget(position, button)
            button.setFixedHeight(
                self.scale_manager.px(28 if page4_mode else 43)
            )
        if self.sidebar_group_cards:
            self.sidebar_group_cards[0].setFixedHeight(
                self.scale_manager.px(180 if page4_mode else 238)
            )
        self._workspace_navigation_mode = page4_mode

    def _apply_analysis_frame_geometry(self):
        """Reapply target-frame geometry after deferred widget scaling settles."""
        if getattr(self, "_current_page_key", None) != "analysis":
            return
        page = getattr(self, "analysis_page", None)
        if page is None:
            return
        page._refresh_analysis_layout(page.width())

    # =========================================================
    # NAVIGATION
    # =========================================================

    def _show_page(self, key):
        page = self.pages[key]
        target = self.page_hosts.get(id(page), page)
        if target is not self.page_stack.currentWidget():
            self._cancel_page_transition()
            # Keep native widget painting during navigation. Applying an
            # opacity effect to the stacked window forces Qt to rasterize the
            # complete page and causes stutters on chart- and table-heavy views.
            self.page_stack.setCurrentWidget(target)
        self._current_page_key = key
        current_scale = self.scale_manager.get_scale()
        page_scales = getattr(self, "_page_scale_values", {})
        if abs(page_scales.get(id(page), -1.0) - current_scale) >= 0.001:
            stylesheet = getattr(self, "_active_scale_stylesheet", "")
            if not stylesheet:
                stylesheet = self._build_scaled_stylesheet(current_scale)
                self._active_scale_stylesheet = stylesheet
            self._apply_scale_stylesheet(stylesheet, (target,))
            self._apply_widget_scale(current_scale, [target])
            self._apply_active_page_layout(page, current_scale)
            page_scales[id(page)] = current_scale
            self._page_scale_values = page_scales
        if getattr(self, "_startup_restore_complete", False):
            self.scale_manager.settings.setValue("last_page_key", key)
        self.page_title.setText(self._page_titles[key])
        self.set_active_button("dashboard" if key == "about" else key)
        self._sync_page_chrome(self.width(), self.height())

    def _cancel_page_transition(self):
        animation = getattr(self, "_page_transition_animation", None)
        if animation is not None:
            animation.stop()
            animation.deleteLater()
            self._page_transition_animation = None
        if self.page_stack.graphicsEffect() is not None:
            self.page_stack.setGraphicsEffect(None)
        self._page_transition_effect = None

    def open_dashboard(self):
        self.dashboard.refresh_summary()
        self._show_page("dashboard")

    def open_upload_page(self):
        if self.current_dataset is not None:
            # Keep the established dataset preview as the main Dataset view.
            # The spreadsheet-style upload preview remains outside this route.
            self.dataset_preview_page.update_dataset(
                self.current_dataset,
                self._current_filename(),
            )
            self.dataset_preview_page._select_tab(0)
            self.datasets_tabs.setCurrentWidget(self.dataset_preview_page)
        else:
            self.upload_page.update_loaded_state(None, "")
            self.datasets_tabs.setCurrentWidget(self.upload_page)
        self._show_page("dataset")

    def open_datasets_page(self):
        # Open the existing dataset preview when data is loaded, or the upload
        # workflow when the application has no active dataset.
        self.open_upload_page()

    def open_analysis_page(self):
        if self.analysis_page is None:
            self._show_page("analysis")
            if not self._analysis_build_scheduled:
                self._analysis_build_scheduled = True
                QTimer.singleShot(0, self._create_analysis_page)
            return
        if self.current_dataset is not None:
            self.analysis_page.update_dataset(
                self.current_dataset,
                self._current_filename(),
            )
        self._show_page("analysis")

    def _create_analysis_page(self):
        self._analysis_build_scheduled = False
        if self.analysis_page is not None:
            return

        page = AnalysisPage(self)
        self.analysis_page = page
        if self.current_dataset is not None:
            page.update_dataset(self.current_dataset, self._current_filename())
        self._replace_lazy_page("analysis", page)

    def open_editor_analysis_page(self):
        if self.editor_analysis_page is None:
            self._show_page("editor_analysis")
            if not self._editor_analysis_build_scheduled:
                self._editor_analysis_build_scheduled = True
                QTimer.singleShot(0, self._create_editor_analysis_page)
            return
        self.editor_analysis_page.set_dataset(
            self.current_dataset,
            self._current_filename() if self.current_dataset is not None else "",
        )
        self._show_page("editor_analysis")

    def _create_editor_analysis_page(self):
        self._editor_analysis_build_scheduled = False
        if self.editor_analysis_page is not None:
            return

        from app.ui.editor_analysis_page import EditorAnalysisPage

        # Build the Python QWidget wrapper before assigning its parent. Passing
        # MainWindow into QWidget.__init__ can make PySide resolve this Python
        # subclass as plain QWidget, dropping its Python slot metaobject.
        page = EditorAnalysisPage()
        page._main_window = self
        page.set_dataset(
            self.current_dataset,
            self._current_filename() if self.current_dataset is not None else "",
        )

        self.editor_analysis_page = page
        self._replace_lazy_page("editor_analysis", page)

    def _replace_lazy_page(self, key, page):
        placeholder = self.pages[key]
        old_host = self.page_hosts.pop(id(placeholder), placeholder)
        old_index = self.page_stack.indexOf(old_host)
        was_current = self.page_stack.currentWidget() is old_host
        if old_index >= 0:
            self.page_stack.removeWidget(old_host)
        self._page_scale_values.pop(id(placeholder), None)
        self._page_scale_hook_cache.pop(id(placeholder), None)
        self._scale_stylesheet_template_cache.pop(id(placeholder), None)
        self._scale_stylesheet_template_cache.pop(id(old_host), None)

        page.setMinimumWidth(0)
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        page.setProperty("pageWidget", True)
        host = page
        if not page.findChildren(QScrollArea):
            host = QScrollArea()
            host.setObjectName("responsivePageScroll")
            host.setWidgetResizable(True)
            host.setFrameShape(QFrame.NoFrame)
            host.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            host.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            host.setWidget(page)
        host.setProperty("pageWidget", True)
        self._apply_canvas_background(page)
        for page_key, current_page in tuple(self.pages.items()):
            if current_page is placeholder:
                self.pages[page_key] = page
        self.page_hosts[id(page)] = host
        self.page_stack.insertWidget(max(0, old_index), host)
        old_host.deleteLater()
        if was_current:
            self._show_page(key)

    def open_node_editor_page(self):
        if not NODE_EDITOR_UI_ENABLED:
            return
        if self.node_editor_page is None:
            self._show_page("node_editor")
            if not self._node_editor_build_scheduled:
                self._node_editor_build_scheduled = True
                QTimer.singleShot(0, self._create_node_editor_page)
            return
        self.node_editor_page.prepare_for_open()
        self._show_page("node_editor")

    def _create_node_editor_page(self):
        self._node_editor_build_scheduled = False
        if self.node_editor_page is not None:
            return
        from app.ui.node_editor import NodeEditorPage

        page = NodeEditorPage(self)
        page.prepare_for_open()
        self.node_editor_page = page
        self._replace_lazy_page("node_editor", page)

    def _schedule_page_build(self, key, flag_name, builder_name):
        self._show_page(key)
        if getattr(self, flag_name):
            return
        setattr(self, flag_name, True)
        QTimer.singleShot(0, getattr(self, builder_name))

    def _create_ml_page(self):
        self._ml_page_build_scheduled = False
        if self.ml_page is not None:
            return
        page = MLIntelligencePage(self)
        self.ml_page = page
        task_result = self._restored_ml_task
        methods_result = self._restored_ml_methods
        if task_result and methods_result:
            page.last_task_result = task_result
            page.last_method_result = {
                key: value for key, value in methods_result.items()
                if key != "_intelligence_result"
            }
            page.last_intelligence_result = methods_result.get(
                "_intelligence_result"
            ) or {}
            page.display_result(task_result, page.last_method_result)
        self._replace_lazy_page("ml", page)
        self.open_ml_page()

    def _ensure_research_engine(self):
        if self.research_page is not None:
            return self.research_page
        page = ResearchPage(self)
        page.hide()
        if self._restored_research_result:
            page._last_ml_result = dict(self._restored_ml_task or {})
            page.restore_result(self._restored_research_result)
        self.research_page = page
        return page

    def _create_research_page(self):
        self._research_page_build_scheduled = False
        if self.papers_tool_page is not None:
            return
        self._ensure_research_engine()
        page = PapersToolPage(self)
        result = self.research_page.research_result
        if result:
            page.update_from_result(result)
        self.papers_tool_page = page
        requested_key = (
            self._current_page_key
            if self._current_page_key in {"research", "papers"}
            else "research"
        )
        self._replace_lazy_page(requested_key, page)
        if requested_key == "papers":
            self.open_papers_tool()
        else:
            self.open_research_page()

    def _create_insights_page(self):
        self._insights_page_build_scheduled = False
        if self.research_insights_page is not None:
            return
        self._ensure_research_engine()
        landscape_page = ResearchLandscapeToolPage(self)
        gap_page = ResearchGapToolPage(self)
        result = self.research_page.research_result
        if result:
            landscape_page.update_from_result(result)
            gap_page.update_from_result(result)
        else:
            landscape_page.show_empty_state()
            gap_page.show_empty_state()
        self.landscape_tool_page = landscape_page
        self.gap_tool_page = gap_page
        page = ResearchInsightsPage(landscape_page, gap_page)
        self.research_insights_page = page
        requested_key = (
            self._current_page_key
            if self._current_page_key in {"insights", "landscape", "gap"}
            else "insights"
        )
        self._replace_lazy_page(requested_key, page)
        page.tabs.setCurrentWidget(
            gap_page if self._pending_insights_tab == 1 else landscape_page
        )
        self.open_insights_page()
        if requested_key == "landscape":
            self.open_landscape_tool()
        elif requested_key == "gap":
            self.open_gap_tool()

    def _create_report_page(self):
        self._report_page_build_scheduled = False
        if self.report_tool_page is not None:
            return
        self._ensure_research_engine()
        page = ResearchReportToolPage(self)
        self.report_tool_page = page
        self._replace_lazy_page("report", page)
        self.open_report_tool()

    def open_ml_page(self):
        if self.ml_page is None:
            self._schedule_page_build("ml", "_ml_page_build_scheduled", "_create_ml_page")
            return
        if self.current_dataset is None:
            self.ml_page.show_empty_state()
        elif self.ml_page.last_task_result and self.ml_page.last_method_result:
            if not self.ml_page.is_result_displayed():
                self.ml_page.display_result(
                    self.ml_page.last_task_result, self.ml_page.last_method_result
                )
        elif self.ml_page.restore_saved_result():
            pass
        elif self.analysis_result:
            self.ml_page.update_dataset()
        else:
            self.ml_page.show_empty_state()
        self._show_page("ml")

    def open_research_page(self):
        if self.papers_tool_page is None:
            self._schedule_page_build(
                "research", "_research_page_build_scheduled", "_create_research_page"
            )
            return
        if self.current_dataset is None or not self.analysis_result:
            self.research_page.show_empty_state()
        self._show_page("research")

    def open_dataset_search_page(self):
        self.datasets_tabs.setCurrentWidget(self.dataset_search_page)
        self._show_page("dataset_search")

    def open_papers_tool(self):
        if self.papers_tool_page is None:
            self._schedule_page_build(
                "papers", "_research_page_build_scheduled", "_create_research_page"
            )
            return
        if not self.analysis_result or not self.research_page.research_result:
            self.research_page.show_empty_state()
        else:
            self.papers_tool_page.update_from_result(self.research_page.research_result)
        self._show_page("papers")

    def open_insights_page(self):
        if self.research_insights_page is None:
            self._schedule_page_build(
                "insights", "_insights_page_build_scheduled", "_create_insights_page"
            )
            return
        result = self.research_page.research_result
        if result:
            self.landscape_tool_page.update_from_result(result)
            self.gap_tool_page.update_from_result(result)
        else:
            self.landscape_tool_page.show_empty_state()
            self.gap_tool_page.show_empty_state()
        self._show_page("insights")
        self.research_insights_page._on_tab_changed(
            self.research_insights_page.tabs.currentIndex()
        )
        self.research_page.ensure_insights()

    def open_landscape_tool(self):
        self._pending_insights_tab = 0
        if self.research_insights_page is None:
            self._schedule_page_build(
                "landscape", "_insights_page_build_scheduled", "_create_insights_page"
            )
            return
        self.open_insights_page()
        self.research_insights_page.tabs.setCurrentWidget(self.landscape_tool_page)
        self._show_page("landscape")

    def open_gap_tool(self):
        self._pending_insights_tab = 1
        if self.research_insights_page is None:
            self._schedule_page_build(
                "gap", "_insights_page_build_scheduled", "_create_insights_page"
            )
            return
        self.open_insights_page()
        self.research_insights_page.tabs.setCurrentWidget(self.gap_tool_page)
        self._show_page("gap")

    def open_report_tool(self):
        if self.report_tool_page is None:
            self._schedule_page_build(
                "report", "_report_page_build_scheduled", "_create_report_page"
            )
            return
        has_data = (
            self.current_dataset is not None
            or bool(self.analysis_result)
            or bool(getattr(self.ml_page, "last_intelligence_result", {}))
            or bool(getattr(self.research_page, "research_result", {}))
        )
        if not has_data:
            self.report_tool_page.show_empty_state()
        else:
            research_result = getattr(self.research_page, "research_result", {}) or {}
            if research_result.get("insights_status") == "NOT_COMPUTED":
                self._show_page("report")
                self.report_tool_page.status.setText(
                    "Menyiapkan landscape dan research gap sebelum menyusun laporan..."
                )
                self.research_page.ensure_insights()
                return
            self.report_tool_page.generate_preview()
        self._show_page("report")

    def open_about(self):
        self._show_page("about")

    def open_copyright(self):
        self._show_page("copyright")

    def open_help(self):
        self._show_page("help")

    def _current_filename(self):
        if self.current_project_id is not None:
            try:
                project = self.repository.get_project(self.current_project_id)
                if project and project.get("dataset_name"):
                    return project["dataset_name"]
            except Exception:
                pass
        if self.current_file_path:
            try:
                return Path(self.current_file_path).name
            except Exception:
                pass
        return "Dataset"

    def set_active_button(self, key):
        for name, button in self.nav_buttons.items():
            active = name == key
            if button.property("active") == active:
                continue
            button.setProperty("active", active)
            style = button.style()
            style.unpolish(button)
            style.polish(button)
            button.update()
            text_label = button.findChild(QLabel, "navText")
            if text_label is not None:
                text_label.setProperty("active", active)
                text_style = text_label.style()
                text_style.unpolish(text_label)
                text_style.polish(text_label)
                text_label.update()
            icon = button.findChild(QLabel, "navIcon")
            icon_path = button.property("iconPath")
            if icon is not None and icon_path and Path(icon_path).exists():
                icon_size = max(1, min(icon.width(), icon.height()) - 1)
                tint = "#0861EF" if active else "#24344D"
                icon.setPixmap(
                    render_icon_pixmap(
                        icon_path, icon_size, icon.devicePixelRatioF(), tint=tint
                    )
                )
                icon.setProperty("renderedTint", tint)
                icon.setProperty("renderedIconSize", icon_size + 1)

    # =========================================================
    # DATASET STATE
    # =========================================================

    @staticmethod
    def _normalized_dataset_path(path):
        if not path:
            return "in-memory"
        value = str(path)
        if value == "in-memory":
            return value
        try:
            return str(Path(value).resolve()).casefold()
        except OSError:
            return value.casefold()

    def _ensure_current_project(self):
        """Ensure the loaded dataset has its own local SQLite project/version."""
        dataframe = self.current_dataset
        if dataframe is None:
            return None

        dataset_path = str(self.current_file_path) if self.current_file_path else "in-memory"
        project = (
            self.repository.get_project(self.current_project_id)
            if self.current_project_id is not None
            else None
        )
        same_dataset = bool(
            project
            and self._normalized_dataset_path(project.get("dataset_path"))
            == self._normalized_dataset_path(dataset_path)
        )
        if project and not same_dataset:
            current_version = self.version_manager.get_current_version(self.current_project_id)
            same_dataset = bool(
                current_version
                and current_version.file_path
                and self._normalized_dataset_path(current_version.file_path)
                == self._normalized_dataset_path(dataset_path)
            )
        if not same_dataset:
            filename = Path(dataset_path).name if dataset_path != "in-memory" else "Dataset"
            self.current_project_id = self.repository.create_project(
                name=filename or "Dataset",
                dataset_name=filename or "Dataset",
                dataset_path=dataset_path,
            )

        if not self.version_manager.has_versions(self.current_project_id):
            if dataset_path != "in-memory" and Path(dataset_path).exists():
                self.version_manager.create_initial_version(
                    self.current_project_id,
                    dataset_path,
                )
            else:
                self.version_manager.create_initial_version(
                    self.current_project_id,
                    dataframe=dataframe,
                )
        self.scale_manager.settings.setValue("active_project_id", self.current_project_id)
        return self.current_project_id

    def _restore_latest_project(self):
        """Start restoring the latest project without delaying the first paint."""
        try:
            preferred_project_id = int(self.scale_manager.settings.value("active_project_id", 0) or 0)
        except (TypeError, ValueError):
            preferred_project_id = 0
        self._project_restore_pending = True
        worker = LatestProjectRestoreWorker(
            self.repository,
            self.version_manager,
            preferred_project_id,
            self,
        )
        self._project_restore_thread = worker
        self.statusBar().showMessage("Memulihkan dataset terakhir di latar belakang…")
        worker.result_ready.connect(self._apply_restored_project)
        worker.finished.connect(self._project_restore_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    @Slot(object)
    def _apply_restored_project(self, snapshot):
        self._project_restore_pending = False
        self.statusBar().clearMessage()
        if self._dataset_session_changed:
            return
        if snapshot is None:
            return
        if snapshot.get("error"):
            print(f"Warning: could not restore local project: {snapshot['error']}")
            return

        project = snapshot["project"]
        project_id = project.get("id")
        dataframe = snapshot["dataframe"]
        state = snapshot["workflow_state"]
        self.current_project_id = project_id
        self.current_dataset = dataframe
        original_path = project.get("dataset_path")
        version = self.version_manager.get_current_version(project_id)
        version_path = version.file_path if version else None
        self.current_file_path = (
            original_path
            if original_path and original_path != "in-memory" and Path(original_path).exists()
            else version_path
        )
        filename = project.get("dataset_name") or project.get("name") or "Dataset"

        self.analysis_result = state.get("analysis") or {}
        self.dashboard.update_dataset(dataframe, filename)
        if self._current_page_key == "dataset":
            self.dataset_preview_page.update_dataset(dataframe, filename)
            self.dataset_preview_page._select_tab(0)
            self.datasets_tabs.setCurrentWidget(self.dataset_preview_page)
        if self.analysis_page is not None:
            self.analysis_page.update_dataset(dataframe, filename)

        task_result = state.get("ml_task") or {}
        methods_result = state.get("ml_methods") or {}
        self._restored_ml_task = task_result
        self._restored_ml_methods = methods_result
        if self.ml_page is not None and task_result and methods_result:
            self.ml_page.last_task_result = task_result
            self.ml_page.last_method_result = {
                key: value for key, value in methods_result.items()
                if key != "_intelligence_result"
            }
            self.ml_page.last_intelligence_result = methods_result.get("_intelligence_result") or {}
            self.ml_page.display_result(task_result, self.ml_page.last_method_result)
        elif self.ml_page is not None and self.analysis_result:
            self.ml_page.update_dataset()

        research_result = state.get("research") or {}
        self._restored_research_result = research_result
        if self.research_page is not None and research_result:
            self.research_page._last_ml_result = dict(task_result or {})
            self.research_page.restore_result(research_result)

        self.dashboard.update_workflow_state(
            has_dataset=True,
            analysis_done=bool(self.analysis_result),
            ml_done=bool(task_result and methods_result),
        )
        self.scale_manager.settings.setValue("active_project_id", project_id)

    @Slot()
    def _project_restore_finished(self):
        self._project_restore_thread = None
        self.statusBar().clearMessage()

    def _restore_last_page(self):
        # Launch into the main workspace every time. The previous route is
        # still saved during shutdown for diagnostics and future preferences,
        # but should not reopen a heavy tool over the Overview on startup.
        self._show_page("dashboard")

    def update_dataset_state(self):
        self._dataset_session_changed = True
        self.analysis_result = {}
        self._restored_ml_task = {}
        self._restored_ml_methods = {}
        self._restored_research_result = {}

        try:
            self._ensure_current_project()
        except Exception as error:
            print(f"Warning: could not initialize local project workflow: {error}")

        filename = self._current_filename()

        self.dashboard.update_dataset(
            self.current_dataset,
            filename,
        )

        self.dataset_preview_page.update_dataset(
            self.current_dataset,
            filename,
        )

        if self.analysis_page is not None:
            self.analysis_page.update_dataset(
                self.current_dataset,
                filename,
            )

        if self.ml_page is not None:
            self.ml_page.show_empty_state()
        if self.research_page is not None:
            self.research_page.show_empty_state()
        if self.landscape_tool_page is not None:
            self.landscape_tool_page.show_empty_state()
        if self.gap_tool_page is not None:
            self.gap_tool_page.show_empty_state()
        if self.report_tool_page is not None:
            self.report_tool_page.show_empty_state()
        self.dashboard.update_workflow_state(
            has_dataset=self.current_dataset is not None,
            analysis_done=False,
            ml_done=False,
        )

    def activate_dataset_version(self, version: int):
        """Switch the shared active dataset to an existing stored version."""
        if self.current_project_id is None:
            raise ValueError("Tidak ada project dataset aktif.")

        dataframe = self.version_manager.load_dataframe(
            self.current_project_id,
            version,
        )
        self.version_manager.set_current_version(
            self.current_project_id,
            version,
        )
        self.current_dataset = dataframe
        self.update_dataset_state()
        if self.analysis_page is None:
            self._create_analysis_page()
        self.analysis_page.run_analysis()

    # =========================================================
    # DATASET ANALYSIS
    # =========================================================

    def prepare_dataset_analysis(self):
        dataframe = self.current_dataset
        if dataframe is None:
            raise ValueError("Belum ada dataset yang dimuat.")

        if self.current_file_path and Path(self.current_file_path).exists():
            if self.current_project_id is None:
                filename = Path(self.current_file_path).name
                self.current_project_id = self.repository.create_project(
                    name=filename,
                    dataset_name=filename,
                    dataset_path=str(self.current_file_path),
                )
            if not self.version_manager.has_versions(self.current_project_id):
                try:
                    self.version_manager.create_initial_version(
                        self.current_project_id, self.current_file_path
                    )
                except Exception as error:
                    print(f"Warning: initial dataset version could not be saved: {error}")
        else:
            if self.current_project_id is None:
                fallback_name = self._current_filename() or "dataset"
                self.current_project_id = self.repository.create_project(
                    name=fallback_name,
                    dataset_name=fallback_name,
                    dataset_path="in-memory",
                )
            if not self.version_manager.has_versions(self.current_project_id):
                try:
                    self.version_manager.create_initial_version(
                        self.current_project_id, dataframe=dataframe
                    )
                except Exception as error:
                    print(f"Warning: in-memory dataset version could not be saved: {error}")

        get_telemetry().track(
            "analysis_started", feature_name="dataset_analysis", status="started"
        )

    def complete_dataset_analysis(self, result, duration_ms):
        self.analysis_result = result
        project_id = self.current_project_id
        if project_id is not None:
            try:
                self.repository.save_workflow_section(project_id, "analysis", result)
                self.repository.save_workflow_section(project_id, "ml_task", None)
                self.repository.save_workflow_section(project_id, "ml_methods", None)
                self.repository.save_workflow_section(project_id, "research", None)
            except Exception as error:
                print(f"Warning: analysis workflow could not be saved: {error}")

        get_telemetry().track(
            "analysis_completed", feature_name="dataset_analysis",
            status="completed", duration_ms=duration_ms,
        )
        self._restored_ml_task = {}
        self._restored_ml_methods = {}
        self._restored_research_result = {}
        if self.ml_page is not None:
            self.ml_page.last_task_result = {}
            self.ml_page.last_method_result = {}
            self.ml_page.last_intelligence_result = {}
            self.ml_page.show_ready_state()
        if self.research_page is not None:
            self.research_page.show_empty_state()
        self.dashboard.update_workflow_state(
            has_dataset=True, analysis_done=True, ml_done=False
        )

    def fail_dataset_analysis(self, duration_ms):
        get_telemetry().track(
            "analysis_failed", feature_name="dataset_analysis",
            status="failed", duration_ms=duration_ms,
        )

    def run_dataset_analysis(self):
        """Synchronous compatibility entry point; the UI uses the worker thread."""
        dataframe = self.current_dataset
        self.prepare_dataset_analysis()
        jobs = {
            "profile": self.profiler.profile,
            "statistics": self.statistics.analyze,
            "missing_values": self.missing_analyzer.analyze,
            "duplicates": self.duplicate_analyzer.analyze,
            "outliers": self.outlier_analyzer.analyze,
            "correlations": self.correlation_analyzer.analyze,
            "fingerprint": self.fingerprint_analyzer.generate,
        }
        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                futures = {name: executor.submit(fn, dataframe) for name, fn in jobs.items()}
                result = {name: future.result() for name, future in futures.items()}
            try:
                result["data_quality"] = self.quality_diagnoser.diagnose(
                    dataframe=dataframe,
                    profile=result.get("profile"),
                    missing_values=result.get("missing_values"),
                    duplicates=result.get("duplicates"),
                    outliers=result.get("outliers"),
                    fingerprint=result.get("fingerprint"),
                )
            except Exception as error:
                print(f"Warning: data quality diagnosis failed: {error}")
                result["data_quality"] = []
            self.complete_dataset_analysis(result, int((time.perf_counter() - started) * 1000))
            return result
        except Exception:
            self.fail_dataset_analysis(int((time.perf_counter() - started) * 1000))
            raise


__all__ = [
    "StatCard",
    "QuickActionCard",
    "PipelineStage",
    "DashboardPage",
    "UploadPage",
    "AnalysisPage",
    "InfoCard",
    "TaskCard",
    "MethodCard",
    "MethodDetailCard",
    "MLIntelligencePage",
    "ResearchPage",
    "PapersToolPage",
    "ResearchLandscapeToolPage",
    "ResearchGapToolPage",
    "ResearchReportToolPage",
    "DatasetSearchPage",
    "MainWindow",
]
