from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html import escape as html_escape
from html.parser import HTMLParser
from pathlib import Path
import re
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

from PySide6.QtCore import Qt, QUrl, QObject, QThread, QTimer, Signal, Slot, QMarginsF, QSize, QPoint, QRectF
from PySide6.QtGui import QIcon, QPixmap, QDesktopServices, QTextDocument, QFont, QFontDatabase, QColor, QPalette, QPageSize, QPageLayout, QPainter
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
)

from app.analyzer.loader import DatasetLoader
from app.analyzer.profiler import DatasetProfiler
from app.analyzer.statistics import DatasetStatistics
from app.analyzer.missing_values import MissingValueAnalyzer
from app.analyzer.missing_values import (
    drop_missing_rows,
    impute_missing_values,
    normalize_missing_values,
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
from app.ml.intelligence import MLIntelligenceEngine
from app.research.intelligence import ResearchIntelligenceEngine
from app.ui.dataset_search_page import DatasetSearchPage
from app.ui.dataset_preview_page import DatasetPreviewPage
from app.ui.editor_analysis_page import EditorAnalysisPage
from app.ui.style_utils import apply_card_shadow
from app.storage import Database, ProjectRepository, DatasetVersionManager
from app.reports import ReportGenerator
from app.core.config import get_data_dir
from app.telemetry import get_telemetry


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

    finished = Signal(str, str)  # (text, source)
    failed = Signal(str)

    def __init__(self, analysis: dict[str, Any]):
        super().__init__()
        self.analysis = analysis

    def run(self):
        try:
            explainer = GeminiDatasetExplainer()
            text, source = explainer.explain_with_source(self.analysis)
            self.finished.emit(text, source)
        except Exception:
            try:
                local_text = LocalAcademicExplainer().explain(self.analysis)
                self.finished.emit(local_text, "local")
            except Exception as local_exc:
                self.failed.emit(str(local_exc))


class DatasetAnalysisWorker(QObject):
    """Run the complete dataset analysis away from the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, analyzers: dict, quality_diagnoser, dataframe):
        super().__init__()
        self.analyzers = analyzers
        self.quality_diagnoser = quality_diagnoser
        self.dataframe = dataframe

    def run(self):
        try:
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

            self.finished.emit(result)
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
            )
            self.finished.emit(result, ml_result)
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

    def __init__(self, gaps):
        super().__init__()
        self.gaps = gaps

    def run(self):
        try:
            explanation, source, reason = ResearchGapExplainer().explain(self.gaps)
            self.finished.emit(explanation, source, reason)
        except Exception as error:
            self.failed.emit(str(error))


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
        self._timer.timeout.connect(self._advance_spinner)
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
            painter.drawText(chart, Qt.AlignCenter, "Run Academic Research to populate this chart")
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
            painter.drawText(chart, Qt.AlignCenter, "No research data yet")
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


class DashboardPage(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()
        self.update_research_result({})

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("dashboardScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("dashboardCanvas")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 24, 30, 28)
        layout.setSpacing(18)

        # ACTIVE DATASET ---------------------------------------
        hero = QFrame()
        hero.setObjectName("dashboardHero")
        hero.setMinimumHeight(186)

        hero_layout = QHBoxLayout(hero)
        self.hero_layout = hero_layout
        hero_layout.setContentsMargins(30, 22, 30, 22)
        hero_layout.setSpacing(24)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(7)
        hero_topline = QHBoxLayout()
        hero_topline.setSpacing(10)
        hero_eyebrow = QLabel("ACTIVE DATASET")
        hero_eyebrow.setObjectName("heroEyebrow")
        hero_topline.addWidget(hero_eyebrow)

        self.dataset_status_badge = QLabel("NO DATASET")
        self.dataset_status_badge.setObjectName("datasetStatusBadge")
        hero_topline.addWidget(self.dataset_status_badge, 0, Qt.AlignVCenter)
        hero_topline.addStretch()
        hero_text.addLayout(hero_topline)

        self.dataset_name_label = QLabel("No dataset loaded")
        self.dataset_name_label.setObjectName("datasetName")
        self.dataset_name_label.setWordWrap(True)
        hero_text.addWidget(self.dataset_name_label)

        self.dataset_description_label = QLabel(
            "Upload a dataset to begin your research workflow."
        )
        self.dataset_description_label.setObjectName("datasetDescription")
        self.dataset_description_label.setWordWrap(True)
        hero_text.addWidget(self.dataset_description_label)

        hero_buttons = QHBoxLayout()
        hero_buttons.setSpacing(10)
        self.analysis_button = QPushButton("Run analysis")
        self.analysis_button.setObjectName("heroPrimaryButton")
        self.analysis_button.setFixedHeight(38)
        self.analysis_button.setCursor(Qt.PointingHandCursor)
        self.analysis_button.clicked.connect(self.main_window.open_analysis_page)
        self.analysis_button.setEnabled(False)
        hero_buttons.addWidget(self.analysis_button)

        self.upload_button = QPushButton("Upload new dataset")
        self.upload_button.setObjectName("heroSecondaryButton")
        self.upload_button.setFixedHeight(38)
        self.upload_button.setCursor(Qt.PointingHandCursor)
        self.upload_button.clicked.connect(self.main_window.open_upload_page)
        hero_buttons.addWidget(self.upload_button)
        hero_buttons.addStretch()
        hero_text.addLayout(hero_buttons)
        hero_text.addStretch(1)
        hero_layout.addLayout(hero_text, 1)

        hero_stats = QHBoxLayout()
        hero_stats.setSpacing(12)
        self.rows_card = StatCard("ROWS", "—", "Records")
        self.columns_card = StatCard("FEATURES", "—", "Columns")
        self.rows_card.setObjectName("heroMetricCard")
        self.columns_card.setObjectName("heroMetricCard")
        hero_stats.addWidget(self.rows_card)
        hero_stats.addWidget(self.columns_card)
        self.hero_stats_layout = hero_stats
        hero_layout.addLayout(hero_stats)
        layout.addWidget(hero)

        # WORKFLOW ---------------------------------------------
        pipeline_header = QHBoxLayout()
        self.pipeline_header_layout = pipeline_header
        pipeline_title = QLabel("Research progress")
        pipeline_title.setObjectName("sectionTitle")
        self.pipeline_description = QLabel("Follow the steps in your research workflow")
        self.pipeline_description.setObjectName("sectionDescription")
        self.pipeline_description.setWordWrap(True)
        pipeline_header.addWidget(pipeline_title)
        pipeline_header.addSpacing(8)
        pipeline_header.addWidget(self.pipeline_description)
        pipeline_header.addStretch()
        layout.addLayout(pipeline_header)

        pipeline_card = QFrame()
        pipeline_card.setObjectName("pipelineCard")
        pipeline_layout = QHBoxLayout(pipeline_card)
        pipeline_layout.setContentsMargins(18, 14, 18, 12)
        pipeline_layout.setSpacing(0)
        self.pipeline_layout = pipeline_layout
        self.pipeline_stages = []
        self.workflow_connectors = []
        for index, title in enumerate(("Dataset", "Analysis", "ML Intelligence", "Academic Research")):
            stage = PipelineStage(index + 1, title, "WAITING")
            self.pipeline_stages.append(stage)
            pipeline_layout.addWidget(stage, 1)
            if index < 3:
                connector = QFrame()
                connector.setObjectName("workflowConnector")
                connector.setFrameShape(QFrame.HLine)
                connector.setFixedHeight(2)
                connector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                self.workflow_connectors.append(connector)
                pipeline_layout.addWidget(connector, 1)
        layout.addWidget(pipeline_card)

        # QUICK ACTIONS -----------------------------------------
        quick_header = QHBoxLayout()
        quick_title = QLabel("Get started")
        quick_title.setObjectName("sectionTitle")
        quick_header.addWidget(quick_title)
        quick_header.addStretch()
        layout.addLayout(quick_header)

        self.actions_layout = QGridLayout()
        self.actions_layout.setHorizontalSpacing(14)
        self.actions_layout.setVerticalSpacing(14)
        self.quick_action_cards = [
            QuickActionCard(chr(0x2191), "Upload dataset", "Import a dataset and start a new project.", self.main_window.open_upload_page),
            QuickActionCard(chr(0x25c7), "Analyze dataset", "Review statistics, missing values, and outliers.", self.main_window.open_analysis_page),
            QuickActionCard(chr(0x2726), "ML Intelligence", "Explore suitable machine learning methods.", self.main_window.open_ml_page),
            QuickActionCard(chr(0x25ce), "Academic Research", "Find related papers and research opportunities.", self.main_window.open_research_page),
        ]
        layout.addLayout(self.actions_layout)
        # RESEARCH SNAPSHOT -------------------------------------
        snapshot_header = QHBoxLayout()
        self.snapshot_header_layout = snapshot_header
        snapshot_title = QLabel("Research overview")
        snapshot_title.setObjectName("sectionTitle")
        snapshot_desc = QLabel("A quick view of your academic research results")
        snapshot_desc.setObjectName("sectionDescription")
        snapshot_desc.setWordWrap(True)
        snapshot_header.addWidget(snapshot_title)
        snapshot_header.addSpacing(8)
        snapshot_header.addWidget(snapshot_desc)
        snapshot_header.addStretch()
        layout.addLayout(snapshot_header)

        self.chart_layout = QGridLayout()
        self.chart_layout.setHorizontalSpacing(12)
        self.chart_layout.setVerticalSpacing(12)
        self.publication_chart = MiniLineChart(
            "Publication Trend",
            "Papers by publication year",
        )
        self.source_chart = MiniBarChart(
            "Paper Sources",
            "Paper candidates returned before deduplication",
        )
        self.metric_chart = MiniBarChart(
            "Research metrics",
            "Relative scale of key outputs",
        )
        self.research_charts = [
            self.publication_chart,
            self.source_chart,
            self.metric_chart,
        ]
        layout.addLayout(self.chart_layout)

        layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)
        self._apply_responsive_layout(self.width())

    @staticmethod
    def _reflow_grid(grid, widgets, columns):
        for column in range(grid.columnCount()):
            grid.setColumnStretch(column, 0)

        for widget in widgets:
            grid.removeWidget(widget)

        for index, widget in enumerate(widgets):
            grid.addWidget(widget, index // columns, index % columns)

        for column in range(columns):
            grid.setColumnStretch(column, 1)

    def _apply_responsive_layout(self, width):
        compact = width < 760
        margins = 16 if compact else 22 if width < 980 else 30
        self.layout().setContentsMargins(margins, 16, margins, 20)

        stacked = width < 760
        self.hero_layout.setDirection(
            QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight
        )
        self.hero_stats_layout.setDirection(
            QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight
        )
        self.pipeline_layout.setDirection(
            QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight
        )
        self.pipeline_header_layout.setDirection(
            QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight
        )
        self.snapshot_header_layout.setDirection(
            QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight
        )
        for connector in self.workflow_connectors:
            connector.setVisible(not stacked)

        action_columns = 4 if width >= 1100 else 2 if width >= 660 else 1
        chart_columns = 3 if width >= 1080 else 2 if width >= 700 else 1
        self._reflow_grid(self.actions_layout, self.quick_action_cards, action_columns)
        self._reflow_grid(self.chart_layout, self.research_charts, chart_columns)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def update_dataset(self, dataframe, filename):
        if dataframe is None:
            return

        self.dataset_name_label.setText(filename)
        self.dataset_description_label.setText(
            "Dataset loaded and ready. Run an analysis to view statistics and data quality insights."
        )
        self.dataset_status_badge.setText("DATASET LOADED")
        self.dataset_status_badge.setProperty("state", "success")
        style = self.dataset_status_badge.style()
        style.unpolish(self.dataset_status_badge)
        style.polish(self.dataset_status_badge)
        self.dataset_status_badge.update()
        self.analysis_button.setEnabled(True)

        self.rows_card.set_value(f"{len(dataframe):,}")
        self.columns_card.set_value(f"{len(dataframe.columns):,}")

        if len(self.pipeline_stages) >= 1:
            self.pipeline_stages[0].set_status("READY", "success")
        if len(self.pipeline_stages) >= 2:
            self.pipeline_stages[1].set_status("NEXT", "active")
        for stage in self.pipeline_stages[2:4]:
            stage.set_status("WAITING", "waiting")
        self._sync_workflow_connectors()

        self.update_research_result({})

    def update_workflow_state(self, has_dataset, analysis_done, ml_done, research_done):
        """Reflect the saved local workflow state in the dashboard pipeline."""
        states = [
            ("READY", "success") if has_dataset else ("UPLOAD", "active"),
            ("READY", "success") if analysis_done else ("NEXT", "active" if has_dataset else "waiting"),
            ("READY", "success") if ml_done else ("WAITING", "waiting"),
            ("READY", "success") if research_done else ("WAITING", "waiting"),
        ]
        for stage, (status, state) in zip(self.pipeline_stages[:4], states):
            stage.set_status(status, state)
        self.analysis_button.setEnabled(bool(has_dataset))
        if research_done:
            self.pipeline_description.setText("All main research steps have saved results")
        elif ml_done:
            self.pipeline_description.setText("ML results are ready; continue with academic research")
        elif analysis_done:
            self.pipeline_description.setText("Analysis is saved; continue with ML Intelligence")
        elif has_dataset:
            self.pipeline_description.setText("Dataset is ready; continue with analysis")
        else:
            self.pipeline_description.setText("Upload a dataset to begin your research workflow")
        self._sync_workflow_connectors()

    def _sync_workflow_connectors(self):
        for index, connector in enumerate(self.workflow_connectors):
            complete = self.pipeline_stages[index].property("state") == "success"
            connector.setProperty("state", "complete" if complete else "pending")
            connector.style().unpolish(connector)
            connector.style().polish(connector)

    def update_research_result(self, result):
        result = result if isinstance(result, dict) else {}
        papers = result.get("papers", [])
        if not isinstance(papers, list):
            papers = []

        # Publication years.
        year_counts = {}
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            year = paper.get("year")
            try:
                year = int(year)
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100:
                year_counts[year] = year_counts.get(year, 0) + 1
        years = sorted(year_counts)
        if years:
            self.publication_chart.set_data(
                years,
                [year_counts[y] for y in years],
            )
        else:
            self.publication_chart.clear_data()

        # Display every supported provider, including zero-result and unavailable
        # providers, so this chart reports participation rather than only sources
        # that happened to survive deduplication.
        search_info = result.get("search", {})
        search_info = search_info if isinstance(search_info, dict) else {}
        academic_search = (result.get("pipeline") or {}).get("academic_search", {})
        academic_search = academic_search if isinstance(academic_search, dict) else {}
        source_counts = search_info.get("source_counts") or search_info.get("sources") or academic_search.get("sources") or {}
        source_counts = source_counts if isinstance(source_counts, dict) else {}
        provider_status = academic_search.get("provider_status", {})
        provider_status = provider_status if isinstance(provider_status, dict) else {}
        providers = ("OpenAlex", "Crossref", "Semantic Scholar", "Europe PMC", "OpenAIRE", "Scopus")
        provider_labels = []
        provider_values = []
        status_text = {
            "NOT_CONFIGURED": "not configured",
            "DISABLED": "disabled",
            "ERROR": "error",
            "RATE_LIMITED": "rate limited",
        }
        for provider in providers:
            try:
                count = int(source_counts.get(provider, 0) or 0)
            except (TypeError, ValueError):
                count = 0
            state = str(provider_status.get(provider, ""))
            if state in status_text:
                label = f"{provider} · {status_text[state]}"
            elif state == "ENABLED" and count == 0:
                label = f"{provider} · 0 results"
            elif state == "ENABLED":
                label = f"{provider} · {count}"
            elif not state:
                # Legacy saved results do not contain provider status; avoid
                # presenting missing metadata as a provider failure.
                label = f"{provider} · {count}" if count else f"{provider} · no data"
            else:
                label = f"{provider} · {state.lower().replace('_', ' ')}"
            provider_labels.append(label)
            provider_values.append(count)
        if any(provider_status.values()) or source_counts:
            self.source_chart._subtitle = "Provider participation · candidates before deduplication"
            self.source_chart.set_data(provider_labels, provider_values)
        else:
            self.source_chart._subtitle = "Provider participation and returned candidates"
            self.source_chart.clear_data()

        # Key metrics as a compact bar chart.
        summary = result.get("summary", {}) if isinstance(result.get("summary", {}), dict) else {}
        gaps = result.get("gaps", {}) if isinstance(result.get("gaps", {}), dict) else {}
        gap_summary = gaps.get("summary", {}) if isinstance(gaps.get("summary", {}), dict) else {}
        keyword_result = result.get("keywords", {}) if isinstance(result.get("keywords", {}), dict) else {}
        keywords = keyword_result.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        keyword_count = len(keywords) if isinstance(keywords, list) else 0
        paper_count = len(papers)
        gap_count = gap_summary.get("gap_count", 0) or 0
        relevance = summary.get("top_relevance_score") or 0
        metrics = [paper_count, keyword_count, gap_count, float(relevance)]
        labels = ["Papers", "Keywords", "Gaps", "Top Score"]
        self.metric_chart.set_data(labels, metrics if any(metrics) else [])

        if papers:
            self.pipeline_stages[3].set_status("READY", "success")
            self.pipeline_description.setText("Academic research results are ready")
            self._sync_workflow_connectors()

    def reset_dataset(self):
        self.dataset_name_label.setText("No dataset loaded")
        self.dataset_description_label.setText(
            "Upload a dataset to begin your research workflow."
        )
        self.dataset_status_badge.setText("NO DATASET")
        self.dataset_status_badge.setProperty("state", "waiting")
        style = self.dataset_status_badge.style()
        style.unpolish(self.dataset_status_badge)
        style.polish(self.dataset_status_badge)
        self.dataset_status_badge.update()
        self.analysis_button.setEnabled(False)
        self.rows_card.set_value("—")
        self.columns_card.set_value("—")
        self.pipeline_description.setText("Upload a dataset to begin your research workflow")
        for index, stage in enumerate(self.pipeline_stages):
            if index == 0:
                stage.set_status("UPLOAD", "active")
            else:
                stage.set_status("WAITING", "waiting")
        self._sync_workflow_connectors()
        self.update_research_result({})

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
        track = QRectF(indicator)
        painter.drawRoundedRect(track, indicator.height() / 2, indicator.height() / 2)

        painter.setBrush(self._thumb)
        thumb_diameter = max(8, indicator.height() - 8)
        thumb_left = (
            indicator.right() - thumb_diameter - 4
            if self.isChecked()
            else indicator.left() + 4
        )
        thumb = QRectF(
            thumb_left,
            indicator.top() + (indicator.height() - thumb_diameter) / 2,
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
        safe_width = min(
            max(1, self.available_geometry.width() - 32),
            max(1, main_window.width() - 32),
        )
        safe_height = min(
            max(1, self.available_geometry.height() - 32),
            max(1, main_window.height() - 32),
        )
        self.dialog_width = min(1027, safe_width)
        self.dialog_height = min(757, safe_height)
        self.compact_layout = self.dialog_width < 900 or self.dialog_height < 700
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
            26 if self.compact_layout else 42,
            24 if self.compact_layout else 49,
            26 if self.compact_layout else 42,
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
        content_layout.addSpacing(6 if self.compact_layout else 10)
        subtitle = QLabel("Choose a CSV file and confirm how DORA should read it.")
        subtitle.setObjectName("uploadDialogSubtitle")
        subtitle.setWordWrap(True)
        content_layout.addWidget(subtitle)
        content_layout.addSpacing(18 if self.compact_layout else 35)

        file_card = QFrame()
        file_card.setObjectName("uploadDialogFileCard")
        file_card.setFixedHeight(108 if self.compact_layout else 134)
        file_layout = QHBoxLayout(file_card)
        if self.compact_layout:
            file_layout.setContentsMargins(16, 12, 16, 12)
        else:
            file_layout.setContentsMargins(26, 20, 26, 20)
        file_layout.setSpacing(16 if self.compact_layout else 26)
        file_icon = QLabel()
        file_icon.setObjectName("uploadDialogFileIcon")
        file_icon.setAlignment(Qt.AlignCenter)
        file_icon.setFixedSize(72 if self.compact_layout else 80, 72 if self.compact_layout else 80)
        icon = QPixmap(str(_resource_path("assets/icons/file-text.svg")))
        if not icon.isNull():
            file_icon.setPixmap(icon.scaled(42, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        file_layout.addWidget(file_icon)
        file_text = QVBoxLayout()
        file_text.setSpacing(4)
        filename = QLabel(self.file_path.name)
        filename.setObjectName("uploadDialogFileName")
        size_kb = self.file_path.stat().st_size / 1024 if self.file_path.exists() else 0
        details_text = (
            f"{max(size_kb, 0.1):.1f} KB · {len(self.columns)} columns detected"
            if self.columns
            else f"{max(size_kb, 0.1):.1f} KB · CSV file selected"
        )
        details = QLabel(details_text)
        details.setObjectName("uploadDialogFileDetails")
        file_text.addWidget(filename)
        file_text.addWidget(details)
        file_layout.addLayout(file_text, 1)
        content_layout.addWidget(file_card)
        content_layout.addSpacing(18 if self.compact_layout else 33)

        dataset_name = QLineEdit(self.file_path.name)
        delimiter = QComboBox()
        delimiter.addItems(["Comma (,)" , "Semicolon (;)" , "Tab"])
        target = QComboBox()
        target.addItems(self.columns or ["Select target column"])
        if "Survived" in self.columns:
            target.setCurrentText("Survived")
        encoding = QComboBox()
        encoding.addItems(["UTF-8", "UTF-8-SIG", "Latin-1"])

        fields = QGridLayout()
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setHorizontalSpacing(10 if self.compact_layout else 14)
        fields.setVerticalSpacing(14 if self.compact_layout else 22)
        fields.addWidget(self._field("DATASET NAME", dataset_name), 0, 0)
        fields.addWidget(self._field("DELIMITER", delimiter), 0, 1)
        fields.addWidget(self._field("TARGET COLUMN", target), 1, 0)
        fields.addWidget(self._field("ENCODING", encoding), 1, 1)
        fields.setColumnStretch(0, 1)
        fields.setColumnStretch(1, 1)
        content_layout.addLayout(fields)
        content_layout.addSpacing(18 if self.compact_layout else 28)

        first_row = SwitchCheckBox(
            "First row contains column names",
            getattr(self.main_window, "_theme_tokens", {}),
        )
        first_row.setObjectName("uploadDialogCheck")
        first_row.setChecked(True)
        content_layout.addWidget(first_row)
        content_layout.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(10 if self.compact_layout else 12)
        actions.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("uploadDialogCancel")
        cancel.setFixedSize(126 if self.compact_layout else 154, 52 if self.compact_layout else 64)
        cancel.clicked.connect(self.reject)
        confirm = QPushButton("Upload && Preview")
        confirm.setObjectName("primaryButton")
        confirm.setFixedSize(230 if self.compact_layout else 290, 52 if self.compact_layout else 64)
        confirm.clicked.connect(self.accept)
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
        left = parent_center.x() - self.width() // 2
        top = parent_center.y() - self.height() // 2
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
        layout.setContentsMargins(36, 30, 24, 36)
        layout.setSpacing(32)

        header = QHBoxLayout()
        self.header_layout = header
        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title = QLabel("Dataset")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Upload, preview and clean your CSV before analysis.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)

        self.upload_button = QPushButton("Upload Dataset")
        self.upload_button.setIcon(QIcon(str(_resource_path("assets/icons/upload.svg"))))
        self.upload_button.setIconSize(QSize(22, 22))
        self.upload_button.setObjectName("primaryButton")
        self.upload_button.setMinimumHeight(62)
        self.upload_button.setMinimumWidth(254)
        self.upload_button.clicked.connect(self.select_dataset)
        header.addWidget(self.upload_button, 0, Qt.AlignTop)
        layout.addLayout(header)

        content_row = QHBoxLayout()
        self.content_row = content_row
        content_row.setSpacing(22)
        upload_card = QFrame()
        self.upload_card = upload_card
        upload_card.setObjectName("uploadCard")
        upload_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        apply_card_shadow(upload_card)
        upload_layout = QVBoxLayout(upload_card)
        upload_layout.setContentsMargins(18, 18, 28, 28)
        upload_layout.setSpacing(40)

        upload_title = QLabel("Upload a dataset")
        upload_title.setObjectName("sectionTitle")
        upload_title.setContentsMargins(18, 20, 0, 0)
        upload_title.setFixedHeight(44)
        upload_layout.addWidget(upload_title)

        dropzone = QFrame()
        dropzone.setObjectName("uploadDropzone")
        dropzone.setFixedHeight(326)
        drop_layout = QVBoxLayout(dropzone)
        drop_layout.setContentsMargins(16, 18, 16, 18)
        drop_layout.setSpacing(10)
        drop_layout.addStretch(1)
        icon_label = QLabel()
        icon_label.setObjectName("uploadDropIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedSize(68, 68)
        upload_pixmap = QPixmap(str(_resource_path("assets/icons/file-arrow-up.svg")))
        if not upload_pixmap.isNull():
            icon_label.setPixmap(upload_pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        drop_layout.addWidget(icon_label, 0, Qt.AlignHCenter)

        drop_title = QLabel("Drag & drop your file CSV file here")
        drop_title.setObjectName("uploadDropTitle")
        drop_title.setAlignment(Qt.AlignCenter)
        drop_title.setWordWrap(True)
        drop_layout.addWidget(drop_title)

        description_row = QGridLayout()
        self.drop_description_layout = description_row
        description_row.setSpacing(5)
        prefix = QLabel("or")
        prefix.setObjectName("uploadDropDescription")
        prefix.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        browse_link = QPushButton("browse from your computer.")
        browse_link.setObjectName("uploadBrowseLink")
        browse_link.setFlat(True)
        browse_link.setCursor(Qt.PointingHandCursor)
        browse_link.clicked.connect(self.select_dataset)
        description_row.addWidget(browse_link, 0, 1, Qt.AlignLeft | Qt.AlignVCenter)
        suffix = QLabel("csv up to 200 MB")
        suffix.setObjectName("uploadDropDescription")
        suffix.setObjectName("uploadDropSizeLimit")
        self.drop_size_label = suffix
        description_row.addWidget(prefix, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        description_row.addWidget(suffix, 0, 2, Qt.AlignLeft | Qt.AlignVCenter)
        drop_layout.addLayout(description_row)
        drop_layout.addStretch(1)
        dropzone_wrap = QWidget()
        dropzone_wrap_layout = QHBoxLayout(dropzone_wrap)
        dropzone_wrap_layout.setContentsMargins(18, 0, 0, 0)
        dropzone_wrap_layout.setSpacing(0)
        dropzone_wrap_layout.addWidget(dropzone)
        dropzone_wrap.setFixedHeight(326)
        upload_layout.addWidget(dropzone_wrap)

        steps_wrap = QWidget()
        steps_row = QGridLayout(steps_wrap)
        self.steps_layout = steps_row
        self.upload_steps = []
        steps_row.setContentsMargins(0, 0, 0, 0)
        steps_row.setHorizontalSpacing(5)
        steps_row.setVerticalSpacing(5)
        for number, heading, detail in (
            ("1", "Upload CSV", "Choose your file"),
            ("2", "Preview", "Check the first rows"),
            ("3", "Quality check", "Missing values & outlier"),
            ("4", "Analyze", "Continue to Analysis"),
        ):
            step = QFrame()
            step.setObjectName("uploadStep")
            step.setMinimumHeight(65)
            step.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            step_layout = QHBoxLayout(step)
            step_layout.setContentsMargins(9, 8, 8, 8)
            step_layout.setSpacing(9)
            badge = QLabel(number)
            badge.setObjectName("uploadStepBadge")
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(40, 40)
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
        upload_layout.addStretch(1)
        content_row.addWidget(upload_card, 215)

        side_column = QVBoxLayout()
        side_column.setSpacing(22)
        search_card = QFrame()
        search_card.setObjectName("uploadSearchCard")
        search_card.setMinimumHeight(244)
        apply_card_shadow(search_card)
        search_layout = QVBoxLayout(search_card)
        search_layout.setContentsMargins(22, 20, 22, 22)
        search_layout.setSpacing(10)
        search_icon = QLabel()
        search_icon.setObjectName("uploadSearchIcon")
        search_icon.setAlignment(Qt.AlignCenter)
        search_icon.setFixedSize(42, 42)
        search_pixmap = QPixmap(str(_resource_path("assets/icons/search.svg")))
        if not search_pixmap.isNull():
            search_icon.setPixmap(search_pixmap.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        search_layout.addWidget(search_icon, 0, Qt.AlignLeft)
        search_title = QLabel("No dataset yet?")
        search_title.setObjectName("cardTitle")
        search_layout.addWidget(search_title)
        search_description = QLabel(
            "Find public dataset from Kaggle and Hugging Face, then send them straight to DORA"
        )
        search_description.setObjectName("cardDescription")
        search_description.setWordWrap(True)
        search_layout.addWidget(search_description)
        search_button = QPushButton("Open Dataset Search")
        search_button.setObjectName("uploadSearchButton")
        search_button.setMinimumHeight(62)
        search_button.setMinimumWidth(254)
        search_button.clicked.connect(self.main_window.open_dataset_search_page)
        search_layout.addWidget(search_button, 0, Qt.AlignLeft)
        side_column.addWidget(search_card)

        tips_card = QFrame()
        tips_card.setObjectName("uploadTipsCard")
        apply_card_shadow(tips_card)
        tips_layout = QVBoxLayout(tips_card)
        tips_layout.setContentsMargins(22, 20, 22, 22)
        tips_layout.setSpacing(12)
        tips_title = QLabel("Tips for a good upload")
        tips_title.setObjectName("cardTitle")
        tips_layout.addWidget(tips_title)
        tips = QLabel(
            "1. First row contains column names\n"
            "2. One table per file, comma separated\n"
            "3. Choose your target column in the next step\n"
            "4. Keep sensitive data out of shared files"
        )
        tips.setObjectName("uploadTipsText")
        tips.setWordWrap(True)
        tips_layout.addWidget(tips)
        tips_layout.addStretch(1)
        side_column.addWidget(tips_card, 1)
        content_row.addLayout(side_column, 100)
        layout.addLayout(content_row, 1)

        self.info_label = QLabel("No dataset selected.")
        self.info_label.setObjectName("cardDescription")
        self.scroll.setWidget(container)
        root.addWidget(self.scroll)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = event.size().width()
        self.header_layout.setDirection(
            QBoxLayout.TopToBottom if width < 700 else QBoxLayout.LeftToRight
        )
        self.content_row.setDirection(
            QBoxLayout.TopToBottom if width < 990 else QBoxLayout.LeftToRight
        )

        description_columns = 1 if width < 480 else 3
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
        columns = 4 if card_width >= 900 else 2 if card_width >= 500 else 1
        for step in self.upload_steps:
            self.steps_layout.removeWidget(step)
        for index, step in enumerate(self.upload_steps):
            self.steps_layout.addWidget(step, index // columns, index % columns)

    # =====================================================
    # SELECT DATASET
    # =====================================================

    def select_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        upload_dialog = UploadDatasetDialog(self.main_window, file_path)
        if upload_dialog.exec_with_overlay() != QDialog.DialogCode.Accepted:
            return

        try:

            dataframe = (
                self.main_window.loader.load_csv(
                    file_path
                )
            )

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
                f"<b>File:</b> {filename}<br>"
                f"<b>Size:</b> {file_size:.2f} KB<br>"
                f"<b>Rows:</b> {len(dataframe):,}<br>"
                f"<b>Columns:</b> "
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
                "Upload Error",
                "Failed to load dataset.\n\n"
                f"{error}"
            )

class AnalysisPage(QWidget):

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self._gemini_thread: QThread | None = None
        self._gemini_worker: GeminiExplainWorker | None = None
        self._gemini_label: QLabel | None = None
        self._visualization_dataframe: pd.DataFrame | None = None
        self._visualization_correlations = None
        self._visualization_missing_values = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker: DatasetAnalysisWorker | None = None
        self._analysis_started_at = 0.0
        self._rendered_result_signature = None

        self.build_ui()

    def build_ui(self):

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        self.page_scroll = QScrollArea(self)
        self.page_scroll.setObjectName("analysisPageScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("analysisPageContent")
        root_layout = QVBoxLayout(content)
        self._root_layout = root_layout

        root_layout.setContentsMargins(36, 29, 28, 16)
        root_layout.setSpacing(18)

        # =================================================
        # HEADER
        # =================================================

        header_widget = QWidget()
        self._header_widget = header_widget
        header_layout = QHBoxLayout(header_widget)
        self._header_layout = header_layout

        title_layout = QVBoxLayout()

        title = QLabel(
            "Dataset Analysis"
        )
        self._title_label = title

        title.setObjectName(
            "pageTitle"
        )

        subtitle = QLabel(
            "Understand the structure, quality, "
            "statistics, and relationships within your dataset."
        )
        self._subtitle_label = subtitle

        subtitle.setObjectName(
            "analysisSubtitle"
        )

        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(title_layout, 1)

        self.analyze_button = QPushButton("Run Analysis")

        self.analyze_button.setObjectName(
            "primaryButton"
        )
        self.analyze_button.setProperty("analysisRun", True)

        self.analyze_button.setMinimumHeight(
            64
        )
        run_icon = _resource_path("assets/icons/run.svg")
        if run_icon.exists():
            self.analyze_button.setIcon(QIcon(str(run_icon)))
            self.analyze_button.setIconSize(QSize(24, 24))

        self.analyze_button.clicked.connect(
            self.run_analysis
        )

        header_layout.addWidget(
            self.analyze_button
        )

        root_layout.addWidget(header_widget)

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
        dataset_banner_layout = QHBoxLayout(self.dataset_label)
        self.dataset_banner_layout = dataset_banner_layout
        dataset_banner_layout.setContentsMargins(32, 0, 32, 0)
        dataset_banner_layout.setSpacing(12)
        self.dataset_name_label = QLabel("No dataset loaded")
        self.dataset_name_label.setObjectName("analysisDatasetName")
        self.dataset_name_label.setWordWrap(True)
        self.dataset_metrics_label = QLabel("")
        self.dataset_metrics_label.setObjectName("analysisDatasetMetrics")
        self.dataset_metrics_label.setWordWrap(True)
        dataset_banner_layout.addWidget(self.dataset_name_label)
        dataset_banner_layout.addStretch(1)
        dataset_banner_layout.addWidget(self.dataset_metrics_label)
        apply_card_shadow(self.dataset_label)

        root_layout.addWidget(self.dataset_label)

        # =================================================
        # RESULT AREA
        # =================================================

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        container = QWidget()

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

        self.show_empty_state()

    # =====================================================
    # EMPTY STATE
    # =====================================================

    def show_empty_state(self):

        self.clear_results()
        self._rendered_result_signature = None

        label = QLabel(
            "No analysis results yet.\n\n"
            "Upload a dataset and click "
            "\"Run Analysis\"."
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
        compact = width < 760
        self._title_label.setMinimumHeight(self._title_label.sizeHint().height())
        self._subtitle_label.setMinimumWidth(0)
        self._subtitle_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        subtitle_height = self._subtitle_label.heightForWidth(max(1, width - 32))
        if subtitle_height > 0:
            self._subtitle_label.setMinimumHeight(subtitle_height)
        if compact:
            self._header_widget.setMinimumHeight(
                self._title_label.minimumHeight()
                + self._subtitle_label.minimumHeight()
                + self.analyze_button.minimumHeight()
                + 18
            )
        else:
            self._header_widget.setMinimumHeight(0)
        self._root_layout.setContentsMargins(
            16 if compact else 36,
            16 if compact else 29,
            16 if compact else 28,
            12 if compact else 16,
        )
        self._header_layout.setDirection(
            QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight
        )
        self.readiness_header_layout.setDirection(
            QBoxLayout.TopToBottom
            if width < 560
            else QBoxLayout.LeftToRight
        )
        self.dataset_banner_layout.setDirection(
            QBoxLayout.TopToBottom
            if width < 720
            else QBoxLayout.LeftToRight
        )

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout(self.width())
        self._apply_overview_layout(self.width())

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
            "duplicates": self.main_window.duplicate_analyzer.analyze,
            "outliers": self.main_window.outlier_analyzer.analyze,
            "correlations": self.main_window.correlation_analyzer.analyze,
            "fingerprint": self.main_window.fingerprint_analyzer.generate,
        }
        self._analysis_thread = QThread(self)
        self._analysis_worker = DatasetAnalysisWorker(
            analyzers, self.main_window.quality_diagnoser, dataframe
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
        label = QLabel(f"Analysis failed.\n\n{error}")
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
            self.analyze_button.setText("Run Analysis")

    def _on_analysis_failed(self, error):
        duration_ms = int((time.perf_counter() - self._analysis_started_at) * 1000)
        self.main_window.fail_dataset_analysis(duration_ms)
        self._show_analysis_error(error)
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("Run Analysis")

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

        # The first viewport mirrors the Figma analysis dashboard. Existing
        # detailed result cards follow below it in the scroll area.
        self._build_visualization_panel()
        overview = QFrame()
        overview.setObjectName("analysisOverview")
        overview.setMinimumHeight(682)
        overview_layout = QHBoxLayout(overview)
        self._overview_layout = overview_layout
        overview_layout.setContentsMargins(0, 0, 0, 0)
        overview_layout.setSpacing(16)
        overview_layout.addWidget(self.visualization_card, 68)

        side_panel = QWidget()
        side_column = QVBoxLayout(side_panel)
        side_column.setContentsMargins(0, 0, 0, 0)
        side_column.setSpacing(18)
        side_column.addWidget(self._build_fingerprint_card(fingerprint))
        side_column.addWidget(self._build_quality_summary_card(), 1)
        overview_layout.addWidget(side_panel, 32)
        self._overview_side_column = side_panel
        self.result_layout.addWidget(overview)

        # Keep the analyzer's detailed result views and their action handlers
        # available below the Figma overview.
        self.add_cell_profiling(profile, fingerprint, statistics)

        # =================================================
        # DATA QUALITY
        # =================================================
        quality_issues = result.get("data_quality") or []
        self.add_cell_data_quality(quality_issues)

        # =================================================
        # MISSING VALUES
        # =================================================
        self.add_cell_missing_values(missing_values)

        # =================================================
        # OUTLIERS
        # =================================================
        self.add_cell_outliers(outliers)

        # =================================================
        # CORRELATIONS
        # =================================================
        self.add_cell_correlations(correlations)

        # =================================================
        # DATASET VERSIONS
        # =================================================
        self.add_versioning_section()

        # =================================================
        # AUTOMATED INTERPRETATION
        # =================================================
        self.add_cell_gemini(result)

        self.result_layout.addStretch()
        self._rendered_result_signature = (
            id(result),
            id(self.main_window.current_dataset),
        )
        self._apply_overview_layout(self.width())

    def _apply_overview_layout(self, width):
        overview_layout = getattr(self, "_overview_layout", None)
        if overview_layout is None:
            return
        stacked = width < 1050
        overview_layout.setDirection(
            QBoxLayout.TopToBottom if stacked else QBoxLayout.LeftToRight
        )
        overview_layout.setStretch(0, 68)
        overview_layout.setStretch(1, 32)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())
        self._apply_overview_layout(event.size().width())

    def _build_fingerprint_card(self, fingerprint):
        card = QFrame()
        card.setObjectName("analysisFingerprintCard")
        card.setMinimumHeight(162)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        title = QLabel("Sidik jari dataset")
        title.setObjectName("analysisPanelTitle")
        layout.addWidget(title)
        algorithm = QLabel("SHA-256")
        algorithm.setObjectName("analysisPanelSubtitle")
        layout.addWidget(algorithm)

        hash_value = fingerprint.get("fingerprint") if isinstance(fingerprint, dict) else ""
        value = QLabel(f"{hash_value[:32]}.." if hash_value else "Belum tersedia")
        value.setObjectName("analysisFingerprintValue")
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value.setToolTip(str(hash_value or ""))
        layout.addWidget(value)
        apply_card_shadow(card)
        return card

    def _build_quality_summary_card(self):
        card = QFrame()
        card.setObjectName("analysisQualitySummaryCard")
        card.setMinimumHeight(500)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(8)

        title = QLabel("Kualitas data")
        title.setObjectName("analysisPanelTitle")
        layout.addWidget(title)
        subtitle = QLabel("Temuan pada nilai kosong, duplikasi, dan outliers")
        subtitle.setObjectName("analysisPanelSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        layout.addSpacing(2)

        for object_name in ("analysisQualityCritical", "analysisQualityWarning", "analysisQualityGood"):
            indicator = QFrame()
            indicator.setObjectName(object_name)
            indicator.setMinimumHeight(92)
            layout.addWidget(indicator)
        layout.addStretch(1)
        apply_card_shadow(card)
        return card

    # =====================================================
    # VISUALIZATION
    # =====================================================

    MAX_PLOT_ROWS = 20_000
    MAX_CATEGORIES = 20

    def _build_visualization_panel(self):
        card = QFrame()
        card.setObjectName("contentCard")
        card.setMinimumHeight(682)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(32, 24, 32, 20)
        layout.setSpacing(12)

        title = QLabel("Visualisasi Data")
        title.setObjectName("analysisVisualizationTitle")
        layout.addWidget(title)
        layout.addSpacing(14)

        controls = QHBoxLayout()
        self.chart_type = QComboBox()
        self.chart_type.setObjectName("analysisControlSelect")
        self.chart_type.addItem("Histogram", "histogram")
        self.chart_type.addItem("Diagram batang", "bar")
        self.chart_type.addItem("Box plot", "box")
        self.chart_type.addItem("Heatmap korelasi", "correlation")
        self.chart_type.addItem("Nilai kosong", "missing")
        self.chart_type.currentIndexChanged.connect(self._update_visualization_options)

        self.chart_column = QComboBox()
        self.chart_column.setObjectName("analysisControlSelect")
        self.chart_column.currentIndexChanged.connect(self._draw_visualization)
        self.chart_type.currentIndexChanged.connect(self._draw_visualization)

        chart_type_label = QLabel("Jenis Grafik")
        chart_type_label.setObjectName("analysisControlLabel")
        controls.addWidget(chart_type_label)
        controls.addWidget(self.chart_type, 1)
        column_label = QLabel("Kolom")
        column_label.setObjectName("analysisControlLabel")
        controls.addWidget(column_label)
        controls.addWidget(self.chart_column, 1)
        layout.addLayout(controls)
        layout.addSpacing(24)

        self.chart_message = QLabel("Jalankan analisis untuk melihat visualisasi.")
        self.chart_message.setObjectName("analysisChartMessage")
        self.chart_message.setAlignment(Qt.AlignCenter)
        self.chart_message.setMinimumHeight(250)
        layout.addWidget(self.chart_message)

        self.chart_figure = Figure(figsize=(8, 4), tight_layout=True)
        self.chart_canvas = FigureCanvasQTAgg(self.chart_figure)
        self.chart_canvas.setObjectName("chartCanvas")
        self.chart_canvas.setMinimumHeight(440)
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
            return [
                column for column in dataframe.select_dtypes(include=["number"]).columns
                if dataframe[column].nunique(dropna=True) > 1
            ]
        if chart_type == "bar":
            return [
                column for column in dataframe.columns
                if 1 < dataframe[column].nunique(dropna=True) <= 100
            ]
        return []

    def _update_visualization_options(self, *_):
        if not hasattr(self, "chart_type"):
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
        if not hasattr(self, "chart_figure"):
            return
        self.chart_figure.clear()
        dataframe = self._visualization_dataframe
        if dataframe is None:
            self._show_chart_message("Belum ada dataset untuk divisualisasikan.")
            return

        chart_type = self.chart_type.currentData()
        axis = self.chart_figure.add_subplot(111)
        if len(dataframe) > self.MAX_PLOT_ROWS:
            sample = dataframe.sample(self.MAX_PLOT_ROWS, random_state=0)
        else:
            sample = dataframe

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
                    self._show_chart_message("Kolom ini belum memiliki variasi untuk histogram.")
                    return
                theme = getattr(self.main_window, "_theme_tokens", {})
                chart_blue = theme.get("chart-blue", "#0d99ff")
                chart_grid = theme.get("chart-grid", "#e5e5e5")
                chart_text = theme.get("chart-text", "#564e4e")
                axis.hist(values, bins="auto", color=chart_blue, edgecolor="white", linewidth=0.7)
                axis.set_title(f"Distribusi {column_name}", fontsize=12, color=chart_text, pad=12)
                axis.set_xlabel("")
                axis.set_ylabel("")
                axis.set_axisbelow(True)
                axis.grid(axis="y", color=chart_grid, linewidth=0.8)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.tick_params(axis="both", colors=chart_text, labelsize=10, length=0, pad=8)
            elif chart_type == "box":
                if values.empty:
                    self._show_chart_message("Tidak ada nilai untuk ditampilkan pada box plot.")
                    return
                axis.boxplot(values, vert=False, patch_artist=True,
                             boxprops={"facecolor": "#DCEAFF", "color": "#4F8CFF"},
                             medianprops={"color": "#172033"})
                axis.set(title=f"Sebaran {column_name}", xlabel=column_name, yticks=[])
            else:
                counts = sample.iloc[:, dataframe.columns.get_loc(column)].dropna()
                counts = counts.astype("string").value_counts().head(self.MAX_CATEGORIES)
                if counts.empty:
                    self._show_chart_message("Tidak ada kategori untuk ditampilkan.")
                    return
                axis.bar(counts.index.astype(str), counts.values, color="#4F8CFF")
                axis.set(title=f"Kategori {column_name}", xlabel=column_name, ylabel="Jumlah")
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
            axis.set_title("Korelasi antar kolom numerik")
            self.chart_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        else:
            missing = self._visualization_missing_values or []
            entries = [item for item in missing if item.get("missing_count", 0) > 0]
            if not entries:
                self._show_chart_message("Tidak ada nilai kosong pada dataset.")
                return
            names = [str(item["column"]) for item in entries[:self.MAX_CATEGORIES]]
            counts = [int(item["missing_count"]) for item in entries[:self.MAX_CATEGORIES]]
            axis.bar(names, counts, color="#4F8CFF")
            axis.set(title="Nilai kosong per kolom", xlabel="Kolom", ylabel="Jumlah")
            axis.tick_params(axis="x", labelrotation=35)

        self.chart_message.hide()
        self.chart_canvas.show()
        self.chart_canvas.draw_idle()

    @staticmethod
    def _empty_chart_message(chart_type):
        if chart_type in ("histogram", "box"):
            return "Tidak ada kolom numerik dengan variasi yang sesuai."
        if chart_type == "bar":
            return "Tidak ada kolom kategori yang sesuai untuk diagram batang."
        if chart_type == "correlation":
            return "Heatmap korelasi belum tersedia untuk dataset ini."
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

    def _on_gemini_finished(self, text: str, _source: str = "local"):
        if self._gemini_label is not None:
            self._gemini_label.setText(text)

    def _on_gemini_failed(self, message: str):
        if self._gemini_label is not None:
            self._gemini_label.setText(
                "Gagal memuat interpretasi cloud. Sistem menggunakan interpretasi lokal."
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
            f"{len(dataframe):,} rows x {len(dataframe.columns):,} columns"
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

    def add_cell_profiling(self, profile, fingerprint, statistics):
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

        # Target Candidates Detection from Fingerprint
        rep = fingerprint.get("representation", {}) if isinstance(fingerprint, dict) else {}
        target_candidates = rep.get("target_candidates", [])
        col_sigs = {c.get("name"): c for c in rep.get("column_signature", []) if isinstance(c, dict)}

        target_box = QFrame()
        target_box.setStyleSheet(
            "background: #EAF7F0; border: 1px solid #D3EFDF; border-radius: 8px; padding: 10px 14px;"
        )
        t_layout = QVBoxLayout(target_box)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(4)

        if target_candidates:
            first_target = target_candidates[0]
            sig = col_sigs.get(first_target, {})
            score = sig.get("target_score", 60)
            reasons = sig.get("target_reasons", [])
            reasons_str = "; ".join(reasons) if reasons else "Indikasi target klasifikasi/regresi"
            u_count = sig.get("unique_count", "-")
            task_type = "Klasifikasi biner" if u_count == 2 else ("Klasifikasi multikelas" if isinstance(u_count, int) and u_count <= 10 else "Regresi / prediksi")
            t_title = QLabel(f"<b>Kandidat Target Terdeteksi:</b> <span style='color: #2C9B68; font-weight: bold;'>{first_target}</span> ({task_type})")
            t_desc = QLabel(f"Skor perkiraan: <b>{score}/100</b> &nbsp;&bull;&nbsp; Dasar: {reasons_str}")
        else:
            t_title = QLabel("<b>Kandidat Target:</b> Tidak terdeteksi kolom target eksplisit.")
            t_desc = QLabel("Belum ada kolom target yang jelas. Analisis tanpa target masih dapat dilakukan.")

        t_title.setStyleSheet("color: #1A3A28; font-size: 12px;")
        t_desc.setStyleSheet("color: #2C6B4A; font-size: 11px;")
        t_layout.addWidget(t_title)
        t_layout.addWidget(t_desc)
        layout.addWidget(target_box)

        # Feature Schema & Column Characteristics Table (Stage 2: Data Understanding)
        if statistics:
            schema_title = QLabel("<b>Karakteristik kolom</b>")
            schema_title.setStyleSheet("font-size: 12px; color: #1C2B40; margin-top: 6px;")
            layout.addWidget(schema_title)

            schema_table = QTableWidget()
            schema_table.setColumnCount(5)
            schema_table.setHorizontalHeaderLabels([
                "Nama kolom", "Tipe data", "Nilai unik", "Kosong (%)", "Ringkasan statistik / modus"
            ])
            schema_table.setRowCount(len(statistics))
            schema_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            schema_table.verticalHeader().setVisible(False)
            schema_table.setEditTriggers(QTableWidget.NoEditTriggers)

            for r_idx, stat in enumerate(statistics):
                col_name = str(stat.get("column", "—"))
                dtype = str(stat.get("dtype", "—"))
                u_cnt = stat.get("unique_count", 0)
                m_pct = stat.get("missing_percentage", 0.0)

                # Summary stats text
                if stat.get("mean") is not None:
                    summary = f"Rentang: [{stat.get('min', 0):.2g} – {stat.get('max', 0):.2g}] | Rata-rata: {stat.get('mean', 0):.2f}"
                elif stat.get("top_values"):
                    top_v = stat["top_values"][0]
                    summary = f"Modus: {top_v.get('value')} ({top_v.get('frequency')}x)"
                else:
                    summary = "—"

                schema_table.setItem(r_idx, 0, QTableWidgetItem(col_name))

                dt_item = QTableWidgetItem(dtype)
                dt_item.setTextAlignment(Qt.AlignCenter)
                schema_table.setItem(r_idx, 1, dt_item)

                u_item = QTableWidgetItem(f"{u_cnt:,}")
                u_item.setTextAlignment(Qt.AlignCenter)
                schema_table.setItem(r_idx, 2, u_item)

                m_item = QTableWidgetItem(f"{m_pct:.1f}%")
                m_item.setTextAlignment(Qt.AlignCenter)
                schema_table.setItem(r_idx, 3, m_item)

                schema_table.setItem(r_idx, 4, QTableWidgetItem(summary))

            schema_table.setFixedHeight(min(200, 36 + len(statistics) * 28))
            layout.addWidget(schema_table)

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
                drop_btn = QPushButton("Drop Baris dengan Missing Values")
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
        b_text = f"{len(active_missing)} Kolom Kosong" if active_missing else "0 Nilai Kosong"
        b_style = "warning" if active_missing else "success"

        card, layout = self._create_cell_frame(
            "Nilai kosong",
            "Kolom yang memiliki data belum terisi.",
            badge_text=b_text,
            badge_style=b_style,
        )

        if not active_missing:
            no_miss = QLabel("Seluruh kolom memiliki data lengkap (0 missing values). Tidak diperlukan perlakuan imputasi.")
            no_miss.setStyleSheet("color: #2C9B68; font-size: 12px; padding: 4px 0;")
            layout.addWidget(no_miss)
            self.result_layout.addWidget(card)
            return

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Kolom", "Jumlah Kosong", "Persentase", "Status Keparahan"])
        table.setRowCount(len(active_missing))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row_idx, item in enumerate(active_missing):
            col_name = str(item.get("column", "—"))
            count = int(self._number(item.get("missing_count", 0)))
            pct = float(self._number(item.get("missing_percentage", 0)))

            table.setItem(row_idx, 0, QTableWidgetItem(col_name))

            c_item = QTableWidgetItem(f"{count:,}")
            c_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 1, c_item)

            p_item = QTableWidgetItem(f"{pct:.1f}%")
            p_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 2, p_item)

            tag_str = "Kritis (>40%)" if pct > 40 else ("Sedang (5-40%)" if pct >= 5 else "Rendah (<5%)")
            s_item = QTableWidgetItem(tag_str)
            s_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 3, s_item)

        table.setFixedHeight(min(160, 36 + len(active_missing) * 30))
        layout.addWidget(table)

        # Action Buttons (Stage 5 Data Prep with Versioning!)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        impute_btn = QPushButton("Terapkan Imputasi (Median / Modus)")
        impute_btn.setObjectName("secondaryButton")
        impute_btn.clicked.connect(self._handle_impute_missing)
        action_row.addWidget(impute_btn)

        drop_btn = QPushButton("Drop Baris dengan Missing Values")
        drop_btn.setObjectName("secondaryButton")
        drop_btn.clicked.connect(self._handle_drop_missing)
        action_row.addWidget(drop_btn)

        action_row.addStretch()
        layout.addLayout(action_row)

        self.result_layout.addWidget(card)

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

        status_note = QLabel(
            "<span style='color: #8492A5; font-size: 10px;'>"
            "Aplikasi menggunakan Hybrid Engine: otomatis beralih ke mesin lokal jika offline atau limit."
            "</span>"
        )
        cfg_row.addWidget(status_note)
        cfg_row.addStretch()
        layout.addLayout(cfg_row)

        self._start_gemini_explanation(result)
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
            df_dropped, cleaning_report = drop_missing_rows(df)
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

        confirm = QMessageBox.question(
            self,
            "Konfirmasi Drop Baris Missing",
            f"Ditemukan {cleaning_report['missing_before']:,} nilai kosong "
            f"pada {cleaning_report['rows_with_missing']:,} baris.\n\n"
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
            dataframe.to_csv(export_path, index=False, encoding="utf-8-sig")
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

        self.setObjectName("contentCard")

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(6)

        title_label = QLabel(
            title.upper()
        )

        title_label.setObjectName(
            "cardLabel"
        )

        self.value_label = QLabel(
            value
        )

        self.value_label.setObjectName(
            "cardValue"
        )

        self.value_label.setWordWrap(
            True
        )

        self.subtitle_label = QLabel(
            subtitle
        )

        self.subtitle_label.setObjectName(
            "cardDescription"
        )

        self.subtitle_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            self.value_label
        )

        layout.addWidget(
            self.subtitle_label
        )

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
                "Why this task?"
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
                    "Unknown Method",
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
                "Why this model?"
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
                f"Scaling: "
                f"{'Required' if scaling else 'Not required'}"
            )
        )

        info_layout.addWidget(
            QLabel(
                f"Nonlinear: "
                f"{'Yes' if nonlinear else 'No'}"
            )
        )

        info_layout.addStretch()

        layout.addLayout(
            info_layout
        )

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
                "Method Details",
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
            f"{'Required' if method.get('scaling_required') else 'Not required'}"
            "\n"
            "Nonlinear: "
            f"{'Yes' if method.get('nonlinear') else 'No'}"
            "\n"
            "Small Data: "
            f"{'Suitable' if method.get('small_data') else 'Limited'}"
            "\n"
            "Large Data: "
            f"{'Suitable' if method.get('large_data') else 'Limited'}"
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
        self.intelligence_engine = MLIntelligenceEngine(
            recommender=self.recommender,
            task_detector=self.detector,
        )
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
            30,
            25,
            30,
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

        title_layout = QVBoxLayout()

        title_layout.setSpacing(
            4
        )

        title = QLabel(
            "ML Intelligence"
        )

        title.setObjectName(
            "pageTitle"
        )

        description = QLabel(
            "Deteksi task machine learning "
            "dan rekomendasi metode berdasarkan "
            "struktur serta karakteristik dataset."
        )

        description.setObjectName(
            "pageDescription"
        )

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
            title_layout
        )

        header.addStretch()

        self.refresh_button = QPushButton(
            "Run ML"
        )

        self.refresh_button.setObjectName(
            "primaryButton"
        )

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
            5,
            10,
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
        compact = width < 760
        self._root_layout.setContentsMargins(
            16 if compact else 30,
            16 if compact else 25,
            16 if compact else 30,
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
        self.refresh_button.setText("Run ML")

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
            "Upload dan analisis dataset terlebih dahulu "
            "untuk mendapatkan estimasi task machine learning "
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
            research_done=bool(getattr(self.main_window.research_page, "last_result", None)),
        )
        self._finish_ml_run()

    def _on_ml_failed(self, error):
        self.show_error(f"Gagal melakukan ML Intelligence:\n{error}")
        self._finish_ml_run()

    def _finish_ml_run(self):
        self.refresh_button.setEnabled(True)
        self.refresh_button.setText("Run ML again" if self.last_task_result else "Run ML")

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

        status_card.setObjectName(
            "contentCard"
        )

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
            "Detection Status"
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

        self.content_layout.addWidget(
            status_card
        )

        # ==================================================
        # PRIMARY TASK ANALYSIS
        # ==================================================

        if primary:

            primary_section = QLabel(
                "Primary Task Analysis"
            )

            primary_section.setObjectName(
                "sectionTitle"
            )

            self.content_layout.addWidget(
                primary_section
            )

            self.content_layout.addWidget(
                TaskCard(
                    primary
                )
            )

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

            for index, method in enumerate(
                recommendations,
                start=1,
            ):

                rank_label = QLabel(
                    f"#{index}"
                )

                rank_label.setObjectName(
                    "cardLabel"
                )

                self.content_layout.addWidget(
                    rank_label
                )

                self.content_layout.addWidget(
                    MethodCard(
                        method
                    )
                )

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
            "Research Note"
        )

        disclaimer_title.setObjectName(
            "cardLabel"
        )

        disclaimer_text = QLabel(
            "ML task detection dan method recommendation "
            "merupakan estimasi berbasis heuristik serta "
            "knowledge base. Performa aktual metode harus "
            "divalidasi melalui eksperimen, cross-validation, "
            "dan metric evaluasi yang sesuai."
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
            "ML Intelligence Error"
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
        self.refresh_button.setText("Run ML again")
        return True

    def show_ready_state(self):
        self.clear_content()
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(30, 40, 30, 40)
        layout.setSpacing(10)
        title = QLabel("Dataset analysis is ready")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        description = QLabel(
            "Tekan Run ML untuk melatih dan membandingkan seluruh metode "
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

class ResearchPage(QWidget):
    """
    Academic Research workspace.

    Menampilkan:
    - Research Overview
    - Ranked Papers
    - Dataset Usage Confidence
    - Research Landscape
    - Research Trend
    - Research Gap
    """

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.engine = ResearchIntelligenceEngine()
        self.task_detector = MLTaskDetector()

        self.research_result: Dict[str, Any] = {}
        self._all_papers = []
        self._all_paper_result_indices = []
        self.filtered_paper_indices = []
        self._research_thread: QThread | None = None
        self._research_worker: AcademicResearchWorker | None = None
        self._sinta_thread: QThread | None = None
        self._sinta_worker: SintaLookupWorker | None = None
        self._sinta_lookup_cache = {}

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
            "Explore related papers, publication trends, and potential research gaps."
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

        self.run_button = QPushButton(
            "Run Academic Research"
        )

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
            "Dataset: No dataset loaded"
        )

        self.dataset_label.setObjectName(
            "statusLabel"
        )

        self.analysis_label = QLabel(
            "Analysis: Not available"
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
            "Searching literature and preparing the research overview, papers, landscape, and potential gaps.",
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

        self.overview_tab = (
            self._scroll_tab(self._create_overview_tab())
        )

        self.papers_tab = self._create_papers_tab()

        self.landscape_tab = self._create_landscape_tab()

        self.trend_tab = self._create_trend_tab()

        self.gap_tab = self._create_gap_tab()

        self.tabs.addTab(
            self.overview_tab,
            "Overview",
        )

        # Keep every research result in one scrollable overview. The sidebar
        # still opens the dedicated Papers, Landscape, and Gap tools.
        overview_content = self.overview_tab.widget()
        overview_layout = overview_content.layout()
        if overview_layout.count() and overview_layout.itemAt(overview_layout.count() - 1).widget() is None:
            overview_layout.takeAt(overview_layout.count() - 1)
        for section in (
            self.landscape_tab,
            self.trend_tab,
            self.gap_tab,
        ):
            divider = QFrame()
            divider.setObjectName("researchSectionDivider")
            divider.setFrameShape(QFrame.NoFrame)
            divider.setFixedHeight(1)
            overview_layout.addWidget(divider)
            overview_layout.addWidget(section)
        overview_layout.addStretch(1)

        root.addWidget(
            self.tabs,
            1,
        )

        # Initial state
        self.show_empty_state()

    @staticmethod
    def _scroll_tab(content_widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(content_widget)
        return scroll

    def _apply_responsive_layout(self, width):
        compact = width < 760
        horizontal_margin = 16 if compact else 30
        vertical_margin = 16 if compact else 25
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
                + 18
            )
        else:
            self.header_widget.setMinimumHeight(0)
        overview_cards = getattr(self, "_overview_cards_layout", None)
        if overview_cards is not None:
            overview_cards.setDirection(
                QBoxLayout.TopToBottom if width < 1000 else QBoxLayout.LeftToRight
            )
        for metrics_layout in (
            getattr(self, "_landscape_metrics_layout", None),
            getattr(self, "_trend_metrics_layout", None),
        ):
            if metrics_layout is not None:
                metrics_layout.setDirection(
                    QBoxLayout.TopToBottom if width < 1000 else QBoxLayout.LeftToRight
                )
        compact = width < 1100
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
                QBoxLayout.TopToBottom if width < 1250 else QBoxLayout.LeftToRight
            )
        chart_stacked = width < 1100
        if (
            self.research_result
            and getattr(self, "_landscape_chart_stacked", None) is not None
            and chart_stacked != self._landscape_chart_stacked
        ):
            landscape = self.research_result.get("landscape") or {}
            self._render_landscape_chart(landscape)
            self._render_trend_chart(
                self.research_result.get("trend") or {}, landscape
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

    def _create_overview_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        layout.setContentsMargins(
            5,
            15,
            5,
            5,
        )

        layout.setSpacing(18)

        # -----------------------------------------------------
        # SUMMARY CARDS
        # -----------------------------------------------------

        cards = QHBoxLayout()
        self._overview_cards_layout = cards

        self.papers_card = (
            self._create_metric_card(
                "Papers Found",
                "0",
            )
        )

        self.score_card = (
            self._create_metric_card(
                "Top Relevance",
                "-",
            )
        )

        self.domain_card = (
            self._create_metric_card(
                "Research Domain",
                "-",
            )
        )

        self.gap_card = (
            self._create_metric_card(
                "Potential Gaps",
                "0",
            )
        )

        cards.addWidget(
            self.papers_card
        )

        cards.addWidget(
            self.score_card
        )

        cards.addWidget(
            self.domain_card
        )

        cards.addWidget(
            self.gap_card
        )

        layout.addLayout(
            cards
        )

        # -----------------------------------------------------
        # INTELLIGENCE
        # -----------------------------------------------------

        intelligence_frame = QFrame()

        intelligence_frame.setObjectName(
            "researchCard"
        )

        intelligence_layout = QVBoxLayout(
            intelligence_frame
        )

        intelligence_title = QLabel(
            "Research Intelligence"
        )

        intelligence_title.setObjectName(
            "sectionTitle"
        )

        intelligence_layout.addWidget(
            intelligence_title
        )

        self.keyword_text = QLabel(
            "Keywords: -"
        )

        self.domain_text = QLabel(
            "Domain: -"
        )

        self.task_text = QLabel(
            "ML Task: -"
        )

        self.method_text = QLabel(
            "Dominant Method: -"
        )

        self.direction_text = QLabel(
            "Research Direction: -"
        )

        self.latest_year_text = QLabel(
            "Latest Publication: -"
        )

        for label in (
            self.keyword_text,
            self.domain_text,
            self.task_text,
            self.method_text,
            self.direction_text,
            self.latest_year_text,
        ):
            label.setWordWrap(True)

            intelligence_layout.addWidget(
                label
            )

        layout.addWidget(
            intelligence_frame
        )

        # -----------------------------------------------------
        # TOP PAPER
        # -----------------------------------------------------

        top_frame = QFrame()

        top_frame.setObjectName(
            "researchCard"
        )

        top_layout = QVBoxLayout(
            top_frame
        )

        top_title = QLabel(
            "Ranked Papers"
        )

        top_title.setObjectName(
            "sectionTitle"
        )

        self.top_paper_title = QLabel(
            "No research has been run yet."
        )

        self.top_paper_title.setWordWrap(
            True
        )

        self.top_paper_info = QLabel(
            ""
        )

        self.top_paper_info.setWordWrap(
            True
        )

        top_layout.addWidget(
            top_title
        )

        top_layout.addWidget(
            self.top_paper_title
        )

        top_layout.addWidget(
            self.top_paper_info
        )

        self.additional_papers_title = QLabel("More relevant papers")
        self.additional_papers_title.setObjectName("sectionTitle")
        self.additional_papers_title.setVisible(False)
        top_layout.addWidget(self.additional_papers_title)

        self.overview_papers_layout = QVBoxLayout()
        self.overview_papers_layout.setContentsMargins(0, 0, 0, 0)
        self.overview_papers_layout.setSpacing(8)
        top_layout.addLayout(self.overview_papers_layout)

        self.overview_papers_status = QLabel(
            "Run Academic Research to see ranked papers here."
        )
        self.overview_papers_status.setObjectName("cardDescription")
        self.overview_papers_status.setWordWrap(True)
        top_layout.addWidget(self.overview_papers_status)

        layout.addWidget(
            top_frame
        )

        layout.addStretch()

        return widget

    # ---------------------------------------------------------

    # ---------------------------------------------------------

    def _create_papers_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(7)

        title = QLabel(
            "Artikel Akademik Terperingkat"
        )

        title.setObjectName(
            "sectionTitle"
        )

        description = QLabel(
            "Tinjau artikel dan sumbernya, lalu filter jurnal berdasarkan status SINTA yang telah dikonfirmasi."
        )
        description.setObjectName("cardDescription")

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

        filter_label = QLabel("Status SINTA")
        filter_label.setObjectName("cardDescription")
        toolbar.addWidget(filter_label)

        self.sinta_filter = QComboBox()
        self.sinta_filter.setMinimumHeight(34)
        self.sinta_filter.setMinimumWidth(185)
        self.sinta_filter.addItem("Semua Status Akreditasi", "ALL")
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
        self.sinta_filter.currentIndexChanged.connect(self._apply_paper_filter)
        toolbar.addWidget(self.sinta_filter)

        self.sinta_batch_button = QPushButton("Periksa jurnal belum dicek")
        self.sinta_batch_button.setObjectName("secondaryButton")
        self.sinta_batch_button.setMinimumHeight(34)
        self.sinta_batch_button.setToolTip(
            "Periksa semua jurnal yang belum memiliki hasil direktori SINTA untuk tahun artikelnya."
        )
        self.sinta_batch_button.clicked.connect(self._check_pending_sinta_batch)
        toolbar.addWidget(self.sinta_batch_button)

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

        self.paper_table.setColumnCount(8)

        self.paper_table.setHorizontalHeaderLabels(
            [
                "#",
                "Judul Artikel",
                "Tahun",
                "Jurnal / Venue",
                "ISSN",
                "Sumber",
                "Relevansi",
                "Status SINTA",
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

        header = (
            self.paper_table.horizontalHeader()
        )

        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
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

        self.paper_table.setMinimumHeight(210)
        self.paper_table.setWordWrap(False)
        self.paper_table.setTextElideMode(Qt.ElideRight)
        self.paper_table.setShowGrid(False)
        self.paper_table.verticalHeader().setDefaultSectionSize(34)
        self.paper_table.verticalHeader().setVisible(False)

        self.paper_table.itemSelectionChanged.connect(
            self._show_selected_paper
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
            "Peringkat diberikan pada jurnal. Gemini dapat membantu mencari profil kandidat; status tetap merujuk pada data resmi SINTA."
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

        selection_panel = QFrame()
        selection_panel.setObjectName("paperSelectionPanel")
        selection_layout = QHBoxLayout(selection_panel)
        self._paper_selection_layout = selection_layout
        selection_layout.setContentsMargins(10, 8, 10, 8)
        selection_layout.setSpacing(12)
        selection_layout.addWidget(self.paper_detail, 3)
        selection_layout.addWidget(self.sinta_screening_panel, 2)
        selection_panel.setMinimumHeight(190)

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        layout.addWidget(filter_panel)
        layout.addLayout(provider_row)

        layout.addWidget(
            self.paper_table,
            1,
        )

        layout.addWidget(selection_panel)

        return widget

    # ---------------------------------------------------------

    def _create_landscape_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        title = QLabel("Research Landscape")
        title.setObjectName("sectionTitle")
        description = QLabel(
            "Distribution of publication sources, methods, and venues in the papers found."
        )
        description.setObjectName("cardDescription")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)

        metrics = QHBoxLayout()
        self._landscape_metrics_layout = metrics
        self.landscape_papers_metric = self._create_metric_card("Papers analyzed", "0")
        self.landscape_methods_metric = self._create_metric_card("Methods identified", "0")
        self.landscape_venues_metric = self._create_metric_card("Publication venues", "0")
        self.landscape_latest_metric = self._create_metric_card("Latest year", "-")
        for card in (self.landscape_papers_metric, self.landscape_methods_metric,
                     self.landscape_venues_metric, self.landscape_latest_metric):
            metrics.addWidget(card)
        layout.addLayout(metrics)

        chart_card = QFrame()
        chart_card.setObjectName("researchCard")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(12, 10, 12, 10)
        self.landscape_figure = Figure(figsize=(11, 3.6))
        self.landscape_axes = list(self.landscape_figure.subplots(1, 3).ravel())
        self._landscape_chart_stacked = None
        self.landscape_canvas = FigureCanvasQTAgg(self.landscape_figure)
        self.landscape_canvas.setMinimumWidth(0)
        self.landscape_canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chart_layout.addWidget(self.landscape_canvas)
        layout.addWidget(chart_card, 1)

        self.landscape_text = QTextEdit()
        self.landscape_text.setReadOnly(True)
        self.landscape_text.setMaximumHeight(150)
        layout.addWidget(self.landscape_text)
        return widget

    # ---------------------------------------------------------

    def _create_trend_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        title = QLabel("Publication Trend")
        title.setObjectName("sectionTitle")
        description = QLabel(
            "Annual publication counts for the papers returned by this literature search."
        )
        description.setObjectName("cardDescription")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)

        metrics = QHBoxLayout()
        self._trend_metrics_layout = metrics
        self.trend_years_metric = self._create_metric_card("Years represented", "0")
        self.trend_peak_metric = self._create_metric_card("Peak publication year", "-")
        self.trend_direction_metric = self._create_metric_card("Recent direction", "-")
        self.trend_growth_metric = self._create_metric_card("Endpoint change", "-")
        for card in (self.trend_years_metric, self.trend_peak_metric,
                     self.trend_direction_metric, self.trend_growth_metric):
            metrics.addWidget(card)
        layout.addLayout(metrics)

        chart_card = QFrame()
        chart_card.setObjectName("researchCard")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(12, 10, 12, 10)
        self.trend_figure = Figure(figsize=(10, 3.8))
        self.trend_axes = self.trend_figure.subplots()
        self.trend_canvas = FigureCanvasQTAgg(self.trend_figure)
        self.trend_canvas.setMinimumWidth(0)
        self.trend_canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chart_layout.addWidget(self.trend_canvas)
        layout.addWidget(chart_card, 1)

        self.trend_text = QTextEdit()
        self.trend_text.setReadOnly(True)
        self.trend_text.setMaximumHeight(115)
        layout.addWidget(self.trend_text)
        return widget

    # ---------------------------------------------------------

    def _create_gap_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        title = QLabel(
            "Potential Research Gap"
        )

        title.setObjectName(
            "sectionTitle"
        )

        description = QLabel(
            "These are potential research directions inferred "
            "from the analyzed literature. They are not claims "
            "that a research gap has been definitively proven."
        )

        description.setWordWrap(
            True
        )

        self.gap_summary_label = QLabel("Potential directions: 0")
        self.gap_summary_label.setObjectName("statusLabel")
        self.gap_text = QTextBrowser()
        self.gap_text.setOpenExternalLinks(True)
        self.gap_text.setReadOnly(True)

        layout.addWidget(
            title
        )

        layout.addWidget(description)
        layout.addWidget(self.gap_summary_label)
        layout.addWidget(self.gap_text, 1)

        return widget

    # =========================================================
    # METRIC CARD
    # =========================================================

    @staticmethod
    def _create_metric_card(
        title: str,
        value: str,
    ):

        frame = QFrame()

        frame.setObjectName(
            "metricCard"
        )

        frame.setMinimumHeight(
            105
        )

        layout = QVBoxLayout(
            frame
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "metricTitle"
        )

        value_label = QLabel(
            value
        )

        value_label.setObjectName(
            "metricValue"
        )

        value_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            value_label
        )

        frame.value_label = value_label

        return frame

    # =========================================================
    # EMPTY STATE
    # =========================================================

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
            "Dataset: No dataset loaded"
        )

        self.analysis_label.setText(
            "Analysis: Not available"
        )

        self._set_metric(
            self.papers_card,
            "0",
        )

        self._set_metric(
            self.score_card,
            "-",
        )

        self._set_metric(
            self.domain_card,
            "-",
        )

        self._set_metric(
            self.gap_card,
            "0",
        )

        self.keyword_text.setText(
            "Keywords: -"
        )

        self.domain_text.setText(
            "Domain: -"
        )

        self.task_text.setText(
            "ML Task: -"
        )

        self.method_text.setText(
            "Dominant Method: -"
        )

        self.direction_text.setText(
            "Research Direction: -"
        )

        self.latest_year_text.setText(
            "Latest Publication: -"
        )

        self.top_paper_title.setText(
            "No research has been run yet."
        )

        self.top_paper_info.setText(
            ""
        )
        self._render_overview_papers([])

        self.paper_table.setRowCount(
            0
        )

        self.paper_detail.clear()
        self.paper_count_label.setText("0 papers | 0 need screening")
        self.provider_status_label.setText(
            "Sources: OpenAlex, Crossref, Semantic Scholar, Europe PMC, and OpenAIRE. "
            "Scopus requires Elsevier API key/access; UCL Discovery is external-only."
        )
        self.sinta_status_combo.setCurrentIndex(0)
        self.sinta_period_confirmed.setChecked(False)
        self.save_sinta_button.setEnabled(False)
        self.check_sinta_button.setEnabled(False)
        self.sinta_lookup_progress.setVisible(False)
        self.sinta_lookup_message.setText(
            "Select a paper and check its journal using ISSN or journal title."
        )

        self.landscape_text.clear()

        self.trend_text.clear()

        self.gap_text.clear()
        self.gap_summary_label.setText("Potential directions: 0")
        for card in (self.landscape_papers_metric, self.landscape_methods_metric,
                     self.landscape_venues_metric, self.landscape_latest_metric):
            self._set_metric(card, "-")
        for card in (self.trend_years_metric, self.trend_peak_metric,
                     self.trend_direction_metric, self.trend_growth_metric):
            self._set_metric(card, "-")
        for axis in self.landscape_axes:
            axis.clear()
            axis.set_axis_off()
        self.landscape_canvas.draw_idle()
        self.trend_axes.clear()
        self.trend_axes.set_axis_off()
        self.trend_canvas.draw_idle()
        self.loading_card.setVisible(False)

        if hasattr(self.main_window, "papers_tool_page"):
            self.main_window.papers_tool_page.show_empty_state()

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

        self.run_button.setEnabled(False)
        self.run_button.setText("Researching...")
        self.progress.setVisible(True)
        self.loading_card.setVisible(True)
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
                QMessageBox.critical(self, "Research Error", str(result.get("error", "Unknown research error.")))
                return
            self.research_result = result
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
            self.main_window.dashboard.update_research_result(result)
            self.main_window.dashboard.update_workflow_state(
                has_dataset=True,
                analysis_done=bool(self.main_window.analysis_result),
                ml_done=bool(self.main_window.ml_page.last_task_result and self.main_window.ml_page.last_method_result),
                research_done=True,
            )
            self.main_window.papers_tool_page.update_from_result(result)
            self.main_window.landscape_tool_page.update_from_result(result)
            self.main_window.gap_tool_page.update_from_result(result)
            self._update_status(self.main_window.current_dataset, self.main_window.analysis_result)
            self._populate_result(result, ml_result)
            self.tabs.setCurrentWidget(self.overview_tab)
        except Exception as error:
            QMessageBox.critical(self, "Academic Research Error", str(error))
        finally:
            self._finish_research()

    def _on_research_failed(self, error):
        QMessageBox.critical(self, "Academic Research Error", str(error))
        self._finish_research()

    def _finish_research(self):
        self.progress.setVisible(False)
        self.loading_card.setVisible(False)
        self.run_button.setEnabled(True)
        self.run_button.setText("Run Academic Research")

    def _cleanup_research_worker(self):
        self._research_worker = None
        self._research_thread = None

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
            f"({dataframe.shape[0]:,} rows × "
            f"{dataframe.shape[1]} columns)"
        )

        self.analysis_label.setText(
            "Analysis: Dataset fingerprint available"
        )

    # =========================================================
    # POPULATE RESULT
    # =========================================================

    def _populate_result(
        self,
        result,
        ml_result,
    ):

        query_list = result.get("queries") or []
        self.ucl_search_query = next(
            (str(query).strip() for query in query_list if str(query).strip()),
            "",
        )

        summary = result.get(
            "summary",
            {}
        )

        papers = result.get(
            "papers",
            []
        )

        academic_search = (result.get("pipeline") or {}).get("academic_search") or {}
        provider_status = academic_search.get("provider_status") or {}
        provider_counts = academic_search.get("sources") or {}
        provider_parts = []
        for provider in ("OpenAlex", "Crossref", "Semantic Scholar", "Europe PMC", "OpenAIRE", "Scopus"):
            state = provider_status.get(provider)
            count = provider_counts.get(provider)
            if state == "NOT_CONFIGURED":
                requirement = "Elsevier key/access needed" if provider == "Scopus" else "key/access needed"
                provider_parts.append(f"{provider}: {requirement}")
            elif state == "DISABLED":
                provider_parts.append(f"{provider}: off")
            elif state == "ERROR":
                provider_parts.append(f"{provider}: temporarily unavailable")
            elif state == "RATE_LIMITED":
                provider_parts.append(f"{provider}: rate limited")
            elif state == "ENABLED":
                candidate_count = int(count or 0)
                provider_parts.append(f"{provider} {candidate_count}")
        if provider_parts:
            self.provider_status_label.setText(
                "Candidates before deduplication  |  " + "  |  ".join(provider_parts)
            )

        keywords_result = result.get(
            "keywords",
            {}
        )

        domain_result = result.get(
            "domain",
            {}
        )

        # -----------------------------------------------------
        # SUMMARY
        # -----------------------------------------------------

        papers_found = summary.get(
            "papers_found",
            len(papers),
        )

        top_score = summary.get(
            "top_relevance_score"
        )

        gap_count = summary.get(
            "potential_gap_count",
            0,
        )

        domain = self._extract_domain(
            domain_result
        )

        self._set_metric(
            self.papers_card,
            str(papers_found),
        )

        self._set_metric(
            self.score_card,
            (
                f"{top_score:.1f}%"
                if isinstance(
                    top_score,
                    (int, float)
                )
                else "-"
            ),
        )

        self._set_metric(
            self.domain_card,
            domain or "-",
        )

        self._set_metric(
            self.gap_card,
            str(gap_count),
        )

        # -----------------------------------------------------
        # KEYWORDS
        # -----------------------------------------------------

        keywords = self._extract_keywords(
            keywords_result
        )

        if keywords:

            self.keyword_text.setText(
                "Keywords: "
                + ", ".join(
                    str(k)
                    for k in keywords
                )
            )

        else:

            self.keyword_text.setText(
                "Keywords: -"
            )

        # -----------------------------------------------------
        # DOMAIN
        # -----------------------------------------------------

        self.domain_text.setText(
            f"Research Domain: {domain or '-'}"
        )

        # -----------------------------------------------------
        # ML TASK
        # -----------------------------------------------------

        task = ml_result.get(
            "primary_task"
        )

        self.task_text.setText(
            f"ML Task: {task or '-'}"
        )

        # -----------------------------------------------------
        # LANDSCAPE SUMMARY
        # -----------------------------------------------------

        landscape = result.get(
            "landscape",
            {}
        )

        landscape_summary = (
            landscape.get(
                "summary",
                {}
            )
            if isinstance(
                landscape,
                dict
            )
            else {}
        )

        dominant_method = (
            landscape_summary.get(
                "dominant_method"
            )
        )

        latest_year = (
            landscape_summary.get(
                "latest_publication_year"
            )
        )

        self.method_text.setText(
            f"Dominant Method: "
            f"{dominant_method or '-'}"
        )

        self.latest_year_text.setText(
            f"Latest Publication: "
            f"{latest_year or '-'}"
        )

        # -----------------------------------------------------
        # TREND
        # -----------------------------------------------------

        trend = result.get(
            "trend",
            {}
        )

        direction = (
            trend.get(
                "recent_direction",
                "UNKNOWN",
            )
            if isinstance(
                trend,
                dict
            )
            else "UNKNOWN"
        )

        self.direction_text.setText(
            f"Research Direction: {direction}"
        )

        # -----------------------------------------------------
        # TOP PAPER
        # -----------------------------------------------------

        top_paper = (
            papers[0]
            if papers
            else None
        )

        if top_paper:

            title = top_paper.get(
                "title",
                "Untitled",
            )

            score = top_paper.get(
                "relevance_score"
            )

            confidence = top_paper.get(
                "dataset_usage_confidence",
                "UNKNOWN",
            )

            self.top_paper_title.setText(
                title
            )

            self.top_paper_info.setText(
                f"Relevance: "
                f"{self._format_score(score)}"
                f"    •    "
                f"Dataset Usage Confidence: "
                f"{confidence}"
            )

        else:

            self.top_paper_title.setText(
                "No academic papers found."
            )

            self.top_paper_info.setText(
                ""
            )

        # -----------------------------------------------------
        # PAPERS
        # -----------------------------------------------------

        self._populate_papers(
            papers
        )
        self._render_overview_papers(papers)

        # -----------------------------------------------------
        # LANDSCAPE
        # -----------------------------------------------------

        landscape = landscape if isinstance(landscape, dict) else {}
        landscape_summary = landscape.get("summary") or {}
        years = landscape.get("publication_years") or {}
        self._set_metric(self.landscape_papers_metric, str(landscape.get("paper_count", len(papers))))
        self._set_metric(self.landscape_methods_metric, str(len(landscape.get("methods") or [])))
        self._set_metric(self.landscape_venues_metric, str(len(landscape.get("top_venues") or [])))
        self._set_metric(
            self.landscape_latest_metric,
            str(landscape_summary.get("latest_publication_year") or (max(years, key=str) if years else "-")),
        )
        self._render_landscape_chart(landscape)
        top_source = (landscape.get("sources") or [{}])[0]
        top_method = (landscape.get("methods") or [{}])[0]
        top_venue = (landscape.get("top_venues") or [{}])[0]
        self.landscape_text.setPlainText(
            "Landscape summary\n"
            f"Papers analyzed: {landscape.get('paper_count', len(papers))}\n"
            f"Latest publication: {landscape_summary.get('latest_publication_year') or '-'}\n"
            f"Most represented source: {top_source.get('name', '-')} ({top_source.get('percentage', 0)}%)\n"
            f"Most identified method: {top_method.get('name', '-')} ({top_method.get('percentage', 0)}%)\n"
            f"Leading venue: {top_venue.get('name', '-')} ({top_venue.get('percentage', 0)}%)"
        )

        # -----------------------------------------------------
        # TREND
        # -----------------------------------------------------

        trend = trend if isinstance(trend, dict) else {}
        trend_rows = trend.get("publication_trend") or landscape.get("research_activity") or []
        growth = trend.get("growth") if isinstance(trend.get("growth"), dict) else {}
        self._set_metric(self.trend_years_metric, str(len(trend_rows)))
        self._set_metric(self.trend_peak_metric, str(trend.get("peak_year") or "-"))
        self._set_metric(self.trend_direction_metric, str(trend.get("recent_direction") or "-"))
        growth_pct = growth.get("percentage")
        self._set_metric(self.trend_growth_metric, f"{growth_pct:+g}%" if isinstance(growth_pct, (int, float)) else "Insufficient data")
        self._render_trend_chart(trend, landscape)
        year_range = trend.get("year_range") if isinstance(trend.get("year_range"), dict) else {}
        self.trend_text.setPlainText(
            "Trend summary\n"
            f"Recent direction: {trend.get('recent_direction', 'UNKNOWN')}\n"
            f"Peak year: {trend.get('peak_year') or '-'} ({trend.get('peak_paper_count', 0)} papers)\n"
            f"Year range: {year_range.get('start', '-')} to {year_range.get('end', '-')}\n"
            f"Endpoint change: {growth_pct:+g}%" if isinstance(growth_pct, (int, float)) else
            "Trend summary\n"
            f"Recent direction: {trend.get('recent_direction', 'UNKNOWN')}\n"
            f"Peak year: {trend.get('peak_year') or '-'} ({trend.get('peak_paper_count', 0)} papers)\n"
            f"Year range: {year_range.get('start', '-')} to {year_range.get('end', '-')}\n"
            "Endpoint change: insufficient data"
        )

        # -----------------------------------------------------
        # GAP
        # -----------------------------------------------------

        gaps = result.get("gaps", {})
        gap_items = []
        if isinstance(gaps, dict):
            gap_items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items") or []
        elif isinstance(gaps, list):
            gap_items = gaps
        self.gap_summary_label.setText(f"Potential directions: {len(gap_items)}")
        self.gap_text.setHtml(self._format_gap_html(gaps))

    def _open_ucl_discovery_search(self):
        base_url = "https://discovery.ucl.ac.uk/cgi/search/simple"
        if self.ucl_search_query:
            url = f"{base_url}?q={quote_plus(self.ucl_search_query)}"
        else:
            url = "https://discovery.ucl.ac.uk/"
        QDesktopServices.openUrl(QUrl(url))

    def _render_landscape_chart(self, landscape):
        stacked = self.width() < 1100
        if stacked != self._landscape_chart_stacked:
            self.landscape_figure.clear()
            axes = (
                self.landscape_figure.subplots(3, 1)
                if stacked
                else self.landscape_figure.subplots(1, 3)
            )
            self.landscape_axes = list(axes.ravel())
            figure_width = max(5.5, min(11.0, (self.width() - 100) / 100))
            self.landscape_figure.set_size_inches(
                figure_width, 7.5 if stacked else 3.6, forward=False
            )
            self.landscape_canvas.setMinimumHeight(750 if stacked else 360)
            self._landscape_chart_stacked = stacked

        groups = [
            ("Sources", landscape.get("sources") or []),
            ("Methods", landscape.get("methods") or []),
            ("Venues", landscape.get("top_venues") or []),
        ]
        for axis, (title, entries) in zip(self.landscape_axes, groups):
            axis.clear()
            entries = [item for item in entries if isinstance(item, dict) and item.get("name")]
            entries = entries[:6]
            axis.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#20324A")
            if not entries:
                axis.text(0.5, 0.5, "No data available", ha="center", va="center", color="#718096", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            axis.set_axis_on()
            names = [str(item["name"]) for item in reversed(entries)]
            if stacked:
                names = [textwrap.fill(name, width=18) for name in names]
            percentages = [float(item.get("percentage", 0) or 0) for item in reversed(entries)]
            bars = axis.barh(names, percentages, color="#4F8CFF", height=0.62)
            axis_limit = max(100, max(percentages, default=0) * 1.16)
            axis.set_xlim(0, axis_limit)
            axis.set_xlabel("Share of analyzed papers (%)", fontsize=8, color="#64748B")
            axis.tick_params(axis="both", labelsize=8, colors="#64748B")
            for spine in ("top", "right", "left"):
                axis.spines[spine].set_visible(False)
            axis.grid(axis="x", color="#E8EEF6", linewidth=0.7)
            axis.set_axisbelow(True)
            for bar, value in zip(bars, percentages):
                near_edge = value >= axis_limit * 0.84
                axis.text(
                    value - 1.0 if near_edge else value + 0.8,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:g}%",
                    ha="right" if near_edge else "left",
                    va="center",
                    fontsize=8,
                    color="#FFFFFF" if near_edge else "#344158",
                )
        # The three side-by-side horizontal charts can have long category
        # names. Reserve room for their y-axis labels explicitly instead of
        # asking tight_layout to shrink each axes until it becomes too narrow.
        if stacked:
            self.landscape_figure.subplots_adjust(
                left=0.30, right=0.98, bottom=0.06, top=0.96, hspace=0.55
            )
        else:
            self.landscape_figure.subplots_adjust(
                left=0.28, right=0.98, bottom=0.22, top=0.90, wspace=0.55
            )
        self.landscape_canvas.draw_idle()

    def _render_trend_chart(self, trend, landscape):
        axis = self.trend_axes
        axis.clear()
        rows = trend.get("publication_trend") or landscape.get("research_activity") or []
        if not rows and isinstance(landscape.get("publication_years"), dict):
            rows = [{"year": year, "paper_count": count} for year, count in landscape["publication_years"].items()]
        rows = [row for row in rows if isinstance(row, dict) and row.get("year") is not None]
        rows.sort(key=lambda row: int(row["year"]))
        axis.set_title("Papers by publication year", loc="left", fontsize=10, fontweight="bold", color="#20324A")
        if not rows:
            axis.text(0.5, 0.5, "No publication year data available", ha="center", va="center", color="#718096", transform=axis.transAxes)
            axis.set_axis_off()
        else:
            axis.set_axis_on()
            years = [str(row["year"]) for row in rows]
            counts = [int(row.get("paper_count", 0) or 0) for row in rows]
            bars = axis.bar(years, counts, color="#4F8CFF", width=0.72)
            axis.set_ylabel("Papers", fontsize=8, color="#64748B")
            axis.set_xlabel("Publication year", fontsize=8, color="#64748B")
            axis.tick_params(axis="both", labelsize=8, colors="#64748B")
            for spine in ("top", "right", "left"):
                axis.spines[spine].set_visible(False)
            axis.grid(axis="y", color="#E8EEF6", linewidth=0.7)
            axis.set_axisbelow(True)
            for bar, count in zip(bars, counts):
                axis.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom", fontsize=8, color="#344158")
        self.trend_figure.subplots_adjust(
            left=0.08, right=0.98, bottom=0.22, top=0.90
        )
        self.trend_canvas.draw_idle()

    @staticmethod
    def _format_gap_html(gaps):
        if not isinstance(gaps, dict):
            gaps = {"gaps": gaps if isinstance(gaps, list) else []}
        items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items") or []
        if not isinstance(items, list):
            items = []
        cards = []
        for index, item in enumerate(items, 1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("gap") or item.get("description") or "Potential research direction"
                description = item.get("description") or item.get("rationale") or item.get("problem") or ""
                evidence = item.get("evidence")
                details = []
                if description and description != title:
                    details.append(f"<p>{html_escape(str(description))}</p>")
                if evidence:
                    evidence_text = ResearchPage._format_section(evidence)
                    details.append(f"<p><b>Evidence</b><br>{html_escape(evidence_text).replace(chr(10), '<br>')}</p>")
                for key, label in (("recommendation", "Suggested direction"), ("method", "Method"), ("dataset", "Dataset")):
                    value = item.get(key)
                    if value:
                        details.append(f"<p><b>{label}:</b> {html_escape(str(value))}</p>")
                body = "".join(details) or "<p>Potential direction inferred from the discovered literature.</p>"
            else:
                title = f"Potential direction {index}"
                body = f"<p>{html_escape(str(item))}</p>"
            cards.append(
                "<div style='background:#FFFFFF;border:1px solid #DCE7F5;border-radius:10px;padding:14px;margin:8px 2px;'>"
                f"<h3 style='color:#20324A;margin:0 0 8px 0;'>{index}. {html_escape(str(title))}</h3>{body}</div>"
            )
        warning = gaps.get("warning") or "Potential research gaps are heuristic signals and should be validated against the literature."
        header = f"<p style='color:#607086;'>{html_escape(str(warning))}</p>"
        if not cards:
            return header + "<p style='color:#607086;'>No potential research directions were identified in this result.</p>"
        return header + "".join(cards)

    # =========================================================
    # PAPER TABLE
    # =========================================================

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
            except Exception as exc:
                print(f"Warning: gagal memuat riwayat screening dari database: {exc}")

        self._all_paper_result_indices = [
            index for index, paper in enumerate(papers) if isinstance(paper, dict)
        ]
        self._all_papers = [papers[index] for index in self._all_paper_result_indices]
        self._apply_paper_filter()

    def _render_overview_papers(self, papers):
        while self.overview_papers_layout.count():
            item = self.overview_papers_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        papers = [
            paper for paper in (papers if isinstance(papers, list) else [])
            if isinstance(paper, dict)
        ]
        additional = papers[1:5]
        self.additional_papers_title.setVisible(bool(additional))

        for rank, paper in enumerate(additional, start=2):
            row = QFrame()
            row.setObjectName("researchPaperPreviewRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 10, 12, 10)
            row_layout.setSpacing(12)

            rank_label = QLabel(f"{rank:02d}")
            rank_label.setObjectName("researchPaperPreviewRank")
            rank_label.setAlignment(Qt.AlignCenter)
            rank_label.setFixedWidth(38)
            row_layout.addWidget(rank_label, 0, Qt.AlignTop)

            details = QVBoxLayout()
            details.setContentsMargins(0, 0, 0, 0)
            details.setSpacing(4)
            title = QLabel(str(paper.get("title") or "Untitled paper"))
            title.setObjectName("researchPaperPreviewTitle")
            title.setWordWrap(True)
            title.setMinimumWidth(0)
            metadata = []
            if paper.get("year"):
                metadata.append(str(paper["year"]))
            if paper.get("venue"):
                metadata.append(str(paper["venue"]))
            score = paper.get("relevance_score")
            if isinstance(score, (int, float)):
                metadata.append(f"Relevance {self._format_score(score)}")
            summary = QLabel("  ·  ".join(metadata) or "Publication details unavailable")
            summary.setObjectName("cardDescription")
            summary.setWordWrap(True)
            summary.setMinimumWidth(0)
            details.addWidget(title)
            details.addWidget(summary)
            row_layout.addLayout(details, 1)
            self.overview_papers_layout.addWidget(row)

        if papers:
            shown_count = min(len(papers), 5)
            self.overview_papers_status.setText(
                f"Showing the top {shown_count} of {len(papers)} papers. "
                "Open Papers from the sidebar for the full list and journal screening."
            )
        else:
            self.overview_papers_status.setText(
                "No academic papers were found for this research run."
            )

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
                return f"SINTA {rank[1:]}"
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
        return compact_labels.get(label, label)

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
            "FAILED": "Belum dapat terhubung ke direktori; coba lagi",
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
        status = str(screening.get("status") or "UNVERIFIED")
        lookup_status = str(screening.get("lookup_status") or "UNVERIFIED")
        if selected_status == "UNVERIFIED":
            return status == "UNVERIFIED" and lookup_status in {"UNVERIFIED", "LEGACY_UNVERIFIED"}
        if selected_status in {f"S{rank}" for rank in range(1, 7)}:
            return str(screening.get("rank") or status) == selected_status
        if selected_status == "NOT_FOUND":
            return lookup_status == "NOT_FOUND"
        return status == selected_status or lookup_status == selected_status

    def _apply_paper_filter(self, *_):
        papers = getattr(self, "_all_papers", [])
        result_indices = getattr(self, "_all_paper_result_indices", [])
        query = self.paper_search_input.text().strip().casefold()
        selected_status = self.sinta_filter.currentData()
        self.filtered_paper_indices = []

        for index, paper in zip(result_indices, papers):
            if not self._matches_sinta_filter(paper, selected_status):
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

            values = [
                str(row + 1),
                title,
                year,
                venue_display,
                issn_str,
                source_label,
                self._format_score(relevance),
                sinta_label,
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(value)

                if column == 0:
                    item.setData(Qt.UserRole, paper_index)
                    item.setTextAlignment(Qt.AlignCenter)
                elif column == 1:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    item.setToolTip(title)
                elif column in {2, 4, 5, 6, 7}:
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
                        f"{_sinta_ui_note(screening) or screening.get('source_url') or 'Pilih baris ini untuk memeriksa akreditasi.'}"
                    )

                if not item.toolTip():
                    item.setToolTip(str(value))

                self.paper_table.setItem(row, column, item)

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
        self.paper_count_label.setText(
            f"{len(self.filtered_paper_indices)} dari {len(self._all_papers)} paper | "
            f"{checked_count} sudah diperiksa | {len(self._all_papers) - checked_count} belum diperiksa"
        )

        if self.filtered_paper_indices:
            selected_row = 0
            if selected_paper_index in self.filtered_paper_indices:
                selected_row = self.filtered_paper_indices.index(selected_paper_index)
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
                self.paper_detail.setPlainText("Tidak ada paper yang cocok dengan pencarian atau filter SINTA ini.")
            self.save_sinta_button.setEnabled(False)
            self.check_sinta_button.setEnabled(False)
            self.sinta_status_combo.setCurrentIndex(0)
            self.sinta_period_confirmed.setChecked(False)
            self.sinta_lookup_message.setText("Tidak ada artikel yang sesuai dengan filter saat ini.")
            self.sinta_lookup_message.setToolTip("")

    # =========================================================
    # PAPER DETAIL
    # =========================================================

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

        issn_values = paper.get("issn", [])
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        issn_clean = [str(i).strip() for i in issn_values if str(i).strip()]
        self.check_sinta_button.setEnabled(
            self._sinta_thread is None
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
        if self._sinta_thread is not None:
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
            papers_page = getattr(self.main_window, "papers_tool_page", None)
            if papers_page is not None:
                papers_page.update_from_result(self.research_result)
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
        self._render_filtered_papers()

    @Slot(int)
    def _finish_sinta_batch(self, total):
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
        papers_page = getattr(self.main_window, "papers_tool_page", None)
        if papers_page is not None:
            papers_page.update_from_result(self.research_result)
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
        if self._sinta_thread is not None:
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

        self.main_window.papers_tool_page.update_from_result(self.research_result)
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
            return "No data available."

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
        self.main_window = main_window
        self.papers = []
        self.filtered_papers = []
        self._result_signature = None
        self._sinta_batch_was_running = False
        self._sinta_last_progress_value = -1
        self._sinta_sync_timer = QTimer(self)
        self._sinta_sync_timer.setInterval(300)
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
            "Browse, filter, and inspect papers discovered by Academic Research."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header.addLayout(title_layout)
        header.addStretch()

        self.count_label = QLabel("0 papers")
        self.count_label.setObjectName("topbarStatus")
        header.addWidget(self.count_label)

        root.addLayout(header)

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
        self.sinta_filter.addItem("Semua Status Akreditasi", "ALL")
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

        root.addLayout(toolbar)

        sinta_status_panel = QFrame()
        sinta_status_panel.setObjectName("researchCard")
        sinta_status_layout = QVBoxLayout(sinta_status_panel)
        sinta_status_layout.setContentsMargins(14, 10, 14, 10)
        sinta_status_layout.setSpacing(6)

        self.sinta_batch_status = QLabel(
            "Filter status SINTA langsung dari daftar. Tombol di atas memeriksa jurnal yang belum dicek."
        )
        self.sinta_batch_status.setObjectName("cardDescription")
        self.sinta_batch_status.setWordWrap(True)
        sinta_status_layout.addWidget(self.sinta_batch_status)

        self.sinta_scope_note = QLabel(
            "Pemeriksaan saat ini hanya mencocokkan jurnal dengan direktori SINTA. "
            "Peringkat dan status akreditasi dapat berubah, jadi hasil ini adalah "
            "snapshot yang perlu diperiksa kembali. Akreditasi atau indeks lain "
            "belum diperiksa."
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
        self.paper_table.setColumnCount(8)
        self._paper_headers = [
            "#",
            "Judul Artikel",
            "Tahun",
            "Jurnal / Venue",
            "ISSN",
            "Sumber",
            "Relevansi",
            "Status SINTA",
        ]
        self._paper_compact_headers = [
            "#", "Artikel", "Thn", "Jurnal", "ISSN", "Sumber", "Skor", "SINTA"
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

        self.paper_table.setMinimumHeight(280)
        self.paper_table.setShowGrid(False)
        self.paper_table.verticalHeader().setDefaultSectionSize(38)
        self.paper_table.verticalHeader().setVisible(False)

        self.paper_table.itemSelectionChanged.connect(
            self._show_selected_paper
        )
        self.paper_table.cellDoubleClicked.connect(
            self._open_paper_source_for_row
        )

        table_layout.addWidget(self.paper_table, 1)
        root.addWidget(table_card, 1)

        # -----------------------------------------------------
        # DETAIL
        # -----------------------------------------------------

        detail_card = QFrame()
        detail_card.setObjectName("researchCard")

        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(16, 14, 16, 14)
        detail_layout.setSpacing(8)

        detail_title = QLabel("Paper Details")
        detail_title.setObjectName("sectionTitle")

        self.paper_detail = QTextEdit()
        self.paper_detail.setReadOnly(True)
        self.paper_detail.setMinimumHeight(165)
        self.paper_detail.setMaximumHeight(220)

        self.open_source_button = QPushButton("Open Paper Source")
        self.open_source_button.setObjectName("secondaryButton")
        self.open_source_button.setEnabled(False)
        self.open_source_button.clicked.connect(self._open_selected_source)

        detail_actions = QHBoxLayout()
        detail_actions.addStretch()
        detail_actions.addWidget(self.open_source_button)

        detail_layout.addWidget(detail_title)
        detail_layout.addWidget(self.paper_detail)
        detail_layout.addLayout(detail_actions)

        root.addWidget(detail_card)

        self.show_empty_state()

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
            "No academic research results available.\n\n"
            "Run Academic Research first to populate this tool."
        )
        self.count_label.setText("0 papers")
        self.open_source_button.setEnabled(False)
        self.sinta_batch_button.setEnabled(False)
        self.sinta_batch_progress.setVisible(False)
        self.sinta_batch_status.setText(
            "Jalankan Academic Research terlebih dahulu untuk memeriksa jurnal di SINTA."
        )

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
        if research_page is None or getattr(research_page, "_sinta_thread", None) is None:
            return
        if not self._sinta_sync_timer.isActive():
            self._sinta_last_progress_value = -1
        self._sinta_batch_was_running = True
        self.sinta_batch_button.setEnabled(False)
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

    def _sync_sinta_batch_state(self):
        research_page = getattr(self.main_window, "research_page", None)
        if research_page is None:
            self._sinta_sync_timer.stop()
            return

        thread = getattr(research_page, "_sinta_thread", None)
        if thread is not None:
            self._sinta_batch_was_running = True
            self.sinta_batch_button.setEnabled(False)
            self.sinta_batch_button.setText("Memeriksa jurnal...")
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
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(value)

                if column in {0, 2, 4, 5, 6, 7}:
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

                self.paper_table.setItem(row, column, item)

        self.paper_table.setSortingEnabled(False)
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

        row = rows[0].row()

        if row < 0 or row >= len(self.filtered_papers):
            return

        paper = self.filtered_papers[row]

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

        if screening.get("journal_name"):
            lines.append(f"JURNAL TERDAFTAR DI SINTA\n{screening.get('journal_name')}")
        if screening.get("issn_matched"):
            lines.append(f"ISSN COCOK DI SINTA\n{screening.get('issn_matched')}")
        if screening.get("period_start") and screening.get("period_end"):
            lines.append(f"MASA BERLAKU SK AKREDITASI\n{screening.get('period_start')} - {screening.get('period_end')}")
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

        row = rows[0].row()

        if row < 0 or row >= len(self.filtered_papers):
            return None

        return self.filtered_papers[row]

    def _open_selected_source(self):
        self._open_paper_source(self._selected_paper())

    def _open_paper_source_for_row(self, row, _column=0):
        if row < 0 or row >= len(self.filtered_papers):
            return

        self._open_paper_source(self.filtered_papers[row])

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
        standard_widths = (42, 64, 175, 105, 110, 85, 145)
        for column, width in zip((0, 2, 3, 4, 5, 6, 7), compact_widths if compact else standard_widths):
            self.paper_table.setColumnWidth(column, width)
        self._root_layout.setContentsMargins(
            16 if compact else 30,
            16 if compact else 25,
            16 if compact else 30,
            16 if compact else 25,
        )


class ResearchLandscapeToolPage(QWidget):
    """Dedicated Research Tool: Research Landscape.

    Reads the already-generated Academic Research landscape result.
    No additional academic search is performed here.
    """

    def __init__(self, main_window):
        super().__init__()
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
        root.setContentsMargins(36, 20, 28, 16)
        root.setSpacing(0)

        header_widget = QWidget()
        self.header_widget = header_widget
        header = QHBoxLayout(header_widget)
        self.header_layout = header
        header.setContentsMargins(0, 0, 0, 0)
        title_layout = QVBoxLayout()
        title = QLabel("Research Landscape")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Explore the methods, topics, and publication structure found in the analyzed literature."
        )
        self.subtitle_label = subtitle
        self.title_label = title
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_layout.setContentsMargins(0, 9, 0, 0)
        title_layout.setSpacing(5)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header.addLayout(title_layout, 1)

        self.refresh_button = QPushButton("Search")
        self.refresh_button.setObjectName("landscapeSearchButton")
        self.refresh_button.setMinimumSize(151, 66)
        self.refresh_button.setCursor(Qt.PointingHandCursor)
        search_icon = _resource_path("assets/icons/search-white.svg")
        if search_icon.exists():
            self.refresh_button.setIcon(QIcon(str(search_icon)))
            self.refresh_button.setIconSize(QSize(24, 24))
        self.refresh_button.clicked.connect(self.refresh)
        header.addWidget(self.refresh_button, 0, Qt.AlignTop | Qt.AlignRight)
        root.addWidget(header_widget)
        root.addSpacing(5)

        self.status_label = QLabel("No academic research result available.")
        self.status_label.setObjectName("landscapeStatus")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addSpacing(25)

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
            "Run Academic Research terlebih dahulu.\n"
            "Setelah literature analysis selesai, hasil landscape akan muncul di sini."
        )
        desc.setObjectName("cardDescription")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        title.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(desc)
        self.content.addWidget(card)
        self.content.addStretch()
        self.status_label.setText("No academic research result available.")

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
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(6)

        title_label = QLabel(title.upper())
        title_label.setObjectName("landscapeMetricTitle")
        title_label.setWordWrap(True)
        value_label = QLabel(self._pretty(value))
        value_label.setObjectName("landscapeMetricValue")
        value_label.setWordWrap(True)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("landscapeMetricSubtitle")
        subtitle_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
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
                    item, "method", "label", "name", "category", default=""
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

    def _add_distribution_card(self, title, rows, show_values=True):
        card = QFrame()
        card.setObjectName("landscapeDistributionCard")
        card.setMinimumHeight(418)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        apply_card_shadow(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 16, 24)
        layout.setSpacing(19)
        heading = QLabel(title)
        heading.setObjectName("landscapeDistributionTitle")
        layout.addWidget(heading)

        if rows:
            maximum = max((amount for _, amount in rows), default=1) or 1
            for label, amount in rows:
                row = QHBoxLayout()
                row.setSpacing(12)
                name = QLabel(label)
                name.setObjectName("landscapeMethodLabel")
                name.setMinimumWidth(100)
                name.setMaximumWidth(259)

                bar = QProgressBar()
                bar.setObjectName("landscapeMethodBar")
                bar.setRange(0, 1000)
                bar.setValue(round(max(0, amount) / maximum * 1000))
                bar.setTextVisible(False)
                bar.setFixedHeight(8)

                row.addWidget(name)
                row.addWidget(bar, 1)
                if show_values:
                    value_label = QLabel(f"{amount:.2f}".rstrip("0").rstrip(".").replace(".", ","))
                    value_label.setObjectName("landscapeMethodValue")
                    value_label.setAlignment(Qt.AlignCenter)
                    row.addWidget(value_label)
                layout.addLayout(row)
            layout.addStretch(1)
        else:
            empty = QLabel("No method distribution available.")
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
            f"Landscape loaded from Academic Research | {paper_count} papers analyzed"
        )

        cards_widget = QWidget()
        cards_widget.setObjectName("landscapeMetricsRow")
        cards = QHBoxLayout(cards_widget)
        self._metrics_layout = cards
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(14)
        cards.addWidget(self._metric_card("Papers", paper_count, "Literature analyzed"))
        cards.addWidget(self._metric_card("Dominan Method", dominant_method, "Most represented method"))
        cards.addWidget(self._metric_card("Latest Publication", latest_year, "Newest publication year"))
        cards.setStretch(0, 46)
        cards.setStretch(1, 49)
        cards.setStretch(2, 50)
        self.content.addWidget(cards_widget)

        distribution_widget = QWidget()
        distribution_widget.setObjectName("landscapeDistributionRow")
        distribution_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        distribution = QHBoxLayout(distribution_widget)
        distribution.setContentsMargins(0, 0, 0, 0)
        distribution.setSpacing(18)
        distribution.addWidget(
            self._add_distribution_card("Method Distribution", method_rows, show_values=True)
        )
        distribution.addWidget(
            self._add_distribution_card("Method Distribution", method_rows, show_values=False)
        )
        distribution.setStretch(0, 1)
        distribution.setStretch(1, 1)
        self._distribution_layout = distribution
        self._distribution_widget = distribution_widget
        self._distribution_area = distribution_widget
        self.content.addWidget(distribution_widget)
        self.content.addStretch()
        self._rendered_result_id = id(result)
        self._apply_responsive_layout(self.width())

    def _apply_responsive_layout(self, width):
        compact = width < 760
        self.title_label.setFixedHeight(self.title_label.sizeHint().height())
        self.subtitle_label.setMinimumWidth(0)
        self.subtitle_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Fixed
        )
        content_width = max(1, width - (32 if compact else 64))
        if not compact:
            content_width = max(
                1, content_width - self.refresh_button.minimumWidth() - 6
            )
        subtitle_height = self.subtitle_label.heightForWidth(content_width)
        if subtitle_height > 0:
            self.subtitle_label.setFixedHeight(subtitle_height)
        if compact:
            self.header_widget.setMinimumHeight(
                self.title_label.minimumHeight()
                + self.subtitle_label.minimumHeight()
                + self.refresh_button.minimumHeight()
                + 34
            )
        else:
            self.header_widget.setMinimumHeight(0)
        self.root_layout.setContentsMargins(
            16 if compact else 36,
            16 if compact else 20,
            16 if compact else 28,
            12 if compact else 16,
        )
        header = getattr(self, "header_layout", None)
        if header is not None:
            header.setDirection(
                QBoxLayout.TopToBottom
                if compact
                else QBoxLayout.LeftToRight
            )
            header.setSpacing(14 if compact else 6)
            header.setSpacing(14 if compact else 6)
        metrics = getattr(self, "_metrics_layout", None)
        if metrics is not None:
            metrics.setDirection(
                QBoxLayout.TopToBottom
                if width < 820
                else QBoxLayout.LeftToRight
            )
        distribution = getattr(self, "_distribution_layout", None)
        if distribution is None:
            return
        distribution.setDirection(
            QBoxLayout.TopToBottom
            if width < 1100
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
        self.main_window = main_window
        self.research_result: Dict[str, Any] = {}
        self._explanation_thread: QThread | None = None
        self._explanation_worker: ResearchGapExplanationWorker | None = None
        self._attempted_explanation_this_session = False
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
        self.explain_button = QPushButton("Jelaskan dengan Gemini")
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
        compact = width < 760
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
        while self.content.count():
            item = self.content.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

    def show_empty_state(self):
        self.research_result = {}
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
        self.research_result = result if isinstance(result, dict) else {}
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
        self.explain_button.setEnabled(bool(items) and self._explanation_thread is None)
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
        metrics.addWidget(InfoCard("Penjelasan", explanation_status, "Memakai key Gemini tersimpan bila tersedia"))
        self.content.addLayout(metrics)

        if self._explanation_thread is not None:
            self.content.addWidget(LoadingCard(
                "Menjelaskan kandidat gap",
                "Gemini sedang menyusun penjelasan sederhana dan rekomendasi penelitian berdasarkan temuan ini.",
                self.container,
            ))
        elif isinstance(explanation, dict) and explanation.get("text"):
            self._add_explanation_card(
                explanation.get("text", ""), explanation_source,
                explanation.get("reason", ""),
            )
        else:
            self._add_explanation_card(
                "Tekan tombol Jelaskan dengan Gemini untuk memahami arti temuan, bukti dan batasannya, serta saran penelitian lanjutan.",
                "menunggu",
            )

        heading = QLabel("Kandidat yang ditemukan dari literatur")
        heading.setObjectName("sectionTitle")
        self.content.addWidget(heading)
        if not items:
            empty = QLabel("Belum ada kandidat gap yang terdeteksi. Periksa kembali jumlah dan kecocokan paper yang ditemukan.")
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
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        title = QLabel("Penjelasan dan saran penelitian")
        title.setObjectName("sectionTitle")
        source_names = {
            "gemini": "Gemini menggunakan API key tersimpan",
            "local": "Penjelasan lokal (Gemini tidak tersedia)",
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
        explanation = self.research_result.get("gap_explanation", {})
        has_key = bool(get_saved_api_key())
        needs_generation = not isinstance(explanation, dict) or not explanation.get("text")
        needs_generation = needs_generation or explanation.get("source") == "error"
        needs_generation = needs_generation or (
            explanation.get("source") == "local" and has_key
        )
        if self.research_result and needs_generation and not self._attempted_explanation_this_session:
            self.explain_gaps()

    def explain_gaps(self):
        gaps = self.research_result.get("gaps", {})
        if not self.research_result or self._explanation_thread is not None:
            return
        self._attempted_explanation_this_session = True
        self.explain_button.setEnabled(False)
        self._explanation_thread = QThread(self)
        self._explanation_worker = ResearchGapExplanationWorker(gaps)
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
        gaps = self.research_result.get("gaps", {})
        if isinstance(gaps, dict):
            items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items") or []
        else:
            items = gaps if isinstance(gaps, list) else []
        self.explain_button.setEnabled(bool(items))
        if self.research_result:
            self.update_from_result(self.research_result)



class ResearchReportToolPage(QWidget):
    """Preview and direct PDF export for a complete research workflow."""

    def __init__(self, main_window):
        super().__init__()
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
        self.preview.setReadOnly(True)
        self.preview.setOpenLinks(False)
        self.preview.setOpenExternalLinks(False)
        root.addWidget(self.preview, 1)
        self.page_scroll.setWidget(content)
        self.show_empty_state()

    def _apply_responsive_layout(self, width):
        compact = width < 760
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
            QBoxLayout.TopToBottom if width < 480 else QBoxLayout.LeftToRight
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
            QMessageBox.warning(self, "Report Generation Error", f"Terjadi kesalahan saat membuat laporan: {exc}")

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
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("researchInsightsTabs")
        self.tabs.addTab(landscape_page, "Landscape")
        self.tabs.addTab(gap_page, "Research Gap")
        root.addWidget(self.tabs)
        self.landscape_page = landscape_page
        self.gap_page = gap_page
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        if self.tabs.widget(index) is self.gap_page:
            self.gap_page.ensure_explanation()


class AboutPage(QWidget):
    """Application information, authorship, licensing, and privacy notes."""

    def __init__(self, parent=None):
        super().__init__(parent)
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

        heading = QLabel("About")
        heading.setObjectName("aboutPageTitle")
        content.addWidget(heading)

        subtitle = QLabel(
            "Informasi singkat tentang aplikasi, pengembang, dan lisensinya."
        )
        subtitle.setObjectName("aboutPageSubtitle")
        subtitle.setWordWrap(True)
        content.addWidget(subtitle)

        hero = QFrame()
        hero.setObjectName("aboutHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(18)

        logo = QLabel()
        logo.setObjectName("aboutLogo")
        logo.setFixedSize(76, 76)
        logo.setAlignment(Qt.AlignCenter)
        logo_path = _resource_path("assets/logo.jpg")
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                logo.setPixmap(pixmap.scaled(76, 76, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText("DR")
        hero_layout.addWidget(logo, 0, Qt.AlignVCenter)

        identity = QVBoxLayout()
        identity.setSpacing(5)
        app_name = QLabel("Dataset Research")
        app_name.setObjectName("aboutHeroTitle")
        identity.addWidget(app_name)
        app_description = QLabel(
            "Analisis dataset dan dukungan riset dalam satu aplikasi desktop."
        )
        app_description.setObjectName("aboutBody")
        app_description.setWordWrap(True)
        identity.addWidget(app_description)
        version_label = QLabel("Versi 4.0.0")
        version_label.setObjectName("aboutVersion")
        identity.addWidget(version_label)
        hero_layout.addLayout(identity, 1)
        content.addWidget(hero)

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)
        content.addLayout(cards)
        self._about_cards_layout = cards
        self._about_cards = []

        def add_card(row, column, title, body):
            card = QFrame()
            card.setObjectName("aboutInfoCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(18, 16, 18, 18)
            card_layout.setSpacing(8)
            card_title = QLabel(title)
            card_title.setObjectName("aboutCardTitle")
            card_layout.addWidget(card_title)
            card_body = QLabel(body)
            card_body.setObjectName("aboutBody")
            card_body.setWordWrap(True)
            card_layout.addWidget(card_body)
            card_layout.addStretch()
            cards.addWidget(card, row, column)
            self._about_cards.append(card)

        add_card(0, 0, "Pengembang", "Abdul Muhis\nSains Data, UIN K.H. Abdurrahman Wahid Pekalongan")
        add_card(
            0, 1, "Copyright",
            "© 2026 UIN HIJAU v5.\n"
            "Teks sementara untuk pratinjau tampilan.",
        )
        add_card(
            1, 0, "Data dan privasi",
            "Telemetri penggunaan hanya memuat event, versi aplikasi, status, dan durasi. "
            "File dataset, path, hasil analisis, dan API key tidak masuk ke telemetri. "
            "Fitur Gemini mengirim permintaan ke layanan Google saat digunakan.",
        )
        add_card(
            1, 1, "Ruang lingkup aplikasi",
            "Profil dan kualitas dataset, evaluasi machine learning, pencarian paper, "
            "research landscape dan gap, serta laporan penelitian.",
        )
        content.addStretch()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cards = getattr(self, "_about_cards_layout", None)
        if cards is None:
            return
        columns = 2 if event.size().width() >= 1000 else 1
        for card in self._about_cards:
            cards.removeWidget(card)
        for index, card in enumerate(self._about_cards):
            cards.addWidget(card, index // columns, index % columns)


class CopyrightPage(QWidget):
    """Preview-only copyright details for the team named in the top bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
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

        heading = QLabel("Copyright")
        heading.setObjectName("aboutPageTitle")
        content.addWidget(heading)

        subtitle = QLabel("Informasi hak cipta untuk pratinjau aplikasi.")
        subtitle.setObjectName("aboutPageSubtitle")
        subtitle.setWordWrap(True)
        content.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("aboutInfoCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 18)
        card_layout.setSpacing(8)

        card_title = QLabel("Hak Cipta")
        card_title.setObjectName("aboutCardTitle")
        card_layout.addWidget(card_title)

        copyright_text = QLabel(
            "© 2026 UIN HIJAU v5.\n"
            "Teks sementara untuk pratinjau tampilan."
        )
        copyright_text.setObjectName("aboutBody")
        copyright_text.setWordWrap(True)
        card_layout.addWidget(copyright_text)
        card_layout.addStretch(1)
        content.addWidget(card)
        content.addStretch(1)


class HelpPage(QWidget):
    """Panduan penggunaan yang bisa dibuka dari menu Help."""

    def __init__(self, main_window):
        super().__init__()
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

        heading = QLabel("Help")
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
                "periksa nama dataset, pemisah kolom, encoding, dan kolom target. Pastikan pilihan "
                "First row contains column names sesuai isi file, lalu pilih Upload & Preview.",
                "Buka Dataset",
                self.main_window.open_datasets_page,
            ),
            (
                "2. Cari dataset online",
                "Buka Dataset Search, ketik kata kunci, pilih Kaggle dataset, Hugging Face dataset, "
                "atau All providers, lalu tekan Search. Pilih berkas CSV yang tersedia dan tekan "
                "Download & Analyze untuk mengunduh dan membukanya.",
                "Buka Dataset Search",
                self.main_window.open_dataset_search_page,
            ),
            (
                "3. Pratinjau dan rapikan data",
                "Di halaman Dataset, periksa beberapa baris, daftar kolom, nilai kosong, dan duplikat. "
                "Gunakan pilihan Quick Cleaning yang tersedia bila diperlukan, lalu pilih "
                "Continue to Analysis.",
                "Buka pratinjau Dataset",
                self.main_window.open_datasets_page,
            ),
            (
                "4. Jalankan analisis dataset",
                "Pada Analysis, tekan Run Analysis. Tinjau ringkasan dataset, statistik kolom, "
                "kualitas data, nilai kosong, duplikat, pencilan, dan hubungan antar kolom. "
                "Gunakan hasilnya untuk menentukan langkah berikutnya.",
                "Buka Analysis",
                self.main_window.open_analysis_page,
            ),
            (
                "5. Tinjau ML Intelligence",
                "Setelah dataset dimuat, buka ML Intelligence untuk melihat perkiraan jenis tugas "
                "machine learning dan metode yang disarankan. Periksa alasan dan keterbatasan "
                "rekomendasi sebelum memilih metode penelitian.",
                "Buka ML Intelligence",
                self.main_window.open_ml_page,
            ),
            (
                "6. Cari referensi penelitian",
                "Setelah analisis tersedia, buka Academic Research dan tekan Run Academic Research. "
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
        columns = 2 if event.size().width() >= 1000 else 1
        if columns == self._card_columns:
            return
        for card in self.cards:
            self.cards_layout.removeWidget(card)
        for index, card in enumerate(self.cards):
            self.cards_layout.addWidget(card, index // columns, index % columns)
        self._card_columns = columns


class MainWindow(QMainWindow):
    """Main application window for Dataset Research.

    The UI pages are intentionally kept in this single module so the
    desktop interface has one clear entry point while the analytical
    backend remains completely independent.
    """

    def __init__(self):
        super().__init__()

        # Load the stylesheet before building the interface so every
        # widget is styled as soon as it is created.
        self._load_stylesheet()

        self.setWindowTitle("Dataset Research")
        app_icon = _resource_path("assets/dataset_research.ico")
        if app_icon.exists():
            self.setWindowIcon(QIcon(str(app_icon)))

        # Keep the window within the usable screen area, including display scaling.
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            max_w = max(480, avail.width() - 32)
            max_h = max(400, avail.height() - 48)
            min_w = min(760, max(520, int(max_w * 0.55)))
            min_h = min(540, max(400, int(max_h * 0.55)))
            initial_w = min(1920, max(min_w, int(max_w * 0.98)))
            initial_h = min(1080, max(min_h, int(max_h * 0.96)))
            self._responsive_min_size = (min_w, min_h)
            self._initial_window_size = (initial_w, initial_h)
            self.resize(initial_w, initial_h)
            self.setMinimumSize(min_w, min_h)
        else:
            self._responsive_min_size = (560, 420)
            self._initial_window_size = (1100, 680)
            self.resize(1100, 680)
            self.setMinimumSize(560, 420)

        self.current_dataset = None
        self.current_file_path = None
        self.analysis_result = {}
        self._shutdown_pending = False
        self._shutdown_idle_checks = 0
        self._session_data_cleaned = False
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

        self._build_window()
        self._restore_latest_project()
        self.setMinimumSize(*self._responsive_min_size)
        self.resize(*self._initial_window_size)
        get_telemetry().track("app_open", status="started")

    def _active_worker_threads(self):
        """Return running background workers that could still write session data."""
        worker_fields = {
            "dataset_search_page": ("_search_thread", "_download_thread"),
            "analysis_page": ("_analysis_thread", "_gemini_thread"),
            "ml_page": ("_ml_thread",),
            "research_page": ("_research_thread", "_sinta_thread"),
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
        return active

    def _poll_shutdown(self):
        if self._active_worker_threads():
            self._shutdown_idle_checks = 0
            self.statusBar().showMessage(
                "Menunggu proses yang sedang berjalan sebelum menghapus data sesi…"
            )
            return

        # Give queued worker-finished slots a short chance to run before cleanup.
        self._shutdown_idle_checks += 1
        if self._shutdown_idle_checks < 2:
            return

        self._shutdown_timer.stop()
        try:
            self._clear_session_workspace()
        except Exception as error:
            self._shutdown_pending = False
            self._shutdown_idle_checks = 0
            self.setWindowTitle("Dataset Research")
            if self.centralWidget() is not None:
                self.centralWidget().setEnabled(True)
            self.statusBar().clearMessage()
            QMessageBox.warning(
                self,
                "Data sesi belum terhapus",
                "Aplikasi tidak jadi ditutup karena data sesi lokal belum dapat "
                f"dibersihkan. Periksa penyimpanan lalu coba tutup kembali.\n\n{error}",
            )
            return

        self.close()

    def _clear_session_workspace(self):
        """Delete app-managed project copies/results while preserving user files and settings."""
        report_page = getattr(self, "report_tool_page", None)
        if report_page is not None:
            report_tempdir = getattr(report_page, "_chart_tempdir", None)
            if report_tempdir is not None:
                report_tempdir.cleanup()
            report_page._chart_tempdir = None
            report_page.chart_dir = None
            report_page.current_html = ""
            report_page._preview_signature = None
            report_page._workflow_source_objects = ()
            report_page._generating_report_signature = None

        for project in self.repository.list_projects():
            project_id = project.get("id")
            if project_id is None:
                continue
            project_id = int(project_id)
            projects_root = (Path(self.version_manager.data_dir) / "projects").resolve()
            project_dir = (projects_root / str(project_id)).resolve()
            if projects_root not in project_dir.parents:
                raise RuntimeError("Lokasi data proyek berada di luar folder kerja aplikasi.")
            self.version_manager.delete_project_files(project_id)

        # Cascades clear each project's workflow/results; journal screening is cleared too.
        self.database.clear_session_data()

        self.current_dataset = None
        self.current_file_path = None
        self.current_project_id = None
        self.analysis_result = {}
        self._session_data_cleaned = True

    def closeEvent(self, event):
        if self._session_data_cleaned:
            event.accept()
            return

        event.ignore()
        if self._shutdown_pending:
            return

        self._shutdown_pending = True
        self._shutdown_idle_checks = 0
        self.setWindowTitle("Dataset Research — Menutup sesi")
        if self.centralWidget() is not None:
            self.centralWidget().setEnabled(False)
        self.statusBar().showMessage(
            "Menyelesaikan proses dan menghapus dataset serta hasil kerja sesi…"
        )
        self._shutdown_timer.start()


    # =========================================================
    # STYLESHEET
    # =========================================================

    def _load_stylesheet(self):
        """Load the UI stylesheet located beside this module."""
        qss_path = Path(__file__).with_name("app.qss")
        theme_path = Path(__file__).with_name("styles") / "theme.qss"

        if not qss_path.exists():
            print(f"Warning: stylesheet not found: {qss_path}")
            return

        try:
            assets_root = Path(__file__).with_name("assets")
            for font_file in (
                assets_root / "fonts" / "Inter-Variable.ttf",
                assets_root / "fonts" / "PlusJakartaSans-Variable.ttf",
                assets_root / "fonts" / "IntelOneMono-Medium.ttf",
            ):
                if font_file.exists():
                    QFontDatabase.addApplicationFont(str(font_file))

            stylesheet = qss_path.read_text(encoding="utf-8")
            if theme_path.exists():
                theme = theme_path.read_text(encoding="utf-8")
                tokens = dict(re.findall(r"/\*\s*@([\w-]+):\s*(.*?)\s*\*/", theme))
                self._theme_tokens = tokens
                theme = re.sub(
                    r"var\(--([\w-]+)\)",
                    lambda match: tokens.get(match.group(1), match.group(0)),
                    theme,
                )
                canvas = tokens.get("canvas", "#f1f7ff")
                palette = self.palette()
                for role in (QPalette.Window, QPalette.Base, QPalette.AlternateBase):
                    palette.setColor(role, QColor(canvas))
                self.setPalette(palette)
                stylesheet += "\n\n" + theme
            self.setStyleSheet(stylesheet)
        except OSError as error:
            print(f"Warning: failed to load stylesheet: {error}")

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

        self.dashboard = DashboardPage(self)
        self.upload_page = UploadPage(self)
        self.dataset_search_page = DatasetSearchPage(self)
        self.dataset_preview_page = DatasetPreviewPage(self)
        self.datasets_tabs = QTabWidget()
        self.datasets_tabs.setObjectName("datasetsTabs")
        self.datasets_tabs.setAttribute(Qt.WA_StyledBackground, True)
        self.datasets_tabs.addTab(self.upload_page, "Upload from computer")
        self.datasets_tabs.addTab(self.dataset_search_page, "Find online")
        self.datasets_tabs.addTab(self.dataset_preview_page, "Preview")
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

        self.analysis_page = AnalysisPage(self)
        self.editor_analysis_page = EditorAnalysisPage(self)
        self.ml_page = MLIntelligencePage(self)
        self.research_page = ResearchPage(self)
        self.papers_tool_page = PapersToolPage(self)
        self.landscape_tool_page = ResearchLandscapeToolPage(self)
        self.gap_tool_page = ResearchGapToolPage(self)
        self.research_insights_page = ResearchInsightsPage(
            self.landscape_tool_page, self.gap_tool_page
        )
        self.report_tool_page = ResearchReportToolPage(self)
        self.about_page = AboutPage()
        self.copyright_page = CopyrightPage()
        self.help_page = HelpPage(self)

        self.pages = {
            "dashboard": self.dashboard,
            "dataset": self.datasets_page,
            "dataset_search": self.datasets_page,
            "analysis": self.analysis_page,
            "editor_analysis": self.editor_analysis_page,
            "ml": self.ml_page,
            "research": self.research_page,
            "papers": self.papers_tool_page,
            "insights": self.research_insights_page,
            # Landscape and gap are tabs inside Research Insights. Reusing
            # their parent page here keeps Qt from pulling the tab widgets
            # out into separate entries in the main page stack.
            "landscape": self.research_insights_page,
            "gap": self.research_insights_page,
            "report": self.report_tool_page,
            "about": self.about_page,
            "copyright": self.copyright_page,
            "help": self.help_page,
        }

        for page in self.pages.values():
            if self.page_stack.indexOf(page) < 0:
                self.page_stack.addWidget(page)

        self._page_titles = {
            "dashboard": "Dashboard",
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
            "about": "About",
            "copyright": "Copyright",
            "help": "Help",
        }

        for widget in (root, content, self.page_stack, self.datasets_page, self.datasets_tabs, self.datasets_tab_pages):
            if widget is None:
                continue
            self._apply_canvas_background(widget)
        seen_page_ids = set()
        for page in self.pages.values():
            if id(page) not in seen_page_ids:
                seen_page_ids.add(id(page))
                self._apply_canvas_background(page)
        for scroll_area in self.findChildren(QScrollArea):
            self._apply_canvas_background(scroll_area)
            self._apply_canvas_background(scroll_area.viewport())

        self.open_dashboard()

    # =========================================================
    # SIDEBAR
    # =========================================================

    def _build_sidebar(self):
        sidebar_scroll = QScrollArea()
        self.sidebar_scroll = sidebar_scroll
        sidebar_scroll.setObjectName("sidebarScroll")
        sidebar_scroll.setFixedWidth(418)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        self.sidebar_layout = layout
        layout.setContentsMargins(28, 78, 28, 24)
        layout.setSpacing(4)

        brand = QHBoxLayout()
        brand.setSpacing(14)
        brand.setContentsMargins(13, 0, 0, 0)
        logo = QLabel("D")
        logo.setObjectName("brandLogo")
        logo.setFixedSize(82, 82)
        logo.setAlignment(Qt.AlignCenter)
        logo_path = _resource_path("assets/logo.jpg")
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path)).copy(275, 130, 560, 650)
            if not pixmap.isNull():
                logo.setPixmap(pixmap.scaled(82, 82, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        brand_text_container = QWidget()
        self.brand_text_container = brand_text_container
        brand_text = QVBoxLayout(brand_text_container)
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(2)
        title = QLabel("DORA")
        title.setObjectName("brandTitle")
        subtitle = QLabel("Data Overview & Research Assistant")
        subtitle.setObjectName("brandSubtitle")
        subtitle.setWordWrap(True)
        brand_text_container.setMaximumWidth(210)
        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)
        brand.addWidget(logo)
        brand.addWidget(brand_text_container, 1)
        layout.addLayout(brand)
        layout.addSpacing(72)

        self.nav_buttons = {}
        self.nav_text_widgets = []
        self.sidebar_section_labels = []
        self._add_nav_section(layout, "WORKSPACE")
        self._add_nav_button(layout, "dataset_search", "dataset-search.svg", "Dataset Search")
        self._add_nav_button(layout, "dataset", "dataset.svg", "Dataset")
        self._add_nav_button(layout, "analysis", "analysis.svg", "Analysis")
        self._add_nav_button(layout, "editor_analysis", "file-text.svg", "Editor Analysis")
        self._add_nav_button(layout, "ml", "ml-intelligence.svg", "ML Intelligence")
        self._add_nav_button(layout, "dashboard", "overview.svg", "Overview")
        layout.addSpacing(24)
        self._add_nav_section(layout, "RESEARCH TOOLS")
        self._add_nav_button(layout, "research", "academic-research.svg", "Academic Research")
        self._add_nav_button(layout, "papers", "papers.svg", "Papers")
        self._add_nav_button(layout, "landscape", "research-landscape.svg", "Research Landscape")
        self._add_nav_button(layout, "gap", "research-gap.svg", "Research Gap")
        self._add_nav_button(layout, "report", "research-report.svg", "Research Report")
        layout.addStretch()

        sidebar_scroll.setWidget(sidebar)
        return sidebar_scroll

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
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(11)

        icon_label = QLabel()
        icon_label.setObjectName("navIcon")
        icon_label.setFixedSize(28, 28)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_path = _resource_path(Path("assets") / "icons" / icon)
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path))
            if not pixmap.isNull():
                icon_label.setPixmap(pixmap.scaled(26, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
            "dataset": self.open_datasets_page,
            "dataset_search": self.open_dataset_search_page,
            "analysis": self.open_analysis_page,
            "editor_analysis": self.open_editor_analysis_page,
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

    # =========================================================
    # TOPBAR
    # =========================================================

    def _build_topbar(self):
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(124)

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(34, 0, 42, 0)
        self.topbar_layout = layout
        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("topbarTitle")
        self.page_title.hide()
        layout.addStretch()

        about = QPushButton("About")
        about.setObjectName("topbarLink")
        about.setIcon(QIcon(str(_resource_path("assets/icons/about.svg"))))
        about.setProperty("active", False)
        about.clicked.connect(self.open_about)
        layout.addWidget(about)
        self.topbar_links = {"about": about}
        self.topbar_link_labels = {"about": "About"}
        for key, label, icon_name in (
            ("copyright", "Copyright", "copyright.svg"),
            ("help", "Help", "help.svg"),
        ):
            link = QPushButton(label)
            link.setObjectName("topbarLink")
            link.setIcon(QIcon(str(_resource_path(Path("assets") / "icons" / icon_name))))
            link.setFlat(True)
            link.setProperty("active", False)
            self.topbar_links[key] = link
            self.topbar_link_labels[key] = label
            if key == "copyright":
                link.clicked.connect(self.open_copyright)
            elif key == "help":
                link.clicked.connect(self.open_help)
            layout.addWidget(link)
        divider = QFrame()
        divider.setObjectName("topbarDivider")
        divider.setFixedSize(1, 34)
        self.topbar_divider = divider
        layout.addSpacing(10)
        layout.addWidget(divider)
        version = QLabel("2026 UIN HIJAU v5")
        version.setObjectName("topbarVersion")
        self.topbar_version = version
        layout.addSpacing(18)
        layout.addWidget(version)
        return topbar

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sidebar = getattr(self, "sidebar_scroll", None)
        if sidebar is None:
            return

        sidebar_width = min(418, max(76, round(event.size().width() * 0.23)))
        compact = sidebar_width < 288
        sidebar.setFixedWidth(sidebar_width)
        self.sidebar_layout.setContentsMargins(
            8 if compact else 28,
            20 if compact else 78,
            8 if compact else 28,
            16 if compact else 24,
        )
        self.brand_text_container.setVisible(not compact)
        for label in self.nav_text_widgets:
            label.setVisible(not compact)
        for label in self.sidebar_section_labels:
            label.setVisible(not compact)

        compact_topbar = event.size().width() < 850
        self.topbar.setFixedHeight(78 if event.size().height() < 620 else 124)
        self.topbar_layout.setContentsMargins(
            12 if compact_topbar else 34,
            0,
            12 if compact_topbar else 42,
            0,
        )
        for key, link in self.topbar_links.items():
            link.setText("" if compact_topbar else self.topbar_link_labels[key])
            link.setToolTip(self.topbar_link_labels[key])
            link.setIconSize(QSize(20 if compact_topbar else 22, 20 if compact_topbar else 22))
        self.topbar_divider.setVisible(not compact_topbar)
        self.topbar_version.setVisible(not compact_topbar)

    # =========================================================
    # NAVIGATION
    # =========================================================

    def _show_page(self, key):
        page = self.pages[key]
        self.page_stack.setCurrentWidget(page)
        self.page_title.setText(self._page_titles[key])
        self.set_active_button(key)
        self._set_topbar_active(key)

    def _set_topbar_active(self, page_key):
        active_link = page_key if page_key in ("about", "copyright", "help") else None
        for key, link in self.topbar_links.items():
            link.setProperty("active", key == active_link)
            link.style().unpolish(link)
            link.style().polish(link)
            link.update()

    def open_dashboard(self):
        self._show_page("dashboard")

    def open_upload_page(self):
        self.datasets_tabs.setCurrentWidget(self.upload_page)
        self._show_page("dataset")

    def open_datasets_page(self):
        target = self.dataset_preview_page if self.current_dataset is not None else self.upload_page
        self.datasets_tabs.setCurrentWidget(target)
        self._show_page("dataset")

    def open_analysis_page(self):
        if self.current_dataset is not None:
            self.analysis_page.update_dataset(
                self.current_dataset,
                self._current_filename(),
            )
        self._show_page("analysis")

    def open_editor_analysis_page(self):
        self.editor_analysis_page.set_dataset(
            self.current_dataset,
            self._current_filename() if self.current_dataset is not None else "",
        )
        self._show_page("editor_analysis")

    def open_ml_page(self):
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
        if self.current_dataset is None or not self.analysis_result:
            self.research_page.show_empty_state()
        self._show_page("research")

    def open_dataset_search_page(self):
        self.datasets_tabs.setCurrentWidget(self.dataset_search_page)
        self._show_page("dataset_search")

    def open_papers_tool(self):
        if not self.analysis_result or not self.research_page.research_result:
            self.research_page.show_empty_state()
        else:
            self.papers_tool_page.update_from_result(self.research_page.research_result)
        self._show_page("papers")

    def open_insights_page(self):
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

    def open_landscape_tool(self):
        self.open_insights_page()
        self.research_insights_page.tabs.setCurrentWidget(self.landscape_tool_page)
        self._show_page("landscape")

    def open_gap_tool(self):
        self.open_insights_page()
        self.research_insights_page.tabs.setCurrentWidget(self.gap_tool_page)
        self._show_page("gap")

    def open_report_tool(self):
        has_data = (
            self.current_dataset is not None
            or bool(self.analysis_result)
            or bool(getattr(self.ml_page, "last_intelligence_result", {}))
            or bool(getattr(self.research_page, "research_result", {}))
        )
        if not has_data:
            self.report_tool_page.show_empty_state()
        else:
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
            button.setProperty("active", active)
            style = button.style()
            style.unpolish(button)
            style.polish(button)
            button.update()

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
        return self.current_project_id

    def _restore_latest_project(self):
        """Restore the newest usable local project and its completed features."""
        for project in self.repository.list_projects():
            project_id = project.get("id")
            if project_id is None or not self.version_manager.has_versions(project_id):
                continue
            try:
                dataframe = self.version_manager.load_current_dataframe(project_id)
            except Exception as error:
                print(f"Warning: could not restore project {project_id}: {error}")
                continue

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
            state = self.repository.get_workflow_state(project_id) or {}

            self.analysis_result = state.get("analysis") or {}
            self.dashboard.update_dataset(dataframe, filename)
            self.dataset_preview_page.update_dataset(dataframe, filename)
            self.analysis_page.update_dataset(dataframe, filename)

            task_result = state.get("ml_task") or {}
            methods_result = state.get("ml_methods") or {}
            if task_result and methods_result:
                self.ml_page.last_task_result = task_result
                self.ml_page.last_method_result = {
                    key: value for key, value in methods_result.items()
                    if key != "_intelligence_result"
                }
                self.ml_page.last_intelligence_result = methods_result.get("_intelligence_result") or {}
                self.ml_page.display_result(task_result, self.ml_page.last_method_result)
            elif self.analysis_result:
                self.ml_page.update_dataset()

            research_result = state.get("research") or {}
            if research_result:
                self.research_page.restore_result(research_result)
                self.dashboard.update_research_result(research_result)
                self.papers_tool_page.update_from_result(research_result)

            self.dashboard.update_workflow_state(
                has_dataset=True,
                analysis_done=bool(self.analysis_result),
                ml_done=bool(task_result and methods_result),
                research_done=bool(research_result),
            )

            return

    def update_dataset_state(self):
        self.analysis_result = {}

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

        self.analysis_page.update_dataset(
            self.current_dataset,
            filename,
        )

        self.ml_page.show_empty_state()
        self.research_page.show_empty_state()
        self.papers_tool_page.show_empty_state()
        self.landscape_tool_page.show_empty_state()
        self.gap_tool_page.show_empty_state()
        self.report_tool_page.show_empty_state()
        self.dashboard.update_workflow_state(
            has_dataset=self.current_dataset is not None,
            analysis_done=False,
            ml_done=False,
            research_done=False,
        )

    # =========================================================
    # DATASET ANALYSIS
    # =========================================================

    def prepare_dataset_analysis(self):
        dataframe = self.current_dataset
        if dataframe is None:
            raise ValueError("No dataset is currently loaded.")

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
        self.ml_page.last_task_result = {}
        self.ml_page.last_method_result = {}
        self.ml_page.last_intelligence_result = {}
        self.ml_page.show_ready_state()
        self.research_page.show_empty_state()
        self.dashboard.update_workflow_state(
            has_dataset=True, analysis_done=True, ml_done=False, research_done=False
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
