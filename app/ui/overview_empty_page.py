"""Page 4 welcome state for the DORA Overview screen."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QRect, Qt, QSize, Signal
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.style_utils import render_icon_pixmap


class OverviewStepCard(QFrame):
    """Keyboard-accessible workflow card connected to its active page."""

    clicked = Signal()

    def __init__(self, title: str, callback, parent=None):
        super().__init__(parent)
        self.setObjectName("overviewEmptyStepCard")
        self.setProperty("interactive", True)
        self.setAccessibleName(title)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(callback)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def _asset_path(relative: str) -> Path:
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return root / "app" / "ui" / relative
    return Path(__file__).resolve().parent / relative


class DatasetIllustrationCrop(QWidget):
    """Render the exact Figma source image through the crop used by its frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("overviewEmptyIllustration")
        self.setMinimumSize(112, 78)
        self.setMaximumSize(257, 178)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._crop = QPixmap()
        self._scaled_cache = QPixmap()
        self._cache_size = QSize()
        image = QPixmap(str(_asset_path("assets/overview-dataset-exploration.png")))
        if not image.isNull():
            # Keep the crop used by the original frame, then scale only this small
            # crop when the available layout size changes.
            source = image.scaled(
                1536, 1024, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            self._crop = source.copy(QRect(304, 388, 257, 178))

    def paintEvent(self, event):
        del event
        if self._crop.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        target_size = self.size()
        if target_size != self._cache_size:
            self._scaled_cache = self._crop.scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._cache_size = target_size
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        x = (self.width() - self._scaled_cache.width()) // 2
        y = (self.height() - self._scaled_cache.height()) // 2
        painter.drawPixmap(x, y, self._scaled_cache)


class Page4OverviewEmptyPage(QWidget):
    """Figma-styled Overview page for empty and populated dataset states."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.icons = _asset_path("assets/icons")
        self._build_ui()
        self._apply_responsive_layout(self.width())

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("overviewEmptyScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.container.setObjectName("overviewEmptyContainer")
        layout = QVBoxLayout(self.container)
        self.content_layout = layout
        layout.setContentsMargins(17, 10, 18, 14)
        layout.setSpacing(12)
        layout.addWidget(self._build_hero())
        layout.addWidget(self._build_dataset_welcome())
        layout.addWidget(self._build_getting_started())
        layout.addWidget(self._build_resources())
        layout.addStretch(1)
        self.scroll.setWidget(self.container)
        root.addWidget(self.scroll)

    def _icon(self, label: QLabel, filename: str, size: int):
        aliases = {
            "figma-overview-step-upload.svg": ("upload.svg", "#00B96A"),
            "figma-overview-dataset.svg": ("dataset.svg", "#9435FF"),
            "figma-overview-analysis.svg": ("analysis.svg", "#FF9000"),
            "figma-overview-ml.svg": ("ml-intelligence.svg", "#9435FF"),
            "figma-overview-report.svg": ("research-report.svg", "#FF3D72"),
            "figma-overview-chevron.svg": ("overview-arrow-right.svg", "#0060FF"),
        }
        asset_name, tint = aliases.get(filename, (filename, None))
        path = self.icons / asset_name
        if path.exists():
            label.setPixmap(
                render_icon_pixmap(path, size, self.devicePixelRatioF(), tint)
            )

    def _button(self, title: str, icon_name: str, *, primary: bool, callback):
        button = QPushButton(title)
        button.setObjectName("overviewEmptyPrimaryButton" if primary else "overviewEmptySecondaryButton")
        button.setCursor(Qt.PointingHandCursor)
        widths = {
            "Cari Dataset Sekarang": 230,
            "Upload Dataset Sendiri": 223,
            "Cari Dataset": 158,
            "Upload Dataset": 165,
        }
        button.setProperty("preferredWidth", widths.get(title, 0))
        button.setMinimumHeight(45)
        button.setMinimumWidth(widths.get(title, 0))
        button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._icon_for_button(button, icon_name, 20, primary=primary)
        button.clicked.connect(callback)
        return button

    def _icon_for_button(
        self, button: QPushButton, filename: str, size: int, *, primary: bool
    ):
        aliases = {
            "figma-overview-search.svg": (
                "search-white.svg" if primary else "search.svg",
                "#FFFFFF" if primary else "#0060FF",
            ),
            "figma-overview-upload.svg": ("upload.svg", "#0060FF"),
            "figma-overview-analysis.svg": (
                "analysis.svg",
                "#FFFFFF" if primary else "#0060FF",
            ),
        }
        asset_name, tint = aliases.get(filename, (filename, None))
        path = self.icons / asset_name
        if path.exists():
            button.setIcon(
                QIcon(render_icon_pixmap(path, size, self.devicePixelRatioF(), tint))
            )
            button.setIconSize(QSize(size, size))

    def _build_hero(self) -> QFrame:
        hero = QFrame()
        hero.setObjectName("overviewEmptyHero")
        hero.setMinimumHeight(220)
        layout = QVBoxLayout(hero)
        self.hero_layout = layout
        layout.setContentsMargins(28, 21, 28, 24)
        layout.setSpacing(12)
        badge = QLabel("Selamat Datang di DORA")
        badge.setObjectName("overviewEmptyWelcomeBadge")
        badge.setAttribute(Qt.WA_StyledBackground, True)
        layout.addWidget(badge, 0, Qt.AlignLeft)
        title = QLabel("Asisten Riset Berbasis Data untuk Peneliti")
        title.setObjectName("overviewEmptyHeroTitle")
        title.setWordWrap(True)
        self.hero_title = title
        layout.addWidget(title)
        description = QLabel(
            "DORA (Data Overview & Research Assistant) membantu Anda mengelola dataset, "
            "menganalisis data, membangun model machine learning, hingga menemukan wawasan "
            "riset, semua dalam satu workspace yang terintegrasi."
        )
        description.setObjectName("overviewEmptyHeroText")
        description.setWordWrap(True)
        description.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(description)
        self.hero_description = description
        layout.addSpacing(8)
        actions = QHBoxLayout()
        self.hero_actions = actions
        actions.setSpacing(14)
        self.hero_search_button = self._button(
            "Cari Dataset Sekarang", "figma-overview-search.svg", primary=True,
            callback=self.main_window.open_dataset_search_page,
        )
        actions.addWidget(self.hero_search_button)
        self.hero_upload_button = self._button(
            "Upload Dataset Sendiri", "figma-overview-upload.svg", primary=False,
            callback=self.main_window.open_upload_page,
        )
        actions.addWidget(self.hero_upload_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return hero

    def _build_dataset_welcome(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setObjectName("overviewEmptyDatasetWrapper")
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(4, 0, 4, 0)
        wrapper_layout.setSpacing(0)
        self.dataset_wrapper_layout = wrapper_layout
        panel = QFrame()
        panel.setObjectName("overviewEmptyDatasetPanel")
        panel.setMinimumWidth(0)
        panel.setMinimumHeight(204)
        layout = QHBoxLayout(panel)
        self.empty_dataset_layout = layout
        layout.setContentsMargins(30, 18, 16, 18)
        layout.setSpacing(22)
        self.dataset_illustration = DatasetIllustrationCrop()
        layout.addWidget(self.dataset_illustration, 0, Qt.AlignVCenter)

        guidance = QWidget()
        guidance.setObjectName("overviewEmptyDatasetGuidance")
        guidance_layout = QVBoxLayout(guidance)
        guidance_layout.setContentsMargins(0, 20, 0, 0)
        guidance_layout.setSpacing(8)
        title = QLabel("Anda belum memiliki dataset")
        title.setObjectName("overviewEmptyDatasetTitle")
        guidance_layout.addWidget(title)
        description = QLabel(
            "Mulai dengan mencari dataset publik sesuai topik penelitian Anda atau upload "
            "dataset sendiri untuk memulai analisis."
        )
        description.setObjectName("overviewEmptyDatasetText")
        description.setWordWrap(True)
        guidance_layout.addWidget(description)
        actions = QHBoxLayout()
        self.empty_dataset_actions = actions
        actions.setSpacing(12)
        actions.addWidget(self._button(
            "Cari Dataset", "figma-overview-search.svg", primary=True,
            callback=self.main_window.open_dataset_search_page,
        ))
        actions.addWidget(self._button(
            "Upload Dataset", "figma-overview-upload.svg", primary=False,
            callback=self.main_window.open_upload_page,
        ))
        guidance_layout.addLayout(actions)
        guidance_layout.addStretch(1)
        layout.addWidget(guidance, 1)
        self.empty_dataset_guidance = guidance
        self.empty_dataset_search_button = actions.itemAt(0).widget()
        self.empty_dataset_upload_button = actions.itemAt(1).widget()

        benefits = QFrame()
        benefits.setObjectName("overviewEmptyBenefits")
        benefits.setMinimumWidth(0)
        benefits.setMinimumHeight(168)
        benefits.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        benefits_layout = QVBoxLayout(benefits)
        benefits_layout.setContentsMargins(22, 18, 12, 17)
        benefits_layout.setSpacing(15)
        heading = QHBoxLayout()
        heading.setSpacing(10)
        lightbulb = QLabel()
        lightbulb.setObjectName("overviewEmptyBenefitsIcon")
        lightbulb.setFixedSize(22, 22)
        self._icon(lightbulb, "figma-overview-lightbulb.svg", 20)
        heading.addWidget(lightbulb)
        heading_title = QLabel("Mengapa dataset penting?")
        heading_title.setObjectName("overviewEmptyBenefitsTitle")
        heading.addWidget(heading_title)
        heading.addStretch(1)
        benefits_layout.addLayout(heading)
        for text in (
            "Menjadi dasar untuk analisis dan visualisasi data",
            "Digunakan untuk membangun model machine learning",
            "Membantu menemukan insight dan menjawab pertanyaan riset",
            "Bisa menggunakan dataset publik atau dataset Anda sendiri",
        ):
            row = QHBoxLayout()
            row.setSpacing(10)
            check = QLabel()
            check.setObjectName("overviewEmptyBenefitCheck")
            check.setFixedSize(16, 16)
            self._icon(check, "figma-overview-check.svg", 16)
            label = QLabel(text)
            label.setObjectName("overviewEmptyBenefitText")
            label.setWordWrap(True)
            row.addWidget(check)
            row.addWidget(label, 1)
            benefits_layout.addLayout(row)
        layout.addWidget(benefits)
        self.empty_dataset_benefits = benefits
        wrapper_layout.addWidget(panel)
        self.empty_dataset_panel = panel
        self.loaded_dataset_panel = self._build_loaded_dataset_panel()
        wrapper_layout.addWidget(self.loaded_dataset_panel)
        return wrapper

    def _build_loaded_dataset_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("overviewEmptyDatasetLoadedPanel")
        panel.setMinimumWidth(0)
        panel.setMinimumHeight(204)
        layout = QHBoxLayout(panel)
        self.loaded_dataset_layout = layout
        layout.setContentsMargins(30, 18, 16, 18)
        layout.setSpacing(28)

        details = QWidget()
        details.setObjectName("overviewEmptyLoadedDetails")
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)
        heading = QHBoxLayout()
        icon = QLabel()
        icon.setFixedSize(23, 23)
        self._icon(icon, "figma-overview-dataset.svg", 22)
        heading.addWidget(icon)
        heading_title = QLabel("Dataset Anda siap digunakan")
        heading_title.setObjectName("overviewEmptyDatasetTitle")
        heading.addWidget(heading_title)
        heading.addStretch(1)
        details_layout.addLayout(heading)

        self.loaded_dataset_name = QLabel("Dataset aktif")
        self.loaded_dataset_name.setObjectName("overviewLoadedDatasetName")
        self.loaded_dataset_name.setWordWrap(True)
        details_layout.addWidget(self.loaded_dataset_name)

        metrics = QHBoxLayout()
        self.loaded_metrics_layout = metrics
        metrics.setSpacing(10)
        self.loaded_metrics = {}
        for key, caption_text in (
            ("rows", "BARIS"),
            ("columns", "KOLOM"),
            ("missing", "NILAI KOSONG"),
        ):
            tile = QFrame()
            tile.setObjectName("overviewLoadedMetric")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(10, 6, 10, 6)
            tile_layout.setSpacing(2)
            caption = QLabel(caption_text)
            caption.setObjectName("overviewLoadedMetricCaption")
            value = QLabel("—")
            value.setObjectName("overviewLoadedMetricValue")
            tile_layout.addWidget(caption)
            tile_layout.addWidget(value)
            metrics.addWidget(tile, 1)
            self.loaded_metrics[key] = value
        details_layout.addLayout(metrics)

        actions = QHBoxLayout()
        self.loaded_actions_layout = actions
        actions.setSpacing(10)
        preview_button = self._button(
            "Buka Dataset",
            "figma-overview-upload.svg",
            primary=False,
            callback=self.main_window.open_datasets_page,
        )
        analysis_button = self._button(
            "Lihat Analisis",
            "figma-overview-analysis.svg",
            primary=True,
            callback=self.main_window.open_analysis_page,
        )
        preview_button.setMinimumWidth(0)
        analysis_button.setMinimumWidth(0)
        preview_button.setMinimumHeight(45)
        analysis_button.setMinimumHeight(45)
        preview_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        analysis_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        actions.addWidget(preview_button)
        actions.addWidget(analysis_button)
        actions.addStretch(1)
        details_layout.addLayout(actions)
        self.loaded_dataset_details = details
        layout.addWidget(details, 1)

        status = QFrame()
        status.setObjectName("overviewEmptyBenefits")
        status.setMinimumWidth(0)
        status.setMinimumHeight(168)
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(22, 18, 12, 17)
        status_layout.setSpacing(14)
        status_title = QLabel("Langkah berikutnya")
        status_title.setObjectName("overviewEmptyBenefitsTitle")
        status_layout.addWidget(status_title)
        self.loaded_analysis_state = QLabel("Analisis: belum dijalankan")
        self.loaded_ml_state = QLabel("ML Intelligence: belum dijalankan")
        for label in (self.loaded_analysis_state, self.loaded_ml_state):
            label.setObjectName("overviewEmptyLoadedStatus")
            status_layout.addWidget(label)
        hint = QLabel("Gunakan tombol analisis untuk memeriksa kualitas dan pola data.")
        hint.setObjectName("overviewEmptyBenefitText")
        hint.setWordWrap(True)
        status_layout.addWidget(hint)
        status_layout.addStretch(1)
        layout.addWidget(status)
        self.loaded_dataset_status = status
        panel.hide()
        return panel

    def set_dataset(self, dataframe, filename: str = "", analysis=None):
        has_dataset = dataframe is not None
        self.empty_dataset_panel.setVisible(not has_dataset)
        self.loaded_dataset_panel.setVisible(has_dataset)
        if not has_dataset:
            return

        analysis = analysis if isinstance(analysis, dict) else {}

        self.loaded_dataset_name.setText(filename or "Dataset aktif")
        self.loaded_metrics["rows"].setText(f"{len(dataframe.index):,}")
        self.loaded_metrics["columns"].setText(f"{len(dataframe.columns):,}")
        missing_rows = (analysis or {}).get("missing_values", [])
        missing_count = 0
        if isinstance(missing_rows, list):
            missing_count = sum(
                int(row.get("missing_count", 0) or 0)
                for row in missing_rows
                if isinstance(row, dict)
            )
        self.loaded_metrics["missing"].setText(f"{missing_count:,}" if analysis else "—")
        self.loaded_analysis_state.setText(
            "Analisis: tersedia" if analysis else "Analisis: belum dijalankan"
        )
        ml_page = getattr(self.main_window, "ml_page", None)
        ml_ready = bool(
            getattr(ml_page, "last_task_result", None)
            and getattr(ml_page, "last_method_result", None)
        )
        self.loaded_ml_state.setText(
            "ML Intelligence: tersedia" if ml_ready else "ML Intelligence: belum dijalankan"
        )

    def _build_getting_started(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("overviewEmptySection")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(11)
        heading = QVBoxLayout()
        heading.setSpacing(3)
        title = QLabel("Mulai dengan 5 Langkah Mudah")
        title.setObjectName("overviewEmptySectionTitle")
        subtitle = QLabel("Ikuti alur berikut untuk melakukan analisis penelitian menggunakan DORA.")
        subtitle.setObjectName("overviewEmptySectionSubtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        layout.addLayout(heading)
        self.steps_layout = QGridLayout()
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setHorizontalSpacing(12)
        self.steps_layout.setVerticalSpacing(10)
        self.step_cards = []
        self.step_arrows = []
        steps = (
            (
                "Upload Dataset",
                "Unggah file data Anda\n(format CSV).",
                "figma-overview-step-upload.svg",
                self.main_window.open_upload_page,
            ),
            (
                "Jelajahi & Pahami Data",
                "Lihat ringkasan, lakukan\neksplorasi, dan bersihkan\ndata jika diperlukan.",
                "figma-overview-dataset.svg",
                self.main_window.open_datasets_page,
            ),
            (
                "Analisis Data",
                "Lakukan analisis statistik\ndan visualisasi data.",
                "figma-overview-analysis.svg",
                self.main_window.open_analysis_page,
            ),
            (
                "ML Intelligence",
                "Bangun, latih, dan evaluasi\nmodel machine learning.",
                "figma-overview-ml.svg",
                self.main_window.open_ml_page,
            ),
            (
                "Hasilkan Insight",
                "Simpan hasil analisis, buat\nlaporan, dan dukung\npenelitian Anda.",
                "figma-overview-report.svg",
                self.main_window.open_report_tool,
            ),
        )
        for index, (name, detail, icon_name, action) in enumerate(steps):
            card = OverviewStepCard(name, action)
            card.setMinimumWidth(0)
            card.setMinimumHeight(158)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 14, 10, 12)
            card_layout.setSpacing(8)
            number = QLabel(str(index + 1), card)
            number.setObjectName("overviewEmptyStepNumber")
            number.setAttribute(Qt.WA_StyledBackground, True)
            number.setAlignment(Qt.AlignCenter)
            number.setFixedSize(27, 27)
            number.move(14, 9)
            icon_tile = QFrame()
            icon_tile.setObjectName("overviewEmptyStepIconTile")
            icon_tile.setProperty("step", index + 1)
            icon_tile.setFixedSize(50, 50)
            icon_layout = QHBoxLayout(icon_tile)
            icon_layout.setContentsMargins(0, 0, 0, 0)
            icon = QLabel()
            icon.setObjectName("overviewEmptyStepIcon")
            self._icon(icon, icon_name, 25)
            icon_layout.addWidget(icon, 0, Qt.AlignCenter)
            card_layout.addWidget(icon_tile, 0, Qt.AlignHCenter)
            title_label = QLabel(name)
            title_label.setObjectName("overviewEmptyStepTitle")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setWordWrap(True)
            card_layout.addWidget(title_label)
            detail_label = QLabel(detail)
            detail_label.setObjectName("overviewEmptyStepDetail")
            detail_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            detail_label.setWordWrap(True)
            card_layout.addWidget(detail_label, 1)
            self.step_cards.append(card)
            if index < len(steps) - 1:
                arrow = QLabel()
                arrow.setObjectName("overviewEmptyStepArrow")
                arrow.setFixedSize(14, 14)
                self._icon(arrow, "figma-overview-chevron.svg", 14)
                self.step_arrows.append(arrow)
        layout.addLayout(self.steps_layout)
        return panel

    def _build_resources(self) -> QWidget:
        row = QWidget()
        row.setObjectName("overviewEmptyResources")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.resources_layout = layout

        features = QFrame()
        features.setObjectName("overviewEmptyFeatures")
        features.setMinimumWidth(0)
        features.setMinimumHeight(220)
        features.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        feature_layout = QVBoxLayout(features)
        feature_layout.setContentsMargins(12, 12, 12, 12)
        feature_layout.setSpacing(7)
        heading = QLabel("Fitur Utama DORA")
        heading.setObjectName("overviewEmptySectionTitle")
        description = QLabel("Semua yang Anda butuhkan untuk penelitian berbasis data, dalam satu platform.")
        description.setObjectName("overviewEmptySectionSubtitle")
        feature_layout.addWidget(heading)
        feature_layout.addWidget(description)
        self.feature_cards_layout = QGridLayout()
        self.feature_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.feature_cards_layout.setHorizontalSpacing(10)
        self.feature_cards_layout.setVerticalSpacing(9)
        self.feature_cards = []
        for title, detail, icon_name, action in (
            ("Manajemen Dataset", "Upload, kelola, dan jelajahi dataset dengan mudah.", "figma-overview-dataset.svg", self.main_window.open_datasets_page),
            ("Analisis Data", "Statistik deskriptif, visualisasi, korelasi, dan analisis lanjutan.", "figma-overview-analysis.svg", self.main_window.open_analysis_page),
            ("ML Intelligence", "Bangun dan evaluasi model machine learning tanpa coding yang kompleks.", "figma-overview-ml.svg", self.main_window.open_ml_page),
            ("Dukungan Riset", "Temukan literatur, analisis research gap, dan buat laporan penelitian.", "figma-overview-report.svg", self.main_window.open_research_page),
        ):
            card = self._feature_card(title, detail, icon_name, action)
            self.feature_cards.append(card)
        feature_layout.addLayout(self.feature_cards_layout, 1)
        self.features_panel = features
        layout.addWidget(features, 2)
        layout.addWidget(self._build_about_card(), 1)
        return row

    def _feature_card(self, title: str, detail: str, icon_name: str, action) -> QFrame:
        card = QFrame()
        card.setObjectName("overviewEmptyFeatureCard")
        content = QVBoxLayout(card)
        content.setContentsMargins(16, 12, 12, 10)
        content.setSpacing(6)
        tile = QFrame()
        tile.setObjectName("overviewEmptyFeatureIconTile")
        tile.setProperty("kind", icon_name)
        tile.setFixedSize(46, 46)
        tile_layout = QHBoxLayout(tile)
        tile_layout.setContentsMargins(0, 0, 0, 0)
        icon = QLabel()
        icon.setObjectName("overviewEmptyFeatureIcon")
        self._icon(icon, icon_name, 25)
        tile_layout.addWidget(icon, 0, Qt.AlignCenter)
        content.addWidget(tile)
        title_label = QLabel(title)
        title_label.setObjectName("overviewEmptyFeatureTitle")
        title_label.setWordWrap(True)
        content.addWidget(title_label)
        description = QLabel(detail)
        description.setObjectName("overviewEmptyFeatureText")
        description.setWordWrap(True)
        content.addWidget(description, 1)
        learn = QPushButton("Pelajari lebih lanjut  →")
        learn.setObjectName("overviewEmptyLearnMore")
        learn.setText("Pelajari lebih lanjut")
        arrow = self.icons / "overview-arrow-right.svg"
        if arrow.exists():
            learn.setIcon(QIcon(str(arrow)))
            learn.setIconSize(QSize(10, 10))
        learn.setLayoutDirection(Qt.RightToLeft)
        learn.clicked.connect(action)
        content.addWidget(learn, 0, Qt.AlignLeft)
        return card

    def _build_about_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("overviewEmptyAboutCard")
        card.setMinimumHeight(220)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(5)
        heading = QHBoxLayout()
        title = QLabel("Tentang DORA")
        title.setObjectName("overviewEmptyAboutTitle")
        heading.addWidget(title)
        heading.addStretch(1)
        learn = QPushButton("Pelajari Lebih Lanjut  →")
        learn.setObjectName("overviewEmptyAboutLink")
        learn.setText("Pelajari Lebih Lanjut")
        arrow = self.icons / "overview-arrow-right.svg"
        if arrow.exists():
            learn.setIcon(QIcon(str(arrow)))
            learn.setIconSize(QSize(10, 10))
        learn.setLayoutDirection(Qt.RightToLeft)
        learn.clicked.connect(self.main_window.open_about)
        heading.addWidget(learn)
        layout.addLayout(heading)
        description = QLabel(
            "DORA adalah platform terpadu untuk membantu peneliti dalam mengelola data, "
            "melakukan analisis, membangun model machine learning, serta mendapatkan wawasan "
            "ilmiah yang relevan."
        )
        description.setObjectName("overviewEmptyAboutText")
        description.setWordWrap(True)
        layout.addWidget(description)
        benefits = QHBoxLayout()
        self.about_benefits_layout = benefits
        benefits.setSpacing(7)
        for title, icon_name, description, kind in (
            ("users", "figma-overview-users.svg", "Ramah untuk\npeneliti non-teknis", "green"),
            ("shield", "figma-overview-shield.svg", "Aman dan\nterjamin privasi", "orange"),
            ("flask", "figma-overview-flask.svg", "Dirancang untuk\nkebutuhan riset", "purple"),
        ):
            benefit = QFrame()
            benefit.setObjectName("overviewEmptyAboutBenefit")
            benefit.setProperty("kind", kind)
            benefit_layout = QVBoxLayout(benefit)
            benefit_layout.setContentsMargins(8, 8, 5, 7)
            benefit_layout.setSpacing(6)
            tile = QFrame()
            tile.setObjectName("overviewEmptyAboutBenefitIconTile")
            tile.setProperty("kind", kind)
            tile.setFixedSize(30, 30)
            tile_layout = QHBoxLayout(tile)
            tile_layout.setContentsMargins(0, 0, 0, 0)
            icon = QLabel()
            icon.setObjectName("overviewEmptyAboutBenefitIcon")
            self._icon(icon, icon_name, 16)
            tile_layout.addWidget(icon, 0, Qt.AlignCenter)
            benefit_layout.addWidget(tile)
            text = QLabel(description)
            text.setObjectName("overviewEmptyAboutBenefitText")
            text.setWordWrap(True)
            benefit_layout.addWidget(text)
            benefits.addWidget(benefit, 1)
        layout.addLayout(benefits)
        return card

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def apply_scale(self, _scale=None):
        self._apply_responsive_layout(self.width())

    def _reflow_grid(self, layout, widgets, columns):
        while layout.count():
            layout.takeAt(0)
        for column in range(max(5, columns)):
            layout.setColumnStretch(column, 0)
        for index, widget in enumerate(widgets):
            layout.addWidget(widget, index // columns, index % columns)
            layout.setColumnStretch(index % columns, 1)

    def _apply_responsive_layout(self, width):
        scale = max(float(self.main_window.scale_manager.get_scale()), 0.01)
        logical_width = width / scale
        self._fit_hero_copy_widths(logical_width)
        compact = logical_width < 760
        stacked_dataset = logical_width < 1080
        stacked_resources = logical_width < 1120
        stacked_metrics = logical_width < 620
        stacked_about_benefits = logical_width < 440
        step_columns = 5 if logical_width >= 1100 else 2 if logical_width >= 660 else 1
        feature_columns = 4 if logical_width >= 1100 else 2 if logical_width >= 680 else 1
        mode = (
            compact,
            stacked_dataset,
            stacked_resources,
            stacked_metrics,
            stacked_about_benefits,
            logical_width < 1100,
            step_columns,
            feature_columns,
        )
        if mode == getattr(self, "_responsive_mode", None):
            self._fit_hero_description()
            return
        self._responsive_mode = mode
        self.content_layout.setContentsMargins(
            17 if logical_width >= 1100 else 12, 10,
            18 if logical_width >= 1100 else 12, 14,
        )
        self.content_layout.setSpacing(12 if logical_width >= 1100 else 9)
        self.hero_layout.setContentsMargins(
            28 if logical_width >= 760 else 18,
            21 if logical_width >= 760 else 16,
            28 if logical_width >= 760 else 18,
            24 if logical_width >= 760 else 18,
        )
        self.hero_actions.setDirection(
            QBoxLayout.LeftToRight if not compact else QBoxLayout.TopToBottom
        )
        for button in (self.hero_search_button, self.hero_upload_button):
            button.setMinimumWidth(0 if compact else int(button.property("preferredWidth") or 0))
        self.empty_dataset_layout.setDirection(
            QBoxLayout.LeftToRight if not stacked_dataset else QBoxLayout.TopToBottom
        )
        self.loaded_dataset_layout.setDirection(
            QBoxLayout.LeftToRight if not stacked_dataset else QBoxLayout.TopToBottom
        )
        self.empty_dataset_layout.setSpacing(22 if not stacked_dataset else 14)
        self.loaded_dataset_layout.setSpacing(22 if not stacked_dataset else 14)
        self.dataset_illustration.setVisible(logical_width >= 760)
        self.dataset_illustration.setMaximumWidth(257 if not stacked_dataset else 220)
        self.empty_dataset_actions.setDirection(
            QBoxLayout.LeftToRight if not compact else QBoxLayout.TopToBottom
        )
        self.loaded_actions_layout.setDirection(
            QBoxLayout.LeftToRight if not compact else QBoxLayout.TopToBottom
        )
        self.loaded_metrics_layout.setDirection(
            QBoxLayout.TopToBottom if stacked_metrics else QBoxLayout.LeftToRight
        )
        self.resources_layout.setDirection(
            QBoxLayout.LeftToRight if not stacked_resources else QBoxLayout.TopToBottom
        )
        while self.steps_layout.count():
            self.steps_layout.takeAt(0)
        for column in range(10):
            self.steps_layout.setColumnStretch(column, 0)
        if step_columns == 5:
            for index, card in enumerate(self.step_cards):
                self.steps_layout.addWidget(card, 0, index * 2)
                self.steps_layout.setColumnStretch(index * 2, 1)
                if index < len(self.step_arrows):
                    arrow = self.step_arrows[index]
                    arrow.setVisible(True)
                    self.steps_layout.addWidget(arrow, 0, index * 2 + 1, Qt.AlignCenter)
        else:
            for arrow in self.step_arrows:
                arrow.hide()
            self._reflow_grid(self.steps_layout, self.step_cards, step_columns)
        self._reflow_grid(self.feature_cards_layout, self.feature_cards, feature_columns)
        self.about_benefits_layout.setDirection(
            QBoxLayout.TopToBottom if stacked_about_benefits else QBoxLayout.LeftToRight
        )
        self._fit_hero_description()

    def _fit_hero_copy_widths(self, logical_width):
        """Give the hero copy enough room on wide screens without breaking small layouts."""
        px = self.main_window.scale_manager.px
        title_width = px(round(min(max(logical_width * 0.62, 680), 1180)))
        description_width = px(round(min(max(logical_width * 0.50, 640), 1060)))
        if self.hero_title.maximumWidth() != title_width:
            self.hero_title.setMaximumWidth(title_width)
        if self.hero_description.maximumWidth() != description_width:
            self.hero_description.setMaximumWidth(description_width)

    def _fit_hero_description(self):
        """Keep the wrapped hero copy tall enough to show its final line."""
        description = getattr(self, "hero_description", None)
        if description is None or description.width() <= 0:
            return
        if self.hero_layout is not None:
            self.hero_layout.activate()
        required_height = description.heightForWidth(description.width())
        if required_height > 0 and description.minimumHeight() != required_height:
            description.setMinimumHeight(required_height)
