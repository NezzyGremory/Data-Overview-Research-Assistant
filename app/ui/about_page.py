"""About page matching the DORA About frame."""

from pathlib import Path
import sys

from PySide6.QtCore import Qt, QSize, QRect
from PySide6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.scale import logical_width, ui_px
from app.ui.style_utils import apply_card_shadow, render_icon_pixmap


def _asset_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return root / "app" / "ui" / relative_path
    return Path(__file__).resolve().parent / relative_path


class AboutPage(QWidget):
    """Application description page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aboutFigmaPage")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("aboutFigmaScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.scroll)

        self.canvas = QWidget()
        self.canvas.setObjectName("aboutFigmaCanvas")
        self.content = QVBoxLayout(self.canvas)
        self.content.setContentsMargins(36, 29, 46, 28)
        self.content.setSpacing(0)
        self.scroll.setWidget(self.canvas)

        self.title = QLabel("Tentang DORA")
        self.title.setObjectName("aboutFigmaTitle")
        self.title.setFixedHeight(41)
        self.content.addWidget(self.title)
        self.content.addSpacing(1)

        self.subtitle = QLabel(
            "Kenali DORA, ruang kerja lokal untuk memahami dataset dan mendukung proses riset Anda."
        )
        self.subtitle.setObjectName("aboutFigmaSubtitle")
        self.subtitle.setFixedHeight(31)
        self.subtitle.setWordWrap(True)
        self.content.addWidget(self.subtitle)
        self.content.addSpacing(12)

        self.application_card = QFrame()
        self.application_card.setObjectName("aboutApplicationCard")
        self.application_card.setFixedSize(1100, 259)
        apply_card_shadow(self.application_card)

        self.application_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.application_layout.setContentsMargins(0, 0, 0, 0)
        self.application_layout.setSpacing(0)
        self.application_card.setLayout(self.application_layout)

        self.application_identity = QWidget()
        self.application_identity.setObjectName("aboutApplicationIdentity")
        self.application_identity.setFixedWidth(416)
        identity_layout = QVBoxLayout(self.application_identity)
        identity_layout.setContentsMargins(24, 10, 0, 0)
        identity_layout.setSpacing(0)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(20)
        brand_row.setAlignment(Qt.AlignLeft)
        brand_mark = QLabel()
        brand_mark.setObjectName("aboutBrandMark")
        brand_mark.setFixedSize(70, 86)
        brand_mark.setAlignment(Qt.AlignCenter)
        brand_mark.setPixmap(
            render_icon_pixmap(
                _asset_path("assets/about-brand-mark.png"),
                QSize(70, 86),
                brand_mark.devicePixelRatioF(),
            )
        )
        brand_row.addWidget(brand_mark, 0, Qt.AlignTop)

        brand_text = QVBoxLayout()
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(0)
        brand_title = QLabel("DORA")
        brand_title.setObjectName("aboutBrandTitle")
        brand_title.setFixedHeight(57)
        brand_text.addWidget(brand_title)
        brand_subtitle = QLabel("Data Overview & Research Assistant")
        brand_subtitle.setObjectName("aboutBrandSubtitle")
        brand_subtitle.setContentsMargins(4, 0, 0, 0)
        brand_subtitle.setFixedWidth(276)
        brand_subtitle.setWordWrap(True)
        brand_text.addWidget(brand_subtitle)
        brand_row.addLayout(brand_text)
        identity_layout.addLayout(brand_row)
        identity_layout.addSpacing(18)

        badges = QHBoxLayout()
        badges.setContentsMargins(13, 0, 0, 0)
        badges.setSpacing(12)
        for label, icon_name, width in (
            ("Dataset", "about-dataset.svg", 82),
            ("Analysis", "about-analysis.svg", 88),
            ("ML Intelligence", "about-ml.svg", 128),
        ):
            badges.addWidget(self._make_badge(label, icon_name, width))
        badges.addStretch(1)
        identity_layout.addLayout(badges)
        identity_layout.addStretch(1)
        self.application_layout.addWidget(self.application_identity)

        description_panel = QWidget()
        description_panel.setObjectName("aboutApplicationDescriptionPanel")
        description_layout = QVBoxLayout(description_panel)
        description_layout.setContentsMargins(0, 22, 10, 0)
        description_layout.setSpacing(12)
        description_title = QLabel("TENTANG APLIKASI")
        description_title.setObjectName("aboutApplicationDescriptionTitle")
        description_title.setFixedHeight(25)
        description_layout.addWidget(description_title)
        description = QLabel(
            "DORA adalah aplikasi desktop yang membantu peneliti memahami dataset, "
            "menemukan metode Machine Learning yang sesuai, dan menjelajahi riset akademik terkait "
            "dalam satu ruang kerja lokal. Mulai dari mengunggah dataset hingga membuat laporan riset, "
            "setiap langkah dirancang agar mudah diikuti."
        )
        description.setObjectName("aboutApplicationDescription")
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.application_description = description
        description_layout.addWidget(description)
        description_layout.addStretch(1)
        self.application_description_panel = description_panel
        self.application_layout.addWidget(description_panel, 1)

        self.content.addWidget(self.application_card, 0, Qt.AlignLeft)

        self.details_section = QWidget()
        self.details_section.setObjectName("aboutDetailsSection")
        self.details_section.setMaximumWidth(1100)
        details_layout = QVBoxLayout(self.details_section)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(12)
        self.details_heading = QLabel("DORA dalam satu ruang kerja")
        self.details_heading.setObjectName("aboutDetailsHeading")
        details_layout.addWidget(self.details_heading)

        self.capability_row = QWidget()
        self.capability_row.setObjectName("aboutCapabilityRow")
        capability_layout = QBoxLayout(QBoxLayout.LeftToRight)
        capability_layout.setContentsMargins(0, 0, 0, 0)
        capability_layout.setSpacing(12)
        self.capability_row.setLayout(capability_layout)
        self.capability_cards = []
        for heading, text, kind in (
            ("Kelola dataset", "Unggah dan jelajahi data dalam workspace lokal.", "dataset"),
            ("Pahami data", "Tinjau ringkasan, statistik, dan kualitas dataset.", "analysis"),
            ("Dukung riset", "Jelajahi literatur dan susun laporan penelitian.", "research"),
        ):
            capability = QFrame()
            capability.setObjectName("aboutCapabilityCard")
            capability.setProperty("kind", kind)
            card_layout = QVBoxLayout(capability)
            card_layout.setContentsMargins(16, 13, 16, 13)
            card_layout.setSpacing(5)
            card_title = QLabel(heading)
            card_title.setObjectName("aboutCapabilityTitle")
            card_layout.addWidget(card_title)
            card_text = QLabel(text)
            card_text.setObjectName("aboutCapabilityText")
            card_text.setWordWrap(True)
            card_layout.addWidget(card_text)
            capability_layout.addWidget(capability, 1)
            self.capability_cards.append((capability, card_text))
        details_layout.addWidget(self.capability_row)
        self.content.addSpacing(18)
        self.content.addWidget(self.details_section, 0, Qt.AlignLeft)

        self.copyright = QLabel("\u00a9 2026 Metriks Studio. All rights reserved.")
        self.copyright.setObjectName("aboutCopyright")
        self.copyright.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.content.addSpacing(20)
        self.content.addWidget(self.copyright)
        self.content.addStretch(1)

        self._apply_responsive_width(self.width())

    @staticmethod
    def _make_badge(label: str, icon_name: str, width: int) -> QFrame:
        badge = QFrame()
        badge.setObjectName("aboutFigmaBadge")
        badge.setFixedSize(width, 23)
        layout = QHBoxLayout(badge)
        layout.setContentsMargins(6, 0, 4, 0)
        layout.setSpacing(0)
        icon = QLabel()
        icon.setFixedSize(17, 17)
        pixmap = render_icon_pixmap(
            _asset_path(Path("assets") / "icons" / icon_name),
            17,
            icon.devicePixelRatioF(),
        )
        icon.setPixmap(pixmap)
        layout.addWidget(icon)
        text = QLabel(label)
        text.setObjectName("aboutFigmaBadgeText")
        layout.addWidget(text)
        return badge

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_width(event.size().width())

    def _apply_responsive_width(self, page_width: int) -> None:
        available = max(ui_px(320), page_width - ui_px(82))
        card_width = min(ui_px(1100), available)
        text_width = max(1, page_width - ui_px(82))
        subtitle_height = self.subtitle.fontMetrics().boundingRect(
            QRect(0, 0, text_width, 1000),
            Qt.TextWordWrap,
            self.subtitle.text(),
        ).height()
        self.subtitle.setFixedHeight(max(ui_px(31), subtitle_height + ui_px(4)))
        self.application_card.setFixedWidth(card_width)
        self.details_section.setFixedWidth(card_width)
        self.capability_row.layout().setDirection(
            QBoxLayout.LeftToRight if logical_width(card_width) >= 720 else QBoxLayout.TopToBottom
        )
        horizontal_capabilities = logical_width(card_width) >= 720
        capability_width = (
            max(1, (card_width - ui_px(24)) // 3)
            if horizontal_capabilities
            else max(1, card_width)
        )
        for capability, text_label in self.capability_cards:
            text_width = max(1, capability_width - ui_px(32))
            text_height = text_label.fontMetrics().boundingRect(
                QRect(0, 0, text_width, 1000),
                Qt.TextWordWrap,
                text_label.text(),
            ).height()
            text_label.setMinimumHeight(text_height + ui_px(5))
            capability.setFixedHeight(max(ui_px(90), text_height + ui_px(62)))
        self.application_layout.setDirection(
            QBoxLayout.LeftToRight if logical_width(card_width) >= 820 else QBoxLayout.TopToBottom
        )
        if logical_width(card_width) >= 820:
            self.application_identity.setFixedWidth(ui_px(416))
            self.application_description_panel.setMinimumHeight(0)
            self.application_description_panel.setMaximumHeight(16777215)
            body_width = max(1, card_width - ui_px(428))
            description_height = self.application_description.fontMetrics().boundingRect(
                QRect(0, 0, body_width, 1000),
                Qt.TextWordWrap,
                self.application_description.text(),
            ).height()
            self.application_description.setFixedHeight(description_height)
            self.application_identity.setMinimumHeight(0)
            self.application_identity.setMaximumHeight(16777215)
            self.application_card.setFixedHeight(max(ui_px(190), description_height + ui_px(66)))
        else:
            self.application_identity.setFixedWidth(max(0, card_width - ui_px(32)))
            body_width = max(1, card_width - ui_px(12))
            description_height = self.application_description.fontMetrics().boundingRect(
                QRect(0, 0, body_width, 1000),
                Qt.TextWordWrap,
                self.application_description.text(),
            ).height()
            self.application_description.setFixedHeight(description_height)
            identity_height = ui_px(179)
            self.application_identity.setFixedHeight(identity_height)
            self.application_description_panel.setFixedHeight(
                description_height + ui_px(79)
            )
            self.application_card.setFixedHeight(
                identity_height + description_height + ui_px(81)
            )
        self.canvas.setMinimumWidth(max(1, page_width))

    def apply_scale(self, _scale=None):
        self._apply_responsive_width(self.width())
