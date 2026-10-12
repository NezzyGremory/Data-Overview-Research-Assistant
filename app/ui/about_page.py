"""About page matching the DORA About frame."""

from pathlib import Path
import sys

from PySide6.QtCore import Qt, QSize, QRect
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.style_utils import apply_card_shadow


def _asset_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return root / "app" / "ui" / relative_path
    return Path(__file__).resolve().parent / relative_path


class AboutPage(QWidget):
    """Application description and the ML recommendation preview."""

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

        self.title = QLabel("About")
        self.title.setObjectName("aboutFigmaTitle")
        self.title.setFixedHeight(41)
        self.content.addWidget(self.title)
        self.content.addSpacing(1)

        self.subtitle = QLabel(
            "Get to know the application, the team behind it, and the expert validation that guided is development"
        )
        self.subtitle.setObjectName("aboutFigmaSubtitle")
        self.subtitle.setFixedHeight(31)
        self.subtitle.setWordWrap(True)
        self.content.addWidget(self.subtitle)
        self.content.addSpacing(12)

        self.application_card = QFrame()
        self.application_card.setObjectName("aboutApplicationCard")
        self.application_card.setFixedSize(935, 259)
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
        brand_mark.setFixedSize(70, 99)
        brand_mark.setAlignment(Qt.AlignCenter)
        mark = QPixmap(str(_asset_path("assets/about-brand-mark.png")))
        if not mark.isNull():
            brand_mark.setPixmap(
                mark.scaled(70, 86, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        identity_layout.addSpacing(39)

        badges = QHBoxLayout()
        badges.setContentsMargins(13, 0, 0, 0)
        badges.setSpacing(12)
        for label, icon_name, width in (
            ("Dataset", "about-dataset.svg", 82),
            ("Analysis", "about-analysis.svg", 88),
            ("ML In", "about-ml.svg", 88),
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
        description_title = QLabel("ABOUT THE APPLICATION")
        description_title.setObjectName("aboutApplicationDescriptionTitle")
        description_title.setFixedHeight(25)
        description_layout.addWidget(description_title)
        description = QLabel(
            "DORA is a desktop application that helps novice researchers understand their datasets, "
            "find suitable machine learning methods, and explore related academic research in a single "
            "local workspace. From uploading a dataset to generating a research report, every step is "
            "designed to be simple and easy to follow."
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
        self.content.addSpacing(45)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(0)
        model_row.addStretch(1)
        self.model_card = self._build_model_card()
        model_row.addWidget(self.model_card, 0, Qt.AlignRight)
        self.content.addLayout(model_row)
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
        pixmap = QIcon(str(_asset_path(Path("assets") / "icons" / icon_name))).pixmap(
            QSize(17, 17)
        )
        icon.setPixmap(pixmap)
        layout.addWidget(icon)
        text = QLabel(label)
        text.setObjectName("aboutFigmaBadgeText")
        layout.addWidget(text)
        return badge

    def _build_model_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("aboutModelCard")
        card.setFixedSize(465, 258)
        apply_card_shadow(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 20, 14)
        layout.setSpacing(0)

        heading = QHBoxLayout()
        heading.setContentsMargins(0, 0, 0, 0)
        heading.setSpacing(9)
        icon_wrap = QFrame()
        icon_wrap.setObjectName("aboutModelIconWrap")
        icon_wrap.setFixedSize(32, 32)
        icon_layout = QHBoxLayout(icon_wrap)
        icon_layout.setContentsMargins(4, 4, 4, 4)
        icon = QLabel()
        icon.setFixedSize(24, 24)
        icon.setPixmap(
            QIcon(str(_asset_path("assets/icons/ml-intelligence.svg"))).pixmap(
                QSize(24, 24)
            )
        )
        icon_layout.addWidget(icon)
        heading.addWidget(icon_wrap)
        title = QLabel("ML Intellegence")
        title.setObjectName("aboutModelTitle")
        heading.addWidget(title)
        heading.addStretch(1)
        layout.addLayout(heading)
        layout.addSpacing(20)

        badges = QHBoxLayout()
        badges.setContentsMargins(0, 0, 0, 0)
        badges.setSpacing(9)
        for label, object_name, width in (
            ("Binary Clasification", "aboutModelTypeBadge", 145),
            ("80% Confidence", "aboutModelConfidenceBadge", 125),
            ("Target: Survived", "aboutModelTargetBadge", 125),
        ):
            badge = QLabel(label)
            badge.setObjectName(object_name)
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(width, 23)
            badges.addWidget(badge)
        layout.addLayout(badges)
        layout.addSpacing(17)

        caption = QLabel("TOP RECOMMENDED MODELS")
        caption.setObjectName("aboutModelCaption")
        caption.setFixedHeight(18)
        layout.addWidget(caption)
        layout.addSpacing(10)

        for name, value, bar_value in (
            ("Random Forest", "82.8%", 90),
            ("Gradient Boosting", "82.1%", 84),
            ("Logistic Regression", "80.7%", 78),
        ):
            row_widget = QWidget()
            row_widget.setObjectName("aboutModelRow")
            row_widget.setFixedHeight(31)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(3, 0, 7, 0)
            row.setSpacing(0)
            model_name = QLabel(name)
            model_name.setObjectName("aboutModelName")
            model_name.setFixedWidth(179)
            bar = QProgressBar()
            bar.setObjectName("aboutModelProgress")
            bar.setRange(0, 100)
            bar.setValue(bar_value)
            bar.setTextVisible(False)
            bar.setFixedSize(188, 8)
            score = QLabel(value)
            score.setObjectName("aboutModelScore")
            score.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            score.setFixedWidth(38)
            row.addWidget(model_name)
            row.addWidget(bar)
            row.addSpacing(14)
            row.addWidget(score)
            layout.addWidget(row_widget)

        details = QHBoxLayout()
        details.setContentsMargins(3, 0, 0, 0)
        details.setSpacing(2)
        details_text = QLabel("View details")
        details_text.setObjectName("aboutModelDetails")
        details.addWidget(details_text)
        arrow = QLabel()
        arrow.setFixedSize(10, 10)
        arrow.setPixmap(
            QIcon(str(_asset_path("assets/icons/arrow-right.svg"))).pixmap(
                QSize(10, 10)
            )
        )
        details.addWidget(arrow)
        details.addStretch(1)
        layout.addLayout(details)
        return card

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_width(event.size().width())

    def _apply_responsive_width(self, page_width: int) -> None:
        available = max(320, page_width - 82)
        card_width = min(935, available)
        text_width = max(1, page_width - 82)
        subtitle_height = self.subtitle.fontMetrics().boundingRect(
            QRect(0, 0, text_width, 1000),
            Qt.TextWordWrap,
            self.subtitle.text(),
        ).height()
        self.subtitle.setFixedHeight(max(31, subtitle_height + 4))
        self.application_card.setFixedWidth(card_width)
        self.application_layout.setDirection(
            QBoxLayout.LeftToRight if card_width >= 820 else QBoxLayout.TopToBottom
        )
        if card_width >= 820:
            self.application_identity.setFixedWidth(416)
            self.application_description_panel.setMinimumHeight(0)
            self.application_description_panel.setMaximumHeight(16777215)
            body_width = max(1, card_width - 428)
            description_height = self.application_description.fontMetrics().boundingRect(
                QRect(0, 0, body_width, 1000),
                Qt.TextWordWrap,
                self.application_description.text(),
            ).height()
            self.application_description.setFixedHeight(description_height)
            self.application_identity.setMinimumHeight(0)
            self.application_identity.setMaximumHeight(16777215)
            self.application_card.setFixedHeight(max(259, description_height + 84))
        else:
            self.application_identity.setFixedWidth(max(0, card_width - 32))
            body_width = max(1, card_width - 12)
            description_height = self.application_description.fontMetrics().boundingRect(
                QRect(0, 0, body_width, 1000),
                Qt.TextWordWrap,
                self.application_description.text(),
            ).height()
            self.application_description.setFixedHeight(description_height)
            identity_height = 179
            self.application_identity.setFixedHeight(identity_height)
            self.application_description_panel.setFixedHeight(
                description_height + 79
            )
            self.application_card.setFixedHeight(
                identity_height + description_height + 81
            )
        self.canvas.setMinimumWidth(max(1, page_width))
