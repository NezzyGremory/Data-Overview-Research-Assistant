"""Fullscreen animated end credits overlay for the DORA desktop app."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QEvent, QPoint, QPropertyAnimation, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.scale import scale_stylesheet


_UI_ROOT = Path(__file__).resolve().parents[1]
_CREDITS_PATH = _UI_ROOT / "data" / "credits.json"
_CREDITS_QSS = _UI_ROOT / "styles" / "credits.qss"
_LOGO_PATH = _UI_ROOT / "assets" / "about-brand-mark.png"
_TOKENS_PATH = _UI_ROOT / "styles" / "theme.qss"


def _load_credits() -> dict:
    with _CREDITS_PATH.open("r", encoding="utf-8") as source:
        credits = json.load(source)
    return credits


class CreditsOverlay(QWidget):
    """A dismissible overlay whose credits roll upward at a constant speed."""

    SCROLL_PIXELS_PER_SECOND = 54

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("creditsOverlay")
        self.setProperty("scaleStylesheetExternally", True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._closing = False
        self._scroll_animation: QPropertyAnimation | None = None
        self._fade_animation: QPropertyAnimation | None = None
        self._base_scroll_duration = 0
        self._scroll_started = False

        credits = _load_credits()
        self._theme_tokens = self._read_theme_tokens()
        application = QApplication.instance()
        scale = float(application.property("uiScale") or 1.0) if application else 1.0
        self.apply_scale(scale)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 22, 0)
        top_bar.addStretch(1)
        self.skip_button = QPushButton("×  Skip")
        self.skip_button.setObjectName("creditsSkip")
        self.skip_button.setCursor(Qt.PointingHandCursor)
        self.skip_button.clicked.connect(self.close_overlay)
        top_bar.addWidget(self.skip_button, 0, Qt.AlignRight | Qt.AlignTop)
        root.addLayout(top_bar)

        self.viewport = QWidget(self)
        self.viewport.setObjectName("creditsViewport")
        self.viewport.setAttribute(Qt.WA_StyledBackground, True)
        self.viewport.setStyleSheet("QWidget#creditsViewport { background: transparent; }")
        root.addWidget(self.viewport, 1)

        self.content = QWidget(self.viewport)
        self.content.setObjectName("creditsContent")
        self.content.setAttribute(Qt.WA_StyledBackground, True)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(24, 0, 24, 36)
        self.content_layout.setSpacing(0)

        self._add_spacer(180)
        logo = QLabel(self.content)
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(116, 116)
        pixmap = QPixmap(str(_LOGO_PATH))
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(112, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText(credits.get("app", "DORA"))
            logo.setObjectName("creditsLogoName")
        logo.setStyleSheet("background: transparent;")
        self.content_layout.addWidget(logo, 0, Qt.AlignHCenter)

        app_name = QLabel(str(credits.get("app", "")), self.content)
        app_name.setObjectName("creditsLogoName")
        app_name.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(app_name)

        tagline = QLabel(str(credits.get("tagline", "")), self.content)
        tagline.setObjectName("creditsTagline")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setWordWrap(True)
        self.content_layout.addWidget(tagline)
        self._add_spacer(72)

        team = QLabel(str(credits.get("team", "")), self.content)
        team.setObjectName("creditsTeam")
        team.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(team)
        self._add_spacer(76)

        for section in credits.get("sections", []):
            role = QLabel(str(section.get("role", "")).upper(), self.content)
            role.setObjectName("creditsRole")
            role.setAlignment(Qt.AlignCenter)
            self.content_layout.addWidget(role)
            self._add_spacer(9)
            for name in section.get("names", []):
                person = QLabel(str(name), self.content)
                person.setObjectName("creditsName")
                person.setAlignment(Qt.AlignCenter)
                person.setWordWrap(True)
                self.content_layout.addWidget(person)
            self._add_spacer(48)

        footer = QLabel(str(credits.get("footer", "")), self.content)
        self.footer = footer
        footer.setObjectName("creditsFooter")
        footer.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(footer)
        self._add_spacer(48)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

    def _add_spacer(self, height: int):
        spacer = QWidget(self.content)
        spacer.setFixedHeight(height)
        spacer.setStyleSheet("background: transparent;")
        self.content_layout.addWidget(spacer)

    @staticmethod
    def _read_theme_tokens() -> dict[str, str]:
        tokens: dict[str, str] = {}
        if _TOKENS_PATH.exists():
            import re

            source = _TOKENS_PATH.read_text(encoding="utf-8")
            tokens = dict(re.findall(r"/\*\s*@([\w-]+):\s*(.*?)\s*\*/", source))
        return tokens

    def _resolve_stylesheet(self) -> str:
        import re

        stylesheet = _CREDITS_QSS.read_text(encoding="utf-8")
        return re.sub(
            r"var\(--([\w-]+)\)",
            lambda match: self._theme_tokens.get(match.group(1), match.group(0)),
            stylesheet,
        )

    def apply_scale(self, scale: float) -> None:
        """Refresh overlay dimensions when the application zoom changes."""
        self.setStyleSheet(scale_stylesheet(self._resolve_stylesheet(), scale))

    def showEvent(self, event):
        super().showEvent(event)
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.setFocus(Qt.OtherFocusReason)
        self._place_content()
        self._start_fade_in()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "viewport"):
            self._place_content(restart=self._scroll_started)

    def _place_content(self, restart: bool = False):
        if not hasattr(self, "content") or self._closing:
            return
        self.content.setFixedWidth(max(self.viewport.width(), 1))
        self.content.adjustSize()
        content_height = self.content.sizeHint().height()
        self.content.resize(max(self.viewport.width(), 1), content_height)
        start_y = self.viewport.height()
        end_y = self._footer_end_position(content_height)
        if restart and self._scroll_started:
            animation = self._scroll_animation
            progress = 0.0
            if animation and animation.duration() > 0:
                progress = animation.currentTime() / animation.duration()
            self._stop_animation("_scroll_animation")
            current_y = round(start_y + (end_y - start_y) * progress)
            remaining = self._duration_for(start_y, end_y) * max(0.0, 1.0 - progress)
            self._start_scroll(current_y, end_y, max(1, int(remaining)))
            return
        self.content.move(0, start_y)
        if self.isVisible() and not self._scroll_started:
            self._start_scroll(start_y, end_y)

    def _duration_for(self, start_y: int, end_y: int) -> int:
        distance = max(1, start_y - end_y)
        return max(25000, min(35000, int(distance / self.SCROLL_PIXELS_PER_SECOND * 1000)))

    def _footer_end_position(self, content_height: int) -> int:
        footer_center = self.footer.y() + self.footer.height() / 2
        return round(self.viewport.height() * 0.62 - footer_center)

    def _start_fade_in(self):
        animation = QPropertyAnimation(self._opacity, b"opacity", self)
        animation.setDuration(180)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_animation = animation
        animation.start()

    def _start_scroll(self, start_y: int, end_y: int, duration: int | None = None):
        if self._closing:
            return
        animation = QPropertyAnimation(self.content, b"pos", self)
        animation.setDuration(duration or self._duration_for(start_y, end_y))
        animation.setStartValue(QPoint(0, start_y))
        animation.setEndValue(QPoint(0, end_y))
        animation.setEasingCurve(QEasingCurve.Linear)
        animation.finished.connect(self._finish_roll)
        self._scroll_animation = animation
        self._scroll_started = True
        animation.start()

    def _finish_roll(self):
        if self._closing:
            return
        self._stop_animation("_scroll_animation")
        animation = QPropertyAnimation(self._opacity, b"opacity", self)
        animation.setDuration(240)
        animation.setStartValue(self._opacity.opacity())
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        animation.finished.connect(self.close_overlay)
        self._fade_animation = animation
        animation.start()

    def _stop_animation(self, attribute: str):
        animation = getattr(self, attribute, None)
        if animation is not None:
            animation.stop()
            animation.deleteLater()
            setattr(self, attribute, None)

    def close_overlay(self, *_):
        if self._closing:
            return
        self._closing = True
        self._stop_animation("_scroll_animation")
        self._stop_animation("_fade_animation")
        parent = self.parentWidget()
        if parent is not None:
            parent.removeEventFilter(self)
        self.hide()
        self.deleteLater()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close_overlay()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.close_overlay()
        event.accept()

    def eventFilter(self, watched, event):
        if (
            watched is self.parentWidget()
            and event.type() in (QEvent.Resize, QEvent.WindowStateChange)
            and not self._closing
        ):
            self.setGeometry(self.parentWidget().rect())
            self._place_content(restart=self._scroll_started)
        return super().eventFilter(watched, event)


def show_credits_overlay(parent: QWidget) -> CreditsOverlay:
    """Show one overlay instance and follow the parent window's geometry."""
    existing = getattr(parent, "_credits_overlay", None)
    if existing is not None and not existing._closing:
        existing.raise_()
        existing.setFocus(Qt.OtherFocusReason)
        return existing

    overlay = CreditsOverlay(parent)
    scale_manager = getattr(parent, "scale_manager", None)
    if scale_manager is not None:
        scale_manager.scaleChanged.connect(overlay.apply_scale)
    parent._credits_overlay = overlay
    parent.installEventFilter(overlay)
    overlay.destroyed.connect(lambda: setattr(parent, "_credits_overlay", None))

    overlay.setGeometry(parent.rect())
    overlay.show()
    return overlay
