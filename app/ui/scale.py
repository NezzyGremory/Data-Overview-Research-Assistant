"""UI scaling settings and stylesheet helpers."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication


SCALE_MIN = 0.6
SCALE_MAX = 1.5
SCALE_STEP = 0.05
SCALE_PRESETS = (
    ("Auto", None),
    ("70%", 0.70),
    ("80%", 0.80),
    ("90%", 0.90),
    ("100%", 1.00),
    ("110%", 1.10),
    ("125%", 1.25),
    ("150%", 1.50),
)
_PIXEL_TOKEN = re.compile(r"@px\((-?\d+(?:\.\d+)?)\)")
_RAW_PIXEL = re.compile(r"(?<![\w])(-?\d+(?:\.\d+)?)px")


@lru_cache(maxsize=32)
def scale_stylesheet(template: str, scale: float) -> str:
    """Resolve @px(n) tokens while leaving QSS braces untouched."""
    template = _RAW_PIXEL.sub(
        lambda match: f"@px({match.group(1)})", template
    )
    return _PIXEL_TOKEN.sub(
        lambda match: (
            f"{match.group(1)}px"
            if scale == 1.0
            else f"{round(float(match.group(1)) * scale)}px"
        ),
        template,
    )


@lru_cache(maxsize=16)
def _build_stylesheet_cached(path: str, scale: float, modified_ns: int) -> str:
    del modified_ns  # Included in the cache key so edits are picked up immediately.
    return scale_stylesheet(Path(path).read_text(encoding="utf-8-sig"), scale)


def build_stylesheet(scale: float, template_path: str | Path | None = None) -> str:
    """Read and scale the shared QSS template."""
    path = Path(template_path) if template_path else Path(__file__).with_name("app.qss")
    path = path.resolve()
    return _build_stylesheet_cached(str(path), float(scale), path.stat().st_mtime_ns)


class ScaleManager(QObject):
    """Store Auto/manual UI scale and emit the active scale on changes."""

    scaleChanged = Signal(float)

    def __init__(self, settings: QSettings | None = None, screen: Any = None):
        super().__init__()
        self.settings = settings or QSettings("DORA", "Dataset Research")
        self._screen = screen
        self._manual_scale: float | None = None
        self._scale = 1.0
        self._read_setting()

    @property
    def is_auto(self) -> bool:
        return self._manual_scale is None

    def get_scale(self) -> float:
        return self._scale

    def set_scale(self, value: float | str | None) -> None:
        """Set a manual scale, or pass None/'auto' to use screen sizing."""
        was_auto = self.is_auto
        if value is None or (isinstance(value, str) and value.strip().casefold() == "auto"):
            self._manual_scale = None
            self.settings.setValue("ui_scale", "auto")
            self._refresh_scale(force=was_auto != self.is_auto)
            return
        try:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError("scale must be finite")
        except (TypeError, ValueError, OverflowError):
            self._manual_scale = None
            self.settings.setValue("ui_scale", "auto")
            self._refresh_scale(force=was_auto != self.is_auto)
            return

        self._manual_scale = self._clamp(parsed)
        self.settings.setValue("ui_scale", self._manual_scale)
        self._refresh_scale(force=was_auto != self.is_auto)

    def zoom_in(self) -> None:
        self.set_scale(self._clamp(self._scale + SCALE_STEP))

    def zoom_out(self) -> None:
        self.set_scale(self._clamp(self._scale - SCALE_STEP))

    def reset(self) -> None:
        self.set_scale(None)

    def auto_scale(self, screen: Any = None) -> float:
        """Choose a compact starting scale from available height and logical DPI."""
        screen = screen or self._screen
        if screen is None:
            app = QApplication.instance()
            screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return 1.0
        height = int(screen.availableGeometry().height())
        dpi = float(screen.logicalDotsPerInch() or 96.0)
        scale = 0.8 if height < 720 else 0.9 if height < 900 else 1.0
        if dpi > 120:
            scale *= 0.95
        return self._round_step(self._clamp(scale))

    def update_screen(self, screen: Any = None) -> None:
        """Re-evaluate Auto after the window moves to another screen."""
        if screen is not None:
            self._screen = screen
        if self.is_auto:
            self._refresh_scale()

    def px(self, value: float) -> int:
        """Scale a fixed pixel value for use in widget geometry."""
        return round(float(value) * self._scale)

    @staticmethod
    def _clamp(value: float) -> float:
        return min(SCALE_MAX, max(SCALE_MIN, value))

    @staticmethod
    def _round_step(value: float) -> float:
        return round(round(value / SCALE_STEP) * SCALE_STEP, 2)

    def _read_setting(self) -> None:
        value = self.settings.value("ui_scale", "auto")
        if isinstance(value, str) and value.strip().casefold() == "auto":
            self._manual_scale = None
        else:
            try:
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError
                self._manual_scale = self._clamp(parsed)
            except (TypeError, ValueError, OverflowError):
                self._manual_scale = None
                self.settings.setValue("ui_scale", "auto")
        self._scale = self.auto_scale() if self.is_auto else self._manual_scale

    def _refresh_scale(self, force: bool = False) -> None:
        value = self.auto_scale() if self.is_auto else float(self._manual_scale)
        if value == self._scale and not force:
            return
        self._scale = value
        self.scaleChanged.emit(value)


def ui_px(value: float) -> int:
    """Scale helper for fixed widget dimensions in UI modules."""
    app = QApplication.instance()
    scale = app.property("uiScale") if app is not None else None
    return round(float(value) * float(scale or 1.0))


def logical_width(value: float) -> float:
    """Convert a widget's current width to its unscaled layout reference."""
    app = QApplication.instance()
    scale = app.property("uiScale") if app is not None else None
    return float(value) / max(float(scale or 1.0), 0.01)
