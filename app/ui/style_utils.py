"""Small Qt helpers shared by Figma-matched UI pages."""

from math import ceil
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


def apply_card_shadow(widget: QWidget) -> None:
    """Apply the subtle shadow used on Figma cards (QSS has no box-shadow)."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(4)
    shadow.setOffset(0, 2)
    shadow.setColor(QColor(0, 0, 0, 13))
    widget.setGraphicsEffect(shadow)


def render_icon_pixmap(
    path: str | Path,
    logical_size: int | QSize,
    device_pixel_ratio: float = 1.0,
    tint: str | QColor | None = None,
) -> QPixmap:
    """Render a vector icon at physical resolution to keep it sharp on HiDPI screens."""
    if isinstance(logical_size, QSize):
        width, height = logical_size.width(), logical_size.height()
    else:
        width = height = int(logical_size)

    ratio = max(1.0, float(device_pixel_ratio or 1.0))
    pixel_size = QSize(max(1, ceil(width * ratio)), max(1, ceil(height * ratio)))
    pixmap = QIcon(str(path)).pixmap(pixel_size)
    if pixmap.isNull():
        return pixmap

    if tint is not None:
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(tint))
        painter.end()

    pixmap.setDevicePixelRatio(ratio)
    return pixmap
