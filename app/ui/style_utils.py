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
    """Render an icon that keeps its requested logical size on HiDPI screens.

    Qt6's QIcon.pixmap() already treats the requested size as logical pixels and
    returns a pixmap with the screen DPR applied, so the size must not be
    pre-multiplied here. Pre-multiplying made every icon ``ratio``x too large on
    125%/150%/175% displays, which clipped fixed-size icon labels.
    """
    if isinstance(logical_size, QSize):
        width, height = logical_size.width(), logical_size.height()
    else:
        width = height = int(logical_size)
    width = max(1, int(width))
    height = max(1, int(height))

    ratio = max(1.0, float(device_pixel_ratio or 1.0))
    pixmap = QIcon(str(path)).pixmap(QSize(width, height))
    if pixmap.isNull():
        return pixmap

    # Mixed-DPI setups: if Qt used another DPR than the widget's screen, re-request
    # so the physical buffer matches, then tag it with the widget's DPR below.
    applied = float(pixmap.devicePixelRatio() or 1.0)
    if applied > 0 and abs(applied - ratio) > 1e-6:
        pixmap = QIcon(str(path)).pixmap(
            QSize(
                max(1, ceil(width * ratio / applied)),
                max(1, ceil(height * ratio / applied)),
            )
        )
        if pixmap.isNull():
            return pixmap

    if tint is not None:
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(tint))
        painter.end()

    pixmap.setDevicePixelRatio(ratio)
    return pixmap
