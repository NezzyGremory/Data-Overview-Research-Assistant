"""Small Qt helpers shared by Figma-matched UI pages."""

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


def apply_card_shadow(widget: QWidget) -> None:
    """Apply the subtle shadow used on Figma cards (QSS has no box-shadow)."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(4)
    shadow.setOffset(0, 2)
    shadow.setColor(QColor(0, 0, 0, 13))
    widget.setGraphicsEffect(shadow)
