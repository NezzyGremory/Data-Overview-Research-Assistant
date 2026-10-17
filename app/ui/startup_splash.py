"""Small animated DORA startup screen shown while the desktop UI loads."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget


def _asset_path(name: str) -> Path:
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
        return root / "app" / "ui" / "assets" / name
    return Path(__file__).resolve().parent / "assets" / name


class StartupSplash(QWidget):
    """A lightweight branded splash with a timer-driven spinner."""

    def __init__(self):
        super().__init__(
            None,
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint,
        )
        self.setObjectName("startupSplash")
        self.setFixedSize(460, 250)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._message = "Memulai aplikasi…"
        self._angle = 0
        self._logo = QPixmap(str(_asset_path("dataset_research.ico")))
        self._timer = QTimer(self)
        self._timer.setInterval(55)
        self._timer.timeout.connect(self._advance_animation)

    def showEvent(self, event):
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.move(
                available.x() + (available.width() - self.width()) // 2,
                available.y() + (available.height() - self.height()) // 2,
            )
        self._timer.start()
        super().showEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

    def set_message(self, message: str):
        self._message = str(message)
        self.update()

    def _advance_animation(self):
        self._angle = (self._angle + 24) % 360
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawRoundedRect(QRectF(1, 1, self.width() - 2, self.height() - 2), 20, 20)

        painter.setBrush(QColor("#EAF3FF"))
        painter.drawRoundedRect(QRectF(1, 1, self.width() - 2, 9), 18, 18)
        painter.drawRect(QRectF(1, 6, self.width() - 2, 8))

        if not self._logo.isNull():
            logo = self._logo.scaled(
                58, 58, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            painter.drawPixmap(36, 42, logo)

        painter.setPen(QColor("#10254B"))
        title_font = QFont("Segoe UI", 23, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(108, 70, "DORA")

        painter.setPen(QColor("#61738F"))
        subtitle_font = QFont("Segoe UI", 9)
        painter.setFont(subtitle_font)
        painter.drawText(110, 91, "Dataset Overview & Research Assistant")

        painter.setPen(QColor("#18325D"))
        message_font = QFont("Segoe UI", 10, QFont.DemiBold)
        painter.setFont(message_font)
        painter.drawText(36, 151, self._message)

        painter.setPen(QPen(QColor("#E3ECF8"), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(QRectF(391, 169, 28, 28), 0, 360 * 16)
        painter.setPen(QPen(QColor("#0866F5"), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(QRectF(391, 169, 28, 28), self._angle * 16, 105 * 16)

        painter.setPen(QColor("#91A2BA"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(36, 205, "Analisis berjalan secara lokal di perangkat ini")

        painter.setPen(QPen(QColor("#E5EEF9"), 1))
        painter.drawLine(36, 225, 424, 225)

    def show_startup_error(self, message: str):
        self._timer.stop()
        QMessageBox.critical(
            self,
            "DORA tidak dapat dibuka",
            f"DORA gagal disiapkan.\n\n{message}",
        )
        self.close()
        application = QApplication.instance()
        if application is not None:
            application.exit(1)
