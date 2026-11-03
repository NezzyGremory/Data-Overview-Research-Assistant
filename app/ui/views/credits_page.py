"""Cinematic end-credits page for DORA."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPauseAnimation,
    QPropertyAnimation,
    QSequentialAnimationGroup,
    Qt,
    QTimer,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.scale import ui_px


_UI_ROOT = Path(__file__).resolve().parents[1]
_CREDITS_PATH = _UI_ROOT / "data" / "credits.json"


class CinematicCreditsPage(QWidget):
    """A full-page credits sequence with native Qt animations and controls."""

    SCROLL_PIXELS_PER_SECOND = 38

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.setObjectName("copyrightPage")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._state = "idle"
        self._animation = None
        self._paused_for_hide = False
        self._paused_by_user = False
        self._saved_shell_visibility = None

        with _CREDITS_PATH.open("r", encoding="utf-8") as source:
            self._credits = json.load(source)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(ui_px(6))

        controls = QWidget(self)
        controls.setObjectName("creditsControls")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(ui_px(18), ui_px(6), ui_px(18), ui_px(6))
        controls_layout.setSpacing(ui_px(10))

        self.back_button = QPushButton("Back to DORA", controls)
        self.back_button.setObjectName("creditsControlButton")
        self.back_button.setAccessibleName("Close credits and return to DORA")
        self.back_button.clicked.connect(self._back_to_app)
        controls_layout.addWidget(self.back_button)
        controls_layout.addStretch(1)

        self.reduce_motion = QCheckBox("Reduce motion", controls)
        self.reduce_motion.setObjectName("creditsReduceMotion")
        self.reduce_motion.setAccessibleName("Show credits without animation")
        self.reduce_motion.toggled.connect(self._set_reduced_motion)
        controls_layout.addWidget(self.reduce_motion)

        self.pause_button = QPushButton("Pause", controls)
        self.pause_button.setObjectName("creditsControlButton")
        self.pause_button.setAccessibleName("Pause credits scrolling")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.pause_button.hide()
        controls_layout.addWidget(self.pause_button)

        self.replay_button = QPushButton("Replay Credits", controls)
        self.replay_button.setObjectName("creditsControlButton")
        self.replay_button.clicked.connect(self._start_experience)
        self.replay_button.hide()
        controls_layout.addWidget(self.replay_button)

        self.skip_button = QPushButton("Skip Credits", controls)
        self.skip_button.setObjectName("creditsControlButton")
        self.skip_button.setAccessibleName("Skip to the closing message")
        self.skip_button.clicked.connect(self._show_closing)
        controls_layout.addWidget(self.skip_button)
        root.addWidget(controls)

        self.stage = QStackedWidget(self)
        self.stage.setObjectName("creditsStage")
        root.addWidget(self.stage, 1)

        self.opening_scene = self._make_centered_scene("creditsOpeningScene")
        self._opening_opacity = QGraphicsOpacityEffect(self.opening_scene)
        self._opening_opacity.setOpacity(0.0)
        self.opening_scene.setGraphicsEffect(self._opening_opacity)
        self._build_opening_scene()
        self.stage.addWidget(self.opening_scene)

        self.roll_area = QScrollArea(self.stage)
        self.roll_area.setObjectName("creditsRollArea")
        self.roll_area.setFrameShape(QFrame.NoFrame)
        self.roll_area.setWidgetResizable(True)
        self.roll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.roll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.roll_area.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.track = QWidget()
        self.track.setObjectName("creditsRollContent")
        self.track_layout = QVBoxLayout(self.track)
        self.track_layout.setContentsMargins(ui_px(24), 0, ui_px(24), 0)
        self.track_layout.setSpacing(0)
        self.top_spacer = QWidget(self.track)
        self.bottom_spacer = QWidget(self.track)
        self.track_layout.addWidget(self.top_spacer)
        self._build_roll_content()
        self.track_layout.addWidget(self.bottom_spacer)
        self.roll_area.setWidget(self.track)
        self.stage.addWidget(self.roll_area)

        self.closing_scene = self._make_centered_scene("creditsClosingScene")
        self._closing_opacity = QGraphicsOpacityEffect(self.closing_scene)
        self._closing_opacity.setOpacity(0.0)
        self.closing_scene.setGraphicsEffect(self._closing_opacity)
        self._build_closing_scene()
        self.stage.addWidget(self.closing_scene)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._sync_roll_geometry)

    @staticmethod
    def _make_centered_scene(object_name: str) -> QWidget:
        scene = QWidget()
        scene.setObjectName(object_name)
        scene.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(scene)
        layout.setContentsMargins(ui_px(24), ui_px(24), ui_px(24), ui_px(24))
        layout.setSpacing(ui_px(12))
        layout.addStretch(1)
        scene._credits_layout = layout
        return scene

    @staticmethod
    def _label(text: str, object_name: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        label.setObjectName(object_name)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setMaximumWidth(ui_px(760))
        label.setMinimumWidth(0)
        label.setProperty("preserveVisibleCopy", True)
        return label

    def _add_scene_label(self, scene: QWidget, text: str, object_name: str):
        label = self._label(text, object_name, scene)
        scene._credits_layout.addWidget(label, 0, Qt.AlignHCenter)
        return label

    def _build_opening_scene(self):
        layout = self.opening_scene._credits_layout
        layout.addStretch(1)
        self._add_scene_label(
            self.opening_scene,
            str(self._credits.get("intro", "A METRIKS STUDIO PRODUCTION")).upper(),
            "creditsProduction",
        )
        self._add_scene_label(self.opening_scene, "PRESENTS", "creditsSmallCaption")
        self._add_scene_label(
            self.opening_scene,
            str(self._credits.get("app", "DORA")).upper(),
            "creditsDoraTitle",
        )
        self._add_scene_label(
            self.opening_scene,
            str(self._credits.get("tagline", "DATA OVERVIEW & RESEARCH ASSISTANT")).upper(),
            "creditsTagline",
        )
        self._add_scene_label(
            self.opening_scene,
            str(self._credits.get("project", "A PROJECT BY METRIKS STUDIO")).upper(),
            "creditsProjectLine",
        )
        layout.addStretch(1)

    def _build_roll_content(self):
        self._add_roll_label(
            str(self._credits.get("intro", "A METRIKS STUDIO PRODUCTION")).upper(),
            "creditsProduction",
            ui_px(22),
        )
        self._add_roll_label("PRESENTS", "creditsSmallCaption", ui_px(32))
        self._add_roll_label(
            str(self._credits.get("app", "DORA")).upper(),
            "creditsDoraTitle",
            ui_px(5),
        )
        self._add_roll_label(
            str(self._credits.get("tagline", "DATA OVERVIEW & RESEARCH ASSISTANT")).upper(),
            "creditsTagline",
            ui_px(14),
        )
        self._add_roll_label(
            str(self._credits.get("project", "A PROJECT BY METRIKS STUDIO")).upper(),
            "creditsProjectLine",
            ui_px(68),
        )

        previous_group = None
        for section in self._credits.get("sections", []):
            group = str(section.get("group", "")).strip().upper()
            if group and group != previous_group:
                self._add_roll_label(group, "creditsGroup", ui_px(20))
                previous_group = group
            self._add_roll_label(
                str(section.get("role", "")).upper(), "creditsRole", ui_px(10)
            )
            for name in section.get("names", []):
                self._add_roll_label(str(name), "creditsName", ui_px(8))
            self.track_layout.addSpacing(ui_px(42))

        self._add_roll_label("SPECIAL THANKS", "creditsGroup", ui_px(18))
        self._add_roll_label(
            str(self._credits.get("thanks", "")), "creditsThanks", ui_px(24)
        )
        self._add_roll_label(
            str(self._credits.get("closing", "THANK YOU FOR USING DORA.")).upper(),
            "creditsRole",
            ui_px(48),
        )
        self._add_roll_label(
            str(self._credits.get("footer_studio", "A METRIKS STUDIO PRODUCTION")).upper(),
            "creditsProduction",
            ui_px(16),
        )
        self._add_roll_label(
            str(
                self._credits.get(
                    "footer", "© 2026 METRIKS STUDIO. ALL RIGHTS RESERVED."
                )
            ).upper(),
            "creditsFooter",
            ui_px(32),
        )

    def _add_roll_label(self, text: str, object_name: str, trailing_space: int):
        label = self._label(text, object_name, self.track)
        self.track_layout.addWidget(label, 0, Qt.AlignHCenter)
        self.track_layout.addSpacing(trailing_space)
        return label

    def _build_closing_scene(self):
        layout = self.closing_scene._credits_layout
        layout.addStretch(1)
        self._add_scene_label(
            self.closing_scene,
            str(
                self._credits.get(
                    "journey_thanks", "THANK YOU FOR BEING PART OF OUR JOURNEY."
                )
            ).upper(),
            "creditsClosingMessage",
        )
        self._add_scene_label(
            self.closing_scene,
            str(self._credits.get("app", "DORA")).upper(),
            "creditsClosingDora",
        )
        self._add_scene_label(
            self.closing_scene,
            str(self._credits.get("footer_studio", "A METRIKS STUDIO PRODUCTION")).upper(),
            "creditsClosingStudio",
        )
        self._add_scene_label(
            self.closing_scene,
            str(
                self._credits.get(
                    "footer", "© 2026 METRIKS STUDIO. ALL RIGHTS RESERVED."
                )
            ).upper(),
            "creditsFooter",
        )
        layout.addStretch(1)

    def showEvent(self, event):
        super().showEvent(event)
        self._resize_timer.start(0)
        QTimer.singleShot(0, self._hide_app_chrome)
        if self._paused_for_hide and self._animation is not None:
            self._animation.resume()
            self._paused_for_hide = False
        elif self._state in ("idle", "finished"):
            if self.reduce_motion.isChecked():
                self._show_static_credits()
            else:
                self._start_experience()

    def hideEvent(self, event):
        self._restore_app_chrome()
        if (
            self._animation is not None
            and self._animation.state() == QAbstractAnimation.State.Running
        ):
            self._animation.pause()
            self._paused_for_hide = True
        super().hideEvent(event)

    def _hide_app_chrome(self):
        window = self.main_window
        if (
            window is None
            or getattr(window, "_current_page_key", None) != "copyright"
            or not self.isVisible()
        ):
            return
        widgets = [
            getattr(window, "sidebar_container", None),
            getattr(window, "topbar", None),
            window.menuBar(),
            window.statusBar(),
        ]
        if self._saved_shell_visibility is None:
            self._saved_shell_visibility = [
                (widget, not widget.isHidden())
                for widget in widgets
                if widget is not None
            ]
        for widget, _was_visible in self._saved_shell_visibility:
            widget.hide()

    def _restore_app_chrome(self):
        if self._saved_shell_visibility is None:
            return
        for widget, was_visible in self._saved_shell_visibility:
            try:
                widget.setVisible(was_visible)
            except RuntimeError:
                pass
        self._saved_shell_visibility = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_resize_timer"):
            self._resize_timer.start(0)

    def _sync_roll_geometry(self):
        if not hasattr(self, "roll_area"):
            return
        self._hide_app_chrome()
        self._sync_text_column_width()
        was_rolling = self._state == "rolling" and self._animation is not None
        was_paused = bool(
            was_rolling
            and self._animation.state() == QAbstractAnimation.State.Paused
        )
        current_value = self.roll_area.verticalScrollBar().value()
        if was_rolling:
            self._stop_animation()
        spacer_height = max(1, self.roll_area.viewport().height())
        self.top_spacer.setFixedHeight(spacer_height)
        self.bottom_spacer.setFixedHeight(spacer_height)
        self.track.adjustSize()
        self.roll_area.widget().adjustSize()
        if was_rolling and self._state == "rolling":
            self._start_scroll(current_value=current_value, paused=was_paused)

    def _sync_text_column_width(self):
        for container in (self.opening_scene, self.closing_scene, self.track):
            width = container.width() or min(self.width(), self.stage.width())
            available = min(
                ui_px(760), max(ui_px(180), width - ui_px(48))
            )
            for label in container.findChildren(QLabel):
                label.setMinimumWidth(available)
                label.setMaximumWidth(available)
            layout = container.layout()
            if layout is not None:
                layout.activate()

    def _start_experience(self):
        self._stop_animation()
        self._state = "opening"
        self._paused_by_user = False
        self.pause_button.setText("Pause")
        self.pause_button.setAccessibleName("Pause credits scrolling")
        self.pause_button.hide()
        self.skip_button.show()
        self.replay_button.hide()
        self.roll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stage.setCurrentWidget(self.opening_scene)
        self._sync_text_column_width()
        self._opening_opacity.setOpacity(0.0)

        sequence = QSequentialAnimationGroup(self)
        sequence.addAnimation(
            self._opacity_animation(self._opening_opacity, 0.0, 1.0, 1100)
        )
        sequence.addAnimation(QPauseAnimation(900, sequence))
        sequence.addAnimation(
            self._opacity_animation(self._opening_opacity, 1.0, 0.0, 650)
        )
        sequence.finished.connect(self._begin_roll)
        self._animation = sequence
        sequence.start()

    @staticmethod
    def _opacity_animation(effect, start: float, end: float, duration: int):
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        return animation

    def _begin_roll(self):
        self._stop_animation()
        if self.reduce_motion.isChecked():
            self._show_static_credits()
            return
        self._state = "rolling"
        self.stage.setCurrentWidget(self.roll_area)
        self._sync_text_column_width()
        self.roll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pause_button.show()
        self.skip_button.show()
        self._resize_timer.start(0)
        QTimer.singleShot(0, self._start_scroll)

    def _start_scroll(self, current_value: int = 0, paused: bool = False):
        if self._state != "rolling" or self.reduce_motion.isChecked():
            return
        scrollbar = self.roll_area.verticalScrollBar()
        maximum = scrollbar.maximum()
        scrollbar.setValue(min(max(0, current_value), maximum))
        distance = max(1, maximum - scrollbar.value())
        duration = max(
            18000,
            int(distance / self.SCROLL_PIXELS_PER_SECOND * 1000),
        )
        animation = QPropertyAnimation(scrollbar, b"value", self)
        animation.setDuration(duration)
        animation.setStartValue(scrollbar.value())
        animation.setEndValue(maximum)
        animation.setEasingCurve(QEasingCurve.Linear)
        animation.finished.connect(self._show_closing)
        self._animation = animation
        animation.start()
        if paused:
            animation.pause()

    def _show_closing(self):
        if self._state in ("closing", "finished", "static"):
            return
        self._stop_animation()
        self._state = "closing"
        self.pause_button.hide()
        self.skip_button.hide()
        self.stage.setCurrentWidget(self.closing_scene)
        self._sync_text_column_width()
        self._closing_opacity.setOpacity(0.0)

        sequence = QSequentialAnimationGroup(self)
        sequence.addAnimation(
            self._opacity_animation(self._closing_opacity, 0.0, 1.0, 1100)
        )
        sequence.addAnimation(QPauseAnimation(2400, sequence))
        sequence.addAnimation(
            self._opacity_animation(self._closing_opacity, 1.0, 0.0, 1700)
        )
        sequence.finished.connect(self._finish_experience)
        self._animation = sequence
        sequence.start()

    def _finish_experience(self):
        self._stop_animation()
        self._state = "finished"
        self.replay_button.show()
        self.skip_button.hide()
        self.pause_button.hide()

    def _show_static_credits(self):
        self._stop_animation()
        self._state = "static"
        self.pause_button.hide()
        self.skip_button.hide()
        self.replay_button.hide()
        self.roll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.stage.setCurrentWidget(self.roll_area)
        self._sync_text_column_width()
        self.roll_area.verticalScrollBar().setValue(0)
        self._resize_timer.start(0)

    def _set_reduced_motion(self, enabled: bool):
        if enabled:
            self._show_static_credits()
        elif self._state == "static":
            self._start_experience()

    def _toggle_pause(self):
        if self._animation is None or self._state != "rolling":
            return
        if self._paused_by_user:
            self._animation.resume()
            self._paused_by_user = False
            self.pause_button.setText("Pause")
            self.pause_button.setAccessibleName("Pause credits scrolling")
        else:
            self._animation.pause()
            self._paused_by_user = True
            self.pause_button.setText("Resume")
            self.pause_button.setAccessibleName("Resume credits scrolling")

    def _stop_animation(self):
        animation = self._animation
        self._animation = None
        if animation is not None:
            animation.stop()
            animation.deleteLater()

    def _back_to_app(self):
        self._stop_animation()
        self._state = "idle"
        self._paused_for_hide = False
        if self.main_window is not None:
            self.main_window._show_page("dashboard")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Backspace):
            self._back_to_app()
            event.accept()
            return
        if event.key() == Qt.Key_Space and self._state == "rolling":
            self._toggle_pause()
            event.accept()
            return
        super().keyPressEvent(event)
