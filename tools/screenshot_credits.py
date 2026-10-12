"""Capture the animated credits overlay at its start, midpoint, and ending."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.views.credits_overlay import show_credits_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("screenshots"))
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(["dora-credits-preview"])
    window = MainWindow()
    window.resize(args.width, args.height)
    window.show()
    app.processEvents()

    overlay = show_credits_overlay(window)
    app.processEvents()
    animation = overlay._scroll_animation
    if animation is None:
        raise RuntimeError("Credits roll animation did not start")
    animation.pause()
    if overlay._fade_animation is not None:
        overlay._fade_animation.stop()
        overlay._opacity.setOpacity(1.0)
    overlay.repaint()
    QTest.qWait(80)
    app.processEvents()

    positions = (
        # Capture after the first part of the roll enters the viewport.
        ("credits-start.png", min(5000, animation.duration() // 5)),
        ("credits-middle.png", animation.duration() // 2),
        ("credits-end.png", max(0, animation.duration() - 250)),
    )
    for filename, position in positions:
        animation.setCurrentTime(position)
        overlay._opacity.setOpacity(1.0)
        app.processEvents()
        if not overlay.isVisible():
            raise RuntimeError("Credits overlay closed before all captures were saved")
        output = args.output_dir / filename
        if not window.grab().save(str(output)):
            raise RuntimeError(f"Could not save {output}")
        print(f"Screenshot saved: {output}")

    overlay.close_overlay()
    app.processEvents()
    window._session_data_cleaned = True
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
