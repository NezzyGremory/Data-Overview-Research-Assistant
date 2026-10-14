import sys

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from app.core.config import init_environment
init_environment()


def _run_packaged_editor_command() -> int | None:
    if not getattr(sys, "frozen", False) or len(sys.argv) < 2:
        return None

    mode = sys.argv[1]
    if mode == "--dora-editor-kernel":
        from app.ui.editor_notebook_kernel import main as kernel_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        kernel_main()
        return 0

    if mode == "--dora-editor-pip":
        from pip._internal.cli.main import main as pip_main

        return int(pip_main(sys.argv[2:]) or 0)

    return None


def main():
    editor_exit_code = _run_packaged_editor_command()
    if editor_exit_code is not None:
        raise SystemExit(editor_exit_code)

    # Preserve fractional scaling for 125%, 150%, and 175% Windows displays.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    QCoreApplication.setOrganizationName("DORA")
    QCoreApplication.setApplicationName("Dataset Research")
    app = QApplication(sys.argv)
    from app.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
