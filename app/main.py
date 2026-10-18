import os
import sys
import threading
import traceback
from pathlib import Path


def _ensure_project_interpreter() -> None:
    """Use the repository venv even when this module is launched directly."""
    if getattr(sys, "frozen", False):
        return

    project_root = Path(__file__).resolve().parents[1]
    project_python = project_root / ".venv" / "Scripts" / "python.exe"
    if not project_python.is_file():
        return
    if Path(sys.executable).resolve() == project_python.resolve():
        return

    launcher = project_root / "run.py"
    os.execv(
        str(project_python),
        [str(project_python), str(launcher), *sys.argv[1:]],
    )


if __name__ == "__main__":
    _ensure_project_interpreter()

from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from app.core.config import init_environment
init_environment()


class _MainWindowModuleLoader(QObject):
    """Import the UI modules off the GUI thread while the splash can animate."""

    finished = Signal(object, object)

    def load(self):
        try:
            from app.ui.main_window import MainWindow
        except BaseException as error:
            self.finished.emit(None, (error, traceback.format_exc()))
            return
        self.finished.emit(MainWindow, None)


class _StartupController(QObject):
    def __init__(self, app, splash):
        super().__init__(app)
        self.app = app
        self.splash = splash
        self.window = None
        self.loader = _MainWindowModuleLoader()
        self.loader.finished.connect(self._on_module_loaded)
        self.thread = threading.Thread(target=self.loader.load, daemon=True)

    def start(self):
        self.splash.show()
        self.app.processEvents()
        self.thread.start()

    @Slot(object, object)
    def _on_module_loaded(self, main_window_class, error_info):
        if error_info is not None:
            error, formatted_traceback = error_info
            sys.stderr.write(formatted_traceback)
            self.splash.set_message(f"DORA tidak dapat dimuat: {error}")
            self.splash.show_startup_error(str(error))
            return

        try:
            self.splash.set_message("Menyiapkan workspace penelitian…")
            self.window = main_window_class(startup_splash=self.splash)
            self.window.show()
            self.splash.close()
            self.splash.deleteLater()
        except BaseException as error:
            traceback.print_exc()
            self.splash.show_startup_error(str(error))


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
    from app.ui.startup_splash import StartupSplash

    splash = StartupSplash()
    startup = _StartupController(app, splash)
    startup.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
