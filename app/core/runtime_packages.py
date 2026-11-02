"""Persistent storage for optional packages used by the frozen desktop app."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def runtime_package_directory() -> Path:
    """Return the per-user package directory shared by desktop features."""
    from PySide6.QtCore import QStandardPaths

    app_data = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
    app_data_root = Path(app_data) if app_data else Path.home() / ".dora"
    return app_data_root / "node_editor" / "site-packages"


def load_runtime_packages() -> Path | None:
    """Expose packages installed by the app to every feature on each launch."""
    if not getattr(sys, "frozen", False):
        return None

    package_directory = runtime_package_directory()
    if package_directory.is_dir():
        resolved = str(package_directory.resolve())
        installed_paths = {
            str(Path(entry).resolve()) for entry in sys.path if entry
        }
        if resolved not in installed_paths:
            sys.path.insert(0, resolved)
        importlib.invalidate_caches()
    return package_directory
