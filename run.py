import os
import sys
from pathlib import Path


def _use_project_venv() -> None:
    """Restart with the project interpreter before importing Qt modules."""
    if getattr(sys, "frozen", False):
        return

    project_root = Path(__file__).resolve().parent
    project_python = project_root / ".venv" / "Scripts" / "python.exe"
    if not project_python.is_file():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == project_python.resolve():
        return

    os.execv(
        str(project_python),
        [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


_use_project_venv()

from app.core.config import init_environment
init_environment()

from app.main import main


if __name__ == "__main__":
    main()
