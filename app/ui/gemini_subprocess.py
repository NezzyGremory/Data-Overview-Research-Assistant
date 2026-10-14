"""Run the optional cloud explanation outside the Qt application's process."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main() -> int:
    try:
        analysis = pickle.load(sys.stdin.buffer)
        from app.ai.gemini_explainer import GeminiDatasetExplainer

        result = GeminiDatasetExplainer().explain_with_source(analysis)
        pickle.dump(result, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
        return 0
    except BaseException as error:
        sys.stderr.write(f"Gemini helper failed: {type(error).__name__}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
