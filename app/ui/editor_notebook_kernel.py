"""Persistent Python worker used by the Editor Analysis notebook UI."""

from __future__ import annotations

import ast
import base64
import contextlib
import importlib
import io
import json
import os
import sys
import traceback
from pathlib import Path


def _format_value(value, output: io.StringIO, rich_outputs: list[dict] | None = None):
    if value is None:
        return
    if hasattr(value, "to_string"):
        plain = value.to_string()
    else:
        plain = repr(value)
    if rich_outputs is None:
        output.write(plain)
        output.write("\n")
        return
    html_value = None
    renderer = getattr(value, "to_html", None)
    if callable(renderer):
        try:
            html_value = renderer(max_rows=50, max_cols=20, border=0)
        except TypeError:
            try:
                html_value = renderer(max_rows=50, border=0)
            except TypeError:
                try:
                    html_value = renderer()
                except Exception:
                    html_value = None
            except Exception:
                html_value = None
        except Exception:
            html_value = None
    if not html_value:
        html_renderer = getattr(value, "_repr_html_", None)
        if callable(html_renderer):
            try:
                html_value = html_renderer()
            except Exception:
                html_value = None
    if isinstance(html_value, str) and html_value and len(html_value) <= 1_500_000:
        rich_outputs.append({
            "data": {"text/plain": plain, "text/html": html_value},
            "metadata": {},
        })
        return
    output.write(plain)
    output.write("\n")


def _execute(request: dict, namespace: dict, pyplot):
    output = io.StringIO()
    rich_outputs = []
    error = False
    source = str(request.get("code", ""))
    original_display = namespace.get("display")

    def capture_display(value):
        _format_value(value, output, rich_outputs)

    try:
        importlib.invalidate_caches()
        tree = ast.parse(source, filename="<DORA Editor Analysis>", mode="exec")
        trailing_expression = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        statements = tree.body[:-1] if trailing_expression else tree.body
        if statements:
            body = ast.Module(body=statements, type_ignores=[])
            ast.fix_missing_locations(body)
        else:
            body = None
        namespace["display"] = capture_display
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if body is not None:
                exec(compile(body, "<DORA Editor Analysis>", "exec"), namespace)
            if trailing_expression is not None:
                expression = ast.Expression(body=trailing_expression.value)
                ast.fix_missing_locations(expression)
                value = eval(compile(expression, "<DORA Editor Analysis>", "eval"), namespace)
                _format_value(value, output, rich_outputs)
    except BaseException:
        error = True
        with contextlib.redirect_stderr(output):
            traceback.print_exc()
    finally:
        namespace["display"] = original_display

    charts = []
    for figure_number in pyplot.get_fignums()[:8]:
        try:
            figure = pyplot.figure(figure_number)
            image = io.BytesIO()
            figure.savefig(image, format="png", dpi=100, bbox_inches="tight")
            charts.append(base64.b64encode(image.getvalue()).decode("ascii"))
        except Exception:
            error = True
            output.write(traceback.format_exc())
    pyplot.close("all")
    return {
        "id": request.get("id"),
        "output": output.getvalue(),
        "charts": charts,
        "rich_outputs": rich_outputs,
        "error": error,
    }


def _write_message(message: dict):
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.__stdout__.buffer.write(payload + b"\n")
    sys.__stdout__.buffer.flush()


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit("Usage: editor_notebook_kernel.py <project_site_packages> <dataset_pickle> <workdir> [app_site_packages]")

    packages_path = Path(sys.argv[1]).resolve()
    dataset_path = Path(sys.argv[2]).resolve()
    workdir = Path(sys.argv[3]).resolve()
    app_site_packages = Path(sys.argv[4]).resolve() if len(sys.argv) == 5 and sys.argv[4] else None
    packages_path.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    sys.path.insert(0, str(packages_path))
    if app_site_packages is not None:
        sys.path.insert(1, str(app_site_packages))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    dataframe = pd.read_pickle(dataset_path)
    dataframe.columns = dataframe.columns.map(str)

    def display(value):
        _format_value(value, sys.stdout)

    namespace = {
        "__name__": "__main__",
        "df": dataframe,
        "pd": pd,
        "np": np,
        "plt": plt,
        "display": display,
    }
    _write_message({"kind": "ready"})

    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if request.get("kind") == "shutdown":
            break
        if request.get("kind") != "execute":
            continue
        _write_message(_execute(request, namespace, plt))


if __name__ == "__main__":
    main()
