# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

project_root = Path(SPECPATH).resolve()
ui_root = project_root / "app" / "ui"
app_icon = ui_root / "assets" / "dataset_research.ico"

runtime_datas = []
for source in (ui_root / "assets").rglob("*"):
    if source.is_file() and source.suffix.lower() in {".ico", ".jpg", ".jpeg", ".png", ".svg", ".ttf", ".otf", ".txt"}:
        runtime_datas.append((str(source), str(source.parent.relative_to(project_root))))
for source in (ui_root / "styles").glob("*.qss"):
    runtime_datas.append((str(source), str(source.parent.relative_to(project_root))))
runtime_datas.append((str(ui_root / "data" / "credits.json"), "app/ui/data"))
runtime_datas.append((str(ui_root / "app.qss"), "app/ui"))
runtime_datas.append((str(project_root / "app" / "reports" / "templates" / "report.html"), "app/reports/templates"))

a = Analysis(
    [str(project_root / "app" / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=runtime_datas + collect_data_files("pip") + copy_metadata("pip"),
    hiddenimports=[
        "app.ui.editor_notebook_kernel",
        "PySide6.QtSvg",
        *collect_submodules("pip"),
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtGraphs",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtQml",
        "PySide6.QtQmlMeta",
        "PySide6.QtQmlModels",
        "PySide6.QtQmlWorkerScript",
        "PySide6.QtQuick",
        "PySide6.QtQuickControls2",
        "PySide6.QtQuickWidgets",
        "PySide6.QtVirtualKeyboard",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebEngineWidgets",
    ],
    noarchive=False,
    optimize=1,
)

# PyInstaller hooks can pick up development files shipped inside wheels. They
# are not used at runtime and can add thousands of files to the installer.
a.datas = [
    entry for entry in a.datas
    if not entry[0].lower().endswith((".lib", ".pxi", ".pxd", ".pyx", ".pyx.tp"))
    and not any(
        marker in ("/" + entry[0].replace("\\", "/").lower())
        for marker in (
            "/pyarrow/include/",
            "/pyarrow/includes/",
            "/pyarrow/src/",
            "/pyarrow/tests/",
            "/pyarrow/benchmark/",
            "/pyarrow/benchmarks/",
        )
    )
    and not entry[0].lower().endswith(("/meson.build", "/cmakelists.txt"))
]

# Keep the local analysis features while omitting optional plugins/backends the
# application never imports (remote Arrow filesystems, Flight/Substrait, AVIF,
# and Qt's QML/Quick/PDF/image-format extras).
unused_binary_markers = (
    "/pyarrow/arrow_flight.dll",
    "/pyarrow/arrow_python_flight.dll",
    "/pyarrow/arrow_python_parquet_encryption.dll",
    "/pyarrow/arrow_substrait.dll",
    "/pyarrow/_azurefs.",
    "/pyarrow/_flight.",
    "/pyarrow/_gcsfs.",
    "/pyarrow/_hdfs.",
    "/pyarrow/_s3fs.",
    "/pyarrow/_dataset_parquet_encryption.",
    "/pyarrow/_parquet_encryption.",
    "/pyarrow/_substrait.",
    "/pyside6/qt6qml",
    "/pyside6/qt6quick",
    "/pyside6/qt6virtualkeyboard.dll",
    "/pyside6/qt6pdf.dll",
    "/pyside6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll",
    "/pyside6/plugins/imageformats/qpdf.dll",
    "/pyside6/plugins/imageformats/qgif.dll",
    "/pyside6/plugins/imageformats/qicns.dll",
    "/pyside6/plugins/imageformats/qtga.dll",
    "/pyside6/plugins/imageformats/qtiff.dll",
    "/pyside6/plugins/imageformats/qwbmp.dll",
    "/pyside6/plugins/imageformats/qwebp.dll",
    "/pil/_avif.",
)
a.binaries = [
    entry for entry in a.binaries
    if not any(
        marker in ("/" + entry[0].replace("\\", "/").lower())
        for marker in unused_binary_markers
    )
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DORA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(app_icon)],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DORA",
)
