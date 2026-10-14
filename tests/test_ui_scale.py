from PySide6.QtCore import QSettings

from app.ui.scale import (
    SCALE_MAX,
    SCALE_MIN,
    SCALE_STEP,
    ScaleManager,
    build_stylesheet,
    scale_stylesheet,
)


class _Geometry:
    def __init__(self, height):
        self._height = height

    def height(self):
        return self._height


class _Screen:
    def __init__(self, height, dpi=96):
        self._geometry = _Geometry(height)
        self._dpi = dpi

    def availableGeometry(self):
        return self._geometry

    def logicalDotsPerInch(self):
        return self._dpi


def _settings(tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    settings.clear()
    return settings


def test_build_stylesheet_resolves_template_at_one_and_point_eight():
    template_path = "app/ui/app.qss"
    with open(template_path, encoding="utf-8-sig") as source:
        template = source.read()

    actual = build_stylesheet(1.0, template_path)
    expected = scale_stylesheet(template, 1.0)
    assert actual == expected
    assert "@px(" not in actual
    assert actual.count("{") == actual.count("}")
    fractional_qss = "QLabel { border-width: 0.5px; padding: 3px; }"
    assert scale_stylesheet(fractional_qss, 1.0) == fractional_qss

    scaled = build_stylesheet(0.8, template_path)
    assert "font-size: 10px" in scaled
    assert "@px(" not in scaled
    assert scaled.count("{") == scaled.count("}")


def test_scale_manager_clamps_zoom_resets_and_scales_pixels(tmp_path):
    manager = ScaleManager(_settings(tmp_path), _Screen(1000))
    manager.set_scale(2.0)
    assert manager.get_scale() == SCALE_MAX
    manager.set_scale(0.1)
    assert manager.get_scale() == SCALE_MIN

    manager.set_scale(1.0)
    manager.zoom_in()
    assert manager.get_scale() == 1.0 + SCALE_STEP
    manager.zoom_out()
    assert manager.get_scale() == 1.0
    assert manager.px(13) == 13

    manager.reset()
    assert manager.is_auto
    assert manager.get_scale() == 1.0


def test_scale_manager_reads_corrupt_setting_as_auto(tmp_path):
    settings = _settings(tmp_path)
    settings.setValue("ui_scale", "not-a-number")
    manager = ScaleManager(settings, _Screen(800))
    assert manager.is_auto
    assert manager.get_scale() == 0.9
    assert settings.value("ui_scale") == "auto"


def test_auto_scale_uses_height_and_logical_dpi(tmp_path):
    manager = ScaleManager(_settings(tmp_path))
    assert manager.auto_scale(_Screen(719, 96)) == 0.8
    assert manager.auto_scale(_Screen(800, 96)) == 0.9
    assert manager.auto_scale(_Screen(900, 96)) == 1.0
    assert manager.auto_scale(_Screen(1000, 144)) == 0.95
