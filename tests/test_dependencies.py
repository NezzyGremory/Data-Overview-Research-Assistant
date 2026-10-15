"""Tests for Stage 4: Dependency check + safe auto-installer."""

import pytest

from app.ml.graph.dependencies import (
    FALLBACK_MAP,
    PACKAGE_WHITELIST,
    install_package,
    preflight_check,
)
from app.ml.graph.engine import GraphEngine


def test_whitelist_contains_pinned_versions():
    for name, spec in PACKAGE_WHITELIST.items():
        assert "pinned" in spec
        assert ">=" in spec["pinned"] or "==" in spec["pinned"]
        assert spec["est_size_mb"] > 0
        assert "import_name" in spec


def test_preflight_check_builtins_ready():
    engine = GraphEngine()
    nodes = [
        {"id": "load", "spec_id": "load_dataset"},
        {"id": "dt", "spec_id": "model:decision_tree_classifier"},
    ]
    res = preflight_check(nodes, engine.specs)
    assert res["ready"] is True
    assert len(res["missing_packages"]) == 0


def test_preflight_check_flags_missing_package_with_fallback():
    engine = GraphEngine()
    # plugin:xgboost:classifier requires xgboost which is not installed
    nodes = [
        {"id": "xgb", "spec_id": "plugin:xgboost:classifier"},
    ]
    res = preflight_check(nodes, engine.specs)
    assert res["ready"] is False
    assert len(res["missing_packages"]) == 1

    missing = res["missing_packages"][0]
    assert missing["package"] == "xgboost"
    assert missing["estimated_size_mb"] == 65
    assert missing["in_whitelist"] is True
    assert missing["fallback_spec_id"] == "model:gradient_boosting_classifier"
    assert "GradientBoostingClassifier" in missing["fallback_reason"]


def test_install_requires_explicit_user_confirmation():
    with pytest.raises(PermissionError, match="Konfirmasi eksplisit pengguna diperlukan"):
        install_package("xgboost", user_confirmed=False)


def test_install_rejects_unwhitelisted_package():
    with pytest.raises(ValueError, match="Hanya paket dalam whitelist resmi"):
        install_package("unauthorized-hack-pkg", user_confirmed=True)


def test_fallback_map_provides_sklearn_alternatives():
    assert "xgboost" in FALLBACK_MAP
    assert FALLBACK_MAP["xgboost"]["fallback_spec_id"] == "model:gradient_boosting_classifier"
    assert "lightgbm" in FALLBACK_MAP
    assert FALLBACK_MAP["lightgbm"]["fallback_spec_id"] == "model:hist_gradient_boosting_classifier"
