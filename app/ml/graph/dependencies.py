"""Preflight dependency check, safe venv-only installer, and fallback mechanism."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Security whitelist with pinned version constraints and approximate download sizes
PACKAGE_WHITELIST: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "pip_name": "xgboost",
        "pinned": "xgboost>=2.0.0",
        "est_size_mb": 65,
        "category": "Gradient Boosting",
        "import_name": "xgboost",
    },
    "lightgbm": {
        "pip_name": "lightgbm",
        "pinned": "lightgbm>=4.0.0",
        "est_size_mb": 25,
        "category": "Gradient Boosting",
        "import_name": "lightgbm",
    },
    "catboost": {
        "pip_name": "catboost",
        "pinned": "catboost>=1.2.0",
        "est_size_mb": 80,
        "category": "Gradient Boosting",
        "import_name": "catboost",
    },
    "imbalanced-learn": {
        "pip_name": "imbalanced-learn",
        "pinned": "imbalanced-learn>=0.12.0",
        "est_size_mb": 15,
        "category": "Imbalanced Data",
        "import_name": "imblearn",
    },
    "optuna": {
        "pip_name": "optuna",
        "pinned": "optuna>=3.6.0",
        "est_size_mb": 20,
        "category": "Hyperparameter Tuning",
        "import_name": "optuna",
    },
    "shap": {
        "pip_name": "shap",
        "pinned": "shap>=0.44.0",
        "est_size_mb": 20,
        "category": "Explainability",
        "import_name": "shap",
    },
    "scikit-learn": {
        "pip_name": "scikit-learn",
        "pinned": "scikit-learn>=1.4.0",
        "est_size_mb": 30,
        "category": "Core ML",
        "import_name": "sklearn",
    },
}

# Fallback recommendations when a library fails to install or is missing
FALLBACK_MAP: Dict[str, Dict[str, str]] = {
    "xgboost": {
        "fallback_spec_id": "model:gradient_boosting_classifier",
        "fallback_name": "Gradient Boosting Classifier",
        "reason": "Scikit-learn GradientBoostingClassifier bawaan tidak memerlukan paket tambahan.",
    },
    "lightgbm": {
        "fallback_spec_id": "model:hist_gradient_boosting_classifier",
        "fallback_name": "Hist Gradient Boosting Classifier",
        "reason": "HistGradientBoostingClassifier bawaan scikit-learn memiliki kecepatan dan akurasi setara LightGBM.",
    },
    "catboost": {
        "fallback_spec_id": "model:random_forest_classifier",
        "fallback_name": "Random Forest Classifier",
        "reason": "Random Forest Classifier bawaan scikit-learn siap digunakan langsung.",
    },
    "imbalanced-learn": {
        "fallback_spec_id": "none",
        "fallback_name": "Class Weight Balanced",
        "reason": "Gunakan parameter class_weight='balanced' pada model klasifikasi bawaan scikit-learn.",
    },
}


def is_package_installed(import_name: str) -> bool:
    """Check if package can be imported in the current environment."""
    return importlib.util.find_spec(import_name) is not None


def get_installed_version(dist_name: str) -> Optional[str]:
    """Get installed package version from metadata."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def preflight_check(nodes: List[Dict[str, Any]], specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Examine all nodes in the graph against engine specifications.
    Identify any required packages that are currently uninstalled.
    """
    missing_packages: Dict[str, Dict[str, Any]] = {}
    installed_packages: Dict[str, str] = {}
    unwhitelisted: List[str] = []

    for node in nodes:
        spec_id = node.get("spec_id") or node.get("id")
        spec = specs.get(spec_id)
        if not spec:
            continue

        requires = getattr(spec, "requires", None) or []
        check_pkg = getattr(spec, "check_package", None)

        # Normalize requirements
        req_list = list(requires)
        if check_pkg and not req_list:
            req_list.append({"package": check_pkg, "version": "any"})

        for req in req_list:
            pkg_name = req.get("package", "")
            if not pkg_name:
                continue

            # Map package name to whitelist key
            whitelist_entry = None
            for key, val in PACKAGE_WHITELIST.items():
                if pkg_name.lower() in (key.lower(), val["pip_name"].lower(), val["import_name"].lower()):
                    whitelist_entry = (key, val)
                    break

            import_name = whitelist_entry[1]["import_name"] if whitelist_entry else pkg_name
            pip_name = whitelist_entry[1]["pip_name"] if whitelist_entry else pkg_name

            if is_package_installed(import_name):
                installed_packages[pip_name] = get_installed_version(pip_name) or "installed"
            else:
                if not whitelist_entry:
                    unwhitelisted.append(pkg_name)
                    continue

                key, val = whitelist_entry
                if key not in missing_packages:
                    fallback = FALLBACK_MAP.get(key, {})
                    missing_packages[key] = {
                        "package": key,
                        "pip_name": val["pip_name"],
                        "pinned_version": val["pinned"],
                        "estimated_size_mb": val["est_size_mb"],
                        "category": val["category"],
                        "nodes_requiring": [node.get("id", spec_id)],
                        "in_whitelist": True,
                        "fallback_spec_id": fallback.get("fallback_spec_id"),
                        "fallback_name": fallback.get("fallback_name"),
                        "fallback_reason": fallback.get("reason"),
                    }
                else:
                    missing_packages[key]["nodes_requiring"].append(node.get("id", spec_id))

    is_ready = len(missing_packages) == 0 and len(unwhitelisted) == 0
    return {
        "ready": is_ready,
        "missing_packages": list(missing_packages.values()),
        "unwhitelisted_packages": unwhitelisted,
        "installed_packages": installed_packages,
        "active_python": sys.executable,
    }


def install_package(
    package_name: str,
    user_confirmed: bool = False,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """
    Install a package safely into the current active virtual environment (.venv).
    Enforces user confirmation, whitelist validation, and pinned versions.
    """
    if not user_confirmed:
        raise PermissionError(
            f"Instalasi paket '{package_name}' dibatalkan: Konfirmasi eksplisit pengguna diperlukan."
        )

    # Validate whitelist
    key = package_name.lower()
    entry = None
    for k, v in PACKAGE_WHITELIST.items():
        if key in (k.lower(), v["pip_name"].lower()):
            entry = v
            break

    if not entry:
        raise ValueError(
            f"Paket '{package_name}' tidak diizinkan. Hanya paket dalam whitelist resmi yang dapat diinstal otomatis."
        )

    pinned_spec = entry["pinned"]
    python_exe = sys.executable

    cmd = [python_exe, "-m", "pip", "install", pinned_spec, "--no-warn-script-location"]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode == 0:
            version = get_installed_version(entry["pip_name"]) or "installed"
            return {
                "success": True,
                "package": entry["pip_name"],
                "version": version,
                "message": f"Paket {entry['pip_name']} ({version}) berhasil dipasang.",
            }
        else:
            fallback = FALLBACK_MAP.get(package_name.lower(), {})
            return {
                "success": False,
                "package": entry["pip_name"],
                "error": proc.stderr.strip() or proc.stdout.strip(),
                "fallback": fallback,
                "message": f"Gagal memasang {entry['pip_name']}. Log: {proc.stderr[:200]}",
            }
    except subprocess.TimeoutExpired:
        fallback = FALLBACK_MAP.get(package_name.lower(), {})
        return {
            "success": False,
            "package": entry["pip_name"],
            "error": "Instalasi timeout (melebihi batas waktu)",
            "fallback": fallback,
            "message": f"Instalasi {entry['pip_name']} melebihi batas waktu {timeout_seconds}s.",
        }
    except Exception as exc:
        fallback = FALLBACK_MAP.get(package_name.lower(), {})
        return {
            "success": False,
            "package": entry["pip_name"],
            "error": str(exc),
            "fallback": fallback,
            "message": f"Terjadi kesalahan saat instalasi: {exc}",
        }
