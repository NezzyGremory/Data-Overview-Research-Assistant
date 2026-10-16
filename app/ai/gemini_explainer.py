from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.ai.local_explainer import LocalAcademicExplainer

# In-memory explanation cache: {fingerprint: (narrative, source)}
_EXPLANATION_CACHE: Dict[str, Tuple[str, str]] = {}
_QUOTA_BACKOFF: Dict[str, Tuple[str, float]] = {}


def get_saved_api_key() -> str | None:
    """
    Mengambil API key dari environment variable atau file konfigurasi lokal.
    Memudahkan aplikasi saat di-compile menjadi .exe tanpa harus setup .env manual.
    """
    from app.core.config import get_data_dir, init_environment
    init_environment()

    # 1. Cek environment variable
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Cek file config lokal data/config.json
    config_path = get_data_dir() / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            saved = data.get("gemini_api_key")
            if saved and str(saved).strip():
                return str(saved).strip()
        except Exception:
            pass

    return None


def save_api_key(api_key: str) -> None:
    """Menyimpan API key ke data/config.json agar persisten across session."""
    from app.core.config import get_data_dir

    config_path = get_data_dir() / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["gemini_api_key"] = api_key.strip()
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class GeminiDatasetExplainer:
    """
    Hybrid Dataset Research Explainer:
    1. Menggunakan Cloud Google Gemini jika API key tersedia dan kuota aktif.
    2. Mendukung multi-key rotation (pisahkan dengan koma).
    3. Caching berdasarkan fingerprint SHA-256 selama proses aplikasi aktif.
    4. Otomatis fallback ke LocalAcademicExplainer jika terkena HTTP 429 (rate limit),
       kuota habis, offline, atau tanpa API key.
    """

    DEFAULT_MODEL = "gemini-3.8-flash"
    FALLBACK_MODELS = (
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    )

    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        raw_key = api_key or get_saved_api_key()
        self.raw_api_key = raw_key
        # Parse multiple keys separated by comma or semicolon
        self.api_keys: List[str] = (
            [k.strip() for k in raw_key.replace(";", ",").split(",") if k.strip()]
            if raw_key
            else []
        )

        self.model = model or os.getenv(
            "GEMINI_MODEL",
            self.DEFAULT_MODEL,
        )

        try:
            configured_timeout = float(os.getenv("GEMINI_TIMEOUT", "15"))
        except (TypeError, ValueError):
            configured_timeout = 15.0
        self.timeout = min(max(configured_timeout, 5.0), 30.0)

        self.retries_per_model = 2
        self.local_explainer = LocalAcademicExplainer()
        self.last_source: str = "local"
        self.last_error: str = ""

    @property
    def enabled(self) -> bool:
        """Sistem selalu siap (hybrid) karena memiliki LocalAcademicExplainer bawaan."""
        return True

    @property
    def cloud_enabled(self) -> bool:
        """True jika terdapat API key Gemini yang terkonfigurasi."""
        return len(self.api_keys) > 0

    def explain(self, analysis: Dict[str, Any]) -> str:
        """Menghasilkan teks narasi penjelasan dataset (backward-compatible)."""
        text, _ = self.explain_with_source(analysis)
        return text

    def explain_with_source(self, analysis: Dict[str, Any]) -> Tuple[str, str]:
        """
        Menghasilkan narasi penjelasan beserta sumbernya: ('teks', 'gemini' | 'local').
        Dijamin tidak akan melempar exception fatal ke UI.
        """
        if not isinstance(analysis, dict):
            return "Data analisis tidak valid.", "local"

        # 1. Cek Caching berdasarkan SHA-256 fingerprint
        fp = analysis.get("fingerprint") or {}
        fp_hash = fp.get("fingerprint") if isinstance(fp, dict) else None
        key_fingerprint = hashlib.sha256("\0".join(self.api_keys).encode("utf-8")).hexdigest()
        if fp_hash and fp_hash in _EXPLANATION_CACHE:
            cached_text, cached_source = _EXPLANATION_CACHE[fp_hash]
            if cached_source == "local_rate_limited":
                backoff = _QUOTA_BACKOFF.get(fp_hash)
                if (
                    backoff
                    and backoff[0] == key_fingerprint
                    and time.monotonic() < backoff[1]
                ):
                    self.last_source = "local"
                    self.last_error = "Batas permintaan Gemini tercapai; hasil lokal dipakai untuk analisis ini."
                    return cached_text, "local"
                _EXPLANATION_CACHE.pop(fp_hash, None)
                _QUOTA_BACKOFF.pop(fp_hash, None)
                cached_source = ""
            # Do not let an old local fallback hide a newly configured API key
            # or prevent a retry after a temporary cloud outage.
            if cached_source == "gemini" or not self.cloud_enabled:
                self.last_source = cached_source
                if cached_source == "local" and not self.cloud_enabled:
                    self.last_error = "API key Gemini belum diatur."
                return cached_text, cached_source

        # 2. Jika tidak ada API key -> langsung gunakan LocalAcademicExplainer
        if not self.cloud_enabled:
            local_text = self.local_explainer.explain(analysis)
            if fp_hash:
                _EXPLANATION_CACHE[fp_hash] = (local_text, "local")
            self.last_source = "local"
            self.last_error = "API key Gemini belum diatur."
            return local_text, "local"

        # 3. Ada API key -> coba panggil Gemini dengan multi-key dan model fallback
        compact = self._compact_analysis(analysis)
        prompt = self._build_prompt(compact)
        models = self._models_to_try()

        gemini_success = False
        gemini_text = ""
        failures: List[str] = []

        # Bound the optional cloud request so a bad key/network cannot leave the
        # analysis worker waiting through every key/model/timeout combination.
        deadline = time.monotonic() + 20.0
        quota_limited = False
        for api_key in self.api_keys:
            key_rejected = False
            for model_name in models:
                if time.monotonic() >= deadline:
                    break

                url = (
                    "https://generativelanguage.googleapis.com/"
                    f"v1beta/models/{model_name}:generateContent"
                )
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 1000,
                        **(
                            {"thinkingConfig": {"thinkingLevel": "low"}}
                            if model_name.startswith("gemini-3")
                            else {}
                        ),
                    },
                }

                for attempt in range(1, self.retries_per_model + 1):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    request_timeout = min(self.timeout, remaining)
                    try:
                        response = httpx.post(
                            url,
                            headers={
                                "x-goog-api-key": api_key,
                                "Content-Type": "application/json",
                            },
                            json=payload,
                            timeout=httpx.Timeout(
                                request_timeout,
                                connect=min(8.0, request_timeout),
                            ),
                        )

                        if response.status_code == 200:
                            extracted = self._extract_text(response.json())
                            if extracted and len(extracted.strip()) > 50:
                                gemini_text = extracted.strip()
                                gemini_success = True
                            else:
                                failures.append(f"{model_name}: respons kosong")
                            break

                        if response.status_code in {401, 403}:
                            response_detail = str(response.text or "").lower()
                            if response.status_code == 403 and any(
                                marker in response_detail
                                for marker in ("quota", "resource_exhausted", "rate limit")
                            ):
                                failures.append(f"{model_name}: kuota atau batas permintaan tercapai (HTTP 403)")
                                quota_limited = True
                                break
                            failures.append(f"API key ditolak (HTTP {response.status_code})")
                            # An authorization error is key-specific; try the next key.
                            key_rejected = True
                            break
                        if response.status_code == 429:
                            failures.append(f"{model_name}: kuota atau batas permintaan tercapai (HTTP 429)")
                            quota_limited = True
                            break
                        if response.status_code in {400, 404}:
                            failures.append(f"{model_name}: HTTP {response.status_code}")
                            # Usually a model/access or request incompatibility; move
                            # to the next supported fallback without repeating it.
                            break
                        if (
                            response.status_code in self.RETRYABLE_STATUS_CODES
                            and attempt < self.retries_per_model
                            and time.monotonic() < deadline
                        ):
                            failures.append(f"{model_name}: HTTP {response.status_code}")
                            time.sleep(min(0.6 * attempt, max(0.0, deadline - time.monotonic())))
                            continue
                        failures.append(f"{model_name}: HTTP {response.status_code}")
                        break
                    except httpx.TimeoutException:
                        failures.append("koneksi Gemini timeout")
                        if attempt < self.retries_per_model and time.monotonic() < deadline:
                            time.sleep(min(0.4 * attempt, max(0.0, deadline - time.monotonic())))
                            continue
                        break
                    except httpx.RequestError:
                        failures.append("layanan Gemini tidak dapat dijangkau")
                        if attempt < self.retries_per_model and time.monotonic() < deadline:
                            time.sleep(min(0.4 * attempt, max(0.0, deadline - time.monotonic())))
                            continue
                        break
                    except Exception:
                        break

                if gemini_success or key_rejected or quota_limited:
                    break

            if gemini_success or quota_limited:
                break

        # 4. Jika Gemini berhasil -> simpan cache dan kembalikan
        if gemini_success and gemini_text:
            if fp_hash:
                _EXPLANATION_CACHE[fp_hash] = (gemini_text, "gemini")
            self.last_source = "gemini"
            self.last_error = ""
            return gemini_text, "gemini"

        # 5. Jika Gemini gagal (limit 429, timeout, offline) -> Auto-Fallback Lokal
        local_text = self.local_explainer.explain(analysis)
        self.last_source = "local"
        self.last_error = "; ".join(dict.fromkeys(failures)) or "Gemini tidak memberi respons."
        if quota_limited and fp_hash:
            # Avoid repeatedly spending requests on the same result after a
            # quota response. Retry after a short cooldown or a key change.
            _EXPLANATION_CACHE[fp_hash] = (local_text, "local_rate_limited")
            _QUOTA_BACKOFF[fp_hash] = (key_fingerprint, time.monotonic() + 60.0)
        return local_text, "local"

    def _models_to_try(self) -> List[str]:
        """Daftar model resmi yang dicoba berurutan."""
        models: List[str] = []
        for name in (self.model, *self.FALLBACK_MODELS):
            norm = str(name).strip().removeprefix("models/")
            if norm and norm not in models:
                models.append(norm)
        return models

    @staticmethod
    def _build_prompt(compact: Dict[str, Any]) -> str:
        analysis_json = json.dumps(
            compact,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        return f"""
Kamu adalah penjelas hasil analisis dataset untuk aplikasi Dataset Research.

Tugasmu HANYA menjelaskan fakta yang SUDAH dihitung oleh program lokal.

ATURAN MUTLAK:
- Jangan menghitung ulang statistik.
- Jangan membuat angka baru.
- Jangan mengubah angka yang diberikan.
- Jangan menebak fakta yang tidak tersedia.
- Jangan melakukan analisis statistik tambahan.
- Jangan memberikan rekomendasi machine learning sembarangan.
- Gunakan hanya DATA ANALYSIS di bawah.
- Jangan menggunakan emoji sama sekali.

Gunakan Bahasa Indonesia yang formal, ilmiah, dan akademis.

Jelaskan secara runtut dalam 4-6 paragraf:
1. Gambaran umum dan struktur dimensi dataset.
2. Kualitas data, nilai kosong (missing values), dan duplikasi observasi.
3. Sebaran data dan outlier jika ada.
4. Karakteristik kandidat target dan korelasi antar-fitur.
5. Kesimpulan kesiapan dataset untuk tahap riset eksperimental berikutnya.

DATA ANALYSIS:
{analysis_json}
""".strip()

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []
            texts: List[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                t = part.get("text")
                if t:
                    texts.append(str(t))
            if texts:
                return "\n".join(texts)
        return ""

    @classmethod
    def _compact_analysis(cls, analysis: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(analysis, dict):
            return {}
        return {
            "profile": cls._limit_mapping(analysis.get("profile", {}), max_items=40),
            "statistics": cls._limit_collection(analysis.get("statistics", []), max_items=20),
            "missing_values": cls._limit_collection(analysis.get("missing_values", []), max_items=20),
            "duplicates": cls._limit_mapping(analysis.get("duplicates", {}), max_items=20),
            "outliers": cls._limit_collection(analysis.get("outliers", []), max_items=20),
            "correlations": cls._limit_correlations(analysis.get("correlations")),
            "fingerprint": analysis.get("fingerprint"),
        }

    @staticmethod
    def _limit_collection(value: Any, max_items: int) -> Any:
        if isinstance(value, (list, tuple)):
            return list(value)[:max_items]
        return value

    @staticmethod
    def _limit_mapping(value: Any, max_items: int) -> Any:
        if not isinstance(value, dict):
            return value
        return dict(list(value.items())[:max_items])

    @classmethod
    def _limit_correlations(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls._limit_mapping(value, 30)
        if isinstance(value, list):
            return cls._limit_collection(value, 20)
        return value
