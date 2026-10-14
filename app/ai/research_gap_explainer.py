from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from app.ai.gemini_explainer import get_saved_api_key


class ResearchGapExplainer:
    """Explain literature-derived research gaps using the app's saved Gemini key."""

    def __init__(self) -> None:
        raw_key = get_saved_api_key() or ""
        self.api_keys = [part.strip() for part in raw_key.replace(";", ",").split(",") if part.strip()]
        self.model = os.getenv("GEMINI_MODEL", "").strip()
        try:
            configured_timeout = float(os.getenv("GEMINI_TIMEOUT", "15"))
        except (TypeError, ValueError):
            configured_timeout = 15.0
        self.timeout = min(max(configured_timeout, 5.0), 18.0)

    def explain(self, gaps: Any) -> tuple[str, str, str]:
        items = self._safe_gaps(gaps)
        if not items:
            return (
                "## Belum ada kandidat yang dapat dijelaskan\n\nJalankan Academic Research dan pastikan paper ditemukan.",
                "local",
                "Tidak ada kandidat gap pada hasil Academic Research.",
            )
        if not self.api_keys:
            reason = "API key Gemini tidak ditemukan di pengaturan aplikasi atau environment."
            return self._local_explanation(items), "local", reason

        prompt = self._build_prompt(items)
        default_model = "gemini-3.8-flash"
        fallback_model = "gemini-3.5-flash"
        models = []
        for candidate in (self.model or default_model, default_model, fallback_model):
            candidate = candidate.removeprefix("models/").strip()
            if candidate and candidate not in models:
                models.append(candidate)

        failures = []
        # Bound total wait even when the provider is repeatedly unavailable.
        deadline = time.monotonic() + 38.0
        transient_statuses = {408, 429, 500, 502, 503, 504}
        # Try at most two configured keys; additional keys rarely help an outage
        # and would make the UI wait too long.
        for model in models:
            for key in self.api_keys[:2]:
                for attempt in range(2):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        response = httpx.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                            headers={
                                "x-goog-api-key": key,
                                "Content-Type": "application/json",
                            },
                            json={
                                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                "generationConfig": {
                                    "temperature": 0.2,
                                    "maxOutputTokens": 3072,
                                    **({"thinkingConfig": {"thinkingLevel": "low"}} if model.startswith("gemini-3") else {}),
                                },
                            },
                            timeout=min(self.timeout, remaining),
                        )
                        if response.status_code == 200:
                            text = self._extract_text(response.json())
                            if len(text.strip()) > 80:
                                return text.strip(), "gemini", ""
                            failures.append(f"{model}: respons tanpa teks penjelasan")
                            break

                        status = response.status_code
                        failures.append(f"{model}: HTTP {status}")
                        if status in transient_statuses and attempt == 0:
                            # Retry-After is advisory and is capped to preserve responsiveness.
                            try:
                                delay = min(max(float(response.headers.get("Retry-After", "0.6")), 0.2), 2.0)
                            except (TypeError, ValueError):
                                delay = 0.6
                            if deadline - time.monotonic() > delay:
                                time.sleep(delay)
                                continue
                        # Invalid model: next model. Auth/quota/provider errors:
                        # try another configured key/model where available.
                        break
                    except httpx.TimeoutException:
                        failures.append(f"{model}: koneksi timeout ({self.timeout:g} detik)")
                        if attempt == 0 and deadline - time.monotonic() > 0.5:
                            time.sleep(0.5)
                            continue
                        break
                    except httpx.RequestError:
                        failures.append(f"{model}: tidak dapat dijangkau")
                        if attempt == 0 and deadline - time.monotonic() > 0.5:
                            time.sleep(0.5)
                            continue
                        break
                    except Exception:
                        failures.append("respons Gemini tidak dapat dibaca")
                        break
                if time.monotonic() >= deadline:
                    break
            if time.monotonic() >= deadline:
                break

        reason = "Gemini belum berhasil memberi penjelasan (" + ", ".join(dict.fromkeys(failures)) + ")."
        return self._local_explanation(items), "local", reason

    @staticmethod
    def _safe_gaps(gaps: Any) -> list[dict[str, Any]]:
        if isinstance(gaps, dict):
            items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items") or []
        elif isinstance(gaps, list):
            items = gaps
        else:
            items = []
        safe = []
        for item in items[:10]:
            if isinstance(item, dict):
                safe.append({
                    key: item.get(key)
                    for key in ("type", "title", "description", "evidence")
                    if item.get(key) not in (None, "", [], {})
                })
            elif item:
                safe.append({"title": str(item)[:500]})
        return safe

    @staticmethod
    def _build_prompt(items: list[dict[str, Any]]) -> str:
        evidence = json.dumps(items, ensure_ascii=False, default=str)
        return f"""Kamu adalah pembimbing penelitian yang menjelaskan hasil kepada mahasiswa.
Tulis dalam Bahasa Indonesia yang mudah dipahami, ringkas, dan gunakan Markdown.
Jangan menampilkan JSON mentah. Gunakan hanya kandidat dan bukti yang diberikan.
Hitungan hanya berlaku untuk paper yang ditemukan aplikasi. Jangan mengklaim gap sebagai fakta pasti atau mengarang angka, sumber, maupun kebaruan.

Awali dengan ringkasan dua kalimat yang menyebutkan cakupan dan keterbatasan pencarian.
Untuk setiap kandidat, pertahankan judul dan tulis tiga bagian singkat:
**Makna** - jelaskan pola dengan bahasa sederhana.
**Bukti dan batasan** - pertahankan angka yang tersedia dan jelaskan apa yang belum dapat disimpulkan.
**Arah penelitian** - beri satu usulan yang dapat diuji dan contoh pertanyaan penelitian; sebutkan pembanding atau evaluasi bila relevan.
Batasi tiap bagian maksimal dua kalimat. Jangan menghilangkan kandidat yang diberikan.
Tutup dengan maksimal tiga langkah praktis untuk memeriksa paper terbaru dan memvalidasi kebaruan.

KANDIDAT GAP DAN BUKTI:
{evidence}"""

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        pieces = []
        for candidate in payload.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if isinstance(part, dict) and part.get("text"):
                    pieces.append(str(part["text"]))
        return "\n".join(pieces)

    @staticmethod
    def _clean(value: Any) -> str:
        text = str(value)
        text = re.sub(r"\s+", " ", text).strip()
        for char in "\\`*_{}[]<>|":
            text = text.replace(char, "\\" + char)
        return text or "(tidak tersedia)"

    @classmethod
    def _local_explanation(cls, items: list[dict[str, Any]]) -> str:
        type_names = {
            "METHOD_DISTRIBUTION": "Perbedaan penggunaan metode",
            "UNDERUSED_RECOMMENDED_METHOD": "Metode yang belum banyak terlihat",
            "TEMPORAL": "Paper terbaru masih sedikit",
            "DATASET_CONCENTRATION": "Paper terkonsentrasi pada dataset tertentu",
            "TOPIC_DIVERSITY": "Konsep yang jarang dibahas",
        }
        sections = [
            "## Ringkasan\n\nBerikut penjelasan awal dari paper yang ditemukan. Ini adalah petunjuk untuk diteliti, bukan bukti bahwa topik tersebut belum pernah diteliti. Angka hanya menggambarkan hasil pencarian aplikasi.",
        ]
        for index, item in enumerate(items, 1):
            kind = str(item.get("type") or "")
            title = type_names.get(kind) or cls._clean(item.get("title") or "Kandidat celah riset")
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}

            if kind == "METHOD_DISTRIBUTION":
                dominant = cls._clean(evidence.get("dominant_method", "metode yang paling sering muncul"))
                less_used = cls._clean(evidence.get("less_common_method", "metode yang lebih jarang muncul"))
                dominant_count = evidence.get("dominant_count", "-")
                less_count = evidence.get("less_common_count", "-")
                meaning = f"Dalam kumpulan paper yang ditemukan, {dominant} lebih sering dibahas daripada {less_used}. Ini menunjukkan perbedaan perhatian dalam hasil pencarian, bukan perbedaan kualitas kedua metode."
                proof = f"{dominant} tercatat pada {dominant_count} paper dan {less_used} pada {less_count} paper. Sebagian paper mungkin tidak ditemukan oleh pencarian, jadi angka ini belum mewakili seluruh penelitian yang ada."
                recommendation = f"Bandingkan {less_used} dengan {dominant} untuk masalah yang sama. Gunakan pembagian data, praproses, dan ukuran evaluasi yang sama agar perbandingannya adil."
                question = f"Apakah {less_used} memberikan hasil yang sebanding dengan {dominant} pada masalah yang diteliti?"
            elif kind == "UNDERUSED_RECOMMENDED_METHOD":
                method = cls._clean(evidence.get("method", "metode yang direkomendasikan"))
                meaning = f"Karakteristik tugas mengarah pada {method}, tetapi metode ini tidak muncul pada paper yang ditemukan."
                proof = "Pencarian terbatas dapat saja melewatkan penelitian yang menggunakan metode tersebut. Karena itu, temuan ini belum membuktikan bahwa metodenya belum pernah dipakai."
                recommendation = f"Cari penelitian tambahan tentang {method}, lalu bandingkan dengan pendekatan yang umum dipakai pada tugas yang sama menggunakan evaluasi yang seragam."
                question = f"Dalam kondisi apa {method} dapat memberi hasil yang lebih baik atau lebih stabil daripada metode pembanding?"
            elif kind == "TEMPORAL":
                recent = evidence.get("recent_papers", "-")
                total = evidence.get("total_papers", "-")
                year = evidence.get("latest_year", "-")
                ratio = evidence.get("recent_ratio")
                ratio_text = f" atau {ratio}% dari paper bertahun" if isinstance(ratio, (int, float)) else ""
                meaning = "Literatur terbaru pada hasil pencarian ini terlihat lebih sedikit dibandingkan publikasi sebelumnya."
                proof = f"Ditemukan {recent} dari {total} paper bertahun pada periode terbaru hingga {year}{ratio_text}. Hal ini juga bisa disebabkan pencarian belum mencakup semua publikasi baru."
                recommendation = "Perbarui pencarian untuk tahun-tahun terbaru dan bandingkan perubahan metode, sumber data, serta cara evaluasi yang digunakan."
                question = "Bagaimana metode dan cara evaluasi pada topik ini berubah dalam publikasi terbaru?"
            elif kind == "DATASET_CONCENTRATION":
                dataset = cls._clean(evidence.get("dominant_dataset", "dataset tertentu"))
                count = evidence.get("paper_count", "-")
                meaning = f"Banyak paper yang ditemukan menggunakan atau membahas dataset {dataset}."
                proof = f"Dataset {dataset} disebut pada {count} paper dalam hasil pencarian. Ini menunjukkan konsentrasi pada kumpulan paper tersebut, bukan bukti bahwa dataset lain belum pernah digunakan."
                recommendation = "Uji apakah temuan tetap berlaku pada dataset lain yang relevan. Bandingkan hasil dengan prosedur praproses dan ukuran evaluasi yang sama."
                question = f"Apakah metode yang dikaji tetap bekerja baik pada dataset selain {dataset}?"
            elif kind == "TOPIC_DIVERSITY":
                concepts = evidence.get("rare_concepts") or []
                concept_text = ", ".join(cls._clean(value) for value in concepts[:8]) if isinstance(concepts, list) else cls._clean(concepts)
                meaning = "Beberapa konsep hanya muncul sedikit pada abstrak yang dianalisis, sehingga mungkin layak diperiksa lebih lanjut."
                proof = f"Konsep yang tercatat sekali: {concept_text or 'rincian konsep tidak tersedia'}. Kemunculan yang jarang saja belum cukup untuk menyatakan suatu topik baru."
                recommendation = "Telusuri konsep tersebut pada sumber akademik lain, baca paper yang paling dekat, lalu persempit satu masalah agar dapat diuji."
                question = "Masalah spesifik apa yang belum terjawab oleh penelitian terkait konsep tersebut?"
            else:
                meaning = cls._clean(item.get("description") or "Pola ini perlu diperiksa lebih lanjut pada literatur yang lebih luas.")
                proof = "Bukti yang tersedia belum cukup untuk memastikan adanya kekosongan penelitian."
                recommendation = "Periksa paper yang mendasari temuan dan cari penelitian terkait yang mungkin terlewat sebelum menentukan arah studi."
                question = "Pertanyaan apa yang belum terjawab setelah membandingkan penelitian paling relevan?"

            sections.append(
                f"## {index}. {title}\n\n"
                f"**Maksudnya**\n\n{meaning}\n\n"
                f"**Bukti dan batasannya**\n\n{proof}\n\n"
                f"**Saran penelitian berikutnya**\n\n{recommendation}\n\n"
                f"**Contoh pertanyaan penelitian**\n\n{question}"
            )
        sections.append(
            "## Langkah berikutnya\n\n"
            "1. Buka paper sumber dan periksa metode serta keterbatasannya.\n"
            "2. Ulangi pencarian dengan kata kunci dan rentang tahun yang lebih luas.\n"
            "3. Setelah kebaruan terverifikasi, tetapkan pertanyaan, data, metode pembanding, dan ukuran evaluasi."
        )
        return "\n\n---\n\n".join(sections)

