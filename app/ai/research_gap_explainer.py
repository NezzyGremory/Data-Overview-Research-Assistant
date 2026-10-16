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

    def explain(self, research: Any) -> tuple[str, str, str]:
        context = self._safe_context(research)
        items = self._safe_gaps(context.get("gaps"))
        if not items:
            return self.explain_local(research)
        if not self.api_keys:
            reason = "API key Gemini tidak ditemukan di pengaturan aplikasi atau environment."
            return self._local_explanation(items, context), "local", reason

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
        deadline = time.monotonic() + 20.0
        transient_statuses = {408, 429, 500, 502, 503, 504}
        # Try at most two configured keys; additional keys rarely help an outage
        # and would make the UI wait too long.
        quota_limited = False
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
                        response_detail = str(response.text or "").lower()
                        quota_error = status == 429 or (
                            status == 403
                            and any(marker in response_detail for marker in ("quota", "resource_exhausted", "rate limit"))
                        )
                        if quota_error:
                            # A rate limit is normally shared by the project;
                            # trying more models and keys only delays local output.
                            quota_limited = True
                            break
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
                if quota_limited:
                    break
            if time.monotonic() >= deadline:
                break
            if quota_limited:
                break

        if quota_limited:
            reason = "Kuota atau batas permintaan Gemini tercapai; hasil ini dijelaskan secara lokal."
        else:
            reason = "Gemini belum berhasil memberi penjelasan (" + ", ".join(dict.fromkeys(failures)) + ")."
        return self._local_explanation(items, context), "local", reason

    def explain_local(self, research: Any, reason: str | None = None) -> tuple[str, str, str]:
        """Build a deterministic explanation without making a network request."""
        context = self._safe_context(research)
        items = self._safe_gaps(context.get("gaps"))
        detail = reason or (
            self._no_candidate_reason(context)
            if not items
            else "Penjelasan lokal digunakan tanpa memanggil Gemini."
        )
        return self._local_explanation(items, context), "local", detail

    @staticmethod
    def _safe_context(research: Any) -> dict[str, Any]:
        """Keep the research facts needed for a useful local synthesis."""
        if not isinstance(research, dict):
            return {"gaps": research}
        looks_like_result = any(
            key in research for key in ("papers", "queries", "search", "landscape", "trend")
        )
        if not looks_like_result:
            return {"gaps": research}
        return {
            "gaps": research.get("gaps"),
            "papers": research.get("papers") if isinstance(research.get("papers"), list) else [],
            "queries": research.get("queries"),
            "keywords": research.get("keywords"),
            "landscape": research.get("landscape"),
            "trend": research.get("trend"),
        }

    @staticmethod
    def _no_candidate_reason(context: dict[str, Any]) -> str:
        gaps = context.get("gaps")
        status = gaps.get("status") if isinstance(gaps, dict) else ""
        if status == "ERROR":
            return "Analisis kandidat gap mengalami kendala; ringkasan lokal menjelaskan status dan data yang tersedia."
        if status == "SKIPPED":
            return "Analisis kandidat gap belum dijalankan; penjelasan lokal memakai jumlah paper dan konteks pencarian yang tersedia."
        if not context.get("papers"):
            return "Belum ada paper yang dapat dianalisis untuk menyimpulkan kandidat gap."
        return "Tidak ada kandidat gap heuristik yang terdeteksi; hasil lokal menjelaskan cakupan dan batas kesimpulannya."

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
    def _local_explanation(
        cls,
        items: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> str:
        context = context if isinstance(context, dict) else {}
        gaps = context.get("gaps")
        gaps = gaps if isinstance(gaps, dict) else {}
        papers = context.get("papers") if isinstance(context.get("papers"), list) else []
        type_names = {
            "METHOD_DISTRIBUTION": "Perbedaan penggunaan metode",
            "UNDERUSED_RECOMMENDED_METHOD": "Metode yang belum banyak terlihat",
            "TEMPORAL": "Paper terbaru masih sedikit",
            "DATASET_CONCENTRATION": "Paper terkonsentrasi pada dataset tertentu",
            "TOPIC_DIVERSITY": "Konsep yang jarang dibahas",
        }
        if items:
            if papers:
                scope = f"Analisis lokal merangkum {len(papers)} paper yang berhasil dikumpulkan. "
            else:
                scope = "Analisis lokal memakai kandidat dan bukti yang tersimpan pada hasil Research Gap. "
            scope += (
                "Kandidat berikut adalah sinyal heuristik untuk ditinjau, bukan bukti bahwa topik tersebut "
                "belum pernah diteliti. Angka hanya menggambarkan literatur yang berhasil ditemukan aplikasi."
            )
            sections = ["## Ringkasan\n\n" + scope]
        else:
            sections = [cls._explain_no_candidates(context, gaps, papers)]
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
        if items:
            sections.append(
                "## Langkah berikutnya\n\n"
                "1. Buka paper sumber dan periksa metode serta keterbatasannya.\n"
                "2. Ulangi pencarian dengan kata kunci dan rentang tahun yang lebih luas.\n"
                "3. Setelah kebaruan terverifikasi, tetapkan pertanyaan, data, metode pembanding, dan ukuran evaluasi."
            )
        return "\n\n---\n\n".join(sections)

    @classmethod
    def _explain_no_candidates(
        cls,
        context: dict[str, Any],
        gaps: dict[str, Any],
        papers: list[Any],
    ) -> str:
        count = len(papers)
        status = str(gaps.get("status") or "").upper()
        summary = gaps.get("summary") if isinstance(gaps.get("summary"), dict) else {}
        queries = context.get("queries")
        if isinstance(queries, dict):
            queries = queries.get("queries") or queries.get("items") or []
        if isinstance(queries, str):
            queries = [queries]
        query_text = ""
        if isinstance(queries, (list, tuple)):
            clean_queries = [str(query).strip() for query in queries if str(query).strip()]
            if clean_queries:
                query_text = " Kata kunci pencarian yang tercatat: " + "; ".join(clean_queries[:3]) + "."

        if status == "SKIPPED":
            headline = "Analisis kandidat gap belum dijalankan"
            explanation = (
                "Hasil Research Gap belum memiliki keluaran analisis, jadi kondisi ini belum berarti bahwa "
                "literatur tidak memiliki celah. Buka atau jalankan tahap analisis Research Gap terlebih dahulu."
            )
        elif status == "ERROR":
            detail = str(gaps.get("error") or "rincian kesalahan tidak tersedia")
            headline = "Analisis kandidat gap belum selesai"
            explanation = (
                f"Pemeriksaan heuristik tidak berhasil diselesaikan ({cls._clean(detail)}). "
                "Karena itu, belum ada dasar untuk menyatakan kandidat gap terdeteksi atau tidak terdeteksi."
            )
        elif count == 0:
            headline = "Belum ada paper untuk dianalisis"
            explanation = (
                "Academic Research belum menyediakan paper yang dapat dibandingkan. Tanpa paper, sistem belum "
                "bisa menyimpulkan pola metode, dataset, tren waktu, atau topik yang kurang dibahas."
            )
        else:
            analyzed_count = summary.get("papers_analyzed", count)
            headline = "Belum ditemukan sinyal gap yang cukup kuat"
            explanation = (
                f"Pemeriksaan lokal membandingkan {analyzed_count} paper yang dianalisis, tetapi aturan heuristik "
                "tidak menandai pola yang cukup kuat sebagai kandidat gap. Pemeriksaan ini melihat perbedaan "
                "metode, metode yang belum banyak digunakan, tahun publikasi, konsentrasi dataset, dan variasi topik. "
                "Hasil kosong bukan bukti bahwa gap tidak ada; sampel paper dan kata kunci pencarian dapat membatasi temuan."
            )

        sections = [f"## {headline}\n\n{explanation}{query_text}"]
        landscape = context.get("landscape")
        landscape = landscape if isinstance(landscape, dict) else {}
        landscape_summary = landscape.get("summary")
        landscape_summary = landscape_summary if isinstance(landscape_summary, dict) else {}
        observations = []
        dominant_method = landscape_summary.get("dominant_method")
        if not dominant_method:
            methods = landscape.get("methods")
            if isinstance(methods, list) and methods and isinstance(methods[0], dict):
                dominant_method = methods[0].get("name")
        if dominant_method:
            method_count = landscape_summary.get("method_count")
            method_detail = f" ({method_count} metode teridentifikasi)" if method_count else ""
            observations.append(f"Metode yang paling sering terlihat: {dominant_method}{method_detail}.")
        latest_year = landscape_summary.get("latest_publication_year")
        if latest_year:
            observations.append(f"Tahun publikasi terbaru pada hasil pencarian: {latest_year}.")

        trend = context.get("trend")
        trend = trend if isinstance(trend, dict) else {}
        direction = str(trend.get("recent_direction") or "").upper()
        direction_names = {
            "INCREASING": "jumlah paper pada data terbaru cenderung meningkat",
            "DECREASING": "jumlah paper pada data terbaru cenderung menurun",
            "STABLE": "jumlah paper pada data terbaru cenderung stabil",
            "INSUFFICIENT_DATA": "jumlah tahun yang tersedia belum cukup untuk membaca tren",
        }
        if direction in direction_names:
            observations.append(f"Tren publikasi: {direction_names[direction]}.")
        if observations:
            sections.append(
                "## Ringkasan hasil yang tersedia\n\n"
                + "\n".join(f"- {observation}" for observation in observations)
                + "\n\nRingkasan ini berasal dari paper yang ditemukan aplikasi dan tetap perlu diperiksa pada sumber aslinya."
            )

        if count:
            years = []
            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                try:
                    year = int(paper.get("year"))
                except (TypeError, ValueError):
                    continue
                if 1900 <= year <= 2100:
                    years.append(year)
            if years:
                sections.append(
                    "## Cakupan data literatur\n\n"
                    f"Rentang tahun pada paper yang ditemukan adalah {min(years)}–{max(years)} "
                    f"({len(set(years))} tahun publikasi berbeda). Angka ini menjelaskan sampel hasil pencarian, "
                    "bukan seluruh publikasi pada bidang tersebut."
                )

        sections.append(
            "## Cara menindaklanjuti\n\n"
            "1. Periksa abstrak dan bagian keterbatasan paper yang paling relevan untuk menemukan masalah yang belum terjawab.\n"
            "2. Ulangi pencarian dengan sinonim, istilah metode, dan rentang tahun yang lebih luas.\n"
            "3. Bandingkan metode, data, populasi, serta ukuran evaluasi sebelum menyatakan kebaruan penelitian."
        )
        return "\n\n---\n\n".join(sections)

