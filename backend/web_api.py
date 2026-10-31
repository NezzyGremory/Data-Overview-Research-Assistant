from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import re
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response

from app.analyzer import (
    CorrelationAnalyzer,
    DataQualityDiagnoser,
    DatasetFingerprint,
    DatasetLoader,
    DatasetProfiler,
    DatasetStatistics,
    DuplicateAnalyzer,
    MissingValueAnalyzer,
    OutlierAnalyzer,
)
from app.analyzer.cleaner import drop_missing
from app.core.exceptions import DatasetLoadError
from app.ml.method_info import get_all_methods
from app.ml.method_recommender import MethodRecommender
from app.ml.task_detector import MLTaskDetector
from app.research.gap_analyzer import ResearchGapAnalyzer
from app.research.landscape import ResearchLandscapeAnalyzer
from app.research.ranking import PaperRanker
from app.research.search import AcademicSearchEngine
from app.research.trend_analyzer import ResearchTrendAnalyzer

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ANALYSIS_COLUMNS = 120
MAX_DISTRIBUTION_COLUMNS = 60
MAX_CORRELATION_COLUMNS = 24
PREVIEW_ROWS = 50
SUPPORTED_SUFFIXES = {".csv"}
PROFILE_REQUESTS_PER_MINUTE = int(os.getenv("PROFILE_REQUESTS_PER_MINUTE", "10"))
RESEARCH_REQUESTS_PER_MINUTE = int(os.getenv("RESEARCH_REQUESTS_PER_MINUTE", "6"))
RESEARCH_CACHE_TTL = 10 * 60
RESEARCH_CACHE_ENTRIES = 32
WEB_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "WEB_ORIGINS",
        "http://localhost:5199,http://127.0.0.1:5199,"
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="DORA Web API", docs_url=None, redoc_url=None)

_analysis_slot = threading.BoundedSemaphore(1)
_rate_lock = threading.Lock()
_rate_requests: dict[str, deque[float]] = defaultdict(deque)
_research_cache: dict[str, tuple[float, dict]] = {}
_research_cache_lock = threading.Lock()

# Limits are process-local. Move them to a shared store before scaling to
# more than one API instance.


@app.middleware("http")
async def reject_oversized_upload(request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/datasets/"):
        try:
            request_size = int(request.headers.get("content-length", "0"))
        except ValueError:
            request_size = 0
        if request_size > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"detail": "Ukuran dataset terlalu besar. Maksimal 10 MB."},
                status_code=413,
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Dora-Report", "Content-Disposition"],
)


def client_key(request: Request) -> str:
    """Identify the caller for rate limiting, preferring the proxy header."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "anonymous"


def limit_requests(subject: str, cap: int | None = None) -> None:
    limit = cap or PROFILE_REQUESTS_PER_MINUTE
    now = time.monotonic()
    with _rate_lock:
        stale = [
            caller
            for caller, hits in _rate_requests.items()
            if not hits or now - hits[-1] >= 60
        ]
        for caller in stale:
            _rate_requests.pop(caller, None)
        hits = _rate_requests[subject]
        while hits and now - hits[0] >= 60:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Batas permintaan sementara tercapai. Coba lagi dalam satu menit.",
            )
        hits.append(now)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ml/catalog")
def ml_catalog() -> dict:
    return {
        "tasks": [
            {"id": task_id, "label": label}
            for task_id, label in MLTaskDetector.TASK_LABELS.items()
        ],
        "methods": get_all_methods(),
    }


@app.post("/api/datasets/profile")
async def profile_dataset(request: Request) -> dict:
    limit_requests(client_key(request))
    temporary_path, filename = await _receive_upload(request)
    try:
        return await run_in_threadpool(_build_profile, temporary_path, filename)
    except DatasetLoadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        temporary_path.unlink(missing_ok=True)
        _analysis_slot.release()


@app.post("/api/datasets/analyze")
async def analyze_dataset(request: Request) -> dict:
    limit_requests(client_key(request))
    temporary_path, filename = await _receive_upload(request)
    try:
        return await run_in_threadpool(_build_analysis, temporary_path, filename)
    except DatasetLoadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        temporary_path.unlink(missing_ok=True)
        _analysis_slot.release()


@app.post("/api/datasets/clean")
async def clean_dataset(request: Request, strategy: str = "impute", columns: str = "") -> dict:
    limit_requests(client_key(request))
    if strategy not in {"impute", "drop_rows"}:
        raise HTTPException(status_code=400, detail="Metode pembersihan tidak dikenal.")
    selected = [name for name in columns.split(",") if name.strip()] or None
    temporary_path, filename = await _receive_upload(request)
    try:
        return await run_in_threadpool(_run_clean, temporary_path, filename, strategy, selected)
    finally:
        temporary_path.unlink(missing_ok=True)
        _analysis_slot.release()


# ---------------------------------------------------------------------------
# Dataset Discovery & Search
# ---------------------------------------------------------------------------


def _search_datasets_sync(query: str, sources: set[str], limit: int = 15) -> list[dict]:
    results: list[dict] = []

    def fetch_hf():
        items = []
        try:
            r = httpx.get(
                "https://huggingface.co/api/datasets",
                params={"search": query, "limit": limit, "sort": "downloads"},
                timeout=7.0,
            )
            if r.status_code == 200:
                for d in r.json():
                    cid = d.get("id") or ""
                    card = d.get("cardData") or {}
                    tags = [
                        *(card.get("task_categories") or [])[:3],
                        *[t for t in d.get("tags", []) if isinstance(t, str) and ":" not in t][:4],
                    ][:6]
                    items.append({
                        "id": f"hf-{cid}",
                        "source": "huggingface",
                        "name": card.get("pretty_name") or cid,
                        "description": card.get("description") or d.get("description") or "",
                        "tags": tags,
                        "downloads": d.get("downloads"),
                        "likes": d.get("likes"),
                        "url": f"https://huggingface.co/datasets/{cid}",
                    })
        except Exception:
            pass
        return items

    def fetch_kg():
        items = []
        try:
            r = httpx.get(
                "https://www.kaggle.com/api/v1/datasets/list",
                params={"search": query, "page": 1, "sortBy": "relevance"},
                timeout=7.0,
            )
            if r.status_code == 200:
                raw = r.json()
                if isinstance(raw, list):
                    for d in raw[:limit]:
                        if not isinstance(d, dict):
                            continue
                        ref = d.get("ref") or str(d.get("id") or "")
                        tags = [
                            t.get("name") if isinstance(t, dict) else str(t)
                            for t in d.get("tags") or []
                        ][:5]
                        items.append({
                            "id": f"kg-{ref}",
                            "source": "kaggle",
                            "name": d.get("title") or ref,
                            "description": d.get("subtitle") or d.get("description") or "",
                            "tags": tags,
                            "downloads": d.get("downloadCount"),
                            "likes": d.get("voteCount"),
                            "url": d.get("url") or f"https://www.kaggle.com/datasets/{ref}",
                        })
        except Exception:
            pass
        return items

    def fetch_zenodo():
        items = []
        try:
            r = httpx.get(
                "https://zenodo.org/api/records",
                params={"q": query, "type": "dataset", "size": min(limit, 10), "sort": "mostrecent"},
                timeout=7.0,
            )
            if r.status_code == 200:
                hits = r.json().get("hits", {}).get("hits", [])
                for d in hits:
                    meta = d.get("metadata") or {}
                    desc = meta.get("description") or ""
                    clean_desc = re.sub(r"<[^>]+>", "", desc)[:160]
                    tags = [*(meta.get("keywords") or [])[:4]][:5]
                    rid = d.get("id")
                    items.append({
                        "id": f"zen-{rid}",
                        "source": "zenodo",
                        "name": meta.get("title") or d.get("title") or f"Zenodo #{rid}",
                        "description": clean_desc,
                        "tags": tags,
                        "downloads": d.get("stats", {}).get("downloads"),
                        "likes": None,
                        "url": d.get("links", {}).get("html") or f"https://zenodo.org/record/{rid}",
                    })
        except Exception:
            pass
        return items

    def fetch_openml():
        items = []
        try:
            first_word = re.sub(r"[^a-zA-Z0-9_]", "", query.split()[0]) if query.split() else query
            if first_word:
                r = httpx.get(
                    f"https://www.openml.org/api/v1/json/data/list/data_name/{first_word}/limit/{min(limit, 10)}",
                    timeout=7.0,
                )
                if r.status_code == 200:
                    raw = r.json().get("data", {}).get("dataset", [])
                    for d in raw:
                        did = d.get("did")
                        items.append({
                            "id": f"oml-{did}",
                            "source": "openml",
                            "name": d.get("name") or f"OpenML #{did}",
                            "description": f"{d.get('format', 'Data')} · status: {d.get('status', 'active')}",
                            "tags": [d["format"]] if d.get("format") else [],
                            "downloads": None,
                            "likes": None,
                            "url": f"https://www.openml.org/d/{did}",
                        })
        except Exception:
            pass
        return items

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        if "huggingface" in sources:
            tasks.append(executor.submit(fetch_hf))
        if "kaggle" in sources:
            tasks.append(executor.submit(fetch_kg))
        if "zenodo" in sources:
            tasks.append(executor.submit(fetch_zenodo))
        if "openml" in sources:
            tasks.append(executor.submit(fetch_openml))
        for t in tasks:
            results.extend(t.result())

    results.sort(key=lambda x: (x["downloads"] is not None, x["downloads"] or 0), reverse=True)
    return results


@app.get("/api/datasets/search")
async def datasets_search(
    q: str = "",
    sources: str = "huggingface,kaggle,openml,zenodo",
    limit: int = 20,
) -> dict:
    cleaned = q.strip()
    if not cleaned:
        return {"query": "", "count": 0, "results": []}
    limit = max(1, min(limit, 40))
    source_set = {s.strip().lower() for s in sources.split(",") if s.strip()}
    results = await run_in_threadpool(_search_datasets_sync, cleaned, source_set, limit)
    return {"query": cleaned, "count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Academic research
# ---------------------------------------------------------------------------


@app.post("/api/research/search")
async def research_search(request: Request) -> dict:
    limit_requests(f"{client_key(request)}:research", cap=RESEARCH_REQUESTS_PER_MINUTE)
    try:
        payload = await request.json()
    except Exception as error:
        raise HTTPException(status_code=400, detail="Body permintaan tidak valid.") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body permintaan tidak valid.")

    query = str(payload.get("query", "")).strip()
    if len(query) < 3:
        raise HTTPException(
            status_code=400,
            detail="Tulis minimal 3 karakter kata kunci pencarian.",
        )
    if len(query) > 300:
        raise HTTPException(status_code=400, detail="Kata kunci terlalu panjang (maksimal 300 karakter).")

    try:
        limit = int(payload.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(5, min(limit, 40))

    fingerprint = payload.get("fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, dict):
        fingerprint = None
    ml_result = payload.get("ml_result") if isinstance(payload.get("ml_result"), dict) else None

    cache_key = json.dumps(
        {"q": query.lower(), "limit": limit, "fp": bool(fingerprint), "ml": bool(ml_result)},
        sort_keys=True,
    )
    now = time.monotonic()
    with _research_cache_lock:
        cached = _research_cache.get(cache_key)
        if cached and now - cached[0] < RESEARCH_CACHE_TTL:
            return {**cached[1], "cached": True}

    result = await run_in_threadpool(_run_research, query, limit, fingerprint, ml_result)
    with _research_cache_lock:
        if len(_research_cache) >= RESEARCH_CACHE_ENTRIES:
            oldest = min(_research_cache.items(), key=lambda item: item[1][0])[0]
            _research_cache.pop(oldest, None)
        _research_cache[cache_key] = (now, result)
    return {**result, "cached": False}


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------


async def _receive_upload(request: Request) -> tuple[Path, str]:
    """Stream the request body to a temp file and release the analysis slot."""
    filename = request.query_params.get("filename", "dataset")
    suffix = Path(filename.replace("\\", "/")).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="DORA versi web hanya membaca file CSV (.csv).",
        )
    if not _analysis_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Analisis lain sedang berjalan. Coba lagi sebentar.",
        )

    temporary_path: Path | None = None
    try:
        total_bytes = 0
        with tempfile.NamedTemporaryFile(
            prefix="dora-web-", suffix=suffix, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            async for chunk in request.stream():
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Ukuran dataset terlalu besar. Maksimal 10 MB.",
                    )
                temporary_file.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="File dataset kosong.")
        return temporary_path, filename
    except HTTPException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        _analysis_slot.release()
        raise
    except Exception as error:  # pragma: no cover - defensive
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        _analysis_slot.release()
        raise HTTPException(status_code=400, detail="File tidak dapat dibaca.") from error


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _load_frame(path: Path):
    return DatasetLoader().load(str(path))


def _build_profile(path: Path, filename: str) -> dict:
    dataframe = _load_frame(path)
    profile = DatasetProfiler().profile(dataframe)
    return {
        "filename": _clean_name(filename),
        "profile": profile,
        "column_names": [str(column) for column in dataframe.columns],
        "preview": _preview_records(dataframe),
        "ml_intelligence": _ml_intelligence(dataframe, profile),
    }


def _build_analysis(path: Path, filename: str) -> dict:
    dataframe = _load_frame(path)
    profile = DatasetProfiler().profile(dataframe)
    total_columns = int(len(dataframe.columns))
    scoped = dataframe.iloc[:, :MAX_ANALYSIS_COLUMNS]

    statistics = DatasetStatistics().analyze(scoped)
    missing_values = MissingValueAnalyzer().analyze(scoped)
    outliers = OutlierAnalyzer().analyze(scoped)
    duplicates = DuplicateAnalyzer().analyze(dataframe)
    correlations = _correlation_payload(scoped)
    fingerprint = DatasetFingerprint().generate(scoped)
    quality = [
        asdict(item)
        for item in DataQualityDiagnoser().diagnose(
            dataframe,
            profile=profile,
            missing_values=missing_values,
            duplicates=duplicates,
            outliers=outliers,
            fingerprint=fingerprint,
        )
    ]

    return {
        "filename": _clean_name(filename),
        "profile": profile,
        "column_names": [str(column) for column in dataframe.columns],
        "preview": _preview_records(dataframe),
        "statistics": _json_safe(statistics),
        "missing_values": _json_safe(missing_values),
        "outliers": _json_safe(outliers),
        "duplicates": _json_safe(duplicates),
        "correlations": correlations,
        "fingerprint": _json_safe(
            {
                "hash": fingerprint.get("fingerprint", ""),
                "representation": {
                    "characteristics": fingerprint.get("representation", {}).get(
                        "characteristics", {}
                    ),
                    "target_candidates": fingerprint.get("representation", {}).get(
                        "target_candidates", []
                    ),
                    "keywords": fingerprint.get("representation", {}).get("keywords", []),
                    "categories": fingerprint.get("representation", {}).get(
                        "categories", {}
                    ),
                    "column_signature": fingerprint.get("representation", {}).get(
                        "column_signature", []
                    ),
                },
            }
        ),
        "quality": _json_safe(quality),
        "distributions": _distributions(scoped),
        "duplicates_detail": _duplicate_sample(dataframe),
        "ml_intelligence": _ml_intelligence(dataframe, profile),
        "limits": {
            "columns_total": total_columns,
            "columns_analyzed": int(len(scoped.columns)),
            "rows_total": int(len(dataframe)),
            "truncated": total_columns > MAX_ANALYSIS_COLUMNS,
        },
    }


def _ml_intelligence(dataframe, profile: dict) -> dict:
    task_detection = MLTaskDetector().detect({}, dataframe=dataframe)
    recommendation = MethodRecommender().recommend(
        {
            "primary_task": task_detection.get("primary_task") or {},
            "dataset": {
                "rows": profile["rows"],
                "columns": profile["columns"],
                "numeric_features": profile["numeric_columns"],
            },
        }
    )
    return {
        "status": "ESTIMATED",
        "task_detection": task_detection,
        "recommendation": recommendation,
        "evaluation_status": "SKIPPED",
    }


def _correlation_payload(scoped) -> dict:
    matrix = CorrelationAnalyzer().analyze(scoped)
    if matrix.empty:
        return {"columns": [], "values": []}
    limited = matrix.iloc[:MAX_CORRELATION_COLUMNS, :MAX_CORRELATION_COLUMNS]
    values = [
        [None if value != value else round(float(value), 3) for value in row]
        for row in limited.to_numpy()
    ]
    return {"columns": [str(column) for column in limited.columns], "values": values}


def _preview_records(dataframe) -> list[dict]:
    head = dataframe.head(PREVIEW_ROWS)
    return json.loads(head.to_json(orient="records", date_format="iso"))


def _distributions(scoped) -> list[dict]:
    """Bins for numeric columns and top categories for everything else."""
    results: list[dict] = []
    sample_cap = 20_000
    for column in list(scoped.columns)[:MAX_DISTRIBUTION_COLUMNS]:
        series = scoped[column]
        values = series.dropna()
        if values.empty:
            continue
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        try:
            if numeric:
                numbers = values.astype(float)
                low = float(numbers.min())
                high = float(numbers.max())
                if low == high:
                    bins = [{"label": _short_number(low), "count": int(numbers.size)}]
                else:
                    counts, edges = np.histogram(numbers.to_numpy(), bins=10)
                    bins = [
                        {
                            "label": f"{_short_number(edges[index])}–{_short_number(edges[index + 1])}",
                            "count": int(counts[index]),
                        }
                        for index in range(len(counts))
                    ]
                kind = "numeric"
            else:
                working = series
                if len(working) > sample_cap:
                    working = working.sample(sample_cap, random_state=0)
                counts = working.astype("string").fillna("(kosong)").value_counts().head(8)
                bins = [
                    {"label": str(label)[:48], "count": int(count)}
                    for label, count in counts.items()
                ]
                kind = "categorical"
        except Exception:
            continue
        results.append(
            {
                "column": str(column),
                "kind": kind,
                "bins": bins,
                "missing": int(series.isna().sum()),
            }
        )
    return results


def _short_number(value: float) -> str:
    if value == int(value) and abs(value) < 1_000_000:
        return str(int(value))
    return f"{value:.3g}"


def _duplicate_sample(dataframe) -> dict:
    duplicated = dataframe.duplicated(keep=False)
    if not bool(duplicated.any()):
        return {"count": 0, "rows": []}
    sample = dataframe[duplicated].head(6)
    return {
        "count": int(dataframe.duplicated().sum()),
        "rows": _json_safe(json.loads(sample.to_json(orient="records", date_format="iso"))),
    }


def _clean_name(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name[:120]


def _json_safe(value):
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# Research pipeline
# ---------------------------------------------------------------------------


def _run_research(
    query: str,
    limit: int,
    fingerprint: dict | None,
    ml_result: dict | None,
) -> dict:
    result = AcademicSearchEngine().search(query, limit=limit)
    papers = result.get("papers") or []
    ranked = False
    if papers and fingerprint:
        try:
            papers = PaperRanker().rank(papers, fingerprint, ml_result=ml_result)
            ranked = True
        except Exception:
            papers = result.get("papers") or []

    landscape = ResearchLandscapeAnalyzer().analyze(papers, ml_result=ml_result)
    trend = ResearchTrendAnalyzer().analyze(papers, landscape=landscape)
    gaps = ResearchGapAnalyzer().analyze(papers, ml_result=ml_result, landscape=landscape)

    return _json_safe(
        {
            "status": result.get("status", "SUCCESS"),
            "query": query,
            "papers": papers,
            "result_count": len(papers),
            "candidate_count": int(result.get("candidate_count") or len(papers)),
            "sources": result.get("sources") or [],
            "provider_status": result.get("provider_status") or {},
            "message": result.get("message") or "",
            "ranked": ranked,
            "landscape": landscape,
            "trend": trend,
            "gaps": gaps,
        }
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _run_clean(
    path: Path,
    filename: str,
    strategy: str,
    selected: list[str] | None,
) -> Response:
    try:
        dataframe = _load_frame(path)
        if selected:
            unknown = [name for name in selected if name not in dataframe.columns]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Kolom tidak ditemukan: {', '.join(unknown[:5])}.",
                )
        result = drop_missing(dataframe, columns=selected, strategy=strategy)
    except DatasetLoadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=400,
            detail="Dataset tidak dapat dibersihkan. Periksa format dan isi file.",
        ) from error

    cleaned = result.cleaned_df
    detail = result.report or {}
    report = {
        "strategy": strategy,
        "rows_before": int(result.rows_before),
        "rows_after": int(result.rows_after),
        "rows_dropped": int(result.rows_dropped),
        "percent_dropped": float(result.percent_dropped),
        "columns_after": int(cleaned.shape[1]),
        "missing_before": int(detail.get("missing_before", 0)),
        "missing_after": int(cleaned.isna().sum().sum()),
        "imputed_columns": [str(name) for name in detail.get("columns_imputed", [])],
        "cells_imputed": int(detail.get("cells_imputed", 0)),
        "dropped_empty_columns": [
            str(name) for name in detail.get("columns_dropped_names", [])
        ],
        "warnings": [str(item) for item in result.warnings[:6]],
    }
    payload = base64.b64encode(
        json.dumps(_json_safe(report)).encode("utf-8")
    ).decode("ascii")
    csv_bytes = cleaned.to_csv(index=False).encode("utf-8-sig")
    download_name = f"{Path(_clean_name(filename)).stem or 'dataset'}_bersih.csv"

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "X-Dora-Report": payload,
            "Cache-Control": "no-store",
        },
    )
