from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx

from app.research.crossref import CrossrefClient
from app.research.deduplication import PaperDeduplicator
from app.research.europe_pmc import EuropePMCClient
from app.research.openalex import OpenAlexClient
from app.research.openaire import OpenAIREClient
from app.research.paper import Paper
from app.research.scopus import ScopusClient
from app.research.semantic_scholar import SemanticScholarClient


class AcademicSearchEngine:
    """Mesin pencarian literatur akademik dengan multi-provider dan deduplikasi.

    Provider yang diaudit dan dipertahankan:
    - OpenAlex: Indeks bibliografi global, bebas key (mendukung email polite pool).
    - Crossref: Registri resmi DOI dan metadata publikasi penerbit.
    - Europe PMC: Repositori literatur ilmu hayati & biomedis publik (EBI/PMC).
    - OpenAIRE: Grafik penelitian publik terintegrasi Eropa.
    - Semantic Scholar: Graf literatur akademik (API key opsional/rate-limited).
    - Scopus: Opsional, hanya aktif jika ELSEVIER_API_KEY dikonfigurasi.
    """

    def __init__(
        self,
        email: Optional[str] = None,
        openalex: Optional[OpenAlexClient] = None,
        crossref: Optional[CrossrefClient] = None,
        semantic_scholar: Optional[SemanticScholarClient] = None,
        europe_pmc: Optional[EuropePMCClient] = None,
        scopus: Optional[ScopusClient] = None,
        openaire: Optional[OpenAIREClient] = None,
        deduplicator: Optional[PaperDeduplicator] = None,
        **kwargs,
    ):
        self.openalex = openalex or OpenAlexClient(email=email, timeout=12.0)
        self.crossref = crossref or CrossrefClient(email=email, timeout=12.0)
        self.semantic_scholar = semantic_scholar or SemanticScholarClient(
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"), timeout=12.0
        )
        self.europe_pmc = europe_pmc or EuropePMCClient(timeout=12.0)
        self.scopus = scopus or ScopusClient(timeout=12.0)
        self.openaire = openaire or OpenAIREClient(timeout=12.0)
        self.deduplicator = deduplicator or PaperDeduplicator()

    def search(
        self,
        query: str,
        limit: int = 20,
        use_crossref: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Menjalankan pencarian paralel ke provider yang aktif dan menggabungkan hasilnya."""
        if not query or not query.strip():
            return {
                "status": "ERROR",
                "query": query,
                "papers": [],
                "result_count": 0,
                "sources": [],
                "provider_status": {},
                "message": "Query tidak boleh kosong.",
            }

        limit = max(1, min(limit, 100))

        all_papers: List[Paper] = []
        candidate_source_counts: Dict[str, int] = {}
        provider_calls = {
            "OpenAlex": lambda: self.openalex.search(query=query, per_page=limit),
        }
        if use_crossref:
            provider_calls["Crossref"] = lambda: self.crossref.search(query=query, rows=limit)
        provider_calls["Europe PMC"] = lambda: self.europe_pmc.search(query=query, limit=limit)
        provider_calls["OpenAIRE"] = lambda: self.openaire.search(query=query, limit=limit)
        provider_calls["Semantic Scholar"] = lambda: self.semantic_scholar.search(query=query, limit=limit)

        provider_status: Dict[str, str] = {name: "ENABLED" for name in provider_calls}
        if self.scopus.enabled:
            provider_calls["Scopus"] = lambda: self.scopus.search(query=query, limit=limit)
            provider_status["Scopus"] = "ENABLED"
        else:
            provider_status["Scopus"] = "NOT_CONFIGURED"

        if not use_crossref:
            provider_status["Crossref"] = "DISABLED"

        # Menjalankan provider secara paralel di thread pool agar tidak memblokir UI
        with ThreadPoolExecutor(max_workers=max(1, len(provider_calls))) as executor:
            futures = {
                executor.submit(search_fn): name
                for name, search_fn in provider_calls.items()
            }
            for future, provider in futures.items():
                try:
                    results = future.result()
                    results = results if isinstance(results, list) else []
                    all_papers.extend(results)
                    candidate_source_counts[provider] = len(results)
                    provider_status[provider] = "ENABLED"
                except httpx.HTTPStatusError as http_err:
                    candidate_source_counts[provider] = 0
                    if http_err.response.status_code == 429:
                        provider_status[provider] = "RATE_LIMITED"
                    elif http_err.response.status_code in (401, 403):
                        provider_status[provider] = "NOT_CONFIGURED"
                    else:
                        provider_status[provider] = "ERROR"
                except Exception:
                    candidate_source_counts[provider] = 0
                    provider_status[provider] = "ERROR"

        # Deduplikasi berurutan: DOI -> external_id -> normalized title
        unique_papers = self.deduplicator.deduplicate(all_papers)
        unique_papers = unique_papers[:limit]

        sources = sorted(
            {
                source.strip()
                for paper in unique_papers
                for source in (paper.source or "").split(",")
                if source.strip()
            }
        )

        active_sources = [
            source
            for source, status in provider_status.items()
            if status == "ENABLED"
        ]

        if not unique_papers:
            status = "PARTIAL" if any(
                value in ("ERROR", "RATE_LIMITED") for value in provider_status.values()
            ) else "SUCCESS"
        else:
            status = "SUCCESS"

        return {
            "status": status,
            "query": query,
            "papers": [paper.to_dict() for paper in unique_papers],
            "result_count": len(unique_papers),
            "sources": sources,
            "source_counts": candidate_source_counts,
            "provider_status": provider_status,
            "active_sources": active_sources,
            "candidate_count": len(all_papers),
            "message": (
                f"Ditemukan {len(unique_papers)} paper setelah deduplikasi "
                f"dari {len(all_papers)} kandidat."
            ),
        }
