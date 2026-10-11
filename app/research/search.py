from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from app.research.crossref import CrossrefClient
from app.research.deduplication import PaperDeduplicator
from app.research.openalex import OpenAlexClient
from app.research.semantic_scholar import SemanticScholarClient
from app.research.europe_pmc import EuropePMCClient
from app.research.scopus import ScopusClient
from app.research.openaire import OpenAIREClient
from app.research.paper import Paper


class AcademicSearchEngine:
    """Search and merge records from several bibliographic indexes."""

    def __init__(
        self,
        email: Optional[str] = None,
        openalex: Optional[OpenAlexClient] = None,
        crossref: Optional[CrossrefClient] = None,
        semantic_scholar: Optional[SemanticScholarClient] = None,
        europe_pmc: Optional[EuropePMCClient] = None,
        scopus: Optional[ScopusClient] = None,
        openaire: Optional[OpenAIREClient] = None,
        scholar: Any = None,
        scholar_api_key: Optional[str] = None,
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
    ) -> Dict:
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
        provider_calls["Semantic Scholar"] = lambda: self.semantic_scholar.search(query=query, limit=limit)
        provider_calls["Europe PMC"] = lambda: self.europe_pmc.search(query=query, limit=limit)
        provider_calls["OpenAIRE"] = lambda: self.openaire.search(query=query, limit=limit)

        provider_status = {name: "ENABLED" for name in provider_calls}
        if self.scopus.enabled:
            provider_calls["Scopus"] = lambda: self.scopus.search(query=query, limit=limit)
            provider_status["Scopus"] = "ENABLED"
        else:
            provider_status["Scopus"] = "NOT_CONFIGURED"
        if not use_crossref:
            provider_status["Crossref"] = "DISABLED"

        provider_attempts = {name: 0 for name in provider_calls}
        provider_errors = {name: 0 for name in provider_calls}
        # Independent providers run in parallel; the GUI research worker owns
        # this operation, so network latency never blocks the desktop window.
        with ThreadPoolExecutor(max_workers=max(1, len(provider_calls))) as executor:
            futures = {
                executor.submit(search_provider): name
                for name, search_provider in provider_calls.items()
            }
            for future, provider in futures.items():
                provider_attempts[provider] += 1
                try:
                    results = future.result()
                    results = results if isinstance(results, list) else []
                    all_papers.extend(results)
                    candidate_source_counts[provider] = len(results)
                except Exception:
                    provider_errors[provider] += 1
                    candidate_source_counts[provider] = 0

        for provider, attempts in provider_attempts.items():
            provider_status[provider] = (
                "ERROR" if attempts and provider_errors[provider] == attempts else "ENABLED"
            )

        unique_papers = self.deduplicator.deduplicate(all_papers)
        unique_papers = unique_papers[:limit]

        sources = sorted(
            {
                paper.source
                for paper in unique_papers
                if paper.source
            }
        )

        active_sources = [
            source
            for source, status in provider_status.items()
            if status == "ENABLED"
        ]

        if not unique_papers:
            status = "PARTIAL" if any(
                value == "ERROR" for value in provider_status.values()
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
