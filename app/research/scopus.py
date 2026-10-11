from __future__ import annotations

import os
from typing import List, Optional

import httpx

from app.research.paper import Paper


class ScopusClient:
    """Optional Scopus API search; requires an entitled Elsevier API key."""

    BASE_URL = "https://api.elsevier.com/content/search/scopus"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 12.0):
        self.api_key = (api_key if api_key is not None else os.getenv("ELSEVIER_API_KEY", "")).strip()
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, limit: int = 20) -> List[Paper]:
        if not self.enabled or not query or not query.strip():
            return []
        response = httpx.get(
            self.BASE_URL,
            params={
                "query": f"TITLE-ABS-KEY({query.strip()})",
                "count": min(max(limit, 1), 100),
                "view": "STANDARD",
            },
            headers={
                "X-ELS-APIKey": self.api_key,
                "Accept": "application/json",
                "User-Agent": "DatasetResearch/4.0",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        entries = response.json().get("search-results", {}).get("entry", [])
        return [self._to_paper(item) for item in entries if isinstance(item, dict) and item.get("dc:title")]

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        doi = item.get("prism:doi")
        eid = item.get("eid")
        year = None
        cover_date = item.get("prism:coverDate") or ""
        if len(cover_date) >= 4 and cover_date[:4].isdigit():
            year = int(cover_date[:4])
        issn = item.get("prism:issn") or ""
        return Paper(
            title=item.get("dc:title") or "Untitled",
            authors=[item["dc:creator"]] if item.get("dc:creator") else [],
            abstract=item.get("dc:description") or "",
            year=year,
            doi=doi,
            venue=item.get("prism:publicationName"),
            url=(f"https://www.scopus.com/record/display.uri?eid={eid}" if eid else None),
            citation_count=item.get("citedby-count") or 0,
            source="Scopus",
            external_id=f"SCOPUS:{eid}" if eid else None,
            issn=[value.strip() for value in issn.split(",") if value.strip()],
        )
