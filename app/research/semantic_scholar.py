from __future__ import annotations

from typing import List, Optional

import httpx

from app.research.paper import Paper


class SemanticScholarClient:
    """Search the Semantic Scholar Academic Graph API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,year,authors,venue,externalIds,url,citationCount"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 12.0):
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, limit: int = 20) -> List[Paper]:
        if not query or not query.strip():
            return []
        headers = {"User-Agent": "DatasetResearch/4.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        response = httpx.get(
            self.BASE_URL,
            params={"query": query.strip(), "limit": min(max(limit, 1), 100), "fields": self.FIELDS},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [self._to_paper(item) for item in payload.get("data", []) if isinstance(item, dict)]

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        authors = [
            author.get("name")
            for author in item.get("authors", [])
            if isinstance(author, dict) and author.get("name")
        ]
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")
        paper_id = item.get("paperId")
        return Paper(
            title=item.get("title") or "Untitled",
            authors=authors,
            abstract=item.get("abstract") or "",
            year=item.get("year"),
            doi=doi,
            venue=item.get("venue"),
            url=item.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None),
            citation_count=item.get("citationCount") or 0,
            source="Semantic Scholar",
            external_id=f"S2:{paper_id}" if paper_id else None,
        )
