from __future__ import annotations

from typing import Any, List, Optional

import httpx

from app.research.paper import Paper


class OpenAIREClient:
    """Search the public OpenAIRE Graph for scholarly publications."""

    BASE_URL = "https://api-beta.openaire.eu/graph/v4/research-products"

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    def search(self, query: str, limit: int = 20) -> List[Paper]:
        if not query or not query.strip():
            return []
        response = httpx.get(
            self.BASE_URL,
            params={
                "search": query.strip(),
                "filter": "type:publication",
                "page_size": min(max(limit, 1), 100),
            },
            headers={"User-Agent": "DatasetResearch/4.0", "Accept": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        return [self._to_paper(item) for item in results if isinstance(item, dict)]

    @staticmethod
    def _to_paper(item: dict[str, Any]) -> Paper:
        authors = []
        for author in item.get("authors") or []:
            if isinstance(author, dict):
                name = author.get("fullName") or author.get("name")
                if name:
                    authors.append(str(name))

        ids = item.get("pids") or []
        doi: Optional[str] = None
        other_id: Optional[str] = None
        if isinstance(ids, list):
            for identifier in ids:
                if not isinstance(identifier, dict):
                    continue
                scheme = str(identifier.get("scheme") or "").lower()
                value = identifier.get("value")
                if not value:
                    continue
                if scheme == "doi" and not doi:
                    doi = str(value)
                elif not other_id:
                    other_id = str(value)

        containers = item.get("container") or {}
        venue = containers.get("name") if isinstance(containers, dict) else None
        if not venue and isinstance(containers, list) and containers:
            first_container = containers[0]
            if isinstance(first_container, dict):
                venue = first_container.get("name")

        descriptions = item.get("descriptions") or []
        abstract = ""
        if isinstance(descriptions, list):
            abstract = next((str(value) for value in descriptions if value), "")
        elif isinstance(descriptions, str):
            abstract = descriptions

        year = None
        publication_date = str(item.get("publicationDate") or "")
        if len(publication_date) >= 4 and publication_date[:4].isdigit():
            year = int(publication_date[:4])

        url = None
        instances = item.get("instances") or []
        if isinstance(instances, list):
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                urls = instance.get("urls") or []
                if urls:
                    url = str(urls[0])
                    break
        if not url and doi:
            url = f"https://doi.org/{doi}"

        citations = (item.get("indicators") or {}).get("citationImpact") or {}
        try:
            citation_count = int(citations.get("citationCount") or 0)
        except (TypeError, ValueError):
            citation_count = 0

        return Paper(
            title=item.get("mainTitle") or "Untitled",
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            venue=venue,
            url=url,
            citation_count=citation_count,
            source="OpenAIRE",
            external_id=f"OpenAIRE:{item.get('id')}" if item.get("id") else other_id,
        )
