from __future__ import annotations

from typing import List, Optional

import httpx

from app.research.paper import Paper


class CrossrefClient:
    BASE_URL = "https://api.crossref.org/works"

    def __init__(
        self,
        email: Optional[str] = None,
        timeout: float = 20.0,
    ):
        self.email = email
        self.timeout = timeout

    def search(
        self,
        query: str,
        rows: int = 20,
        offset: int = 0,
    ) -> List[Paper]:

        if not query or not query.strip():
            return []

        params = {
            "query": query.strip(),
            "rows": min(max(rows, 1), 100),
            "offset": max(offset, 0),
        }

        headers = {
            "User-Agent": "DatasetResearch/1.0"
        }

        if self.email:
            headers["User-Agent"] = (
                f"DatasetResearch/1.0 (mailto:{self.email})"
            )

        response = httpx.get(
            self.BASE_URL,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        items = message.get("items", [])

        return [
            self._parse_item(item)
            for item in items
            if isinstance(item, dict)
        ]

    def _parse_item(self, item: dict) -> Paper:
        authors = []

        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")

            name = f"{given} {family}".strip()

            if name:
                authors.append(name)

        title_data = item.get("title") or []

        title = (
            title_data[0]
            if title_data
            else "Untitled"
        )

        abstract = item.get("abstract") or ""

        year = None

        published = item.get("published-print") or \
            item.get("published-online") or \
            item.get("issued")

        if published:
            date_parts = published.get("date-parts", [])

            if date_parts and date_parts[0]:
                try:
                    year = int(date_parts[0][0])
                except (TypeError, ValueError):
                    year = None

        doi = item.get("DOI")

        url = item.get("URL")

        container = item.get("container-title") or []

        venue = (
            container[0]
            if container
            else None
        )

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            venue=venue,
            url=url,
            citation_count=item.get(
                "is-referenced-by-count",
                0,
            ) or 0,
            source="Crossref",
            external_id=item.get("URL"),
            keywords=[],
            issn=item.get("ISSN", []) or [],
        )
