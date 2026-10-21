from __future__ import annotations

from typing import List, Optional

import httpx

from app.research.paper import Paper


class OpenAlexClient:
    BASE_URL = "https://api.openalex.org/works"

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
        per_page: int = 20,
        page: int = 1,
    ) -> List[Paper]:

        if not query or not query.strip():
            return []

        params = {
            "search": query.strip(),
            "per-page": min(max(per_page, 1), 100),
            "page": max(page, 1),
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

        results = data.get("results", [])

        return [
            self._parse_work(item)
            for item in results
            if isinstance(item, dict)
        ]

    def _parse_work(self, item: dict) -> Paper:
        authors = []

        for authorship in item.get("authorships", []):
            author = authorship.get("author", {})

            display_name = author.get("display_name")

            if display_name:
                authors.append(display_name)

        abstract = self._reconstruct_abstract(
            item.get("abstract_inverted_index")
        )

        primary_location = item.get("primary_location") or {}

        source = primary_location.get("source") or {}

        venue = source.get("display_name")

        doi = item.get("doi")

        url = item.get("id")

        if doi:
            url = doi

        keywords = []

        for keyword in item.get("keywords", []):
            display_name = keyword.get("display_name")

            if display_name:
                keywords.append(display_name)

        year = item.get("publication_year")

        return Paper(
            title=item.get("title") or "Untitled",
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            venue=venue,
            url=url,
            citation_count=item.get("cited_by_count", 0) or 0,
            source="OpenAlex",
            external_id=item.get("id"),
            keywords=keywords,
            issn=(source.get("issn") or ([source.get("issn_l")] if source.get("issn_l") else [])),
            source_urls=(
                [{"source": "OpenAlex", "url": str(item.get("id"))}]
                if item.get("id")
                else []
            ),
        )

    @staticmethod
    def _reconstruct_abstract(
        inverted_index: Optional[dict],
    ) -> str:

        if not inverted_index:
            return ""

        words = []

        for word, positions in inverted_index.items():
            for position in positions:
                words.append((position, word))

        words.sort(key=lambda x: x[0])

        return " ".join(word for _, word in words)
