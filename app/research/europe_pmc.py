from __future__ import annotations

from typing import List

import httpx

from app.research.paper import Paper


class EuropePMCClient:
    """Search Europe PMC's public REST service for life-sciences literature."""

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    def search(self, query: str, limit: int = 20) -> List[Paper]:
        if not query or not query.strip():
            return []
        response = httpx.get(
            self.BASE_URL,
            params={
                "query": query.strip(),
                "format": "json",
                "resultType": "core",
                "pageSize": min(max(limit, 1), 100),
            },
            headers={"User-Agent": "DatasetResearch/4.0"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [self._to_paper(item) for item in payload.get("resultList", {}).get("result", []) if isinstance(item, dict)]

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        journal_info = item.get("journalInfo") or {}
        journal = journal_info.get("journal") or {}
        published = item.get("firstPublicationDate") or ""
        year = item.get("pubYear")
        if not year and len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])
        issn_values = []
        for key in ("issnOfPublication", "essn"):
            value = journal.get(key)
            if isinstance(value, str) and value and value not in issn_values:
                issn_values.append(value)
        doi = item.get("doi")
        pmid = item.get("pmid")
        record_id = item.get("id")
        record_source = item.get("source") or "MED"
        return Paper(
            title=item.get("title") or "Untitled",
            authors=[item["authorString"]] if item.get("authorString") else [],
            abstract=item.get("abstractText") or "",
            year=year,
            doi=doi,
            venue=journal.get("title") or item.get("journalTitle"),
            url=(
                f"https://europepmc.org/article/MED/{pmid}"
                if pmid
                else f"https://europepmc.org/article/{record_source}/{record_id}"
                if record_id
                else None
            ),
            citation_count=item.get("citedByCount") or 0,
            source="Europe PMC",
            external_id=f"PMID:{pmid}" if pmid else (f"EPMC:{item.get('id')}" if item.get("id") else None),
            issn=issn_values,
        )
