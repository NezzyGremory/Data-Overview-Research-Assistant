from __future__ import annotations

from typing import Dict, List

from app.research.paper import Paper


class PaperDeduplicator:

    def deduplicate(self, papers: List[Paper]) -> List[Paper]:
        unique: Dict[str, Paper] = {}

        for paper in papers:
            key = self._build_key(paper)

            if key not in unique:
                unique[key] = paper
            else:
                unique[key] = self._merge(
                    unique[key],
                    paper,
                )

        return list(unique.values())

    def _build_key(self, paper: Paper) -> str:
        if paper.normalized_doi:
            return f"doi:{paper.normalized_doi}"

        if paper.external_id:
            return f"id:{paper.external_id.strip().lower()}"

        return f"title:{paper.normalized_title}"

    def _merge(
        self,
        first: Paper,
        second: Paper,
    ) -> Paper:

        if not first.abstract and second.abstract:
            first.abstract = second.abstract

        if not first.doi and second.doi:
            first.doi = second.doi

        if not first.url and second.url:
            first.url = second.url

        if not first.venue and second.venue:
            first.venue = second.venue

        if not first.year and second.year:
            first.year = second.year

        if second.citation_count > first.citation_count:
            first.citation_count = second.citation_count

        existing_authors = set(
            author.lower()
            for author in first.authors
        )

        for author in second.authors:
            if author.lower() not in existing_authors:
                first.authors.append(author)
                existing_authors.add(author.lower())

        existing_keywords = set(
            keyword.lower()
            for keyword in first.keywords
        )

        for keyword in second.keywords:
            if keyword.lower() not in existing_keywords:
                first.keywords.append(keyword)
                existing_keywords.add(keyword.lower())

        existing_issns = {
            "".join(character for character in issn.upper() if character.isalnum())
            for issn in first.issn
        }
        for issn in second.issn:
            normalized_issn = "".join(
                character for character in issn.upper() if character.isalnum()
            )
            if normalized_issn and normalized_issn not in existing_issns:
                first.issn.append(issn)
                existing_issns.add(normalized_issn)

        if first.source and second.source:
            if second.source not in first.source:
                first.source = (
                    f"{first.source}, {second.source}"
                )
        elif second.source:
            first.source = second.source

        if not first.sinta_screening and second.sinta_screening:
            first.sinta_screening = second.sinta_screening

        return first
