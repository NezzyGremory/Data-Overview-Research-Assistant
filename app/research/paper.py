from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    citation_count: int = 0
    source: str = ""
    external_id: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    issn: List[str] = field(default_factory=list)
    sinta_screening: Dict[str, Any] = field(default_factory=dict)
    accreditation_screening: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "venue": self.venue,
            "url": self.url,
            "citation_count": self.citation_count,
            "source": self.source,
            "external_id": self.external_id,
            "keywords": self.keywords,
            "issn": self.issn,
            "sinta_screening": self.sinta_screening,
            "accreditation_screening": self.accreditation_screening,
        }

    @property
    def normalized_doi(self) -> Optional[str]:
        if not self.doi:
            return None

        doi = self.doi.strip().lower()

        for prefix in (
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
            "doi:",
        ):
            if doi.startswith(prefix):
                doi = doi[len(prefix):]

        return doi.strip()

    @property
    def normalized_title(self) -> str:
        import re

        title = self.title.lower()
        title = re.sub(r"[^a-z0-9\s]", " ", title)
        title = re.sub(r"\s+", " ", title)

        return title.strip()
