"""Automatic checks for journal indexes other than SINTA.

Only exact ISSN matches are accepted. Index membership is kept separate from
national accreditation and from quartile rankings.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

import httpx


class JournalIndexVerifier:
    DOAJ_URL = "https://doaj.org/api/search/journals/issn:{}"
    SCOPUS_URL = "https://api.elsevier.com/content/serial/title"

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.elsevier_api_key = os.getenv("ELSEVIER_API_KEY", "").strip()

    @staticmethod
    def normalize_issn(value: str) -> str:
        return re.sub(r"[^0-9X]", "", str(value or "").upper())

    @classmethod
    def _doaj_match(cls, payload: dict, issns: set[str]) -> dict | None:
        results = payload.get("results", [])
        if not isinstance(results, list):
            return None
        for result in results:
            bibjson = result.get("bibjson", {}) if isinstance(result, dict) else {}
            candidate_issns = bibjson.get("issn", []) if isinstance(bibjson, dict) else []
            if isinstance(candidate_issns, str):
                candidate_issns = [candidate_issns]
            normalized = {
                cls.normalize_issn(value)
                for value in candidate_issns
                if cls.normalize_issn(value)
            }
            if issns.intersection(normalized):
                return {
                    "status": "INDEXED",
                    "journal_name": str(bibjson.get("title") or "").strip(),
                    "source_url": "https://doaj.org/",
                }
        return None

    def _check_doaj(self, issns: list[str]) -> dict:
        if not issns:
            return {"status": "NO_METADATA", "source_url": "https://doaj.org/"}
        errors = []
        for issn in issns:
            try:
                normalized_issn = self.normalize_issn(issn)
                query_issn = f"{normalized_issn[:4]}-{normalized_issn[4:]}"
                response = httpx.get(
                    self.DOAJ_URL.format(query_issn),
                    timeout=self.timeout,
                    headers={"Accept": "application/json", "User-Agent": "DORA-DatasetResearch/1.0"},
                )
                response.raise_for_status()
                matched = self._doaj_match(response.json(), {self.normalize_issn(issn)})
                if matched:
                    matched["issn"] = issn
                    return matched
            except (httpx.HTTPError, ValueError) as error:
                errors.append(str(error))
        if errors:
            return {
                "status": "FAILED",
                "message": "DOAJ belum bisa dihubungi; hasil indeks DOAJ belum dapat dipastikan.",
                "source_url": "https://doaj.org/",
            }
        return {"status": "NOT_INDEXED", "source_url": "https://doaj.org/"}

    def _check_scopus(self, issns: list[str]) -> dict:
        if not self.elsevier_api_key:
            return {
                "status": "NOT_CONFIGURED",
                "message": "API Elsevier belum dikonfigurasi.",
                "source_url": "https://www.scopus.com/",
            }
        if not issns:
            return {"status": "NO_METADATA", "source_url": "https://www.scopus.com/"}

        errors = []
        for issn in issns:
            try:
                normalized_issn = self.normalize_issn(issn)
                query_issn = f"{normalized_issn[:4]}-{normalized_issn[4:]}"
                response = httpx.get(
                    self.SCOPUS_URL,
                    params={"issn": query_issn, "view": "STANDARD"},
                    timeout=self.timeout,
                    headers={
                        "Accept": "application/json",
                        "X-ELS-APIKey": self.elsevier_api_key,
                        "User-Agent": "DORA-DatasetResearch/1.0",
                    },
                )
                if response.status_code in {401, 403}:
                    return {
                        "status": "UNAVAILABLE",
                        "message": "Akses API Scopus tidak tersedia untuk kunci ini.",
                        "source_url": "https://www.scopus.com/",
                    }
                response.raise_for_status()
                payload = response.json()
                metadata = payload.get("serial-metadata-response", {})
                entries = metadata.get("entry", []) if isinstance(metadata, dict) else []
                if isinstance(entries, dict):
                    entries = [entries]
                expected = self.normalize_issn(issn)
                for entry in entries if isinstance(entries, list) else []:
                    if not isinstance(entry, dict):
                        continue
                    raw_issns = []
                    for field in ("prism:issn", "prism:eIssn"):
                        value = entry.get(field) or []
                        raw_issns.extend([value] if isinstance(value, str) else value)
                    if expected in {self.normalize_issn(value) for value in raw_issns}:
                        return {
                            "status": "INDEXED",
                            "journal_name": str(entry.get("dc:title") or "").strip(),
                            "source_url": "https://www.scopus.com/",
                            "issn": issn,
                        }
            except (httpx.HTTPError, ValueError) as error:
                errors.append(str(error))
        if errors:
            return {
                "status": "FAILED",
                "message": "Scopus belum bisa diperiksa; status indeks belum dapat dipastikan.",
                "source_url": "https://www.scopus.com/",
            }
        return {"status": "NOT_INDEXED", "source_url": "https://www.scopus.com/"}

    def verify(self, issn_values: list[str] | str) -> dict:
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        normalized = {}
        for value in issn_values or []:
            clean = str(value or "").strip()
            key = self.normalize_issn(clean)
            if key and len(key) == 8:
                normalized.setdefault(key, clean)
        issns = [f"{key[:4]}-{key[4:]}" for key in normalized]

        if not issns:
            return {
                "status": "NO_METADATA",
                "indexes": [],
                "sources": {
                    "DOAJ": {"status": "NO_METADATA", "source_url": "https://doaj.org/"},
                    "SCOPUS": {"status": "NO_METADATA", "source_url": "https://www.scopus.com/"},
                },
                "checked_at": datetime.now().astimezone().isoformat(timespec="minutes"),
                "issns": [],
            }

        sources = {
            "DOAJ": self._check_doaj(issns),
            "SCOPUS": self._check_scopus(issns),
        }
        statuses = [source.get("status") for source in sources.values()]
        positives = [name for name, source in sources.items() if source.get("status") == "INDEXED"]
        if positives:
            overall = "INDEXED"
        elif any(status in {"FAILED", "UNAVAILABLE", "NOT_CONFIGURED"} for status in statuses):
            overall = "PARTIAL"
        elif any(status == "NO_METADATA" for status in statuses):
            overall = "NO_METADATA"
        else:
            overall = "NOT_FOUND"

        return {
            "status": overall,
            "indexes": positives,
            "sources": sources,
            "checked_at": datetime.now().astimezone().isoformat(timespec="minutes"),
            "issns": issns,
        }
