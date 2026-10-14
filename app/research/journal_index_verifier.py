"""Automatic checks for journal indexes other than SINTA.

Only exact ISSN matches are accepted. Index membership is kept separate from
national accreditation and from quartile rankings.
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime

import httpx


class JournalIndexVerifier:
    DOAJ_URL = "https://doaj.org/api/search/journals/issn:{}"
    SCOPUS_URL = "https://api.elsevier.com/content/serial/title"
    OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
    OPENALEX_SOURCE_LISTS_URL = "https://api.openalex.org/source-lists"
    OPENALEX_CACHE_TTL = 6 * 60 * 60
    _openalex_assessment_cache: dict[str, tuple[float, dict]] = {}
    _openalex_assessment_lock = threading.Lock()
    _openalex_catalog_cache: tuple[float, dict] | None = None
    _openalex_catalog_lock = threading.Lock()

    _ACCREDITATION_LIST_IDS = {
        "kci-excellent", "kci-registered", "kci-candidate",
    }
    _RATING_LIST_PREFIXES = (
        "norway-", "jufo-", "ki-jl-", "abdc-", "russia-white-list-",
        "poland-", "fnege-", "ccf-", "publindex-", "anvur-", "dongbi-",
        "jpps-",
    )

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.elsevier_api_key = os.getenv("ELSEVIER_API_KEY", "").strip()

    @staticmethod
    def normalize_issn(value: str) -> str:
        return re.sub(r"[^0-9X]", "", str(value or "").upper())

    @classmethod
    def _fetch_openalex_list_catalog(cls, timeout: float) -> tuple[dict, bool]:
        now = time.monotonic()
        with cls._openalex_catalog_lock:
            cached = cls._openalex_catalog_cache
            if cached and cached[0] > now:
                return cached[1], True

        response = httpx.get(
            cls.OPENALEX_SOURCE_LISTS_URL,
            params={
                "per-page": 100,
                "select": "id,display_name,maintainer,url,list_version,description",
            },
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "DORA-DatasetResearch/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        catalog = {}
        results = payload.get("results", []) if isinstance(payload, dict) else []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            list_id = str(item.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
            if list_id:
                catalog[list_id.casefold()] = item
        with cls._openalex_catalog_lock:
            cls._openalex_catalog_cache = (now + cls.OPENALEX_CACHE_TTL, catalog)
        return catalog, True

    @classmethod
    def _assessment_kind(cls, list_id: str) -> str:
        normalized = str(list_id or "").casefold()
        if normalized in cls._ACCREDITATION_LIST_IDS:
            return "ACCREDITATION"
        if normalized.startswith(cls._RATING_LIST_PREFIXES):
            return "RATING_OR_CLASSIFICATION"
        return "DIRECTORY_OR_INDEX"

    def verify_official_assessments(self, issn_values: list[str] | str) -> dict[str, dict]:
        """Look up current external journal ratings/accreditations by exact ISSN.

        OpenAlex exposes externally maintained source-list memberships. Those
        memberships are reported with the maintaining list and edition, and are
        kept separate from article indexing (for example, DOAJ or Scopus).
        """
        if isinstance(issn_values, str):
            issn_values = [issn_values]
        normalized = sorted({
            key for value in (issn_values or [])
            if (key := self.normalize_issn(value)) and len(key) == 8
        })
        if not normalized:
            return {}

        now = time.monotonic()
        results: dict[str, dict] = {}
        with self._openalex_assessment_lock:
            for issn in normalized:
                cached = self._openalex_assessment_cache.get(issn)
                if cached and cached[0] > now:
                    results[issn] = dict(cached[1])
        missing = [issn for issn in normalized if issn not in results]
        if not missing:
            return results

        # A single request can resolve many ISSNs. Split long result sets so
        # broad searches stay fast and do not exceed URL/API limits.
        chunks = [missing[index:index + 60] for index in range(0, len(missing), 60)]
        from concurrent.futures import ThreadPoolExecutor, as_completed

        source_payloads = {}
        source_errors = {}

        def fetch_sources(chunk):
            values = [f"{issn[:4]}-{issn[4:]}" for issn in chunk]
            response = httpx.get(
                self.OPENALEX_SOURCES_URL,
                params={
                    "filter": "issn:" + "|".join(values),
                    "per-page": 100,
                    "select": "display_name,issn,issn_l,listed_in",
                },
                timeout=self.timeout,
                headers={"Accept": "application/json", "User-Agent": "DORA-DatasetResearch/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            return chunk, payload

        with ThreadPoolExecutor(max_workers=min(4, len(chunks) + 1)) as executor:
            catalog_future = executor.submit(self._fetch_openalex_list_catalog, self.timeout)
            futures = {executor.submit(fetch_sources, chunk): chunk for chunk in chunks}
            try:
                catalog, _ = catalog_future.result()
            except Exception:
                catalog = {}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    _chunk, payload = future.result()
                    for source in (payload.get("results", []) if isinstance(payload, dict) else []):
                        if not isinstance(source, dict):
                            continue
                        source_issns = source.get("issn", [])
                        if isinstance(source_issns, str):
                            source_issns = [source_issns]
                        if source.get("issn_l"):
                            source_issns = [*source_issns, source.get("issn_l")]
                        matching = {
                            self.normalize_issn(value)
                            for value in source_issns
                            if self.normalize_issn(value) in chunk
                        }
                        list_ids = source.get("listed_in", [])
                        if isinstance(list_ids, str):
                            list_ids = [list_ids]
                        assessments = []
                        for raw_list_id in list_ids if isinstance(list_ids, list) else []:
                            list_id = str(raw_list_id or "").rstrip("/").rsplit("/", 1)[-1].casefold()
                            if not list_id:
                                continue
                            metadata = catalog.get(list_id, {})
                            assessments.append({
                                "list_id": list_id,
                                "system_name": str(metadata.get("display_name") or list_id),
                                "rank": str(metadata.get("display_name") or list_id),
                                "kind": self._assessment_kind(list_id),
                                "maintainer": str(metadata.get("maintainer") or ""),
                                "list_version": str(metadata.get("list_version") or ""),
                                "source_url": str(metadata.get("url") or "https://openalex.org/"),
                            })
                        for issn in matching:
                            matched_source = source_payloads.setdefault(issn, {
                                "status": "CHECKED",
                                "source_name": str(source.get("display_name") or ""),
                                "assessments": [],
                            })
                            known_list_ids = {
                                str(item.get("list_id") or "")
                                for item in matched_source["assessments"]
                                if isinstance(item, dict)
                            }
                            for assessment in assessments:
                                if assessment["list_id"] not in known_list_ids:
                                    matched_source["assessments"].append(assessment)
                except Exception as error:
                    for issn in chunk:
                        source_errors[issn] = str(error)

        cache_expiry = time.monotonic() + self.OPENALEX_CACHE_TTL
        for issn in missing:
            if issn in source_errors:
                result = {
                    "status": "FAILED",
                    "assessments": [],
                    "message": "Daftar penilaian jurnal OpenAlex belum dapat diperiksa.",
                }
            else:
                result = source_payloads.get(issn, {
                    "status": "CHECKED",
                    "source_name": "",
                    "assessments": [],
                })
            results[issn] = result
            if result["status"] == "CHECKED":
                with self._openalex_assessment_lock:
                    self._openalex_assessment_cache[issn] = (cache_expiry, dict(result))
        return results

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
