from __future__ import annotations

import re
import time
import math
import html
import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlsplit

import httpx


@dataclass
class SintaVerificationResult:
    """Hasil verifikasi akreditasi jurnal di direktori SINTA."""

    found: bool = False
    rank: Optional[str] = None  # S1, S2, S3, S4, S5, S6
    status: str = "UNVERIFIED"  # S1-S6, NOT_ACCREDITED, UNVERIFIED, NOT_IN_SINTA
    lookup_status: str = "UNVERIFIED"  # VERIFIED, NOT_FOUND, FAILED, NO_MATCH, NO_METADATA, RANK_UNAVAILABLE, AMBIGUOUS, PERIOD_UNVERIFIED
    journal_name: Optional[str] = None
    issn_matched: Optional[str] = None
    all_issns: List[str] = field(default_factory=list)
    publisher: Optional[str] = None
    source_url: str = "https://sinta.kemdiktisaintek.go.id/journals"
    verification_source: str = "Direktori Resmi SINTA (sinta.kemdiktisaintek.go.id)"
    verification_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    match_method: str = "NONE"  # "ISSN", "JOURNAL_NAME", "MANUAL"
    is_ambiguous: bool = False
    ambiguity_reason: str = ""
    period_start: Optional[int] = None
    period_end: Optional[int] = None
    period_status: str = "UNKNOWN"  # "VALID", "EXPIRED", "PERIOD_CHECK_REQUIRED", "NOT_APPLICABLE"
    period_confirmed: bool = False
    checked_for_year: Optional[int] = None
    notes: str = ""
    assistive_candidate_url: Optional[str] = None
    assistive_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "rank": self.rank,
            "status": self.status,
            "lookup_status": self.lookup_status,
            "journal_name": self.journal_name,
            "issn_matched": self.issn_matched,
            "all_issns": self.all_issns,
            "publisher": self.publisher,
            "source_url": self.source_url,
            "verification_source": self.verification_source,
            "verification_date": self.verification_date,
            "match_method": self.match_method,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "period_status": self.period_status,
            "period_confirmed": self.period_confirmed,
            "checked_for_year": self.checked_for_year,
            "notes": self.notes,
            "assistive_candidate_url": self.assistive_candidate_url,
            "assistive_note": self.assistive_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SintaVerificationResult":
        if not isinstance(data, dict):
            return cls()
        return cls(
            found=bool(data.get("found", False)),
            rank=data.get("rank"),
            status=str(data.get("status") or "UNVERIFIED"),
            lookup_status=str(data.get("lookup_status") or "UNVERIFIED"),
            journal_name=data.get("journal_name"),
            issn_matched=data.get("issn_matched"),
            all_issns=data.get("all_issns") or [],
            publisher=data.get("publisher"),
            source_url=str(data.get("source_url") or "https://sinta.kemdiktisaintek.go.id/journals"),
            verification_source=str(data.get("verification_source") or "Direktori Resmi SINTA (sinta.kemdiktisaintek.go.id)"),
            verification_date=str(data.get("verification_date") or time.strftime("%Y-%m-%d %H:%M")),
            match_method=str(data.get("match_method") or "NONE"),
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            ambiguity_reason=str(data.get("ambiguity_reason") or ""),
            period_start=data.get("period_start"),
            period_end=data.get("period_end"),
            period_status=str(data.get("period_status") or "UNKNOWN"),
            period_confirmed=bool(data.get("period_confirmed", False)),
            checked_for_year=data.get("checked_for_year"),
            notes=str(data.get("notes") or ""),
            assistive_candidate_url=data.get("assistive_candidate_url"),
            assistive_note=str(data.get("assistive_note") or ""),
        )


class _SintaDirectoryHtmlParser(HTMLParser):
    """Parser teks direktori SINTA untuk mengekstrak entri jurnal secara aman."""

    _BLOCK_TAGS = {"br", "p", "div", "article", "li", "h1", "h2", "h3", "h4", "tr", "td"}
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    _RECORD_CLASSES = {
        "card", "journal-card", "card-journal", "journal-item", "journal-list-item",
        "journal-result", "journal-entry", "result-item", "search-result", "list-group-item",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.records: List[List[str]] = []
        self._record_parts: Optional[List[str]] = None
        self._record_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs: Any):
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").casefold().split())
        is_journal_record_class = any(
            re.search(r"(?:^|[-_])journal[-_](?:card|item|result|entry)(?:$|[-_])", value)
            for value in classes
        )
        is_record = (
            tag.lower() == "article"
            or bool(classes & self._RECORD_CLASSES)
            or is_journal_record_class
        )
        if self._record_parts is None and is_record:
            self._record_parts = []
            self._record_stack = [tag.lower()]
        elif self._record_parts is not None and tag.lower() not in self._VOID_TAGS:
            self._record_stack.append(tag.lower())
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")
            if self._record_parts is not None:
                self._record_parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")
            if self._record_parts is not None:
                self._record_parts.append("\n")
        if self._record_parts is not None and tag.lower() in self._record_stack:
            match_index = len(self._record_stack) - 1 - self._record_stack[::-1].index(tag.lower())
            del self._record_stack[match_index:]
            if not self._record_stack:
                self.records.append(self._lines(self._record_parts))
                self._record_parts = None

    def handle_data(self, data: str):
        self.parts.append(data)
        if self._record_parts is not None:
            self._record_parts.append(data)

    def get_lines(self) -> List[str]:
        return self._lines(self.parts)

    def get_records(self) -> List[List[str]]:
        return [record for record in self.records if record]

    def get_records_for_issns(self, target_issns: List[str]) -> List[List[str]]:
        """Fallback extraction for directory pages whose card markup changed."""
        targets = {SintaVerifier.normalize_issn(value) for value in target_issns}
        targets.discard("")
        if not targets:
            return []

        lines = self.get_lines()
        markers = []
        for index, line in enumerate(lines):
            values = {
                SintaVerifier.normalize_issn(value)
                for value in re.findall(
                    r"(?:P-ISSN|E-ISSN|ISSN)\s*[:#]?\s*([0-9Xx-]+)", line, re.I
                )
            }
            values.discard("")
            if values:
                markers.append((index, values))

        groups = []
        for index, values in markers:
            if (
                groups
                and index - groups[-1]["last"] <= 3
                and not any(
                    re.search(r"\bSINTA\s*[1-6]\b|\bS[1-6]\s+Accredited\b", line, re.I)
                    for line in lines[groups[-1]["last"] + 1:index]
                )
            ):
                groups[-1]["last"] = index
                groups[-1]["issns"].update(values)
            else:
                groups.append({"start": index, "last": index, "issns": set(values)})

        found = []
        for position, group in enumerate(groups):
            if not (group["issns"] & targets):
                continue
            end = groups[position + 1]["start"] if position + 1 < len(groups) else len(lines)
            found.append(lines[group["start"]:end])
        return found

    @staticmethod
    def _lines(parts: List[str]) -> List[str]:
        raw = " ".join(parts)
        return [line.strip() for line in raw.splitlines() if line.strip()]


class SintaVerifier:
    """Verifikator akreditasi jurnal SINTA (Kemdiktisaintek RI).

    Aturan verifikasi:
    1. Verifikasi dilakukan pada JURNAL, bukan artikel.
    2. Pencocokan utama melalui ISSN (P-ISSN / E-ISSN).
    3. Jika ISSN tidak tersedia, gunakan nama jurnal sebagai alternatif dan tandai jika ambigu.
    4. Peringkat SINTA hanya berlaku jika periode masa berlaku SK akreditasi mencakup tahun terbit artikel.
       Jika periode tidak bisa dipastikan secara otomatis, status ditetapkan sebagai 'UNVERIFIED'
       (perlu pemeriksaan tahun oleh pengguna).
    5. Indeksasi di Scopus, Crossref, atau provider lain TIDAK dianggap sebagai bukti akreditasi SINTA.
    """

    BASE_URL = "https://sinta.kemdiktisaintek.go.id/journals"

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self._search_cache: Dict[str, str] = {}
        self._profile_cache: Dict[str, str] = {}
        self._gemini_candidate_cache: Dict[str, Optional[Dict[str, str]]] = {}
        self._client: Optional[httpx.Client] = None

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    @staticmethod
    def normalize_issn(issn: Optional[str]) -> str:
        """Menghapus strip, spasi, dan menormalkan huruf X."""
        if not issn:
            return ""
        return re.sub(r"[^0-9Xx]", "", str(issn)).upper()

    @staticmethod
    def normalize_name(name: Optional[str]) -> str:
        """Normalisasi nama jurnal untuk perbandingan teks."""
        if not name:
            return ""
        name = name.lower()
        name = re.sub(r"[^a-z0-9]", "", name)
        return name

    @staticmethod
    def format_issn(raw: Optional[str]) -> str:
        """Format ISSN standar: 1234-567X."""
        cleaned = SintaVerifier.normalize_issn(raw)
        if len(cleaned) == 8:
            return f"{cleaned[:4]}-{cleaned[4:]}"
        return raw or ""

    @staticmethod
    def normalize_year(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        year = int(number)
        return year if 1000 <= year <= 2200 else None

    @staticmethod
    def normalize_doi(value: Optional[str]) -> str:
        if not value:
            return ""
        doi = html.unescape(str(value)).strip().casefold()
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
        doi = re.sub(r"^doi\s*:\s*", "", doi)
        return doi.strip().rstrip(".,;:)]}>")

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://sinta.kemdiktisaintek.go.id/journals",
            }
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=min(5.0, self.timeout)),
                follow_redirects=True,
                headers=headers,
            )
        return self._client

    def search_sinta_directory(self, query: str) -> str:
        """Mengambil halaman pencarian direktori SINTA untuk query tertentu."""
        if not query or not query.strip():
            return ""
        cache_key = query.strip().casefold()
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]
        encoded_query = quote_plus(query.strip())
        urls = (
            f"{self.BASE_URL}?q={encoded_query}",
            f"{self.BASE_URL}/index/?q={encoded_query}",
        )
        last_error = None
        for url in urls:
            try:
                response = self._get_client().get(url)
                response.raise_for_status()
                self._search_cache[cache_key] = response.text
                return response.text
            except httpx.HTTPStatusError as exc:
                last_error = exc
                continue
        if last_error:
            raise last_error
        raise RuntimeError("Direktori SINTA tidak mengembalikan respons.")

    def fetch_sinta_profile(self, profile_url: str) -> str:
        """Fetch only a journal profile URL discovered on official SINTA search results."""
        parts = urlsplit(str(profile_url or ""))
        if (
            parts.scheme != "https"
            or parts.hostname != "sinta.kemdiktisaintek.go.id"
            or not re.fullmatch(r"/journals/profile/\d+/?", parts.path)
        ):
            raise ValueError("Tautan profil bukan URL profil jurnal SINTA resmi.")
        normalized_url = f"https://{parts.hostname}{parts.path.rstrip('/')}/"
        if normalized_url not in self._profile_cache:
            response = self._get_client().get(normalized_url)
            response.raise_for_status()
            self._profile_cache[normalized_url] = response.text
        return self._profile_cache[normalized_url]

    @classmethod
    def _profile_url_for_issn(cls, html_text: str, issn: Optional[str]) -> Optional[str]:
        target = cls.normalize_issn(issn)
        if not target:
            return None
        issn_match = next(
            (
                match
                for match in re.finditer(r"(?<!\d)(?:\d[\s-]*){7}[0-9Xx](?![\w])", html_text)
                if cls.normalize_issn(match.group()) == target
            ),
            None,
        )
        if not issn_match:
            return None
        links = list(re.finditer(
            r"href\s*=\s*['\"]([^'\"]*/journals/profile/\d+[^'\"]*)['\"]",
            html_text,
            re.I,
        ))
        before = [link for link in links if link.start() <= issn_match.start()]
        link = before[-1] if before else next(
            (candidate for candidate in links if candidate.start() > issn_match.start()),
            None,
        )
        if not link:
            return None
        return urljoin(cls.BASE_URL + "/", link.group(1))

    @staticmethod
    def _extract_rank_from_lines(lines: List[str], require_current_label: bool = False):
        rank_patterns = (
            re.compile(r"\bSINTA\s*[-:]?\s*([1-6])\b", re.I),
            re.compile(r"\bS([1-6])\b\s*(?:Accredited|Akreditasi)\b", re.I),
        )
        occurrences = []
        for line_index, line in enumerate(lines):
            for pattern in rank_patterns:
                occurrences.extend(
                    (line_index, f"S{match.group(1)}", match.start(), match.end())
                    for match in pattern.finditer(line)
                )

        current_label = re.compile(
            r"current\s+(?:ac{1,2}reditation(?:\s+status)?|ranking|rank|status)|"
            r"(?:akreditasi|peringkat)\s+(?:saat\s+ini|terkini)",
            re.I,
        )
        current_ranks = []
        for line_index, line in enumerate(lines):
            label_match = current_label.search(line)
            if not label_match:
                continue
            same_line = [
                (rank, start, end)
                for candidate_line, rank, start, end in occurrences
                if candidate_line == line_index
            ]
            if same_line:
                closest_distance = min(
                    0 if start <= label_match.start() <= end
                    else min(abs(start - label_match.start()), abs(end - label_match.start()))
                    for _, start, end in same_line
                )
                current_ranks.extend(
                    rank
                    for rank, start, end in same_line
                    if (
                        0 if start <= label_match.start() <= end
                        else min(abs(start - label_match.start()), abs(end - label_match.start()))
                    ) == closest_distance
                )
                continue

            # The value may be in the next element or immediately before the label.
            for neighbor in (line_index + 1, line_index - 1):
                if not 0 <= neighbor < len(lines):
                    continue
                neighbor_ranks = [
                    rank for candidate_line, rank, _, _ in occurrences
                    if candidate_line == neighbor
                ]
                if neighbor_ranks:
                    current_ranks.extend(neighbor_ranks)
                    break

        preferred = set(current_ranks)
        all_ranks = {rank for _, rank, _, _ in occurrences}
        has_current_label = bool(current_ranks)
        if len(preferred) == 1:
            return next(iter(preferred)), False, "", True
        if len(preferred) > 1:
            return None, True, "Bagian peringkat terkini memuat beberapa peringkat yang berbeda.", True
        if require_current_label:
            if len(all_ranks) > 1:
                return None, True, "Profil memuat beberapa peringkat tetapi label peringkat terkini tidak terbaca.", False
            return None, False, "Label peringkat terkini tidak ditemukan pada profil jurnal.", False
        if len(all_ranks) == 1:
            return next(iter(all_ranks)), False, "", has_current_label
        if len(all_ranks) > 1:
            return None, True, "Entri memuat beberapa peringkat SINTA dan tidak menandai dengan jelas mana yang terkini.", False
        return None, False, "", False

    @classmethod
    def _article_record_rank(cls, profile_html: str, article_doi: str):
        target_doi = cls.normalize_doi(article_doi)
        if not target_doi:
            return None
        parser = _SintaDirectoryHtmlParser()
        parser.feed(profile_html)
        for line in parser.get_lines():
            if target_doi not in cls.normalize_doi(line):
                continue
            accredited = bool(re.search(r"\b(?:accred(?:itation)?|akreditasi)\b", line, re.I))
            mentions = cls._extract_rank_from_lines([line])
            rank, ambiguous, reason, _ = mentions
            if ambiguous:
                return {"found": True, "rank": None, "ambiguous": True, "notes": reason}
            if accredited and rank:
                return {"found": True, "rank": rank, "ambiguous": False, "notes": ""}
            return {
                "found": True,
                "rank": None,
                "ambiguous": False,
                "notes": "DOI artikel ditemukan di profil resmi SINTA, tetapi label peringkat akreditasinya tidak tersedia.",
            }
        return None

    def _resolve_from_official_profile(
        self,
        result: SintaVerificationResult,
        issn_list: List[str],
        article_year: Optional[int],
        article_doi: Optional[str] = None,
    ) -> SintaVerificationResult:
        """Resolve a directory candidate against its official journal profile."""
        profile_url = result.source_url
        if not re.fullmatch(
            r"https://sinta\.kemdiktisaintek\.go\.id/journals/profile/\d+/?",
            profile_url or "",
        ):
            result.rank = None
            result.status = "UNVERIFIED"
            result.lookup_status = "RANK_UNAVAILABLE"
            result.period_status = "UNKNOWN"
            result.is_ambiguous = False
            result.ambiguity_reason = ""
            if article_doi:
                reason = "Profil resmi jurnal tidak ditemukan, sehingga DOI artikel tidak dapat dicocokkan langsung di SINTA."
            else:
                reason = "Hasil pencarian tidak menyediakan tautan profil jurnal resmi; peringkat tidak ditetapkan dari cuplikan pencarian."
            result.notes = reason
            return result

        try:
            profile_html = self.fetch_sinta_profile(profile_url)
        except Exception as error:
            result.rank = None
            result.status = "UNVERIFIED"
            result.lookup_status = "FAILED"
            result.period_status = "UNKNOWN"
            result.is_ambiguous = False
            result.ambiguity_reason = ""
            result.notes = (
                "ISSN cocok dengan hasil pencarian, tetapi profil resmi SINTA tidak dapat dibaca. "
                f"Peringkat tidak ditetapkan. Detail: {error}"
            )
            return result

        profile_parser = _SintaDirectoryHtmlParser()
        profile_parser.feed(profile_html)
        profile_lines = profile_parser.get_lines()
        expected_issns = {
            self.normalize_issn(value) for value in (issn_list or [])
            if self.normalize_issn(value)
        }
        profile_issns = {
            self.normalize_issn(value)
            for value in re.findall(
                r"(?:P-ISSN|E-ISSN|ISSN)\s*[:#]?\s*([0-9Xx-]+)",
                " ".join(profile_lines),
                re.I,
            )
            if self.normalize_issn(value)
        }
        matched_issn = next(iter(expected_issns & profile_issns), None)
        profile_journal_name = self._journal_name_from_profile(profile_lines)
        exact_name_match = bool(
            result.match_method == "JOURNAL_NAME"
            and self.normalize_name(result.journal_name)
            and self.normalize_name(result.journal_name)
            == self.normalize_name(profile_journal_name)
        )
        if not matched_issn and not exact_name_match:
            result.found = False
            result.rank = None
            result.status = "UNVERIFIED"
            result.lookup_status = "NO_MATCH"
            result.is_ambiguous = False
            result.match_method = "NONE"
            result.notes = (
                "Profil resmi yang ditautkan dari hasil pencarian tidak memuat ISSN artikel. "
                "Peringkat tidak ditetapkan karena identitas jurnal tidak cocok."
            )
            return result

        result.found = True
        result.source_url = profile_url
        result.issn_matched = self.format_issn(matched_issn) if matched_issn else None
        result.all_issns = [self.format_issn(value) for value in sorted(profile_issns)]
        result.journal_name = profile_journal_name or result.journal_name
        profile_text = " ".join(profile_lines)
        period_match = re.search(
            r"(?:masa\s+berlaku|periode(?:\s+akreditasi)?|akreditasi\s+berlaku|"
            r"valid(?:ity)?\s+period|accreditation\s+period|sk\s+akreditasi)"
            r"[^0-9]{0,80}(20\d{2})\s*(?:-|\u2013|\u2014|sampai|hingga|to|s\.d\.)\s*(20\d{2})",
            profile_text,
            re.I,
        )
        result.period_start = int(period_match.group(1)) if period_match else None
        result.period_end = int(period_match.group(2)) if period_match else None

        # If SINTA itself lists this exact article DOI, its article-level
        # accreditation label is the strongest available evidence.
        article_record = self._article_record_rank(profile_html, article_doi or "")
        if article_record is not None:
            result.match_method = "DOI"
            result.checked_for_year = self.normalize_year(article_year)
            result.is_ambiguous = bool(article_record.get("ambiguous"))
            result.ambiguity_reason = str(article_record.get("notes") or "")
            result.rank = article_record.get("rank")
            result.period_start = None
            result.period_end = None
            result.period_confirmed = False
            if result.is_ambiguous:
                result.status = "UNVERIFIED"
                result.lookup_status = "AMBIGUOUS"
                result.period_status = "PERIOD_CHECK_REQUIRED"
                result.notes = result.ambiguity_reason
            elif result.rank:
                result.status = result.rank
                result.lookup_status = "VERIFIED"
                result.period_status = "ARTICLE_RECORD"
                result.period_confirmed = True
                result.notes = (
                    f"DOI artikel ditemukan langsung pada profil resmi SINTA dengan label Accred: {result.rank}."
                )
            else:
                result.status = "UNVERIFIED"
                result.lookup_status = "RANK_UNAVAILABLE"
                result.period_status = "UNKNOWN"
                result.notes = str(article_record.get("notes") or "Peringkat artikel tidak tercantum pada profil resmi SINTA.")
            return result

        current_rank, ambiguous, reason, explicit_current = self._extract_rank_from_lines(
            profile_lines,
            require_current_label=True,
        )
        result.period_confirmed = False
        result.checked_for_year = self.normalize_year(article_year)
        result.match_method = "ISSN" if matched_issn else "JOURNAL_NAME"
        if ambiguous:
            result.rank = None
            result.status = "UNVERIFIED"
            result.lookup_status = "AMBIGUOUS"
            result.period_status = "PERIOD_CHECK_REQUIRED"
            result.is_ambiguous = True
            result.ambiguity_reason = reason
            result.notes = reason
        elif current_rank and explicit_current:
            result.rank = current_rank
            result.status = "UNVERIFIED"
            result.lookup_status = "PERIOD_UNVERIFIED"
            result.period_status = "PERIOD_CHECK_REQUIRED"
            result.is_ambiguous = False
            result.ambiguity_reason = ""
            result.notes = (
                f"Profil resmi SINTA menandai peringkat jurnal saat ini sebagai {current_rank}. "
                "Masa berlaku SK untuk tahun artikel tidak tercantum dengan jelas."
            )
        else:
            result.rank = None
            result.status = "UNVERIFIED"
            result.lookup_status = "RANK_UNAVAILABLE"
            result.period_status = "UNKNOWN"
            result.is_ambiguous = False
            result.ambiguity_reason = ""
            result.notes = reason or "Profil resmi tidak menampilkan satu peringkat terkini yang dapat dibaca."
        return self.evaluate_for_year(result, article_year)

    @staticmethod
    def _journal_name_from_profile(lines: List[str]) -> Optional[str]:
        for line in lines:
            candidate = line.strip()
            if (
                len(candidate) >= 6
                and not candidate.isdigit()
                and not re.search(
                    r"ISSN|SINTA|Accredited|Accred|Current|History|Google|Citation|Impact|Garuda|Scopus|H-Index",
                    candidate,
                    re.I,
                )
            ):
                return candidate
        return None

    def _verify_official(
        self,
        issn_list: Optional[List[str]] = None,
        journal_name: Optional[str] = None,
        article_year: Optional[int] = None,
        article_doi: Optional[str] = None,
    ) -> SintaVerificationResult:
        """Verifikasi identitas jurnal, lalu evaluasi masa berlaku per artikel."""
        clean_issns = list(dict.fromkeys(
            normalized
            for value in (issn_list or [])
            if (normalized := self.normalize_issn(value))
        ))
        clean_name = (journal_name or "").strip()
        article_year = self.normalize_year(article_year)

        if not clean_issns and not clean_name:
            return SintaVerificationResult(
                found=False,
                status="UNVERIFIED",
                lookup_status="UNVERIFIED",
                notes="Tidak ada ISSN atau nama jurnal yang tersedia untuk diverifikasi.",
                checked_for_year=article_year,
            )

        queries = [(value, "ISSN") for value in clean_issns]
        if clean_name:
            queries.append((clean_name, "JOURNAL_NAME"))
        exact_matches = []
        name_matches = []
        explicit_empty = []
        successful_queries = 0
        failed_queries = []
        unrecognized_response = False
        for query, query_kind in queries:
            if query_kind == "JOURNAL_NAME" and exact_matches:
                break
            try:
                search_html = self.search_sinta_directory(query)
            except Exception as exc:
                failed_queries.append(str(exc))
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if (
                    isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))
                    or status_code in {401, 403, 429}
                    or (status_code is not None and status_code >= 500)
                ):
                    unrecognized_response = True
                    break
                continue
            if not search_html:
                unrecognized_response = True
                continue
            successful_queries += 1
            result = self.parse_and_evaluate(
                html_text=search_html,
                issn_list=clean_issns,
                journal_name=clean_name,
                query_used=query,
            )
            if result.lookup_status == "FAILED":
                unrecognized_response = True
            elif result.lookup_status == "NOT_FOUND":
                explicit_empty.append(result)
            elif result.found and result.match_method == "ISSN":
                exact_matches.append(result)
            elif result.found:
                name_matches.append(result)

        if (
            not exact_matches
            and name_matches
            and clean_issns
            and (failed_queries or unrecognized_response)
        ):
            return SintaVerificationResult(
                found=False,
                status="UNVERIFIED",
                lookup_status="FAILED",
                journal_name=clean_name or None,
                checked_for_year=article_year,
                notes=(
                    "Pencarian ISSN belum berhasil diperiksa seluruhnya. "
                    "Kandidat nama tidak digunakan sebagai pengganti kecocokan ISSN yang gagal."
                ),
            )

        candidates = exact_matches or name_matches
        if candidates:
            unique = {}
            for result in candidates:
                key = (
                    self.normalize_name(result.journal_name),
                    tuple(sorted(self.normalize_issn(value) for value in result.all_issns)),
                    result.rank,
                    result.period_start,
                    result.period_end,
                )
                unique.setdefault(key, result)
            chosen = next(iter(unique.values()))
            if len(unique) > 1:
                chosen.is_ambiguous = True
                chosen.lookup_status = "AMBIGUOUS"
                chosen.status = "UNVERIFIED"
                chosen.period_status = "PERIOD_CHECK_REQUIRED"
                chosen.ambiguity_reason = (
                    "Beberapa entri jurnal berbeda cocok dengan metadata pencarian. "
                    "Periksa nama dan ISSN pada sumber resmi."
                )
                chosen.notes = chosen.ambiguity_reason
            return self._resolve_from_official_profile(
                chosen,
                clean_issns,
                article_year,
                article_doi=article_doi,
            )

        if queries and len(explicit_empty) == len(queries):
            result = explicit_empty[0]
            result.checked_for_year = article_year
            return result

        if not successful_queries or unrecognized_response:
            return SintaVerificationResult(
                found=False,
                status="UNVERIFIED",
                lookup_status="FAILED",
                journal_name=clean_name or None,
                checked_for_year=article_year,
                notes=(
                    f"Pemeriksaan direktori SINTA gagal: {failed_queries[-1]}"
                    if failed_queries
                    else "Respons SINTA kosong atau formatnya tidak dikenali. Status jurnal tidak disimpulkan."
                ),
            )
        return SintaVerificationResult(
            found=False,
            status="UNVERIFIED",
            lookup_status="NO_MATCH",
            journal_name=clean_name or None,
            checked_for_year=article_year,
            notes=(
                "Pencarian menghasilkan kandidat, tetapi identitas jurnal tidak cocok persis. "
                "Hasil ini tidak cukup untuk menyatakan jurnal tidak terdaftar."
            ),
        )

    @staticmethod
    def _gemini_api_key() -> Optional[str]:
        """Return the first configured Gemini key without exposing it to prompts."""
        try:
            from app.ai.gemini_explainer import get_saved_api_key

            raw_key = get_saved_api_key() or ""
        except Exception:
            raw_key = ""
        keys = [part.strip() for part in raw_key.replace(";", ",").split(",") if part.strip()]
        return keys[0] if keys else None

    def _find_gemini_profile_candidate(
        self,
        issn_list: List[str],
        journal_name: str,
    ) -> tuple[Optional[Dict[str, str]], str]:
        """Use grounded Gemini search only to locate a profile URL, never to assign a rank."""
        key = self._gemini_api_key()
        if not key:
            return None, "Bantuan Gemini belum aktif karena API key belum dikonfigurasi."

        normalized_issns = list(dict.fromkeys(
            self.normalize_issn(value) for value in (issn_list or [])
            if self.normalize_issn(value)
        ))
        cache_key = "|".join((";".join(normalized_issns), self.normalize_name(journal_name)))
        if cache_key in self._gemini_candidate_cache:
            cached = self._gemini_candidate_cache[cache_key]
            return cached, "" if cached else "Gemini belum menemukan profil resmi yang dapat dicocokkan."

        try:
            import os

            configured_model = (
                os.getenv("GEMINI_SINTA_MODEL", "").strip()
                or os.getenv("GEMINI_MODEL", "").strip()
            )
            default_model = "gemini-3.8-flash"
            models = [configured_model or default_model]
            if models[0] != default_model:
                models.append(default_model)
            prompt = (
                "Gunakan Google Search untuk menemukan halaman profil jurnal yang tepat di direktori resmi SINTA Kemdiktisaintek. "
                "Cari berdasarkan ISSN persis terlebih dahulu, lalu nama jurnal. Metadata yang dicari adalah data bibliografis publik. "
                "Kembalikan URL profil hanya jika merupakan halaman https://sinta.kemdiktisaintek.go.id/journals/profile/ID/ "
                "yang terlihat relevan dengan ISSN atau nama jurnal. Jangan menebak URL, ISSN, atau nama jurnal. "
                "JANGAN memberikan, memperkirakan, atau menyimpulkan peringkat SINTA. Jika tidak ada profil yang jelas, "
                "kembalikan profile_url kosong. Ringkas bukti pencocokan secara singkat.\n\n"
                f"Nama jurnal: {journal_name or '(tidak tersedia)'}\n"
                f"ISSN: {', '.join(self.format_issn(value) for value in normalized_issns) or '(tidak tersedia)'}"
            )
            schema = {
                "type": "OBJECT",
                "properties": {
                    "profile_url": {"type": "STRING"},
                    "journal_name": {"type": "STRING"},
                    "matched_issn": {"type": "STRING"},
                    "evidence": {"type": "STRING"},
                },
                "required": ["profile_url", "journal_name", "matched_issn", "evidence"],
                "propertyOrdering": ["profile_url", "journal_name", "matched_issn", "evidence"],
            }
            response = None
            for model in models:
                response = httpx.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "tools": [{"googleSearch": {}}],
                        "generationConfig": {
                            "temperature": 0,
                            "maxOutputTokens": 500,
                            "responseMimeType": "application/json",
                            "responseSchema": schema,
                            **({"thinkingConfig": {"thinkingLevel": "low"}} if model.startswith("gemini-3") else {}),
                        },
                    },
                    timeout=httpx.Timeout(16.0, connect=5.0),
                )
                if response.status_code in {400, 404} and model != models[-1]:
                    continue
                response.raise_for_status()
                break
            if response is None:
                return None, "Gemini belum dapat menemukan profil; pemeriksaan dapat dicoba kembali."
            payload = response.json()
            parts = (
                payload.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            response_text = "\n".join(
                str(part.get("text") or "") for part in parts if isinstance(part, dict)
            ).strip()
            response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text, flags=re.I)
            parsed = json.loads(response_text)
            url = str(parsed.get("profile_url") or "").strip()
            if not re.fullmatch(
                r"https://sinta\.kemdiktisaintek\.go\.id/journals/profile/\d+/?",
                url,
            ):
                self._gemini_candidate_cache[cache_key] = None
                return None, "Gemini belum menemukan profil resmi yang dapat dicocokkan."
            candidate = {
                "profile_url": url,
                "journal_name": str(parsed.get("journal_name") or journal_name).strip(),
                "matched_issn": self.normalize_issn(parsed.get("matched_issn")),
                "evidence": str(parsed.get("evidence") or "").strip()[:300],
            }
            self._gemini_candidate_cache[cache_key] = candidate
            return candidate, ""
        except Exception:
            self._gemini_candidate_cache[cache_key] = None
            return None, "Gemini belum dapat menemukan profil; pemeriksaan dapat dicoba kembali."

    def verify(
        self,
        issn_list: Optional[List[str]] = None,
        journal_name: Optional[str] = None,
        article_year: Optional[int] = None,
        article_doi: Optional[str] = None,
    ) -> SintaVerificationResult:
        """Verify against SINTA; Gemini may locate a candidate but never supplies a rank."""
        clean_issns = list(dict.fromkeys(
            normalized
            for value in (issn_list or [])
            if (normalized := self.normalize_issn(value))
        ))
        clean_name = (journal_name or "").strip()
        year = self.normalize_year(article_year)
        result = self._verify_official(
            issn_list=clean_issns,
            journal_name=clean_name,
            article_year=year,
            article_doi=article_doi,
        )
        if result.lookup_status not in {"FAILED", "NO_MATCH", "NOT_FOUND", "RANK_UNAVAILABLE"}:
            return result
        if (
            result.lookup_status == "RANK_UNAVAILABLE"
            and result.found
            and re.fullmatch(
                r"https://sinta\.kemdiktisaintek\.go\.id/journals/profile/\d+/?",
                result.source_url or "",
            )
        ):
            # The exact official page is already loaded; Gemini cannot improve
            # a rank that SINTA itself does not expose in a readable form.
            return result

        candidate, helper_note = self._find_gemini_profile_candidate(clean_issns, clean_name)
        if not candidate:
            result.assistive_note = helper_note
            return result

        # Gemini's proposed profile is fetched from SINTA itself and must match
        # the supplied ISSN, or the exact journal name when no ISSN is present.
        candidate_result = SintaVerificationResult(
            found=False,
            status="UNVERIFIED",
            lookup_status="UNVERIFIED",
            journal_name=candidate.get("journal_name") or clean_name or None,
            source_url=candidate["profile_url"],
            match_method="ISSN" if clean_issns else "JOURNAL_NAME",
            checked_for_year=year,
            notes="Profil ditemukan sebagai kandidat dengan bantuan pencarian Gemini.",
        )
        try:
            official_result = self._resolve_from_official_profile(
                candidate_result,
                clean_issns,
                year,
                article_doi=article_doi,
            )
        except Exception:
            official_result = candidate_result
            official_result.found = False
            official_result.rank = None
            official_result.status = "UNVERIFIED"
            official_result.lookup_status = "FAILED"

        if not official_result.found:
            result.assistive_candidate_url = candidate["profile_url"]
            result.assistive_note = (
                "Gemini menemukan halaman kandidat, tetapi ISSN atau nama jurnal belum cocok "
                "dengan profil resmi. Status peringkat tidak diubah."
            )
            return result

        official_result.assistive_candidate_url = candidate["profile_url"]
        official_result.assistive_note = (
            "Gemini membantu menemukan profil; kecocokan dan status diambil ulang dari halaman resmi SINTA."
        )
        return official_result

    def evaluate_article_doi(
        self,
        journal_result: SintaVerificationResult | Dict[str, Any],
        article_doi: Optional[str],
        article_year: Any,
    ) -> SintaVerificationResult:
        """Apply a journal-profile result to one exact article DOI, if listed."""
        result = (
            SintaVerificationResult.from_dict(journal_result)
            if isinstance(journal_result, dict)
            else journal_result
        )
        if not article_doi or not result.source_url:
            return self.evaluate_for_year(result, article_year)
        profile_url = result.source_url.rstrip("/") + "/"
        if profile_url not in self._profile_cache:
            result.notes = (
                (result.notes + " ")
                + "Profil resmi tidak berhasil dimuat; DOI artikel belum dapat diverifikasi secara langsung."
            ).strip()
            return self.evaluate_for_year(result, article_year)
        issns = result.all_issns or ([result.issn_matched] if result.issn_matched else [])
        return self._resolve_from_official_profile(
            result,
            issns,
            article_year,
            article_doi=article_doi,
        )

    def parse_and_evaluate(
        self,
        html_text: str,
        issn_list: List[str],
        journal_name: str,
        article_year: Optional[int] = None,
        query_used: str = "",
    ) -> SintaVerificationResult:
        """Mem-parsing hasil HTML direktori SINTA dan mengevaluasi masa berlaku akreditasi."""
        article_year = self.normalize_year(article_year)
        parser = _SintaDirectoryHtmlParser()
        parser.feed(html_text)
        entries = parser.get_records()
        issn_fallback_entries = parser.get_records_for_issns(issn_list or [])
        structured_exact = any(
            any(
                self.normalize_issn(value) in {
                    self.normalize_issn(target) for target in (issn_list or [])
                }
                for value in re.findall(
                    r"(?:P-ISSN|E-ISSN|ISSN)\s*[:#]?\s*([0-9Xx-]+)",
                    " ".join(entry),
                    re.I,
                )
            )
            for entry in entries
        )
        # Keep the structured journal record intact when it contains the exact
        # ISSN. Splitting it merely because it contains multiple rank mentions
        # can detach the current accreditation from the journal profile and
        # select an unrelated historical/article-level rank.
        used_issn_fallback = bool(issn_fallback_entries) and not structured_exact
        if used_issn_fallback:
            entries = issn_fallback_entries

        source_url = f"{self.BASE_URL}?q={quote_plus(query_used)}" if query_used else self.BASE_URL
        match_method = "NONE"
        is_ambiguous = False
        ambiguity_reason = ""

        clean_target_issns = [self.normalize_issn(i) for i in (issn_list or []) if self.normalize_issn(i)]
        if not entries:
            page_text = " ".join(parser.get_lines())
            if re.search(
                r"no\s+(?:journals?|results?)\s+(?:were\s+)?found|"
                r"0\s+(?:journals?|results?)|tidak ditemukan|tidak ada jurnal",
                page_text,
                re.I,
            ):
                return SintaVerificationResult(
                    found=False,
                    status="NOT_IN_SINTA",
                    lookup_status="NOT_FOUND",
                    journal_name=journal_name,
                    source_url=source_url,
                    checked_for_year=article_year,
                    notes="Direktori SINTA menyatakan tidak ada entri yang cocok dengan pencarian.",
                )
            return SintaVerificationResult(
                found=False,
                status="UNVERIFIED",
                lookup_status="FAILED",
                journal_name=journal_name,
                source_url=source_url,
                checked_for_year=article_year,
                notes="Struktur hasil SINTA tidak dapat dipisahkan per jurnal; peringkat tidak ditetapkan.",
            )

        matched_entries = []
        partial_entries = []
        target_norm = self.normalize_name(journal_name)
        for entry in entries:
            entry_text = " ".join(entry)
            entry_issns = [
                self.normalize_issn(value)
                for value in re.findall(r"(?:P-ISSN|E-ISSN|ISSN)\s*[:#]?\s*([0-9Xx-]+)", entry_text, re.I)
            ]
            matched_issn = next((value for value in clean_target_issns if value in entry_issns), None)
            if matched_issn:
                matched_entries.append((entry, matched_issn, "ISSN"))
                continue
            if target_norm:
                name_lines = [
                    self.normalize_name(line)
                    for line in entry
                    if len(self.normalize_name(line)) >= 4
                    and not re.search(r"ISSN|SINTA|Accredited|Akreditasi|Scopus|Garuda", line, re.I)
                ]
                if target_norm in name_lines:
                    matched_entries.append((entry, None, "JOURNAL_NAME"))
                elif any(
                    len(target_norm) >= 8
                    and len(value) >= 8
                    and (target_norm in value or value in target_norm)
                    for value in name_lines
                ):
                    partial_entries.append(entry)

        if matched_entries:
            exact_issn_matches = [item for item in matched_entries if item[2] == "ISSN"]
            selected_matches = exact_issn_matches or matched_entries
            context_lines, matched_issn, match_method = selected_matches[0]
            if len(selected_matches) > 1:
                is_ambiguous = True
                ambiguity_reason = "Metadata cocok dengan beberapa entri jurnal SINTA."
        elif partial_entries:
            context_lines = partial_entries[0]
            matched_issn = None
            match_method = "JOURNAL_NAME"
            is_ambiguous = True
            ambiguity_reason = (
                f"Kecocokan nama jurnal memiliki ambiguitas dan hanya sebagian ('{journal_name}'); "
                "periksa entri resmi secara manual."
            )
        else:
            return SintaVerificationResult(
                found=False,
                status="UNVERIFIED",
                lookup_status="NO_MATCH",
                journal_name=journal_name,
                source_url=source_url,
                checked_for_year=article_year,
                notes=(
                    "Respons berisi entri lain, tetapi tidak ada kecocokan jurnal yang cukup kuat. "
                    "Jurnal tidak dinyatakan tidak terdaftar."
                ),
            )

        context_text = " ".join(context_lines)

        detected_rank, rank_ambiguous, rank_reason, _ = self._extract_rank_from_lines(context_lines)
        if rank_ambiguous:
            is_ambiguous = True
            ambiguity_reason = rank_reason

        # 2. Nama jurnal di direktori (biasanya berada sebelum baris ISSN atau di judul entri)
        found_journal_name = journal_name
        if not used_issn_fallback:
            for line in context_lines:
                if not re.search(
                    r"(?:ISSN|SINTA|Accredited|Garuda|Google|Scopus|H-Index|Citations|"
                    r"masa berlaku|periode|akreditasi|\bSK\b|current accreditation)",
                    line,
                    re.I,
                ):
                    if len(line) > 5 and not line.isdigit():
                        found_journal_name = line
                        break

        # 3. Semua ISSN yang ada di entri
        all_issns_found = re.findall(r"(?:P-ISSN|E-ISSN|ISSN)\s*[:#]?\s*([0-9Xx-]+)", context_text, re.I)
        formatted_issns = list(dict.fromkeys(self.format_issn(i) for i in all_issns_found if i))

        # 4. Deteksi periode akreditasi (contoh: "2020 - 2025" atau "Vol. 1 No. 1 (2019) - Vol. 5 No. 2 (2024)")
        # Only accept a year range explicitly associated with accreditation.
        period_match = re.search(
            r"(?:masa\s+berlaku|periode(?:\s+akreditasi)?|akreditasi\s+berlaku|"
            r"valid(?:ity)?\s+period|accreditation\s+period|sk\s+akreditasi)"
            r"[^0-9]{0,80}(20\d{2})\s*(?:-|\u2013|\u2014|sampai|hingga|to|s\.d\.)\s*(20\d{2})",
            context_text,
            re.I,
        )
        period_start = int(period_match.group(1)) if period_match else None
        period_end = int(period_match.group(2)) if period_match else None

        # Evaluasi masa berlaku akreditasi terhadap tahun terbit artikel
        period_status = "UNKNOWN"
        final_status = "UNVERIFIED"
        notes = ""

        if not detected_rank:
            explicitly_unaccredited = bool(re.search(r"belum\s+terakreditasi|not\s+accredited|tidak\s+terakreditasi", context_text, re.I))
            if is_ambiguous:
                final_status = "UNVERIFIED"
                period_status = "PERIOD_CHECK_REQUIRED"
                notes = ambiguity_reason
            else:
                final_status = "NOT_ACCREDITED" if explicitly_unaccredited else "UNVERIFIED"
                period_status = "NOT_APPLICABLE" if explicitly_unaccredited else "UNKNOWN"
                notes = (
                    "Entri menyatakan jurnal belum terakreditasi."
                    if explicitly_unaccredited
                    else "Jurnal ditemukan, tetapi peringkat akreditasi tidak dapat dipastikan dari entri ini."
                )
        else:
            # Jurnal memiliki peringkat SINTA
            if is_ambiguous:
                final_status = "UNVERIFIED"
                period_status = "PERIOD_CHECK_REQUIRED"
                notes = "Kandidat jurnal ambigu; cocokkan ISSN dan masa berlaku pada sumber resmi sebelum menetapkan peringkat."
            elif article_year and period_start and period_end:
                if period_start <= article_year <= period_end:
                    final_status = detected_rank
                    period_status = "VALID"
                    notes = (
                        f"Jurnal terakreditasi {detected_rank}. Tahun terbit artikel ({article_year}) "
                        f"berada dalam masa berlaku akreditasi ({period_start}-{period_end})."
                    )
                else:
                    final_status = "NOT_ACCREDITED"
                    period_status = "EXPIRED"
                    notes = (
                        f"Peringkat direktori saat ini adalah {detected_rank}, namun tahun terbit artikel "
                        f"({article_year}) berada di luar masa berlaku akreditasi yang tertera ({period_start}-{period_end})."
                    )
            elif article_year:
                # Periode tidak bisa dipastikan secara pasti dari listing pencarian
                final_status = "UNVERIFIED"
                period_status = "PERIOD_CHECK_REQUIRED"
                notes = (
                    f"Jurnal terdaftar dengan peringkat {detected_rank} di SINTA. Namun, masa berlaku SK akreditasi "
                    f"untuk tahun terbit artikel ({article_year}) perlu diverifikasi secara manual oleh pengguna sebelum menetapkan peringkat."
                )
            else:
                final_status = "UNVERIFIED"
                period_status = "PERIOD_CHECK_REQUIRED"
                notes = (
                    f"Jurnal terdaftar dengan peringkat {detected_rank} di SINTA. Tahun publikasi artikel tidak diketahui; "
                    "perlu konfirmasi manual pengguna."
                )

        result = SintaVerificationResult(
            found=True,
            rank=detected_rank,
            status=final_status,
            lookup_status=(
                "AMBIGUOUS" if is_ambiguous
                else "VERIFIED" if not detected_rank and final_status == "NOT_ACCREDITED"
                else "PERIOD_UNVERIFIED" if detected_rank
                else "UNVERIFIED"
            ),
            journal_name=found_journal_name or journal_name,
            issn_matched=self.format_issn(matched_issn) if matched_issn else None,
            all_issns=formatted_issns,
            source_url=self._profile_url_for_issn(html_text, matched_issn) or source_url,
            verification_source="Direktori Resmi SINTA (sinta.kemdiktisaintek.go.id)",
            verification_date=time.strftime("%Y-%m-%d %H:%M"),
            match_method=match_method,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            period_start=period_start,
            period_end=period_end,
            period_status=period_status,
            period_confirmed=False,
            checked_for_year=article_year,
            notes=notes,
        )
        return self.evaluate_for_year(result, article_year)

    @classmethod
    def evaluate_for_year(
        cls,
        result: SintaVerificationResult | Dict[str, Any],
        article_year: Any,
    ) -> SintaVerificationResult:
        """Gunakan hasil identitas jurnal yang sama untuk tahun tiap artikel."""
        if isinstance(result, dict):
            result = SintaVerificationResult.from_dict(result)
        year = cls.normalize_year(article_year)
        result.checked_for_year = year
        if not result.found:
            return result
        if result.is_ambiguous or result.lookup_status == "AMBIGUOUS":
            result.status = "UNVERIFIED"
            result.lookup_status = "AMBIGUOUS"
            result.period_status = "PERIOD_CHECK_REQUIRED"
            return result
        if result.status == "NOT_ACCREDITED" and not result.rank:
            result.lookup_status = "VERIFIED"
            result.period_status = "NOT_APPLICABLE"
            return result
        if not result.rank:
            result.status = "UNVERIFIED"
            result.lookup_status = "RANK_UNAVAILABLE"
            result.period_status = "UNKNOWN"
            return result
        if year is None:
            result.status = "UNVERIFIED"
            result.lookup_status = "PERIOD_UNVERIFIED"
            result.period_status = "PERIOD_CHECK_REQUIRED"
            result.notes = (
                f"Peringkat terkini yang ditemukan adalah {result.rank}. "
                "Tahun artikel tidak tersedia untuk mengevaluasi masa berlaku SK."
            )
            return result
        if result.period_start is None or result.period_end is None:
            result.status = "UNVERIFIED"
            result.lookup_status = "PERIOD_UNVERIFIED"
            result.period_status = "PERIOD_CHECK_REQUIRED"
            result.notes = (
                f"Peringkat terkini yang ditemukan adalah {result.rank}; "
                f"masa berlaku SK untuk tahun {year} belum dapat dipastikan."
            )
            return result
        if result.period_start <= year <= result.period_end:
            result.status = result.rank
            result.lookup_status = "VERIFIED"
            result.period_status = "VALID"
            result.notes = (
                f"Peringkat {result.rank} tertera pada entri jurnal yang cocok. "
                f"Rentang SK {result.period_start}-{result.period_end} mencakup tahun artikel {year}."
            )
            return result

        result.status = "UNVERIFIED"
        result.lookup_status = "PERIOD_UNVERIFIED"
        result.period_status = "EXPIRED"
        result.notes = (
            f"Peringkat terkini adalah {result.rank} dengan rentang SK yang tampil "
            f"{result.period_start}-{result.period_end}; tahun artikel {year} berada di luar rentang itu. "
            "Periksa SK historis. Ini tidak membuktikan jurnal tidak terakreditasi pada tahun tersebut."
        )
        return result
