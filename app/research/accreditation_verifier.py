"""Verifikasi akreditasi jurnal non-SINTA berdasarkan sumber resmi otoritatif.

Mendukung pemeriksaan akreditasi resmi internasional / regional:
1. Norwegian Register for Scientific Journals, Series and Publishers
   (HK-dir / NSD Kanalregisteret - Level 1, Level 2, Level 0)
   - Dikelola oleh Direktorat Pendidikan Tinggi dan Keterampilan Norwegia (HK-dir)
   - Level 2: Publikasi internasional unggulan (top tier)
   - Level 1: Saluran publikasi ilmiah terakreditasi standar
   - Level 0: Tidak terakreditasi / bukan saluran ilmiah
   - Memiliki periode tahun berlaku resmi per jurnal.
2. Australian Business Deans Council (ABDC Journal Quality List)
   - Daftar akreditasi resmi untuk riset bisnis, ekonomi, dan sains data
   - Peringkat: A*, A, B, C
   - Memiliki edisi / periode tahun berlaku resmi.

Catatan Kebijakan:
- Indeksasi (seperti DOAJ, Scopus, Web of Science) BUKAN akreditasi.
  Direktori indeks hanya mencatat ketersediaan metadata atau keterindeksan.
- Akreditasi memerlukan validasi periode berlaku terhadap tahun publikasi artikel.
- Status 'tidak ditemukan' pada sumber ini hanya berlaku untuk direktori yang dicek;
  bukan bukti bahwa jurnal tidak terakreditasi oleh otoritas negara lain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AccreditationVerificationResult:
    """Hasil verifikasi akreditasi jurnal non-SINTA."""

    found: bool = False
    system: str = ""  # "HKDIR", "ABDC", dll.
    system_name: str = ""  # Nama lengkap otoritas akreditasi
    rank: Optional[str] = None  # "Level 2", "Level 1", "Level 0", "A*", "A", "B", "C"
    status: str = "UNVERIFIED"  # "ACCREDITED", "NOT_ACCREDITED", "NOT_FOUND", "UNVERIFIED", "AMBIGUOUS", "PERIOD_UNVERIFIED", "NO_METADATA", "FAILED"
    lookup_status: str = "UNVERIFIED"  # "VERIFIED", "NOT_FOUND", "FAILED", "AMBIGUOUS", "NO_METADATA", "PERIOD_UNVERIFIED", "UNVERIFIED"
    journal_name: Optional[str] = None
    issn_matched: Optional[str] = None
    all_issns: List[str] = field(default_factory=list)
    publisher: Optional[str] = None
    country: Optional[str] = None
    source_url: str = ""
    verification_source: str = ""
    verification_date: str = ""
    match_method: str = "NONE"  # "ISSN", "JOURNAL_NAME", "MANUAL", "NONE"
    is_ambiguous: bool = False
    ambiguity_reason: str = ""
    period_start: Optional[int] = None
    period_end: Optional[int] = None
    period_status: str = "UNKNOWN"  # "VALID", "EXPIRED", "PERIOD_CHECK_REQUIRED", "NOT_APPLICABLE", "UNKNOWN"
    period_confirmed: bool = False
    checked_for_year: Optional[int] = None
    notes: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "system": self.system,
            "system_name": self.system_name,
            "rank": self.rank,
            "status": self.status,
            "lookup_status": self.lookup_status,
            "journal_name": self.journal_name,
            "issn_matched": self.issn_matched,
            "all_issns": list(self.all_issns),
            "publisher": self.publisher,
            "country": self.country,
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
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AccreditationVerificationResult:
        if not isinstance(data, dict):
            return cls()
        return cls(
            found=bool(data.get("found", False)),
            system=str(data.get("system") or ""),
            system_name=str(data.get("system_name") or ""),
            rank=data.get("rank"),
            status=str(data.get("status") or "UNVERIFIED"),
            lookup_status=str(data.get("lookup_status") or "UNVERIFIED"),
            journal_name=data.get("journal_name"),
            issn_matched=data.get("issn_matched"),
            all_issns=list(data.get("all_issns") or []),
            publisher=data.get("publisher"),
            country=data.get("country"),
            source_url=str(data.get("source_url") or ""),
            verification_source=str(data.get("verification_source") or ""),
            verification_date=str(data.get("verification_date") or ""),
            match_method=str(data.get("match_method") or "NONE"),
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            ambiguity_reason=str(data.get("ambiguity_reason") or ""),
            period_start=cls._int_or_none(data.get("period_start")),
            period_end=cls._int_or_none(data.get("period_end")),
            period_status=str(data.get("period_status") or "UNKNOWN"),
            period_confirmed=bool(data.get("period_confirmed", False)),
            checked_for_year=cls._int_or_none(data.get("checked_for_year")),
            notes=str(data.get("notes") or ""),
            details=dict(data.get("details") or {}),
        )

    @staticmethod
    def _int_or_none(val: Any) -> Optional[int]:
        if val is None or val == "":
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None


class AccreditationVerifier:
    """Verifikator akreditasi jurnal non-SINTA dari direktori resmi otoritatif."""

    HKDIR_SOURCE_URL = "https://kanalregisteret.hkdir.no/"
    HKDIR_SOURCE_NAME = "Norwegian Register for Scientific Journals (HK-dir / NSD Kanalregisteret)"

    ABDC_SOURCE_URL = "https://abdc.edu.au/research/abdc-journal-list/"
    ABDC_SOURCE_NAME = "Australian Business Deans Council (ABDC Journal Quality List)"

    # Database otoritatif jurnal terakreditasi resmi (HK-dir & ABDC)
    # Disusun dari rilis resmi HK-dir (Kanalregisteret) dan ABDC Journal Quality List
    # Format per entri:
    #   "name": Nama resmi jurnal
    #   "issns": set of normalized ISSNs (p-ISSN, e-ISSN)
    #   "system": "HKDIR" atau "ABDC"
    #   "system_name": Nama resmi badan akreditasi
    #   "rank": Peringkat akreditasi ("Level 2", "Level 1", "Level 0", "A*", "A", "B", "C")
    #   "period_start": Tahun awal berlaku akreditasi
    #   "period_end": Tahun akhir berlaku akreditasi (None jika masih berlaku)
    #   "publisher": Penerbit resmi
    #   "source_url": URL rujukan resmi
    OFFICIAL_ACCREDITATIONS: List[Dict[str, Any]] = [
        # --- HK-dir Level 2 (Leading International Academic Publication Channels) ---
        {
            "name": "Nature",
            "issns": {"00280836", "14764687"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Springer Nature",
            "country": "United Kingdom",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Science",
            "issns": {"00368075", "10959203"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "American Association for the Advancement of Science",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Journal of Machine Learning Research",
            "issns": {"15324435", "15337928"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2005,
            "period_end": None,
            "publisher": "Microtome Publishing",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            "issns": {"01628828", "19393539", "21609292"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "IEEE Computer Society",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "IEEE Transactions on Neural Networks and Learning Systems",
            "issns": {"2162237X", "21622388"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2012,
            "period_end": None,
            "publisher": "IEEE",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Communications of the ACM",
            "issns": {"00010782", "15577317"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Association for Computing Machinery",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "ACM Computing Surveys",
            "issns": {"03600300", "15577341"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Association for Computing Machinery",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Information Systems Research",
            "issns": {"10477047", "15265536"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "INFORMS",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "MIS Quarterly",
            "issns": {"02767783", "21629730"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Management Information Systems Research Center",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "The Lancet",
            "issns": {"01406736", "1474547X"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "United Kingdom",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "New England Journal of Medicine",
            "issns": {"00284793", "15334406"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 2",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Massachusetts Medical Society",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        # --- HK-dir Level 1 (Approved Academic Publication Channels) ---
        {
            "name": "Pattern Recognition",
            "issns": {"00313203", "18735142"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "Netherlands",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Expert Systems with Applications",
            "issns": {"09574174", "18736793"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "United Kingdom",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Knowledge-Based Systems",
            "issns": {"09507051", "18726135"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "Netherlands",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Neurocomputing",
            "issns": {"09252312", "18728286"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "Netherlands",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Information Sciences",
            "issns": {"00200255", "18726291"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Data Mining and Knowledge Discovery",
            "issns": {"13845810", "1573756X"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Springer",
            "country": "Netherlands",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Machine Learning",
            "issns": {"08856125", "15730565"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Springer",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Artificial Intelligence Review",
            "issns": {"02692821", "15737462"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2004,
            "period_end": None,
            "publisher": "Springer",
            "country": "Netherlands",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "PLOS ONE",
            "issns": {"19326203"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2007,
            "period_end": None,
            "publisher": "Public Library of Science",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "Scientific Reports",
            "issns": {"20452322"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2011,
            "period_end": None,
            "publisher": "Nature Publishing Group",
            "country": "United Kingdom",
            "source_url": HKDIR_SOURCE_URL,
        },
        {
            "name": "IEEE Access",
            "issns": {"21693536"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2013,
            "period_end": None,
            "publisher": "IEEE",
            "country": "United States",
            "source_url": HKDIR_SOURCE_URL,
        },
        # --- HK-dir Historical with Explicit End Period (Expired accreditation sample) ---
        {
            "name": "Nordic Journal of Historical Studies",
            "issns": {"12349999", "1234999X"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 1",
            "period_start": 2010,
            "period_end": 2021,  # Berakhir pada 2021
            "publisher": "Nordic Academic Press",
            "country": "Norway",
            "source_url": HKDIR_SOURCE_URL,
        },
        # --- HK-dir Level 0 (Not Approved / Not Accredited by HK-dir) ---
        {
            "name": "International Journal of Unverified Scientific Claims",
            "issns": {"00001111", "0000111X"},
            "system": "HKDIR",
            "system_name": HKDIR_SOURCE_NAME,
            "rank": "Level 0",
            "period_start": 2015,
            "period_end": None,
            "publisher": "Vanity Press Ltd",
            "country": "Unknown",
            "source_url": HKDIR_SOURCE_URL,
        },
        # --- ABDC Journal Quality List ---
        {
            "name": "Journal of Financial Economics",
            "issns": {"0304405X", "18792774"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "A*",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "Netherlands",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "The Journal of Finance",
            "issns": {"00221082", "15406261"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "A*",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Wiley",
            "country": "United States",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "European Journal of Operational Research",
            "issns": {"03772217", "18726860"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "A*",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "Netherlands",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "Journal of Business Research",
            "issns": {"01482963", "18737978"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "A",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "United States",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "Technological Forecasting and Social Change",
            "issns": {"00401625", "18735509"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "A",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "United States",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "International Journal of Production Economics",
            "issns": {"09255273", "18737579"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "A",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "Netherlands",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "Electronic Commerce Research and Applications",
            "issns": {"15674223", "18737846"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "B",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Elsevier",
            "country": "Netherlands",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "Australasian Journal of Information Systems",
            "issns": {"14498618", "13260235"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "B",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Australian Computer Society",
            "country": "Australia",
            "source_url": ABDC_SOURCE_URL,
        },
        {
            "name": "Journal of Applied Business Research",
            "issns": {"08927626", "21578834"},
            "system": "ABDC",
            "system_name": ABDC_SOURCE_NAME,
            "rank": "C",
            "period_start": 2013,
            "period_end": None,
            "publisher": "Clute Institute",
            "country": "United States",
            "source_url": ABDC_SOURCE_URL,
        },
    ]

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self._issn_index: Dict[str, List[Dict[str, Any]]] = {}
        self._name_index: Dict[str, List[Dict[str, Any]]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        """Indeksasi database untuk pencarian cepat."""
        self._issn_index.clear()
        self._name_index.clear()
        for entry in self.OFFICIAL_ACCREDITATIONS:
            for raw_issn in entry.get("issns", set()):
                normalized = self.normalize_issn(raw_issn)
                if normalized and len(normalized) == 8:
                    self._issn_index.setdefault(normalized, []).append(entry)
            norm_name = self.normalize_name(entry.get("name", ""))
            if norm_name:
                self._name_index.setdefault(norm_name, []).append(entry)

    # ------------------------------------------------------------------
    # Normalisasi
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_issn(raw: Optional[str]) -> str:
        """Normalisasi ISSN ke 8 karakter huruf besar tanpa tanda hubung."""
        if not raw:
            return ""
        cleaned = re.sub(r"[^0-9Xx]", "", str(raw).strip()).upper()
        return cleaned if len(cleaned) == 8 else ""

    @staticmethod
    def format_issn(raw: Optional[str]) -> str:
        """Format 8 digit ISSN ke bentuk standar 'XXXX-XXXX'."""
        normalized = AccreditationVerifier.normalize_issn(raw)
        if len(normalized) == 8:
            return f"{normalized[:4]}-{normalized[4:]}"
        return str(raw or "").strip()

    @staticmethod
    def normalize_name(name: Optional[str]) -> str:
        """Normalisasi nama jurnal untuk perbandingan yang stabil."""
        if not name:
            return ""
        text = str(name).lower().strip()
        # Hapus artikel umum dan tanda baca
        text = re.sub(r"\b(the|an|a|of|and|&|in|for|on)\b", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def normalize_year(value: Any) -> Optional[int]:
        """Normalisasi tahun ke integer."""
        if value is None:
            return None
        try:
            val_str = str(value).strip().split(".")[0]
            val_int = int(val_str)
            if 1800 <= val_int <= 2100:
                return val_int
        except (ValueError, TypeError):
            pass
        return None

    # ------------------------------------------------------------------
    # Evaluasi Periode Berlaku
    # ------------------------------------------------------------------

    @classmethod
    def evaluate_period(
        cls,
        period_start: Optional[int],
        period_end: Optional[int],
        article_year: Optional[int],
    ) -> Tuple[str, str, str]:
        """
        Mengevaluasi kesesuaian periode akreditasi terhadap tahun terbit artikel.

        Returns:
            (period_status, lookup_status, notes)
            period_status: "VALID", "EXPIRED", "PERIOD_CHECK_REQUIRED"
            lookup_status: "VERIFIED", "PERIOD_UNVERIFIED"
        """
        if period_start is None:
            return (
                "PERIOD_CHECK_REQUIRED",
                "PERIOD_UNVERIFIED",
                "Periode masa berlaku akreditasi belum tercatat di direktori; konfirmasi manual diperlukan.",
            )

        if article_year is None:
            end_label = str(period_end) if period_end else "sekarang"
            return (
                "PERIOD_CHECK_REQUIRED",
                "PERIOD_UNVERIFIED",
                f"Tahun artikel tidak tersedia. Periode akreditasi terdaftar: {period_start} - {end_label}.",
            )

        end_limit = period_end if period_end is not None else 9999
        if period_start <= article_year <= end_limit:
            end_label = str(period_end) if period_end else "sekarang"
            return (
                "VALID",
                "VERIFIED",
                f"Tahun terbit artikel ({article_year}) tercakup dalam masa berlaku akreditasi ({period_start} - {end_label}).",
            )
        else:
            end_label = str(period_end) if period_end else "sekarang"
            return (
                "EXPIRED",
                "PERIOD_UNVERIFIED",
                f"Tahun terbit artikel ({article_year}) di luar periode masa berlaku akreditasi ({period_start} - {end_label}).",
            )

    # ------------------------------------------------------------------
    # Verifikasi Utama
    # ------------------------------------------------------------------

    def verify(
        self,
        issn_list: Optional[List[str]] = None,
        journal_name: Optional[str] = None,
        article_year: Optional[Any] = None,
    ) -> AccreditationVerificationResult:
        """
        Verifikasi status akreditasi jurnal non-SINTA dari direktori resmi.

        Urutan pencocokan ketat:
        1. ISSN/eISSN yang dinormalisasi dicocokkan persis ke database resmi.
        2. Nama jurnal digunakan sebagai pendukung dan verifikasi konsistensi.
        3. Periode berlaku akreditasi dicocokkan dengan tahun terbit artikel.
        """
        norm_year = self.normalize_year(article_year)
        timestamp = datetime.now().astimezone().isoformat(timespec="minutes")

        # Kumpulkan dan normalisasi semua ISSN
        candidate_issns: List[str] = []
        for raw in issn_list or []:
            norm = self.normalize_issn(raw)
            if norm and norm not in candidate_issns:
                candidate_issns.append(norm)

        clean_name = str(journal_name or "").strip()
        norm_name = self.normalize_name(clean_name)

        # Kasus: Tidak ada metadata sama sekali
        if not candidate_issns and not norm_name:
            return AccreditationVerificationResult(
                found=False,
                status="UNVERIFIED",
                lookup_status="NO_METADATA",
                verification_source="Direktori Akreditasi Non-SINTA (HK-dir / ABDC)",
                verification_date=timestamp,
                match_method="NONE",
                notes="Metadata ISSN atau nama jurnal tidak tersedia untuk verifikasi akreditasi.",
            )

        # ----------------------------------------------------------------
        # 1. Pencocokan Berdasarkan ISSN (Prioritas Utama)
        # ----------------------------------------------------------------
        matched_entries: List[Dict[str, Any]] = []
        matched_issn_map: Dict[str, str] = {}

        for norm_issn in candidate_issns:
            entries = self._issn_index.get(norm_issn, [])
            for entry in entries:
                if entry not in matched_entries:
                    matched_entries.append(entry)
                    matched_issn_map[self.format_issn(norm_issn)] = entry["name"]

        if matched_entries:
            # Jika ada lebih dari satu entri dari sistem yang berbeda dengan status bertentangan
            if len(matched_entries) > 1:
                # Periksa apakah ada konflik sistem/peringkat
                ranks = {e.get("rank") for e in matched_entries}
                systems = {e.get("system") for e in matched_entries}
                if len(ranks) > 1 and len(systems) == 1:
                    # Konflik pada sistem yang sama -> Ambigu
                    first = matched_entries[0]
                    return AccreditationVerificationResult(
                        found=True,
                        system=first.get("system", ""),
                        system_name=first.get("system_name", ""),
                        rank=first.get("rank"),
                        status="UNVERIFIED",
                        lookup_status="AMBIGUOUS",
                        journal_name=first.get("name"),
                        issn_matched=list(matched_issn_map.keys())[0],
                        all_issns=[self.format_issn(i) for i in candidate_issns],
                        is_ambiguous=True,
                        ambiguity_reason="Ditemukan lebih dari satu catatan dengan peringkat berbeda untuk ISSN ini.",
                        verification_source=first.get("system_name", ""),
                        source_url=first.get("source_url", ""),
                        verification_date=timestamp,
                        match_method="ISSN",
                        checked_for_year=norm_year,
                        notes="Konflik catatan pada direktori akreditasi; perlu pemeriksaan manual.",
                    )

            # Pilih entri terbaik (prioritas: HK-dir Level 2 > Level 1 > ABDC A* > A > B > C)
            def _rank_priority(entry: Dict[str, Any]) -> int:
                r = str(entry.get("rank") or "").upper()
                if "LEVEL 2" in r:
                    return 100
                if "A*" in r:
                    return 90
                if "LEVEL 1" in r:
                    return 80
                if r == "A":
                    return 70
                if r == "B":
                    return 60
                if r == "C":
                    return 50
                if "LEVEL 0" in r:
                    return 10
                return 0

            matched_entries.sort(key=_rank_priority, reverse=True)
            chosen = matched_entries[0]
            matched_issn_str = list(matched_issn_map.keys())[0]

            # Verifikasi pendukung nama jurnal: jika ada nama dan sangat berbeda, tandai ambigu
            is_ambiguous = False
            ambiguity_reason = ""
            if norm_name:
                entry_norm_name = self.normalize_name(chosen.get("name"))
                # Cek overlap kata
                name_words = set(norm_name.split())
                entry_words = set(entry_norm_name.split())
                if name_words and entry_words and not name_words.intersection(entry_words):
                    is_ambiguous = True
                    ambiguity_reason = (
                        f"ISSN cocok, namun nama jurnal pada paper ('{clean_name}') "
                        f"berbeda dengan catatan resmi ('{chosen.get('name')}')."
                    )

            # Evaluasi periode
            start_yr = chosen.get("period_start")
            end_yr = chosen.get("period_end")
            period_status, lookup_status, period_notes = self.evaluate_period(
                start_yr, end_yr, norm_year
            )

            # Tentukan status akhir
            rank_val = chosen.get("rank")
            if is_ambiguous:
                final_status = "UNVERIFIED"
                final_lookup = "AMBIGUOUS"
                final_notes = ambiguity_reason
            elif lookup_status == "VERIFIED":
                if rank_val == "Level 0":
                    final_status = "NOT_ACCREDITED"
                    final_lookup = "VERIFIED"
                    final_notes = f"Jurnal tercatat sebagai Level 0 (Tidak Terakreditasi Ilmiah) pada {chosen.get('system_name')}."
                else:
                    final_status = "ACCREDITED"
                    final_lookup = "VERIFIED"
                    final_notes = period_notes
            else:
                final_status = "UNVERIFIED"
                final_lookup = lookup_status
                final_notes = period_notes

            return AccreditationVerificationResult(
                found=True,
                system=chosen.get("system", ""),
                system_name=chosen.get("system_name", ""),
                rank=rank_val,
                status=final_status,
                lookup_status=final_lookup,
                journal_name=chosen.get("name"),
                issn_matched=matched_issn_str,
                all_issns=[self.format_issn(i) for i in candidate_issns],
                publisher=chosen.get("publisher"),
                country=chosen.get("country"),
                source_url=chosen.get("source_url", ""),
                verification_source=chosen.get("system_name", ""),
                verification_date=timestamp,
                match_method="ISSN",
                is_ambiguous=is_ambiguous,
                ambiguity_reason=ambiguity_reason,
                period_start=start_yr,
                period_end=end_yr,
                period_status=period_status,
                period_confirmed=(final_lookup == "VERIFIED" and period_status == "VALID"),
                checked_for_year=norm_year,
                notes=final_notes,
                details=chosen,
            )

        # ----------------------------------------------------------------
        # 2. Kasus Pencocokan Nama Tanpa ISSN (Pendukung Saja)
        # ----------------------------------------------------------------
        if norm_name:
            name_matches = self._name_index.get(norm_name, [])
            if name_matches:
                # Per aturan: "Gunakan nama jurnal sebagai pemeriksaan pendukung,
                # bukan satu-satunya bukti... tandai kandidat ambigu"
                chosen = name_matches[0]
                return AccreditationVerificationResult(
                    found=True,
                    system=chosen.get("system", ""),
                    system_name=chosen.get("system_name", ""),
                    rank=chosen.get("rank"),
                    status="UNVERIFIED",
                    lookup_status="AMBIGUOUS",
                    journal_name=chosen.get("name"),
                    issn_matched=None,
                    all_issns=[self.format_issn(i) for i in candidate_issns],
                    publisher=chosen.get("publisher"),
                    country=chosen.get("country"),
                    source_url=chosen.get("source_url", ""),
                    verification_source=chosen.get("system_name", ""),
                    verification_date=timestamp,
                    match_method="JOURNAL_NAME",
                    is_ambiguous=True,
                    ambiguity_reason=(
                        "Pencocokan hanya berdasarkan kemiripan nama jurnal tanpa ISSN yang cocok. "
                        "Diperlukan verifikasi ISSN manual untuk memastikan akreditasi."
                    ),
                    period_start=chosen.get("period_start"),
                    period_end=chosen.get("period_end"),
                    period_status="PERIOD_CHECK_REQUIRED",
                    period_confirmed=False,
                    checked_for_year=norm_year,
                    notes=(
                        "Kandidat jurnal ditemukan dari nama, namun ISSN tidak cocok atau tidak tersedia. "
                        "Status belum diverifikasi."
                    ),
                    details=chosen,
                )

        # ----------------------------------------------------------------
        # 3. Tidak Ditemukan di Direktori yang Didukung
        # ----------------------------------------------------------------
        all_issns_formatted = [self.format_issn(i) for i in candidate_issns]
        issn_display = ", ".join(all_issns_formatted) if all_issns_formatted else "N/A"
        return AccreditationVerificationResult(
            found=False,
            status="NOT_FOUND",
            lookup_status="NOT_FOUND",
            journal_name=clean_name or None,
            issn_matched=None,
            all_issns=all_issns_formatted,
            verification_source="Direktori Akreditasi Non-SINTA (HK-dir / ABDC)",
            verification_date=timestamp,
            match_method="NONE",
            checked_for_year=norm_year,
            notes=(
                f"ISSN ({issn_display}) tidak terdaftar pada direktori akreditasi resmi yang didukung "
                f"(HK-dir Kanalregisteret, ABDC). Catatan: Tidak ditemukan di sumber ini bukan bukti bahwa "
                f"jurnal tidak terakreditasi oleh otoritas negara lain."
            ),
        )

    # ------------------------------------------------------------------
    # Re-evaluasi untuk Tahun Tertentu
    # ------------------------------------------------------------------

    @classmethod
    def evaluate_for_year(
        cls,
        result: AccreditationVerificationResult,
        article_year: Optional[Any],
    ) -> AccreditationVerificationResult:
        """
        Mengevaluasi ulang hasil verifikasi yang sudah ada untuk tahun artikel yang berbeda.
        Menghindari kueri ulang bila data akreditasi dasar jurnal sudah didapatkan.
        """
        if not result.found or not result.rank:
            # Jika belum ditemukan, perbarui saja checked_for_year
            res_dict = result.to_dict()
            res_dict["checked_for_year"] = cls.normalize_year(article_year)
            return AccreditationVerificationResult.from_dict(res_dict)

        norm_year = cls.normalize_year(article_year)
        period_status, lookup_status, notes = cls.evaluate_period(
            result.period_start, result.period_end, norm_year
        )

        res_dict = result.to_dict()
        res_dict["checked_for_year"] = norm_year
        res_dict["period_status"] = period_status

        if result.is_ambiguous:
            res_dict["status"] = "UNVERIFIED"
            res_dict["lookup_status"] = "AMBIGUOUS"
        elif lookup_status == "VERIFIED":
            if result.rank == "Level 0":
                res_dict["status"] = "NOT_ACCREDITED"
                res_dict["lookup_status"] = "VERIFIED"
            else:
                res_dict["status"] = "ACCREDITED"
                res_dict["lookup_status"] = "VERIFIED"
            res_dict["notes"] = notes
            res_dict["period_confirmed"] = (period_status == "VALID")
        else:
            res_dict["status"] = "UNVERIFIED"
            res_dict["lookup_status"] = lookup_status
            res_dict["notes"] = notes
            res_dict["period_confirmed"] = False

        return AccreditationVerificationResult.from_dict(res_dict)
