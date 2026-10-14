from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import quote

import httpx

from app.core.config import get_data_dir
from app.dataset_search.huggingface import DatasetFile, DatasetSearchResult


class KaggleDatasetClient:
    """
    Kaggle Dataset Discovery and Download Client.

    Supports both:
    1. Authenticated mode (via ~/.kaggle/kaggle.json, .env, or data/config.json)
    2. Anonymous public mode for searching and downloading public datasets via Kaggle API v1.
    """

    BASE_URL = "https://www.kaggle.com/api/v1"

    SUPPORTED_TABULAR = {
        ".csv",
        ".tsv",
        ".parquet",
        ".xlsx",
        ".xls",
        ".json",
    }

    def __init__(
        self,
        username: Optional[str] = None,
        key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.timeout = timeout
        self.username, self.key = self._resolve_credentials(username, key)

    @property
    def auth(self) -> Optional[Tuple[str, str]]:
        if self.username and self.key:
            return (self.username, self.key)
        return None

    @property
    def is_authenticated(self) -> bool:
        return self.auth is not None

    def search(self, query: str, limit: int = 20) -> List[DatasetSearchResult]:
        """Search Kaggle datasets by query."""
        if not query or not query.strip():
            return []

        cleaned_query = query.strip()
        url = f"{self.BASE_URL}/datasets/list"
        params = {
            "search": cleaned_query,
            "page": 1,
            "sortBy": "relevance",
        }

        headers = {
            "User-Agent": "DatasetResearchApp/1.0",
            "Accept": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(
                    url,
                    params=params,
                    headers=headers,
                    auth=self.auth,
                )
                response.raise_for_status()
                items = response.json()
        except Exception as exc:
            # Re-raise so caller or UI worker can display error
            raise RuntimeError(f"Gagal mencari dataset di Kaggle: {exc}") from exc

        if not isinstance(items, list):
            return []

        results: List[DatasetSearchResult] = []
        query_words = set(re.findall(r"\w+", cleaned_query.lower()))

        for item in items[:limit]:
            if not isinstance(item, dict):
                continue

            ref = item.get("ref") or ""
            if not ref or "/" not in ref:
                continue

            title = item.get("title") or item.get("titleNullable") or ref.split("/")[-1]
            owner = item.get("ownerName") or item.get("creatorName") or ref.split("/")[0]
            description = (
                item.get("subtitle")
                or item.get("subtitleNullable")
                or item.get("description")
                or ""
            )

            total_bytes = item.get("totalBytes") or item.get("totalBytesNullable")
            downloads = item.get("downloadCount", 0) or 0
            votes = item.get("voteCount", 0) or 0
            last_updated = item.get("lastUpdated", "") or ""
            usability = float(item.get("usabilityRating", 0.0) or 0.0)

            # Files representation: Kaggle provides entire dataset bundle
            dataset_name = ref.split("/")[-1]
            files = [
                DatasetFile(
                    filename=f"{dataset_name}.csv (Auto-Extract)",
                    size_bytes=total_bytes,
                )
            ]

            # Calculate relevance score
            title_lower = title.lower()
            desc_lower = description.lower()
            matched = [w for w in query_words if w in title_lower or w in desc_lower]
            relevance = (len(matched) * 35.0) + min(votes * 0.5, 30.0) + min(downloads * 0.001, 20.0) + (usability * 15.0)

            # Construct standardized DatasetSearchResult
            # Prefix dataset_id with 'kaggle:' to distinguish from Hugging Face
            result = DatasetSearchResult(
                dataset_id=f"kaggle:{ref}",
                author=owner,
                description=description,
                downloads=downloads,
                likes=votes,
                last_modified=last_updated,
                tags=["kaggle", "tabular"],
                files=files,
                matched_terms=matched,
                match_reason=f"Kaggle dataset ({votes} votes, usability {int(usability * 100)}%)",
                relevance_score=round(relevance, 2),
            )
            results.append(result)

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def download_dataset(
        self,
        dataset_id: str,
        destination_dir: Path,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Download Kaggle dataset zip archive and extract tabular files.
        Returns the Path to the primary extracted data file (.csv / .parquet).
        """
        extracted_files = self.download_dataset_files(
            dataset_id,
            destination_dir,
            filename=filename,
        )
        return extracted_files[0]

    def download_dataset_files(
        self,
        dataset_id: str,
        destination_dir: Path,
        filename: Optional[str] = None,
    ) -> List[Path]:
        """Download a Kaggle archive and return every supported tabular file."""
        clean_ref = dataset_id.replace("kaggle:", "").strip()
        if "/" not in clean_ref:
            raise ValueError(f"Format dataset id Kaggle tidak valid: {clean_ref}")

        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        url = f"{self.BASE_URL}/datasets/download/{clean_ref}"
        headers = {
            "User-Agent": "DatasetResearchApp/1.0",
        }

        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                response = client.get(url, headers=headers, auth=self.auth)
                response.raise_for_status()
                content = response.content
        except Exception as exc:
            raise RuntimeError(f"Gagal mengunduh dataset dari Kaggle: {exc}") from exc

        # Extract zip in destination subfolder
        safe_name = clean_ref.replace("/", "__")
        target_dir = destination_dir / safe_name
        target_dir.mkdir(parents=True, exist_ok=True)

        extracted_files: List[Path] = []

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                for member in z.infolist():
                    # Security check: prevent zip slip
                    member_path = Path(member.filename)
                    if ".." in member_path.parts or member.filename.startswith(("/", "\\")):
                        continue

                    # Extract file
                    out_path = target_dir / member_path.name
                    with z.open(member) as source, open(out_path, "wb") as target:
                        target.write(source.read())

                    if out_path.suffix.lower() in self.SUPPORTED_TABULAR:
                        extracted_files.append(out_path)
        except zipfile.BadZipFile:
            # If not a zip, it might be a direct CSV or tabular file
            direct_file = target_dir / f"{safe_name}.csv"
            direct_file.write_bytes(content)
            return [direct_file]

        if not extracted_files:
            # List any extracted files
            all_extracted = list(target_dir.glob("*"))
            if all_extracted:
                return [all_extracted[0]]
            raise RuntimeError("Dataset berhasil diunduh tetapi tidak ditemukan file tabular di dalamnya.")

        # Pick the best tabular file (prioritize CSV, then largest size)
        extracted_files.sort(
            key=lambda p: (1 if p.suffix.lower() == ".csv" else 0, p.stat().st_size),
            reverse=True,
        )
        return extracted_files

    # =====================================================================
    # HELPERS
    # =====================================================================

    def _resolve_credentials(
        self,
        username: Optional[str] = None,
        key: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        if username and key:
            return (username.strip(), key.strip())

        # 1. Environment variables
        env_user = os.getenv("KAGGLE_USERNAME")
        env_key = os.getenv("KAGGLE_KEY")
        if env_user and env_key:
            return (env_user.strip(), env_key.strip())

        # 2. Local config data/config.json
        config_path = get_data_dir() / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                u = data.get("kaggle_username")
                k = data.get("kaggle_key")
                if u and k:
                    return (str(u).strip(), str(k).strip())
            except Exception:
                pass

        # 3. Standard ~/.kaggle/kaggle.json
        home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
        if home_kaggle.exists():
            try:
                data = json.loads(home_kaggle.read_text(encoding="utf-8"))
                u = data.get("username")
                k = data.get("key")
                if u and k:
                    return (str(u).strip(), str(k).strip())
            except Exception:
                pass

        return (None, None)

