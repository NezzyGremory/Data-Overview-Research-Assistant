from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from app.storage.database import Database


class ProjectRepository:
    """
    Repository untuk mengelola Project dan Research Result.

    Struktur data Research Intelligence yang didukung:

    result = {
        "status": ...,
        "keywords": ...,
        "domain": ...,
        "queries": ...,
        "search": ...,
        "papers": [...],
        "top_papers": [...],
        "landscape": ...,
        "trend": ...,
        "gaps": ...,
        "summary": ...,
        "transparency": ...
    }
    """

    def __init__(self, database: Database):
        self.database = database

    # ============================================================
    # JSON HELPERS
    # ============================================================

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=ProjectRepository._json_default,
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert dataclasses and common NumPy/Pandas values to JSON safely."""
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, set):
            return sorted(value, key=str)
        return str(value)

    @staticmethod
    def _from_json(
        value: str | None,
        default: Any = None,
    ) -> Any:
        if value is None:
            return default

        try:
            return json.loads(value)
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return default

    # ============================================================
    # PROJECT CRUD
    # ============================================================

    def create_project(
        self,
        name: str,
        description: str | None = None,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> int:

        return self.database.execute(
            """
            INSERT INTO projects (
                name,
                description,
                dataset_name,
                dataset_path
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                description,
                dataset_name,
                dataset_path,
            ),
        )

    def get_project(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:

        row = self.database.fetch_one(
            """
            SELECT
                id,
                name,
                description,
                dataset_name,
                dataset_path,
                created_at,
                updated_at
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        )

        if row is None:
            return None

        return dict(row)

    def list_projects(self) -> list[dict[str, Any]]:

        rows = self.database.fetch_all(
            """
            SELECT
                id,
                name,
                description,
                dataset_name,
                dataset_path,
                created_at,
                updated_at
            FROM projects
            ORDER BY updated_at DESC, id DESC
            """
        )

        return [
            dict(row)
            for row in rows
        ]

    def update_project(
        self,
        project_id: int,
        name: str | None = None,
        description: str | None = None,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> None:

        current = self.get_project(project_id)

        if current is None:
            raise ValueError(
                f"Project dengan ID {project_id} tidak ditemukan."
            )

        final_name = (
            name
            if name is not None
            else current["name"]
        )

        final_description = (
            description
            if description is not None
            else current["description"]
        )

        final_dataset_name = (
            dataset_name
            if dataset_name is not None
            else current["dataset_name"]
        )

        final_dataset_path = (
            dataset_path
            if dataset_path is not None
            else current["dataset_path"]
        )

        self.database.execute(
            """
            UPDATE projects
            SET
                name = ?,
                description = ?,
                dataset_name = ?,
                dataset_path = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                final_name,
                final_description,
                final_dataset_name,
                final_dataset_path,
                project_id,
            ),
        )
        return True

    def save_workflow_section(
        self,
        project_id: int,
        section: str,
        value: Any,
    ) -> None:
        """Persist one feature result locally, keyed by its dataset project."""
        columns = {
            "analysis": "analysis_result_json",
            "ml_task": "ml_task_result_json",
            "ml_methods": "ml_methods_result_json",
            "research": "research_result_json",
        }
        column = columns.get(section)
        if column is None:
            raise ValueError(f"Bagian workflow tidak dikenal: {section}")
        if not self.project_exists(project_id):
            raise ValueError(f"Project dengan ID {project_id} tidak ditemukan.")

        payload = self._json(value)
        self.database.execute(
            f"""
            INSERT INTO project_workflows (project_id, {column})
            VALUES (?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                {column} = excluded.{column},
                updated_at = CURRENT_TIMESTAMP
            """,
            (project_id, payload),
        )
        self.database.execute(
            "UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (project_id,),
        )

    def get_workflow_state(self, project_id: int) -> dict[str, Any] | None:
        """Load saved feature results for a project without loading its dataset."""
        row = self.database.fetch_one(
            """
            SELECT analysis_result_json, ml_task_result_json,
                   ml_methods_result_json, research_result_json, updated_at
            FROM project_workflows
            WHERE project_id = ?
            """,
            (project_id,),
        )
        if row is None:
            return None
        return {
            "analysis": self._from_json(row["analysis_result_json"], {}),
            "ml_task": self._from_json(row["ml_task_result_json"], {}),
            "ml_methods": self._from_json(row["ml_methods_result_json"], {}),
            "research": self._from_json(row["research_result_json"], {}),
            "updated_at": row["updated_at"],
        }

    def delete_project(
        self,
        project_id: int,
    ) -> bool:

        self.database.execute(
            """
            DELETE FROM projects
            WHERE id = ?
            """,
            (project_id,),
        )
        return True

    # ============================================================
    # RESEARCH RESULT
    # ============================================================

    def save_research_result(
        self,
        project_id: int,
        result: dict[str, Any],
    ) -> None:
        """
        Menyimpan hasil Research Intelligence.

        Paper utama diambil dari:

            result["papers"]

        Jika result["papers"] kosong, repository akan mencoba:

            result["top_papers"]

        Hal ini membuat storage lebih tahan terhadap variasi
        struktur result dari Research Intelligence Engine.
        """

        if not self.project_exists(project_id):
            raise ValueError(
                f"Project dengan ID {project_id} tidak ditemukan."
            )

        if not isinstance(result, dict):
            raise TypeError(
                "result harus berupa dictionary."
            )

        status = result.get("status")

        keywords = result.get(
            "keywords",
            [],
        )

        domain = result.get(
            "domain",
            {},
        )

        queries = result.get(
            "queries",
            [],
        )

        search = result.get(
            "search",
            {},
        )

        papers = result.get(
            "papers",
            [],
        )

        top_papers = result.get(
            "top_papers",
            [],
        )

        landscape = result.get(
            "landscape",
            {},
        )

        trend = result.get(
            "trend",
            {},
        )

        gaps = result.get(
            "gaps",
            {},
        )

        summary = result.get(
            "summary",
            {},
        )

        transparency = result.get(
            "transparency",
            {},
        )

        # --------------------------------------------------------
        # Normalisasi paper
        # --------------------------------------------------------

        if not isinstance(papers, list):
            papers = list(papers) if papers else []

        if not isinstance(top_papers, list):
            top_papers = (
                list(top_papers)
                if top_papers
                else []
            )

        # Jika papers kosong tetapi top_papers tersedia,
        # gunakan top_papers sebagai sumber penyimpanan.
        if not papers and top_papers:
            papers = top_papers

        if not papers and isinstance(result.get("ranking"), dict):
            ranking_papers = result["ranking"].get("papers", [])
            if isinstance(ranking_papers, list):
                papers = ranking_papers

        # --------------------------------------------------------
        # Research Result
        # --------------------------------------------------------

        existing = self.database.fetch_one(
            """
            SELECT id
            FROM research_results
            WHERE project_id = ?
            """,
            (project_id,),
        )

        research_payload = self._json(
            {
                "search": search,
                "top_papers": top_papers,
                "transparency": transparency,
            }
        )

        if existing is None:

            self.database.execute(
                """
                INSERT INTO research_results (
                    project_id,
                    status,
                    keywords_json,
                    domain_json,
                    queries_json,
                    ml_result_json,
                    landscape_json,
                    trend_json,
                    gaps_json,
                    summary_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    status,
                    self._json(keywords),
                    self._json(domain),
                    self._json(queries),
                    research_payload,
                    self._json(landscape),
                    self._json(trend),
                    self._json(gaps),
                    self._json(summary),
                ),
            )

        else:

            self.database.execute(
                """
                UPDATE research_results
                SET
                    status = ?,
                    keywords_json = ?,
                    domain_json = ?,
                    queries_json = ?,
                    ml_result_json = ?,
                    landscape_json = ?,
                    trend_json = ?,
                    gaps_json = ?,
                    summary_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE project_id = ?
                """,
                (
                    status,
                    self._json(keywords),
                    self._json(domain),
                    self._json(queries),
                    research_payload,
                    self._json(landscape),
                    self._json(trend),
                    self._json(gaps),
                    self._json(summary),
                    project_id,
                ),
            )

        # --------------------------------------------------------
        # Papers
        # --------------------------------------------------------

        self.delete_papers(project_id)

        saved_paper_count = 0

        for paper in papers:

            if not isinstance(paper, dict):
                continue

            title = paper.get(
                "title",
                "",
            )

            if title is None:
                title = ""

            title = str(title).strip()

            if not title:
                continue

            paper_data = dict(paper)

            paper_data["title"] = title

            self.save_paper(
                project_id=project_id,
                paper=paper_data,
            )

            saved_paper_count += 1

        # --------------------------------------------------------
        # Update Project Timestamp
        # --------------------------------------------------------

        self.database.execute(
            """
            UPDATE projects
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (project_id,),
        )

    # ============================================================
    # GET RESEARCH RESULT
    # ============================================================

    def get_research_result(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:

        row = self.database.fetch_one(
            """
            SELECT
                id,
                project_id,
                status,
                keywords_json,
                domain_json,
                queries_json,
                ml_result_json,
                landscape_json,
                trend_json,
                gaps_json,
                summary_json,
                created_at,
                updated_at
            FROM research_results
            WHERE project_id = ?
            """,
            (project_id,),
        )

        if row is None:
            return None

        ml_result = self._from_json(
            row["ml_result_json"],
            {},
        )

        if not isinstance(ml_result, dict):
            ml_result = {}

        result = {
            "id": row["id"],
            "project_id": row["project_id"],
            "status": row["status"],
            "keywords": self._from_json(
                row["keywords_json"],
                [],
            ),
            "domain": self._from_json(
                row["domain_json"],
                {},
            ),
            "queries": self._from_json(
                row["queries_json"],
                [],
            ),
            "search": ml_result.get(
                "search",
                {},
            ),
            "top_papers": ml_result.get(
                "top_papers",
                [],
            ),
            "transparency": ml_result.get(
                "transparency",
                {},
            ),
            "landscape": self._from_json(
                row["landscape_json"],
                {},
            ),
            "trend": self._from_json(
                row["trend_json"],
                {},
            ),
            "gaps": self._from_json(
                row["gaps_json"],
                {},
            ),
            "summary": self._from_json(
                row["summary_json"],
                {},
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

        result["papers"] = self.get_papers(
            project_id
        )

        return result

    # ============================================================
    # PAPER STORAGE
    # ============================================================

    def save_paper(
        self,
        project_id: int,
        paper: dict[str, Any],
    ) -> int:

        if not isinstance(paper, dict):
            raise TypeError(
                "paper harus berupa dictionary."
            )

        title = paper.get(
            "title",
            "",
        )

        if title is None:
            title = ""

        title = str(title).strip()

        if not title:
            raise ValueError(
                "Paper harus memiliki title."
            )

        return self.database.execute(
            """
            INSERT INTO papers (
                project_id,
                title,
                authors_json,
                abstract,
                year,
                doi,
                venue,
                url,
                citation_count,
                source,
                external_id,
                keywords_json,
                issn_json,
                sinta_screening_json,
                accreditation_screening_json,
                relevance_score,
                score_breakdown_json,
                reasons_json,
                ranking_status,
                rank
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                project_id,
                title,
                self._json(
                    paper.get(
                        "authors",
                        [],
                    )
                ),
                paper.get("abstract"),
                paper.get("year"),
                paper.get("doi"),
                paper.get("venue"),
                paper.get("url"),
                paper.get(
                    "citation_count",
                    0,
                ),
                paper.get("source"),
                paper.get("external_id"),
                self._json(
                    paper.get(
                        "keywords",
                        [],
                    )
                ),
                self._json(paper.get("issn", [])),
                self._json(paper.get("sinta_screening", {})),
                self._json(paper.get("accreditation_screening", {})),
                paper.get(
                    "relevance_score"
                ),
                self._json(
                    paper.get(
                        "score_breakdown",
                        {},
                    )
                ),
                self._json(
                    paper.get(
                        "reasons",
                        [],
                    )
                ),
                paper.get(
                    "ranking_status"
                ),
                paper.get("rank"),
            ),
        )

    def get_papers(
        self,
        project_id: int,
    ) -> list[dict[str, Any]]:

        rows = self.database.fetch_all(
            """
            SELECT
                id,
                project_id,
                title,
                authors_json,
                abstract,
                year,
                doi,
                venue,
                url,
                citation_count,
                source,
                external_id,
                keywords_json,
                issn_json,
                sinta_screening_json,
                accreditation_screening_json,
                relevance_score,
                score_breakdown_json,
                reasons_json,
                ranking_status,
                rank,
                created_at
            FROM papers
            WHERE project_id = ?
            ORDER BY
                CASE
                    WHEN rank IS NULL THEN 1
                    ELSE 0
                END,
                rank ASC,
                relevance_score DESC,
                id ASC
            """,
            (project_id,),
        )

        papers = []

        for row in rows:

            papers.append(
                {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "title": row["title"],
                    "authors": self._from_json(
                        row["authors_json"],
                        [],
                    ),
                    "abstract": row["abstract"],
                    "year": row["year"],
                    "doi": row["doi"],
                    "venue": row["venue"],
                    "url": row["url"],
                    "citation_count": row[
                        "citation_count"
                    ],
                    "source": row["source"],
                    "external_id": row[
                        "external_id"
                    ],
                    "keywords": self._from_json(
                        row["keywords_json"],
                        [],
                    ),
                    "issn": self._from_json(row["issn_json"], []),
                    "sinta_screening": self._from_json(
                        row["sinta_screening_json"],
                        {},
                    ),
                    "accreditation_screening": self._from_json(
                        row["accreditation_screening_json"] if "accreditation_screening_json" in row.keys() else None,
                        {},
                    ),
                    "relevance_score": row[
                        "relevance_score"
                    ],
                    "score_breakdown": self._from_json(
                        row[
                            "score_breakdown_json"
                        ],
                        {},
                    ),
                    "reasons": self._from_json(
                        row["reasons_json"],
                        [],
                    ),
                    "ranking_status": row[
                        "ranking_status"
                    ],
                    "rank": row["rank"],
                    "created_at": row[
                        "created_at"
                    ],
                }
            )

        return papers

    def get_paper(
        self,
        paper_id: int,
    ) -> dict[str, Any] | None:

        row = self.database.fetch_one(
            """
            SELECT
                id,
                project_id,
                title,
                authors_json,
                abstract,
                year,
                doi,
                venue,
                url,
                citation_count,
                source,
                external_id,
                keywords_json,
                issn_json,
                sinta_screening_json,
                accreditation_screening_json,
                relevance_score,
                score_breakdown_json,
                reasons_json,
                ranking_status,
                rank,
                created_at
            FROM papers
            WHERE id = ?
            """,
            (paper_id,),
        )

        if row is None:
            return None

        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "title": row["title"],
            "authors": self._from_json(
                row["authors_json"],
                [],
            ),
            "abstract": row["abstract"],
            "year": row["year"],
            "doi": row["doi"],
            "venue": row["venue"],
            "url": row["url"],
            "citation_count": row[
                "citation_count"
            ],
            "source": row["source"],
            "external_id": row[
                "external_id"
            ],
            "keywords": self._from_json(
                row["keywords_json"],
                [],
            ),
            "issn": self._from_json(row["issn_json"], []),
            "sinta_screening": self._from_json(
                row["sinta_screening_json"],
                {},
            ),
            "accreditation_screening": self._from_json(
                row["accreditation_screening_json"] if "accreditation_screening_json" in row.keys() else None,
                {},
            ),
            "relevance_score": row[
                "relevance_score"
            ],
            "score_breakdown": self._from_json(
                row[
                    "score_breakdown_json"
                ],
                {},
            ),
            "reasons": self._from_json(
                row["reasons_json"],
                [],
            ),
            "ranking_status": row[
                "ranking_status"
            ],
            "rank": row["rank"],
            "created_at": row[
                "created_at"
            ],
        }

    def delete_papers(
        self,
        project_id: int,
    ) -> None:

        self.database.execute(
            """
            DELETE FROM papers
            WHERE project_id = ?
            """,
            (project_id,),
        )

    # ============================================================
    # COMPLETE PROJECT
    # ============================================================

    def get_project_with_research(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:

        project = self.get_project(
            project_id
        )

        if project is None:
            return None

        project["research"] = (
            self.get_research_result(
                project_id
            )
        )

        return project

    # ============================================================
    # UTILITIES
    # ============================================================

    def project_exists(
        self,
        project_id: int,
    ) -> bool:

        row = self.database.fetch_one(
            """
            SELECT 1
            FROM projects
            WHERE id = ?
            LIMIT 1
            """,
            (project_id,),
        )

        return row is not None

    def research_exists(
        self,
        project_id: int,
    ) -> bool:

        row = self.database.fetch_one(
            """
            SELECT 1
            FROM research_results
            WHERE project_id = ?
            LIMIT 1
            """,
            (project_id,),
        )

        return row is not None

    def count_papers(
        self,
        project_id: int,
    ) -> int:

        row = self.database.fetch_one(
            """
            SELECT COUNT(*) AS count
            FROM papers
            WHERE project_id = ?
            """,
            (project_id,),
        )

        if row is None:
            return 0

        return int(row["count"])

    # ============================================================
    # JOURNAL SCREENING & SINTA PERSISTENCE
    # ============================================================

    def save_journal_screening(
        self,
        paper_key: str,
        screening_data: dict[str, Any],
        doi: str | None = None,
    ) -> None:
        """Menyimpan hasil screening akreditasi jurnal ke SQLite."""
        if not paper_key and not doi:
            return

        doi_val = (doi or "").strip().lower() or None
        key_val = (paper_key or "").strip().lower() or None
        status = screening_data.get("status") or "UNVERIFIED"
        rank = screening_data.get("rank")
        journal_name = screening_data.get("journal_name") or screening_data.get("journal")
        issn_matched = screening_data.get("issn_matched")
        period_start = screening_data.get("period_start")
        period_end = screening_data.get("period_end")
        period_confirmed = 1 if screening_data.get("period_confirmed") else 0
        checked_for_year = screening_data.get("checked_for_year")
        verification_source = screening_data.get("verification_source") or "Direktori Resmi SINTA"
        source_url = screening_data.get("source_url")
        is_ambiguous = 1 if screening_data.get("is_ambiguous") else 0
        ambiguity_reason = screening_data.get("ambiguity_reason") or ""
        notes = screening_data.get("notes") or ""
        payload = self._json(screening_data)

        # Cek apakah sudah ada record berdasarkan DOI atau paper_key
        existing = None
        if doi_val:
            existing = self.database.fetch_one(
                "SELECT id FROM journal_screening WHERE doi = ?",
                (doi_val,),
            )
        if not existing and key_val:
            existing = self.database.fetch_one(
                "SELECT id FROM journal_screening WHERE paper_key = ?",
                (key_val,),
            )

        if existing:
            self.database.execute(
                """
                UPDATE journal_screening
                SET
                    doi = coalesce(?, doi),
                    paper_key = coalesce(?, paper_key),
                    journal_name = ?,
                    issn_matched = ?,
                    rank = ?,
                    status = ?,
                    period_start = ?,
                    period_end = ?,
                    period_confirmed = ?,
                    checked_for_year = ?,
                    verification_source = ?,
                    source_url = ?,
                    is_ambiguous = ?,
                    ambiguity_reason = ?,
                    notes = ?,
                    screening_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    doi_val,
                    key_val,
                    journal_name,
                    issn_matched,
                    rank,
                    status,
                    period_start,
                    period_end,
                    period_confirmed,
                    checked_for_year,
                    verification_source,
                    source_url,
                    is_ambiguous,
                    ambiguity_reason,
                    notes,
                    payload,
                    existing["id"],
                ),
            )
        else:
            self.database.execute(
                """
                INSERT INTO journal_screening (
                    doi,
                    paper_key,
                    journal_name,
                    issn_matched,
                    rank,
                    status,
                    period_start,
                    period_end,
                    period_confirmed,
                    checked_for_year,
                    verification_source,
                    source_url,
                    is_ambiguous,
                    ambiguity_reason,
                    notes,
                    screening_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doi_val,
                    key_val,
                    journal_name,
                    issn_matched,
                    rank,
                    status,
                    period_start,
                    period_end,
                    period_confirmed,
                    checked_for_year,
                    verification_source,
                    source_url,
                    is_ambiguous,
                    ambiguity_reason,
                    notes,
                    payload,
                ),
            )

    def get_journal_screening(
        self,
        doi: str | None = None,
        paper_key: str | None = None,
    ) -> dict[str, Any] | None:
        """Mengambil hasil screening jurnal tersimpan."""
        row = None
        if doi:
            row = self.database.fetch_one(
                """
                SELECT * FROM journal_screening
                WHERE doi = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (doi.strip().lower(),),
            )
        if not row and paper_key:
            row = self.database.fetch_one(
                """
                SELECT * FROM journal_screening
                WHERE paper_key = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (paper_key.strip().lower(),),
            )
        if not row:
            return None

        screening_dict = self._from_json(row["screening_json"], {})
        if not screening_dict:
            screening_dict = {
                "status": row["status"],
                "rank": row["rank"],
                "journal_name": row["journal_name"],
                "issn_matched": row["issn_matched"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "period_confirmed": bool(row["period_confirmed"]),
                "checked_for_year": row["checked_for_year"],
                "verification_source": row["verification_source"],
                "source_url": row["source_url"],
                "is_ambiguous": bool(row["is_ambiguous"]),
                "ambiguity_reason": row["ambiguity_reason"],
                "notes": row["notes"],
                "checked_at": row["checked_at"],
            }
        return screening_dict

    def list_journal_screenings(self) -> dict[str, dict[str, Any]]:
        """Mengambil seluruh riwayat screening yang tersimpan (keyed by DOI and paper_key)."""
        rows = self.database.fetch_all(
            """
            SELECT * FROM journal_screening
            ORDER BY updated_at DESC
            """
        )
        results: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = self._from_json(row["screening_json"], {})
            if not data:
                data = {
                    "status": row["status"],
                    "rank": row["rank"],
                    "journal_name": row["journal_name"],
                    "issn_matched": row["issn_matched"],
                    "period_confirmed": bool(row["period_confirmed"]),
                    "checked_for_year": row["checked_for_year"],
                    "source_url": row["source_url"],
                }
            if row["doi"]:
                results[f"doi:{row['doi']}"] = data
            if row["paper_key"]:
                results[f"key:{row['paper_key']}"] = data
        return results

    # ============================================================
    # NON-SINTA ACCREDITATION PERSISTENCE
    # ============================================================

    def save_non_sinta_screening(
        self,
        paper_key: str,
        screening_data: dict[str, Any],
        doi: str | None = None,
    ) -> None:
        """Menyimpan hasil screening akreditasi non-SINTA ke SQLite."""
        if not paper_key and not doi:
            return

        doi_val = (doi or "").strip().lower() or None
        key_val = (paper_key or "").strip().lower() or None
        system = screening_data.get("system") or ""
        status = screening_data.get("status") or "UNVERIFIED"
        rank = screening_data.get("rank")
        journal_name = screening_data.get("journal_name") or screening_data.get("journal")
        issn_matched = screening_data.get("issn_matched")
        period_start = screening_data.get("period_start")
        period_end = screening_data.get("period_end")
        period_confirmed = 1 if screening_data.get("period_confirmed") else 0
        checked_for_year = screening_data.get("checked_for_year")
        verification_source = screening_data.get("verification_source") or "Direktori Akreditasi Non-SINTA"
        source_url = screening_data.get("source_url") or ""
        is_ambiguous = 1 if screening_data.get("is_ambiguous") else 0
        ambiguity_reason = screening_data.get("ambiguity_reason") or ""
        notes = screening_data.get("notes") or ""
        payload = self._json(screening_data)

        existing = None
        if doi_val:
            existing = self.database.fetch_one(
                "SELECT id FROM non_sinta_screening WHERE doi = ?",
                (doi_val,),
            )
        if not existing and key_val:
            existing = self.database.fetch_one(
                "SELECT id FROM non_sinta_screening WHERE paper_key = ?",
                (key_val,),
            )

        if existing:
            self.database.execute(
                """
                UPDATE non_sinta_screening
                SET
                    doi = coalesce(?, doi),
                    paper_key = coalesce(?, paper_key),
                    system = ?,
                    journal_name = ?,
                    issn_matched = ?,
                    rank = ?,
                    status = ?,
                    period_start = ?,
                    period_end = ?,
                    period_confirmed = ?,
                    checked_for_year = ?,
                    verification_source = ?,
                    source_url = ?,
                    is_ambiguous = ?,
                    ambiguity_reason = ?,
                    notes = ?,
                    screening_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    doi_val,
                    key_val,
                    system,
                    journal_name,
                    issn_matched,
                    rank,
                    status,
                    period_start,
                    period_end,
                    period_confirmed,
                    checked_for_year,
                    verification_source,
                    source_url,
                    is_ambiguous,
                    ambiguity_reason,
                    notes,
                    payload,
                    existing["id"],
                ),
            )
        else:
            self.database.execute(
                """
                INSERT INTO non_sinta_screening (
                    doi,
                    paper_key,
                    system,
                    journal_name,
                    issn_matched,
                    rank,
                    status,
                    period_start,
                    period_end,
                    period_confirmed,
                    checked_for_year,
                    verification_source,
                    source_url,
                    is_ambiguous,
                    ambiguity_reason,
                    notes,
                    screening_json
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    doi_val,
                    key_val,
                    system,
                    journal_name,
                    issn_matched,
                    rank,
                    status,
                    period_start,
                    period_end,
                    period_confirmed,
                    checked_for_year,
                    verification_source,
                    source_url,
                    is_ambiguous,
                    ambiguity_reason,
                    notes,
                    payload,
                ),
            )

    def get_non_sinta_screening(
        self,
        doi: str | None = None,
        paper_key: str | None = None,
    ) -> dict[str, Any] | None:
        """Mengambil hasil screening non-SINTA dari SQLite berdasarkan DOI atau paper_key."""
        row = None
        if doi:
            row = self.database.fetch_one(
                """
                SELECT * FROM non_sinta_screening
                WHERE doi = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (doi.strip().lower(),),
            )
        if not row and paper_key:
            row = self.database.fetch_one(
                """
                SELECT * FROM non_sinta_screening
                WHERE paper_key = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (paper_key.strip().lower(),),
            )
        if not row:
            return None

        screening_dict = self._from_json(row["screening_json"], {})
        if not screening_dict:
            screening_dict = {
                "system": row["system"],
                "status": row["status"],
                "rank": row["rank"],
                "journal_name": row["journal_name"],
                "issn_matched": row["issn_matched"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "period_confirmed": bool(row["period_confirmed"]),
                "checked_for_year": row["checked_for_year"],
                "verification_source": row["verification_source"],
                "source_url": row["source_url"],
                "is_ambiguous": bool(row["is_ambiguous"]),
                "ambiguity_reason": row["ambiguity_reason"],
                "notes": row["notes"],
                "checked_at": row["checked_at"],
            }
        return screening_dict

    def list_non_sinta_screenings(self) -> dict[str, dict[str, Any]]:
        """Mengambil seluruh riwayat screening non-SINTA yang tersimpan."""
        rows = self.database.fetch_all(
            """
            SELECT * FROM non_sinta_screening
            ORDER BY updated_at DESC
            """
        )
        results: dict[str, dict[str, Any]] = {}
        for row in rows:
            data = self._from_json(row["screening_json"], {})
            if not data:
                data = {
                    "system": row["system"],
                    "status": row["status"],
                    "rank": row["rank"],
                    "journal_name": row["journal_name"],
                    "issn_matched": row["issn_matched"],
                    "period_confirmed": bool(row["period_confirmed"]),
                    "checked_for_year": row["checked_for_year"],
                    "source_url": row["source_url"],
                }
            if row["doi"]:
                results[f"doi:{row['doi']}"] = data
            if row["paper_key"]:
                results[f"key:{row['paper_key']}"] = data
        return results

