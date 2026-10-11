import sqlite3

from app.storage.database import Database
from app.storage.project_repository import ProjectRepository


def test_batch_screening_metadata_round_trips_through_local_research_storage(tmp_path):
    repository = ProjectRepository(Database(tmp_path / "research.db"))
    project_id = repository.create_project(
        name="SINTA batch persistence",
        description="Offline persistence test",
        dataset_name="fixture.csv",
        dataset_path="fixture.csv",
    )
    screening = {
        "found": True,
        "status": "UNVERIFIED",
        "lookup_status": "PERIOD_UNVERIFIED",
        "rank": "S2",
        "journal_name": "Journal A",
        "issn_matched": "1234-5678",
        "all_issns": ["1234-5678", "8765-4321"],
        "match_method": "ISSN",
        "period_start": None,
        "period_end": None,
        "period_status": "PERIOD_CHECK_REQUIRED",
        "checked_for_year": 2024,
        "verification_source": "Direktori Resmi SINTA",
        "source_url": "https://sinta.kemdiktisaintek.go.id/journals?q=12345678",
        "verification_date": "2026-10-11 12:00",
        "metadata_key": "issn:12345678|journal:journala|year:2024",
        "notes": "Masa SK belum tersedia.",
    }
    result = {
        "status": "SUCCESS",
        "keywords": [],
        "domain": {},
        "queries": [],
        "search": {},
        "papers": [{
            "title": "Paper A",
            "authors": [],
            "year": 2024,
            "doi": "10.1234/paper-a",
            "venue": "Journal A",
            "url": "https://example.org/paper-a",
            "source": "Fixture",
            "issn": ["1234-5678", "8765-4321"],
            "sinta_screening": screening,
        }],
        "summary": {},
    }

    repository.save_research_result(project_id, result)
    repository.save_workflow_section(project_id, "research", result)
    repository.save_journal_screening(
        paper_key="paper-a-key",
        doi="10.1234/paper-a",
        screening_data=screening,
    )

    loaded = repository.get_research_result(project_id)
    assert loaded["papers"][0]["sinta_screening"] == screening
    assert repository.get_journal_screening(doi="10.1234/paper-a") == screening
    workflow = repository.get_workflow_state(project_id)
    assert workflow["research"]["papers"][0]["sinta_screening"] == screening


def test_existing_screening_table_receives_missing_metadata_columns(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE journal_screening (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                paper_key TEXT,
                journal_name TEXT,
                issn_matched TEXT,
                rank TEXT,
                status TEXT NOT NULL,
                period_start INTEGER,
                checked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )

    Database(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(journal_screening)")}

    assert {
        "period_end",
        "period_confirmed",
        "checked_for_year",
        "verification_source",
        "source_url",
        "is_ambiguous",
        "ambiguity_reason",
        "notes",
        "screening_json",
    }.issubset(columns)
