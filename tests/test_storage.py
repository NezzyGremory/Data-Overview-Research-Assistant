from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from app.storage.database import Database
from app.storage.project_repository import ProjectRepository


def main() -> None:
    print("=== SQLITE PROJECT STORAGE ===")

    with TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"

        database = Database(db_path)
        repository = ProjectRepository(database)

        print("Database initialization: PASSED")

        project_id = repository.create_project(
            name="Passenger Survival Research",
            description="Research project for passenger survival dataset.",
            dataset_name="Titanic Dataset",
            dataset_path="data/titanic.csv",
        )

        assert project_id > 0

        project = repository.get_project(project_id)

        assert project is not None
        assert project["name"] == "Passenger Survival Research"
        assert project["dataset_name"] == "Titanic Dataset"

        print("Project creation: PASSED")

        projects = repository.list_projects()

        assert len(projects) == 1
        assert projects[0]["id"] == project_id

        print("Project listing: PASSED")

        research_result = {
            "status": "SUCCESS",
            "keywords": [
                "passenger",
                "survival",
                "classification",
            ],
            "domain": {
                "primary_domain": "transportation",
                "confidence": 0.85,
            },
            "queries": [
                "passenger survival classification transportation",
                "Survived prediction passenger survival classification",
            ],
            "ml": {
                "task": "binary classification",
                "target": "Survived",
                "recommendations": [
                    {
                        "method": "Random Forest",
                        "score": 95,
                    },
                    {
                        "method": "Logistic Regression",
                        "score": 85,
                    },
                ],
            },
            "ranking": {
                "papers": [
                    {
                        "title": (
                            "Passenger Survival Prediction "
                            "Using Random Forest"
                        ),
                        "authors": ["Researcher A"],
                        "abstract": (
                            "Machine learning classification "
                            "for passenger survival prediction."
                        ),
                        "year": 2024,
                        "doi": "10.1234/survival",
                        "venue": "Machine Learning Journal",
                        "url": "https://example.com/paper",
                        "citation_count": 25,
                        "source": "Mock",
                        "external_id": "mock-1",
                        "issn": ["1234-5678"],
                        "sinta_screening": {
                            "status": "S2",
                            "checked_for_year": 2024,
                            "period_confirmed": True,
                        },
                        "keywords": [
                            "machine learning",
                            "survival",
                        ],
                        "relevance_score": 83.0,
                        "score_breakdown": {
                            "keyword": 100.0,
                            "domain": 65.0,
                            "target": 100.0,
                            "method": 50.0,
                        },
                        "reasons": [
                            "Keyword similarity 100.0",
                            "Domain similarity 65.0",
                            "Target similarity 100.0",
                            "ML method similarity 50.0",
                        ],
                        "ranking_status": "HEURISTIC",
                        "rank": 1,
                    }
                ]
            },
            "landscape": {
                "status": "SUCCESS",
                "methods": {
                    "Random Forest": 100.0,
                },
            },
            "trend": {
                "status": "SUCCESS",
                "recent_direction": "INCREASING",
            },
            "gaps": {
                "status": "SUCCESS",
                "gaps": [
                    {
                        "type": "METHOD",
                        "title": "Potential Method Gap",
                    }
                ],
            },
            "summary": {
                "papers_found": 1,
                "top_relevance_score": 83.0,
                "potential_gap_count": 1,
                "research_direction": "INCREASING",
                "dominant_method": "Random Forest",
                "latest_publication_year": "2024",
            },
        }

        repository.save_research_result(
            project_id,
            research_result,
        )

        print("Research result saving: PASSED")

        loaded_result = repository.get_research_result(project_id)

        assert loaded_result is not None
        assert loaded_result["status"] == "SUCCESS"

        assert loaded_result["keywords"] == [
            "passenger",
            "survival",
            "classification",
        ]

        assert (
            loaded_result["domain"]["primary_domain"]
            == "transportation"
        )

        assert loaded_result["summary"]["papers_found"] == 1

        print("Research result loading: PASSED")

        papers = repository.get_papers(project_id)

        assert len(papers) == 1

        paper = papers[0]

        assert (
            paper["title"]
            == "Passenger Survival Prediction Using Random Forest"
        )

        assert paper["year"] == 2024
        assert paper["relevance_score"] == 83.0
        assert paper["rank"] == 1
        assert paper["issn"] == ["1234-5678"]
        assert paper["sinta_screening"] == {
            "status": "S2",
            "checked_for_year": 2024,
            "period_confirmed": True,
        }

        assert paper["score_breakdown"]["keyword"] == 100.0
        assert paper["score_breakdown"]["domain"] == 65.0

        print("Paper storage: PASSED")

        full_project = repository.get_project_with_research(
            project_id
        )

        assert full_project is not None
        assert full_project["name"] == "Passenger Survival Research"
        assert full_project["research"] is not None
        assert len(full_project["research"]["papers"]) == 1

        print("Project + research loading: PASSED")

        repository.delete_project(project_id)

        deleted_project = repository.get_project(project_id)

        assert deleted_project is None

        deleted_research = repository.get_research_result(project_id)

        assert deleted_research is None

        deleted_papers = repository.get_papers(project_id)

        assert deleted_papers == []

        print("Cascade delete: PASSED")

    print()
    print("================================")
    print("Semua SQLite Storage test berhasil.")
    print("================================")


if __name__ == "__main__":
    main()