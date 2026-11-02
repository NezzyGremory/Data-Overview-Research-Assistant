"""ML run history uses an additive SQLite migration and preserves prior runs."""

from app.storage.database import Database
from app.storage.project_repository import ProjectRepository


def test_experiment_history_keeps_runs_and_configuration(tmp_path):
    database = Database(tmp_path / "existing.db")
    repository = ProjectRepository(database)
    project_id = repository.create_project("Dataset A")
    fingerprint = {"rows": 100, "columns": ["x", "y"]}
    options = {"task_override": "binary_classification", "target_override": "y",
               "requested_methods": ["logistic_regression"]}
    result = {"primary_task": {"task": "binary_classification", "target": "y"},
              "best_method": {"method_id": "logistic_regression", "method": "Logistic Regression"},
              "evaluation": {"ranking_metric": "f1_score", "final_test": {
                  "status": "EVALUATED", "metrics": {"f1_score": 91.0}}}}

    first_id = repository.record_ml_experiment(project_id, fingerprint, options, result)
    second_id = repository.record_ml_experiment(project_id, fingerprint, options, result)
    runs = repository.list_ml_experiments(project_id)

    assert [run["id"] for run in runs] == [second_id, first_id]
    assert runs[0]["options"] == options
    assert runs[0]["fingerprint"] == fingerprint
    assert runs[0]["result"]["evaluation"]["final_test"]["metrics"]["f1_score"] == 91.0

    # Reopening the database must not discard previously recorded runs.
    reopened = ProjectRepository(Database(tmp_path / "existing.db"))
    assert len(reopened.list_ml_experiments(project_id)) == 2


def test_experiments_follow_project_deletion(tmp_path):
    database = Database(tmp_path / "runs.db")
    repository = ProjectRepository(database)
    project_id = repository.create_project("Dataset B")
    repository.record_ml_experiment(project_id, {}, {}, {})

    database.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    assert repository.list_ml_experiments(project_id) == []
