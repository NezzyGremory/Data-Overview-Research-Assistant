from __future__ import annotations

from pathlib import Path

import pytest

from app.ai.evaluation_benchmark import (
    BenchmarkError,
    BenchmarkOutput,
    BenchmarkTarget,
    TDEBBenchmarkBuilder,
    TDEBBenchmarkRunner,
    TDEBBenchmarkStore,
)
from app.ai.fine_tuning import TrainingHardware
from app.ai.tdid_dataset import TDIDDatasetBuilder


def _training_dataset():
    return TDIDDatasetBuilder(
        [
            {
                "record_id": f"train-{index}",
                "messages": [
                    {"role": "user", "content": f"Training prompt {index}"},
                    {"role": "assistant", "content": f"Training answer {index}"},
                ],
                "metadata": {
                    "category": "statistics",
                    "difficulty": "beginner",
                    "instruction_type": "knowledge",
                    "verified": True,
                    "source": f"training-reference-{index}",
                    "source_group": f"training-source-{index}",
                },
            }
            for index in range(3)
        ]
    )


def _hardware():
    return TrainingHardware(8, 16, 8, False, None, False, None, None, None, None, None)


def _cases():
    return TDEBBenchmarkBuilder(
        [
            {
                "case_id": "heldout-1",
                "prompt": "What does a median describe?",
                "category": "statistics",
                "expected_answer": "The median is the middle value.",
                "required_terms": ["middle value"],
                "expected_citations": ["S1"],
                "expected_tools": ["get_statistical_summary"],
                "source_group": "benchmark-source-1",
            },
            {
                "case_id": "heldout-2",
                "prompt": "How should uncertainty be reported?",
                "category": "research_methodology",
                "required_terms": ["limitations"],
                "source_group": "benchmark-source-2",
            },
        ]
    )


def test_benchmark_version_requires_leakage_check_and_rejects_training_overlap(tmp_path: Path):
    store = TDEBBenchmarkStore(tmp_path / "benchmark")
    cases = _cases()
    training = _training_dataset()
    version = store.save_version("heldout-v1", cases, training_dataset=training)

    assert version.leakage_checked
    assert version.leakage_report["independent"]
    assert len(store.load_version(version.version_id)) == 2

    overlapping = TDEBBenchmarkBuilder(
        [
            {
                "case_id": "leak",
                "prompt": "Training prompt 0",
                "category": "statistics",
                "source_group": "benchmark-source",
            }
        ]
    )
    with pytest.raises(BenchmarkError, match="overlap"):
        store.save_version("leaky", overlapping, training_dataset=training)

    unchecked = store.save_version("unchecked", cases)
    runner = TDEBBenchmarkRunner(store, hardware_probe=_hardware)
    with pytest.raises(BenchmarkError, match="leakage check"):
        runner.run(unchecked.version_id, BenchmarkTarget("test", "mock", "base", lambda _case: ""))


def test_benchmark_runs_save_real_outputs_latency_and_transparent_proxy_metrics(tmp_path: Path):
    store = TDEBBenchmarkStore(tmp_path / "benchmark")
    version = store.save_version("heldout-v1", _cases(), training_dataset=_training_dataset())
    runner = TDEBBenchmarkRunner(store, hardware_probe=_hardware)
    target = BenchmarkTarget(
        name="Mock local model",
        model_name="test-model",
        mode="base+rag+tools",
        generate=lambda case: BenchmarkOutput(
            answer=(
                "The median is the middle value."
                if case.case_id == "heldout-1"
                else "Report limitations clearly."
            ),
            citations=("S1",),
            tool_calls=("get_statistical_summary",),
        ),
    )

    run = runner.run(version.version_id, target)
    assert run.status == "completed"
    assert run.summary["case_count"] == 2
    assert run.summary["exact_match_rate"] == 1.0
    assert run.summary["required_term_recall"] == 1.0
    assert run.summary["citation_recall"] == 1.0
    assert run.summary["tool_coverage"] == 1.0
    assert run.summary["mean_latency_seconds"] >= 0
    assert "do not measure factual accuracy" in run.summary["quality_note"]

    review = {"factual_accuracy": 4, "reasoning_quality": 3, "citation_accuracy": 5}
    store.review_case(
        run.run_id,
        "heldout-1",
        scores=review,
        hallucination_observed=False,
        notes="Human-reviewed example.",
    )
    assert store.get_reviews(run.run_id)["heldout-1"]["scores"] == review

    second = runner.run(
        version.version_id,
        BenchmarkTarget("Second model", "test-model-2", "base", lambda _case: "different response"),
    )
    assert {item.run_id for item in store.compare_runs([run.run_id, second.run_id])} == {run.run_id, second.run_id}


def test_benchmark_records_per_case_failures_without_inventing_metrics(tmp_path: Path):
    store = TDEBBenchmarkStore(tmp_path / "benchmark")
    version = store.save_version("heldout-v1", _cases(), training_dataset=_training_dataset())

    def generate(case):
        if case.case_id == "heldout-1":
            raise ConnectionError("Ollama unavailable")
        return "limitations should be reported"

    run = TDEBBenchmarkRunner(store, hardware_probe=_hardware).run(
        version.version_id,
        BenchmarkTarget("local", "test-model", "base", generate),
    )

    assert run.summary["error_count"] == 1
    assert run.case_results[0].error.startswith("ConnectionError:")
    assert run.summary["exact_match_rate"] is None
    assert len(store.list_runs(version.version_id)) == 1

