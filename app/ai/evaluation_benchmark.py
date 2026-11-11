"""Versioned held-out Tanya DORA benchmark datasets and local evaluation runs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from app.ai.fine_tuning import TrainingHardware, detect_training_hardware
from app.ai.tdid_dataset import TDIDDatasetBuilder


class BenchmarkError(RuntimeError):
    """Raised when a benchmark set or run is invalid."""


@dataclass(frozen=True, slots=True)
class TDEBCase:
    case_id: str
    prompt: str
    category: str
    expected_answer: str = ""
    required_terms: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    source_group: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkIssue:
    case_id: str
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetVersion:
    version_id: str
    name: str
    created_at: str
    case_count: int
    sha256: str
    snapshot_path: str
    leakage_checked: bool
    leakage_report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkOutput:
    answer: str
    citations: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    name: str
    model_name: str
    mode: str
    generate: Callable[[TDEBCase], Any]

    @classmethod
    def ollama(cls, manager: Any, model_name: str, *, mode: str = "base") -> "BenchmarkTarget":
        selected = str(model_name or "").strip()
        if not selected:
            raise ValueError("Select an installed local Ollama model for the benchmark.")
        return cls(
            name=f"{selected} ({mode})",
            model_name=selected,
            mode=mode,
            generate=lambda case: manager.generate(
                case.prompt,
                model=selected,
                temperature=0,
                system=(
                    "Answer the benchmark question accurately and concisely. "
                    "Do not invent measurements, papers, citations, or tool results."
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    case_id: str
    answer: str
    citations: tuple[str, ...]
    tool_calls: tuple[str, ...]
    latency_seconds: float
    exact_match: bool | None
    required_term_recall: float | None
    citation_recall: float | None
    tool_coverage: float | None
    error: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    run_id: str
    version_id: str
    target_name: str
    model_name: str
    mode: str
    status: str
    created_at: str
    case_results: tuple[BenchmarkCaseResult, ...]
    summary: dict[str, Any]
    hardware_before: dict[str, Any]
    hardware_after: dict[str, Any]


class TDEBBenchmarkBuilder:
    MAX_CASES = 20_000
    MAX_PROMPT_CHARS = 20_000

    def __init__(self, cases: Iterable[TDEBCase | Mapping[str, Any]] = ()) -> None:
        self.cases = [_case_from_value(case) for case in cases]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TDEBBenchmarkBuilder":
        source = Path(path).expanduser()
        if not source.is_file():
            raise BenchmarkError("Benchmark JSONL file does not exist.")
        cases: list[dict[str, Any]] = []
        issues: list[BenchmarkIssue] = []
        try:
            with source.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, Mapping):
                            raise ValueError("Each JSONL row must be an object.")
                        cases.append(dict(row))
                    except (ValueError, TypeError) as error:
                        issues.append(BenchmarkIssue(str(line_number), "json", str(error)))
        except OSError as error:
            raise BenchmarkError(f"Benchmark JSONL could not be read: {error}") from error
        builder = cls(cases)
        builder.import_issues = issues
        return builder

    def validate(self) -> tuple[BenchmarkIssue, ...]:
        issues = list(getattr(self, "import_issues", ()))
        if len(self.cases) > self.MAX_CASES:
            issues.append(BenchmarkIssue("", "cases", f"At most {self.MAX_CASES:,} cases are allowed."))
        ids: dict[str, int] = {}
        prompts: dict[str, int] = {}
        for index, case in enumerate(self.cases, start=1):
            if not case.case_id.strip():
                issues.append(BenchmarkIssue(str(index), "case_id", "Case ID cannot be empty."))
            elif case.case_id in ids:
                issues.append(BenchmarkIssue(case.case_id, "case_id", "Case ID is duplicated."))
            else:
                ids[case.case_id] = index
            prompt = _normalize(case.prompt)
            if not prompt:
                issues.append(BenchmarkIssue(case.case_id or str(index), "prompt", "Prompt cannot be empty."))
            elif len(case.prompt) > self.MAX_PROMPT_CHARS:
                issues.append(BenchmarkIssue(case.case_id, "prompt", "Prompt exceeds the size limit."))
            elif prompt in prompts:
                issues.append(BenchmarkIssue(case.case_id, "prompt", "Prompt duplicates another benchmark case."))
            else:
                prompts[prompt] = index
            if not case.category.strip():
                issues.append(BenchmarkIssue(case.case_id or str(index), "category", "Category cannot be empty."))
            for field_name, values in (
                ("required_terms", case.required_terms),
                ("expected_citations", case.expected_citations),
                ("expected_tools", case.expected_tools),
            ):
                if any(not str(value).strip() for value in values):
                    issues.append(BenchmarkIssue(case.case_id or str(index), field_name, "Values cannot be empty."))
        return tuple(issues)

    def leakage_report(self, training_dataset: TDIDDatasetBuilder) -> dict[str, Any]:
        training_prompts = {
            _normalize(str(message.get("content") or ""))
            for record in training_dataset.records
            for message in record.get("messages", ())
            if isinstance(message, Mapping) and message.get("role") == "user"
        }
        training_groups = {
            _normalize(str(metadata.get(key) or ""))
            for record in training_dataset.records
            if isinstance((metadata := record.get("metadata")), Mapping)
            for key in ("source_group", "source_id", "source")
            if metadata.get(key)
        }
        overlaps = []
        source_overlaps = []
        for case in self.cases:
            if _normalize(case.prompt) in training_prompts:
                overlaps.append(case.case_id)
            if case.source_group and _normalize(case.source_group) in training_groups:
                source_overlaps.append(case.case_id)
        return {
            "checked": True,
            "training_record_count": len(training_dataset.records),
            "prompt_overlap_case_ids": overlaps,
            "source_group_overlap_case_ids": source_overlaps,
            "independent": not overlaps and not source_overlaps,
        }

    def export_jsonl(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for case in self.cases:
                handle.write(json.dumps(asdict(case), ensure_ascii=False, allow_nan=False) + "\n")
        return destination


class TDEBBenchmarkStore:
    def __init__(self, storage_dir: str | Path | None = None) -> None:
        if storage_dir is None:
            from app.core.config import get_data_dir

            storage_dir = get_data_dir() / "tanya_dora" / "benchmark"
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.version_dir = self.storage_dir / "versions"
        self.db_path = self.storage_dir / "tdeb.db"
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS benchmark_versions (
                    version_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
                    case_count INTEGER NOT NULL, sha256 TEXT NOT NULL, snapshot_path TEXT NOT NULL,
                    leakage_checked INTEGER NOT NULL, leakage_report_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY, version_id TEXT NOT NULL, target_name TEXT NOT NULL,
                    model_name TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, result_json TEXT NOT NULL,
                    hardware_before_json TEXT NOT NULL, hardware_after_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS benchmark_reviews (
                    run_id TEXT NOT NULL, case_id TEXT NOT NULL, scores_json TEXT NOT NULL,
                    hallucination_observed INTEGER NOT NULL, notes TEXT NOT NULL,
                    PRIMARY KEY(run_id, case_id)
                )"""
            )

    def save_version(
        self,
        name: str,
        builder: TDEBBenchmarkBuilder,
        *,
        training_dataset: TDIDDatasetBuilder | None = None,
    ) -> BenchmarkDatasetVersion:
        normalized_name = " ".join(str(name or "").split())
        if not normalized_name or len(normalized_name) > 100:
            raise ValueError("Benchmark version name must contain 1-100 characters.")
        issues = builder.validate()
        if issues:
            raise BenchmarkError(f"Cannot save benchmark data with {len(issues)} validation issue(s).")
        if not builder.cases:
            raise BenchmarkError("Benchmark dataset must contain at least one case.")
        report = builder.leakage_report(training_dataset) if training_dataset is not None else {
            "checked": False,
            "independent": False,
            "prompt_overlap_case_ids": [],
            "source_group_overlap_case_ids": [],
        }
        if report.get("checked") and not report.get("independent"):
            raise BenchmarkError("Benchmark cases overlap TDID training prompts or source groups.")
        version_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        snapshot = self.version_dir / f"{version_id}.jsonl"
        builder.export_jsonl(snapshot)
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        relative = snapshot.relative_to(self.storage_dir).as_posix()
        try:
            with self._connection() as connection:
                connection.execute(
                    """INSERT INTO benchmark_versions
                       (version_id, name, created_at, case_count, sha256, snapshot_path,
                        leakage_checked, leakage_report_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (version_id, normalized_name, created_at, len(builder.cases), digest,
                     relative, int(bool(report.get("checked"))), _json(report)),
                )
        except Exception:
            snapshot.unlink(missing_ok=True)
            raise
        return self.get_version(version_id)

    def get_version(self, version_id: str) -> BenchmarkDatasetVersion:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM benchmark_versions WHERE version_id = ?", (str(version_id),)
            ).fetchone()
        if row is None:
            raise BenchmarkError("Benchmark dataset version was not found.")
        report = _json_object(row["leakage_report_json"])
        return BenchmarkDatasetVersion(
            version_id=str(row["version_id"]), name=str(row["name"]),
            created_at=str(row["created_at"]), case_count=int(row["case_count"]),
            sha256=str(row["sha256"]), snapshot_path=str(row["snapshot_path"]),
            leakage_checked=bool(row["leakage_checked"]), leakage_report=report,
        )

    def list_versions(self) -> tuple[BenchmarkDatasetVersion, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT version_id FROM benchmark_versions ORDER BY created_at DESC, version_id"
            ).fetchall()
        return tuple(self.get_version(str(row["version_id"])) for row in rows)

    def load_version(self, version_id: str) -> tuple[TDEBCase, ...]:
        version = self.get_version(version_id)
        path = self._resolve_snapshot(version.snapshot_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != version.sha256:
            raise BenchmarkError("Benchmark snapshot hash does not match its manifest.")
        builder = TDEBBenchmarkBuilder.from_jsonl(path)
        issues = builder.validate()
        if issues:
            raise BenchmarkError("Stored benchmark snapshot failed validation.")
        return tuple(builder.cases)

    def record_run(self, run: BenchmarkRun) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO benchmark_runs
                   (run_id, version_id, target_name, model_name, mode, status, created_at,
                    result_json, hardware_before_json, hardware_after_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run.run_id, run.version_id, run.target_name, run.model_name, run.mode,
                 run.status, run.created_at, _json(_run_dict(run)), _json(run.hardware_before),
                 _json(run.hardware_after)),
            )

    def list_runs(self, version_id: str | None = None) -> tuple[BenchmarkRun, ...]:
        with self._connection() as connection:
            if version_id:
                rows = connection.execute(
                    "SELECT * FROM benchmark_runs WHERE version_id = ? ORDER BY created_at DESC",
                    (str(version_id),),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM benchmark_runs ORDER BY created_at DESC"
                ).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def review_case(
        self,
        run_id: str,
        case_id: str,
        *,
        scores: Mapping[str, int],
        hallucination_observed: bool,
        notes: str = "",
    ) -> None:
        run = next((item for item in self.list_runs() if item.run_id == str(run_id)), None)
        if run is None or not any(item.case_id == str(case_id) for item in run.case_results):
            raise BenchmarkError("Benchmark run or case was not found.")
        normalized: dict[str, int] = {}
        for key, value in scores.items():
            rating = int(value)
            if not str(key).strip() or not 1 <= rating <= 5:
                raise ValueError("Human review scores must use named criteria rated from 1 to 5.")
            normalized[str(key)] = rating
        with self._connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO benchmark_reviews
                   (run_id, case_id, scores_json, hallucination_observed, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(run_id), str(case_id), _json(normalized), int(hallucination_observed), str(notes)[:4000]),
            )

    def get_reviews(self, run_id: str) -> dict[str, dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM benchmark_reviews WHERE run_id = ?", (str(run_id),)
            ).fetchall()
        return {
            str(row["case_id"]): {
                "scores": _json_object(row["scores_json"]),
                "hallucination_observed": bool(row["hallucination_observed"]),
                "notes": str(row["notes"]),
            }
            for row in rows
        }

    def compare_runs(self, run_ids: Sequence[str]) -> tuple[BenchmarkRun, ...]:
        if len(run_ids) < 2:
            raise ValueError("Choose at least two benchmark runs to compare.")
        by_id = {run.run_id: run for run in self.list_runs()}
        try:
            runs = tuple(by_id[str(run_id)] for run_id in run_ids)
        except KeyError as error:
            raise BenchmarkError("A selected benchmark run was not found.") from error
        if len({run.version_id for run in runs}) != 1:
            raise BenchmarkError("Models can only be compared on the same held-out dataset version.")
        return runs

    def _resolve_snapshot(self, relative_path: str) -> Path:
        root = self.version_dir.resolve()
        candidate = (self.storage_dir / relative_path).resolve()
        if root not in candidate.parents or candidate.suffix.casefold() != ".jsonl":
            raise BenchmarkError("Benchmark snapshot path is invalid.")
        if not candidate.is_file():
            raise BenchmarkError("Benchmark snapshot file is missing.")
        return candidate


class TDEBBenchmarkRunner:
    def __init__(
        self,
        store: TDEBBenchmarkStore,
        *,
        hardware_probe: Callable[[], TrainingHardware] = detect_training_hardware,
    ) -> None:
        self.store = store
        self.hardware_probe = hardware_probe

    def run(
        self,
        version_id: str,
        target: BenchmarkTarget,
        *,
        cancel_event: threading.Event | None = None,
    ) -> BenchmarkRun:
        version = self.store.get_version(version_id)
        if not version.leakage_checked or not version.leakage_report.get("independent"):
            raise BenchmarkError("Run a leakage check against the TDID training set before benchmarking.")
        cases = self.store.load_version(version_id)
        cancel = cancel_event or threading.Event()
        before = self.hardware_probe()
        results: list[BenchmarkCaseResult] = []
        status = "completed"
        for case in cases:
            if cancel.is_set():
                status = "cancelled"
                break
            start = time.perf_counter()
            answer = ""
            citations: tuple[str, ...] = ()
            tools: tuple[str, ...] = ()
            error_detail = None
            try:
                output = target.generate(case)
                if isinstance(output, str):
                    answer = output
                else:
                    answer = str(getattr(output, "answer", "") or "")
                    citations = tuple(
                        str(value) for value in getattr(output, "citations", ())
                    )
                    tools = tuple(str(value) for value in getattr(output, "tool_calls", ()))
                    if not citations and hasattr(output, "sources"):
                        citations = tuple(
                            str(getattr(item, "source_id", "")) for item in output.sources
                            if getattr(item, "source_id", None)
                        )
                    if not tools and hasattr(output, "tool_results"):
                        tools = tuple(
                            str(getattr(item, "name", "")) for item in output.tool_results
                            if getattr(item, "name", None)
                        )
            except Exception as error:
                error_detail = f"{type(error).__name__}: {error}"
            latency = max(0.0, time.perf_counter() - start)
            results.append(_score_case(case, answer, citations, tools, latency, error_detail))
        after = self.hardware_probe()
        summary = _summarize(results, status)
        run = BenchmarkRun(
            run_id=uuid4().hex,
            version_id=version_id,
            target_name=str(target.name),
            model_name=str(target.model_name),
            mode=str(target.mode),
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            case_results=tuple(results),
            summary=summary,
            hardware_before=asdict(before),
            hardware_after=asdict(after),
        )
        self.store.record_run(run)
        return run


def _case_from_value(value: TDEBCase | Mapping[str, Any]) -> TDEBCase:
    if isinstance(value, TDEBCase):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Benchmark case must be a TDEBCase or mapping.")
    return TDEBCase(
        case_id=str(value.get("case_id") or ""),
        prompt=str(value.get("prompt") or ""),
        category=str(value.get("category") or ""),
        expected_answer=str(value.get("expected_answer") or ""),
        required_terms=_string_tuple(value.get("required_terms", ())),
        expected_citations=_string_tuple(value.get("expected_citations", ())),
        expected_tools=_string_tuple(value.get("expected_tools", ())),
        source_group=str(value.get("source_group") or ""),
    )


def _score_case(
    case: TDEBCase,
    answer: str,
    citations: tuple[str, ...],
    tools: tuple[str, ...],
    latency: float,
    error: str | None,
) -> BenchmarkCaseResult:
    normalized_answer = _normalize(answer)
    exact = _normalize(case.expected_answer) == normalized_answer if case.expected_answer else None
    term_recall = (
        sum(_normalize(term) in normalized_answer for term in case.required_terms) / len(case.required_terms)
        if case.required_terms else None
    )
    citation_recall = (
        sum(expected in citations for expected in case.expected_citations) / len(case.expected_citations)
        if case.expected_citations else None
    )
    tool_coverage = (
        sum(expected in tools for expected in case.expected_tools) / len(case.expected_tools)
        if case.expected_tools else None
    )
    return BenchmarkCaseResult(
        case_id=case.case_id,
        answer=answer,
        citations=citations,
        tool_calls=tools,
        latency_seconds=latency,
        exact_match=exact,
        required_term_recall=term_recall,
        citation_recall=citation_recall,
        tool_coverage=tool_coverage,
        error=error,
    )


def _summarize(results: Sequence[BenchmarkCaseResult], status: str) -> dict[str, Any]:
    successful = [item for item in results if item.error is None]
    latencies = [item.latency_seconds for item in successful]
    summary: dict[str, Any] = {
        "status": status,
        "case_count": len(results),
        "completed_count": len(successful),
        "error_count": sum(item.error is not None for item in results),
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "p50_latency_seconds": _percentile(latencies, 0.50),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "exact_match_rate": _mean_optional([item.exact_match for item in successful]),
        "required_term_recall": _mean_optional([item.required_term_recall for item in successful]),
        "citation_recall": _mean_optional([item.citation_recall for item in successful]),
        "tool_coverage": _mean_optional([item.tool_coverage for item in successful]),
        "quality_note": "Automatic exact/term/citation checks are proxies; they do not measure factual accuracy.",
        "human_rated_quality": None,
    }
    return summary


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * quantile)))
    return sorted_values[index]


def _mean_optional(values: Sequence[Any]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return statistics.fmean(selected) if selected else None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("Benchmark list fields must be arrays of strings.")
    return tuple(str(item) for item in value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_dict(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "version_id": run.version_id,
        "target_name": run.target_name,
        "model_name": run.model_name,
        "mode": run.mode,
        "status": run.status,
        "created_at": run.created_at,
        "case_results": [asdict(item) for item in run.case_results],
        "summary": run.summary,
    }


def _run_from_row(row: sqlite3.Row) -> BenchmarkRun:
    result = _json_object(row["result_json"])
    cases = tuple(BenchmarkCaseResult(**item) for item in result.get("case_results", ()) if isinstance(item, Mapping))
    return BenchmarkRun(
        run_id=str(row["run_id"]), version_id=str(row["version_id"]),
        target_name=str(row["target_name"]), model_name=str(row["model_name"]),
        mode=str(row["mode"]), status=str(row["status"]), created_at=str(row["created_at"]),
        case_results=cases, summary=dict(result.get("summary") or {}),
        hardware_before=_json_object(row["hardware_before_json"]),
        hardware_after=_json_object(row["hardware_after_json"]),
    )
