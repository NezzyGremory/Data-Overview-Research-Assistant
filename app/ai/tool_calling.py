"""Bounded, read-only tool registry for Tanya DORA."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolCallingError(RuntimeError):
    """Raised when a model requests an invalid or unavailable DORA tool."""


@dataclass(frozen=True, slots=True)
class DoraTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    requires_confirmation: bool = False

    def to_ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: Any
    success: bool
    requires_confirmation: bool = False

    @property
    def provides_evidence(self) -> bool:
        if not self.success:
            return False
        if not isinstance(self.result, Mapping):
            return True
        if self.result.get("available") is False:
            return False
        if self.result.get("status") == "NO_RESULTS":
            return False
        if "result_count" in self.result and self.result.get("result_count") == 0:
            if "sources" in self.result or "papers" in self.result:
                return False
        for key in ("sources", "papers"):
            value = self.result.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and not value:
                return False
        return not bool(self.result.get("error"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "requires_confirmation": self.requires_confirmation,
        }

    def to_json(self) -> str:
        return json.dumps(_json_safe(self.to_dict()), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True)
class ToolChatResult:
    answer: str
    executions: tuple[ToolExecution, ...]
    cancelled: bool = False


class DoraToolRegistry:
    """Register explicit functions; never evaluate model-produced code."""

    MAX_ARGUMENTS = 12
    MAX_RESULT_BYTES = 64_000

    def __init__(self) -> None:
        self._tools: dict[str, DoraTool] = {}

    def register(self, tool: DoraTool) -> None:
        name = str(tool.name or "").strip()
        if not name or not name.replace("_", "").isalnum():
            raise ValueError("Tool name must contain only letters, numbers, or underscores.")
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        if not isinstance(tool.parameters, dict) or tool.parameters.get("type") != "object":
            raise ValueError("Tool parameters must use a JSON object schema.")
        properties = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        if not isinstance(properties, dict) or len(properties) > self.MAX_ARGUMENTS:
            raise ValueError("Tool schema has too many or invalid properties.")
        if not isinstance(required, list) or any(key not in properties for key in required):
            raise ValueError("Tool schema has invalid required properties.")
        self._tools[name] = tool

    def definitions(self) -> tuple[dict[str, Any], ...]:
        return tuple(tool.to_ollama_schema() for tool in self._tools.values())

    def has_tools(self) -> bool:
        return bool(self._tools)

    def execute(self, name: str, arguments: Mapping[str, Any] | None) -> ToolExecution:
        normalized_name = str(name or "").strip()
        tool = self._tools.get(normalized_name)
        if tool is None:
            return ToolExecution(
                normalized_name,
                {},
                {"error": "Tool tidak tersedia untuk sesi ini."},
                False,
            )
        if not isinstance(arguments, Mapping):
            return ToolExecution(normalized_name, {}, {"error": "Argumen tool harus berupa objek."}, False)
        normalized_arguments = dict(arguments)
        properties = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        unknown = set(normalized_arguments) - set(properties)
        missing = set(required) - set(normalized_arguments)
        invalid = False
        for key, value in normalized_arguments.items():
            property_schema = properties.get(key)
            if not isinstance(property_schema, Mapping):
                invalid = True
                break
            expected_type = property_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                invalid = True
            elif expected_type == "integer" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                invalid = True
            elif expected_type == "number" and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                invalid = True
            elif expected_type == "boolean" and not isinstance(value, bool):
                invalid = True
            if "enum" in property_schema and value not in property_schema["enum"]:
                invalid = True
            if expected_type in {"integer", "number"} and not isinstance(value, bool):
                if "minimum" in property_schema and value < property_schema["minimum"]:
                    invalid = True
                if "maximum" in property_schema and value > property_schema["maximum"]:
                    invalid = True
        if unknown or missing or invalid:
            return ToolExecution(
                normalized_name,
                _json_safe(normalized_arguments),
                {"error": "Argumen tool tidak sesuai dengan schema."},
                False,
            )
        if tool.requires_confirmation:
            return ToolExecution(
                normalized_name,
                _json_safe(normalized_arguments),
                {"error": "Tool ini memerlukan konfirmasi pengguna sebelum dijalankan."},
                False,
                requires_confirmation=True,
            )
        try:
            result = _json_safe(tool.handler(normalized_arguments))
            serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
            if len(serialized.encode("utf-8")) > self.MAX_RESULT_BYTES:
                raise ToolCallingError("Hasil tool melebihi batas ukuran respons.")
        except Exception as error:
            detail = str(error).strip() or "Tool tidak dapat dijalankan."
            result = {"error": detail[:500]}
            return ToolExecution(normalized_name, _json_safe(normalized_arguments), result, False)
        return ToolExecution(normalized_name, _json_safe(normalized_arguments), result, True)


def build_dora_tool_registry(
    context: Mapping[str, Any] | None,
    knowledge_base: Any,
    *,
    allow_online_research: bool = False,
    include_knowledge_base: bool = True,
) -> DoraToolRegistry:
    """Build the read-only tool set from a UI-thread snapshot of DORA state."""
    snapshot = dict(context) if isinstance(context, Mapping) else {}
    dataset = snapshot.get("active_dataset")
    analysis = snapshot.get("analysis")
    dataset = dataset if isinstance(dataset, Mapping) else None
    analysis = analysis if isinstance(analysis, Mapping) else {}
    registry = DoraToolRegistry()

    registry.register(
        DoraTool(
            "get_active_dataset",
            "Read the active dataset name, shape, column names and data types. Does not read rows.",
            _object_schema({}),
            lambda _args: (
                {"available": False, "detail": "Tidak ada dataset aktif."}
                if dataset is None
                else {"available": True, **_json_safe(dataset)}
            ),
        )
    )

    def add_analysis_tool(name: str, description: str, result_key: str, schema=None, handler=None):
        def read_result(arguments):
            if handler is not None:
                return handler(arguments)
            result = analysis.get(result_key)
            if result is None:
                return {
                    "available": False,
                    "detail": f"DORA belum memiliki hasil '{result_key}' untuk dataset aktif.",
                }
            return {"available": True, "dataset": dataset, "result": result}

        registry.register(DoraTool(name, description, schema or _object_schema({}), read_result))

    add_analysis_tool(
        "get_dataset_profile",
        "Read the already computed DORA dataset profile. Do not estimate row or column counts.",
        "profile",
    )
    add_analysis_tool(
        "get_missing_values",
        "Read DORA's computed missing-value analysis for the active dataset.",
        "missing_values",
    )
    add_analysis_tool(
        "get_duplicate_analysis",
        "Read DORA's computed duplicate-row analysis for the active dataset.",
        "duplicates",
    )

    def statistics(arguments):
        result = analysis.get("statistics")
        if result is None:
            return {"available": False, "detail": "DORA belum menghitung statistik dataset ini."}
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
            return {"available": True, "dataset": dataset, "result": result}
        column = str(arguments.get("column") or "").strip()
        rows = [item for item in result if isinstance(item, Mapping)]
        if column:
            rows = [item for item in rows if str(item.get("column")) == column]
            if not rows:
                return {"available": False, "detail": f"Kolom '{column}' tidak memiliki statistik tersimpan."}
        return {"available": True, "dataset": dataset, "result": rows[:100]}

    add_analysis_tool(
        "get_statistical_summary",
        "Read computed DORA column statistics; pass a column name to limit the response.",
        "statistics",
        _object_schema(
            {"column": {"type": "string", "description": "Optional exact dataset column name."}}
        ),
        statistics,
    )
    add_analysis_tool(
        "get_correlation_results",
        "Read DORA's computed correlation result. Never derive correlation values in the model.",
        "correlations",
    )

    if include_knowledge_base:
        categories = sorted(getattr(knowledge_base, "CATEGORIES", ()))

        def search_knowledge(arguments):
            query = str(arguments.get("query") or "").strip()
            if not query:
                raise ValueError("Query pencarian DKB tidak boleh kosong.")
            category = str(arguments.get("category") or "").strip() or None
            limit = max(1, min(int(arguments.get("limit", 3)), 5))
            results = knowledge_base.search(query, category=category, limit=limit)
            return {
                "query": query,
                "result_count": len(results),
                "sources": [
                    {
                        "citation": result.citation,
                        "title": result.title,
                        "author_or_org": result.author_or_org,
                        "year": result.publication_year,
                        "category": result.category,
                        "page_or_section": result.page_or_section,
                        "doi": result.doi,
                        "url": result.source_url,
                        "excerpt": result.text[:1800],
                    }
                    for result in results
                ],
            }

        category_schema: dict[str, Any] = {"type": "string"}
        if categories:
            category_schema["enum"] = categories
        registry.register(
            DoraTool(
                "search_knowledge_base",
                "Search locally indexed DKB documents. Use returned source metadata for citations.",
                _object_schema(
                    {
                        "query": {"type": "string", "description": "Short search query."},
                        "category": {**category_schema, "description": "Optional DKB category slug."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    required=("query",),
                ),
                search_knowledge,
            )
        )

    if allow_online_research:
        registry.register(
            DoraTool(
                "search_academic_papers",
                "Search OpenAlex and Crossref for paper metadata. Use only when online research is enabled; report returned providers and sources, and do not claim the search is exhaustive.",
                _object_schema(
                    {
                        "query": {"type": "string", "description": "Academic search terms."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    required=("query",),
                ),
                _search_openalex_crossref,
            )
        )
    return registry


def _search_openalex_crossref(arguments: dict[str, Any]) -> dict[str, Any]:
    query = " ".join(str(arguments.get("query") or "").split())[:500]
    if not query:
        raise ValueError("Query akademik tidak boleh kosong.")
    limit = max(1, min(int(arguments.get("limit", 5)), 10))
    from app.research.crossref import CrossrefClient
    from app.research.deduplication import PaperDeduplicator
    from app.research.openalex import OpenAlexClient

    providers = {
        "OpenAlex": lambda: OpenAlexClient(timeout=12.0).search(query=query, per_page=limit),
        "Crossref": lambda: CrossrefClient(timeout=12.0).search(query=query, rows=limit),
    }
    provider_status: dict[str, str] = {}
    papers = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(search) for name, search in providers.items()}
        for name, future in futures.items():
            try:
                results = future.result(timeout=20)
                provider_status[name] = "SUCCESS"
                papers.extend(results if isinstance(results, list) else [])
            except Exception:
                provider_status[name] = "ERROR"

    unique = PaperDeduplicator().deduplicate(papers)[:limit]
    return {
        "query": query,
        "status": "SUCCESS" if unique else "NO_RESULTS",
        "provider_status": provider_status,
        "result_count": len(unique),
        "search_is_exhaustive": False,
        "papers": [
            {
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "doi": paper.doi,
                "url": paper.url,
                "abstract": str(paper.abstract or "")[:1800],
                "source": paper.source,
            }
            for paper in unique
        ],
    }


def _object_schema(
    properties: dict[str, Any],
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 12:
        return "[nested result omitted]"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key)[:160]: _json_safe(item, _depth=_depth + 1)
            for key, item in list(value.items())[:500]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item, _depth=_depth + 1) for item in value[:500]]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item(), _depth=_depth + 1)
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)[:1000]
