"""FastAPI router for ML Graph execution, validation, and node discovery."""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.ml.graph.engine import GraphEngine, GraphError
from app.ml.graph.node_registry import get_registry_summary

router = APIRouter(prefix="/api/graph", tags=["ML Graph"])


# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================

class NodeModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    spec_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    position: Tuple[float, float] = (0.0, 0.0)


class GraphDefinition(BaseModel):
    nodes: List[NodeModel]
    connections: List[List[str]]  # [source, output, target, input_name]


class GraphRunRequest(BaseModel):
    graph: GraphDefinition
    dataframe: Optional[List[Dict[str, Any]]] = None  # Rows as list of dicts
    csv_content: Optional[str] = None  # Raw CSV string
    random_state: int = 42
    auto_fix_leakage: bool = False
    allow_leakage_warning: bool = True


def _make_json_safe(val: Any) -> Any:
    """Ensure value can be returned in JSON response."""
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_make_json_safe(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _make_json_safe(v) for k, v in val.items()}
    if isinstance(val, pd.DataFrame):
        return {
            "columns": list(val.columns),
            "rows": val.head(50).to_dict(orient="records"),
            "total_rows": len(val),
        }
    return str(val)


# =====================================================================
# ENDPOINTS
# =====================================================================

@router.get("/nodes")
def list_available_nodes():
    """Return catalog of all available graph nodes (auto-discovered and plugins)."""
    return {"nodes": get_registry_summary()}


@router.post("/validate")
def validate_graph(definition: GraphDefinition):
    """Validate graph topology, port types, and detect potential data leakage."""
    engine = GraphEngine()
    for n in definition.nodes:
        if n.spec_id not in engine.specs:
            return {
                "valid": False,
                "errors": [f"Node spec tidak dikenal: {n.spec_id}"],
                "warnings": [],
                "leakages": [],
            }
        engine.add_node(n.spec_id, n.id, n.parameters, n.position)

    for conn in definition.connections:
        if len(conn) != 4:
            return {
                "valid": False,
                "errors": [f"Format koneksi tidak valid: {conn}"],
                "warnings": [],
                "leakages": [],
            }
        try:
            engine.connect(conn[0], conn[1], conn[2], conn[3])
        except GraphError as exc:
            return {
                "valid": False,
                "errors": [str(exc)],
                "warnings": [],
                "leakages": [],
            }

    validation = engine.validate_graph()
    return validation


@router.post("/run")
def execute_graph(req: GraphRunRequest):
    """Execute ML experiment graph and return evaluation metrics & results."""
    start_time = time.perf_counter()
    engine = GraphEngine()

    # Reconstruct nodes
    for n in req.graph.nodes:
        if n.spec_id not in engine.specs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Node spec '{n.spec_id}' tidak terdaftar pada engine.",
            )
        engine.add_node(n.spec_id, n.id, n.parameters, n.position)

    # Reconstruct connections
    for conn in req.graph.connections:
        if len(conn) != 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Koneksi harus berupa [source, output, target, input_name], got: {conn}",
            )
        try:
            engine.connect(conn[0], conn[1], conn[2], conn[3])
        except GraphError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Parse dataset
    df: Optional[pd.DataFrame] = None
    if req.dataframe is not None:
        df = pd.DataFrame(req.dataframe)
    elif req.csv_content is not None:
        df = pd.read_csv(io.StringIO(req.csv_content))

    # Data leakage handling
    leakages = engine.detect_leakage()
    fixes_applied = 0
    if leakages and req.auto_fix_leakage:
        fixes_applied = engine.auto_fix_leakage()
        leakages = engine.detect_leakage()

    if leakages and not req.allow_leakage_warning:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=leakages[0]["message"],
        )

    # Execute graph
    engine.strict_leakage = not req.allow_leakage_warning
    try:
        raw_results = engine.run(dataframe=df)
    except GraphError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kesalahan saat mengeksekusi graf: {exc}",
        ) from exc

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    safe_results = {k: _make_json_safe(v) for k, v in raw_results.items()}

    return {
        "success": True,
        "results": safe_results,
        "warnings": [l["message"] for l in leakages],
        "leakage_detected": len(leakages) > 0,
        "fixes_applied": fixes_applied,
        "random_state": req.random_state,
        "duration_ms": duration_ms,
    }


class InstallRequest(BaseModel):
    package_name: str
    user_confirmed: bool = False


@router.post("/dependencies/preflight")
def check_dependencies(definition: GraphDefinition):
    """Preflight check: verify that all packages required by graph nodes are installed."""
    from app.ml.graph.dependencies import preflight_check
    engine = GraphEngine()
    nodes = [n.model_dump() for n in definition.nodes]
    return preflight_check(nodes, engine.specs)


@router.post("/dependencies/install")
def install_missing_package(req: InstallRequest):
    """Safely install a whitelisted package into the active .venv with user confirmation."""
    from app.ml.graph.dependencies import install_package
    try:
        res = install_package(req.package_name, user_confirmed=req.user_confirmed)
        return res
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


class TaskDetectRequest(BaseModel):
    dataframe: Optional[List[Dict[str, Any]]] = None
    csv_content: Optional[str] = None
    target_column: Optional[str] = None


class CustomTemplateRequest(BaseModel):
    name: str
    description: str = ""
    graph: GraphDefinition


@router.post("/task/detect")
def detect_dataset_task(req: TaskDetectRequest):
    """Detect task type and profile dataset."""
    from app.ml.graph.task_detector import detect_task
    df = None
    if req.dataframe is not None:
        df = pd.DataFrame(req.dataframe)
    elif req.csv_content is not None:
        df = pd.read_csv(io.StringIO(req.csv_content))
    else:
        raise HTTPException(status_code=400, detail="Dataframe atau csv_content wajib disertakan.")
    return detect_task(df, req.target_column)


@router.post("/task/templates")
def get_task_templates(req: TaskDetectRequest):
    """Generate dynamic, ready-to-run template graphs for dataset."""
    from app.ml.graph.task_detector import generate_template_graphs
    df = None
    if req.dataframe is not None:
        df = pd.DataFrame(req.dataframe)
    elif req.csv_content is not None:
        df = pd.read_csv(io.StringIO(req.csv_content))
    else:
        raise HTTPException(status_code=400, detail="Dataframe atau csv_content wajib disertakan.")
    return {"templates": generate_template_graphs(df, req.target_column)}


@router.get("/templates/custom")
def list_saved_templates():
    from app.ml.graph.task_detector import list_custom_templates
    return {"custom_templates": list_custom_templates()}


@router.post("/templates/custom")
def save_new_custom_template(req: CustomTemplateRequest):
    from app.ml.graph.task_detector import save_custom_template
    graph_dict = req.graph.model_dump()
    return save_custom_template(req.name, graph_dict, req.description)


@router.delete("/templates/custom/{template_id}")
def delete_saved_template(template_id: str):
    from app.ml.graph.task_detector import delete_custom_template
    deleted = delete_custom_template(template_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template tidak ditemukan.")
    return {"success": True}



