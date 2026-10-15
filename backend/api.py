"""FastAPI receiver for privacy-limited desktop usage events."""

from __future__ import annotations

import os
import hmac
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import mysql.connector
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from backend.graph_api import router as graph_router

app = FastAPI(title="Dataset Research Usage API", docs_url=None, redoc_url=None)
app.include_router(graph_router)
logger = logging.getLogger("dataset_research.telemetry_api")
# Local development convenience: load backend/.env when present. Existing process
# environment (e.g. production secret injection) always takes precedence.
load_dotenv(Path(__file__).with_name(".env"), override=False)
EventName = Literal["app_open", "analysis_started", "analysis_completed", "analysis_failed"]
EventStatus = Literal["started", "completed", "failed"]


class UsageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    installation_id: str = Field(min_length=36, max_length=36, pattern=r"^[0-9a-fA-F-]{36}$")
    event_name: EventName
    feature_name: str | None = Field(default=None, max_length=80)
    status: EventStatus
    duration_ms: int | None = Field(default=None, ge=0, le=86_400_000)
    app_version: str | None = Field(default=None, max_length=30)
    created_at: datetime


def _connect():
    required = ("MYSQL_HOST", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing database configuration: " + ", ".join(missing))
    ssl_ca = os.getenv("MYSQL_SSL_CA") or None
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        connection_timeout=5,
        # MySQL accounts configured with REQUIRE SSL need TLS even for local
        # development. Verify the server identity when a CA is configured.
        ssl_disabled=False,
        ssl_ca=ssl_ca,
        ssl_verify_cert=bool(ssl_ca),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/events", status_code=status.HTTP_202_ACCEPTED)
def receive_event(event: UsageEvent, authorization: str | None = Header(default=None)):
    # Optional shared API credential for controlled deployments. Prefer placing
    # the service behind an authenticated gateway; never ship this value in the app.
    expected_token = os.getenv("TELEMETRY_API_TOKEN", "")
    if expected_token:
        supplied_token = (authorization or "").removeprefix("Bearer ")
        expected_hash = hashlib.sha256(expected_token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(supplied_token, expected_hash):
            raise HTTPException(status_code=401, detail="unauthorized")
    # Require the status to agree with event type; this keeps reports reliable.
    expected = {
        "app_open": "started",
        "analysis_started": "started",
        "analysis_completed": "completed",
        "analysis_failed": "failed",
    }[event.event_name]
    if event.status != expected:
        raise HTTPException(status_code=422, detail="status does not match event_name")
    created_at = event.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    connection = None
    try:
        connection = _connect()
        cursor = connection.cursor()
        cursor.execute(
            """INSERT INTO app_events
               (installation_id, event_name, feature_name, status, duration_ms, app_version, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (event.installation_id, event.event_name, event.feature_name, event.status,
             event.duration_ms, event.app_version, created_at),
        )
        connection.commit()
        cursor.close()
    except Exception as exc:
        if connection is not None:
            connection.rollback()
        # Log the failure type/message server-side for local diagnosis. The
        # client still receives a generic response without database details.
        logger.exception("Could not store telemetry event (%s)", type(exc).__name__)
        # Avoid exposing connection details or credentials to clients.
        raise HTTPException(status_code=503, detail="event storage unavailable") from exc
    finally:
        if connection is not None:
            connection.close()
    return {"accepted": True}
