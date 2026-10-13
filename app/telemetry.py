"""Privacy-limited, best-effort usage telemetry for the desktop client."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_data_dir

_ALLOWED_EVENTS = {"app_open", "analysis_started", "analysis_completed", "analysis_failed"}
_ALLOWED_STATUSES = {"started", "completed", "failed"}
_logger = logging.getLogger("dataset_research.telemetry")


def _installation_id() -> str:
    """Return an opaque random install ID, kept outside project/SQLite data."""
    path = get_data_dir() / "installation_id"
    try:
        value = path.read_text(encoding="ascii").strip()
        if len(value) == 36:
            return value
    except OSError:
        pass
    value = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="ascii")
    except OSError:
        # Avoid preventing app use in read-only installations.
        pass
    return value


class TelemetryClient:
    """Sends allowlisted aggregate events; never accepts arbitrary event data."""

    def __init__(self, api_url: str | None = None, timeout: float = 2.0):
        self.api_url = (api_url if api_url is not None else os.getenv("TELEMETRY_API_URL", "")).strip().rstrip("/")
        self.api_token = os.getenv("TELEMETRY_API_TOKEN", "")
        self.timeout = timeout
        self.installation_id = _installation_id() if self.api_url else ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_url)

    def track(self, event_name: str, *, feature_name: str | None = None,
              status: str | None = None, duration_ms: int | None = None) -> None:
        """Queue one event using a fixed schema; network errors are intentionally ignored."""
        if not self.enabled or event_name not in _ALLOWED_EVENTS:
            return
        safe_feature = feature_name if feature_name in {"dataset_analysis"} else None
        safe_status = status if status in _ALLOWED_STATUSES else ("started" if event_name.endswith("started") or event_name == "app_open" else None)
        if duration_ms is not None:
            duration_ms = max(0, min(int(duration_ms), 86_400_000))
        payload: dict[str, Any] = {
            "installation_id": self.installation_id,
            "event_name": event_name,
            "feature_name": safe_feature,
            "status": safe_status or "completed",
            "duration_ms": duration_ms,
            "app_version": os.getenv("APP_VERSION", "5.0.0")[:30],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        threading.Thread(target=self._send, args=(payload,), daemon=True).start()

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                # Hash the server token before attaching it, so the raw secret
                # is never embedded in a distributed desktop package.
                token = hashlib.sha256(self.api_token.encode("utf-8")).hexdigest()
                headers["Authorization"] = f"Bearer {token}"
            request = Request(
                f"{self.api_url}/api/events",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=self.timeout) as response:
                response.read(256)
        except HTTPError as error:
            # Report only HTTP status, never headers/payload/token.
            _logger.warning("Telemetry API rejected an event (HTTP %s).", error.code)
        except (OSError, URLError, ValueError) as error:
            _logger.warning("Telemetry event could not be sent (%s).", type(error).__name__)
            return


_default_client: TelemetryClient | None = None


def get_telemetry() -> TelemetryClient:
    global _default_client
    if _default_client is None:
        _default_client = TelemetryClient()
    return _default_client
