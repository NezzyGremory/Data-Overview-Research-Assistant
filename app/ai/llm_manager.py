"""Local Ollama connection and inference manager.

This module only talks to a loopback Ollama HTTP server. It never pulls or
installs a model; callers can select only a model reported by ``/api/tags``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import threading
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from app.ai.model_registry import LocalModel, ModelRegistry
from app.ai.tool_calling import DoraToolRegistry, ToolChatResult


class LocalLLMError(RuntimeError):
    """Base error for local model discovery and inference failures."""


class OllamaUnavailableError(LocalLLMError):
    """Raised when the configured local Ollama server cannot be reached."""


class ModelNotSelectedError(LocalLLMError):
    """Raised when an inference request has no explicit model selection."""


class ModelNotInstalledError(LocalLLMError):
    """Raised when the requested model is not in Ollama's local inventory."""


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    available: bool
    selected_model: str | None
    installed_model: bool
    running_model: bool
    installed_model_count: int
    detail: str

    @property
    def label(self) -> str:
        if not self.available:
            return "Ollama unavailable"
        if not self.selected_model:
            return "Select a local model"
        if not self.installed_model:
            return "Selected model is not installed"
        if self.running_model:
            return "Model is running"
        return "Model is installed"


class OllamaLLMManager:
    """Discover local models and run chat and embedding requests."""

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    _ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
    _LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = self._validate_local_base_url(base_url)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )
        self._registry = ModelRegistry()
        self._selected_model: str | None = None
        self._stream_lock = threading.Lock()
        self._active_stream: httpx.Response | None = None

    @property
    def selected_model(self) -> str | None:
        return self._selected_model

    def close(self) -> None:
        self.cancel_generation()
        self._client.close()

    def __enter__(self) -> "OllamaLLMManager":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def refresh_models(self) -> tuple[LocalModel, ...]:
        """Refresh the inventory of models already installed on this device."""
        payload = self._request_json("GET", "/api/tags")
        try:
            return self._registry.replace_from_payload(payload)
        except ValueError as error:
            raise LocalLLMError(str(error)) from error

    def list_models(self) -> tuple[LocalModel, ...]:
        """Return cached models; call ``refresh_models`` for a fresh inventory."""
        return self._registry.list_models()

    def select_cached_model(self, name: str) -> LocalModel:
        """Select a model from the last refreshed local inventory without I/O."""
        normalized_name = str(name or "").strip()
        model = self._registry.get(normalized_name)
        if model is None:
            raise ModelNotInstalledError(
                f"Model '{normalized_name}' is not in the last local inventory."
            )
        self._selected_model = model.name
        return model

    def select_model(self, name: str) -> LocalModel:
        """Select an installed model after refreshing Ollama's inventory."""
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ModelNotInstalledError("Select an installed local model first.")

        models = self.refresh_models()
        model = next((item for item in models if item.name == normalized_name), None)
        if model is None:
            raise ModelNotInstalledError(
                f"Model '{normalized_name}' is not installed in local Ollama."
            )

        self._selected_model = model.name
        return model

    def running_models(self) -> tuple[str, ...]:
        """Return names of models currently loaded by the local Ollama server."""
        payload = self._request_json("GET", "/api/ps")
        entries = payload.get("models")
        if not isinstance(entries, list):
            raise LocalLLMError("Ollama running-model response has an invalid format.")

        names = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or entry.get("model") or "").strip()
            if name and name not in names:
                names.append(name)
        return tuple(names)

    def get_status(self) -> OllamaStatus:
        """Report server reachability and the selected model's local state."""
        try:
            models = self.refresh_models()
        except LocalLLMError as error:
            return OllamaStatus(
                available=False,
                selected_model=self._selected_model,
                installed_model=False,
                running_model=False,
                installed_model_count=0,
                detail=str(error),
            )

        installed_names = {model.name for model in models}
        selected_is_installed = self._selected_model in installed_names
        running_names: tuple[str, ...] = ()
        status_detail = f"Found {len(models)} installed local model(s)."
        try:
            running_names = self.running_models()
        except LocalLLMError as error:
            status_detail = f"Ollama is reachable; running-model status is unavailable: {error}"

        return OllamaStatus(
            available=True,
            selected_model=self._selected_model,
            installed_model=selected_is_installed,
            running_model=(
                self._selected_model in running_names
                if self._selected_model
                else False
            ),
            installed_model_count=len(models),
            detail=status_detail,
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate one response for a prompt using an installed local model."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must contain text.")

        messages: list[dict[str, str]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, temperature=temperature)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> tuple[tuple[float, ...], ...]:
        """Create local vectors with an installed Ollama embedding model."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
            raise ValueError("At least one text is required for embedding.")
        normalized_texts = []
        for index, value in enumerate(texts):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Embedding input {index + 1} must contain text.")
            normalized_texts.append(value)

        model_name = str(model or "").strip()
        if not model_name:
            raise ModelNotSelectedError("Select an installed local embedding model first.")
        installed_models = self.refresh_models()
        if not any(item.name == model_name for item in installed_models):
            raise ModelNotInstalledError(
                f"Embedding model '{model_name}' is not installed in local Ollama."
            )

        payload = self._request_json(
            "POST",
            "/api/embed",
            json={"model": model_name, "input": normalized_texts, "truncate": False},
        )
        raw_embeddings = payload.get("embeddings")
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(normalized_texts):
            raise LocalLLMError("Ollama returned an unexpected number of embeddings.")

        embeddings: list[tuple[float, ...]] = []
        expected_dimensions: int | None = None
        for index, raw_vector in enumerate(raw_embeddings):
            if not isinstance(raw_vector, list) or not raw_vector:
                raise LocalLLMError(f"Ollama returned an invalid embedding at position {index + 1}.")
            try:
                vector = tuple(float(value) for value in raw_vector)
            except (TypeError, ValueError, OverflowError) as error:
                raise LocalLLMError(
                    f"Ollama returned a non-numeric embedding at position {index + 1}."
                ) from error
            if not all(math.isfinite(value) for value in vector):
                raise LocalLLMError(f"Ollama returned a non-finite embedding at position {index + 1}.")
            if expected_dimensions is None:
                expected_dimensions = len(vector)
            elif len(vector) != expected_dimensions:
                raise LocalLLMError("Ollama returned embeddings with inconsistent dimensions.")
            embeddings.append(vector)
        return tuple(embeddings)

    def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        on_delta: Callable[[str], None],
        model: str | None = None,
        temperature: float = 0.7,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Stream a local chat response; closing the response stops the stream."""
        normalized_messages = self._normalize_messages(messages)
        normalized_temperature = self._validate_temperature(temperature)
        selected_name = str(model or self._selected_model or "").strip()
        if not selected_name:
            raise ModelNotSelectedError("Select an installed local model first.")
        if cancel_event is not None and cancel_event.is_set():
            return ""

        installed_models = self.refresh_models()
        if not any(item.name == selected_name for item in installed_models):
            raise ModelNotInstalledError(
                f"Model '{selected_name}' is not installed in local Ollama."
            )

        request_json = {
            "model": selected_name,
            "messages": normalized_messages,
            "stream": True,
            "options": {"temperature": normalized_temperature},
        }
        content, _tool_calls, _cancelled = self._stream_chat_request(
            request_json,
            on_delta=on_delta,
            cancel_event=cancel_event,
        )
        return content

    def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tool_registry: DoraToolRegistry,
        on_delta: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, Mapping[str, Any]], None] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        cancel_event: threading.Event | None = None,
        max_tool_rounds: int = 4,
        max_tool_calls: int = 8,
    ) -> ToolChatResult:
        """Run a bounded local Ollama tool loop with registered functions only."""
        normalized_messages = self._normalize_tool_messages(messages)
        normalized_temperature = self._validate_temperature(temperature)
        selected_name = str(model or self._selected_model or "").strip()
        if not selected_name:
            raise ModelNotSelectedError("Select an installed local model first.")
        tool_definitions = tool_registry.definitions()
        if not tool_definitions:
            raise ValueError("At least one registered tool is required.")
        if not 1 <= int(max_tool_rounds) <= 8 or not 1 <= int(max_tool_calls) <= 16:
            raise ValueError("Tool loop limits must be between 1 and 8 rounds and 1 and 16 calls.")
        installed_models = self.refresh_models()
        if not any(item.name == selected_name for item in installed_models):
            raise ModelNotInstalledError(
                f"Model '{selected_name}' is not installed in local Ollama."
            )

        executions = []
        for _round in range(int(max_tool_rounds) + 1):
            if cancel_event is not None and cancel_event.is_set():
                return ToolChatResult("", tuple(executions), True)
            request_json = {
                "model": selected_name,
                "messages": normalized_messages,
                "tools": tool_definitions,
                "stream": True,
                "options": {"temperature": normalized_temperature},
            }
            content, raw_calls, cancelled = self._stream_chat_request(
                request_json,
                on_delta=on_delta,
                cancel_event=cancel_event,
            )
            if cancelled:
                return ToolChatResult(content, tuple(executions), True)
            if not raw_calls:
                return ToolChatResult(content, tuple(executions), False)
            normalized_calls = self._normalize_tool_calls(raw_calls)
            normalized_messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": normalized_calls,
                }
            )
            for tool_call in normalized_calls:
                if cancel_event is not None and cancel_event.is_set():
                    return ToolChatResult(content, tuple(executions), True)
                function = tool_call["function"]
                name = function["name"]
                arguments = function["arguments"]
                if len(executions) >= int(max_tool_calls):
                    raise LocalLLMError("The local model exceeded the allowed number of tool calls.")
                if on_tool_call is not None:
                    on_tool_call(name, arguments)
                execution = tool_registry.execute(name, arguments)
                executions.append(execution)
                normalized_messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": execution.to_json(),
                    }
                )
        raise LocalLLMError("The local model did not finish within the tool-loop limit.")

    def _stream_chat_request(
        self,
        request_json: Mapping[str, Any],
        *,
        on_delta: Callable[[str], None] | None,
        cancel_event: threading.Event | None,
    ) -> tuple[str, tuple[dict[str, Any], ...], bool]:
        parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        try:
            with self._client.stream("POST", "/api/chat", json=request_json) as response:
                with self._stream_lock:
                    self._active_stream = response
                try:
                    if response.is_error:
                        response.read()
                        detail = self._response_error(response)
                        raise LocalLLMError(
                            f"Ollama POST /api/chat failed ({response.status_code}): {detail}"
                        )

                    for line in response.iter_lines():
                        if cancel_event is not None and cancel_event.is_set():
                            break
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except (TypeError, ValueError) as error:
                            raise LocalLLMError("Ollama returned an invalid streaming response.") from error
                        if not isinstance(payload, dict):
                            raise LocalLLMError("Ollama returned an invalid streaming response.")
                        if payload.get("error"):
                            raise LocalLLMError(str(payload["error"]))
                        message = payload.get("message")
                        delta = message.get("content") if isinstance(message, Mapping) else None
                        if delta:
                            if not isinstance(delta, str):
                                raise LocalLLMError("Ollama returned a non-text chat delta.")
                            parts.append(delta)
                            if on_delta is not None:
                                on_delta(delta)
                        raw_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
                        if isinstance(raw_calls, list):
                            tool_calls.extend(
                                dict(item) for item in raw_calls if isinstance(item, Mapping)
                            )
                        if payload.get("done"):
                            break
                except (httpx.HTTPError, RuntimeError) as error:
                    if cancel_event is not None and cancel_event.is_set():
                        return "".join(parts), tuple(tool_calls), True
                    raise OllamaUnavailableError(
                        f"The local Ollama stream ended unexpectedly: {error}"
                    ) from error
                finally:
                    with self._stream_lock:
                        if self._active_stream is response:
                            self._active_stream = None
        except httpx.RequestError as error:
            if cancel_event is not None and cancel_event.is_set():
                return "".join(parts), tuple(tool_calls), True
            raise OllamaUnavailableError(
                f"Cannot connect to local Ollama at {self.base_url}: {error}"
            ) from error
        return "".join(parts), tuple(tool_calls), bool(cancel_event is not None and cancel_event.is_set())

    @staticmethod
    def _normalize_tool_calls(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for call in calls:
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = None
            if not isinstance(arguments, Mapping):
                arguments = {}
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": dict(arguments),
                    },
                }
            )
        return normalized

    @classmethod
    def _normalize_tool_messages(
        cls,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
            raise ValueError("At least one chat message is required.")
        normalized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"Chat message {index + 1} must be a mapping.")
            role = str(message.get("role") or "").strip()
            content = message.get("content", "")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Chat message {index + 1} has an unsupported role.")
            if not isinstance(content, str):
                raise ValueError(f"Chat message {index + 1} content must be text.")
            normalized_message: dict[str, Any] = {"role": role, "content": content}
            if role == "tool":
                tool_name = str(message.get("tool_name") or "").strip()
                if not tool_name:
                    raise ValueError(f"Tool message {index + 1} is missing its tool name.")
                normalized_message["tool_name"] = tool_name
            if role == "assistant" and message.get("tool_calls") is not None:
                calls = message.get("tool_calls")
                if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
                    raise ValueError(f"Assistant message {index + 1} has invalid tool calls.")
                normalized_message["tool_calls"] = cls._normalize_tool_calls(calls)
            normalized.append(normalized_message)
        return normalized

    def cancel_generation(self) -> None:
        """Close the active response stream, if a generation is running."""
        with self._stream_lock:
            active_stream = self._active_stream
        if active_stream is not None:
            active_stream.close()

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Run a chat completion with an explicit installed model and history."""
        normalized_messages = self._normalize_messages(messages)
        normalized_temperature = self._validate_temperature(temperature)
        selected_name = str(model or self._selected_model or "").strip()
        if not selected_name:
            raise ModelNotSelectedError("Select an installed local model first.")

        installed_models = self.refresh_models()
        if not any(item.name == selected_name for item in installed_models):
            raise ModelNotInstalledError(
                f"Model '{selected_name}' is not installed in local Ollama."
            )

        payload = self._request_json(
            "POST",
            "/api/chat",
            json={
                "model": selected_name,
                "messages": normalized_messages,
                "stream": False,
                "options": {"temperature": normalized_temperature},
            },
        )
        message = payload.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise LocalLLMError("Ollama returned a chat response without text content.")
        return content

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as error:
            raise OllamaUnavailableError(
                f"Cannot connect to local Ollama at {self.base_url}: {error}"
            ) from error

        if response.is_error:
            detail = self._response_error(response)
            raise LocalLLMError(
                f"Ollama {method} {path} failed ({response.status_code}): {detail}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise LocalLLMError(f"Ollama {method} {path} returned invalid JSON.") from error
        if not isinstance(payload, dict):
            raise LocalLLMError(f"Ollama {method} {path} returned an invalid response.")
        if payload.get("error"):
            raise LocalLLMError(str(payload["error"]))
        return payload

    @staticmethod
    def _response_error(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()[:240] or "No error detail was returned."
        if isinstance(payload, Mapping):
            return str(payload.get("error") or response.text or "No error detail was returned.")[:240]
        return response.text.strip()[:240] or "No error detail was returned."

    @classmethod
    def _normalize_messages(
        cls,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
            raise ValueError("At least one chat message is required.")

        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"Chat message {index + 1} must be a mapping.")
            role = str(message.get("role") or "").strip()
            content = message.get("content")
            if role not in cls._ALLOWED_ROLES:
                raise ValueError(f"Chat message {index + 1} has an unsupported role.")
            if not isinstance(content, str):
                raise ValueError(f"Chat message {index + 1} content must be text.")
            normalized.append({"role": role, "content": content})

        return normalized

    @staticmethod
    def _validate_temperature(value: float) -> float:
        try:
            temperature = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Temperature must be a number between 0 and 2.") from error
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Temperature must be a number between 0 and 2.")
        return temperature

    @classmethod
    def _validate_local_base_url(cls, value: str) -> str:
        candidate = str(value or "").strip().rstrip("/")
        try:
            parsed = urlsplit(candidate)
        except ValueError as error:
            raise ValueError("Ollama base URL must be a valid loopback URL.") from error
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in cls._LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama base URL must point to localhost or a loopback address.")
        return candidate
