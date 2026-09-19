"""Generic API executor.

`ToolDefinition` + runtime parameters -> an HTTP request. N APIs therefore need
zero per-API Python functions. This layer performs no retries and no planning:
it executes once and reports a structured result. Recovery decisions belong to
the recovery handler.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from app.config import Settings, get_settings
from app.observability import Telemetry, get_telemetry
from app.tools.credentials import CredentialError, CredentialManager
from app.tools.models import ExecutionResult, ToolDefinition
from app.tools.schema import BODY_ARGUMENT

VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")
DEFAULT_TIMEOUT = 30.0
MAX_TEXT_CHARS = 4000
INTERNAL_METHOD = "INTERNAL"


class ToolExecutor:
    """Turns a normalized tool definition into a real HTTP call."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        credentials: CredentialManager | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        internal_handlers: Mapping[str, Callable[[dict[str, Any]], Awaitable[Any]]] | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.credentials = credentials or CredentialManager(
            settings=self.settings, client=client
        )
        self._client = client
        self._owns_client = client is None
        self.timeout = timeout
        self.internal_handlers = dict(internal_handlers or {})
        self.telemetry = telemetry or get_telemetry()
        self.request_id = ""

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------- building --
    def _resolve_variables(self, template: str, values: Mapping[str, Any]) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name == "base_url":
                return self.settings.paypal_base_url.rstrip("/")
            return str(values.get(name, match.group(0)))

        return VARIABLE_PATTERN.sub(replace, template)

    def build_url(self, tool: ToolDefinition, arguments: Mapping[str, Any]) -> str:
        base = self.settings.paypal_base_url.rstrip("/")
        path = tool.path_template
        missing: list[str] = []
        for parameter in tool.path_parameters:
            value = arguments.get(parameter.name)
            if value in (None, ""):
                missing.append(parameter.name)
                continue
            path = path.replace("{" + parameter.name + "}", str(value))
        if missing:
            raise ValueError(f"missing required path parameter(s): {', '.join(missing)}")
        path = self._resolve_variables(path, arguments)
        return f"{base}{path}"

    def build_query(self, tool: ToolDefinition, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query: dict[str, Any] = {}
        declared = {p.name for p in tool.query_parameters}
        for name in declared:
            if name in arguments and arguments[name] not in (None, ""):
                query[name] = arguments[name]
        return query

    async def build_headers(
        self, tool: ToolDefinition, arguments: Mapping[str, Any], *, force_refresh: bool = False
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if tool.body_content_type:
            headers["Content-Type"] = tool.body_content_type
        for parameter in tool.headers:
            value = arguments.get(parameter.name)
            if value not in (None, ""):
                headers[parameter.name] = str(value)
        if tool.auth.required:
            # Injected here, never supplied by the model.
            headers["Authorization"] = await self.credentials.authorization_header(
                tool.auth.type, force_refresh=force_refresh
            )
        return headers

    def build_body(self, tool: ToolDefinition, arguments: Mapping[str, Any]) -> Any:
        if tool.body_mode == "none":
            return None
        body = arguments.get(BODY_ARGUMENT)
        if body is None:
            declared = set((tool.request_body_schema or {}).get("properties", {}))
            body = {k: v for k, v in arguments.items() if k in declared} or None
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return body
        return body

    # ------------------------------------------------------------ executing --
    async def execute(
        self,
        tool: ToolDefinition,
        arguments: Mapping[str, Any] | None = None,
        *,
        attempt: int = 1,
        force_token_refresh: bool = False,
    ) -> ExecutionResult:
        arguments = dict(arguments or {})
        started = time.perf_counter()

        if tool.method == INTERNAL_METHOD:
            result = await self._execute_internal(tool, arguments, started, attempt)
            self._record(tool, result)
            return result

        result = await self._execute_http(
            tool, arguments, started, attempt, force_token_refresh
        )
        self._record(tool, result)
        return result

    def _record(self, tool: ToolDefinition, result: ExecutionResult) -> None:
        self.telemetry.log(
            self.request_id or "adhoc",
            "tool_call",
            tool_id=tool.id,
            domain=tool.domain,
            name=tool.name,
            status="success" if result.success else "error",
            latency_ms=result.latency_ms,
            retry_count=max(0, result.attempt - 1),
            error_type=result.error_type,
            error_message=(result.error or "")[:300] or None,
            detail={"status_code": result.status_code, "call": result.request_summary},
        )

    async def _execute_internal(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        started: float,
        attempt: int,
    ) -> ExecutionResult:
        handler = self.internal_handlers.get(tool.id)
        if handler is None:
            return self._failure(
                tool, "internal", f"no handler registered for {tool.id}", started, attempt,
                f"INTERNAL {tool.path_template}",
            )
        try:
            data = await handler(arguments)
        except Exception as exc:  # a capability bug must not crash the workflow
            return self._failure(
                tool, "internal", str(exc), started, attempt, f"INTERNAL {tool.path_template}"
            )
        return ExecutionResult(
            tool_id=tool.id,
            success=not (isinstance(data, dict) and data.get("error")),
            status_code=200,
            data=data,
            error=data.get("error") if isinstance(data, dict) else None,
            error_type="validation" if isinstance(data, dict) and data.get("error") else None,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempt=attempt,
            request_summary=f"INTERNAL {tool.path_template}",
        )

    async def _execute_http(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        started: float,
        attempt: int,
        force_token_refresh: bool,
    ) -> ExecutionResult:
        try:
            url = self.build_url(tool, arguments)
            params = self.build_query(tool, arguments)
            headers = await self.build_headers(
                tool, arguments, force_refresh=force_token_refresh
            )
            body = self.build_body(tool, arguments)
        except ValueError as exc:
            return ExecutionResult(
                tool_id=tool.id,
                success=False,
                error=str(exc),
                error_type="validation",
                attempt=attempt,
                latency_ms=(time.perf_counter() - started) * 1000,
                request_summary=f"{tool.method} {tool.path_template}",
            )
        except CredentialError as exc:
            return ExecutionResult(
                tool_id=tool.id,
                success=False,
                error=str(exc),
                error_type="authentication",
                attempt=attempt,
                latency_ms=(time.perf_counter() - started) * 1000,
                request_summary=f"{tool.method} {tool.path_template}",
            )

        request_kwargs: dict[str, Any] = {"params": params, "headers": headers}
        if body is not None:
            if tool.body_mode == "urlencoded":
                request_kwargs["data"] = body
            elif isinstance(body, (dict, list)):
                request_kwargs["json"] = body
            else:
                request_kwargs["content"] = body

        summary = f"{tool.method} {tool.path_template}"
        try:
            response = await self.client.request(tool.method, url, **request_kwargs)
        except httpx.TimeoutException as exc:
            return self._failure(tool, "timeout", str(exc), started, attempt, summary)
        except httpx.HTTPError as exc:
            return self._failure(tool, "network", str(exc), started, attempt, summary)

        latency_ms = (time.perf_counter() - started) * 1000
        payload, text = self._parse_response(response)

        if response.is_success:
            return ExecutionResult(
                tool_id=tool.id,
                success=True,
                status_code=response.status_code,
                data=payload,
                text=None if payload is not None else text,
                latency_ms=latency_ms,
                attempt=attempt,
                request_summary=summary,
                response_headers=self._safe_headers(response.headers),
                produced_context=extract_context(tool, payload),
            )

        return ExecutionResult(
            tool_id=tool.id,
            success=False,
            status_code=response.status_code,
            data=payload,
            text=None if payload is not None else text,
            error=self._error_message(payload, text, response.status_code),
            error_type="http",
            latency_ms=latency_ms,
            attempt=attempt,
            request_summary=summary,
            response_headers=self._safe_headers(response.headers),
        )

    def _failure(
        self,
        tool: ToolDefinition,
        error_type: str,
        message: str,
        started: float,
        attempt: int,
        summary: str,
    ) -> ExecutionResult:
        return ExecutionResult(
            tool_id=tool.id,
            success=False,
            error=message,
            error_type=error_type,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempt=attempt,
            request_summary=summary,
        )

    @staticmethod
    def _parse_response(response: httpx.Response) -> tuple[Any, str | None]:
        if not response.content:
            return None, None
        try:
            return response.json(), None
        except (json.JSONDecodeError, ValueError):
            return None, response.text[:MAX_TEXT_CHARS]

    @staticmethod
    def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
        keep = {"retry-after", "content-type", "paypal-debug-id", "x-ratelimit-reset"}
        return {k: v for k, v in headers.items() if k.lower() in keep}

    @staticmethod
    def _error_message(payload: Any, text: str | None, status_code: int) -> str:
        if isinstance(payload, dict):
            for key in ("message", "error_description", "error", "name"):
                if isinstance(payload.get(key), str):
                    detail = payload[key]
                    issues = payload.get("details") or []
                    if isinstance(issues, list) and issues:
                        detail = f"{detail} :: {json.dumps(issues[:3], default=str)[:600]}"
                    return detail
        return (text or f"HTTP {status_code}")[:MAX_TEXT_CHARS]


def _walk(node: Any, depth: int = 0):
    """Breadth-ish walk over nested JSON, bounded so pathological payloads can't stall."""
    if depth > 8:
        return
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value, depth + 1)
    elif isinstance(node, list):
        for entry in node[:20]:
            yield from _walk(entry, depth + 1)


def _scalar(value: Any) -> Any | None:
    return value if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None


def extract_named(payload: Any, keys: Sequence[str]) -> dict[str, Any]:
    """Pull named workflow values out of an arbitrary JSON response.

    Two generic REST conventions are honoured, in order:
      1. a field literally named `capture_id`
      2. `capture.id` / `captures[0].id` for a key of the form `<noun>_id`
    """
    found: dict[str, Any] = {}
    if payload is None:
        return found

    wanted = [k for k in keys if k]
    for node in _walk(payload):
        for key in wanted:
            if key in found:
                continue
            value = _scalar(node.get(key))
            if value is not None:
                found[key] = value

    for key in wanted:
        if key in found or not key.endswith("_id"):
            continue
        noun = key[:-3]
        containers = {noun, f"{noun}s", f"{noun}es"}
        for node in _walk(payload):
            for name in containers & node.keys():
                target = node[name]
                if isinstance(target, list):
                    target = target[0] if target else None
                if isinstance(target, dict):
                    value = _scalar(target.get("id"))
                    if value is not None:
                        found[key] = value
                        break
            if key in found:
                break
    return found


def _singular(name: str) -> str:
    if name.endswith("ies"):
        return f"{name[:-3]}y"
    if name.endswith("ses") or name.endswith("xes"):
        return name[:-2]
    return name[:-1] if name.endswith("s") else name


def harvest_identifiers(payload: Any, *, limit: int = 25) -> dict[str, Any]:
    """Collect identifiers a later step might need, without per-API code.

    Recognises `<thing>_id` fields directly, and the REST shape where a nested
    `captures: [{id: ...}]` implies `capture_id`. Declared `produces` metadata
    is authoritative; this is the safety net for read endpoints that declare none.
    """
    found: dict[str, Any] = {}
    for node in _walk(payload):
        for key, value in node.items():
            if len(found) >= limit:
                return found
            scalar = _scalar(value)
            if scalar is not None and key.endswith("_id") and key not in found:
                found[key] = scalar
                continue
            target = value[0] if isinstance(value, list) and value else value
            if isinstance(target, dict):
                nested = _scalar(target.get("id"))
                name = f"{_singular(key)}_id"
                if nested is not None and name not in found:
                    found[name] = nested
    return found


def extract_context(tool: ToolDefinition, payload: Any) -> dict[str, Any]:
    """Map a response onto the workflow variables this tool declares it produces.

    The declared `produces` names come from the source collection, so no
    per-API extraction code is needed.
    """
    if not tool.produces or payload is None:
        return {}

    found = extract_named(payload, tool.produces)

    # A resource-creating call returns its own identifier as `id`.
    primary = tool.produces[0]
    if primary not in found and isinstance(payload, dict) and isinstance(payload.get("id"), str):
        found[primary] = payload["id"]
    return found
