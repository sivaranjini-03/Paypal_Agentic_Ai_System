"""Structured telemetry.

Every tool invocation, LLM call and workflow transition is recorded as a flat
event. This backs three things: debugging, the System Search tool's "what
happened to my last request", and the benchmark's token/latency numbers.

Events carry metadata only. Payloads and credentials are never stored.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal["tool_call", "llm_call", "workflow", "recovery"]


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class TelemetryEvent(BaseModel):
    request_id: str
    kind: EventKind
    timestamp: float = Field(default_factory=time.time)
    name: str = ""
    domain: str = ""
    agent: str = ""
    tool_id: str = ""
    status: str = ""
    latency_ms: float = 0.0
    retry_count: int = 0
    error_type: str | None = None
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        label = self.tool_id or self.name or self.kind
        suffix = f" ({self.error_type}: {self.error_message})" if self.error_type else ""
        return f"{when} [{self.kind}] {label} -> {self.status} in {self.latency_ms:.0f}ms{suffix}"


class Telemetry:
    """In-memory event log with an optional JSONL sink."""

    def __init__(self, *, sink: Path | None = None, max_events: int = 5000) -> None:
        self._events: list[TelemetryEvent] = []
        self._sink = sink
        self._max_events = max_events

    def record(self, event: TelemetryEvent) -> TelemetryEvent:
        self._events.append(event)
        if len(self._events) > self._max_events:
            del self._events[: len(self._events) - self._max_events]
        if self._sink is not None:
            self._sink.parent.mkdir(parents=True, exist_ok=True)
            with self._sink.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.model_dump(mode="json"), default=str) + "\n")
        return event

    def log(self, request_id: str, kind: EventKind, **fields: Any) -> TelemetryEvent:
        return self.record(TelemetryEvent(request_id=request_id, kind=kind, **fields))

    def events(
        self, *, request_id: str | None = None, kind: EventKind | None = None, limit: int = 50
    ) -> list[TelemetryEvent]:
        selected = [
            event
            for event in reversed(self._events)
            if (request_id is None or event.request_id == request_id)
            and (kind is None or event.kind == kind)
        ]
        return list(reversed(selected[:limit]))

    def requests(self, limit: int = 10) -> list[str]:
        seen: list[str] = []
        for event in reversed(self._events):
            if event.request_id not in seen:
                seen.append(event.request_id)
            if len(seen) >= limit:
                break
        return seen

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        # An empty log is still a valid log; never let `x or default` swap it out.
        return True


_TELEMETRY: Telemetry | None = None


def get_telemetry() -> Telemetry:
    global _TELEMETRY
    if _TELEMETRY is None:
        _TELEMETRY = Telemetry()
    return _TELEMETRY
