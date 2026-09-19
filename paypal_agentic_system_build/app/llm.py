"""LLM access.

One place constructs the reasoning model and turns free-form completions into
validated Pydantic objects, so agents never parse raw strings themselves and
token/latency telemetry is captured uniformly.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings

JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when no reasoning model is configured."""


class LLMCall(BaseModel):
    """Telemetry for a single completion."""

    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StructuredResult(BaseModel):
    """A validated model plus the telemetry of the call that produced it."""

    value: Any = None
    raw: str = ""
    call: LLMCall = Field(default_factory=LLMCall)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None


def llm_available(settings: Settings | None = None) -> bool:
    return (settings or get_settings()).llm_available


def get_chat_model(
    settings: Settings | None = None, *, model: str | None = None, temperature: float | None = None
) -> Any:
    settings = settings or get_settings()
    if not settings.llm_available:
        raise LLMUnavailable("GROQ_API_KEY is not configured")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        api_key=settings.groq_api_key,
        max_retries=settings.llm_max_retries,
    )


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a completion that may be fenced or chatty."""
    candidates = [match.group(1) for match in JSON_BLOCK.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        for opening, closing in (("{", "}"), ("[", "]")):
            start = candidate.find(opening)
            end = candidate.rfind(closing)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON object found in model output")


def _usage(response: Any) -> tuple[int, int]:
    metadata = getattr(response, "usage_metadata", None) or {}
    if metadata:
        return int(metadata.get("input_tokens", 0)), int(metadata.get("output_tokens", 0))
    metadata = (getattr(response, "response_metadata", None) or {}).get("token_usage", {})
    return int(metadata.get("prompt_tokens", 0)), int(metadata.get("completion_tokens", 0))


def call_structured(
    llm: Any,
    messages: Sequence[tuple[str, str]],
    schema: type[ModelT],
    *,
    max_attempts: int = 2,
) -> StructuredResult:
    """Invoke the model and validate its JSON answer against `schema`."""
    started = time.perf_counter()
    conversation = list(messages)
    input_tokens = output_tokens = 0
    raw = ""
    error: str | None = None

    for attempt in range(1, max_attempts + 1):
        response = llm.invoke(conversation)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        used_in, used_out = _usage(response)
        input_tokens += used_in
        output_tokens += used_out
        try:
            return StructuredResult(
                value=schema.model_validate(extract_json(raw)),
                raw=raw,
                call=LLMCall(
                    latency_ms=(time.perf_counter() - started) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attempts=attempt,
                    model=getattr(llm, "model_name", getattr(llm, "model", "")),
                ),
            )
        except (ValueError, ValidationError) as exc:
            error = str(exc)
            conversation = [
                *messages,
                ("assistant", raw[:2000]),
                (
                    "user",
                    f"That response was invalid ({error[:300]}). "
                    f"Reply with JSON only, matching this schema: "
                    f"{json.dumps(schema.model_json_schema())}",
                ),
            ]

    return StructuredResult(
        raw=raw,
        error=error,
        call=LLMCall(
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attempts=max_attempts,
            model=getattr(llm, "model_name", getattr(llm, "model", "")),
        ),
    )
