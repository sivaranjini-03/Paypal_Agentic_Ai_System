"""Postman Collection v2.1 -> normalized ToolDefinition parser.

Responsibility (and nothing else): turn API definitions into clean, normalized,
searchable tool definitions. This module never calls an API, never talks to an
LLM, never embeds anything and never decides which tool to use.

The parser is fully generic: it knows about the Postman schema, not about any
particular API provider. Adding new requests/folders to the collection yields
new tools with no code change here.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

from app.tools.models import (
    AuthRequirement,
    OperationType,
    Parameter,
    ParseResult,
    ParserIssue,
    ResponseMetadata,
    ToolDefinition,
)

# Credential names must never reach the registry, the retrieval index, a prompt
# or a log line. The match is deliberately narrow: identifiers such as
# `authorization_id` or `payment_token_id` are workflow data, not secrets.
SECRET_NAME_PATTERN = re.compile(
    r"secret|password|passwd|credential|private_key|api[_-]?key"
    r"|access_token|site_token|client_id|auth_assertion",
    re.IGNORECASE,
)
SECRET_VALUE_PATTERN = re.compile(r"^\s*(bearer|basic)\s+\S", re.IGNORECASE)

# Collection variables that are infrastructure/credentials rather than workflow
# data, so they are not treated as workflow dependencies.
NON_WORKFLOW_VARIABLES = {
    "base_url",
    "client_id",
    "client_secret",
    "access_token",
    "access_token_expiry",
    "access_token_for",
    "managed_path_client_id",
    "managed_path_client_secret",
    "managed_path_access_token",
    "managed_path_access_token_expiry",
    "managed_path_access_token_for",
    "paypal_auth_assertion",
    "paypal_partner_attribution_id",
    "paypal_client_metadata_Id",
    "prefer_representation_detailed",
    "prefer_representation_min",
    "prefer_representation_minimal",
    "webhook_site_token",
    "webhook_url",
    "todays_date",
}

# Headers that carry credentials: dropped entirely from the tool surface.
CREDENTIAL_HEADERS = {"authorization", "paypal-auth-assertion"}

VERSION_SEGMENT = re.compile(r"^v\d+(\.\d+)?$")
VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")
SET_VARIABLE_PATTERN = re.compile(
    r"pm\.(?:collectionVariables|environment|globals|variables)\.set\(\s*[\"']([^\"']+)[\"']"
)
TAG_PATTERN = re.compile(r"<[^>]+>")
REQUIRED_PREFIX = re.compile(r"^\s*\(required\)", re.IGNORECASE)
SLUG_STRIP = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return SLUG_STRIP.sub("_", text.strip().lower()).strip("_")


def _clean_text(text: Any, limit: int = 4000) -> str:
    """Strip markup/entities from a Postman description field."""
    if isinstance(text, dict):
        text = text.get("content", "")
    if not isinstance(text, str):
        return ""
    text = TAG_PATTERN.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _is_secret(name: str, value: str = "") -> bool:
    return bool(SECRET_NAME_PATTERN.search(name or "")) or bool(
        value and SECRET_VALUE_PATTERN.match(value)
    )


def _variable_refs(value: Any) -> list[str]:
    """Collection variables referenced anywhere inside a (possibly nested) value."""
    blob = value if isinstance(value, str) else json.dumps(value, default=str)
    return [m for m in VARIABLE_PATTERN.findall(blob) if not m.startswith("$")]


def _safe_example(name: str, value: Any) -> tuple[str | None, str | None]:
    """Return (example, variable_ref) with credentials scrubbed."""
    if not isinstance(value, str) or not value:
        return None, None
    refs = VARIABLE_PATTERN.findall(value)
    ref = refs[0] if refs else None
    if _is_secret(name, value) or (ref and _is_secret(ref)):
        return None, ref
    if ref:
        # Pure variable placeholders carry no example information.
        return (None, ref) if value.strip() == f"{{{{{ref}}}}}" else (value, ref)
    if value.startswith("{{$"):  # postman dynamic variable
        return None, None
    return value[:120], None


def _required_flag(description: str, *, default: bool = False) -> bool:
    return bool(REQUIRED_PREFIX.match(description)) or default


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


# --------------------------------------------------------------------------- #
# URL / parameter parsing
# --------------------------------------------------------------------------- #
def _path_template(url: dict[str, Any]) -> str:
    segments = url.get("path") or []
    if isinstance(segments, str):
        segments = [s for s in segments.split("/") if s]
    rendered = []
    for segment in segments:
        segment = str(segment)
        if segment.startswith(":"):
            rendered.append("{" + segment[1:] + "}")
        else:
            rendered.append(VARIABLE_PATTERN.sub(lambda m: "{" + m.group(1) + "}", segment))
    return "/" + "/".join(rendered)


def _path_parameters(url: dict[str, Any], path_template: str) -> list[Parameter]:
    declared = {v.get("key"): v for v in (url.get("variable") or []) if v.get("key")}
    params: list[Parameter] = []
    for name in re.findall(r"\{([^{}]+)\}", path_template):
        spec = declared.get(name, {})
        description = _clean_text(spec.get("description"), limit=500)
        example, ref = _safe_example(name, spec.get("value", ""))
        params.append(
            Parameter(
                name=name,
                location="path",
                required=True,  # a templated path segment is always needed
                description=description,
                example=example,
                variable_ref=ref,
            )
        )
    return params


def _query_parameters(url: dict[str, Any]) -> list[Parameter]:
    params: list[Parameter] = []
    for spec in url.get("query") or []:
        name = spec.get("key")
        if not name:
            continue
        description = _clean_text(spec.get("description"), limit=500)
        example, ref = _safe_example(name, spec.get("value", ""))
        enabled = not spec.get("disabled", False)
        params.append(
            Parameter(
                name=name,
                location="query",
                required=_required_flag(description) and enabled,
                description=description,
                example=example,
                enabled=enabled,
                variable_ref=ref,
            )
        )
    return params


def _header_parameters(headers: Iterable[dict[str, Any]]) -> list[Parameter]:
    params: list[Parameter] = []
    seen: set[str] = set()
    for spec in headers or []:
        name = spec.get("key")
        if not name or name.lower() in CREDENTIAL_HEADERS:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        description = _clean_text(spec.get("description"), limit=500)
        example, ref = _safe_example(name, spec.get("value", ""))
        params.append(
            Parameter(
                name=name,
                location="header",
                required=_required_flag(description),
                description=description,
                example=example,
                enabled=not spec.get("disabled", False),
                variable_ref=ref,
            )
        )
    return params


# --------------------------------------------------------------------------- #
# body parsing
# --------------------------------------------------------------------------- #
def _loads_templated_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Bodies often contain bare {{variable}} placeholders outside of strings.
        patched = VARIABLE_PATTERN.sub('"__var__"', raw)
        return json.loads(patched)


def _infer_schema(value: Any, *, top_level: bool = False) -> dict[str, Any]:
    kind = _json_type(value)
    if kind == "object":
        properties = {k: _infer_schema(v) for k, v in value.items()}
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if top_level:
            # Fields present in the reference example: a strong hint, not a contract.
            schema["required"] = [k for k, v in value.items() if v not in (None, "", [], {})]
        return schema
    if kind == "array":
        items = _infer_schema(value[0]) if value else {"type": "object"}
        return {"type": "array", "items": items}
    return {"type": kind}


def _parse_body(
    body: dict[str, Any] | None, item_name: str, hierarchy: list[str], issues: list[ParserIssue]
) -> tuple[str, str | None, dict[str, Any] | None]:
    if not body:
        return "none", None, None

    mode = body.get("mode") or "none"

    if mode == "raw":
        raw = body.get("raw") or ""
        language = ((body.get("options") or {}).get("raw") or {}).get("language", "json")
        content_type = "application/json" if language == "json" else f"text/{language}"
        if not raw.strip():
            return "none", None, None
        try:
            parsed = _loads_templated_json(raw)
        except (json.JSONDecodeError, ValueError):
            issues.append(
                ParserIssue(
                    level="warning",
                    item=item_name,
                    hierarchy=hierarchy,
                    reason="raw body is not valid JSON; exposed as opaque payload",
                )
            )
            return "raw", content_type, {"type": "string", "description": "Raw request payload"}
        return "raw", content_type, _infer_schema(parsed, top_level=True)

    if mode in ("urlencoded", "formdata"):
        properties = {}
        required = []
        for spec in body.get(mode) or []:
            name = spec.get("key")
            if not name:
                continue
            description = _clean_text(spec.get("description"), limit=300)
            properties[name] = {"type": "string", "description": description}
            if not spec.get("disabled", False):
                required.append(name)
        content_type = (
            "application/x-www-form-urlencoded" if mode == "urlencoded" else "multipart/form-data"
        )
        return mode, content_type, {"type": "object", "properties": properties, "required": required}

    return "none", None, None


# --------------------------------------------------------------------------- #
# capability metadata
# --------------------------------------------------------------------------- #
def _literal_segments(path_template: str) -> list[str]:
    return [
        s
        for s in path_template.strip("/").split("/")
        if s and not s.startswith("{") and not VERSION_SEGMENT.match(s)
    ]


def _infer_operation(method: str, path_template: str) -> tuple[OperationType, str]:
    segments = path_template.strip("/").split("/")
    literals = _literal_segments(path_template)
    resource = literals[-1] if literals else ""
    ends_with_param = bool(segments) and segments[-1].startswith("{")
    has_param = any(s.startswith("{") for s in segments)
    method = method.upper()

    if method == "GET":
        return (OperationType.READ if ends_with_param else OperationType.LIST), resource
    if method == "DELETE":
        return OperationType.DELETE, resource
    if method in ("PUT", "PATCH"):
        return OperationType.UPDATE, resource
    if method == "POST":
        if has_param and not ends_with_param:
            # e.g. /captures/{id}/refund -> an action on an existing resource
            return OperationType.ACTION, resource
        return OperationType.CREATE, resource
    return OperationType.ACTION, resource


def _produced_variables(item: dict[str, Any]) -> list[str]:
    """Variables the request's test scripts publish for later steps."""
    produced: list[str] = []
    for event in item.get("event") or []:
        if event.get("listen") != "test":  # pre-request scripts only set run-control flags
            continue
        script = (event.get("script") or {}).get("exec") or []
        body = "\n".join(script) if isinstance(script, list) else str(script)
        for name in SET_VARIABLE_PATTERN.findall(body):
            if name not in produced and name not in NON_WORKFLOW_VARIABLES:
                produced.append(name)
    return produced


def _required_context(request: dict[str, Any], path_params: list[Parameter]) -> list[str]:
    """Workflow inputs this tool needs before it can run."""
    required = [p.variable_ref or p.name for p in path_params]
    for ref in _variable_refs(request.get("body") or {}):
        if ref not in required:
            required.append(ref)
    return [r for r in required if r not in NON_WORKFLOW_VARIABLES and not _is_secret(r)]


def _auth_requirement(request: dict[str, Any], inherited: AuthRequirement) -> AuthRequirement:
    auth = request.get("auth")
    if not auth:
        return inherited.model_copy()
    auth_type = auth.get("type", "bearer")
    if auth_type == "noauth":
        return AuthRequirement(required=False, type="none", scheme_source="request")
    return AuthRequirement(required=True, type=auth_type, scheme_source="request")


# --------------------------------------------------------------------------- #
# parsing entry points
# --------------------------------------------------------------------------- #
def _is_folder(item: dict[str, Any]) -> bool:
    return isinstance(item.get("item"), list)


def _parse_request(
    item: dict[str, Any],
    hierarchy: list[str],
    *,
    source: str,
    inherited_auth: AuthRequirement,
    issues: list[ParserIssue],
) -> ToolDefinition | None:
    name = item.get("name") or "unnamed request"
    request = item.get("request")

    if isinstance(request, str):  # shorthand "GET https://..." form
        method, _, raw_url = request.partition(" ")
        request = {"method": method or "GET", "url": {"raw": raw_url}}
    if not isinstance(request, dict):
        issues.append(
            ParserIssue(level="skipped", item=name, hierarchy=hierarchy, reason="missing request")
        )
        return None

    url = request.get("url")
    if isinstance(url, str):
        url = {"raw": url}
    if not isinstance(url, dict) or not (url.get("raw") or url.get("path")):
        issues.append(
            ParserIssue(level="skipped", item=name, hierarchy=hierarchy, reason="missing url")
        )
        return None

    method = (request.get("method") or "GET").upper()
    path_template = _path_template(url)
    url_template = url.get("raw") or path_template

    path_params = _path_parameters(url, path_template)
    query_params = _query_parameters(url)
    headers = _header_parameters(request.get("header") or [])
    body_mode, content_type, body_schema = _parse_body(request.get("body"), name, hierarchy, issues)

    description = _clean_text(request.get("description") or item.get("description"))
    if not description:
        issues.append(
            ParserIssue(
                level="warning", item=name, hierarchy=hierarchy, reason="no description available"
            )
        )

    operation_type, resource = _infer_operation(method, path_template)
    domain = hierarchy[0] if hierarchy else "general"

    fingerprint = hashlib.sha1(
        f"{method}|{path_template}|{'>'.join(hierarchy)}|{name}".encode("utf-8")
    ).hexdigest()[:12]

    responses = [
        ResponseMetadata(
            name=r.get("name", ""),
            status=r.get("status", ""),
            code=r.get("code"),
        )
        for r in (item.get("response") or [])
        if isinstance(r, dict)
    ]

    return ToolDefinition(
        id=f"{_slug(domain)}.{_slug(name)}",
        name=name,
        domain=domain,
        hierarchy=list(hierarchy),
        description=description,
        summary=f"{method} {path_template}",
        method=method,
        url_template=url_template,
        path_template=path_template,
        path_parameters=path_params,
        query_parameters=query_params,
        headers=headers,
        body_mode=body_mode,  # type: ignore[arg-type]
        body_content_type=content_type,
        request_body_schema=body_schema,
        response_metadata=responses,
        auth=_auth_requirement(request, inherited_auth),
        operation_type=operation_type,
        resource=resource,
        requires=_required_context(request, path_params),
        produces=_produced_variables(item),
        source=source,
        source_ref=item.get("id") or item.get("_postman_id"),
        fingerprint=fingerprint,
    )


def _walk_items(
    items: list[dict[str, Any]],
    hierarchy: list[str],
    *,
    source: str,
    inherited_auth: AuthRequirement,
    tools: list[ToolDefinition],
    issues: list[ParserIssue],
) -> None:
    """Depth-first traversal over arbitrarily nested folders."""
    for item in items:
        if not isinstance(item, dict):
            continue
        if _is_folder(item):
            folder_auth = _auth_requirement(item, inherited_auth)
            _walk_items(
                item["item"],
                hierarchy + [item.get("name", "unnamed folder")],
                source=source,
                inherited_auth=folder_auth,
                tools=tools,
                issues=issues,
            )
            continue
        tool = _parse_request(
            item, hierarchy, source=source, inherited_auth=inherited_auth, issues=issues
        )
        if tool is not None:
            tools.append(tool)


def _deduplicate_ids(tools: list[ToolDefinition]) -> None:
    """Keep ids stable and unique; collisions fall back to the fingerprint."""
    seen: set[str] = set()
    for tool in tools:
        if tool.id in seen:
            tool.id = f"{tool.id}.{tool.fingerprint[:6]}"
        seen.add(tool.id)


def _validate(tools: list[ToolDefinition], issues: list[ParserIssue]) -> None:
    for tool in tools:
        missing = [
            field
            for field in ("id", "name", "method", "path_template", "domain")
            if not getattr(tool, field)
        ]
        if missing:
            issues.append(
                ParserIssue(
                    level="warning",
                    item=tool.name,
                    hierarchy=tool.hierarchy,
                    reason=f"incomplete tool definition: missing {', '.join(missing)}",
                )
            )


def parse_collection(path: str | Path) -> ParseResult:
    """Parse a Postman Collection v2.1 file into normalized tool definitions."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        collection = json.load(handle)

    if not isinstance(collection.get("item"), list):
        raise ValueError(f"{path.name} is not a valid Postman collection (no 'item' array)")

    info = collection.get("info") or {}
    collection_auth = collection.get("auth") or {}
    inherited_auth = AuthRequirement(
        required=bool(collection_auth) and collection_auth.get("type") != "noauth",
        type=collection_auth.get("type", "none") if collection_auth else "none",
        scheme_source="collection" if collection_auth else "none",
    )

    tools: list[ToolDefinition] = []
    issues: list[ParserIssue] = []
    _walk_items(
        collection["item"],
        [],
        source=path.name,
        inherited_auth=inherited_auth,
        tools=tools,
        issues=issues,
    )
    _deduplicate_ids(tools)
    _validate(tools, issues)

    return ParseResult(
        source=path.name,
        collection_name=info.get("name", ""),
        tools=tools,
        issues=issues,
        # Names only, credential-shaped names excluded: values may hold secrets.
        variables=[
            v["key"]
            for v in (collection.get("variable") or [])
            if v.get("key") and not _is_secret(v["key"])
        ],
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse a Postman collection into tool definitions")
    parser.add_argument("collection", help="Path to the Postman collection JSON")
    parser.add_argument("--out", help="Optional path to write parsed tools as JSON")
    parser.add_argument("--show", type=int, default=0, help="Print N parsed tools")
    args = parser.parse_args()

    result = parse_collection(args.collection)
    print(result.report())
    domains = sorted({t.domain for t in result.tools})
    print(f"Domains ({len(domains)}): {', '.join(domains)}")
    for issue in result.issues:
        print(f"  [{issue.level}] {' > '.join(issue.hierarchy)} :: {issue.item} -> {issue.reason}")
    for tool in result.tools[: args.show]:
        print(json.dumps(tool.model_dump(mode="json"), indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    _main()
