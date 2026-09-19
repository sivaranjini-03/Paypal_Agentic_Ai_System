"""Tool Registry: the system's catalogue of available capabilities.

The registry is generated from the source collection, never hand-maintained.
It answers structural questions cheaply (what domains exist, what tools live in
a domain, which tool produces `capture_id`) and persists a deterministic
snapshot to disk. Semantic retrieval lives in the retriever, not here.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from app.config import get_settings
from app.tools.models import OperationType, ParserIssue, ToolDefinition
from app.tools.postman_parser import parse_collection

REGISTRY_VERSION = "1.0"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class DomainSummary(BaseModel):
    """Routing-facing description of a domain: what it is, not how to call it."""

    domain: str
    tool_count: int
    categories: list[str] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    sample_tools: list[str] = Field(default_factory=list)

    def as_prompt_line(self) -> str:
        operations = ", ".join(self.operations) or "actions"
        examples = "; ".join(self.sample_tools[:2]) or "general capabilities"
        return (
            f"- {self.domain} ({self.tool_count} tools; {operations}): "
            f"{', '.join(self.resources[:5]) or 'general'}; examples: {examples}"
        )


class RegistryDocument(BaseModel):
    """On-disk snapshot of the registry."""

    version: str = REGISTRY_VERSION
    source: str = ""
    collection_name: str = ""
    tool_count: int = 0
    domains: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    issues: list[ParserIssue] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class ToolRegistry:
    """In-memory index over normalized tool definitions."""

    def __init__(self, document: RegistryDocument) -> None:
        self._document = document
        self._by_id: dict[str, ToolDefinition] = {}
        self._by_domain: dict[str, list[ToolDefinition]] = defaultdict(list)
        self._producers: dict[str, list[ToolDefinition]] = defaultdict(list)
        self._consumers: dict[str, list[ToolDefinition]] = defaultdict(list)
        for tool in document.tools:
            self._index(tool)

    # ---------------------------------------------------------------- build --
    @classmethod
    def from_collection(cls, path: str | Path | None = None) -> "ToolRegistry":
        settings = get_settings()
        path = settings.resolve(Path(path) if path else settings.collection_path)
        result = parse_collection(path)
        document = RegistryDocument(
            source=result.source,
            collection_name=result.collection_name,
            tool_count=len(result.tools),
            domains=sorted({t.domain for t in result.tools}),
            variables=result.variables,
            issues=result.issues,
            tools=result.tools,
        )
        return cls(document)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ToolRegistry":
        settings = get_settings()
        path = settings.resolve(Path(path) if path else settings.registry_path)
        document = RegistryDocument.model_validate_json(path.read_text(encoding="utf-8"))
        return cls(document)

    @classmethod
    def load_or_build(cls, registry_path: str | Path | None = None) -> "ToolRegistry":
        settings = get_settings()
        path = settings.resolve(Path(registry_path) if registry_path else settings.registry_path)
        if path.exists():
            return cls.load(path)
        registry = cls.from_collection()
        registry.save(path)
        return registry

    def save(self, path: str | Path | None = None) -> Path:
        settings = get_settings()
        path = settings.resolve(Path(path) if path else settings.registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._document.model_dump(mode="json"), indent=2, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def _index(self, tool: ToolDefinition) -> None:
        self._by_id[tool.id] = tool
        self._by_domain[tool.domain].append(tool)
        for produced in tool.produces:
            self._producers[produced].append(tool)
        for required in tool.requires:
            self._consumers[required].append(tool)

    def register_tool(self, tool: ToolDefinition) -> None:
        """Add a capability that did not come from the collection (RAG, system search)."""
        if tool.id in self._by_id:
            raise ValueError(f"tool id already registered: {tool.id}")
        self._document.tools.append(tool)
        self._document.tool_count = len(self._document.tools)
        if tool.domain not in self._document.domains:
            self._document.domains = sorted([*self._document.domains, tool.domain])
        self._index(tool)

    # ----------------------------------------------------------------- read --
    @property
    def document(self) -> RegistryDocument:
        return self._document

    def __len__(self) -> int:
        return len(self._by_id)

    def __bool__(self) -> bool:
        # An empty registry is still a valid registry; see Telemetry.__bool__.
        return True

    def get_tool(self, tool_id: str) -> ToolDefinition | None:
        return self._by_id.get(tool_id)

    def require_tool(self, tool_id: str) -> ToolDefinition:
        tool = self._by_id.get(tool_id)
        if tool is None:
            raise KeyError(f"unknown tool id: {tool_id}")
        return tool

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._by_id.values())

    def list_domains(self) -> list[str]:
        return sorted(self._by_domain)

    def get_tools_by_domain(self, domain: str) -> list[ToolDefinition]:
        """Domain lookup is case-insensitive so routing output need not match exactly."""
        exact = self._by_domain.get(domain)
        if exact is not None:
            return list(exact)
        lowered = domain.lower()
        for name, tools in self._by_domain.items():
            if name.lower() == lowered:
                return list(tools)
        return []

    def list_categories(self, domain: str | None = None) -> list[str]:
        tools = self.get_tools_by_domain(domain) if domain else self.list_tools()
        return sorted({" > ".join(t.hierarchy) for t in tools if t.hierarchy})

    def tools_producing(self, capability: str) -> list[ToolDefinition]:
        return list(self._producers.get(capability, []))

    def tools_requiring(self, capability: str) -> list[ToolDefinition]:
        return list(self._consumers.get(capability, []))

    def domain_summary(self, domain: str) -> DomainSummary:
        tools = self.get_tools_by_domain(domain)
        return DomainSummary(
            domain=domain,
            tool_count=len(tools),
            categories=sorted({" > ".join(t.hierarchy) for t in tools if len(t.hierarchy) > 1}),
            operations=sorted({t.operation_type.value for t in tools}),
            resources=sorted({t.resource for t in tools if t.resource}),
            sample_tools=[t.name for t in tools[:5]],
        )

    def domain_summaries(self) -> list[DomainSummary]:
        """Compact catalogue used for domain routing and the system search tool."""
        return [self.domain_summary(d) for d in self.list_domains()]

    def stats(self) -> dict[str, int]:
        return {domain: len(tools) for domain, tools in sorted(self._by_domain.items())}

    # --------------------------------------------------------------- search --
    def filter_tools(
        self,
        *,
        domains: Iterable[str] | None = None,
        methods: Iterable[str] | None = None,
        operation_types: Iterable[OperationType | str] | None = None,
        requires: Iterable[str] | None = None,
        produces: Iterable[str] | None = None,
        available_context: Iterable[str] | None = None,
    ) -> list[ToolDefinition]:
        """Structural (non-semantic) narrowing: stage 1 and 2 of retrieval."""
        tools = self.list_tools()
        if domains is not None:
            wanted = {d.lower() for d in domains}
            tools = [t for t in tools if t.domain.lower() in wanted]
        if methods is not None:
            wanted_methods = {m.upper() for m in methods}
            tools = [t for t in tools if t.method in wanted_methods]
        if operation_types is not None:
            wanted_ops = {
                (o.value if isinstance(o, OperationType) else str(o)).lower()
                for o in operation_types
            }
            tools = [t for t in tools if t.operation_type.value in wanted_ops]
        if requires is not None:
            wanted_req = set(requires)
            tools = [t for t in tools if wanted_req & set(t.requires)]
        if produces is not None:
            wanted_prod = set(produces)
            tools = [t for t in tools if wanted_prod & set(t.produces)]
        if available_context is not None:
            # Only tools whose hard inputs the workflow can already satisfy.
            have = set(available_context)
            tools = [t for t in tools if set(t.requires) <= have]
        return tools

    def search_tools(
        self,
        query: str = "",
        *,
        domains: Iterable[str] | None = None,
        methods: Iterable[str] | None = None,
        operation_types: Iterable[OperationType | str] | None = None,
        limit: int = 10,
    ) -> list[tuple[ToolDefinition, float]]:
        """Lexical capability search. Cheap, deterministic, dependency-free."""
        candidates = self.filter_tools(
            domains=domains, methods=methods, operation_types=operation_types
        )
        tokens = set(_tokenize(query))
        if not tokens:
            return [(t, 0.0) for t in candidates[:limit]]

        scored: list[tuple[ToolDefinition, float]] = []
        for tool in candidates:
            name_tokens = set(_tokenize(tool.name))
            path_tokens = set(_tokenize(f"{tool.path_template} {tool.resource}"))
            domain_tokens = set(_tokenize(f"{tool.domain} {' '.join(tool.hierarchy)}"))
            description_tokens = set(_tokenize(tool.description))
            score = (
                3.0 * len(tokens & name_tokens)
                + 2.0 * len(tokens & path_tokens)
                + 1.5 * len(tokens & domain_tokens)
                + 1.0 * len(tokens & description_tokens)
            ) / len(tokens)
            if score > 0:
                scored.append((tool, round(score, 4)))

        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:limit]


_REGISTRY: ToolRegistry | None = None


def get_registry(*, refresh: bool = False) -> ToolRegistry:
    """Process-wide registry singleton."""
    global _REGISTRY
    if _REGISTRY is None or refresh:
        _REGISTRY = ToolRegistry.load_or_build()
    return _REGISTRY


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build or inspect the tool registry")
    parser.add_argument("command", choices=["build", "stats", "search"])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--domain")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.command == "build":
        registry = ToolRegistry.from_collection()
        path = registry.save(args.out)
        print(f"Registry written to {path}")
        print(f"Tools: {len(registry)}  Domains: {len(registry.list_domains())}")
        return

    registry = get_registry()
    if args.command == "stats":
        for domain, count in registry.stats().items():
            print(f"{count:>4}  {domain}")
        print(f"{len(registry):>4}  TOTAL")
        return

    for tool, score in registry.search_tools(
        args.query, domains=[args.domain] if args.domain else None
    ):
        print(f"{score:>6.2f}  {tool.id:<55} {tool.method:<6} {tool.path_template}")


if __name__ == "__main__":
    _main()
