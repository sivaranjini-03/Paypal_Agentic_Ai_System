"""RAG pipeline tool.

Retrieves passages from a local knowledge base so the agent can answer
conceptual questions and understand API preconditions. It returns context; it
does not call an LLM and it does not touch the API executor.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT, Settings, get_settings
from app.tools.models import Parameter, ToolDefinition
from app.tools.retriever import BM25Index, default_embedder

RAG_TOOL_ID = "knowledge.search_documentation"
HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
MIN_CHUNK_CHARS = 80


class Passage(BaseModel):
    source: str
    heading: str
    text: str
    score: float = 0.0


class RagResult(BaseModel):
    query: str
    passages: list[Passage] = Field(default_factory=list)
    documents_searched: int = 0

    def as_context(self, max_chars: int = 2500) -> str:
        blocks = [f"[{p.source} :: {p.heading}]\n{p.text}" for p in self.passages]
        return "\n\n".join(blocks)[:max_chars]


def rag_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        id=RAG_TOOL_ID,
        name="Search product documentation",
        domain="Knowledge",
        hierarchy=["Knowledge"],
        description=(
            "Search the product knowledge base for guidance, concepts, payment and "
            "dispute lifecycles, error meanings, preconditions and best practices. "
            "Use this to answer how-does-X-work questions or to learn which operation "
            "must happen before another. Returns documentation passages, not live data."
        ),
        summary="Retrieval-augmented lookup over local documentation",
        method="INTERNAL",
        url_template="internal://knowledge/search",
        path_template="/knowledge/search",
        query_parameters=[
            Parameter(
                name="query",
                location="query",
                required=True,
                description="Natural-language question or topic to look up",
            ),
            Parameter(
                name="top_k",
                location="query",
                required=False,
                type="integer",
                description="How many passages to return (default 4)",
            ),
        ],
        auth={"required": False, "type": "none", "scheme_source": "none"},
        operation_type="read",
        resource="documentation",
        source="internal",
    )


def _chunk(text: str, source: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) chunks at heading boundaries."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return [(source, text.strip())] if text.strip() else []

    chunks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append((heading, f"{heading}\n{body}"))
    return chunks


class RagTool:
    """Hybrid lexical + embedding retrieval over a directory of markdown docs."""

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        *,
        settings: Settings | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.knowledge_dir = knowledge_dir or (PROJECT_ROOT / "data" / "knowledge")
        self._embedder = embedder
        self._embedder_resolved = embedder is not None
        self._sources: list[str] = []
        self._headings: list[str] = []
        self._texts: list[str] = []
        self._bm25: BM25Index | None = None
        self._vectors: np.ndarray | None = None
        self._loaded = False

    @property
    def embedder(self) -> Any | None:
        if not self._embedder_resolved:
            self._embedder = default_embedder(self.settings)
            self._embedder_resolved = True
        return self._embedder

    def load(self) -> None:
        if self._loaded:
            return
        for path in sorted(self.knowledge_dir.glob("**/*.md")):
            text = path.read_text(encoding="utf-8")
            for heading, body in _chunk(text, path.stem):
                self._sources.append(path.stem)
                self._headings.append(heading)
                self._texts.append(body)
        self._bm25 = BM25Index(self._texts)
        self._loaded = True

    def _semantic(self, query: str) -> np.ndarray | None:
        if self.embedder is None or not self._texts:
            return None
        if self._vectors is None:
            vectors = self.embedder.embed_documents(self._texts)
            self._vectors = vectors / np.clip(
                np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None
            )
        query_vector = self.embedder.embed_query(query)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-9)
        return self._vectors @ query_vector

    def search(self, query: str, top_k: int = 4) -> RagResult:
        self.load()
        if not self._texts:
            return RagResult(query=query, documents_searched=0)

        indices = list(range(len(self._texts)))
        lexical = self._bm25.scores(query, indices)  # type: ignore[union-attr]
        highest = float(lexical.max()) if lexical.size else 0.0
        lexical = lexical / highest if highest > 0 else lexical

        semantic = self._semantic(query)
        scores = lexical if semantic is None else 0.5 * lexical + 0.5 * np.clip(semantic, 0, None)

        order = np.argsort(-scores)[: max(1, top_k)]
        return RagResult(
            query=query,
            documents_searched=len({*self._sources}),
            passages=[
                Passage(
                    source=self._sources[i],
                    heading=self._headings[i],
                    text=self._texts[i][:1200],
                    score=round(float(scores[i]), 4),
                )
                for i in order
                if scores[i] > 0
            ],
        )

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}
        top_k = int(arguments.get("top_k") or 4)
        result = self.search(query, top_k=top_k)
        return {
            "query": result.query,
            "documents_searched": result.documents_searched,
            "passages": [p.model_dump() for p in result.passages],
            "context": result.as_context(),
        }
