"""Staged tool retrieval: 500 tools -> a handful the LLM actually sees.

    domain filter -> metadata filter -> lexical+semantic scoring -> optional rerank

Each stage is cheap before the expensive one runs, and every stage is optional,
so the same retriever serves the benchmark's three architectures.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.tools.models import OperationType, ToolDefinition
from app.tools.registry import ToolRegistry, get_registry

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with snake/camel identifiers split apart."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return TOKEN_PATTERN.findall(text.lower())


class ToolCandidate(BaseModel):
    """What the agent receives instead of the full registry."""

    tool_id: str
    name: str
    description: str
    domain: str
    method: str
    path: str
    operation_type: str = ""
    required_parameters: list[str] = Field(default_factory=list)
    optional_parameters: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    score: float = 0.0
    stage_scores: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_tool(cls, tool: ToolDefinition, **scores: float) -> "ToolCandidate":
        return cls(
            tool_id=tool.id,
            name=tool.name,
            description=(tool.description or tool.summary)[:400],
            domain=tool.domain,
            method=tool.method,
            path=tool.path_template,
            operation_type=tool.operation_type.value,
            required_parameters=[p.name for p in tool.required_parameters],
            optional_parameters=[p.name for p in tool.parameters if not p.required][:12],
            requires=tool.requires,
            produces=tool.produces,
            score=round(scores.get("final", 0.0), 4),
            stage_scores={k: round(v, 4) for k, v in scores.items() if k != "final"},
        )

    def as_prompt_block(self) -> str:
        required = ", ".join(self.required_parameters) or "none"
        return (
            f"tool_id: {self.tool_id}\n"
            f"  name: {self.name}\n"
            f"  call: {self.method} {self.path}\n"
            f"  required: {required}\n"
            f"  needs_context: {', '.join(self.requires) or 'none'}\n"
            f"  produces: {', '.join(self.produces) or 'none'}\n"
            f"  about: {self.description[:240]}"
        )


class Embedder(Protocol):
    """Minimal embedding interface so backends stay swappable."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local MiniLM embeddings: no API cost, no data leaving the process."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:  # loaded lazily: importing torch is expensive
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), batch_size=32, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text])[0], dtype=np.float32)


def default_embedder(settings: Settings | None = None) -> Embedder | None:
    """Local embeddings when available; otherwise retrieval stays lexical-only."""
    settings = settings or get_settings()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    return SentenceTransformerEmbedder(settings.embedding_model)


class Reranker(Protocol):
    """Final, most expensive stage: reorder a shortlist."""

    def rerank(
        self, query: str, candidates: list[ToolCandidate], top_k: int
    ) -> list[ToolCandidate]: ...


class _RerankResponse(BaseModel):
    tool_ids: list[str] = Field(default_factory=list)


class LLMReranker:
    """Reorders a shortlist with the reasoning model. Never sees the full registry."""

    SYSTEM = (
        "You rank API tools by how well they satisfy a user request. "
        "Reply with JSON only: {\"tool_ids\": [\"...\"]}, best first, "
        "using only tool_ids from the candidate list."
    )

    def __init__(self, llm: Any | None = None, settings: Settings | None = None) -> None:
        self._llm = llm
        self._settings = settings
        self.last_call: Any = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from app.llm import get_chat_model

            self._llm = get_chat_model(self._settings)
        return self._llm

    def rerank(
        self, query: str, candidates: list[ToolCandidate], top_k: int
    ) -> list[ToolCandidate]:
        from app.llm import call_structured

        listing = "\n\n".join(candidate.as_prompt_block() for candidate in candidates)
        result = call_structured(
            self.llm,
            [
                ("system", self.SYSTEM),
                ("user", f"Request: {query}\n\nCandidates:\n{listing}\n\nReturn the best {top_k}."),
            ],
            _RerankResponse,
        )
        self.last_call = result.call
        if not result.ok:
            return candidates[:top_k]

        by_id = {candidate.tool_id: candidate for candidate in candidates}
        ordered = [by_id[tid] for tid in result.value.tool_ids if tid in by_id]
        remaining = [c for c in candidates if c.tool_id not in {o.tool_id for o in ordered}]
        return (ordered + remaining)[:top_k]


class BM25Index:
    """Okapi BM25 over the tools' semantic descriptions."""

    def __init__(self, documents: Sequence[str]) -> None:
        self._docs = [tokenize(doc) for doc in documents]
        self._lengths = np.array([len(d) or 1 for d in self._docs], dtype=np.float32)
        self._avg_length = float(self._lengths.mean()) if len(self._docs) else 1.0
        self._term_frequencies = [Counter(doc) for doc in self._docs]
        document_frequency: Counter[str] = Counter()
        for doc in self._docs:
            document_frequency.update(set(doc))
        total = len(self._docs) or 1
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: str, indices: Sequence[int]) -> np.ndarray:
        terms = tokenize(query)
        scores = np.zeros(len(indices), dtype=np.float32)
        for position, index in enumerate(indices):
            frequencies = self._term_frequencies[index]
            length = self._lengths[index]
            total = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / self._avg_length
                )
                total += self._idf.get(term, 0.0) * frequency * (BM25_K1 + 1) / denominator
            scores[position] = total
        return scores


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    highest = float(values.max())
    return values / highest if highest > 0 else np.zeros_like(values)


def diversify(candidates: Sequence[ToolCandidate], limit: int) -> list[ToolCandidate]:
    """Reserve slots for unseen operation types.

    Descriptions inside one domain are near-identical, so pure similarity tends
    to return eight variations of the same action and drop the `list` endpoint
    that actually answers a question.
    """
    if limit <= 2 or len(candidates) <= limit:
        return list(candidates[:limit])

    reserved = max(1, limit // 3)
    chosen: list[ToolCandidate] = list(candidates[: limit - reserved])
    seen = {candidate.operation_type for candidate in chosen}

    for candidate in candidates[limit - reserved :]:
        if len(chosen) >= limit:
            break
        if candidate.operation_type not in seen:
            chosen.append(candidate)
            seen.add(candidate.operation_type)

    for candidate in candidates:
        if len(chosen) >= limit:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen


class ToolRetriever:
    """Hybrid (lexical + optional embedding) retrieval over the registry."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry()
        self.embedder = embedder if embedder is not None else default_embedder(self.settings)
        self.reranker = reranker

        self._tools: list[ToolDefinition] = self.registry.list_tools()
        self._position: dict[str, int] = {t.id: i for i, t in enumerate(self._tools)}
        self._documents = [t.semantic_text() for t in self._tools]
        self._bm25 = BM25Index(self._documents)
        self._vectors: np.ndarray | None = None
        self._cache_key = hashlib.sha1(
            "|".join([getattr(self.embedder, "model_name", "none"), *self._documents]).encode(
                "utf-8"
            )
        ).hexdigest()

    # ------------------------------------------------------------ embedding --
    def _cache_path(self) -> Path:
        return self.settings.resolve(self.settings.embedding_cache_path)

    def _load_cached_vectors(self) -> np.ndarray | None:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            cached = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if str(cached.get("key", "")) != self._cache_key:
            return None  # registry or model changed: recompute
        vectors = cached["vectors"]
        return vectors if vectors.shape[0] == len(self._documents) else None

    def _store_vectors(self, vectors: np.ndarray) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=vectors, key=np.array(self._cache_key))

    def _ensure_vectors(self) -> np.ndarray | None:
        if self.embedder is None:
            return None
        if self._vectors is None:
            cached = self._load_cached_vectors()
            if cached is not None:
                self._vectors = cached
            else:
                vectors = self.embedder.embed_documents(self._documents)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                self._vectors = vectors / np.clip(norms, 1e-9, None)
                self._store_vectors(self._vectors)
        return self._vectors

    def _semantic_scores(self, query: str, indices: Sequence[int]) -> np.ndarray | None:
        vectors = self._ensure_vectors()
        if vectors is None:
            return None
        query_vector = self.embedder.embed_query(query)  # type: ignore[union-attr]
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-9)
        return vectors[list(indices)] @ query_vector

    # ------------------------------------------------------------ retrieval --
    def retrieve(
        self,
        query: str,
        *,
        domains: Iterable[str] | None = None,
        methods: Iterable[str] | None = None,
        operation_types: Iterable[OperationType | str] | None = None,
        available_context: Iterable[str] | None = None,
        top_k: int = 5,
        rerank_top_n: int | None = None,
        diversity: bool = True,
    ) -> list[ToolCandidate]:
        # Stage 1 + 2: structural narrowing, no model calls.
        pool = self.registry.filter_tools(
            domains=domains,
            methods=methods,
            operation_types=operation_types,
            available_context=available_context,
        )
        if not pool:
            pool = self.registry.filter_tools(domains=domains)
        if not pool:
            return []

        indices = [self._position[t.id] for t in pool if t.id in self._position]
        if not indices:
            return []

        # Stage 3: scoring over the surviving pool only.
        lexical = _normalize(self._bm25.scores(query, indices))
        semantic = self._semantic_scores(query, indices)
        if semantic is None:
            final = lexical
            stage_names = {"lexical": lexical}
        else:
            semantic = _normalize(np.clip(semantic, 0, None))
            final = 0.5 * lexical + 0.5 * semantic
            stage_names = {"lexical": lexical, "semantic": semantic}

        shortlist_size = rerank_top_n or max(top_k * 3, top_k)
        order = np.argsort(-final)[:shortlist_size]

        candidates = [
            ToolCandidate.from_tool(
                self._tools[indices[position]],
                final=float(final[position]),
                **{name: float(values[position]) for name, values in stage_names.items()},
            )
            for position in order
        ]

        # Stage 4: optional LLM rerank over a small shortlist.
        if self.reranker is not None and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, top_k)
            return candidates[:top_k]
        return diversify(candidates, top_k) if diversity else candidates[:top_k]

    def retrieve_all_tools(self) -> list[ToolCandidate]:
        """Baseline A: everything, unranked. Exists to be measured against."""
        return [ToolCandidate.from_tool(tool) for tool in self._tools]

    def route_domains(self, query: str, *, top_n: int = 2) -> list[str]:
        """Cheap lexical domain routing, used as a fallback when no LLM is available."""
        scores: dict[str, float] = {}
        query_tokens = set(tokenize(query))
        for summary in self.registry.domain_summaries():
            text = " ".join(
                [summary.domain, *summary.categories, *summary.resources, *summary.sample_tools]
            )
            tokens = set(tokenize(text))
            overlap = len(query_tokens & tokens)
            if overlap:
                scores[summary.domain] = overlap / max(len(query_tokens), 1)
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return [domain for domain, _ in ranked[:top_n]]


def format_candidates(candidates: Sequence[ToolCandidate]) -> str:
    """Prompt-ready rendering of the final candidate set."""
    return "\n".join(candidate.as_prompt_block() for candidate in candidates)


def candidate_payload(candidates: Sequence[ToolCandidate]) -> list[dict[str, Any]]:
    return [candidate.model_dump() for candidate in candidates]
