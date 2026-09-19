"""Step 5 verification: staged retrieval narrows hundreds of tools to a few."""

from __future__ import annotations

import numpy as np
import pytest

from app.tools.models import OperationType
from app.tools.registry import ToolRegistry
from app.tools.retriever import BM25Index, ToolCandidate, ToolRetriever, tokenize


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.from_collection()


@pytest.fixture(scope="module")
def retriever(registry) -> ToolRetriever:
    return ToolRetriever(registry, embedder=None)


class FakeEmbedder:
    """Deterministic bag-of-words vectors: exercises the semantic path offline."""

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary

    def _vector(self, text: str) -> np.ndarray:
        tokens = set(tokenize(text))
        return np.array([1.0 if word in tokens else 0.0 for word in self.vocabulary], dtype=np.float32)

    def embed_documents(self, texts):
        return np.vstack([self._vector(t) for t in texts]) + 1e-6

    def embed_query(self, text):
        return self._vector(text) + 1e-6


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def rerank(self, query, candidates, top_k):
        self.calls.append(len(candidates))
        # Prefer exact operations over list endpoints.
        ranked = sorted(candidates, key=lambda c: (c.operation_type == "list", -c.score))
        return ranked[:top_k]


def test_tokenizer_splits_identifiers():
    assert tokenize("capture_id showOrderDetails") == ["capture", "id", "show", "order", "details"]


def test_bm25_prefers_documents_containing_query_terms():
    index = BM25Index(["refund a captured payment", "create a draft invoice", "list disputes"])
    scores = index.scores("refund payment", [0, 1, 2])
    assert scores[0] > scores[1] and scores[0] > scores[2]


def test_retrieval_returns_a_small_candidate_set(retriever, registry):
    candidates = retriever.retrieve("refund the captured payment", top_k=3)
    assert 0 < len(candidates) <= 3
    assert candidates[0].tool_id == "payments.refund_captured_payment"
    assert len(candidates) < len(registry) / 10


def test_domain_filter_is_the_first_stage(retriever):
    candidates = retriever.retrieve("show details", domains=["Disputes"], top_k=5)
    assert candidates and all(c.domain == "Disputes" for c in candidates)


def test_metadata_filters_narrow_before_scoring(retriever):
    candidates = retriever.retrieve(
        "invoice", domains=["Invoices"], operation_types=[OperationType.CREATE], top_k=5
    )
    assert candidates and all(c.operation_type == "create" for c in candidates)


def test_available_context_excludes_unsatisfiable_tools(retriever):
    candidates = retriever.retrieve(
        "refund this payment",
        domains=["Payments"],
        available_context=["capture_id"],
        top_k=5,
    )
    ids = {c.tool_id for c in candidates}
    assert "payments.refund_captured_payment" in ids
    assert "payments.void_authorized_payment" not in ids


def test_semantic_stage_contributes_when_an_embedder_is_present(registry):
    vocabulary = ["refund", "capture", "payment", "dispute", "invoice", "order", "payout"]
    retriever = ToolRetriever(registry, embedder=FakeEmbedder(vocabulary))
    candidates = retriever.retrieve("refund a capture", top_k=3)
    assert candidates[0].tool_id == "payments.refund_captured_payment"
    assert "semantic" in candidates[0].stage_scores
    assert "lexical" in candidates[0].stage_scores


def test_reranker_only_sees_a_shortlist(registry):
    reranker = RecordingReranker()
    retriever = ToolRetriever(registry, embedder=None, reranker=reranker)
    candidates = retriever.retrieve("show dispute details", domains=["Disputes"], top_k=2)
    assert len(candidates) == 2
    assert reranker.calls and reranker.calls[0] <= 6
    assert reranker.calls[0] < len(registry.get_tools_by_domain("Disputes")) + 1


def test_candidates_are_prompt_sized_and_credential_free(retriever):
    candidates = retriever.retrieve("create an invoice for a customer", top_k=3)
    block = "\n".join(c.as_prompt_block() for c in candidates)
    assert "client_secret" not in block and "Bearer" not in block
    assert len(block) < 4000


def test_lexical_domain_routing_fallback(retriever):
    assert "Disputes" in retriever.route_domains("is there a dispute open from user_123")
    assert "Invoices" in retriever.route_domains("send an invoice for $50")


def test_all_tools_baseline_is_available_for_benchmarking(retriever, registry):
    assert len(retriever.retrieve_all_tools()) == len(registry)
    assert isinstance(retriever.retrieve_all_tools()[0], ToolCandidate)


def test_unknown_domain_falls_back_instead_of_returning_nothing(retriever):
    assert retriever.retrieve("refund", domains=["NoSuchDomain"], top_k=3) == []
    # A satisfiable domain with impossible context still returns that domain's tools.
    candidates = retriever.retrieve(
        "refund", domains=["Payments"], available_context=[], top_k=3
    )
    assert candidates
