"""Retrieval against a temporary Chroma collection seeded with SYNTHETIC docs."""

import pytest

from spark_rag_guardrail import rag

pytestmark = pytest.mark.chroma


def test_collection_uses_cosine_space(seeded_collection):
    assert seeded_collection.metadata.get("hnsw:space") == "cosine"
    assert seeded_collection.name == "notebook_sources"


def test_alignment_query_hits_alignment_doc(seeded_collection, embedder):
    chunks = rag.retrieve(
        "What is AI alignment research?",
        collection=seeded_collection,
        embedder=embedder,
    )
    assert chunks, "expected at least one chunk above threshold"
    top = chunks[0]
    assert top["filename"] == "synthetic_alignment_notes.pdf"
    assert top["page"] == 3
    assert "alignment" in top["text"].lower()
    assert top["score"] >= rag.SCORE_THRESHOLD
    # Unrelated synthetic docs must not pass the threshold.
    assert all(c["filename"] == "synthetic_alignment_notes.pdf" for c in chunks)


def test_scores_are_one_minus_cosine_distance(seeded_collection, embedder):
    # threshold=-1 keeps everything so we can inspect scores.
    chunks = rag.retrieve(
        "AI alignment", collection=seeded_collection, embedder=embedder, threshold=-1.0
    )
    assert len(chunks) == 3
    assert all(-1.0 <= c["score"] <= 1.0 for c in chunks)
    assert chunks[0]["score"] >= chunks[-1]["score"]


def test_threshold_filters_unrelated_query(seeded_collection, embedder):
    chunks = rag.retrieve(
        "Best sourdough bread recipe for weekend baking",
        collection=seeded_collection,
        embedder=embedder,
    )
    assert chunks == []


def test_import_does_not_load_models():
    import importlib

    module = importlib.reload(rag)
    assert module._embedder is None
    assert module._collection is None
