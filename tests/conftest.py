"""Shared fixtures. All documents here are SYNTHETIC and exist only for tests."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import pytest

from spark_rag_guardrail import rag

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "what",
    "which", "with",
}

# SYNTHETIC fixture documents (not real sources).
FIXTURE_DOCS = [
    {
        "id": "doc-alignment",
        "text": (
            "AI alignment is the research field concerned with making AI "
            "systems pursue the goals and values intended by their designers."
        ),
        "metadata": {"filename": "synthetic_alignment_notes.pdf", "page": 3},
    },
    {
        "id": "doc-photosynthesis",
        "text": (
            "Photosynthesis converts light energy into chemical energy stored "
            "in glucose inside plant chloroplasts."
        ),
        "metadata": {"filename": "synthetic_biology_primer.pdf", "page": 12},
    },
    {
        "id": "doc-tides",
        "text": (
            "Ocean tides are caused mainly by the gravitational pull of the "
            "moon acting on Earth's oceans."
        ),
        "metadata": {"filename": "synthetic_earth_science.pdf", "page": 7},
    },
]


class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder (L2-normalised).

    No model download; similarity is driven purely by shared content words.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        return [
            t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS
        ]

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                idx = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def seeded_collection(tmp_path, embedder):
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = rag.get_collection(client=client)
    collection.add(
        ids=[d["id"] for d in FIXTURE_DOCS],
        documents=[d["text"] for d in FIXTURE_DOCS],
        metadatas=[d["metadata"] for d in FIXTURE_DOCS],
        embeddings=embedder([d["text"] for d in FIXTURE_DOCS]),
    )
    return collection


@pytest.fixture(autouse=True)
def _reset_rag_cache():
    rag.reset_cache()
    yield
    rag.reset_cache()
