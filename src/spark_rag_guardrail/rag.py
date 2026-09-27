"""RAG answer function with a relevance guardrail.

Retrieval uses ChromaDB (collection ``notebook_sources``, cosine space) and
sentence-transformers ``all-MiniLM-L6-v2`` embeddings by default.  Generation
uses a local Ollama server.  If no retrieved chunk reaches ``SCORE_THRESHOLD``
(similarity = 1 - cosine distance), the LLM is never called and a guardrail
response is returned instead.

Heavy dependencies (chromadb client, sentence-transformers model) are loaded
lazily so that importing this module never downloads a model.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import httpx

SCORE_THRESHOLD = 0.40
COLLECTION_NAME = "notebook_sources"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_TOP_K = 5

GUARDRAIL_MESSAGE = (
    "[GUARDRAIL] I could not find sufficiently relevant material in the "
    "knowledge base to answer reliably."
)

SYSTEM_INSTRUCTIONS = (
    "Answer the question using ONLY the supplied sources. "
    "Every factual statement must include source markers like [S1]. "
    "If the sources do not establish an answer, state that clearly. "
    "Do not invent citations."
)

# An embedder maps a list of texts to a list of vectors.
Embedder = Callable[[Sequence[str]], list[list[float]]]

_embedder: Optional[Embedder] = None
_collection: Any = None


# --------------------------------------------------------------------------
# Configuration (read at call time, so env changes are honoured)
# --------------------------------------------------------------------------
def ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def chroma_path() -> str:
    return os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH)


# --------------------------------------------------------------------------
# Embedder (lazy, injectable)
# --------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers; the model loads on first use."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "sentence-transformers is not installed. Install the "
                    "'embeddings' extra or inject a custom embedder with "
                    "set_embedder()."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def set_embedder(embedder: Optional[Embedder]) -> None:
    """Inject a custom embedder (e.g. a deterministic fake in tests)."""
    global _embedder
    _embedder = embedder


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


# --------------------------------------------------------------------------
# Collection (lazy, injectable)
# --------------------------------------------------------------------------
def get_collection(client: Any = None, name: str = COLLECTION_NAME) -> Any:
    """Return the ``notebook_sources`` collection, configured for cosine space.

    If ``client`` is given it is used as-is (e.g. an EphemeralClient in tests);
    otherwise a module-level PersistentClient at ``CHROMA_PATH`` is created
    on first use and cached.
    """
    global _collection
    if client is not None:
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
    if _collection is None:
        import chromadb

        persistent = chromadb.PersistentClient(path=chroma_path())
        _collection = persistent.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def reset_cache() -> None:
    """Forget the cached embedder and collection."""
    global _embedder, _collection
    _embedder = None
    _collection = None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(
    question: str,
    collection: Any = None,
    embedder: Optional[Embedder] = None,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return chunks whose similarity (1 - cosine distance) >= ``threshold``."""
    collection = collection if collection is not None else get_collection()
    embedder = embedder if embedder is not None else get_embedder()

    query_embedding = embedder([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns one list per query embedding: index [0] for our single query.
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    chunks: list[dict[str, Any]] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        similarity = 1.0 - distance
        if similarity < threshold:
            continue
        meta = meta or {}
        chunks.append(
            {
                "text": text,
                "filename": meta.get("filename", "unknown"),
                "page": meta.get("page", "?"),
                "score": round(float(similarity), 4),
            }
        )
    return chunks


# --------------------------------------------------------------------------
# Prompting and generation
# --------------------------------------------------------------------------
def build_context(chunks: Sequence[dict[str, Any]]) -> str:
    blocks = [
        f"[S{i}: {c['filename']}, page {c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


async def generate(prompt: str) -> str:
    payload = {"model": ollama_model(), "prompt": prompt, "stream": False}
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(ollama_url(), json=payload)
        response.raise_for_status()
        data = response.json()
    return str(data.get("response", "")).strip()


def _guardrail_response() -> dict[str, Any]:
    return {"answer": GUARDRAIL_MESSAGE, "sources": [], "guardrail_triggered": True}


async def answer_question(
    question: str,
    collection: Any = None,
    embedder: Optional[Embedder] = None,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Retrieve, apply the guardrail, and (only if it passes) ask the LLM."""
    chunks = retrieve(
        question,
        collection=collection,
        embedder=embedder,
        top_k=top_k,
        threshold=threshold,
    )
    if not chunks:
        return _guardrail_response()

    prompt = build_prompt(question, build_context(chunks))
    answer = await generate(prompt)
    sources = [
        {
            "marker": f"S{i}",
            "filename": c["filename"],
            "page": c["page"],
            "score": c["score"],
        }
        for i, c in enumerate(chunks, start=1)
    ]
    return {"answer": answer, "sources": sources, "guardrail_triggered": False}
