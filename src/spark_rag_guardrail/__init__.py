"""Source-grounded RAG answering with a retrieval-relevance guardrail."""

from .rag import (
    GUARDRAIL_MESSAGE,
    SCORE_THRESHOLD,
    answer_question,
    build_context,
    build_prompt,
    get_collection,
    get_embedder,
    retrieve,
    set_embedder,
)

__all__ = [
    "GUARDRAIL_MESSAGE",
    "SCORE_THRESHOLD",
    "answer_question",
    "build_context",
    "build_prompt",
    "get_collection",
    "get_embedder",
    "retrieve",
    "set_embedder",
]

__version__ = "0.1.1"
