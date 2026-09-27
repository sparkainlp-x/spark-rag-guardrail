"""Guardrail: no chunk above threshold => no LLM call."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spark_rag_guardrail import rag

pytestmark = pytest.mark.mocked

EXPECTED = {
    "answer": (
        "[GUARDRAIL] I could not find sufficiently relevant material in the "
        "knowledge base to answer reliably."
    ),
    "sources": [],
    "guardrail_triggered": True,
}


@pytest.mark.asyncio
async def test_empty_retrieval_triggers_guardrail_without_llm(embedder):
    empty_collection = MagicMock()
    empty_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_post = AsyncMock()
    with patch("spark_rag_guardrail.rag.httpx.AsyncClient.post", mock_post):
        result = await rag.answer_question(
            "anything", collection=empty_collection, embedder=embedder
        )
    assert result == EXPECTED
    mock_post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.chroma
async def test_below_threshold_triggers_guardrail_without_llm(
    seeded_collection, embedder
):
    mock_post = AsyncMock()
    with patch("spark_rag_guardrail.rag.httpx.AsyncClient.post", mock_post):
        result = await rag.answer_question(
            "Best sourdough bread recipe for weekend baking",
            collection=seeded_collection,
            embedder=embedder,
        )
    assert result == EXPECTED
    mock_post.assert_not_called()
