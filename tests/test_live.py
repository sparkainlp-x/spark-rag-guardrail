"""Optional live test against a real Ollama server. Skipped unless reachable."""

import os
from urllib.parse import urlsplit

import httpx
import pytest

from spark_rag_guardrail import rag

pytestmark = pytest.mark.live


def _ollama_reachable() -> bool:
    url = os.environ.get("OLLAMA_URL", rag.DEFAULT_OLLAMA_URL)
    parts = urlsplit(url)
    try:
        httpx.get(f"{parts.scheme}://{parts.netloc}/api/tags", timeout=2)
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.asyncio
@pytest.mark.chroma
async def test_live_answer(seeded_collection, embedder):
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at OLLAMA_URL")
    result = await rag.answer_question(
        "What is AI alignment research?",
        collection=seeded_collection,
        embedder=embedder,
    )
    assert result["guardrail_triggered"] is False
    assert isinstance(result["answer"], str) and result["answer"]
