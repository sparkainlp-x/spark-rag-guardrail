"""Generation with a mocked Ollama endpoint (no network)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spark_rag_guardrail import rag

pytestmark = [pytest.mark.mocked, pytest.mark.chroma]


def _mock_response(text: str) -> MagicMock:
    # MagicMock (not AsyncMock): httpx.Response.json() is synchronous.
    response = MagicMock()
    response.json.return_value = {"response": text}
    response.raise_for_status.return_value = None
    return response


@pytest.mark.asyncio
async def test_answer_includes_source_marker_and_grounded_prompt(
    seeded_collection, embedder, monkeypatch
):
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test:11434/api/generate")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    mock_post = AsyncMock(
        return_value=_mock_response("AI alignment aims at intended goals [S1].")
    )

    with patch("spark_rag_guardrail.rag.httpx.AsyncClient.post", mock_post):
        result = await rag.answer_question(
            "What is AI alignment research?",
            collection=seeded_collection,
            embedder=embedder,
        )

    assert result["guardrail_triggered"] is False
    assert "[S1]" in result["answer"]
    assert result["sources"][0]["marker"] == "S1"
    assert result["sources"][0]["filename"] == "synthetic_alignment_notes.pdf"

    mock_post.assert_awaited_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://ollama.test:11434/api/generate"
    payload = kwargs["json"]
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    prompt = payload["prompt"]
    assert "Answer the question using ONLY the supplied sources." in prompt
    assert "Every factual statement must include source markers like [S1]." in prompt
    assert "Do not invent citations." in prompt
    assert "[S1: synthetic_alignment_notes.pdf, page 3]\nAI alignment is" in prompt
    assert "Question: What is AI alignment research?" in prompt


def test_build_context_format():
    chunks = [
        {"filename": "a.pdf", "page": 1, "text": "alpha"},
        {"filename": "b.pdf", "page": 2, "text": "beta"},
    ]
    assert rag.build_context(chunks) == "[S1: a.pdf, page 1]\nalpha\n\n[S2: b.pdf, page 2]\nbeta"


def test_defaults_when_env_unset(monkeypatch):
    for var in ("OLLAMA_URL", "OLLAMA_MODEL", "CHROMA_PATH"):
        monkeypatch.delenv(var, raising=False)
    assert rag.ollama_url() == "http://localhost:11434/api/generate"
    assert rag.ollama_model() == "llama3.2:3b"
    assert rag.chroma_path() == "./chroma_db"
