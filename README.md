# spark-rag-guardrail

[![CI](https://github.com/sparkainlp-x/spark-rag-guardrail/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkainlp-x/spark-rag-guardrail/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#evidence-status-unrun)

Source-grounded retrieval-augmented generation (RAG) with a **retrieval-relevance
guardrail**: if nothing in the knowledge base is relevant enough, the LLM is never
called and the system says so instead of guessing.

- Vector store: ChromaDB (`PersistentClient`, collection `notebook_sources`, cosine space)
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (lazy-loaded, optional; injectable)
- Generation: local Ollama (`llama3.2:3b` by default) via `httpx.AsyncClient`

## How the guardrail works

1. The question is embedded and the top-k chunks are retrieved from `notebook_sources`.
2. Each chunk gets `similarity = 1 - cosine_distance` (the collection is created with
   `metadata={"hnsw:space": "cosine"}` so this is meaningful).
3. Chunks with `similarity < SCORE_THRESHOLD` (default `0.40`) are dropped.
4. **If no chunk survives**, `answer_question` returns immediately, without calling the LLM:

   ```python
   {"answer": "[GUARDRAIL] I could not find sufficiently relevant material in the knowledge base to answer reliably.",
    "sources": [], "guardrail_triggered": True}
   ```

5. Otherwise the surviving chunks are formatted as context blocks
   `[S{i}: {filename}, page {page}]\n{text}` and sent with a source-grounded instruction
   (answer using ONLY the supplied sources, cite every factual statement with markers
   like `[S1]`, say so if the sources don't establish an answer, never invent citations).
   The result is `{"answer": ..., "sources": [...], "guardrail_triggered": False}`.

## Usage

```bash
pip install -e ".[embeddings]"   # sentence-transformers is only needed at runtime
```

```python
import asyncio
from spark_rag_guardrail import answer_question

result = asyncio.run(answer_question("What is AI alignment?"))
print(result["answer"], result["sources"], result["guardrail_triggered"])
```

Configuration (environment variables, read at call time):

| Variable       | Default                               |
|----------------|---------------------------------------|
| `OLLAMA_URL`   | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | `llama3.2:3b`                         |
| `CHROMA_PATH`  | `./chroma_db`                         |

Nothing heavy happens at import time: the Chroma client and the embedding model are
created on first use. You can inject your own collection and embedder
(`answer_question(q, collection=..., embedder=...)` or `set_embedder(...)`); an embedder is any
callable mapping `list[str] -> list[list[float]]`.

## Tests

```bash
pip install -e ".[test]"
pytest -m "not live"
```

| Marker   | Meaning                                                              |
|----------|----------------------------------------------------------------------|
| `chroma` | Temporary ChromaDB collection seeded with synthetic fixture docs     |
| `mocked` | Ollama is mocked (no network)                                        |
| `live`   | Talks to a real Ollama at `OLLAMA_URL`; skipped if not reachable     |

Tests use a small deterministic hashed bag-of-words embedder, so no model is downloaded.

## Evidence status: UNRUN

The tests use **SYNTHETIC** fixture documents and a fake embedder. They verify the
plumbing (cosine scoring, threshold filtering, guardrail short-circuit, prompt and citation
format), not answer quality. **No accuracy or hallucination-reduction numbers are claimed
or have been measured.** Any such evaluation is UNRUN.

## Citation

See [CITATION.cff](CITATION.cff). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE). Author: Jean-François Brisson / Spark AI NLP, https://sparkainlpx.xyz
