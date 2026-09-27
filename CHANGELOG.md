# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-27

### Added
- `answer_question()` with a retrieval-relevance guardrail (cosine similarity threshold, default 0.40) that returns an explicit refusal without calling the LLM when nothing relevant is retrieved.
- ChromaDB persistent collection (cosine space), lazy optional sentence-transformers embedder, injectable collection and embedder.
- Ollama generation via `httpx.AsyncClient` with a sources-only, citation-required prompt.
- Test suite (10 offline tests plus 1 opt-in live test) using SYNTHETIC fixtures; CI on Python 3.11 and 3.12.
- `CITATION.cff` and `.zenodo.json` metadata.
