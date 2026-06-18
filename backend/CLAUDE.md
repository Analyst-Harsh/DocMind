# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DocMind is a Retrieval-Augmented Generation (RAG) backend: it ingests a fixed document corpus (PDFs + READMEs) into Qdrant, retrieves relevant chunks for a question, and generates a grounded answer via an LLM — with full Langfuse tracing of cost/latency/usage on every query.

## Commands

All commands run from `backend/` (the project root for tooling; the repo root only contains `backend/`).

- Install deps: `uv sync`
- Run the API: `uvicorn main:app --reload` (serves `/health`, `/query`, `/query/stream`)
- Start Qdrant: `docker compose up -d` (Qdrant REST on `:6333`, gRPC on `:6334`, data in the `qdrant_data` volume)
- Run ingestion (loads corpus → chunks → embeds → upserts into Qdrant): `python -m scripts.ingest`
- Download/refresh the corpus + manifest: `python -m scripts.download_corpus`
- Run tests: `pytest`
- Run a single test: `pytest app/ingestion/test/test_chunker.py::test_name`
- Lint: `ruff check .`

Qdrant server version (`docker-compose.yml`) and the `qdrant-client` Python package version must stay compatible — `query_points` and other newer client calls require a matching-enough server version, or you'll get 404s from the REST API.

## Architecture

Request flow for `/query` (`main.py`): retrieve → generate → trace, all wrapped in nested Langfuse observations (root span → retrieval span → generation span) created via `get_langfuse().start_as_current_observation(...)`. `langfuse` here is the v4 OTEL-based SDK — there is no `trace()/span()/generation()`/`.end()` API; use `start_as_current_observation(as_type="span"|"generation", ...)` as a context manager and `.update(...)` to attach output/usage/cost before the block exits.

Pipeline stages, each its own module under `app/`:

- **`app/ingestion/`** — offline corpus → vector pipeline, driven by `scripts/ingest.py`:
  - `loader.py` reads `corpus/manifest.yaml` and loads each document (markdown/code as text, PDF via PyMuPDF) into a `Document`.
  - `chunker.py` (`FixedSizeChunker`) splits `Document.text` into overlapping token-based `Chunk`s using `tiktoken` (`cl100k_base`), matching OpenAI's billing tokenization.
  - `embedder.py` batches `Chunk`s through OpenAI embeddings (`embedding_model`, default `text-embedding-3-small`, 1536-dim).
  - `indexer.py` creates/ensures the Qdrant collection (`ensure_collection`, vector size must match the embedding model's dimensionality) and upserts `(chunk, vector)` pairs as Qdrant points, with all chunk metadata stored in the point payload (this is what `searcher.py` reads back at query time).
- **`app/retrieval/searcher.py`** — embeds the incoming query and calls `client.query_points` against the configured collection, returning `RetrievedChunk`s reconstructed from point payloads (mirrors the `Chunk` fields written during ingestion).
- **`app/generation/`**:
  - `prompts.py` — `build_qa_prompt` renders the single grounded-QA Jinja2 template (`GROUNDED_QA_TEMPLATE`) that enforces context-only answers with `[doc_title, chunk N]` citations and a fixed refusal string when the answer isn't in context.
  - `generator.py` — `generate_answer` (sync, full response) and `stream_answer` (token-streaming generator) both call the chat completions endpoint with `temperature=0.0` for reproducibility. Cost is computed client-side from hardcoded per-token constants (`COST_PER_INPUT_TOKEN`/`COST_PER_OUTPUT_TOKEN`, currently GPT-4o-mini pricing) rather than relying solely on Langfuse's model price table — update these constants if `llm_model` changes.
- **`app/tracing/langfuse.py`** — `get_langfuse()` returns a cached (`lru_cache`) `Langfuse` client built from `Settings`.
- **`app/config.py`** — single `Settings` (pydantic-settings) object via `get_settings()` (also `lru_cache`d), reading `backend/.env`. This module is also where `.env` loading (`load_dotenv()`) happens, before `Settings` is defined — keep that ordering here rather than in `main.py`/scripts, since downstream modules (e.g. `embedder.py`'s `OpenAI()` client) read `OPENAI_API_KEY` straight from `os.environ` at import time, not from the `Settings` object.

### Data identity

- `Chunk.chunk_id` is `"{doc_id}_{chunk_index}"`. Qdrant point IDs are derived by hashing `chunk_id` (`abs(hash(...)) % 2**63`), since Qdrant requires int/UUID IDs.
- `doc_id`/manifest entries are the join key between `corpus/manifest.yaml`, `Document`, `Chunk`, and the Qdrant payload — if you add a document, it must go through both `download_corpus.py` (or be placed manually) and a `manifest.yaml` entry before `ingest.py` will pick it up.
