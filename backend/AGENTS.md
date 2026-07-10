# AGENTS.md — backend

FastAPI RAG service: ingests a fixed corpus into Qdrant, retrieves via dense/hybrid/reranked/agentic/graph search, generates grounded cited answers, traces every query with Langfuse. See the [root AGENTS.md](../AGENTS.md) for how this fits into the rest of the repo, and [`README.md`](./README.md) for the human-facing directory tour and API reference.

This file is the authoritative architecture/commands/conventions reference for `backend/` — read it before non-trivial changes. `CLAUDE.md` in this directory just points here.

## Setup

```bash
uv sync
docker compose up -d              # Qdrant :6333/:6334, Redis :6379, Neo4j :7474/:7687
cp .env.example .env              # required: OPENAI_API_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
python -m scripts.download_corpus
python -m scripts.ingest --strategy all   # add --embedding-model BAAI/bge-large-en-v1.5 for local embeddings
uvicorn main:app --reload         # serves /health, /query, /query/stream, /agent/query, /documents
```

Qdrant server version (`docker-compose.yml`) and the `qdrant-client` pip package version must stay compatible, or `query_points` and other newer client calls 404 against the REST API.

## Testing

```bash
pytest                                                                     # full suite
pytest app/ingestion/chunker/tests/test_recursive_chunker.py::test_name   # single test
python -m scripts.eval_chunking --strategies recursive --embedding-model BAAI/bge-large-en-v1.5   # retrieval eval
ruff check .                                                               # lint
uv run mypy app/                                                          # type check
```

Tests are colocated per module (`app/{agent,caching,ingestion,retrieval,streaming}/tests/`, 17 files). **`app/documents/` and `app/graph/` have no tests** — don't assume coverage there.

**`pytest` needs no live services and no API keys.** Every test mocks Qdrant/Redis/Neo4j/OpenAI/Langfuse (`unittest.mock.MagicMock`/`monkeypatch`, `app/caching/tests/fakes.py`'s `FakeRedis`, `app/agent/tests/conftest.py`'s tracing no-ops) — you do not need `docker compose up` or a populated `.env` to run the suite. If a test *does* seem to need a live service, that's a sign something's wrong, not a sign you need to start Docker.

Every retrieval/generation change should be measured against `app/eval/` (`scripts/eval_*.py`, `scripts/run_*_eval.py`) before being treated as an improvement — this is the project's core practice; see `EXPERIMENTS.md` for the full log of hypothesis → measurement → decision this produced.

**Both `ruff check .` and `uv run mypy app/` are clean** (0 errors, `[[tool.mypy.overrides]]` in `pyproject.toml` silences `fitz`'s missing stubs, since PyMuPDF ships none). Treat any output from either as a real issue introduced by your change, not pre-existing debt to route around — if you see one, fix it or explain why not, don't assume it was already there.

## Architecture

### Request flow — `POST /query` (`main.py`)

embed query → check semantic cache → retrieve (dense / hybrid / hybrid+reranked, per `request.hybrid` and `settings.use_reranker`) → generate → return `{answer, sources, cost_usd, latency_ms, trace_id, cache_hit}`. Wrapped in nested Langfuse spans: root → cache-check → retrieval → generation. `POST /query/stream` runs the same retrieval/cache logic in `main.py` then hands off to `app/streaming/pipeline.py` for SSE generation — see below.

### Ingestion (`app/ingestion/`, driven by `scripts/ingest.py`)

- `loader.py` reads `corpus/manifest.yaml` and loads each document (markdown/code as text, PDF via PyMuPDF) into a `Document`.
- `chunker/` is a package: `base_chunker.py` defines the `Chunk` dataclass, `ChunkStrategy` enum, and abstract `BaseChunker` (token counting via `tiktoken`/`cl100k_base`, matching OpenAI's billing tokenization). `fixed_size_chunker.py`, `recursive_chunker.py` (production default — see Experiment 1), and `structure_aware_chunker.py` are the three prose strategies; `table_chunker.py`/`figure_chunker.py` handle multimodal content. `chunk_registry.py`'s `get_chunker(strategy, chunk_size, chunk_overlap)` is the factory `scripts/ingest.py` and `scripts/eval_chunking.py` use.
- `embedder.py` dispatches `embed_chunks`/`embed_query` by the `model` argument: OpenAI (default `text-embedding-3-small`, 1536-dim, batched) or a local `sentence-transformers` model (`LOCAL_MODELS`, currently `BAAI/bge-large-en-v1.5`, 1024-dim, `@lru_cache`-loaded). BGE-family models require queries (not passages) prefixed with `BGE_QUERY_INSTRUCTION` and L2-normalized embeddings (`normalize_embeddings=True`) to match how they were contrastively trained — get this wrong and similarity scores degrade silently, no error. `sparse_embedder.py` adds BM25 sparse vectors (`fastembed`) for hybrid search.
- `table_extractor.py` (Docling TableFormer, page-by-page to avoid `std::bad_alloc`) and `figure_extractor.py` (Docling picture detection + GPT-4o Vision captioning, `MIN_FIGURE_PX=100` filter) feed the multimodal collection.
- `indexer.py` creates/ensures the Qdrant collection (`ensure_collection`/`ensure_hybrid_collection`, vector size from `get_embedding_dim`) and upserts `(chunk, vector)` pairs as points, with all chunk metadata in the payload (what `searcher.py` reads back). `collection_name_for(strategy, embedding_model, hybrid=...)` sanitizes `/` to `-` in HF model ids before building the collection name. `HYBRID_STRATEGY = "recursive"` / `HYBRID_MODEL = "text-embedding-3-small"` (also defined in `main.py`, `app/agent/router.py`, `app/documents/service.py`) name the one collection the live API actually serves from.

### Retrieval (`app/retrieval/searcher.py`)

`retrieve` (dense-only), `retrieve_hybrid` (dense + BM25 via Qdrant's native RRF), `retrieve_reranked` (hybrid over a `candidate_pool_size=20` pool, cross-encoder re-scored down to `top_k` — the default whenever `settings.use_reranker`), `retrieve_with_multimodal_quota` (two independently-filtered searches reserving fixed slots for table/figure vs. prose chunks, see Experiment 8's follow-up). `app/retrieval/reranker.py`'s `rerank()` uses `BAAI/bge-reranker-base`, `@lru_cache`-loaded once.

### Generation (`app/generation/`)

`generator.py`'s `generate_answer` / `generate_partial_answer` / `stream_answer` call chat completions at `temperature=0.0` for reproducibility, using `PromptRegistry` (`prompts/prompt_registry.py` + `prompt_registry.json`, which tracks version/description/`introduced_in`/`eval_scores` per template) to render versioned Jinja2 templates (`v1`/`v2` of `grounded_qa`, `query_reformulation`, `sufficiency_assessment`, plus `v1_partial_answer`). The grounded-QA prompt enforces context-only answers with `[doc_title, chunk N]` citations, a fixed refusal string when unsupported, and (as of the v2 rewrite in Experiment 5) explicit permission to synthesize facts stated across multiple chunks while still forbidding inference beyond what's stated. Cost is computed client-side from hardcoded per-token constants (`COST_PER_INPUT_TOKEN`/`COST_PER_OUTPUT_TOKEN`, currently GPT-4o-mini pricing) — update these if `llm_model` changes.

### Agentic retrieval loop (`app/agent/`, `POST /agent/query`)

`router.py` mirrors `main.py`'s cache-check → retrieve → generate shape, but retrieval is `loop.py`'s `run_agent_loop`: up to `MAX_ITERATIONS = 3` rounds of retrieve → dedupe-accumulate (by `chunk_id`, so repeated retrievals across iterations don't double-count) → LLM sufficiency-check (`sufficiency.py`, v2 prompt — v1 caused near-universal cap-reaching on this bounded corpus) → if insufficient, LLM query reformulation (`reformulation.py`, v2 prompt passes full `query_history` so the loop doesn't regenerate the same query) for the next iteration. Terminates on `sufficiency_reached` or the iteration cap; the router then does one final rerank over all accumulated chunks (`_finalize_chunks`) before generating — `generate_answer` if sufficiency was reached, `generate_partial_answer` (with `missing_aspects`) if the loop hit the cap. Response includes `iterations_used` and `loop_terminated_by` (`"sufficiency_reached"` / `"cap_reached"` / `"cache_hit"`). See Experiment 7: on this corpus 8/10 hard multi-hop queries terminate at iteration 1, so the loop is usually naive RAG plus one extra LLM call — kept opt-in, not the default, for that reason.

### Graph-augmented retrieval (`app/graph/`, eval-only — not wired into any router)

`extractor.py` does LLM entity/relation extraction per chunk; `writer.py` writes `Document`/`Chunk`/`Entity` nodes and `MENTIONS`/`RELATED_TO` edges into Neo4j, plus a native vector index on `Chunk.embedding` (`schema.py`, `VECTOR_INDEX_NAME`). `graph_searcher.py`'s `retrieve_graph()` runs a Neo4j vector-index query for `top_k * 3` candidates, then (when `rerank=True`) always expands via a 1-hop shared-entity traversal *and* a 2-hop `RELATED_TO` traversal, merges direct hits + both expansion pools, and re-scores everything with the same cross-encoder reranker used elsewhere. Both traversals weight candidate chunks by **inverse entity degree** (`sum(1/degree(entity))`, the graph analogue of IDF) rather than raw mention/path count — a plain count lets high-degree "hub" entities (e.g. "Transformer", degree 33 in this corpus) dominate purely by being mentioned everywhere; this was caught empirically when it surfaced an off-topic title-page chunk as the top 2-hop candidate. `rerank=False` is a cheaper fallback that only backfills direct hits with 1-hop candidates when they undershoot `top_k`, using a fabricated score floor. Historical note from Experiment 9: the original expansion path had a dead-code bug (`needs_expansion` was unreachable) so early "Graph RAG" numbers were silently plain vector search — fixed and re-measured before the findings in `EXPERIMENTS.md` were written.

### Semantic caching (`app/caching/`)

`cache.py`'s `SemanticCache` (Redis-backed) is namespaced by `build_key_prefix(embedding_model, retrieval_mode)` — cache entries from different embedding models or retrieval modes never collide or cross-match. `write()` stores `{query, embedding, response, ts}` as a Redis hash with a TTL (`semantic_cache_ttl_seconds`, default 86400s). `check()` does an **O(n) scan** (`scan_iter` + Python cosine similarity) over every live entry in the namespace and returns the best match if its similarity clears `semantic_cache_similarity_threshold` (default `0.75`, calibrated in Experiment 6 to sit near the empirical paraphrase floor rather than the calibration midpoint, trading some false misses for fewer false hits). Documented as fine at this project's scale (dozens–low hundreds of entries); a production port would move to Redis's native vector index (RediSearch HNSW / `FT.SEARCH ... KNN`) behind the same `check()`/`write()`/`flush()` interface. `flush()` (called automatically after every document upload — see below) deletes every `semcache:*` key regardless of namespace.

### Document catalog + upload (`app/documents/`, no tests yet)

`GET /documents` (`router.py` → `service.py`'s `list_documents_with_chunk_counts`) reads `corpus/manifest.yaml` and joins in live per-`doc_id` chunk counts by scrolling the hybrid Qdrant collection (`HYBRID_STRATEGY`/`HYBRID_MODEL`). `POST /documents/upload`: accepts `.pdf`/`.md` only, 20MB max; sanitizes the filename to its basename (`Path(...).name`, blocks path traversal); derives `doc_id` via `slugify()` and rejects duplicates (409); saves the file to `corpus/uploads/` (`save_upload`); appends a manifest entry (`append_manifest_entry` — preserves every prior entry, unlike `download_corpus.py`'s manifest writer which overwrites from a hardcoded list); loads and ingests just that one document into the live hybrid collection by reusing `scripts.ingest.ingest_strategy` with a single-document list (`ingest_uploaded_document`, leaves every other document's points untouched); then flushes the semantic cache so stale cached answers can't shadow the new document (a flush failure is logged and swallowed, not raised — the doc is already searchable at that point, so failing the request over a cache miss would be the wrong tradeoff). If ingestion itself fails after the file is saved and cataloged, the endpoint returns 500 but does **not** roll back the save/catalog — a re-run of `scripts.ingest` would pick the document up.

### Streaming (`app/streaming/pipeline.py`, `POST /query/stream`)

`stream_query_pipeline()` is a generator yielding SSE-formatted events in a fixed order: zero or more `event: token` frames (either the full cached answer as one frame on a cache hit, or streamed deltas from `client.chat.completions.create(..., stream=True, stream_options={"include_usage": True})` on a miss), then one `event: done`, then one `event: metadata` frame carrying `{sources, cost_usd, latency_ms, cache_hit, trace_id}` as JSON. Retrieval and the cache check happen in the calling endpoint (`main.py`) *before* this generator starts — it only handles generation and event framing. Cost is computed from the final chunk's `usage` object, same pricing constants as the sync path.

### Tracing (`app/tracing/`)

`langfuse.py`'s `get_langfuse()` returns an `lru_cache`d client built from `Settings`. `spans.py` provides `root_span`/`traced_span` context managers plus `new_trace_id()`/`flush_traces()`; when `settings.enable_tracing` is `False` (the default), every one of these becomes a no-op (`NoOpObservation`) instead of branching call sites — so tracing code is unconditionally safe to leave in regardless of the flag. This is the **v4 OTEL-based** Langfuse SDK — there is no `trace()`/`span()`/`generation()`/`.end()` API; use `start_as_current_observation(as_type="span"|"generation", ...)` as a context manager and `.update(...)` to attach output/usage/cost before the block exits.

### Evaluation (`app/eval/`, `scripts/eval_*.py`, `scripts/run_*_eval.py`)

`golden_dataset.py` loads `eval/golden_dataset.yaml` (`GoldenQuery`/`GoldenItem`) and validates every snippet is a literal (whitespace-normalized) substring of its source doc, failing fast if not. `matcher.py`'s `is_relevant()` decides relevance via `rapidfuzz.fuzz.partial_ratio` between a golden snippet and a retrieved chunk (same `doc_id` required) — deliberately not an embedding/LLM judge, so swapping the embedding model under test doesn't bias its own scoring. `metrics.py`/`runner.py` aggregate precision/recall/MRR@k per strategy. `ragas_dataset.py`/`ragas_runner.py` drive full-pipeline RAGAS (LLM-as-judge) evaluation. Every technique in `EXPERIMENTS.md` was adopted or rejected using this harness.

### Config (`app/config.py`)

Single `Settings` (pydantic-settings) via `get_settings()` (`lru_cache`d), reading `backend/.env`. `load_dotenv()` runs at module import time, **before** `Settings` is defined — keep that ordering, since downstream modules (e.g. `embedder.py`'s `OpenAI()` client) read `OPENAI_API_KEY` straight from `os.environ` at import time, not from the `Settings` object. See [`.env.example`](./.env.example) / [`README.md`](./README.md) for the full variable table.

## Cost-incurring operations

These hit paid external APIs — don't run them speculatively while debugging something unrelated, and don't loop/retry them blindly on failure:

- `python -m scripts.ingest` and any `scripts/ingest_*.py` — OpenAI embedding calls for every chunk (unless `--embedding-model` is a local BGE model). `scripts/ingest_figures.py` additionally calls GPT-4o Vision per figure, which is the most expensive ingestion path in the repo.
- `scripts/run_ragas_eval.py`, `scripts/run_comparison_eval.py`, `scripts/run_graph_comparison_eval.py`, `scripts/run_multimodal_comparison.py`, `scripts/eval_graph.py` — each makes an LLM call per question (generation) plus more LLM calls for RAGAS's own judge metrics. These are the scripts behind `EXPERIMENTS.md` and can cost real money to re-run at full scale.
- Live `/query`, `/query/stream`, and `/agent/query` requests against a running server — the agentic path (`/agent/query`) can trigger multiple LLM calls per request (see Experiment 7).
- If you need to sanity-check ingestion/retrieval logic without spending anything, prefer the debugging tools below or a local BGE embedding model over re-running a full OpenAI-backed ingest.

## Debugging tools

Prefer these over ad hoc scripting when investigating a retrieval/ingestion problem — they already exist and print exactly the kind of internal state you'd otherwise reach for `print()` to inspect:

- `scripts/inspect_chunking.py` — shows how a document splits under a given chunking strategy.
- `scripts/inspect_graph.py` — inspects the Neo4j entity graph for a document/query (node/edge counts, degree, what a query's traversal actually pulls in).
- `scripts/check_tables.py` — shows what Docling's table extraction produced for the corpus's PDFs.

## Generated / data files — don't hand-edit

- `uv.lock` — committed, but regenerate via `uv add`/`uv sync`, never hand-edit.
- `.ruff_cache/` — tool-managed, self-ignoring.
- `eval/results/*.json` — raw output of the eval scripts above; regenerate by re-running the script that produced it, don't hand-tweak numbers to "fix" a result.
- `app/generation/prompts/prompt_registry.json` — update in lockstep with a new `.jinja2` template file, not standalone: adding a prompt version means adding both the template *and* a registry entry (`name`/`version`/`file`/`description`/`introduced_in`/`eval_scores`).
- `corpus/manifest.yaml` — hand-editable, but see Data identity below; prefer the upload endpoint or `append_manifest_entry`-style appends over rewriting the whole file.

## Data identity

- `Chunk.chunk_id` is `"{doc_id}_{strategy_name}_{chunk_index}"` (`base_chunker.py`'s `_make_chunk`). Qdrant point IDs are `uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)`, since Qdrant requires int/UUID IDs.
- `doc_id` / manifest entries are the join key between `corpus/manifest.yaml`, `Document`, `Chunk`, and Qdrant/Neo4j payloads — adding a document requires both a file on disk (or an upload) and a `manifest.yaml` entry before `scripts/ingest.py` (or the upload endpoint) will pick it up.
- Qdrant collections are named per `(chunking_strategy, embedding_model)` pair (`collection_name_for`) — don't assume there's a single collection; the live API only ever queries the one named by `HYBRID_STRATEGY`/`HYBRID_MODEL`.

## Conventions

- Measure before adopting: every retrieval/generation change in this codebase has a paired eval run in `EXPERIMENTS.md`. Extend that log rather than skipping straight to "this should be better."
- Secrets never get committed — `.env` is gitignored; use `.env.example` as the template.
- `corpus/uploads/` is runtime-generated data (documents added via the upload feature), not a fixture to hand-edit.
