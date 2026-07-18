# DocMind Backend

FastAPI service that ingests a fixed document corpus into Qdrant, retrieves relevant chunks for a question, and generates a grounded, cited answer via an LLM — with hybrid search, reranking, semantic caching, an agentic retrieval loop, graph-augmented retrieval, and full Langfuse tracing on every query.

For the overall project story (architecture, key decisions, experiment findings, how the frontend fits in), see the [root README](../README.md). For a deep, code-level architecture reference written for agentic coding tools (and equally useful for humans), see [AGENTS.md](./AGENTS.md) (`CLAUDE.md` just points there). For the full experiment log this README's findings are condensed from, see [EXPERIMENTS.md](./EXPERIMENTS.md).

## Directory structure

```
backend/
├── main.py                     # FastAPI app: /health, /query, /query/stream + agent/documents routers
├── pyproject.toml, uv.lock     # uv-managed Python 3.13 project, dependencies, ruff/mypy/pytest config
├── docker-compose.yml          # Qdrant + Redis + Neo4j
├── CLAUDE.md                   # deep architecture/agentic-dev reference
├── EXPERIMENTS.md              # full RAG-quality experiment log (10 experiments)
│
├── app/
│   ├── config.py                # Settings (pydantic-settings), loaded from .env
│   │
│   ├── ingestion/                # offline corpus → vector pipeline
│   │   ├── loader.py              # reads corpus/manifest.yaml, loads PDF (PyMuPDF) / markdown into Document
│   │   ├── chunker/                # pluggable chunking strategies
│   │   │   ├── base_chunker.py     # Chunk dataclass, ChunkStrategy enum, BaseChunker (tiktoken cl100k_base)
│   │   │   ├── fixed_size_chunker.py
│   │   │   ├── recursive_chunker.py     # production default (see Experiment 1)
│   │   │   ├── structure_aware_chunker.py
│   │   │   ├── table_chunker.py / figure_chunker.py
│   │   │   └── chunk_registry.py    # get_chunker(strategy, size, overlap) factory
│   │   ├── embedder.py             # embed_chunks/embed_query — OpenAI (default) or local BGE via sentence-transformers
│   │   ├── sparse_embedder.py      # BM25 sparse vectors (fastembed) for hybrid search
│   │   ├── indexer.py              # Qdrant collection mgmt + upsert (dense/hybrid, table/figure variants)
│   │   ├── table_extractor.py      # Docling TableFormer table extraction
│   │   └── figure_extractor.py     # Docling picture detection + GPT-4o Vision captioning
│   │
│   ├── retrieval/
│   │   ├── searcher.py             # retrieve / retrieve_hybrid / retrieve_reranked / retrieve_with_multimodal_quota
│   │   └── reranker.py             # cross-encoder rerank() via BAAI/bge-reranker-base
│   │
│   ├── generation/
│   │   ├── generator.py            # generate_answer / generate_partial_answer / stream_answer (OpenAI chat)
│   │   └── prompts/                 # versioned Jinja2 prompt templates
│   │       ├── prompt_registry.py    # PromptRegistry — versioned template loader
│   │       ├── prompt_registry.json  # version metadata + eval scores per prompt
│   │       └── v1/v2 templates       # grounded_qa, partial_answer, query_reformulation, sufficiency_assessment
│   │
│   ├── agent/                      # agentic RAG: iterative retrieve → assess → reformulate loop
│   │   ├── router.py                # POST /agent/query
│   │   ├── loop.py                  # run_agent_loop() — max 3 iterations
│   │   ├── reformulation.py         # LLM query reformulation given missing aspects
│   │   └── sufficiency.py           # LLM judge: "is retrieved context sufficient?"
│   │
│   ├── graph/                      # Neo4j GraphRAG — entity-graph-augmented retrieval (eval-only, not in the live API)
│   │   ├── extractor.py             # LLM entity/relation extraction
│   │   ├── writer.py                # writes Document/Chunk/Entity nodes + MENTIONS/RELATED_TO edges
│   │   └── graph_searcher.py        # vector search + 1-hop/2-hop entity expansion + rerank
│   │
│   ├── caching/                    # Redis-backed semantic response cache
│   │   └── cache.py                 # SemanticCache: check()/write()/flush(), cosine-similarity scan
│   │
│   ├── documents/                  # document catalog + upload API
│   │   ├── router.py                # GET /documents, POST /documents/upload
│   │   └── service.py               # save upload, manifest append, ingest-on-upload, cache flush
│   │
│   ├── repo_ingest/                # GitHub repo ingestion (per-repo hybrid Qdrant collections)
│   │   ├── router.py                # POST /ingest/repo, POST /ingest/files, GET /ingest/status/{job_id}
│   │   ├── service.py               # run_full_ingest / run_incremental_ingest orchestration
│   │   ├── github.py                # tarball fetch, ref→SHA resolution, compare, file content
│   │   ├── filters.py               # ingestable-file selection, Document construction
│   │   └── job_store.py             # Redis-backed job records, per-repo lock, ingest watermark
│   │
│   ├── streaming/pipeline.py       # SSE generator for /query/stream (token → done → metadata events)
│   ├── tracing/                    # Langfuse (v4 OTEL SDK) spans — root/retrieval/generation
│   └── eval/                       # retrieval + RAGAS evaluation harness
│       ├── golden_dataset.py        # loads eval/golden_dataset.yaml, validates snippet⊂doc
│       ├── matcher.py               # is_relevant() via rapidfuzz partial-ratio (not an embedding judge)
│       ├── metrics.py / runner.py   # precision/recall/MRR@k aggregation
│       └── ragas_dataset.py / ragas_runner.py  # RAGAS-based LLM-judge eval
│
├── corpus/
│   ├── manifest.yaml            # doc_id/title/path/type/tags catalog — the join key across the whole pipeline
│   ├── pdfs/                     # Attention Is All You Need, RAG (Lewis 2020), RAGAS (Es 2023)
│   ├── repos/                    # README snapshots used as a markdown corpus: FastAPI, Langfuse, Qdrant, RAGAS, tiktoken
│   └── uploads/                  # documents added via POST /documents/upload land here
│
├── eval/                        # golden/RAGAS/calibration datasets + eval/results/*.json (raw experiment outputs)
└── scripts/                     # ~22 CLI entrypoints — ingestion, evaluation, calibration, inspection
```

## Architecture

**Request flow for `POST /query`** (`main.py`): embed query → check semantic cache → retrieve (dense, hybrid, or hybrid+reranked, per settings) → generate a grounded answer → return `{answer, sources, cost_usd, latency_ms, trace_id, cache_hit}`. The whole request is wrapped in nested Langfuse spans (root → cache-check → retrieval → generation).

**Ingestion** (`app/ingestion/`, driven by `scripts/ingest.py`): `loader.py` reads `corpus/manifest.yaml` and loads each document; `chunker/` splits it (recursive chunking is the production default — see Experiment 1 below); `embedder.py` embeds chunks via OpenAI or a local BGE model; `sparse_embedder.py` adds BM25 sparse vectors for hybrid search; `indexer.py` upserts everything into a Qdrant collection named for its `(chunking_strategy, embedding_model)` pair, so multiple strategies/models can be compared side by side without collisions. `table_extractor.py` / `figure_extractor.py` (Docling-based) extract structured table and figure content into a separate multimodal collection.

**Retrieval** (`app/retrieval/searcher.py`) has three modes: `retrieve` (dense-only), `retrieve_hybrid` (dense + BM25 fused server-side via Qdrant RRF), `retrieve_reranked` (hybrid over a 20-candidate pool, cross-encoder re-scored down to top_k — the default when `use_reranker=True`). `retrieve_with_multimodal_quota` reserves separate slots for table/figure chunks vs. prose.

**Generation** (`app/generation/generator.py`) calls the chat completions API at `temperature=0.0` for reproducibility, using versioned Jinja2 prompts (`PromptRegistry`) that enforce context-only answers with `[doc_title, chunk N]` citations and a fixed refusal string when the answer isn't supported by context. Cost is computed client-side from hardcoded per-token pricing constants.

**Agentic retrieval** (`app/agent/`) is a separate, opt-in path (`POST /agent/query`): up to 3 iterations of retrieve → LLM sufficiency-check → LLM query reformulation on missing aspects, terminating on `sufficiency_reached` or the iteration cap. See Experiment 7 for why this isn't the default.

**Graph-augmented retrieval** (`app/graph/`) extracts entities/relations from chunks into Neo4j (`Document`/`Chunk`/`Entity` nodes, `MENTIONS`/`RELATED_TO` edges) and retrieves via vector search + entity-graph expansion. **It's fully implemented and evaluated (Experiment 9) but not exposed through the live API** — only invoked from `scripts/eval_graph.py` / `scripts/run_graph_comparison_eval.py`.

**Semantic caching** (`app/caching/`) is a Redis-backed cache keyed by `(scope, embedding_model, retrieval_mode)` — `scope` is `"docs"` for the fixed corpus or a repo slug for a repo-scoped query, so a cached docs answer can never be served for a repo question or vice versa — checked via a cosine-similarity scan before retrieval runs and flushed automatically whenever a new document or repo is ingested (see Experiment 6 for threshold calibration).

**Repo ingestion** (`app/repo_ingest/`, `POST /ingest/repo` / `POST /ingest/files`) indexes a GitHub repo into its own hybrid Qdrant collection, separate from the fixed docs corpus, so `POST /query`/`POST /query/stream` can pass an optional `repo` field to search it instead. `POST /ingest/repo` downloads the repo tarball pinned to a resolved commit SHA, chunks every ingestable file with a language-aware `CodeChunker` (`app/ingestion/chunker/code_chunker.py` — separator hierarchies keep functions/classes intact per language, e.g. splitting Python on `class`/`def` before falling back to blank lines), embeds, and upserts — then sweeps every point whose `commit_sha` doesn't match the run (mark-and-sweep), so deleted/shrunk files don't linger. `POST /ingest/files` is the incremental counterpart: it diffs against the last successfully ingested commit (tracked as a per-repo "watermark") via GitHub's compare API and applies only the changed files, falling back to a full re-ingest when there's no watermark yet, the history diverged (force-push), or too many files changed. Both run as a FastAPI `BackgroundTasks` job behind a per-repo Redis lock (so concurrent ingests for the same repo 409 instead of racing) and report progress via `GET /ingest/status/{job_id}`. Point IDs are deterministic (`uuid5` of a chunk ID that embeds the file path), so re-ingesting the same repo/commit is idempotent.

**Evaluation** (`app/eval/`, `scripts/eval_*.py`, `scripts/run_*_eval.py`) is not a bolt-on — every retrieval/generation change in this project was measured before being adopted. Retrieval-only eval uses `rapidfuzz` fuzzy string matching against a hand-verified golden set (deliberately *not* an embedding-based judge, to avoid biasing results toward whichever embedding model is under test). End-to-end pipeline eval uses RAGAS (LLM-as-judge) against a 35-question dataset spanning single-doc, multi-doc-synthesis, and deliberately-unanswerable questions.

## API reference

| Method | Path | Router | Description |
|---|---|---|---|
| `GET` | `/health` | `main.py` | Liveness check |
| `POST` | `/query` | `main.py` | Full sync RAG pipeline: embed → cache check → retrieve (dense/hybrid/reranked) → generate → trace. Optional `repo` field queries a repo ingested via `/ingest/repo` instead of the docs corpus (404 if it hasn't been ingested) |
| `POST` | `/query/stream` | `main.py` | Same pipeline (incl. optional `repo`), server-sent events (`token` → `done` → `metadata`) |
| `POST` | `/agent/query` | `app/agent/router.py` | Agentic RAG: iterative retrieve/assess/reformulate loop, up to 3 iterations |
| `GET` | `/documents` | `app/documents/router.py` | Lists cataloged documents with live chunk counts from Qdrant |
| `POST` | `/documents/upload` | `app/documents/router.py` | Uploads a `.pdf`/`.md` (max 20MB), ingests it immediately into the live hybrid collection, flushes the semantic cache |
| `POST` | `/ingest/repo` | `app/repo_ingest/router.py` | `{repo, ref?}` — bulk-ingests a GitHub repo into its own hybrid collection. Returns `202 {job_id, ...}` immediately; runs as a background job |
| `POST` | `/ingest/files` | `app/repo_ingest/router.py` | Same request/response shape; incrementally re-ingests a repo by diffing against the last ingested commit instead of re-downloading the whole tarball |
| `GET` | `/ingest/status/{job_id}` | `app/repo_ingest/router.py` | Job status (`pending`/`running`/`completed`/`failed`) plus file/chunk counters |

## MCP server

`app/mcp/` exposes 4 of the operations above as [MCP](https://modelcontextprotocol.io) tools over stdio, for use from an MCP-aware client (Claude Code, Claude Desktop) instead of the HTTP API directly:

| Tool | Wraps | Notes |
|---|---|---|
| `ingest_repo(repo, ref?)` | `POST /ingest/repo` | Full ingest; returns a `job_id` immediately |
| `sync_repo_incremental(repo, ref?)` | `POST /ingest/files` | Incremental re-ingest; much cheaper once a repo is already indexed |
| `get_ingest_status(job_id)` | `GET /ingest/status/{job_id}` | Poll a job started by either ingest tool |
| `query_repo(repo, question, top_k?)` | `POST /query` (with `repo` set) | Hybrid+reranked Q&A against an already-ingested repo |

**Not exposed**: docs-corpus query, `/agent/query`, `/documents` catalog/upload, `/query/stream` — out of scope for this tool surface. The 4 tools are thin wrappers around the same service-layer functions (`app/query/service.py::run_query`, `app/repo_ingest/service.py::prepare_ingest_job`/`run_full_ingest`/`run_incremental_ingest`) the HTTP routes call, so behavior (validation, locking, tracing) is identical either way.

Run it from `backend/` (same `.env` as the FastAPI app — no separate config), matching this project's `python -m` convention for every other entrypoint (`scripts.download_corpus`, `scripts.ingest`, ...):

```bash
uv run python -m app.mcp.server
```

Example client config (same shape for Claude Code's `.mcp.json` and Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "docmind": {
      "command": "uv",
      "args": ["run", "python", "-m", "app.mcp.server"],
      "cwd": "/absolute/path/to/DocMind/backend"
    }
  }
}
```

No auth is required or enforced — stdio's trust boundary is "whoever can spawn this local subprocess," matching the rest of this no-auth, local-dev-only backend. To manually inspect the 4 tools/schemas without a full client, use FastMCP's own bundled CLI: `PYTHONPATH=. uv run fastmcp inspect app/mcp/server.py` (add `--format fastmcp` for the full JSON report). `PYTHONPATH=.` is required because the CLI loads `app/mcp/server.py` as a standalone file rather than via `python -m`, so `backend/` needs to be on `sys.path` explicitly for its `app.*` imports to resolve — the server itself doesn't need this when run via `python -m app.mcp.server`, which already puts the current directory on `sys.path`.

## Setup & running

Requires Python 3.13, [`uv`](https://docs.astral.sh/uv/), Docker, an OpenAI API key, and a Langfuse project.

```bash
cd backend
uv sync                                    # install dependencies
docker compose up -d                       # start Qdrant (:6333/:6334), Redis (:6379), Neo4j (:7474/:7687)
cp .env.example .env                       # then fill in the required keys below
python -m scripts.download_corpus          # fetch/refresh the corpus + manifest
python -m scripts.ingest --strategy all    # chunk → embed → upsert into Qdrant
uvicorn main:app --reload                  # serves on :8000
```

### Environment variables (`.env`)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | **yes** | — | Embeddings + chat completions |
| `LANGFUSE_SECRET_KEY` | **yes** | — | Tracing |
| `LANGFUSE_PUBLIC_KEY` | **yes** | — | Tracing |
| `LANGFUSE_BASE_URL` | **yes** | — | Tracing (self-hosted or cloud Langfuse) |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` | Set to `BAAI/bge-large-en-v1.5` to use the local embedding model instead |
| `LLM_MODEL` | no | `gpt-4o-mini` | Generation model |
| `USE_RERANKER` | no | `true` | Enable cross-encoder reranking on the hybrid path |
| `ENABLE_TRACING` | no | `false` | Toggle Langfuse tracing |
| `ENABLE_SEMANTIC_CACHE` | no | `false` | Toggle the Redis semantic cache |
| `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | no | `0.75` | Cache hit threshold (see Experiment 6) |
| `QDRANT_HOST` / `QDRANT_PORT` | no | `localhost` / `6333` | Vector store |
| `REDIS_HOST` / `REDIS_PORT` | no | `localhost` / `6379` | Semantic cache |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | no | `bolt://localhost:7687` / `neo4j` / `password` | GraphRAG (eval scripts only) |
| `GITHUB_TOKEN` | no | — | Fine-grained PAT for repo ingestion. Without it, only public repos work and the GitHub API is capped at 60 req/hr (vs. 5000/hr authenticated) |
| `INGEST_JOB_TTL_SECONDS` | no | `604800` (7 days) | How long a repo-ingestion job record stays queryable via `GET /ingest/status/{job_id}` after it finishes |
| `MCP_INGEST_MAX_CONCURRENCY` | no | `2` | Max concurrent repo-ingest jobs the MCP server (`app/mcp/`) will run at once, across repos |

See [`.env.example`](./.env.example) for a ready-to-copy template. The four "required" variables above have no defaults — the app will fail to start without them.

## Ingestion & corpus

`corpus/manifest.yaml` is the source of truth for what gets ingested: 3 foundational PDFs (Attention Is All You Need, RAG (Lewis 2020), RAGAS (Es 2023)) plus 5 tool READMEs (FastAPI, Langfuse, Qdrant, RAGAS, tiktoken) used as a markdown corpus, plus whatever has been added via `POST /documents/upload` (lands in `corpus/uploads/`). Adding a document requires both the file on disk and a `manifest.yaml` entry before `scripts/ingest.py` will pick it up. Qdrant collections are named per `(chunking_strategy, embedding_model)` — re-ingesting with a different strategy or model creates a new collection rather than overwriting the existing one, which is what makes side-by-side comparison (Experiments 1–2) possible.

GitHub repos ingested via `POST /ingest/repo` are a separate track from the docs corpus: each repo gets its own hybrid Qdrant collection (`docmind_repo_{owner-name}_{model}_hybrid`), keyed only by the repo slug — no `manifest.yaml` entry needed. The `code` chunking strategy this uses is deliberately excluded from `python -m scripts.ingest --strategy all` (see `scripts/ingest.py`'s `CORPUS_STRATEGIES`), so running the corpus ingest script never touches repo collections or spends embedding budget on them.

## Scripts (`scripts/`)

| Category | Scripts |
|---|---|
| Ingestion | `ingest.py`, `ingest_tables.py`, `ingest_figures.py`, `ingest_graph.py`, `download_corpus.py` |
| Retrieval eval | `eval_chunking.py`, `eval_hybrid.py`, `eval_rerank.py`, `eval_graph.py` |
| Pipeline / RAGAS eval | `run_comparison_eval.py`, `run_graph_comparison_eval.py`, `run_multimodal_comparison.py`, `run_ragas_eval.py`, `analyze_ragas_results.py` |
| Caching | `calibrate_cache_threshold.py`, `measure_cache_performance.py` |
| Judge calibration | `build_judge_calibration_sample.py`, `analyze_judge_calibration.py` |
| Golden dataset | `generate_golden_dataset.py`, `validate_ragas_golden_set.py` |
| Inspection | `inspect_chunking.py`, `inspect_graph.py`, `check_tables.py` |

## Testing

```bash
pytest                                                                       # full suite
pytest app/ingestion/chunker/tests/test_recursive_chunker.py::test_name     # single test
ruff check .                                                                 # lint
```

Tests are colocated per module: `app/agent/tests/`, `app/caching/tests/`, `app/ingestion/tests/` + `app/ingestion/chunker/tests/`, `app/retrieval/tests/`, `app/streaming/tests/`, `app/repo_ingest/tests/`, `scripts/tests/`, `tests/` (root-level, for `main.py`). **`app/documents/` and `app/graph/` currently have no tests.**

## Further reading

- [`EXPERIMENTS.md`](./EXPERIMENTS.md) — full hypothesis → measurement → finding → decision write-up for all 10 experiments (chunking, embeddings, hybrid search, reranking, RAGAS eval, semantic caching, agentic RAG, multimodal retrieval, GraphRAG, LLM-judge calibration).
- [`AGENTS.md`](./AGENTS.md) — deep architecture and conventions reference (data-identity rules, config-loading order caveats, embedding-provider dispatch details, per-module breakdowns for the agent loop, caching, document upload, graph retrieval, streaming, and tracing). `CLAUDE.md` in this directory just points here.
