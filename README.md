# DocMind

**An eval-driven Retrieval-Augmented Generation system** — a Next.js chat UI backed by a FastAPI service that answers questions grounded in an ingested document corpus, with citations, streaming, semantic caching, and full cost/latency tracing. Every retrieval technique in the pipeline (chunking, hybrid search, reranking, agentic iteration, multimodal retrieval, knowledge-graph retrieval) was added by hypothesis → measurement → decision, not by assumption — see [Key Findings](#key-findings-from-experiments) below.

## Project summary

DocMind ingests a fixed corpus — three foundational papers (*Attention Is All You Need*, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, *RAGAS*) plus five tool READMEs (FastAPI, Langfuse, Qdrant, RAGAS, tiktoken) — into Qdrant, and lets a user ask natural-language questions about them through a chat interface. Answers are generated only from retrieved context, cited back to their source chunk, and streamed token-by-token. Users can also upload their own `.pdf`/`.md` documents through the UI, which are ingested into the live index immediately.

What makes this more than a LangChain-wrapper demo is the methodology: the ~3.5-week build (see `git log`) is a sequence of deliberate, measured upgrades — chunking strategy → embedding model → hybrid search → cross-encoder reranking → RAGAS evaluation → semantic caching → an agentic iterative-retrieval loop → multimodal (table/figure) retrieval → knowledge-graph retrieval → LLM-judge calibration — where each step is backed by a written hypothesis, a measured result against a fixed golden query set, and an explicit decision (adopt, reject, or "it's a tradeoff, here's when to use it"). Two of those experiments caught real bugs through measurement alone: Graph RAG's entity-expansion query turning out to be dead code, and a cross-encoder reranker silently picking the wrong table for a numeric answer. That log lives in full in [`backend/EXPERIMENTS.md`](./backend/EXPERIMENTS.md).

## New to RAG? Start here

If you're using this repo to *learn* RAG rather than to evaluate it as a portfolio project, the sections below (and `backend/EXPERIMENTS.md`) assume you already know the vocabulary. This is a curriculum-ordered path through the same code and experiments, going from the basic four-step RAG loop to the advanced retrieval strategies, each with the exact files and experiment number to read next.

1. **What is RAG, and what does the basic loop look like?** Four steps: chunk a document, embed the chunks, retrieve the closest ones to a query, generate an answer from them. Read `backend/app/ingestion/loader.py` → `chunker/` → `embedder.py` → `indexer.py` (how a document becomes searchable vectors), then `backend/app/retrieval/searcher.py`'s `retrieve()` (dense-only search) and `backend/app/generation/generator.py`'s `generate_answer()` (turning retrieved chunks into an answer). Everything else in this repo is a refinement of one of these four steps.
2. **Does chunking strategy matter?** Experiment 1 in `backend/EXPERIMENTS.md` — code in `backend/app/ingestion/chunker/`.
3. **Does embedding model choice matter?** Experiment 2 — code in `backend/app/ingestion/embedder.py`.
4. **Beyond vector similarity: lexical/keyword search.** Experiment 3 (hybrid dense + BM25 via Reciprocal Rank Fusion) — code in `backend/app/retrieval/searcher.py`'s `retrieve_hybrid`.
5. **Refining the ranking: cross-encoder reranking.** Experiment 4 — code in `backend/app/retrieval/reranker.py`.
6. **How do you know if any of this is actually working?** This is evaluation, arguably the most important skill in the whole repo. Read `backend/app/eval/` and `backend/eval/golden_dataset.yaml`, then Experiment 5 (full-pipeline RAGAS/LLM-as-judge evaluation).
7. **Production concerns: cost and latency.** Experiment 6 (semantic caching) — code in `backend/app/caching/`.
8. **Advanced retrieval strategies** — each is a different answer to "what if one retrieval pass over plain text isn't enough?":
   - Agentic RAG (iterative retrieve → assess → reformulate) — Experiment 7, `backend/app/agent/`.
   - Multimodal retrieval (tables/figures, not just prose) — Experiment 8, `backend/app/ingestion/table_extractor.py` / `figure_extractor.py`.
   - GraphRAG (entity-graph traversal alongside vector search) — Experiment 9, `backend/app/graph/`.
9. **Can you trust your own evaluator?** Experiment 10 — the meta-lesson that closes the loop: even an LLM-as-judge eval harness needs to be evaluated before you trust its fine-grained scores.

### RAG concepts used in this repo, and where they live

| Concept | What it means | Code | Experiment |
|---|---|---|---|
| Chunking | Splitting a document into small retrievable pieces — you can't hand an LLM an entire PDF as context | `backend/app/ingestion/chunker/` | 1 |
| Embedding | Turning text into a vector, so "similar meaning" becomes "close together in vector space" | `backend/app/ingestion/embedder.py` | 2 |
| Dense retrieval | Finding chunks whose embedding is closest to the query's embedding (cosine similarity) | `backend/app/retrieval/searcher.py` (`retrieve`) | 1, 2 |
| Sparse retrieval (BM25) | Classic keyword/lexical search — scores exact term overlap, no embeddings involved | `backend/app/ingestion/sparse_embedder.py` | 3 |
| Hybrid search + RRF | Running dense and sparse search in parallel and fusing the two rankings (Reciprocal Rank Fusion), catching both semantic and exact-keyword matches | `backend/app/retrieval/searcher.py` (`retrieve_hybrid`) | 3 |
| Reranking (cross-encoder) | A second, slower model that scores a (query, chunk) pair jointly instead of comparing precomputed vectors — more accurate at picking the single best match | `backend/app/retrieval/reranker.py` | 4 |
| Precision / Recall / MRR@k | Precision = how many of the *k* returned chunks are actually relevant; recall = how many relevant chunks you found out of all that exist; MRR = how high up the list the first relevant one lands | `backend/app/eval/metrics.py` | 1–4 |
| RAGAS / LLM-as-judge | Using an LLM to score answer quality (faithfulness, relevancy, etc.) against retrieved context — there's no simple string match for "correct" on open-ended questions | `backend/app/eval/ragas_runner.py` | 5, 10 |
| Semantic caching | Skipping the LLM call entirely for a paraphrased repeat question by comparing the new query's embedding against past queries | `backend/app/caching/` | 6 |
| Agentic RAG | Letting the system judge its own retrieved context as insufficient, reformulate the question, and retrieve again — instead of one fixed retrieval pass | `backend/app/agent/` | 7 |
| Multimodal retrieval | Extracting and indexing tables/figures, not just prose, so questions about numbers in a table can be answered | `backend/app/ingestion/table_extractor.py`, `figure_extractor.py` | 8 |
| GraphRAG | Extracting entities/relationships into a graph database and traversing it — not just vector similarity — to pull in connected context a pure vector search would miss | `backend/app/graph/` | 9 |

## How to navigate this repo

This README is the front door. Suggested reading order, depending on how deep you want to go:

1. **You are here** — skim [Architecture](#architecture), [Key Engineering Decisions](#key-engineering-decisions), and [Key Findings from Experiments](#key-findings-from-experiments) below for the whole-project picture.
2. **[`backend/EXPERIMENTS.md`](./backend/EXPERIMENTS.md)** — the full experiment log: 10 write-ups, each with a hypothesis, a results table, and a decision. This is the strongest evidence of how the system was actually built.
3. **[`backend/README.md`](./backend/README.md)** and **[`frontend/README.md`](./frontend/README.md)** — a file-by-file tour of each half of the codebase: directory structure, module responsibilities, API reference, setup instructions.
4. **[`backend/AGENTS.md`](./backend/AGENTS.md)** and **[`frontend/AGENTS.md`](./frontend/AGENTS.md)** — the deepest layer: written as operating instructions for agentic coding tools, so they're terse, precise, and call out non-obvious gotchas (data-identity rules, stale-closure fixes, config-loading order) rather than re-explaining what the code already shows. Each directory's `CLAUDE.md` just points to its `AGENTS.md`.

If you just want specific code: the RAG pipeline core is `backend/app/{ingestion,retrieval,generation}/`, the experiment scripts are `backend/scripts/`, and the chat UI is `frontend/app/page.tsx` + `frontend/components/chat/`.

## Architecture

**The generic RAG shape**, independent of any tool in this stack — every RAG system is some version of this:

```
INGESTION (offline, once per document)
  Document → Chunk → Embed → Index (vector store)

QUERY (online, per user question)
  Question → Embed → Retrieve (± rerank) → Augment prompt with retrieved chunks → Generate → Answer + citations
```

**How DocMind implements it**, service by service:

```
 Browser (React)          Next.js 16 (App Router)              FastAPI backend
 components/chat/         app/api/**/route.ts                  main.py + routers
 components/documents/  →  (same-origin BFF proxy, SSE)  →     /query  /query/stream
                                                                /agent/query  /documents
                                                                          │
                              ┌───────────────┬───────────────┼───────────────┬───────────────┐
                              │               │               │               │               │
                          Qdrant           Redis           Neo4j           OpenAI          Langfuse
                       vector store   semantic cache   GraphRAG (eval)  embed + chat        tracing
```

The browser never talks to FastAPI directly — every request goes through Next.js Route Handlers that proxy to the backend, keeping the backend URL server-side only. See [`frontend/README.md`](./frontend/README.md) for the proxy pattern and [`backend/README.md`](./backend/README.md) for the API surface.

**Pipeline components** (all under `backend/app/`):
- **Ingestion** — loads the corpus (`corpus/manifest.yaml`), chunks it (pluggable strategies; recursive chunking is the production default), embeds it (OpenAI or local BGE), and indexes it into Qdrant. Table/figure content is extracted separately via Docling + GPT-4o Vision for multimodal retrieval.
- **Retrieval** — three modes: dense-only, hybrid (dense + BM25 via Qdrant RRF), and hybrid+reranked (cross-encoder over a 20-candidate pool) — the last is the default.
- **Generation** — grounded, cited answers via versioned Jinja2 prompts, streamed over SSE, with client-side cost accounting per request.
- **Agentic retrieval** (`POST /agent/query`) — an opt-in iterative retrieve → assess-sufficiency → reformulate loop, capped at 3 iterations.
- **Graph-augmented retrieval** (`backend/app/graph/`) — Neo4j entity-graph expansion on top of vector search. **Fully built and evaluated, but not wired into the live API** — it's an eval-only alternative retrieval mode today.
- **Semantic caching** — Redis-backed, cosine-similarity threshold calibrated against real paraphrase pairs, flushed automatically on every document upload.
- **Tracing** — every query is wrapped in nested Langfuse (v4 OTEL SDK) spans, so cost, latency, and token usage are attached to every request, not just logged separately.
- **Evaluation** — a first-class part of the system, not an afterthought: retrieval-only eval (`rapidfuzz`-based, avoids embedding-judge bias) and full-pipeline eval (RAGAS, LLM-as-judge) both run against fixed golden datasets, and every pipeline change in `EXPERIMENTS.md` is backed by one of these.

## Key engineering decisions

Condensed from the full write-ups in [`backend/EXPERIMENTS.md`](./backend/EXPERIMENTS.md):

- **Recursive chunking**, not fixed-size or structure-aware — won on precision/recall/MRR outright, even on markdown/code docs where structure-aware was hypothesized to win.
- **OpenAI `text-embedding-3-small`**, not local BGE — BGE is free but drops recall 10–12.5pp across every chunking strategy on this corpus; the cost of hosted embeddings buys back real retrieval quality here.
- **Hybrid (dense + BM25 via Qdrant RRF) as the default retrieval mode** — beats dense-only on every metric (recall +9pp, MRR +14%) at the cost of one extra local BM25 encode per query.
- **Cross-encoder reranking, kept despite a precision/recall dip** — biggest MRR gain in the pipeline (+13%), because scoring (query, chunk) pairs jointly beats comparing independently-computed scores; a QA pipeline cares more about finding *a* correct chunk fast than recovering every relevant span.
- **One Qdrant collection per `(chunking_strategy, embedding_model)` pair** — makes side-by-side comparison experiments (like the two above) possible without overwriting data, at the cost of some storage duplication.
- **Semantic cache threshold set to 0.75, closer to the empirical paraphrase floor than the calibrated midpoint** — deliberately trades some cache misses for fewer false hits, because serving a wrong cached answer is a worse failure than paying full retrieval+generation cost again.
- **Agentic retrieval loop kept opt-in, not the default** — context_precision improves (+9.3pp) but context_recall is exactly flat and cost is 2.5x higher; the corpus ceiling, not iteration count, is the binding constraint, so the tradeoff isn't a clean win.
- **Graph RAG positioned as a third retrieval option, not a replacement** — cheaper and lower-latency than agentic RAG, better than naive RAG on multi-hop answer_relevancy and recall, but not exposed via the live API yet.
- **Prompt versioning via a `PromptRegistry`** (Jinja2 templates + JSON metadata) rather than inline strings — every generation-prompt change (e.g. the multi-doc-synthesis refusal fix in Experiment 5) is tracked with its own eval scores.
- **A second LLM run as judge-of-the-judge** — RAGAS scores were independently re-scored to measure inter-judge disagreement (30.4%), which is what surfaced RAGAS's blind spot on KV-formatted table text and confirmed RAGAS is reliable for ranking pipelines but not for sub-0.1 precision comparisons.

## How to run the complete project

**Prerequisites:** Python 3.13 + [`uv`](https://docs.astral.sh/uv/), Node.js + npm, Docker, an OpenAI API key, and a Langfuse project (cloud or self-hosted).

```bash
# 1. Start the data services
cd backend
docker compose up -d          # Qdrant, Redis, Neo4j

# 2. Configure and start the backend
uv sync
# create backend/.env — see backend/README.md for the full variable table
python -m scripts.download_corpus
python -m scripts.ingest --strategy all
uvicorn main:app --reload     # http://localhost:8000

# 3. Start the frontend (separate terminal)
cd frontend
npm install
# optional: create frontend/.env.local with API_URL if the backend isn't on localhost:8000
npm run dev                   # http://localhost:3000
```

| Service | Port |
|---|---|
| Frontend (Next.js) | 3000 |
| Backend (FastAPI) | 8000 |
| Qdrant (REST / gRPC) | 6333 / 6334 |
| Redis | 6379 |
| Neo4j (browser / bolt) | 7474 / 7687 |

Full details — every environment variable, all API endpoints, the ingestion pipeline, and test commands — are in [`backend/README.md`](./backend/README.md) and [`frontend/README.md`](./frontend/README.md).

## Key findings from experiments

All ten experiments run against a fixed golden query set (baseline: fixed-size chunking, 500 tokens/50 overlap). Full detail — hypotheses, result tables, and reasoning — in [`backend/EXPERIMENTS.md`](./backend/EXPERIMENTS.md).

| # | Experiment | Headline result | Decision |
|---|---|---|---|
| 1 | Chunking strategy | Recursive: best recall (0.850) and MRR (0.617) | Recursive chunking adopted as default |
| 2 | Embedding model | Local BGE trails OpenAI by 10–12.5pp recall at zero cost | Kept OpenAI `text-embedding-3-small` |
| 3 | Hybrid search (dense+BM25) | Recall +9pp, MRR +14% over dense-only, same chunks | Hybrid adopted as default |
| 4 | Cross-encoder reranking | MRR +13% (0.703→0.796), small precision/recall dip | Reranking kept — biggest single quality lever |
| 5 | End-to-end RAGAS eval | Multi-doc questions were refusing instead of synthesizing (relevancy 0.27→0.50 after fix) | Rewrote the grounded-QA prompt to allow cross-chunk synthesis |
| 6 | Semantic caching | 22.3% cost reduction, 22.2% hit rate, ~16x faster on hits | Deployed with a precision-conservative 0.75 threshold |
| 7 | Agentic RAG vs. naive | Context precision +9.3pp, but recall flat and cost +151% | Opt-in, not default — corpus ceiling limits iteration's upside |
| 8 | Multimodal (table/figure) retrieval | Faithfulness +0.138, but precision/recall regressed; found a reranker bug picking the wrong table | Root-caused to a reranking bug and a RAGAS format-sensitivity blind spot, not the retrieval design |
| 9 | GraphRAG (Neo4j) | Found the graph-expansion path was dead code; fixing it made GraphRAG beat naive on relevancy (+0.15) and recall (+0.05) | Positioned as a third retrieval option between naive and agentic |
| 10 | LLM-judge calibration | 30.4% inter-judge disagreement, concentrated in context_precision | RAGAS trusted for ranking pipelines, not for fine-grained score differences |

The standout signal here isn't any single number — it's that two of these experiments (#8 and #9) found real implementation bugs purely by noticing a metric didn't move the way it should have, then tracing it back to root cause instead of accepting the number at face value.

## Tech stack

**Backend:** FastAPI, Python 3.13 (`uv`), Qdrant, Redis, Neo4j, OpenAI (embeddings + GPT-4o-mini/GPT-4o), Langfuse (v4 OTEL tracing), RAGAS, Docling, `sentence-transformers`, `fastembed`, `tiktoken`.
**Frontend:** Next.js 16 (App Router, Turbopack), React 19, TypeScript, Tailwind CSS v4, shadcn/ui (Radix + Lucide), `react-markdown`.

## Project structure

```
DocMind/
├── backend/     # FastAPI RAG service — ingestion, retrieval, generation, caching, tracing, eval
│                # see backend/README.md for the full file tour
└── frontend/    # Next.js chat UI + document uploader
                 # see frontend/README.md for the full file tour
```

## Limitations / known gaps

Documented honestly rather than glossed over, since this is part of what the experiment log is demonstrating:

- **No CI and no LICENSE** — the repo has no `.github/workflows` and no license file yet.
- **Graph RAG is not exposed via the live API** — it's fully implemented and evaluated (Experiment 9) but only reachable through eval scripts.
- **RAGAS has known format-sensitivity blind spots** on structured/KV-formatted content (Experiments 8 and 10) — treat sub-0.1 faithfulness/context_precision differences as noise, not signal, without a manual spot-check.
