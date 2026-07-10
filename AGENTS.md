# AGENTS.md

## Project

DocMind is a Retrieval-Augmented Generation system: `backend/` is a FastAPI service that ingests a fixed document corpus into Qdrant and answers questions grounded in it (hybrid search, reranking, agentic and graph-augmented retrieval variants, semantic caching, Langfuse tracing); `frontend/` is a Next.js 16 chat UI + document uploader that talks to it. Full context: [`README.md`](./README.md) (project overview, architecture, key decisions, experiment findings).

This is a monorepo with two independently-run halves. **More specific instructions win**: `backend/AGENTS.md` governs `backend/`, `frontend/AGENTS.md` governs `frontend/` (each directory's `CLAUDE.md` just points to its `AGENTS.md` — that's where the actual content lives). Read the relevant one before making changes in that directory — they cover per-app conventions and gotchas this file doesn't repeat.

## Setup

```bash
# backend
cd backend && uv sync && docker compose up -d   # Qdrant/Redis/Neo4j; needs .env — see backend/README.md
uvicorn main:app --reload                        # http://localhost:8000

# frontend
cd frontend && npm install
npm run dev                                       # http://localhost:3000
```

## Testing

```bash
cd backend && pytest && ruff check .
cd frontend && npx tsc --noEmit && npm run lint   # no test suite yet — tsc is the correctness gate
```

Run the relevant suite after any change in that half of the repo. Backend changes to retrieval/generation/prompts should also be considered against the eval harness in `backend/app/eval/` and `backend/scripts/eval_*.py` — this project's convention is to measure a retrieval/generation change before adopting it (see `backend/EXPERIMENTS.md`), not to assume it's an improvement.

**Backend `ruff check .` and `mypy app/` are both clean.** Frontend `npx tsc --noEmit` is clean; `npm run lint` has one pre-existing, unrelated issue (see `frontend/AGENTS.md` § Testing for exactly where). Treat a lint/type failure as something your change introduced unless it's that one known frontend exception — don't assume there's a backlog of pre-existing debt to route around.

**Backend `pytest` needs no live services or API keys** — everything is mocked. Don't spin up `docker compose` or populate `.env` just to run the test suite.

## Cost & external calls

Backend ingestion (`scripts/ingest*.py`), evaluation (`scripts/run_*_eval.py`, `scripts/eval_*.py`), and live `/query`/`/agent/query` requests all make real, paid OpenAI API calls (embeddings, chat completions, and GPT-4o Vision for figure captioning) — the agentic path can make several calls per request. Don't re-run these speculatively while debugging something unrelated; see `backend/AGENTS.md` § Cost-incurring operations and § Debugging tools for cheaper alternatives (the `scripts/inspect_*.py` tools, or a local BGE embedding model).

## Conventions that apply repo-wide

- **Secrets never get committed.** `.env` files are gitignored; required keys (`OPENAI_API_KEY`, `LANGFUSE_*`) are documented in `backend/README.md`, not hardcoded anywhere.
- **`backend/corpus/uploads/` is runtime-generated data** (documents added via the upload feature), not source — don't treat it as a fixture to edit by hand.
- **Backend data identity**: `corpus/manifest.yaml` is the join key across the whole ingestion pipeline (`doc_id` ties `Document`/`Chunk`/Qdrant payloads together) and Qdrant collections are named per `(chunking_strategy, embedding_model)` pair — see `backend/AGENTS.md` § Data identity before changing ingestion code.
- **Frontend never calls FastAPI directly from the browser** — all requests go through same-origin `app/api/**/route.ts` proxies; see `frontend/AGENTS.md` before adding a new backend call.
- **Frontend is on Next.js 16**, which has real breaking changes vs. most training data — `frontend/AGENTS.md` has the details; read it before writing Next.js code.
- **Commit messages are short, lowercase, imperative, and unscoped** (`add hybrid search`, `fix test`, `add doc upload api`) — no conventional-commit prefixes (`feat:`/`fix:`) and no `(scope):` markers. Match this style if asked to commit.
- **Lock files are committed but machine-generated** (`uv.lock`, `frontend/package-lock.json`) — regenerate via the tool (`uv add`/`uv sync`, `npm install`), never hand-edit.

## Where to look for more

- `backend/EXPERIMENTS.md` — every retrieval/generation technique in this system, with its measured result and the decision made from it.
- `backend/AGENTS.md` / `frontend/AGENTS.md` — deep per-app architecture references.
- `backend/README.md` / `frontend/README.md` — file-by-file directory tours and API references.
