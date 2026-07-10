# DocMind Frontend

Next.js 16 (App Router) chat UI for DocMind: ask questions grounded in the ingested corpus, see streamed answers with inline citations linked to their source chunks, and upload new documents into the live index.

For the overall project story (architecture, key decisions, experiment findings, how the backend fits in), see the [root README](../README.md). For a deep, code-level architecture reference written for agentic coding tools (and equally useful for humans) — including the state-management patterns and gotchas below, and the load-bearing Next.js-16-breaking-changes warning — see [AGENTS.md](./AGENTS.md) (`CLAUDE.md` just points there).

## Directory structure

```
frontend/
├── app/
│   ├── page.tsx                      # the app — single-page "ChatPage" client component
│   ├── layout.tsx                    # root layout: Geist fonts, metadata
│   ├── globals.css                   # Tailwind v4 config lives here (@theme, @plugin) — no tailwind.config.ts
│   ├── error.tsx / loading.tsx / not-found.tsx
│   └── api/                          # Next.js Route Handlers — same-origin BFF proxy to FastAPI
│       ├── health/route.ts            # GET  → proxies FastAPI /health
│       ├── query/route.ts             # POST → proxies FastAPI /query
│       ├── query/stream/route.ts      # POST → proxies FastAPI /query/stream (SSE passthrough)
│       └── documents/
│           ├── route.ts               # GET  → proxies FastAPI /documents
│           └── upload/route.ts        # POST → proxies FastAPI /documents/upload (multipart, 120s timeout)
│
├── components/
│   ├── chat/
│   │   ├── MessageList.tsx            # scrollable message log, empty-state with example questions
│   │   ├── MessageBubble.tsx          # renders sending / error / complete states
│   │   ├── CitationMarker.tsx         # parses "[1]" markers in markdown into clickable badges
│   │   ├── SourcesSection.tsx         # collapsible list of retrieved chunks under an answer
│   │   ├── SourceCard.tsx             # doc title, relevance score badge, chunk index, doc_id
│   │   └── InputArea.tsx              # auto-growing textarea, char counter, send button
│   ├── documents/                     # the document uploader
│   │   ├── DocumentUpload.tsx          # dropzone / file picker, upload progress + error states
│   │   ├── DocumentList.tsx            # list with skeleton loading + empty state
│   │   └── DocumentListItem.tsx        # icon by type, title, type badge, chunk count
│   ├── layout/
│   │   ├── TopBar.tsx                  # sidebar toggle, app name, connection status
│   │   ├── Sidebar.tsx                 # branding, last-query metadata, documents panel
│   │   └── ConnectionStatus.tsx        # Wifi/WifiOff/Loader2 indicator, backed by useBackendHealth
│   └── ui/                            # shadcn/ui primitives (badge, button, scroll-area, skeleton, ...)
│
├── hooks/
│   ├── useChat.ts                     # all chat state + SSE streaming logic (the core hook)
│   ├── useDocuments.ts                # document list + upload state
│   └── useBackendHealth.ts            # polls /api/health every 30s
│
├── lib/
│   ├── api.ts                         # the only place that calls fetch/axios — all backend calls live here
│   ├── config.ts                      # single CONFIG const — every magic value/copy string
│   ├── types.ts                       # Message, Source, DocumentSummary, ApiError, ...
│   └── utils.ts                       # cn(), scoreColor(), formatCost/Latency/TraceId/Timestamp
│
├── AGENTS.md / CLAUDE.md              # agentic-coding-tool guidance (Next.js 16 breaking-change warning + architecture)
├── components.json                    # shadcn/ui generator config ("Nova" preset)
└── package.json / next.config.ts / tsconfig.json
```

## Architecture

**The browser never calls FastAPI directly.** Every request is same-origin:

```
Browser → lib/api.ts → app/api/**/route.ts (Next.js server) → FastAPI :8000
```

`app/api/**/route.ts` route handlers read a server-only `API_URL` env var (default `http://localhost:8000`) and proxy to the FastAPI backend documented in [`backend/README.md`](../backend/README.md). This keeps the backend URL out of the browser bundle — `API_URL` must never be given a `NEXT_PUBLIC_` prefix.

**State ownership** is plain React hooks, no external state library:
- `useChat` — owns `messages`, `isLoading`, `lastQueryMeta`; exposes `sendMessage`/`retryMessage`, both funneled through a shared `executeQuery` helper that opens an SSE connection via `streamBackend()` in `lib/api.ts`. Uses a `messagesRef` synced every render (via `useEffect`) so `retryMessage` never reads a stale closure, and aborts any in-flight request on unmount.
- `useDocuments` — owns the document list + upload state for the sidebar.
- `useBackendHealth` — polls `/api/health` every 30s for the connection-status indicator.
- Components only ever receive props and call hooks — never `fetch`/`axios` directly; only `lib/api.ts` does that.

**Document upload flow** (`components/documents/DocumentUpload.tsx` → `useDocuments.handleUpload` → `lib/api.ts`'s `uploadDocument()` → `app/api/documents/upload/route.ts` → FastAPI `POST /documents/upload`): the upload uses a *separate* axios instance (`uploadClient`) with no default `Content-Type` header, since axios's JSON transformer would otherwise corrupt the multipart body. On success the document list is refreshed from `GET /documents`.

**Streaming**: `lib/api.ts`'s `streamBackend()` is a manual `fetch`-based SSE reader (axios doesn't stream well) that parses `event:`/`data:` lines for `token`, `metadata`, and `error` events and feeds them into `useChat`'s message state as tokens arrive.

**Styling**: Tailwind CSS v4 — there is no `tailwind.config.ts`; all configuration (`@theme`, `@plugin`) lives in `app/globals.css`. Components use shadcn/ui (Nova preset: Radix primitives, Lucide icons, Geist font, Zinc base color) plus `cn()` (`lib/utils.ts`) for class composition.

## Setup & running

Requires Node.js and the backend running (see [`backend/README.md`](../backend/README.md)).

```bash
cd frontend
npm install
# create .env.local with: API_URL=http://localhost:8000   (only if the backend runs elsewhere than the default)
npm run dev          # Turbopack dev server → http://localhost:3000
```

Other commands:

```bash
npx tsc --noEmit     # type check — the primary correctness gate (no test suite exists yet)
npm run lint          # eslint
npm run build         # production build
npm run start          # serve production build
```

## Further reading

- [`AGENTS.md`](./AGENTS.md) — deeper architecture notes: the `messagesRef` stale-closure fix, the `executeQuery` sharing pattern, the `CONFIG` convention, `ApiError` typing, shadcn/ui gotchas (e.g. `ScrollArea` doesn't work reliably in flex-grow contexts), and the load-bearing warning that this is Next.js 16, which has breaking changes versus most training data. `CLAUDE.md` in this directory just points here.
