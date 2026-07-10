<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# AGENTS.md — frontend

Next.js 16 (App Router) chat UI for DocMind: ask questions grounded in the backend's ingested corpus, see streamed cited answers, upload new documents into the live index. See the [root AGENTS.md](../AGENTS.md) for how this fits into the rest of the repo, and [`README.md`](./README.md) for the human-facing directory tour.

This file is the authoritative architecture/commands/conventions reference for `frontend/` — read it before non-trivial changes. `CLAUDE.md` in this directory just points here.

## Setup

```bash
npm install
# create .env.local with API_URL=<backend URL> only if it's not http://localhost:8000 — never NEXT_PUBLIC_-prefix it
npm run dev          # Turbopack dev server → http://localhost:3000
```

## Testing

```bash
npx tsc --noEmit     # primary correctness gate — run after every change; no test suite exists yet
npm run lint          # eslint
npm run build         # production build
```

**`npx tsc --noEmit` is currently clean** — treat any type error it reports as real and yours to fix. **`npm run lint` currently reports one pre-existing issue** unrelated to typical changes: `components/chat/CitationMarker.tsx` has a `react/display-name` error plus an unused-`_n` warning. Don't treat that as something you broke unless you're actually editing that file, and don't fix it opportunistically as a drive-by unless asked.

**There is no test suite, so `tsc`/`lint` passing is necessary but not sufficient.** Before claiming a UI change works, actually start `npm run dev` and exercise the changed flow in the browser (or use the `run`/`verify` skill) — don't rely on the type checker alone to confirm behavior.

## Architecture

### Request flow

The browser **never** calls FastAPI directly. All traffic is same-origin:

```
Browser → lib/api.ts (queryBackend / streamBackend / listDocuments / uploadDocument / checkHealth)
        → app/api/**/route.ts  (Next.js server, proxies to FastAPI)
        → FastAPI :8000
```

`API_URL` (FastAPI base URL, default `http://localhost:8000`) is a server-only env var read inside `app/api/**/route.ts`. It must never have a `NEXT_PUBLIC_` prefix — it must not appear in the browser JS bundle. The five backend calls the frontend makes: `POST /query` (non-streaming, currently unused by the chat UI but present in `lib/api.ts`), `POST /query/stream` (SSE, what the chat UI actually calls), `GET /documents`, `POST /documents/upload`, `GET /health`.

### State ownership

All chat state lives in `hooks/useChat.ts`. Components receive props only — no component calls `fetch()` or `axios` directly; only `lib/api.ts` does, and only hooks call `lib/api.ts`.

- `useChat` returns `{ messages, isLoading, lastQueryMeta, sendMessage, retryMessage }`.
- `useDocuments` owns the document list + upload state for the sidebar.
- `useBackendHealth` polls `/api/health` every 30s and returns a `ConnectionStatus` string.
- `app/page.tsx` owns `input` and `isSidebarOpen` state; everything else flows from `useChat`.

### Key patterns

**`messagesRef` / stale closure fix** — `useChat` syncs a ref after every render via `useEffect(() => { messagesRef.current = messages; })`. `retryMessage` reads this ref (never the closure variable) so it always sees the current messages array — it needs the array *as of when the user clicked retry*, not as of when `retryMessage` was defined. Do not move this back to a plain render-time assignment — React 19 bans writing refs during render.

**`executeQuery` helper** — shared by both `sendMessage` and `retryMessage` (`hooks/useChat.ts`). `sendMessage` appends two new messages (user bubble + assistant placeholder, both new IDs via `crypto.randomUUID()`); `retryMessage` resets the existing error bubble in-place (same array index/ID, no new messages, `content` cleared back to `""`). Both then call `executeQuery(question, targetAssistantId)`, which opens the SSE stream via `streamBackend()`, appends tokens to the target message as they arrive, and on the terminal `metadata` event builds a `citationMap` (1-indexed `Record<number, Source>` from `meta.sources`) before marking the message `"complete"`.

**Abort on unmount** — `useChat` keeps an `abortRef`; a `useEffect` with `[]` deps returns a cleanup that calls `abortRef.current?.abort()`, cancelling any in-flight SSE request when the component unmounts. `executeQuery` treats an `AbortError` from `streamBackend` as a silent no-op, not a user-facing error.

**SSE parsing** — `lib/api.ts`'s `streamBackend()` is a manual `fetch`-based reader (axios doesn't stream response bodies well): it reads `response.body`'s reader in a loop, buffers partial lines across chunks, and parses `event: `/`data: ` pairs terminated by a blank line into `token` (append to message), `metadata` (terminal — sources/cost/latency/cache_hit/trace_id), and `error` (throws `ApiError`) callbacks.

**Upload's separate axios instance** — `lib/api.ts` has two axios instances: `client` (JSON, default `Content-Type: application/json`) and `uploadClient` (multipart, deliberately **no** default `Content-Type`). Axios's default `transformRequest` JSON-stringifies `FormData` bodies when a JSON content-type is already set, which corrupts the upload — leaving the header unset lets the browser attach the correct `multipart/form-data` boundary itself. Never merge these two clients.

**Error typing** — `lib/types.ts` exports `ApiError extends Error` with an optional `detail` field. Every function in `lib/api.ts` normalizes failures (axios errors, timeouts, aborts, network errors, backend `{detail}` bodies) into `ApiError` via a shared `toApiError()` helper (or inline in `streamBackend`, which can't reuse the axios-specific helper). `useChat` catches `ApiError` and sets `message.status = "error"` with `errorMessage = err.detail ?? err.message`.

### Tailwind v4

There is **no `tailwind.config.ts`**. All configuration is in `app/globals.css` via `@import "tailwindcss"`, `@theme`, and `@plugin`. Font variables (`--font-sans`, `--font-mono`) are set there and consumed via Geist CSS variables injected by `app/layout.tsx`.

### CONFIG

All magic values (timeouts, thresholds, UI copy, example questions, `topK`) live in `lib/config.ts` as the `CONFIG` const. No magic literals in components — always import from `CONFIG`.

### shadcn/ui

Components live in `components/ui/`. The preset is **Nova** (Lucide icons, Geist font, Radix primitives, Zinc base color — see `components.json`). Add new components with `npx shadcn@latest add <component>`. `ScrollArea` from shadcn does not work reliably in flex-grow contexts — use a plain `div` with `overflow-y-auto min-h-0` instead.

### Document uploader flow

`components/documents/DocumentUpload.tsx` (dropzone/file picker) → `hooks/useDocuments.ts`'s `handleUpload(file)` → `lib/api.ts`'s `uploadDocument()` (via `uploadClient`, see above) → `app/api/documents/upload/route.ts` (proxies to FastAPI `POST /documents/upload` with a 120s timeout) → on success, `useDocuments` refetches the list via `listDocuments()` (`GET /documents`).

## Conventions

- The browser never calls FastAPI directly — every backend call goes through `app/api/**/route.ts` same-origin proxies. Never give a backend-URL env var a `NEXT_PUBLIC_` prefix.
- Components receive props and call hooks only — `fetch`/`axios` calls live exclusively in `lib/api.ts`.
- No magic literals in components — timeouts, thresholds, copy all live in `lib/config.ts`'s `CONFIG`.
- Run `npx tsc --noEmit` after every change — it's the only correctness gate until a test suite exists.

## Generated files — don't hand-edit

`package-lock.json` (regenerate via `npm install`), `next-env.d.ts` and `*.tsbuildinfo` (Next.js/TS-managed), `.next/` (build output, gitignored). `components/ui/*` are shadcn-generated — prefer `npx shadcn@latest add <component>` to re-scaffold rather than hand-writing a new primitive from scratch.
