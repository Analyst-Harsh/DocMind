# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **This is NOT the Next.js you know.** This version (16.x) has breaking changes — APIs, conventions, and file structure may differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.

## Commands

All commands run from `frontend/`.

- Dev server (Turbopack): `npm run dev` → http://localhost:3000
- Type check: `npx tsc --noEmit`
- Lint: `npm run lint`
- Production build: `npm run build`

There are no tests yet. TypeScript (`npx tsc --noEmit`) is the primary correctness check — run it after every change.

## Architecture

### Request flow

The browser **never** calls FastAPI directly. All traffic is same-origin:

```
Browser → lib/api.ts (/api/query, /api/health)
        → app/api/*/route.ts  (Next.js server, proxies to FastAPI)
        → FastAPI :8000
```

`API_URL` (FastAPI base URL) is a server-only env var in `.env.local`. It must never have a `NEXT_PUBLIC_` prefix — it must not appear in the browser JS bundle.

### State ownership

All chat state lives in `hooks/useChat.ts`. Components receive props only — no component calls `fetch()` or `axios` directly.

- `useChat` returns: `{ messages, isLoading, lastQueryMeta, sendMessage, retryMessage }`
- `useBackendHealth` polls `/api/health` every 30 s and returns a `ConnectionStatus` string
- `app/page.tsx` owns `input` and `isSidebarOpen` state; everything else flows from `useChat`

### Key patterns

**`messagesRef` / stale closure fix** — `useChat` syncs a ref after every render via `useEffect(() => { messagesRef.current = messages; })`. `retryMessage` reads this ref (never the closure variable) so it always sees the current messages array. Do not move this back to a plain render-time assignment — React 19 bans writing refs during render.

**`executeQuery` helper** — shared by both `sendMessage` and `retryMessage`. `sendMessage` appends two new messages (user bubble + assistant placeholder); `retryMessage` resets the existing error bubble in-place (same array index, no new messages). Both then call `executeQuery(question, targetAssistantId)`.

**Abort on unmount** — `abortRef.current?.abort()` in a `useEffect` cleanup with `[]` deps cancels any in-flight request when the component unmounts.

### Tailwind v4

There is **no `tailwind.config.ts`**. All configuration is in `app/globals.css` via `@import "tailwindcss"`, `@theme`, and `@plugin`. Font variables (`--font-sans`, `--font-mono`) are set there and consumed via Geist CSS variables injected by `app/layout.tsx`.

### CONFIG

All magic values (timeouts, thresholds, UI copy, example questions) live in `lib/config.ts` as the `CONFIG` const. No magic literals in components — always import from CONFIG.

### Error types

`lib/types.ts` exports `ApiError extends Error` with an optional `detail` field. `lib/api.ts` always throws `ApiError`; `useChat` catches it and sets `message.status = "error"` with `errorMessage = err.detail ?? err.message`.

### shadcn/ui

Components live in `components/ui/`. The preset is **Nova** (Lucide icons, Geist font, Radix primitives, Zinc base color). Add new components with `npx shadcn@latest add <component>`. `ScrollArea` from shadcn does not work reliably in flex-grow contexts — use a plain `div` with `overflow-y-auto min-h-0` instead.
