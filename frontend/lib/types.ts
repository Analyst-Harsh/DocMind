export interface Source {
  doc_title: string;
  chunk_index: number;
  score: number;
  doc_id: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  cost_usd: number;
  latency_ms: number;
  trace_id: string;
  cache_hit: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  citationMap?: Record<number, Source>;
  costUsd?: number;
  latencyMs?: number;
  cacheHit?: boolean;
  traceId?: string;
  timestamp: Date;
  status: "sending" | "streaming" | "complete" | "error";
  errorMessage?: string;
}

export interface QueryMeta {
  costUsd: number;
  latencyMs: number;
  cacheHit: boolean;
  traceId: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export type ConnectionStatus = "connected" | "disconnected" | "checking";

export interface StreamMetadata {
  sources: Source[];
  cost_usd: number;
  latency_ms: number;
  cache_hit: boolean;
  trace_id: string;
}
