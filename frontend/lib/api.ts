import axios, { type AxiosInstance } from "axios";
import { CONFIG } from "./config";
import {
  ApiError,
  type DocumentSummary,
  type QueryResponse,
  type StreamMetadata,
  type UploadDocumentResponse,
} from "./types";

const client: AxiosInstance = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: CONFIG.api.timeoutMs,
});

client.interceptors.request.use((config) => config);
client.interceptors.response.use(
  (res) => res,
  (err) => Promise.reject(err),
);

// Separate instance with no default Content-Type: axios's default
// transformRequest JSON-stringifies FormData bodies when a
// "application/json" Content-Type is already set, which would corrupt
// a file upload. Leaving it unset lets the browser add the correct
// multipart/form-data boundary itself.
const uploadClient: AxiosInstance = axios.create({
  baseURL: "/api",
  timeout: CONFIG.documents.uploadTimeoutMs,
});

function toApiError(err: unknown): ApiError {
  if (axios.isAxiosError(err)) {
    if (err.code === "ECONNABORTED") {
      return new ApiError(
        "Request timed out",
        `No response after ${CONFIG.api.timeoutMs / 1000}s`,
      );
    }
    if (err.code === "ERR_CANCELED") {
      return new ApiError("Request cancelled");
    }
    const detail = (err.response?.data as { detail?: string } | undefined)
      ?.detail;
    return new ApiError(
      `HTTP ${err.response?.status ?? "error"}`,
      detail ?? err.message,
    );
  }
  return new ApiError("Network error. Could not reach the backend.");
}

export async function queryBackend(
  question: string,
  signal: AbortSignal,
): Promise<QueryResponse> {
  try {
    const { data } = await client.post<QueryResponse>(
      "/query",
      { question, top_k: CONFIG.api.topK },
      { signal },
    );
    return data;
  } catch (err) {
    if (axios.isAxiosError(err)) {
      if (err.code === "ECONNABORTED") {
        throw new ApiError(
          "Request timed out",
          `No response after ${CONFIG.api.timeoutMs / 1000}s`,
        );
      }
      if (err.code === "ERR_CANCELED") {
        throw new ApiError("Request cancelled");
      }
      const detail = (err.response?.data as { detail?: string } | undefined)
        ?.detail;
      throw new ApiError(
        `HTTP ${err.response?.status ?? "error"}`,
        detail ?? err.message,
      );
    }
    throw new ApiError("Network error. Could not reach the backend.");
  }
}

export async function streamBackend(
  question: string,
  signal: AbortSignal,
  onToken: (token: string) => void,
  onMetadata: (meta: StreamMetadata) => void,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: CONFIG.api.topK }),
      signal,
    });
  } catch (err) {
    if ((err as { name?: string }).name === "AbortError") throw err;
    throw new ApiError("Network error. Could not reach the backend.");
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(
      `HTTP ${response.status}`,
      (data as { detail?: string }).detail,
    );
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    let eventType = "";
    let eventData = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        eventData = line.slice(6);
      } else if (line === "") {
        if (eventType === "token" && eventData) {
          onToken(eventData);
        } else if (eventType === "metadata" && eventData) {
          onMetadata(JSON.parse(eventData) as StreamMetadata);
        } else if (eventType === "error" && eventData) {
          const err = JSON.parse(eventData) as { message: string };
          throw new ApiError(err.message);
        }
        eventType = "";
        eventData = "";
      }
    }
  }
}

export async function listDocuments(
  signal?: AbortSignal,
): Promise<DocumentSummary[]> {
  try {
    const { data } = await client.get<{ documents: DocumentSummary[] }>(
      "/documents",
      { signal },
    );
    return data.documents;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function uploadDocument(
  file: File,
  signal?: AbortSignal,
): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const { data } = await uploadClient.post<UploadDocumentResponse>(
      "/documents/upload",
      formData,
      { signal },
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const { data } = await client.get<{ status: string }>("/health", {
      timeout: 5_000,
    });
    return data.status === "ok";
  } catch {
    return false;
  }
}
