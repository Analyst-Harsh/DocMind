import axios, { type AxiosInstance } from "axios";
import { CONFIG } from "./config";
import { ApiError, type QueryResponse } from "./types";

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
