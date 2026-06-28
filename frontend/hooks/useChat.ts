"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { streamBackend } from "@/lib/api";
import { CONFIG } from "@/lib/config";
import { ApiError, type Message, type QueryMeta } from "@/lib/types";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastQueryMeta, setLastQueryMeta] = useState<QueryMeta | null>(null);

  const messagesRef = useRef<Message[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  // Sync ref after every render so retryMessage never reads a stale snapshot
  useEffect(() => {
    messagesRef.current = messages;
  });

  // Abort in-flight request on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  /**
   * Update a message in the messages state by its ID.
   * This is used to update the assistant's placeholder message with the actual response or error.
   */
  const updateMessage = useCallback(
    (id: string, updates: Partial<Message>) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...updates } : m)),
      ),
    [],
  );

  const executeQuery = useCallback(
    async (question: string, targetId: string) => {
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamBackend(
          question,
          controller.signal,
          (token) =>
            setMessages((prev) =>
              prev.map((m) =>
                m.id === targetId
                  ? { ...m, content: m.content + token, status: "streaming" }
                  : m,
              ),
            ),
          (meta) => {
            updateMessage(targetId, {
              status: "complete",
              sources: meta.sources,
              costUsd: meta.cost_usd,
              latencyMs: meta.latency_ms,
              cacheHit: meta.cache_hit,
              traceId: meta.trace_id,
            });
            setLastQueryMeta({
              costUsd: meta.cost_usd,
              latencyMs: meta.latency_ms,
              cacheHit: meta.cache_hit,
              traceId: meta.trace_id,
            });
          },
        );
      } catch (err) {
        if ((err as { name?: string }).name === "AbortError") return;
        const errorMessage =
          err instanceof ApiError
            ? (err.detail ?? err.message)
            : "An unexpected error occurred.";
        updateMessage(targetId, { status: "error", errorMessage });
      } finally {
        setIsLoading(false);
      }
    },
    [updateMessage],
  );

  const sendMessage = useCallback(
    async (question: string) => {
      if (isLoading) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: question,
        timestamp: new Date(),
        status: "complete",
      };

      const assistantId = crypto.randomUUID();
      const placeholder: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        status: "sending",
      };

      setMessages((prev) => [...prev, userMsg, placeholder]);
      setIsLoading(true);
      await executeQuery(question, assistantId);
    },
    [isLoading, executeQuery],
  );

  const retryMessage = useCallback(
    async (assistantMessageId: string) => {
      const current = messagesRef.current;
      const idx = current.findIndex((m) => m.id === assistantMessageId);
      if (idx < 1) return;

      const userMsg = current[idx - 1];
      if (!userMsg || userMsg.role !== "user") return;

      // Replace error bubble in-place — array length stays constant
      setMessages((prev) =>
        prev.map((m, i) =>
          i === idx
            ? {
                ...m,
                status: "sending" as const,
                errorMessage: undefined,
                content: "",
              }
            : m,
        ),
      );
      setIsLoading(true);
      await executeQuery(userMsg.content, assistantMessageId);
    },
    [executeQuery],
  );

  return { messages, isLoading, lastQueryMeta, sendMessage, retryMessage };
}
