"use client";

import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import { CONFIG } from "@/lib/config";
import { cn } from "@/lib/utils";
import type { Message } from "@/lib/types";

interface MessageListProps {
  messages: Message[]; // The list of messages to display in the chat.
  onRetry: (id: string) => void; // Callback function to retry sending a message when an error occurs.
  onQuestionSelect: (q: string) => void; // Callback function to handle the selection of a suggested question from the empty state.
}

export function MessageList({
  messages,
  onRetry,
  onQuestionSelect,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  return (
    <div
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
      className="flex-1 min-h-0 overflow-hidden flex flex-col"
    >
      {messages.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-6 px-8">
          <div className="text-center space-y-1.5">
            <h2 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
              {CONFIG.ui.appName}
            </h2>
            <p className="text-[15px] text-zinc-500">
              {CONFIG.ui.emptyStateTagline}
            </p>
          </div>
          <div className="flex flex-wrap gap-2 justify-center max-w-lg">
            {CONFIG.exampleQuestions.map((q) => (
              <button
                key={q}
                onClick={() => onQuestionSelect(q)}
                className={cn(
                  "text-[13px] px-3 py-1.5 rounded-lg",
                  "border border-zinc-200 dark:border-zinc-700",
                  "text-zinc-600 dark:text-zinc-400",
                  "hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1",
                )}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="flex flex-col gap-4 p-6">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onRetry={onRetry}
              />
            ))}
            <div ref={bottomRef} aria-hidden="true" />
          </div>
        </div>
      )}
    </div>
  );
}
