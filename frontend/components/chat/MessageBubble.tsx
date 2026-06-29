"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { CitationMarker } from "./CitationMarker";
import { SourcesSection } from "./SourcesSection";
import { formatTimestamp } from "@/lib/utils";
import type { Message } from "@/lib/types";

interface MessageBubbleProps {
  message: Message;
  onRetry: (id: string) => void;
}

const HIGHLIGHT_DURATION_MS = 2000;

export function MessageBubble({ message, onRetry }: MessageBubbleProps) {
  const [highlightedCitation, setHighlightedCitation] = useState<
    number | null
  >(null);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    },
    [],
  );

  const handleCitationClick = useCallback((n: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setHighlightedCitation(n);
    setSourcesExpanded(true);
    timerRef.current = setTimeout(
      () => setHighlightedCitation(null),
      HIGHLIGHT_DURATION_MS,
    );
  }, []);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[70%] bg-indigo-600 text-white rounded-lg px-4 py-3 text-[15px] leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.status === "sending") {
    return (
      <div className="flex justify-start">
        <div className="max-w-[80%] w-full bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg px-4 py-3 space-y-3">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-3 w-24 mt-4" />
        </div>
      </div>
    );
  }

  if (message.status === "error") {
    return (
      <div className="flex justify-start">
        <div className="max-w-[80%] bg-white dark:bg-zinc-900 border border-red-200 dark:border-red-900 rounded-lg px-4 py-3 space-y-2">
          <div className="flex items-start gap-2 text-red-600 dark:text-red-400 text-[14px]">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <p>{message.errorMessage}</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onRetry(message.id)}
            className="text-[13px] h-7 focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <RefreshCw className="h-3 w-3 mr-1.5" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  const citationMap = message.citationMap ?? {};

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg px-4 py-3">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[12px] text-zinc-400">
            {formatTimestamp(message.timestamp)}
          </span>
          {message.cacheHit && (
            <Badge
              variant="secondary"
              className="text-[11px] h-4 px-1.5 bg-indigo-50 text-indigo-600 dark:bg-indigo-950 dark:text-indigo-400 border-0"
            >
              ⚡ Cached
            </Badge>
          )}
        </div>

        <CitationMarker
          content={message.content}
          citationMap={citationMap}
          onCitationClick={handleCitationClick}
        />

        {message.sources && message.sources.length > 0 && (
          <SourcesSection
            sources={message.sources}
            highlightedCitation={highlightedCitation}
            expanded={sourcesExpanded}
            onToggle={() => setSourcesExpanded((prev) => !prev)}
          />
        )}
      </div>
    </div>
  );
}
