"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { SourceCard } from "./SourceCard";
import type { Source } from "@/lib/types";

interface SourcesSectionProps {
  sources: Source[];
  highlightedCitation?: number | null;
  expanded: boolean;
  onToggle: () => void;
}

export function SourcesSection({
  sources,
  highlightedCitation,
  expanded,
  onToggle,
}: SourcesSectionProps) {
  if (sources.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800">
      <button
        onClick={onToggle}
        aria-expanded={expanded}
        className={cn(
          "flex items-center gap-1 text-[13px] text-zinc-500",
          "hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 rounded",
        )}
      >
        {sources.length} {sources.length === 1 ? "source" : "sources"}
        {expanded ? (
          <ChevronUp className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3" />
        )}
      </button>

      <div
        className={cn(
          "overflow-hidden transition-[max-height,opacity] duration-150 ease-out",
          expanded ? "max-h-96 opacity-100 mt-2" : "max-h-0 opacity-0",
        )}
      >
        <div className="flex gap-3 overflow-x-auto pb-2">
          {sources.map((source, i) => (
            <SourceCard
              key={`${source.doc_id}-${source.chunk_index}-${i}`}
              source={source}
              isHighlighted={highlightedCitation === i + 1}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
