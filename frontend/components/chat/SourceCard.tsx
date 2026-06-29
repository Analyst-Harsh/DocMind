"use client";

import { useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn, scoreColor } from "@/lib/utils";
import { CONFIG } from "@/lib/config";
import type { Source } from "@/lib/types";

interface SourceCardProps {
  source: Source;
  isHighlighted?: boolean;
}

export function SourceCard({ source, isHighlighted = false }: SourceCardProps) {
  const pct = Math.round(source.score * 100);
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isHighlighted && cardRef.current) {
      cardRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "center",
      });
    }
  }, [isHighlighted]);

  return (
    <div
      ref={cardRef}
      className={cn(
        "min-w-[220px] max-w-[260px] shrink-0 p-3 rounded-lg border flex flex-col gap-1.5 transition-all duration-200",
        isHighlighted
          ? "border-indigo-400 dark:border-indigo-500 ring-2 ring-indigo-200 dark:ring-indigo-900 bg-indigo-50/50 dark:bg-indigo-950/30"
          : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <p
          className="font-medium text-[13px] text-zinc-800 dark:text-zinc-200 leading-snug line-clamp-2 flex-1"
          title={source.doc_title}
        >
          {source.doc_title}
        </p>
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="secondary"
                className={cn(
                  "shrink-0 text-[11px] font-medium cursor-default select-none",
                  scoreColor(source.score),
                )}
              >
                {pct}%
              </Badge>
            </TooltipTrigger>
            <TooltipContent
              side="top"
              className="max-w-[200px] text-center text-[12px]"
            >
              {CONFIG.ui.scoreTooltip}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <p className="text-[12px] text-zinc-500">chunk {source.chunk_index}</p>
      <p className="font-mono text-[11px] text-zinc-400 truncate">
        {source.doc_id}
      </p>
    </div>
  );
}
