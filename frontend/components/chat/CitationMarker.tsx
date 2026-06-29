"use client";

import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { Source } from "@/lib/types";

interface CitationMarkerProps {
  content: string;
  citationMap: Record<number, Source>;
  onCitationClick: (n: number) => void;
}

const CITATION_RE = /(\[\d+\])/g;

function CitationBadge({
  n,
  source,
  onClick,
}: {
  n: number;
  source: Source | undefined;
  onClick: () => void;
}) {
  const btn = (
    <sup>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
        className="inline-flex items-center justify-center min-w-[1rem] h-4 px-0.5 text-[10px] font-bold rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300 hover:bg-indigo-200 dark:hover:bg-indigo-800 transition-colors cursor-pointer border-0 leading-none"
        aria-label={
          source ? `Citation ${n}: ${source.doc_title}` : `Citation ${n}`
        }
      >
        {n}
      </button>
    </sup>
  );

  if (!source) return btn;

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>{btn}</TooltipTrigger>
        <TooltipContent side="top" className="text-[12px] max-w-[200px]">
          {source.doc_title}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function expandText(
  text: string,
  citationMap: Record<number, Source>,
  onCitationClick: (n: number) => void,
  keyPrefix: string,
): React.ReactNode[] {
  return text.split(CITATION_RE).map((part, i) => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (m) {
      const n = Number(m[1]);
      return (
        <CitationBadge
          key={`${keyPrefix}-${i}`}
          n={n}
          source={citationMap[n]}
          onClick={() => onCitationClick(n)}
        />
      );
    }
    return part;
  });
}

function walkChildren(
  children: React.ReactNode,
  citationMap: Record<number, Source>,
  onCitationClick: (n: number) => void,
  depth: string,
): React.ReactNode {
  return React.Children.map(children, (child, i) => {
    const key = `${depth}-${i}`;
    if (typeof child === "string") {
      const parts = expandText(child, citationMap, onCitationClick, key);
      if (parts.length === 1 && typeof parts[0] === "string") return parts[0];
      return <React.Fragment key={key}>{parts}</React.Fragment>;
    }
    if (React.isValidElement(child)) {
      const props = child.props as { children?: React.ReactNode };
      if (props.children != null) {
        return React.cloneElement(
          child as React.ReactElement<{ children?: React.ReactNode }>,
          {},
          walkChildren(props.children, citationMap, onCitationClick, key),
        );
      }
    }
    return child;
  });
}

export function CitationMarker({
  content,
  citationMap,
  onCitationClick,
}: CitationMarkerProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const components = useMemo<Record<string, React.ComponentType<any>>>(() => {
    const makeRenderer =
      (tag: string) =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ({ children, node: _n, ...rest }: any) =>
        React.createElement(
          tag,
          rest,
          walkChildren(children, citationMap, onCitationClick, tag),
        );

    return {
      p: makeRenderer("p"),
      li: makeRenderer("li"),
      h1: makeRenderer("h1"),
      h2: makeRenderer("h2"),
      h3: makeRenderer("h3"),
    };
  }, [citationMap, onCitationClick]);

  return (
    <div className="prose prose-zinc dark:prose-invert prose-sm max-w-none text-[15px]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
