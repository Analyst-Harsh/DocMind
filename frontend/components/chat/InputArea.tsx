"use client";

import { type KeyboardEvent, useCallback, useRef } from "react";
import { SendHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CONFIG } from "@/lib/config";
import { cn } from "@/lib/utils";

interface InputAreaProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
}

export function InputArea({
  value,
  onChange,
  onSend,
  isLoading,
}: InputAreaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxH = 24 * CONFIG.input.maxRows + 16;
    el.style.height = Math.min(el.scrollHeight, maxH) + "px";
  }, []);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSend();
    }
  };

  const canSend = value.trim().length > 0 && !isLoading;
  const remaining = CONFIG.input.maxChars - value.length;
  const showCounter = value.length > CONFIG.input.charCountWarningThreshold;

  return (
    <div className="shrink-0 border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 px-4 py-3">
      <div className="flex gap-2 items-end">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            adjustHeight();
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents…"
          maxLength={CONFIG.input.maxChars}
          rows={1}
          aria-label="Question input"
          className={cn(
            "flex-1 resize-none rounded-lg",
            "border border-zinc-200 dark:border-zinc-700",
            "bg-zinc-50 dark:bg-zinc-900",
            "text-[15px] text-zinc-900 dark:text-zinc-100",
            "placeholder:text-zinc-400",
            "px-3 py-2 leading-6",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1",
            "transition-shadow duration-150"
          )}
          style={{
            minHeight: "40px",
            maxHeight: `${24 * CONFIG.input.maxRows + 16}px`,
          }}
        />
        <Button
          onClick={onSend}
          disabled={!canSend}
          aria-label="Send question"
          aria-busy={isLoading}
          className="h-10 w-10 p-0 shrink-0 bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-50 focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1"
        >
          <SendHorizontal className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex items-center justify-between mt-1.5">
        <p className="text-[11px] text-zinc-400">{CONFIG.ui.inputDisclaimer}</p>
        {showCounter && (
          <p
            className={cn(
              "text-[11px] shrink-0 ml-4",
              remaining < 100 ? "text-amber-500" : "text-zinc-400"
            )}
          >
            {remaining} remaining
          </p>
        )}
      </div>
    </div>
  );
}
