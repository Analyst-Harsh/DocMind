"use client";

import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { ConnectionStatus } from "./ConnectionStatus";
import { CONFIG } from "@/lib/config";
import { cn } from "@/lib/utils";

interface TopBarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function TopBar({ isOpen, onToggle }: TopBarProps) {
  return (
    <header className="h-12 shrink-0 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 flex items-center px-4 gap-3">
      <button
        onClick={onToggle}
        aria-label={isOpen ? "Close sidebar" : "Open sidebar"}
        className={cn(
          "p-1.5 rounded-md text-zinc-500",
          "hover:text-zinc-700 dark:hover:text-zinc-300",
          "hover:bg-zinc-100 dark:hover:bg-zinc-800",
          "transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1"
        )}
      >
        {isOpen ? (
          <PanelLeftClose className="h-4 w-4" />
        ) : (
          <PanelLeftOpen className="h-4 w-4" />
        )}
      </button>

      <span className="text-[14px] font-semibold tracking-tight text-zinc-800 dark:text-zinc-200">
        {CONFIG.ui.appName}
      </span>

      <div className="ml-auto">
        <ConnectionStatus />
      </div>
    </header>
  );
}
