import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { CONFIG } from "./config";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function scoreColor(score: number): string {
  if (score >= CONFIG.score.highThreshold)
    return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
  if (score >= CONFIG.score.midThreshold)
    return "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400";
  return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400";
}

export function formatCost(usd: number): string {
  return `$${usd.toFixed(6)}`;
}

export function formatLatency(ms: number): string {
  return `${ms.toLocaleString()} ms`;
}

export function formatTraceId(id: string): string {
  return id.slice(0, 8);
}

export function formatTimestamp(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
