"use client";

import { Loader2, Wifi, WifiOff } from "lucide-react";
import { useBackendHealth } from "@/hooks/useBackendHealth";
import { cn } from "@/lib/utils";

const STATUS_CONFIG = {
  checking: {
    Icon: Loader2,
    label: "Checking…",
    className: "text-zinc-400",
    iconClassName: "animate-spin",
  },
  connected: {
    Icon: Wifi,
    label: "Connected",
    className: "text-green-600 dark:text-green-400",
    iconClassName: "",
  },
  disconnected: {
    Icon: WifiOff,
    label: "Backend unreachable",
    className: "text-red-500 dark:text-red-400",
    iconClassName: "",
  },
} as const;

export function ConnectionStatus() {
  const status = useBackendHealth();
  const { Icon, label, className, iconClassName } = STATUS_CONFIG[status];

  return (
    <div className={cn("flex items-center gap-1.5 text-[12px]", className)}>
      <Icon className={cn("h-3 w-3", iconClassName)} />
      <span>{label}</span>
    </div>
  );
}
