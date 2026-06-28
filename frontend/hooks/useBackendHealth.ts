"use client";

import { useEffect, useState } from "react";
import { checkHealth } from "@/lib/api";
import { CONFIG } from "@/lib/config";
import type { ConnectionStatus } from "@/lib/types";

export function useBackendHealth(): ConnectionStatus {
  const [status, setStatus] = useState<ConnectionStatus>("checking");

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      const ok = await checkHealth();
      if (!cancelled) setStatus(ok ? "connected" : "disconnected");
    };

    poll();
    const id = setInterval(poll, CONFIG.api.healthPollIntervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return status;
}
