import { cn, formatCost, formatLatency, formatTraceId } from "@/lib/utils";
import { CONFIG } from "@/lib/config";
import { Separator } from "@/components/ui/separator";
import type { QueryMeta } from "@/lib/types";

interface SidebarProps {
  lastQueryMeta: QueryMeta | null;
  isOpen: boolean;
}

export function Sidebar({ lastQueryMeta, isOpen }: SidebarProps) {
  return (
    <aside
      className={cn(
        "flex flex-col shrink-0 h-full",
        "border-r border-zinc-200 dark:border-zinc-800",
        "bg-white dark:bg-zinc-950",
        "transition-[width] duration-200 ease-in-out overflow-hidden",
        isOpen ? "w-60 xl:w-72" : "w-0 border-r-0"
      )}
    >
      {/* min-w prevents content from reflowing during the width animation */}
      <div className="flex flex-col gap-4 px-4 py-6 min-w-[240px]">
        <div>
          <h2 className="font-semibold text-[18px] tracking-tight text-zinc-900 dark:text-zinc-50 whitespace-nowrap">
            {CONFIG.ui.appName}
          </h2>
          <p className="text-[12px] text-zinc-500 mt-0.5 whitespace-nowrap">
            {CONFIG.ui.sidebarTagline}
          </p>
        </div>

        {lastQueryMeta && (
          <>
            <Separator />
            <div>
              <p className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider mb-2 whitespace-nowrap">
                Last Query
              </p>
              <dl className="flex flex-col gap-1.5">
                <MetaRow label="Cost" value={formatCost(lastQueryMeta.costUsd)} mono />
                <MetaRow label="Latency" value={formatLatency(lastQueryMeta.latencyMs)} />
                <MetaRow
                  label="Cache"
                  value={lastQueryMeta.cacheHit ? "Hit" : "Miss"}
                  valueClassName={
                    lastQueryMeta.cacheHit
                      ? "text-indigo-600 dark:text-indigo-400"
                      : "text-zinc-400"
                  }
                />
                <MetaRow label="Trace" value={formatTraceId(lastQueryMeta.traceId)} mono />
              </dl>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

function MetaRow({
  label,
  value,
  mono = false,
  valueClassName,
}: {
  label: string;
  value: string;
  mono?: boolean;
  valueClassName?: string;
}) {
  return (
    <div className="flex items-center justify-between text-[12px] gap-2">
      <dt className="text-zinc-500 whitespace-nowrap">{label}</dt>
      <dd
        className={cn(
          "text-zinc-700 dark:text-zinc-300 whitespace-nowrap",
          mono && "font-mono",
          valueClassName
        )}
      >
        {value}
      </dd>
    </div>
  );
}
