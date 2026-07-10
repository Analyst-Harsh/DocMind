import { File, FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { DocumentSummary } from "@/lib/types";

interface DocumentListItemProps {
  document: DocumentSummary;
}

export function DocumentListItem({ document }: DocumentListItemProps) {
  const Icon = document.type === "markdown" ? FileText : File;

  return (
    <div className="flex items-center gap-2 py-1.5 text-[12px]">
      <Icon className="size-3.5 shrink-0 text-zinc-400" />
      <span
        className="flex-1 truncate text-zinc-700 dark:text-zinc-300"
        title={document.title}
      >
        {document.title}
      </span>
      <Badge variant="outline" className="shrink-0 text-[10px]">
        {document.type}
      </Badge>
      <span className="shrink-0 text-zinc-400 tabular-nums">
        {document.chunk_count}
      </span>
    </div>
  );
}
