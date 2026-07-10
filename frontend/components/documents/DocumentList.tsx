import { Skeleton } from "@/components/ui/skeleton";
import { CONFIG } from "@/lib/config";
import type { DocumentSummary } from "@/lib/types";
import { DocumentListItem } from "./DocumentListItem";

interface DocumentListProps {
  documents: DocumentSummary[];
  isLoading: boolean;
}

export function DocumentList({ documents, isLoading }: DocumentListProps) {
  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-5 w-full" />
        ))}
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <p className="text-[12px] text-zinc-400">
        {CONFIG.documents.emptyStateText}
      </p>
    );
  }

  return (
    <div className="flex flex-col divide-y divide-zinc-100 dark:divide-zinc-800">
      {documents.map((doc) => (
        <DocumentListItem key={doc.doc_id} document={doc} />
      ))}
    </div>
  );
}
