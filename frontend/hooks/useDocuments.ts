"use client";

import { useCallback, useEffect, useState } from "react";
import { listDocuments, uploadDocument } from "@/lib/api";
import { ApiError, type DocumentSummary } from "@/lib/types";

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [isLoadingList, setIsLoadingList] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    setIsLoadingList(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch {
      // Leave the previous list in place on a transient refresh failure.
    } finally {
      setIsLoadingList(false);
    }
  }, []);

  // Mirrors useBackendHealth's inline-fetch-in-effect pattern (an effect
  // calling a useCallback identifier directly trips
  // react-hooks/set-state-in-effect).
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setIsLoadingList(true);
      try {
        const docs = await listDocuments();
        if (!cancelled) setDocuments(docs);
      } catch {
        // Leave the initial empty list in place on a fetch failure.
      } finally {
        if (!cancelled) setIsLoadingList(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleUpload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setUploadError(null);
      try {
        await uploadDocument(file);
        await refreshDocuments();
      } catch (err) {
        const message =
          err instanceof ApiError
            ? (err.detail ?? err.message)
            : "An unexpected error occurred.";
        setUploadError(message);
      } finally {
        setIsUploading(false);
      }
    },
    [refreshDocuments],
  );

  return {
    documents,
    isLoadingList,
    isUploading,
    uploadError,
    uploadDocument: handleUpload,
  };
}
