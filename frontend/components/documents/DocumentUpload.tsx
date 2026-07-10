"use client";

import { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";
import { CONFIG } from "@/lib/config";

interface DocumentUploadProps {
  onUpload: (file: File) => void;
  isUploading: boolean;
  error: string | null;
}

export function DocumentUpload({
  onUpload,
  isUploading,
  error,
}: DocumentUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        disabled={isUploading}
        className={cn(
          "flex flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed px-3 py-4 text-center transition-colors",
          "border-zinc-300 dark:border-zinc-700",
          isDragging &&
            "border-indigo-400 bg-indigo-50/50 dark:bg-indigo-950/30",
          isUploading && "opacity-60 cursor-not-allowed",
        )}
      >
        <UploadCloud className="size-4 text-zinc-400" />
        <span className="text-[11px] text-zinc-500 leading-snug">
          {isUploading
            ? CONFIG.documents.uploadingLabel
            : CONFIG.documents.dropzoneLabel}
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={CONFIG.documents.acceptAttr}
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      {error && (
        <p className="text-[11px] text-red-600 dark:text-red-400">{error}</p>
      )}
    </div>
  );
}
