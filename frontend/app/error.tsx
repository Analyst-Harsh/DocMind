"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex h-screen flex-col items-center justify-center gap-4 bg-zinc-50 dark:bg-zinc-950">
      <p className="text-[15px] font-medium text-zinc-800 dark:text-zinc-200">
        Something went wrong
      </p>
      <p className="font-mono text-[12px] text-zinc-400">{error.message}</p>
      <Button
        variant="outline"
        onClick={reset}
        className="focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        Try again
      </Button>
    </div>
  );
}
