import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="flex h-screen flex-col items-center justify-center gap-4 bg-zinc-50 dark:bg-zinc-950">
      <p className="text-2xl font-semibold text-zinc-800 dark:text-zinc-200">
        404
      </p>
      <p className="text-[14px] text-zinc-500">This page doesn&apos;t exist.</p>
      <Button
        asChild
        variant="outline"
        className="focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        <Link href="/">Back to DocMind</Link>
      </Button>
    </div>
  );
}
