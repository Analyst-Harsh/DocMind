import { type NextRequest, NextResponse } from "next/server";
import axios from "axios";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  const incoming = await request.formData();
  const file = incoming.get("file");
  if (typeof file === "string" || file === null) {
    return NextResponse.json({ detail: "No file provided" }, { status: 400 });
  }

  const outgoing = new FormData();
  outgoing.append("file", file, file.name);

  try {
    const { data } = await axios.post(
      `${BACKEND}/documents/upload`,
      outgoing,
      { timeout: 120_000 },
    );
    return NextResponse.json(data, { status: 201 });
  } catch (err) {
    if (axios.isAxiosError(err)) {
      const status = err.response?.status ?? 502;
      const detail =
        (err.response?.data as { detail?: string } | undefined)?.detail ??
        err.message;
      return NextResponse.json({ detail }, { status });
    }
    return NextResponse.json({ detail: "Proxy error" }, { status: 500 });
  }
}
