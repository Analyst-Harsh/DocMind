import { type NextRequest, NextResponse } from "next/server";
import axios from "axios";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { data } = await axios.post(`${BACKEND}/query`, body, {
      timeout: 35_000,
    });
    return NextResponse.json(data);
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
