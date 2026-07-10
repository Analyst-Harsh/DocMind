import { NextResponse } from "next/server";
import axios from "axios";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const { data } = await axios.get(`${BACKEND}/documents`, {
      timeout: 10_000,
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
