import { NextResponse } from "next/server";
import axios from "axios";

const BACKEND = process.env.API_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const { data } = await axios.get(`${BACKEND}/health`, { timeout: 5_000 });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ status: "error" }, { status: 502 });
  }
}
