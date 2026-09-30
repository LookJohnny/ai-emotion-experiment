// Stores one participant session (JSON) in a PRIVATE Vercel Blob store. No personal data is collected by the page.
import { put } from "@vercel/blob";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  const raw = JSON.stringify(body);
  if (raw.length > 400_000 || !Array.isArray(body.trials)) return res.status(400).json({ error: "bad payload" });
  const pid = String(body.participant ?? "x").replace(/[^0-9a-zA-Z_-]/g, "").slice(0, 8) || "x";
  body.received = new Date().toISOString();
  await put(`responses/p${pid}.json`, JSON.stringify(body), { access: "private", addRandomSuffix: true, contentType: "application/json" });
  return res.status(200).json({ ok: true });
}
