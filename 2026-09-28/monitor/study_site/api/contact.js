// Pilot inquiry from nous.center -> PRIVATE Vercel Blob (leads/). Only what the visitor typed; no tracking.
import { put } from "@vercel/blob";

const LIMITS = { name: 40, org: 80, contact: 120, scene: 40, model: 80, note: 1000 };

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  let body = req.body;
  if (typeof body === "string") { try { body = JSON.parse(body || "{}"); } catch { return res.status(400).json({ error: "bad json" }); } }
  body = body || {};
  const lead = {};
  for (const [k, n] of Object.entries(LIMITS)) lead[k] = String(body[k] ?? "").trim().slice(0, n);
  if (!lead.name || !lead.contact) return res.status(400).json({ error: "name and contact required" });
  lead.received = new Date().toISOString();
  await put(`leads/${lead.received.slice(0, 10)}.json`, JSON.stringify(lead), { access: "private", addRandomSuffix: true, contentType: "application/json" });
  return res.status(200).json({ ok: true });
}
