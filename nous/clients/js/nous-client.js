// Nous JS client (browser or Node >= 18; ESM, no dependencies).
//   import { gatewayStream, Companion } from "./nous-client.js";
//   for await (const { text, risk } of gatewayStream("https://host", key, messages)) { ... }
//   const c = new Companion("wss://host", deviceKey); await c.open();
//   for await (const ev of c.chat("我今天考了第一名！")) { if (ev.type === "expression") face(ev.expr); if (ev.type === "sentence") speak(ev.text); }

export async function* gatewayStream(baseUrl, apiKey, messages, opts = {}) {
  const { action, concepts, ...rest } = opts;
  const body = { messages, stream: true, ...rest };
  if (action || concepts) body.x_nous = { ...(action && { action }), ...(concepts && { concepts }) };
  const r = await fetch(baseUrl.replace(/\/$/, "") + "/v1/chat/completions", {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: "Bearer " + apiKey }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error("nous gateway " + r.status);
  const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { value, done } = await rd.read(); if (done) break;
    buf += dec.decode(value, { stream: true }); let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
      if (!line.startsWith("data: {")) continue;
      const d = JSON.parse(line.slice(6)), ch = (d.choices || [])[0] || {};
      yield { text: (ch.delta && ch.delta.content) || "", risk: d.x_risk, finish: ch.finish_reason || null };
    }
  }
}

export class Companion {
  constructor(baseWs, deviceKey) { this.uri = baseWs.replace(/\/$/, "") + "/v1/companion/ws?key=" + encodeURIComponent(deviceKey); this.q = []; this.wait = null; }
  open() {
    return new Promise((ok, bad) => {
      this.ws = new WebSocket(this.uri);
      this.ws.onmessage = e => { const m = JSON.parse(e.data); if (m.type === "hello") { this.hello = m; ok(this); return; } this.q.push(m); this.wait && this.wait(); };
      this.ws.onerror = bad; this.ws.onclose = e => { if (e.code === 4401) bad(new Error("invalid device key")); };
    });
  }
  async next() { while (!this.q.length) await new Promise(r => (this.wait = r)); return this.q.shift(); }
  async *chat(text) {
    this.ws.send(JSON.stringify({ type: "chat", text }));
    for (;;) { const ev = await this.next(); yield ev; if (ev.type === "done" || ev.type === "error") return; }
  }
  close() { this.ws.close(); }
}
