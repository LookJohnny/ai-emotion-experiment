"""Companion engine (hardware SDK): expression pre-sync + child safety + care, over one WebSocket per device.

  ws  /v1/companion/ws?key=<device key>
  in : {"type":"chat","text":"..."} | {"type":"reset"} | {"type":"ping"}
       {"type":"seed","messages":[...]}   (admin/console connections only: preload history for demos and QA)
  out: {"type":"expression","phase":"pre"|"token","expr":{"happy","sad","angry","neutral"}}   before the first word / while speaking
       {"type":"sentence","text":"...","expr":{...}}      text released for TTS, one sentence at a time
       {"type":"care","score":p}                          the child's message reads as distressed (C2) -> soften, parent notified
       {"type":"safety","action":"rewrite"|"fallback"}    a confirmed hostility alarm stopped the reply before it was released
       {"type":"done","reply":"..."}  {"type":"error","message":"..."}
Safety design: text is held until `hold_tokens` (= min_tokens + sustain) so a confirmed alarm can fire before anything is spoken;
if the pre-reply warning (C1_pre) is on, the whole reply is held until it completes. A confirmed alarm discards the held text,
regenerates once with a safety hint, and falls back to a fixed reply if that is flagged too."""
import asyncio, re, time
import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

WORD = re.compile(r"[\w\u4e00-\u9fff]")


def split_ready(buf, force=False):
    """-> (sentences ready to speak, remainder). A sentence ends at 。！？!?；…/newline; long clauses (>=28 chars) may end at a comma."""
    out, start = [], 0
    for m in re.finditer(r"[。！？!?；;…\n]+[”’\"」』）)】]*|[，,]", buf):
        seg = buf[start:m.end()]
        if m.group()[0] in "，," and len(seg.strip()) < 28: continue
        if seg.strip(): out.append(seg.strip())
        start = m.end()
    rest = buf[start:]
    if force and rest.strip(): out.append(rest.strip()); rest = ""
    merged = []
    for s in out:                                   # punctuation-only fragments ("”", "）") belong to the previous sentence
        if merged and not WORD.search(s): merged[-1] += s
        else: merged.append(s)
    return merged, rest


class Companion:
    def __init__(self, engine, cfg, audit):
        self.E, self.C, self.A, self.cfg = engine, cfg.companion, audit, cfg
        self.hist = {}                                                                  # device -> [messages]

    async def notify(self, device, kind, **detail):
        if not self.C.parent_webhook: return
        try:
            async with httpx.AsyncClient(timeout=5) as hc:
                await hc.post(self.C.parent_webhook, json={"device": device, "kind": kind, "ts": time.time(), **detail})
        except Exception: pass

    async def turn(self, device, text, send):
        C = self.C; h = self.hist.setdefault(device, []); h.append({"role": "user", "content": text})
        base = h[-2 * C.history_turns:]; spoken = []; last_expr = None

        async def expr(phase, e, force=False):
            nonlocal last_expr
            if e is None: return
            if force or last_expr is None or max(abs(e[k] - last_expr.get(k, 0)) for k in e) >= C.expression_step:
                last_expr = e; await send({"type": "expression", "phase": phase, "expr": e})

        for attempt in range(2):
            system = C.persona + ("\n" + C.rewrite_hint if attempt else "")
            buf, blocked, hold_all, pending = "", False, False, []
            async for ev in self.E.stream([{"role": "system", "content": system}] + base, max_tokens=C.max_tokens, temperature=C.temperature,
                                          concepts=["C1", "C2"], emotion=self.E.emotion is not None, stop_on=("C1",)):
                if ev["phase"] == "pre":
                    await expr("pre", (ev.get("emo") or {}).get("expr"), force=True)
                    hold_all = ev["alarm"].get("C1_pre", False)
                    if attempt == 0 and ev["alarm"].get("C2"):
                        await send({"type": "care", "score": ev["C2"]}); self.A.event(device, "care", ev["C2"], text=text)
                        asyncio.create_task(self.notify(device, "care", score=ev["C2"], text=text))
                elif ev["phase"] == "token":
                    await expr("token", (ev.get("emo") or {}).get("expr"))
                    if ev["alarm"].get("C1"): blocked = True; break
                    buf += ev["delta"]
                    if not hold_all and ev["i"] >= C.hold_tokens:
                        ready, buf = split_ready(buf)
                        for s in pending + ready: await send({"type": "sentence", "text": s, "expr": last_expr}); spoken.append(s)
                        pending = []
                    else:
                        ready, buf = split_ready(buf); pending += ready
            if not blocked:
                ready, _ = split_ready(buf, force=True)
                for s in pending + ready: await send({"type": "sentence", "text": s, "expr": last_expr}); spoken.append(s)
                reply = "".join(spoken); h.append({"role": "assistant", "content": reply})
                if attempt: self.A.event(device, "rewrite_ok", None, reply=reply[:500])
                await send({"type": "done", "reply": reply}); return
            held = "".join(pending) + buf
            self.A.event(device, "block", None, attempt=attempt, held_text=held[:500], already_spoken="".join(spoken)[:500], user=text)
            asyncio.create_task(self.notify(device, "block", attempt=attempt, user=text))
            await send({"type": "safety", "action": "rewrite" if attempt == 0 else "fallback"})
        spoken.append(C.fallback_reply); await send({"type": "sentence", "text": C.fallback_reply, "expr": {"happy": 0.2, "sad": 0, "angry": 0, "neutral": 0.8}})
        reply = "".join(spoken); h.append({"role": "assistant", "content": reply}); self.A.event(device, "fallback", None)
        await send({"type": "done", "reply": reply})


def build(engine, cfg, audit, auth):
    r = APIRouter(); comp = Companion(engine, cfg, audit)

    @r.websocket("/v1/companion/ws")
    async def ws(sock: WebSocket):
        key = sock.query_params.get("key") or (sock.headers.get("authorization", "")[7:] if sock.headers.get("authorization", "").lower().startswith("bearer ") else "")
        device = auth.device(key)
        if device is None: await sock.close(code=4401); return
        seedable = device in ("console", "dev-open")
        device = sock.query_params.get("device", "demo") if seedable else device
        await sock.accept(); await sock.send_json({"type": "hello", "device": device, "emotion": engine.emotion is not None})
        try:
            while True:
                m = await sock.receive_json(); t = m.get("type")
                if t == "chat" and (m.get("text") or "").strip():
                    try: await comp.turn(device, m["text"].strip()[:1000], sock.send_json)
                    except WebSocketDisconnect: raise
                    except Exception as e: await sock.send_json({"type": "error", "message": str(e)[:200]})
                elif t == "reset": comp.hist.pop(device, None); await sock.send_json({"type": "reset_ok"})
                elif t == "seed" and seedable:        # demo / QA only: preload a conversation history (not allowed for device keys)
                    comp.hist[device] = [{"role": x["role"], "content": str(x["content"])[:2000]} for x in m.get("messages", []) if x.get("role") in ("user", "assistant")]
                    await sock.send_json({"type": "seeded", "turns": len(comp.hist[device])})
                elif t == "ping": await sock.send_json({"type": "pong"})
        except WebSocketDisconnect: pass

    @r.get("/v1/companion/events")
    def events(request: Request, device: str = None, kind: str = None, n: int = 100):
        auth.admin(request); return audit.recent_events(device, kind, min(n, 500))

    return r
