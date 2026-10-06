"""Nous Lens local service. Binds to 127.0.0.1 only; conversations never leave this computer.
python -m nous_lens.server --model ~/.nous-lens/models/Qwen3-4B-Instruct-2507 [--port 7869]"""
import argparse, asyncio, collections, json, os, threading, time, webbrowser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.4.0"


def build(lens):
    app = FastAPI(title="Nous Lens", version=VERSION, docs_url=None, redoc_url=None)
    lock = asyncio.Lock()
    recent = collections.deque(maxlen=50); subs = set()          # in memory only: nothing is written to disk

    def summary(res, body, reply):
        h, s, f, g = res["hostility"], res.get("sycophancy"), res.get("fabrication"), res.get("danger")
        kind = "hostile" if h["flag"] else "danger" if g and g["flag"] else "syco" if s and s["flag"] else "fabric" if f and f["flag"] else "ok"
        msgs = body.get("messages") or []; u = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        return {"ts": time.time(), "source": str(body.get("source") or "")[:40], "kind": kind, "hostility": h["score"],
                "sycophancy": s and s["score"], "fabrication": f and f["score"], "danger": g and g["score"], "reply": reply[:160], "user": u[:120],
                "hot": [reply[x["start"]:x["end"]].strip()[:80] for x in h["sentences"] if x.get("hot")][:2]}

    @app.middleware("http")
    async def same_machine_only(request: Request, call_next):
        # Web pages must not use the engine: only the extension (background worker, no Origin or chrome-extension://) and our own page.
        o = request.headers.get("origin")
        if o and not (o.startswith("chrome-extension://") or o.startswith("moz-extension://") or o in ("http://127.0.0.1:7869", "http://localhost:7869", "tauri://localhost", "http://tauri.localhost")):
            return JSONResponse({"error": "origin not allowed"}, 403)
        r = await call_next(request)
        if o: r.headers["Access-Control-Allow-Origin"] = o
        return r

    @app.get("/health")
    def health():
        return {"ok": True, "version": VERSION, "model": lens.model_name, "device": lens.device, "load_s": lens.load_s, "auto_adapted": lens.auto_adapted,
                "readouts": [k for k, p in (("hostility", lens.p_host), ("distress", lens.p_dist), ("fabrication", lens.p_fab), ("sycophancy", lens.p_syco), ("danger", lens.p_danger)) if p]}

    @app.post("/read")
    async def read(body: dict):
        reply = str(body.get("reply") or "")[:8000]
        if not reply.strip(): return JSONResponse({"error": "reply required"}, 400)
        for q in list(subs): q.put_nowait({"kind": "reading", "source": str(body.get("source") or "")[:40]})
        async with lock:                                     # one model, one reading at a time
            res = await asyncio.to_thread(lens.read, body.get("messages") or [], reply, bool(body.get("fabrication", True)))
        ev = summary(res, body, reply); ev["id"] = f"{time.time_ns():x}"; recent.append((ev, {"messages": body.get("messages") or [], "reply": reply, "res": res}))
        for q in list(subs): q.put_nowait(ev)
        return res

    @app.post("/notice")
    async def notice(body: dict):
        # something a front end handled by itself (e.g. the extension stopped a prompt with a key from being sent).
        # Only a kind and a source travel: never the text.
        kind = str(body.get("kind") or "")[:40]
        if kind not in ("prompt_secret_blocked", "prompt_secret_sent_anyway", "paste_secret_warned"):
            return JSONResponse({"error": "unknown kind"}, 400)
        ev = {"ts": time.time(), "kind": "notice", "notice": kind, "source": str(body.get("source") or "")[:40], "id": f"{time.time_ns():x}"}
        for q in list(subs): q.put_nowait(ev)
        return {"ok": True}

    @app.get("/events/recent")
    def events_recent():
        return [e for e, _ in recent][::-1]

    @app.get("/events/item/{eid}")
    def events_item(eid: str):
        for e, full in recent:
            if e["id"] == eid: return {**full, "event": e}
        return JSONResponse({"error": "not found"}, 404)

    @app.get("/events")
    async def events(request: Request):
        q = asyncio.Queue(); subs.add(q)
        async def gen():
            try:
                yield "retry: 3000\n\n"
                while not await request.is_disconnected():
                    try: ev = await asyncio.wait_for(q.get(), 15)
                    except asyncio.TimeoutError: yield ": keepalive\n\n"; continue
                    yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
            finally: subs.discard(q)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/samples")
    def samples():
        return json.load(open(os.path.join(HERE, "web", "samples.json"), encoding="utf-8"))

    @app.get("/chat")
    def chat():                                              # simulated AI chat site, for demonstrating the browser extension
        return HTMLResponse(open(os.path.join(HERE, "web", "chat.html"), encoding="utf-8").read())

    @app.get("/")
    def panel():
        return HTMLResponse(open(os.path.join(HERE, "web", "panel.html"), encoding="utf-8").read())

    @app.get("/fonts/{name}")
    def fonts(name: str):
        from fastapi.responses import FileResponse
        p = os.path.join(HERE, "web", "fonts", os.path.basename(name))
        return FileResponse(p) if os.path.isfile(p) else JSONResponse({}, 404)
    return app


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.expanduser("~/.nous-lens/models/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--pack", default=os.path.join(HERE, "pack")); ap.add_argument("--port", type=int, default=7869)
    ap.add_argument("--device", default=None); ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print("Nous 读心：正在加载本地模型（首次约 1 分钟）…", flush=True)
    from .engine import Lens
    lens = Lens(a.model, a.pack, a.device)
    lens.warmup()                                            # compile kernels for the common lengths now, not on the first real request
    print(f"已就绪：{lens.model_name} · {lens.device} · 加载 {lens.load_s}s · http://127.0.0.1:{a.port}", flush=True)
    if not a.no_browser: threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    uvicorn.run(build(lens), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__": main()
