"""Nous Lens local service. Binds to 127.0.0.1 only; conversations never leave this computer.
python -m nous_lens.server --model ~/.nous-lens/models/Qwen3-4B-Instruct-2507 [--port 7869]"""
import argparse, asyncio, json, os, threading, webbrowser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.1.0"


def build(lens):
    app = FastAPI(title="Nous Lens", version=VERSION, docs_url=None, redoc_url=None)
    lock = asyncio.Lock()

    @app.middleware("http")
    async def same_machine_only(request: Request, call_next):
        # Web pages must not use the engine: only the extension (background worker, no Origin or chrome-extension://) and our own page.
        o = request.headers.get("origin")
        if o and not (o.startswith("chrome-extension://") or o.startswith("moz-extension://") or o in ("http://127.0.0.1:7869", "http://localhost:7869")):
            return JSONResponse({"error": "origin not allowed"}, 403)
        r = await call_next(request)
        if o: r.headers["Access-Control-Allow-Origin"] = o
        return r

    @app.get("/health")
    def health():
        return {"ok": True, "version": VERSION, "model": lens.model_name, "device": lens.device, "load_s": lens.load_s}

    @app.post("/read")
    async def read(body: dict):
        reply = str(body.get("reply") or "")[:8000]
        if not reply.strip(): return JSONResponse({"error": "reply required"}, 400)
        async with lock:                                     # one model, one reading at a time
            return await asyncio.to_thread(lens.read, body.get("messages") or [], reply, bool(body.get("fabrication", True)))

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
    print("Nous 知情助手：正在加载本地模型（首次约 1 分钟）…", flush=True)
    from .engine import Lens
    lens = Lens(a.model, a.pack, a.device)
    lens.warmup()                                            # compile kernels for the common lengths now, not on the first real request
    print(f"已就绪：{lens.model_name} · {lens.device} · 加载 {lens.load_s}s · http://127.0.0.1:{a.port}", flush=True)
    if not a.no_browser: threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    uvicorn.run(build(lens), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__": main()
