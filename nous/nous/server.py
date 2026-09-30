"""Nous server: one process, one GPU model, two products (gateway + companion) and the web pages."""
import json, os, time
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from . import __version__
from .audit import Audit
from .auth import Auth

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def create_app(cfg, engine=None):
    if engine is None:
        from .engine import Engine
        engine = Engine(cfg.model, cfg.pack, device=cfg.device, visualize=cfg.visualize, min_tokens=cfg.min_tokens, sustain=cfg.sustain)
    audit, auth = Audit(cfg.db), Auth(cfg)
    app = FastAPI(title="Nous", version=__version__, docs_url="/docs", redoc_url=None); app.state.engine = engine
    if cfg.gateway.enabled:
        from . import gateway; app.include_router(gateway.build(engine, cfg, audit, auth))
    if cfg.companion.enabled:
        from . import companion; app.include_router(companion.build(engine, cfg, audit, auth))

    def info_(request: Request):
        auth.admin(request)
        return {**engine.info(), "version": __version__, "gateway": {"enabled": cfg.gateway.enabled, "mode": cfg.gateway.mode, "upstream": cfg.gateway.upstream_model or None,
                "action": cfg.gateway.action, "concepts": cfg.gateway.concepts}, "companion": {"enabled": cfg.companion.enabled},
                "thresholds": engine.pack.thr, "sustain": cfg.sustain, "min_tokens": cfg.min_tokens}
    app.get("/v1/nous/info")(info_); app.get("/info", include_in_schema=False)(info_)

    def audit_(request: Request, n: int = 50):
        auth.admin(request); return audit.recent_requests(min(n, 500))
    app.get("/v1/nous/audit")(audit_); app.get("/audit", include_in_schema=False)(audit_)

    @app.get("/v1/nous/stats")
    def stats(request: Request, hours: float = 24):
        auth.admin(request); return audit.stats(time.time() - hours * 3600)

    @app.get("/circuit", include_in_schema=False)
    def circuit(request: Request):
        auth.admin(request)
        return {"edges": engine.inner.edges if engine.inner else [], "rule": "potential wiring from weights (V-composition, random-baseline corrected)"}

    @app.get("/presets", include_in_schema=False)
    def presets():
        return json.load(open(os.path.join(WEB, "presets.json"), encoding="utf-8"))

    @app.get("/vrm/index.json", include_in_schema=False)
    def avatars():
        names = json.load(open(os.path.join(WEB, "vrm", "names.json"), encoding="utf-8")) if os.path.exists(os.path.join(WEB, "vrm", "names.json")) else {}
        return [{"file": f, "name": names.get(f, f[:-4].capitalize())} for f in sorted(os.listdir(os.path.join(WEB, "vrm"))) if f.endswith(".vrm")]

    @app.get("/healthz", include_in_schema=False)
    def health(): return {"ok": True, "model": engine.model_name, "version": __version__}

    page = lambda f: (lambda: HTMLResponse(open(os.path.join(WEB, f), encoding="utf-8").read()))
    app.get("/console", include_in_schema=False)(page("console.html"))
    app.get("/monitor", include_in_schema=False)(page("console.html"))
    app.get("/parent", include_in_schema=False)(page("parent.html"))
    app.get("/companion", include_in_schema=False)(page("companion.html"))
    app.get("/", include_in_schema=False)(page("home.html"))
    for d in ("vendor", "fonts", "vrm"): app.mount("/" + d, StaticFiles(directory=os.path.join(WEB, d)), name=d)

    @app.get("/static/{name}", include_in_schema=False)
    def static(name: str):
        p = os.path.join(WEB, os.path.basename(name))
        return FileResponse(p) if os.path.isfile(p) else JSONResponse({"error": "not found"}, 404)
    return app
