"""nous serve  --config nous.yaml       start the server (gateway + companion + console)
   nous check  --config nous.yaml       validate config and pack without loading the model
   nous keygen [--n 1]                 print random keys for nous.yaml"""
import argparse, os, secrets, sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nous"); sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("serve", "check"):
        p = sub.add_parser(c); p.add_argument("--config", "-c", default=os.environ.get("NOUS_CONFIG", "nous.yaml"))
    k = sub.add_parser("keygen"); k.add_argument("--n", type=int, default=1); k.add_argument("--prefix", default="nk_")
    a = ap.parse_args(argv)
    if a.cmd == "keygen":
        for _ in range(a.n): print(a.prefix + secrets.token_urlsafe(24))
        return
    from .config import Config
    cfg = Config.load(a.config); problems = check(cfg)
    for p in problems: print(("ERROR " if p[0] else "WARN  ") + p[1], file=sys.stderr)
    if any(p[0] for p in problems): sys.exit(2)
    if a.cmd == "check": print("config ok"); return
    import uvicorn
    from .server import create_app
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


def check(cfg):
    out = []
    if not cfg.model or not os.path.exists(cfg.model): out.append((True, f"model path not found: {cfg.model!r}"))
    if not os.path.exists(os.path.join(cfg.pack, "manifest.json")): out.append((True, f"pack manifest not found in {cfg.pack}"))
    else:
        from .engine.pack import Pack
        try:
            p = Pack(cfg.pack); miss = [c for c in cfg.gateway.concepts if c not in p.concepts]
            if miss: out.append((True, f"gateway concepts not in pack: {miss}"))
        except Exception as e: out.append((True, f"pack failed to load: {e}"))
    if cfg.gateway.mode not in ("local", "reader"): out.append((True, "gateway.mode must be local or reader"))
    if cfg.gateway.mode == "reader" and not cfg.gateway.upstream_base_url: out.append((True, "reader mode needs gateway.upstream_base_url"))
    if cfg.gateway.action not in ("log", "stop"): out.append((True, "gateway.action must be log or stop"))
    if not cfg.admin_keys: out.append((False, "no admin_keys: console / audit are OPEN (development only)"))
    if cfg.gateway.enabled and not cfg.api_keys and not cfg.admin_keys: out.append((False, "no api_keys: gateway is OPEN"))
    if cfg.companion.enabled and not cfg.device_keys: out.append((False, "no device_keys: only admin keys can connect companions"))
    return out


if __name__ == "__main__": main()
