"""API-key auth. Keys come from `Authorization: Bearer <key>`, `x-api-key`, or `?key=` (for devices / WebSocket clients).
If no keys of a class are configured, that class is open (development only -- `nous serve` warns)."""
import hmac
from fastapi import HTTPException, Request


def extract(req):
    h = req.headers.get("authorization", "")
    if h.lower().startswith("bearer "): return h[7:].strip()
    return req.headers.get("x-api-key") or req.query_params.get("key") or ""


def match(key, keys):
    return any(hmac.compare_digest(key, k) for k in keys)


class Auth:
    def __init__(self, cfg): self.cfg = cfg

    def client(self, req: Request):
        """gateway callers: api keys or admin keys."""
        keys = self.cfg.api_keys + self.cfg.admin_keys
        if not keys: return "open"
        k = extract(req)
        if match(k, keys): return "admin" if match(k, self.cfg.admin_keys) else "key:" + k[-4:]
        raise HTTPException(401, "invalid api key")

    def admin(self, req: Request):
        if not self.cfg.admin_keys: return "open"
        if match(extract(req), self.cfg.admin_keys): return "admin"
        raise HTTPException(401, "admin key required")

    def device(self, key):
        """companion devices -> device id; admin keys may connect as 'console'."""
        if not self.cfg.device_keys and not self.cfg.admin_keys: return "dev-open"
        for k, dev in self.cfg.device_keys.items():
            if hmac.compare_digest(key, k): return dev
        if match(key, self.cfg.admin_keys): return "console"
        return None
