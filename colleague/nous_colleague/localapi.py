"""Local endpoint for AI tools on this computer (Claude Code / Codex hooks, the browser extension).
127.0.0.1 only, and every request must carry the token in ~/.nous-colleague/local.token, so web pages and other users
cannot use it. The hooks do their own blocking checks locally; this endpoint adds what needs the chat platforms:
  POST /notify   {source, title, level, lines, quote?}            -> DM the owner(s)
  POST /approve  {source, title, lines, quote?, timeout}          -> DM the owner(s) with Allow / Deny, wait -> {"decision"}
  POST /approve_async {digest, source, title, lines}             -> never waits: pending | granted (once) | denied
  POST /read     {source, reply, context?}                        -> Nous 读心; flagged -> DM the owner(s) -> {"findings"}"""
import json, os, secrets, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .cards import Card, clip
from .reader import LABEL, ReaderDown, findings

TOKEN_PATH = os.path.expanduser("~/.nous-colleague/local.token")
PORT = 7870


def local_token():
    if not os.path.exists(TOKEN_PATH):
        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f: f.write(secrets.token_urlsafe(32))
    return open(TOKEN_PATH).read().strip()


class LocalAPI:
    def __init__(self, brain, adapters, port=PORT, log=print):
        self.brain, self.adapters, self.port, self.log = brain, adapters, port, log
        self.token = local_token()
        self.pending, self.lock = {}, threading.Lock()          # digest -> approval id (async approvals)

    def _to_owners(self, card):
        sent = 0
        for p, uid in self.brain.owners().items():
            ad = self.adapters.get(p)
            if not ad: continue
            from .brain import Out
            ad.send([Out("dm", uid, card)]); sent += 1
        return sent

    def notify(self, b):
        card = Card(b.get("title", "Nous 读心"), b.get("level", "info"), list(b.get("lines") or []), quote=clip(b.get("quote", ""), 120),
                    note=b.get("source", ""))
        return {"sent": self._to_owners(card)}

    def approve(self, b):
        timeout = max(10, min(int(b.get("timeout", 120)), 300))
        aid = self.brain.new_approval(timeout)
        card = Card(b.get("title", "需要你确认"), "warn", list(b.get("lines") or []) + [f"{timeout // 60 or 1} 分钟内不回应会自动拒绝。"],
                    quote=clip(b.get("quote", ""), 160), note=b.get("source", ""),
                    buttons=[("允许这一次", {"ap": aid, "d": "allow"}), ("拒绝", {"ap": aid, "d": "deny"})])
        if not self._to_owners(card): return {"decision": "unavailable"}
        return {"decision": self.brain.wait_approval(aid, timeout)}

    def approve_async(self, b):
        """For agents that give up on a slow tool call (Codex): never wait. The first call sends the card and returns pending;
        once the owner allows it, the next call with the same digest (same project + tool + exact input) is granted once."""
        dg = str(b.get("digest", ""))[:128]
        if not dg: return {"status": "denied"}
        with self.lock:
            aid = self.pending.get(dg)
            a = self.brain.approvals.get(aid) if aid else None
            if a and a["decision"] == "allow":
                self.pending.pop(dg, None); return {"status": "granted"}
            if a and a["decision"] in ("deny", "timeout"):
                self.pending.pop(dg, None); return {"status": "denied"}
            if a: return {"status": "pending"}
            ttl = 600; aid = self.brain.new_approval(ttl); self.pending[dg] = aid
        card = Card(b.get("title", "需要你确认"), "warn", list(b.get("lines") or []) + ["允许后，它再试一次同样的操作就能执行（10 分钟内有效）。"],
                    quote=clip(b.get("quote", ""), 160), note=b.get("source", ""),
                    buttons=[("允许这一次", {"ap": aid, "d": "allow"}), ("拒绝", {"ap": aid, "d": "deny"})])
        if not self._to_owners(card):
            with self.lock: self.pending.pop(dg, None)
            return {"status": "unavailable"}
        return {"status": "pending"}

    def note(self, b):
        """Something handled quietly (a masked read, a routine risky command): only counted for the daily digest."""
        self.brain.state.bump("local", "note:" + str(b.get("kind", "其他"))[:40]); return {"ok": True}

    def read(self, b):
        try: res = self.brain.reader.read(str(b.get("reply", "")), b.get("context"), source=b.get("source", "本机"))
        except ReaderDown: return {"error": "reader down"}
        fs = findings(res); self.brain.state.bump("local", "ai_read")
        if fs:
            for k, *_ in fs: self.brain.state.bump("local", "flag_" + k)
            k, _, pct, validated, _ = fs[0]
            self.notify({"source": b.get("source", ""), "title": f"{b.get('source') or '本机 AI'} 的回复：{LABEL[k]}", "level": "bad" if validated else "warn",
                         "lines": [f"**{LABEL[k2]}**{'' if v2 else '（实验性读数）'}" for k2, _, _, v2, _ in fs] +
                                  (["它可能只是在顺着你说。要据此做决定，可以让它列出反对理由。"] if k == "sycophancy" else
                                   ["涉及事实、数字、出处的内容，请先核对。"] if k == "fabrication" else []),
                         "quote": b.get("reply", "")})
        return {"findings": [k for k, *_ in fs]}

    def serve(self):
        api = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                if self.headers.get("Authorization") != "Bearer " + api.token or self.headers.get("Origin"):
                    self.send_response(403); self.end_headers(); return
                try: body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except Exception: body = None
                fn = {"/notify": api.notify, "/approve": api.approve, "/approve_async": api.approve_async, "/read": api.read, "/note": api.note}.get(self.path)
                if not fn or not isinstance(body, dict): self.send_response(404); self.end_headers(); return
                try: out = fn(body); code = 200
                except Exception as e: out, code = {"error": repr(e)[:200]}, 500
                data = json.dumps(out, ensure_ascii=False).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)

        srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.log(f"[本机] AI 工具接口已就绪：127.0.0.1:{self.port}")
        return srv
