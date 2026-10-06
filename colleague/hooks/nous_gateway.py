#!/usr/bin/env python3
"""Nous 读心 gateway for Claude Code (ANTHROPIC_BASE_URL=http://127.0.0.1:7871). Standard library only.

Every request to the model passes through here on its way to api.anthropic.com. Before it leaves this computer, every
key in it is swapped for a placeholder [[NOUS:NAME]] and the key itself goes to the Nous vault (macOS keychain).
The model only ever sees placeholders; the PreToolUse hook swaps the real value back in when a command or file uses one.
So pasting a key into Claude Code works exactly as before, and the key never reaches the model provider.

- Headers (including the person's own Claude login) are passed through untouched and never logged.
- Responses are streamed back byte for byte.
- If the body is not JSON, it is passed through unchanged."""
import http.client, json, os, re, ssl, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nous_vault

UPSTREAM = os.environ.get("NOUS_UPSTREAM", "api.anthropic.com")
PORT = int(os.environ.get("NOUS_GATEWAY_PORT", "7871"))
LOG = os.path.expanduser("~/.nous-colleague/gateway.log")
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "accept-encoding"}


def placeholder(name): return f"[[NOUS:{name}]]"


class Vault:
    """name -> value cache, refreshed when the vault index changes."""
    def __init__(self): self.mtime, self.values, self.lock = None, {}, threading.Lock()

    def current(self):
        try: m = os.path.getmtime(nous_vault.INDEX)
        except OSError: m = None
        with self.lock:
            if m != self.mtime:
                self.values = {n: v for n in nous_vault.index() if (v := nous_vault.get(n))}
                self.mtime = m
            return dict(self.values)


VAULT = Vault()


def scrub(obj, values, found):
    """Replace known key values, and newly seen keys, in every string of the request body."""
    if isinstance(obj, str):
        s = obj
        for name, v in values.items():
            if v and v in s: s = s.replace(v, placeholder(name)); found.add(name)
        for hint, v in nous_vault.find_secrets(s):           # a key the vault has not seen yet
            if v in s:
                name = nous_vault.name_for(hint, v, s); nous_vault.put(name, v); values[name] = v
                s = s.replace(v, placeholder(name)); found.add(name)
        return s
    if isinstance(obj, list): return [scrub(x, values, found) for x in obj]
    if isinstance(obj, dict): return {k: scrub(v, values, found) for k, v in obj.items()}
    return obj


def log(msg):
    try:
        with open(LOG, "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError: pass


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a): pass

    def _proxy(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if self.command in ("POST", "PUT", "PATCH") else None
        if body and "json" in (self.headers.get("Content-Type") or ""):
            try:
                found = set(); data = scrub(json.loads(body), VAULT.current(), found)
                if found:
                    body = json.dumps(data, ensure_ascii=False).encode(); log(f"{self.path} keys replaced: {','.join(sorted(found))}")
            except ValueError:
                pass
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if body is not None: headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPSConnection(UPSTREAM, 443, timeout=600, context=ssl.create_default_context())
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            r = conn.getresponse()
        except Exception as e:
            msg = json.dumps({"type": "error", "error": {"type": "api_error", "message": f"Nous gateway could not reach {UPSTREAM}: {e}"}}).encode()
            self.send_response(502); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(msg)))
            self.end_headers(); self.wfile.write(msg); return
        self.send_response(r.status, r.reason)
        for k, v in r.getheaders():
            if k.lower() not in HOP: self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk: break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _proxy


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    srv.daemon_threads = True
    log(f"listening on 127.0.0.1:{PORT} -> {UPSTREAM}")
    srv.serve_forever()
