"""ChatGPT desktop app, read through "Nous 读心助手" (Accessibility). The helper posts the window's accessibility tree whenever
it changes; here it is turned into conversation turns, and a reply is read by Nous 读心 once it has stopped changing.
The ChatGPT app marks each turn with a heading ("你说：" / "ChatGPT 说：", or English "You said:" / "ChatGPT said:");
text between two headings belongs to that turn. Replies already on screen when a conversation is first seen are not read.
Nothing is written to disk; the latest snapshot lives in memory only.
The composer is checked for keys too: the app offers no way to hold a send, so Nous shows a macOS notification instead."""
import hashlib, json, re, subprocess, threading, time, urllib.request

USER_H = ("你说：", "You said:")
AI_H = ("ChatGPT 说：", "ChatGPT said:")
SKIP_TEXT = re.compile(r"^(\d{1,2}:\d{2}|ChatGPT 说：|你说：|ChatGPT said:|You said:)$")
SETTLE = 3.0


def turns(snapshot):
    """-> (list of (role, text)), composer text"""
    toks, composer = [], ""

    def walk(n):
        nonlocal composer
        r, t = n.get("r"), n.get("t") or n.get("d") or ""
        if r == "AXHeading" and t in USER_H + AI_H:
            toks.append(("H", "user" if t in USER_H else "assistant")); return
        if r == "AXHeading" and toks and toks[-1][0] != "H" and t in ("最新一条回复", "Latest reply"):
            toks.append(("END", "")); return
        if r == "AXTextArea":
            composer = n.get("v") or ""; toks.append(("END", "")); return
        if r in ("AXButton", "AXPopUpButton"):
            return                                            # action buttons ("复制", "评价回复", "用时 …")
        if r in ("AXStaticText", "AXListMarker"):
            v = (n.get("v") or "").strip("\n")
            if v and not SKIP_TEXT.match(v.strip()): toks.append(("T", v))
        for c in n.get("c", []): walk(c)

    for w in snapshot.get("windows", []): walk(w)
    out, cur = [], None
    for k, v in toks:
        if k == "H": cur = [v, []]; out.append(cur)
        elif k == "END": cur = None
        elif cur is not None: cur[1].append(v)
    res = [(role, re.sub(r"[ \t]*\n[ \t]*", "\n", "".join(p if p.endswith(("\n", " ", "• ")) else p + "\n" for p in parts)).strip())
           for role, parts in out]
    placeholder = ("随心输入", "Ask anything", "有问题，尽管问", "询问任何问题")
    c = composer.strip()
    return [x for x in res if x[1]], ("" if c in placeholder else c)


def notify(title, body):
    """A macOS notification (no permission needed for osascript notifications)."""
    s = lambda x: x.replace("\\", "\\\\").replace('"', '\\"')
    try: subprocess.run(["osascript", "-e", f'display notification "{s(body)}" with title "{s(title)}"'], timeout=5)
    except Exception: pass


class DesktopReader:
    def __init__(self, brain, engine_url="http://127.0.0.1:7869", secrets_in=None, log=print):
        self.brain, self.url, self.log, self.secrets_in = brain, engine_url.rstrip("/"), log, secrets_in
        self.lock = threading.Lock()
        self.seen_convs, self.done = set(), set()
        self.latest = None                                    # (app, context, reply, hash, since)
        self.warned_composer = None
        threading.Thread(target=self._loop, daemon=True).start()

    def status(self):
        return {"last_snapshot": getattr(self, "last_at", None), "turns": getattr(self, "last_turns", None), "conversations": len(self.seen_convs)}

    def ingest(self, app, snapshot):
        ts, composer = turns(snapshot)
        self.last_at, self.last_turns = time.time(), len(ts)
        self._check_composer(app, composer)
        ai = [i for i, (r, _) in enumerate(ts) if r == "assistant"]
        if not ai: return {"turns": len(ts)}
        conv = hashlib.sha1((ts[0][1][:200]).encode()).hexdigest()   # a conversation is known by its first turn
        with self.lock:
            if conv not in self.seen_convs:                    # history already on screen: do not read it
                self.seen_convs.add(conv)
                for i in ai: self.done.add(hashlib.sha1(ts[i][1].encode()).hexdigest())
                return {"turns": len(ts), "history": len(ai)}
            i = ai[-1]; reply = ts[i][1]; h = hashlib.sha1(reply.encode()).hexdigest()
            ctx = next((t for r, t in reversed(ts[:i]) if r == "user"), None)
            if h in self.done: return {"turns": len(ts)}
            if not self.latest or self.latest[3] != h: self.latest = (app, ctx, reply, h, time.time())
        return {"turns": len(ts), "pending": True}

    def _check_composer(self, app, text):
        if not text or not self.secrets_in: self.warned_composer = None; return
        found = self.secrets_in(text)
        key = hashlib.sha1(text.encode()).hexdigest()
        if found and self.warned_composer != key:
            self.warned_composer = key
            self.brain.state.bump("local", "note:桌面版提醒输入框里有密钥")
            notify("Nous 读心", f"{app} 输入框里像是有{found}。发给 AI 后就收不回了，发送前请删掉。")

    def _loop(self):
        while True:
            time.sleep(1)
            with self.lock:
                l = self.latest
                if not l or time.time() - l[4] < SETTLE: continue   # still streaming
                self.latest = None; self.done.add(l[3])
            app, ctx, reply, _, _ = l
            if len(reply) < 8: continue
            msgs = [{"role": "user", "content": ctx}] if ctx else []
            body = json.dumps({"messages": msgs, "reply": reply[:8000], "source": f"app:{app} 桌面版"}).encode()
            try:
                urllib.request.urlopen(urllib.request.Request(self.url + "/read", data=body, headers={"Content-Type": "application/json"}), timeout=60).read()
            except Exception as e:
                self.log(f"[桌面版] 读心引擎不可用：{e!r}")
            # the engine's event stream (engine_events.py) turns the result into the digest / a DM
