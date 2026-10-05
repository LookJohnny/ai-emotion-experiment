"""Follow the 读心 engine's event stream (127.0.0.1:7869/events) for AI use outside the chat platforms:
the browser extension (source "web:<site>") and the desktop-app reader (source "app:<name>").
Quiet by default: the page already shows its own marker, so only a hostile reply becomes a DM; everything else is counted
for the daily digest. Events carry scores and short excerpts, never stored."""
import json, threading, time, urllib.request
from .brain import Out
from .cards import Card, clip

NOTICE = {"prompt_secret_blocked": "网页里拦下含密钥的消息", "prompt_secret_sent_anyway": "含密钥仍坚持发送",
          "paste_secret_warned": "提醒粘贴了密钥"}
SITE = {"chatgpt.com": "ChatGPT 网页版", "chat.openai.com": "ChatGPT 网页版", "claude.ai": "Claude 网页版", "gemini.google.com": "Gemini",
        "chat.deepseek.com": "DeepSeek", "www.kimi.com": "Kimi", "kimi.com": "Kimi", "kimi.moonshot.cn": "Kimi", "www.doubao.com": "豆包", "127.0.0.1": "Nous 测试页", "localhost": "Nous 测试页"}


def label(source):
    kind, _, name = source.partition(":")
    return SITE.get(name, name) if kind == "web" else name or source


class EngineEvents:
    def __init__(self, brain, adapters, url="http://127.0.0.1:7869", log=print):
        self.brain, self.adapters, self.url, self.log = brain, adapters, url.rstrip("/"), log

    def handle(self, ev):
        src = str(ev.get("source") or "")
        if not src.startswith(("web:", "app:")): return            # chat-platform and CLI reads are handled where they start
        st = self.brain.state
        if ev.get("kind") == "notice":
            st.bump("local", "note:" + NOTICE.get(ev.get("notice"), "其他")); return
        if ev.get("kind") in (None, "reading"): return
        st.bump("local", "ai_read")
        k = {"hostile": "hostility", "syco": "sycophancy", "fabric": "fabrication"}.get(ev.get("kind"))
        if not k: return
        st.bump("local", "flag_" + k)
        if k != "hostility": return                                 # experimental readouts: the page marker + the digest
        for p, uid in self.brain.owners().items():
            ad = self.adapters.get(p)
            if ad and self.brain._dm_allowed(uid):
                ad.send([Out("dm", uid, Card(f"{label(src)} 的回复在贬低你", "bad", ["页面上已经标出来了。这类回复不必放在心上，也可以让它换个说法重答。"],
                                             quote=clip(ev.get("reply", ""), 100), note=label(src)))])

    def run(self):
        while True:
            try:
                with urllib.request.urlopen(self.url + "/events", timeout=60) as r:
                    for raw in r:
                        line = raw.decode("utf-8", "replace").strip()
                        if line.startswith("data: "):
                            try: self.handle(json.loads(line[6:]))
                            except Exception as e: self.log(f"[读心事件] 处理出错：{e!r}")
            except Exception:
                time.sleep(5)                                        # engine restarting or not up yet

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
