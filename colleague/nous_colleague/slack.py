"""Slack adapter: Socket Mode (no public server needed), Block Kit cards, button callbacks.
Streaming AI apps post a message and then edit it many times (message_changed); the colleague reads a message once it has
been quiet for SETTLE seconds, so it reads the final answer, not a half-written one."""
import os, threading, time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from .brain import Message

SETTLE = 6.0
EMOJI = {"bad": ":red_circle:", "warn": ":large_orange_circle:", "ok": ":large_green_circle:", "info": ":large_blue_circle:"}


def text_of(ev):
    """Text of a Slack message, including bots that put their answer in blocks or attachments."""
    parts = [ev.get("text") or ""]

    def walk(x):
        if isinstance(x, dict):
            if x.get("type") in ("mrkdwn", "plain_text", "text") and isinstance(x.get("text"), str): parts.append(x["text"])
            for v in x.values():
                if isinstance(v, (dict, list)): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(ev.get("blocks") or []); walk(ev.get("attachments") or [])
    seen, out = set(), []
    for p in parts:
        p = p.strip()
        if p and p not in seen: seen.add(p); out.append(p)
    return "\n".join(out)


def blocks(card):
    md = lambda s: s.replace("**", "*")
    b = [{"type": "section", "text": {"type": "mrkdwn", "text": f"{EMOJI.get(card.level, '')} *{card.title}*\n" + "\n".join(md(l) for l in card.lines)}}]
    if card.quote: b.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "AI 原话：" + card.quote}]})
    if card.buttons:
        import json
        b.append({"type": "actions", "elements": [{"type": "button", "text": {"type": "plain_text", "text": lab}, "action_id": f"nous_fb_{i}",
                                                   "value": json.dumps(val)} for i, (lab, val) in enumerate(card.buttons)]})
    if card.note: b.append({"type": "context", "elements": [{"type": "plain_text", "text": card.note}]})
    return b


class Slack:
    def __init__(self, brain, bot_token, app_token, log=print):
        self.brain, self.log, self.app_token = brain, log, app_token
        self.app = App(token=bot_token)
        auth = self.app.client.auth_test(); self.me_user, self.me_bot = auth["user_id"], auth.get("bot_id")
        self.names, self.pending, self.plock = {}, {}, threading.Lock()
        self.app.event("message")(self._on_message)
        self.app.action({"action_id": "nous_fb_0"})(self._on_button); self.app.action({"action_id": "nous_fb_1"})(self._on_button)

    def _chat_name(self, cid):
        if cid not in self.names:
            try: self.names[cid] = "#" + self.app.client.conversations_info(channel=cid)["channel"].get("name", "")
            except Exception: self.names[cid] = ""
        return self.names[cid]

    def _on_message(self, event, ack=None):
        ev = event
        if ev.get("subtype") == "message_changed":
            ev = dict(ev.get("message") or {}, channel=event.get("channel"), channel_type=event.get("channel_type"))
        elif ev.get("subtype") not in (None, "bot_message", "file_share", "thread_broadcast"):
            return
        if ev.get("user") == self.me_user or (self.me_bot and ev.get("bot_id") == self.me_bot): return
        text = text_of(ev)
        if not text: return
        is_bot = bool(ev.get("bot_id")) or ev.get("subtype") == "bot_message"
        msg = Message("slack", ev["channel"], "p2p" if ev.get("channel_type") == "im" else "group", ev["ts"],
                      ev.get("user") or ev.get("bot_id") or "", is_bot, text,
                      chat_name=self._chat_name(ev["channel"]) if ev.get("channel_type") != "im" else "",
                      parent_id=ev.get("thread_ts") if ev.get("thread_ts") != ev["ts"] else "")
        if is_bot and msg.chat_type == "group":
            self._settle(msg)                 # AI answers may still be streaming: wait until the message stops changing
        else:
            threading.Thread(target=self._process, args=(msg,), daemon=True).start()

    def _settle(self, msg):
        key = (msg.chat_id, msg.msg_id)
        with self.plock:
            old = self.pending.get(key)
            if old: old.cancel()
            t = threading.Timer(SETTLE, self._fire, args=(key, msg)); self.pending[key] = t; t.start()

    def _fire(self, key, msg):
        with self.plock: self.pending.pop(key, None)
        self._process(msg)

    def _process(self, msg):
        try: self.send(self.brain.handle(msg))
        except Exception as e: self.log(f"[slack] 处理消息出错：{e!r}")

    def _on_button(self, ack, body, respond):
        import json
        ack()
        a = (body.get("actions") or [{}])[0]; val = json.loads(a.get("value") or "{}"); uid = body.get("user", {}).get("id", "")
        if "ap" in val:                                   # approval for an AI tool on this computer: replace the buttons with the outcome
            msg = self.brain.resolve_approval("slack", val, uid)
            old = (body.get("message") or {}).get("blocks") or []
            keep = [b for b in old if b.get("type") != "actions"]
            respond({"replace_original": True, "text": msg, "blocks": keep + [{"type": "context", "elements": [{"type": "mrkdwn", "text": "*" + msg + "*"}]}]})
            return
        toast = self.brain.feedback("slack", val, uid)
        respond({"response_type": "ephemeral", "replace_original": False, "text": toast})

    def send(self, outs):
        for o in outs:
            for attempt in range(3):                      # the network to Slack can drop a TLS connection; retry
                try:
                    self._send_one(o); break
                except Exception as e:
                    if attempt == 2: self.log(f"[slack] 发送失败：{e!r}")
                    else: time.sleep(1.5 * (attempt + 1))

    def ship(self, user_id):
        """Export this computer's ledger and DM it to user_id; on failure DM what to fix instead."""
        from .ship import ShipError, ship
        from .cards import Card
        try: ship(self.app.client, user_id)
        except ShipError as e:
            self.log(f"[slack] 证据包发送失败：{e}")
            ch = self.app.client.conversations_open(users=user_id)["channel"]["id"]
            c = Card("证据包没能发出", "warn", [str(e)])
            self.app.client.chat_postMessage(channel=ch, blocks=blocks(c), text=c.text()[:300])

    def team_report(self, user_id, days=30):
        """Collect every computer's bundle from the bot's own uploads, check them, DM the report as an HTML file."""
        from . import team
        from .cards import Card
        try:
            self.ship(user_id)                                       # include this computer's latest record
            paths = team.collect_from_slack(self.app.client, self.me_user, self.app.client.token, days)
            if not paths: raise RuntimeError("最近没有收到任何证据包。")
            rep = team.build(paths, days, os.path.join(team.TEAM_HOME, "team.json"))
            out = os.path.join(team.TEAM_HOME, time.strftime("团队风险证明-%Y%m%d-%H%M.html"))
            fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(team.render(rep))
            ch = self.app.client.conversations_open(users=user_id)["channel"]["id"]
            self.app.client.files_upload_v2(channel=ch, file=out, filename=os.path.basename(out), title="团队 AI 风险证明",
                                            initial_comment=team.summary(rep))
        except Exception as e:
            err = getattr(getattr(e, "response", None), "get", lambda *_: None)("error")
            why = ("Slack 应用缺少 files:read 或 files:write 权限：在 api.slack.com/apps → OAuth & Permissions 里添加，然后重新安装到工作区。"
                   if err == "missing_scope" else str(e) if isinstance(e, RuntimeError) else f"出错了：{err or repr(e)[:120]}")
            self.log(f"[slack] 团队报告失败：{why}")
            ch = self.app.client.conversations_open(users=user_id)["channel"]["id"]
            c = Card("团队报告没能生成", "warn", [why])
            self.app.client.chat_postMessage(channel=ch, blocks=blocks(c), text=c.text()[:300])

    def _send_one(self, o):
        if o.kind == "ship": return self.ship(o.target)
        if o.kind == "team": return self.team_report(o.target)
        if o.kind == "reply":
            self.app.client.chat_postMessage(channel=o.chat_id, thread_ts=o.target, blocks=blocks(o.card), text=o.card.text()[:300])
        else:
            ch = self.app.client.conversations_open(users=o.target)["channel"]["id"]
            self.app.client.chat_postMessage(channel=ch, blocks=blocks(o.card), text=o.card.text()[:300])

    def run(self):
        self.log("[slack] 已连接（Socket Mode）。")
        SocketModeHandler(self.app, self.app_token).start()      # blocks
