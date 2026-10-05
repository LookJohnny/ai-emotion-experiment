"""Feishu (飞书) adapter: long-connection events (no public server needed), interactive cards, card-button callbacks.
Needs a self-built app with a bot, these scopes, and the event im.message.receive_v1 + callback card.action.trigger
on the long connection (see FEISHU_SETUP.md):
  im:message.group_msg:readonly   (people's messages in groups: context + secret warnings)
  im:message.group_bot_msg:readonly   (other bots' messages in groups: the AI replies to read)
  im:message.p2p_msg:readonly  im:message:send_as_bot"""
import json, threading
import lark_oapi as lark
from lark_oapi.api.im.v1 import (CreateMessageRequest, CreateMessageRequestBody, GetChatRequest, ReplyMessageRequest,
                                 ReplyMessageRequestBody)
from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTriggerResponse
from .brain import Message

COLOR = {"bad": "red", "warn": "orange", "ok": "green", "info": "blue"}


def text_of(msg_type, content):
    """Plain text of a Feishu message: text, rich text (post) and interactive cards (what most AI bots send)."""
    try: c = json.loads(content or "{}")
    except Exception: return ""
    if msg_type == "text": return c.get("text", "")
    out = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("tag") in ("text", "lark_md", "plain_text", "markdown", "md", "a") and isinstance(x.get("text") or x.get("content"), str):
                out.append(x.get("text") or x.get("content"))
            for k, v in x.items():
                if k not in ("text", "content") or not isinstance(v, str): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    if msg_type in ("post", "interactive"):
        walk(c.get("title")) if isinstance(c.get("title"), (dict, list)) else (c.get("title") and out.append(c["title"]))
        walk(c)
    return "\n".join(s for s in out if s.strip())


def card_json(card):
    els = [{"tag": "div", "text": {"tag": "lark_md", "content": l}} for l in card.lines]
    if card.quote: els.append({"tag": "note", "elements": [{"tag": "lark_md", "content": "AI 原话：" + card.quote}]})
    if card.buttons:
        els.append({"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": lab},
                                                  "type": "default", "value": val} for lab, val in card.buttons]})
    if card.note: els.append({"tag": "note", "elements": [{"tag": "plain_text", "content": card.note}]})
    return json.dumps({"config": {"wide_screen_mode": True},
                       "header": {"template": COLOR.get(card.level, "blue"), "title": {"tag": "plain_text", "content": card.title}},
                       "elements": els}, ensure_ascii=False)


class Feishu:
    def __init__(self, brain, app_id, app_secret, log=print):
        self.brain, self.log = brain, log
        self.api = lark.Client.builder().app_id(app_id).app_secret(app_secret).log_level(lark.LogLevel.WARNING).build()
        self.me = self._bot_open_id(); self.chat_names = {}
        handler = (lark.EventDispatcherHandler.builder("", "")
                   .register_p2_im_message_receive_v1(self._on_message)
                   .register_p2_card_action_trigger(self._on_card)
                   .build())
        self.ws = lark.ws.Client(app_id, app_secret, event_handler=handler, log_level=lark.LogLevel.WARNING)

    def _bot_open_id(self):
        req = lark.BaseRequest.builder().http_method(lark.HttpMethod.GET).uri("/open-apis/bot/v3/info").token_types({lark.AccessTokenType.TENANT}).build()
        r = self.api.request(req)
        try: return json.loads(r.raw.content)["bot"]["open_id"]
        except Exception: raise RuntimeError(f"飞书：无法获取机器人信息（{getattr(r, 'msg', '')}）。请检查 App ID / Secret，并确认已启用机器人能力。")

    def _chat_name(self, chat_id):
        if chat_id not in self.chat_names:
            r = self.api.im.v1.chat.get(GetChatRequest.builder().chat_id(chat_id).build())
            self.chat_names[chat_id] = r.data.name if r.success() and r.data else ""
        return self.chat_names[chat_id]

    def _on_message(self, data):
        ev = data.event; m, s = ev.message, ev.sender
        sid = (s.sender_id.open_id if s.sender_id else None) or ""
        if sid and sid == self.me: return
        text = text_of(m.message_type, m.content)
        if not text: return
        msg = Message("feishu", m.chat_id, "p2p" if m.chat_type == "p2p" else "group", m.message_id, sid,
                      s.sender_type in ("bot", "app"), text, chat_name=self._chat_name(m.chat_id) if m.chat_type != "p2p" else "",
                      parent_id=m.parent_id or "")
        # do the reading off the event thread: Feishu expects the handler to return quickly
        threading.Thread(target=self._process, args=(msg,), daemon=True).start()

    def _process(self, msg):
        try: self.send(self.brain.handle(msg))
        except Exception as e: self.log(f"[feishu] 处理消息出错：{e!r}")

    def _on_card(self, data):
        ev = data.event; val = (ev.action.value if ev.action else None) or {}
        uid = ev.operator.open_id if ev.operator else ""
        toast = self.brain.resolve_approval("feishu", val, uid) if "ap" in val else self.brain.feedback("feishu", val, uid)
        return P2CardActionTriggerResponse({"toast": {"type": "success", "content": toast}})

    def send(self, outs):
        for o in outs:
            content = card_json(o.card)
            if o.kind == "reply":
                req = ReplyMessageRequest.builder().message_id(o.target).request_body(
                    ReplyMessageRequestBody.builder().msg_type("interactive").content(content).build()).build()
                r = self.api.im.v1.message.reply(req)
            else:
                req = CreateMessageRequest.builder().receive_id_type("open_id").request_body(
                    CreateMessageRequestBody.builder().receive_id(o.target).msg_type("interactive").content(content).build()).build()
                r = self.api.im.v1.message.create(req)
            if not r.success(): self.log(f"[feishu] 发送失败 {r.code}: {r.msg}")

    def run(self):
        self.log(f"[feishu] 已连接（长连接）。机器人 open_id 已获取。")
        self.ws.start()                       # blocks
