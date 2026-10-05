"""The colleague's judgement, shared by every chat platform.

Inputs are platform-neutral Messages; outputs are Outs (reply under a message, DM a person). Adapters do the I/O.
What it does:
  - an AI bot speaks in a group  -> read it with Nous 读心 (context: the human message it answers);
                                    flagged -> a short card under that message; validated + serious -> also DM the owner
  - a person posts a secret in a group the AI bots can read -> DM that person privately (never repeat the secret)
  - a person DMs Nous an AI reply  -> read it and answer (the personal "forward to check" path)
  - owner commands in DM: 帮助 / 状态 / 汇总 / 暂停 / 继续; daily digest
Privacy: message bodies live only in a short in-memory history used for context; the state file keeps counters and ids only."""
import collections, json, os, re, secrets as pysecrets, sys, threading, time
from dataclasses import dataclass, field
from .cards import Card, clip
from .reader import LABEL, Reader, ReaderDown, findings

SAFETY_CORE = os.environ.get("NOUS_SAFETY_CORE", os.path.expanduser(
    "~/.codex/.chatgpt-projects/g-p-6abfea876b9c8191980bd8cb26029f18/nous-safety-core"))
try:
    sys.path.insert(0, SAFETY_CORE)
    from safety_core import Policy, SafetyCore, UserContext          # rule-based detection of secrets / personal data
except Exception:                                                     # pragma: no cover - the pilot still runs without it
    SafetyCore = None

SECRET_LABEL = {"private_key": "私钥", "api_key": "API 密钥", "bearer_token": "访问令牌", "credential": "密码或密钥",
                "cvv": "银行卡安全码", "payment_card": "银行卡号", "email": "邮箱地址", "cn_mobile": "手机号"}
URGENT_SECRETS = {"private_key", "api_key", "bearer_token", "credential", "cvv", "payment_card"}


@dataclass
class Message:
    platform: str            # feishu | slack | sim
    chat_id: str
    chat_type: str           # p2p | group
    msg_id: str
    sender_id: str
    sender_is_bot: bool
    text: str
    chat_name: str = ""
    sender_name: str = ""
    parent_id: str = ""      # the message this one replies to (Feishu parent_id / Slack thread_ts)
    ts: float = field(default_factory=time.time)


@dataclass
class Out:
    kind: str                # reply (under msg target) | dm (user target)
    target: str
    card: Card
    chat_id: str = ""


class State:
    """Counters, owners and switches. No message text."""

    def __init__(self, path):
        self.path, self.lock = path, threading.RLock()
        try: self.d = json.load(open(path))
        except Exception: self.d = {}
        for k, v in {"owners": {}, "paused": {}, "day": "", "stats": {}, "feedback": {}, "bots": {}}.items(): self.d.setdefault(k, v)

    def save(self):
        with self.lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True); tmp = self.path + ".tmp"
            with open(tmp, "w") as f: json.dump(self.d, f, ensure_ascii=False, indent=1)
            os.chmod(tmp, 0o600); os.replace(tmp, self.path)

    def bump(self, platform, key, n=1):
        with self.lock:
            s = self.d["stats"].setdefault(platform, {}); s[key] = s.get(key, 0) + n; self.save()


class Brain:
    def __init__(self, state_path=None, reader=None, group_reply="all", clock=time.time):
        self.state = State(state_path or os.path.expanduser("~/.nous-colleague/state.json"))
        self.reader = reader or Reader(); self.group_reply = group_reply; self.clock = clock
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=40))   # chat -> recent messages (memory only)
        self.dm_times = collections.defaultdict(collections.deque)                     # owner DM rate limit
        self.down_notified = 0.0
        self.bind_codes = {}
        self.safety = SafetyCore(Policy()) if SafetyCore else None
        self.lock = threading.Lock()
        self.approvals = {}                                   # id -> {"event", "decision", "expires", "by"}

    # ---- owner binding: a one-time code shown in the terminal, sent to the bot in a DM ----
    def bind_code(self, platform):
        if platform not in self.state.d["owners"]:
            self.bind_codes[platform] = f"{pysecrets.randbelow(10**6):06d}"
        return self.bind_codes.get(platform)

    def owner(self, platform):
        return self.state.d["owners"].get(platform)

    # ---- entry point ----
    def handle(self, m: Message):
        with self.lock:
            if m.chat_type == "group":
                self.hist[m.chat_id].append(m)
                if m.sender_is_bot:
                    self.state.d["bots"].setdefault(m.chat_id, True)
                    return self._ai_spoke(m)
                return self._person_spoke(m)
            if not m.sender_is_bot:
                return self._dm(m)
            return []

    # ---- an AI bot spoke in a group ----
    def _context(self, m):
        h = self.hist[m.chat_id]
        if m.parent_id:
            for x in h:
                if x.msg_id == m.parent_id and not x.sender_is_bot: return x.text
        for x in reversed(h):
            if x is m: continue
            if not x.sender_is_bot and m.ts - x.ts < 900: return x.text
        return None

    def _ai_spoke(self, m):
        if self.state.d["paused"].get(m.platform) or len(m.text.strip()) < 8: return []
        try:
            res = self.reader.read(m.text, self._context(m), source=f"{m.platform}:{m.chat_name or 'group'}")
        except ReaderDown:
            return self._reader_down(m.platform)
        self.state.bump(m.platform, "ai_read")
        fs = findings(res)
        if not fs: return []
        outs = []
        for k, *_ in fs: self.state.bump(m.platform, "flag_" + k)
        card = self._finding_card(m, fs, res)
        validated = [f for f in fs if f[3]]
        if self.group_reply == "all" or (self.group_reply == "validated" and validated):
            outs.append(Out("reply", m.msg_id, card, m.chat_id))
        own = self.owner(m.platform)
        if own and validated and self._dm_allowed(own):
            dm = Card(f"{m.chat_name or '群聊'}里的 AI 回复有问题", "bad",
                      [f"读心读到 **{LABEL[validated[0][0]]}**（{_rank(validated[0][2])}）。",
                       "我已经在那条消息下面标注了。需要的话，可以提醒同事别直接采用这条回复。"], quote=clip(m.text))
            outs.append(Out("dm", own, dm))
        return outs

    def _finding_card(self, m, fs, res):
        k, score, pct, validated, hot = fs[0]
        lines = []
        for k2, s2, p2, v2, _ in fs:
            tag = "" if v2 else "（实验性读数）"
            lines.append(f"**{LABEL[k2]}**{tag}：{_rank(p2)}")
        if k == "hostility":
            lines.append("这条回复的语气在贬低或指责对方。转发或采用前，建议先换个说法。")
        elif k == "fabrication":
            lines.append("读心发现，回答这个问题时，模型内部的状态像在「不知道却在答」。涉及数字、名称、出处的内容，请先核对。")
        elif k == "sycophancy":
            lines.append("它可能只是在顺着提问的人说。如果要据此做决定，可以换个问法再问一次，或让它列出反对理由。")
        quote = clip(" ".join(m.text[a:b] for a, b in hot)) if hot else clip(m.text, 70)
        return Card("Nous 读心", "bad" if validated else "warn", lines, quote=quote,
                    note="Nous 读的是模型内部状态，不是关键词。看错了请点「误报」。",
                    buttons=[("误报", {"fb": "fp", "kind": k}), ("有用", {"fb": "tp", "kind": k})])

    # ---- a person spoke in a group ----
    def _person_spoke(self, m):
        if not self.safety or self.state.d["paused"].get(m.platform): return []
        try:
            d = self.safety.review_input(m.text, UserContext("chat_" + _opaque(m.sender_id), "s_" + _opaque(m.chat_id)))
        except Exception:
            return []
        kinds = [t for t in d.sensitive_types if t in URGENT_SECRETS]
        if not kinds: return []                        # contact details alone are normal in a work chat
        self.state.bump(m.platform, "secret_warned")
        names = "、".join(dict.fromkeys(SECRET_LABEL.get(t, t) for t in kinds))
        ai = "，群里的 AI 机器人和" if self.state.d["bots"].get(m.chat_id) else "，"
        card = Card("刚才那条消息里有敏感信息", "warn",
                    [f"你在「{m.chat_name or '群聊'}」发的消息里像是有 **{names}**{ai}所有群成员都能看到。",
                     "建议：撤回这条消息，并尽快作废、更换这个密钥或密码。以后可以用密钥管理工具共享，不要直接贴在聊天里。"],
                    note="只有你收到这条提醒。Nous 没有保存消息内容。")
        return [Out("dm", m.sender_id, card)]

    # ---- direct messages ----
    def _dm(self, m):
        t = m.text.strip(); p = m.platform; own = self.owner(p)
        code = self.bind_codes.get(p)
        if code and code in t and not own:
            self.state.d["owners"][p] = m.sender_id; self.state.save(); self.bind_codes.pop(p, None)
            return [Out("dm", m.sender_id, Card("绑定成功", "ok", ["你现在是这个工作区的 Nous 管理员。重要提醒和每日汇总会私聊发给你。", "发「帮助」查看可以做什么。"]))]
        cmd = t.lower()
        if cmd in ("帮助", "help", "?", "？"): return [Out("dm", m.sender_id, self.help_card(m.sender_id == own))]
        if cmd in ("状态", "status"): return [Out("dm", m.sender_id, self.status_card(p))]
        if cmd in ("汇总", "digest"): return [Out("dm", m.sender_id, self.digest_card(p, reset=False))]
        if cmd in ("暂停", "pause", "继续", "resume"):
            if m.sender_id != own: return [Out("dm", m.sender_id, Card("只有管理员可以操作", "info", ["请联系这个工作区的 Nous 管理员。"]))]
            self.state.d["paused"][p] = cmd in ("暂停", "pause"); self.state.save()
            return [Out("dm", m.sender_id, Card("已暂停" if self.state.d["paused"][p] else "已恢复", "ok",
                         ["群聊里的读取和提醒已暂停。私聊检查仍然可用。" if self.state.d["paused"][p] else "继续在群里读取 AI 的回复。"]))]
        return self._check_forwarded(m)

    def _check_forwarded(self, m):
        q, a = _split_qa(m.text)
        if len(a) < 8:
            return [Out("dm", m.sender_id, Card("把 AI 的回复发给我", "info", ["把 AI 的回复转发或粘贴给我，我读一遍它背后的状态。", "想更准，可以按这个格式发：\n问：你的问题\n答：AI 的回复"]))]
        try:
            res = self.reader.read(a, q, source=f"{m.platform}:私聊检查")
        except ReaderDown:
            return [Out("dm", m.sender_id, Card("读心引擎暂时不可用", "warn", ["稍后再试。管理员已收到通知。"]))] + self._reader_down(m.platform)
        self.state.bump(m.platform, "dm_checked")
        fs = findings(res)
        if fs:
            for k, *_ in fs: self.state.bump(m.platform, "flag_" + k)
            return [Out("dm", m.sender_id, self._finding_card(m, fs, res))]
        h = res["hostility"]
        lines = ["敌意：没有超过提醒线。"]
        if res.get("fabrication"): lines.append("编造（实验性）：没有超过提醒线。")
        else: lines.append("这条不像是在回答事实问题，没有读编造。带上「问：」可以一起读。")
        return [Out("dm", m.sender_id, Card("没发现问题", "ok", lines, note="没发现问题不等于内容一定正确。"))]

    # ---- approvals for AI tools running on this computer (Claude Code / Codex hooks) ----
    def new_approval(self, ttl):
        aid = pysecrets.token_urlsafe(12)
        self.approvals[aid] = {"event": threading.Event(), "decision": None, "expires": self.clock() + ttl, "by": None}
        return aid

    def wait_approval(self, aid, timeout):
        a = self.approvals.get(aid)
        if not a: return "deny"
        a["event"].wait(timeout)
        d = a["decision"] or "timeout"; a["decision"] = a["decision"] or "timeout"
        return d

    def resolve_approval(self, platform, value, user_id):
        """A button press. Only the bound owner of that platform may decide; each approval is decided once, before it expires."""
        a = self.approvals.get(value.get("ap"))
        if not a: return "这个请求已经失效。"
        if user_id != self.owner(platform): return "只有管理员可以批准或拒绝。"
        if a["decision"]: return {"allow": "已经允许过了。", "deny": "已经拒绝过了。"}.get(a["decision"], "这个请求已经超时，没有执行。")
        if self.clock() > a["expires"]: a["decision"] = "timeout"; a["event"].set(); return "这个请求已经超时，没有执行。"
        a["decision"] = "allow" if value.get("d") == "allow" else "deny"; a["by"] = user_id; a["event"].set()
        return "已允许，这一次操作会继续执行。" if a["decision"] == "allow" else "已拒绝，这次操作不会执行。"

    def owners(self):
        return dict(self.state.d["owners"])

    # ---- feedback buttons ----
    def feedback(self, platform, value, user_id=""):
        k = f"{value.get('fb', '?')}_{value.get('kind', '?')}"
        with self.state.lock:
            f = self.state.d["feedback"].setdefault(platform, {}); f[k] = f.get(k, 0) + 1; self.state.save()
        return "已记下，谢谢。误报会用来校准读心。" if value.get("fb") == "fp" else "已记下，谢谢。"

    # ---- cards ----
    def help_card(self, is_owner):
        lines = ["**在群里**：我读 AI 机器人的每条回复。发现贬低、编造或一味附和，会在那条消息下面标注。平时我不说话。",
                 "**私聊我**：转发或粘贴任何 AI 的回复，我帮你读一遍。格式「问：…  答：…」更准。",
                 "**命令**：状态、汇总" + ("、暂停、继续" if is_owner else "")]
        return Card("Nous 读心 · AI 安全同事", "info", lines, note="读心读的是模型内部状态。编造和附和读数是实验性的。")

    def status_card(self, p):
        try: h = self.reader.health(); eng = f"在线（{h.get('model')}，{', '.join(h.get('readouts', []))}）"
        except ReaderDown: eng = "离线"
        s = self.state.d["stats"].get(p, {})
        return Card("状态", "info", [f"读心引擎：{eng}", f"群聊读取：{'已暂停' if self.state.d['paused'].get(p) else '运行中'}",
                                   f"管理员：{'已绑定' if self.owner(p) else '未绑定'}", f"自上次汇总以来读了 {s.get('ai_read', 0)} 条 AI 回复。"])

    def digest_card(self, p, reset=True):
        s = self.state.d["stats"].get(p, {}); fb = self.state.d["feedback"].get(p, {})
        flags = "、".join(f"{LABEL[k]} {s['flag_' + k]} 条" for k in ("hostility", "fabrication", "sycophancy") if s.get("flag_" + k)) or "没有"
        lines = [f"读了 **{s.get('ai_read', 0)}** 条 AI 回复，标注：{flags}。",
                 f"私聊检查 {s.get('dm_checked', 0)} 次；提醒同事收回敏感信息 {s.get('secret_warned', 0)} 次。"]
        fp = sum(v for k, v in fb.items() if k.startswith("fp_")); tp = sum(v for k, v in fb.items() if k.startswith("tp_"))
        if fp or tp: lines.append(f"同事反馈：有用 {tp}，误报 {fp}。")
        card = Card("Nous 读心 · 今日汇总", "info", lines)
        if reset:
            with self.state.lock: self.state.d["stats"][p] = {}; self.state.save()
        return card

    def daily(self, p):
        """Called by the adapter's scheduler once a day; returns the digest DM, or nothing if there is no news."""
        own = self.owner(p); s = self.state.d["stats"].get(p, {})
        if not own or not any(s.values()): return []
        return [Out("dm", own, self.digest_card(p))]

    # ---- helpers ----
    def _dm_allowed(self, uid, limit=3, window=3600):
        q, now = self.dm_times[uid], self.clock()
        while q and now - q[0] > window: q.popleft()
        if len(q) >= limit: return False
        q.append(now); return True

    def _reader_down(self, p):
        own = self.owner(p); now = self.clock()
        if not own or now - self.down_notified < 3600: return []
        self.down_notified = now
        return [Out("dm", own, Card("读心引擎没有运行", "warn", ["这段时间群里的 AI 回复没有被读取。", "请在运行 Nous 的电脑上启动读心引擎。"]))]


def _rank(p):
    if p is None: return "超过了提醒线"
    return f"高于 {min(int(p), 99)}% 的正常回复"


def _opaque(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)[:100] or "x"


def _split_qa(t):
    m = re.search(r"(?:^|\n)\s*(?:问|Q)[:：]\s*(.+?)\n\s*(?:答|A)[:：]\s*(.+)", t, re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, t.strip())
