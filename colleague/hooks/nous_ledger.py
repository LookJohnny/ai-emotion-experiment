"""Nous 证据账本: what the AI tools on this computer tried, what Nous saw in their replies, and what Nous did about it.
The answer to an insurer's or regulator's first question after an incident: "what did you know, and what did you do?"

- Append-only JSONL, one record per event, each carrying the hash of the one before it (a hash chain): editing,
  deleting or reordering any record breaks every hash after it. `verify()` checks the chain.
- No content is stored. A command or reply is kept only as an HMAC under a key that stays on this computer, so the
  ledger can be shared without revealing what was said, and the owner can later prove a given text is the one recorded.
- The chain alone does not stop someone with full control of this computer rewriting it from the start. `anchor()`
  gives the current head; sending it somewhere the owner does not control (the daily Slack/Feishu digest, the insurer)
  makes everything before it fixed from that moment.
Standard library only: the hooks import it too, and several processes append at once (serialised with flock).
  python3 nous_ledger.py verify | head | stamp [TSA_URL] | stamps | profile [days] | report OUT.html [days] | evidence SEQ [TEXTFILE]"""
import fcntl, glob, hashlib, hmac, json, os, re, secrets, shutil, subprocess, sys, time, urllib.request

HOME = os.environ.get("NOUS_LEDGER_HOME", os.path.expanduser("~/.nous-colleague/ledger"))
PATH, KEYPATH = os.path.join(HOME, "ledger.jsonl"), os.path.join(HOME, "ledger.key")
GENESIS = "0" * 64
VERSION = 1

# what an underwriter asks about, in the words of the AI-agent insurance market (see e.g. mount.insure's categories)
CATEGORY = {
    "data_exposure": "数据与密钥泄露",
    "unauthorized_action": "越权或不可撤销的操作",
    "risky_command": "高风险命令（放行并记录）",
    "prompt_injection": "提示注入（外部内容里夹带指令）",
    "hallucination": "编造（错误信息）",
    "harmful_advice": "危险建议",
    "hostility": "敌意 / 贬低用户",
    "sycophancy": "附和（顺着用户说）",
}
# mitigated: Nous stopped it or made it safe. residual: it happened, Nous recorded it (and told someone).
OUTCOME = {
    "blocked": "已拦下", "masked": "已打码后放行", "vaulted": "已收进保险箱", "denied": "管理员拒绝", "timeout": "无人确认，自动拒绝",
    "held": "等待管理员批准，暂不执行", "approved": "管理员批准后执行", "asked_in_tool": "连不上管理员，交由使用者确认", "noted": "放行并记录",
    "flagged": "已标注并提醒", "clean": "未见异常",
}
MITIGATED = {"blocked", "masked", "vaulted", "held", "denied", "timeout"}
READOUT_VALIDATED = {"hostility": True, "hallucination": False, "sycophancy": False, "harmful_advice": False}


def _key():
    os.makedirs(HOME, mode=0o700, exist_ok=True)
    try: return open(KEYPATH, "rb").read()
    except FileNotFoundError:
        try:
            fd = os.open(KEYPATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f: f.write(secrets.token_bytes(32))
        except FileExistsError: pass
        return open(KEYPATH, "rb").read()


def digest(text):
    return hmac.new(_key(), (text or "").encode("utf-8"), hashlib.sha256).hexdigest()


def _hash(rec):
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _last(f):
    """Last record, reading only the tail. Bytes, not text: the tail may start inside a multi-byte character."""
    f.seek(0, 2); end = f.tell()
    if not end: return None
    n = 4096
    while True:
        f.seek(max(0, end - n)); tail = f.read()
        lines = [l for l in tail.splitlines() if l.strip()]
        if len(lines) >= 2 or n >= end: return json.loads(lines[-1].decode("utf-8"))
        n *= 4


def append(category, outcome, source, title="", text=None, scores=None, kind="guard"):
    """Record one event. Never raises: Nous must not break the session it protects."""
    try:
        os.makedirs(HOME, mode=0o700, exist_ok=True)
        fd = os.open(PATH, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "rb+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            last = _last(f)
            rec = {"v": VERSION, "seq": (last["seq"] + 1) if last else 0, "ts": round(time.time(), 3), "kind": kind,
                   "category": category, "outcome": outcome, "source": str(source)[:80], "title": str(title)[:80],
                   "digest": digest(text) if text else None,
                   "scores": {k: round(float(v), 4) for k, v in (scores or {}).items()} or None,
                   "prev": last["hash"] if last else GENESIS}
            rec["hash"] = _hash(rec)
            f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")); f.flush(); os.fsync(f.fileno())
            return rec
    except Exception:
        return None


def records():
    try: return [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()]
    except FileNotFoundError: return []


def verify(recs=None):
    """-> (ok, n, problem). Checks sequence numbers, each record's own hash, and the link to the record before."""
    recs = records() if recs is None else recs; prev = GENESIS
    for i, r in enumerate(recs):
        if r.get("seq") != i: return False, i, f"第 {i} 条的序号是 {r.get('seq')}（记录被删除或插入）"
        if r.get("prev") != prev: return False, i, f"第 {i} 条与前一条接不上（前面的记录被改动）"
        if _hash(r) != r.get("hash"): return False, i, f"第 {i} 条的内容与它的哈希不符（这一条被改动）"
        prev = r["hash"]
    return True, len(recs), ""


def anchor():
    recs = records()
    return {"seq": recs[-1]["seq"], "hash": recs[-1]["hash"], "ts": recs[-1]["ts"]} if recs else {"seq": -1, "hash": GENESIS, "ts": None}


def profile(days=30, recs=None, now=None):
    """Risk profile over the last `days`: per category how often it happened and how much of it Nous stopped,
    readout rates per AI reply read, and which protections were seen working. Every number is a plain count or ratio."""
    recs = records() if recs is None else recs; now = now or time.time()
    w = [r for r in recs if r["ts"] >= now - days * 86400]
    reads = [r for r in w if r["kind"] == "reading"]
    cats = {}
    for r in w:
        if r["kind"] == "reading": continue                       # replies: counted under readouts (flagged, not stopped)
        c = cats.setdefault(r["category"], {"events": 0, "mitigated": 0, "outcomes": {}})
        c["events"] += 1; c["mitigated"] += r["outcome"] in MITIGATED; c["outcomes"][r["outcome"]] = c["outcomes"].get(r["outcome"], 0) + 1
    readouts = {}
    for k in READOUT_VALIDATED:
        n = sum(1 for r in reads if r["category"] == k and r["outcome"] == "flagged")
        if n or k in ("hostility", "hallucination", "sycophancy"):
            readouts[k] = {"flagged": n, "per_1000_replies": round(1000 * n / len(reads), 1) if reads else None, "validated": READOUT_VALIDATED[k]}
    asks = [r for r in w if r["category"] == "unauthorized_action"]
    sources = sorted({r["source"].split(" · ")[0] for r in w})
    controls = {"发送前拦截密钥": any(r["category"] == "data_exposure" and r["outcome"] in ("blocked", "vaulted") for r in w),
                "读取密钥文件自动打码": any(r["outcome"] == "masked" for r in w),
                "不可撤销操作需管理员批准": any(r["outcome"] in ("approved", "denied", "timeout") for r in w),
                "逐条读心（模型内部状态）": bool(reads)}
    return {"window_days": days, "from": min((r["ts"] for r in w), default=None), "to": max((r["ts"] for r in w), default=None),
            "records": len(w), "events": sum(c["events"] for c in cats.values()), "ai_replies_read": len(reads), "sources": sources, "categories": cats, "readouts": readouts,
            "oversight": {"asked": len(asks), "approved": sum(r["outcome"] == "approved" for r in asks),
                          "refused": sum(r["outcome"] in ("denied", "timeout", "held", "blocked") for r in asks),
                          "no_admin": sum(r["outcome"] == "asked_in_tool" for r in asks)},
            "controls_seen": controls}


# ---- third-party time stamps (RFC 3161): what makes the chain hold even against someone in full control of this computer ----
TSA = os.environ.get("NOUS_TSA", "http://timestamp.digicert.com")
ANCHORS = os.path.join(HOME, "anchors")


def _openssl():
    for c in ("/opt/homebrew/bin/openssl", "/usr/local/bin/openssl", shutil.which("openssl")):   # macOS's LibreSSL has no `ts`
        if c and os.path.exists(c):
            r = subprocess.run([c, "ts", "-help"], capture_output=True)
            if b"Usage: ts" in r.stdout + r.stderr: return c
    raise RuntimeError("需要支持 ts 的 OpenSSL（brew install openssl）")


def _cafile():
    for c in (os.environ.get("NOUS_TSA_CAFILE"), "/opt/homebrew/etc/openssl@3/cert.pem", "/usr/local/etc/openssl@3/cert.pem", "/etc/ssl/cert.pem"):
        if c and os.path.exists(c): return c
    raise RuntimeError("找不到根证书文件（设置 NOUS_TSA_CAFILE）")


def stamp(tsa=None):
    """Send the ledger head's hash (nothing else) to a time-stamping authority and keep its signed reply.
    Afterwards nobody, including whoever controls this computer, can change records up to the head without it showing."""
    h = anchor()
    if h["seq"] < 0: return None
    tsa = tsa or TSA; ossl = _openssl(); os.makedirs(ANCHORS, mode=0o700, exist_ok=True)
    base = os.path.join(ANCHORS, f"{h['seq']:08d}")
    q = subprocess.run([ossl, "ts", "-query", "-digest", h["hash"], "-sha256", "-cert"], capture_output=True, check=True).stdout
    req = urllib.request.Request(tsa, data=q, headers={"Content-Type": "application/timestamp-query"})
    with urllib.request.urlopen(req, timeout=30) as r: tsr = r.read()
    open(base + ".tsr", "wb").write(tsr)
    txt = subprocess.run([ossl, "ts", "-reply", "-in", base + ".tsr", "-text"], capture_output=True, text=True).stdout
    if "Granted" not in txt: raise RuntimeError("时间戳机构没有签发：" + (re.search(r"Status:.*", txt) or [""])[0])
    meta = {"seq": h["seq"], "hash": h["hash"], "tsa": tsa, "time": (re.search(r"Time stamp: (.*)", txt) or [None, None])[1], "stamped_at": time.time()}
    json.dump(meta, open(base + ".json", "w"), ensure_ascii=False, indent=1)
    return meta


def stamps():
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ANCHORS, "*.json")))]


def check_stamp(meta, recs=None):
    """The TSA's signature is valid for that hash, and that hash is still the record at that position in this ledger."""
    recs = records() if recs is None else recs
    in_chain = 0 <= meta["seq"] < len(recs) and recs[meta["seq"]]["hash"] == meta["hash"]
    try:
        r = subprocess.run([_openssl(), "ts", "-verify", "-digest", meta["hash"], "-in", os.path.join(ANCHORS, f"{meta['seq']:08d}.tsr"),
                            "-CAfile", _cafile()], capture_output=True, text=True)
        sig = "Verification: OK" in r.stdout + r.stderr
    except Exception:
        sig = False
    return {"signature_ok": sig, "in_chain": in_chain, "ok": sig and in_chain}


def covering_stamp(seq, recs=None):
    """The earliest time stamp taken at or after record `seq`: it fixes that record."""
    for m in stamps():
        if m["seq"] >= seq: return {**m, **check_stamp(m, recs)}
    return None


def evidence(seq, text=None):
    """Claim pack for one event: the record, proof that it sits unaltered in the chain, and (optionally) proof that a
    given text is the one recorded. Contains no content unless the owner supplies it."""
    recs = records(); ok, n, why = verify(recs)
    if not 0 <= seq < len(recs): raise SystemExit(f"没有第 {seq} 条记录")
    r = recs[seq]
    out = {"record": r, "chain": {"intact": ok, "length": n, "problem": why or None, "head": anchor()},
           "how_to_check": "逐条重算 sha256(去掉 hash 字段、键排序的紧凑 JSON)，每条的 prev 必须等于上一条的 hash；"
                           "time_stamp 是第三方时间戳机构（RFC 3161）对某条记录哈希的签名：openssl ts -verify -digest <hash> -in <seq>.tsr -CAfile <根证书> 通过，"
                           "且该位置的记录哈希相同，就说明这条记录在那个时间之前就已存在、之后没有被改写。"}
    out["time_stamp"] = covering_stamp(seq, recs)
    if text is not None: out["text_matches_record"] = r.get("digest") is not None and hmac.compare_digest(digest(text), r["digest"])
    return out


def _t(ts): return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "—"


def report_html(days=30):
    """Self-contained page for an underwriter: period, chain check, what happened, what Nous did, method and limits."""
    from html import escape as e
    recs = records(); ok, n, why = verify(recs); p = profile(days, recs); h = anchor()
    rows = "".join(
        f"<tr><td>{e(CATEGORY.get(k, k))}</td><td class=n>{c['events']}</td><td class=n>{c['mitigated']}</td>"
        f"<td class=n>{round(100 * c['mitigated'] / c['events']) if c['events'] else 0}%</td>"
        f"<td>{e('；'.join(f'{OUTCOME.get(o, o)} {v}' for o, v in sorted(c['outcomes'].items(), key=lambda x: -x[1])))}</td></tr>"
        for k, c in sorted(p["categories"].items(), key=lambda x: -x[1]["events"])) or "<tr><td colspan=5>这段时间没有记录到风险事件。</td></tr>"
    ro = "".join(f"<tr><td>{e(CATEGORY[k])}</td><td class=n>{v['flagged']}</td><td class=n>{'—' if v['per_1000_replies'] is None else v['per_1000_replies']}</td>"
                 f"<td>{'已验证' if v['validated'] else '实验性'}</td></tr>" for k, v in p["readouts"].items())
    ctl = "".join(f"<li class={'on' if v else 'off'}>{'●' if v else '○'} {e(k)}{'' if v else '（这段时间没有触发记录）'}</li>" for k, v in p["controls_seen"].items())
    o = p["oversight"]; st = stamps()
    if st:
        m = st[-1]; c = check_stamp(m, recs); fixed = sum(1 for r in recs if r["seq"] <= m["seq"])
        stamp_html = (f"<h2>第三方时间戳</h2><div class=box><b class={'ok' if c['ok'] else 'bad'}>{'签名有效，且与账本一致' if c['ok'] else '时间戳无法验证'}</b>：前 {fixed} 条记录"
                      f"（到第 {m['seq']} 条）已由 {e(m['tsa'].split('//')[-1])} 于 {e(m['time'] or '—')} 签发时间戳。<br><span class=mut>时间戳机构只收到账本指纹，没有收到任何记录内容。"
                      f"在那之前的记录，任何人（包括这台电脑的管理员）改动都会被发现。共签发 {len(st)} 次。</span></div>")
    else:
        stamp_html = "<h2>第三方时间戳</h2><div class=box><span class=bad>尚未签发。</span><span class=mut>没有第三方时间戳时，完整性只能证明到最近一次发出的账本指纹为止。</span></div>"
    return f"""<!doctype html><html lang=zh-CN><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>AI 使用风险证明</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a65;--line:#e4e1d9;--ok:#2f7a4f;--bad:#a8402f;--acc:#24527a}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"PingFang SC","Helvetica Neue",sans-serif;margin:0;padding:32px 16px}}
main{{max-width:860px;margin:auto}}h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 8px;color:var(--acc)}}
.mut{{color:var(--mut);font-size:13px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:var(--mut);font-weight:500}}.n{{text-align:right;font-variant-numeric:tabular-nums}}.box{{border:1px solid var(--line);border-radius:8px;padding:12px 16px;background:#fff}}
.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}code{{font:12px ui-monospace,Menlo,monospace;word-break:break-all}}ul{{padding-left:0;list-style:none}}li.off{{color:var(--mut)}}
@media print{{body{{padding:0}}}}</style></head><body><main>
<h1>AI 使用风险证明</h1><div class=mut>由 Nous 读心生成 · {_t(time.time())} · 统计最近 {days} 天（{_t(p['from'])} 至 {_t(p['to'])}）</div>
<h2>账本完整性</h2><div class=box><b class={'ok' if ok else 'bad'}>{'哈希链完整' if ok else '哈希链断裂：' + e(why)}</b>，共 {n} 条记录。<br>
账本指纹（第 {h['seq']} 条）：<code>{h['hash']}</code><br><span class=mut>这个指纹每天随汇总发到管理员的 Slack / 飞书。指纹对得上，说明那天以前的记录没有被改写过。
账本里不存任何对话或命令原文，只存本机密钥下的 HMAC 摘要；需要举证时，由使用方提供原文来核对。</span></div>
{stamp_html}
<h2>覆盖范围</h2><div>受保护的 AI 工具：{e('、'.join(p['sources']) or '—')}；记录到需要处置的操作 {p['events']} 次，逐条读心读了 {p['ai_replies_read']} 条 AI 回复。</div>
<h2>AI 的操作：风险事件与处置</h2><table><tr><th>类别</th><th class=n>次数</th><th class=n>Nous 已化解</th><th class=n>化解率</th><th>处置结果</th></tr>{rows}</table>
<h2>AI 的回复：读数（模型内部状态）</h2><div class=mut>回复已经到达使用者，Nous 做的是标注和提醒，不是拦截。</div><table><tr><th>读数</th><th class=n>标注</th><th class=n>每千条回复</th><th>状态</th></tr>{ro}</table>
<h2>人工监督</h2><div>不可撤销的操作共申请 {o['asked']} 次：批准 {o['approved']}，拒绝或超时 {o['refused']}，连不上管理员时交由使用者确认 {o['no_admin']}。</div>
<h2>这段时间实际起作用的防护</h2><ul>{ctl}</ul>
<h2>方法与局限</h2><ul class=mut>
<li>所有数字都是计数或比例，没有加权总分。在有足够的赔付数据之前，我们不给不透明的评分。</li>
<li>“已化解”指拦下、打码、收进保险箱、等待批准期间暂不执行，或管理员拒绝 / 超时自动拒绝。标注提醒、批准后执行和放行记录不算化解。</li>
<li>敌意读数已通过预注册验证；编造、附和、危险建议读数仍是实验性的，只作风险信号，不作定责依据。</li>
<li>只覆盖已接入 Nous 的工具（见“覆盖范围”）。这台电脑以外的 AI 使用、接入前发生的事件都不在统计内。</li>
</ul></main></body></html>"""


if __name__ == "__main__":
    a = sys.argv[1:] or ["verify"]
    if a[0] == "verify": ok, n, why = verify(); print(("账本完整，" if ok else "账本有问题：" + why + "；") + f"共 {n} 条。"); sys.exit(0 if ok else 1)
    elif a[0] == "head": print(json.dumps(anchor()))
    elif a[0] == "stamp": print(json.dumps(stamp(a[1] if len(a) > 1 else None), ensure_ascii=False))
    elif a[0] == "stamps": print(json.dumps([{**m, **check_stamp(m)} for m in stamps()], ensure_ascii=False, indent=1))
    elif a[0] == "profile": print(json.dumps(profile(int(a[1]) if len(a) > 1 else 30), ensure_ascii=False, indent=1))
    elif a[0] == "report":
        open(a[1], "w", encoding="utf-8").write(report_html(int(a[2]) if len(a) > 2 else 30)); print("已生成", a[1])
    elif a[0] == "evidence":
        print(json.dumps(evidence(int(a[1]), open(a[2], encoding="utf-8").read() if len(a) > 2 else None), ensure_ascii=False, indent=1))
    else: sys.exit(__doc__)
