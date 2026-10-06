#!/usr/bin/env python3
"""Nous 读心 hook for Claude Code and Codex (`nous_hook.py codex` for Codex). Standard library only; hook JSON on stdin.

Principle: protect without getting in the way. Nous fixes what it can by itself and asks a person only for the few actions
that cannot be undone.
  block   a prompt containing a secret (before the model sees it); a command or request that would send a secret, or a
          secret file, to the outside. The owner gets a short DM.
  mask    reading a secret file (.env, keys, credentials): allowed, but the agent sees ●●● instead of values. Claude Code's
          Read is pointed at a masked copy; shell reads are piped through nous_mask.py. Counted in the daily digest.
  ask     only irreversible, rare actions: deleting a home or system directory, force-pushing a main branch, erasing a disk,
          dumping the keychain. Claude Code waits for the owner's tap in Slack / Feishu (2 min, then denied); Codex gives up
          on slow tools, so it is denied at once and the owner's "allow" lets exactly that action through on the next try.
  note    risky but routine (curl | sh, uploads, sudo, git reset --hard, chmod 777): allowed, counted in the daily digest.
  Stop    (async) the final answer is read by Nous 读心; flagged -> DM.
~/.nous-colleague/hook.log records decisions without prompts, commands or secrets."""
import hashlib, json, os, re, sys, time, urllib.request

HOME = os.path.expanduser("~/.nous-colleague")
HERE = os.path.dirname(os.path.abspath(__file__))
MASK = os.path.join(HERE, "nous_mask.py")
API = os.environ.get("NOUS_LOCAL_API", "http://127.0.0.1:7870")
SAFETY_CORE = os.environ.get("NOUS_SAFETY_CORE", os.path.expanduser(
    "~/.codex/.chatgpt-projects/g-p-6abfea876b9c8191980bd8cb26029f18/nous-safety-core"))
URGENT = {"private_key", "api_key", "bearer_token", "credential", "cvv", "payment_card"}
NAME = {"private_key": "私钥", "api_key": "API 密钥", "bearer_token": "访问令牌", "credential": "密码或密钥", "cvv": "银行卡安全码", "payment_card": "银行卡号"}

SECRET_FILE = re.compile(r"""(?ix)(^|[\s/'"=@])(
    \.env(?!\.(example|sample|template|dist))(\.[\w-]+)?\b | \.ssh/ | id_(rsa|ed25519|ecdsa|dsa)\b | \.(pem|p12|pfx|keystore|jks)\b |
    [\w.-]+\.key\b | \.aws/credentials | \.netrc\b | \.npmrc\b | \.pypirc\b | \.git-credentials | \.docker/config\.json |
    \.kube/config | secrets?\.(env|json|ya?ml|toml)\b | credentials\.json\b | \.nous-colleague/local\.token | \w+\.env\b(?!\.)
)""")
SHOWS = {"cat", "less", "more", "head", "tail", "bat", "grep", "rg", "egrep", "awk", "sed", "base64", "xxd", "od", "strings", "nl", "cut",
         "sort", "uniq", "diff", "jq", "yq", "tac", "column", "printenv"}          # print file content -> masked
SENDS = {"curl", "wget", "scp", "rsync", "nc", "ftp", "sftp", "http", "https"}     # with a secret file as argument -> blocked
PREFIXES = {"sudo", "env", "command", "time", "nice", "nohup", "xargs", "exec"}
IRREVERSIBLE = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*[rR])[a-zA-Z]*\s+(/|~|\$HOME|/Users/[^/\s]+|/home/[^/\s]+)/?(\s|$|\*)"),
     "删除整个主目录或系统目录", "这会删掉大量文件，无法恢复。"),
    (re.compile(r"\bgit\s+push\b(?=.*\s(--force\b|-f\b|\+))(?=.*\b(main|master|release|prod|production)\b)"),
     "强制推送到主分支", "这会覆盖远端的提交历史，别人的提交可能丢失。"),
    (re.compile(r"\b(mkfs(\.\w+)?|diskutil\s+(erase\w*|zeroDisk|secureErase)|dd\s+[^|;]*of=/dev/)"), "格式化或直接写磁盘", "磁盘上的数据会被清除，无法恢复。"),
    (re.compile(r"\bsecurity\s+(dump-keychain|find-(generic|internet)-password\b[^|;]*\s-w)"), "导出系统钥匙串里的密码", "钥匙串里存着你各个账号的密码。"),
]
ROUTINE = [
    (re.compile(r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(sh|bash|zsh|python3?|node)\b"), "下载脚本并直接执行"),
    (re.compile(r"\bcurl\b.*\s(-F|--form|-T|--upload-file|--data-binary)\b|\bscp\b.*\s[\w.-]+@?[\w.-]+:|\brsync\b.*\s[\w.-]+@[\w.-]+:"), "上传文件到外部"),
    (re.compile(r"(^|[;&|]\s*)sudo\b"), "以管理员权限执行"),
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f)"), "丢弃本地未提交的修改"),
    (re.compile(r"\bchmod\s+(-R\s+)?777\b"), "把文件权限开放给所有人"),
]
OUTBOUND = re.compile(r"\b(curl|wget|http|nc|scp|rsync|ssh|ftp|git\s+push|gh\s+api|aws\s+s3\s+cp)\b")


def log(event, decision, why=""):
    try:
        os.makedirs(HOME, exist_ok=True)
        with open(os.path.join(HOME, "hook.log"), "a") as f:
            f.write(json.dumps({"ts": round(time.time()), "event": event, "decision": decision, "why": why}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def ledger(category, outcome, source, title="", text=None):
    """One record in the Nous 证据账本 (hash chain, no content kept). Never raises."""
    try:
        sys.path.insert(0, HERE)
        import nous_ledger
        nous_ledger.append(category, outcome, source, title, text)
    except Exception:
        pass


def secrets_in(text):
    try:
        sys.path.insert(0, SAFETY_CORE)
        from safety_core.core import _redact
        return [t for t in _redact(text)[1] if t in URGENT]
    except Exception:                       # rules unavailable: a minimal fallback for the most common keys
        pats = [r"\bsk-[A-Za-z0-9_-]{16,}", r"\bxox[abprs]-[A-Za-z0-9-]{10,}", r"\bgh[pousr]_[A-Za-z0-9]{20,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
        return ["api_key"] if any(re.search(p, text) for p in pats) else []


def call(path, body, timeout):
    try: tok = open(os.path.join(HOME, "local.token")).read().strip()
    except OSError: return None
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except Exception:
        return None


def segments(cmd):
    """Simple commands of a shell line, quote-aware (; | & ( ) inside awk/sed scripts do not split). Heredoc bodies dropped."""
    import shlex
    cmd = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\s*(\n|$)", "\n", cmd, flags=re.S)
    cmd = cmd.replace("`", " ; ").replace("\n", " ; ")
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True); lex.whitespace_split = True; lex.commenters = ""
        toks = list(lex)
    except ValueError:
        toks = cmd.split()
    segs, cur = [], []
    for t in toks:
        if t and set(t) <= set(";|&()"): segs.append(cur); cur = []
        else: cur.append(t)
    segs.append(cur)
    out = []
    for seg in segs:
        while seg and (re.fullmatch(r"\w+=\S*", seg[0]) or seg[0] in PREFIXES or seg[0] == "$"): seg = seg[1:]
        if seg: out.append(seg)
    return out


def secret_args(seg):
    return [t for t in seg[1:] if (not t.startswith("-") or "=" in t) and SECRET_FILE.search(t)] + \
           [b for a, b in zip(seg, seg[1:]) if a in ("<", "<<<") and SECRET_FILE.search(b)]


def short(cmd, n=80):
    c = " ".join(cmd.split())
    return c if len(c) <= n else ""


def classify(d):
    """-> (action, title, detail)  action in {None, 'block', 'mask', 'ask', 'note'}"""
    tool, ti = d.get("tool_name", ""), d.get("tool_input") or {}
    if tool in ("Bash", "exec_command", "shell", "local_shell"):
        cmd = ti.get("command") or ti.get("cmd") or ""          # Codex code-mode exec_command sends `cmd`
        cmd = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        for seg in segments(cmd):            # the command word itself must be the keychain tool, not a mention in text
            if os.path.basename(seg[0]) == "security" and any("center.nous.vault" in t for t in seg[1:]) and \
                    any(t.startswith("find-") or t.startswith("dump-") for t in seg[1:]):
                return "block", "要从 Nous 保险箱里取出密钥明文", "AI 只能通过 nous-run 使用密钥。"
        found = secrets_in(cmd)
        if found and OUTBOUND.search(cmd):
            return "block", f"要把{NAME.get(found[0], '密钥')}发到外部", "命令里直接带着密钥。"
        segs = segments(cmd)
        for seg in segs:
            if secret_args(seg) and os.path.basename(seg[0]) in SENDS:
                f = re.split(r"[=@]", secret_args(seg)[0])[-1]
                return "block", f"要把密钥文件 {os.path.basename(f)} 发到外部", ""
        for rx, title, why in IRREVERSIBLE:
            if rx.search(cmd): return "ask", title, why
        if any(secret_args(s) and os.path.basename(s[0]) in SHOWS for s in segs):
            f = next(secret_args(s)[0] for s in segs if secret_args(s) and os.path.basename(s[0]) in SHOWS)
            return "mask", f"读取 {os.path.basename(f)}", ""
        for rx, title in ROUTINE:
            if rx.search(cmd): return "note", title, ""
        return None, "", ""
    if tool == "Read":
        p = str(ti.get("file_path") or "")
        if SECRET_FILE.search(p) and os.path.isfile(p): return "mask", f"读取 {os.path.basename(p)}", ""
        return None, "", ""
    if tool == "WebFetch" or tool.startswith("mcp__"):
        found = secrets_in(json.dumps(ti, ensure_ascii=False))
        if found: return "block", f"要把{NAME.get(found[0], '密钥')}发给外部服务", ""
    return None, "", ""


def masked_copy(path):
    """A masked copy of a secret file for Claude Code's Read tool (0600, under ~/.nous-colleague/masked)."""
    sys.path.insert(0, HERE)
    from nous_mask import mask
    d = os.path.join(HOME, "masked"); os.makedirs(d, mode=0o700, exist_ok=True)
    dst = os.path.join(d, hashlib.sha256(os.path.realpath(path).encode()).hexdigest()[:10] + "-" + os.path.basename(path) + ".masked")
    with open(path, encoding="utf-8", errors="replace") as f: text = f.read()
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"# Nous 读心：这是 {os.path.basename(path)} 的打码副本，密钥值已隐藏，只保留名称和结构。\n" + mask(text))
    return dst


def out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False)); sys.stdout.flush()


def decide(kind, reason, updated=None):
    o = {"hookEventName": "PreToolUse", "permissionDecision": kind, "permissionDecisionReason": reason}
    if updated is not None: o["updatedInput"] = updated
    out({"hookSpecificOutput": o})


def last_user_text(path):
    try:
        lines = open(path, encoding="utf-8").read().splitlines()[-400:]
    except OSError:
        return None
    for l in reversed(lines):
        try: e = json.loads(l)
        except ValueError: continue
        m = e.get("message") or {}
        if e.get("type") == "user" and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip(): return c[:2000]
            if isinstance(c, list):
                t = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text").strip()
                if t: return t[:2000]
    return None


def main():
    try: d = json.load(sys.stdin)
    except ValueError: return
    codex = "codex" in sys.argv[1:]; app = "Codex" if codex else "Claude Code"
    ev = d.get("hook_event_name", ""); project = os.path.basename(d.get("cwd") or "") or "本机"
    src = f"{app} · {project}"
    if ev == "UserPromptSubmit":
        prompt = str(d.get("user_input") or d.get("prompt") or "")
        sys.path.insert(0, HERE)
        import nous_vault
        stored = []
        if not codex and "127.0.0.1:7871" in os.environ.get("ANTHROPIC_BASE_URL", ""):
            # Claude Code goes through the Nous gateway: the key is swapped for a placeholder before it leaves this
            # computer, so nothing needs to be held. Just store it now and tell the model how it will see it.
            names = []
            for hint, v in nous_vault.find_secrets(prompt):
                n = nous_vault.name_for(hint, v, prompt); nous_vault.put(n, v); names.append(n)
            names += [n for n in nous_vault.index() if n in prompt and n not in names]
            if names:
                log(ev, "vaulted-seamless", ",".join(names)); ledger("data_exposure", "vaulted", src, "对话里给出的密钥", prompt); call("/note", {"kind": "密钥由保险箱托管"}, 2)
                out({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext":
                     "Nous 读心：用户给的密钥由 Nous 保险箱托管，在你看来是占位符 " + "、".join(f"[[NOUS:{n}]]" for n in names) +
                     "。需要用到时，在命令、配置文件或代码里原样写这个占位符即可，执行时 Nous 会自动换成真实值。"
                     "不要让用户重发明文，也不要尝试读取钥匙串。"}})
            return
        if secrets_in(prompt) or nous_vault.find_secrets(prompt):
            # the person is handing a key to the AI: keep it in the keychain and give the AI a name, never the value.
            # Neither Claude Code nor Codex lets a hook rewrite the prompt, so this one prompt is held and a ready-to-send
            # version (key replaced by its name) is shown.
            try: rewritten, stored = nous_vault.stash(prompt)
            except Exception as e: log(ev, "vault-error", repr(e)[:80])
            if stored:
                log(ev, "vaulted", ",".join(stored)); ledger("data_exposure", "vaulted", src, "对话里给出的密钥", prompt); call("/note", {"kind": "密钥存入保险箱"}, 2)
                try:
                    if os.environ.get("NOUS_NO_CLIPBOARD"): raise RuntimeError("clipboard disabled (tests)")
                    import subprocess; subprocess.run(["pbcopy"], input=rewritten, text=True, timeout=3); copied = True
                except Exception: copied = False
                out({"decision": "block", "reason":
                     f"Nous 读心：密钥已存进保险箱（{'、'.join(stored)}），模型不会看到明文。"
                     + ("换好名字的消息已复制，按 ⌘V 再回车发送即可。" if copied else f"请重新发送：{rewritten}")})
                return
        names = [n for n in nous_vault.index() if n in prompt]
        if names:                            # tell the model how to use stored keys without ever seeing them
            out({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext":
                 f"Nous 保险箱（macOS 钥匙串）里有这些密钥：{', '.join(names)}。不要向用户索要明文，也不要尝试读取钥匙串。"
                 f"需要时用 `nous-run {names[0]} -- <命令>` 运行程序：程序从同名环境变量拿到密钥，输出里出现的密钥会被打码。"}})
            return
        found = secrets_in(prompt)
        if found:
            names = "、".join(dict.fromkeys(NAME[t] for t in found))
            log(ev, "block", ",".join(found)); ledger("data_exposure", "blocked", src, "消息里带着密钥", prompt)
            call("/notify", {"source": src, "title": f"已拦下：你发给 {app} 的消息里有{names}", "level": "warn",
                             "lines": ["这条消息没有发给模型。删掉密钥后重发即可。"]}, 3)
            out({"decision": "block", "reason": f"Nous 读心：这条消息里像是有{names}，已拦下，没有发给模型。请删掉密钥后重发；需要用到时，让 AI 从环境变量或密钥文件读取。"})
        return
    if ev == "PreToolUse":
        raw = json.dumps(d.get("tool_input") or {}, ensure_ascii=False)
        via_gateway = "127.0.0.1:7871" in os.environ.get("ANTHROPIC_BASE_URL", "")
        if "[[NOUS:" in raw and via_gateway:          # elsewhere a placeholder is just text (e.g. code about Nous itself)
            # the model uses a key it only knows as a placeholder: put the real value in, at execution time only
            sys.path.insert(0, HERE)
            import nous_vault
            missing = []
            def sub(m):
                v = nous_vault.get(m.group(1))
                if v is None: missing.append(m.group(1)); return m.group(0)
                return v
            filled = re.sub(r"\[\[NOUS:([A-Za-z0-9_]+)\]\]", sub, raw)
            if filled != raw:                         # at least one stored key used (unknown names stay as text)
                log(ev, "vault-fill", d.get("tool_name", ""))
                # keep Claude Code's own permission flow: auto-approve only where the person already chose not to be asked
                mode = d.get("permission_mode", "default")
                decide("allow" if mode in ("bypassPermissions", "dontAsk", "auto", "acceptEdits") or codex else "ask",
                       "Nous 读心：已在执行时填入保险箱里的密钥（模型看不到）。", json.loads(filled))
                return
        action, title, why = classify(d)
        if action is None: return
        ti = dict(d.get("tool_input") or {})
        log(ev, action, title)
        cmd_text = json.dumps(ti, ensure_ascii=False, sort_keys=True)
        if action in ("block", "mask", "note"):
            ledger({"block": "data_exposure", "mask": "data_exposure", "note": "risky_command"}[action],
                   {"block": "blocked", "mask": "masked", "note": "noted"}[action], src, title, cmd_text)
        if action == "block":
            call("/notify", {"source": src, "title": f"已拦下：{app} {title}", "level": "warn", "lines": [(why + " " if why else "") + "没有执行。"]}, 3)
            decide("deny", f"Nous 读心：{title}，已拦下。不要在命令或请求里直接使用密钥或密钥文件；需要时让程序从环境变量读取。")
        elif action == "mask":
            call("/note", {"kind": "masked"}, 2)
            if d.get("tool_name") == "Read":
                decide("allow", f"Nous 读心：{title} 时已自动隐藏密钥值。", {**ti, "file_path": masked_copy(ti["file_path"])})
            else:
                key = "cmd" if "cmd" in ti and "command" not in ti else "command"
                wrapped = f"( {ti.get(key)} ) 2>&1 | python3 '{MASK}'"
                upd = {**ti, key: wrapped}
                if codex: upd["command"] = wrapped                # Codex requires a string `command` in updatedInput
                decide("allow", f"Nous 读心：{title} 时已自动隐藏密钥值。", upd)
        elif action == "note":
            call("/note", {"kind": title}, 2)
        elif action == "ask":
            cmd = ti.get("command") or ti.get("cmd") or ""
            lines = [why, f"项目：{project}"] + ([f"命令：`{short(str(cmd))}`"] if short(str(cmd)) else [])
            card = {"source": src, "title": f"{app} 想{title}", "lines": lines}
            if codex:                # Codex gives up on slow tools: never wait; "allow" lets exactly this action through once
                dg = hashlib.sha256(json.dumps([d.get("cwd"), d.get("tool_name"), cmd], ensure_ascii=False).encode()).hexdigest()
                st = (call("/approve_async", {**card, "digest": dg}, 8) or {}).get("status", "unavailable")
                log(ev, "async-" + st, title)
                ledger("unauthorized_action", {"granted": "approved", "pending": "held", "denied": "denied"}.get(st, "blocked"), src, title, cmd_text)
                if st == "granted": decide("allow", "Nous 读心：管理员已允许这一次。")
                else:
                    decide("deny", {"pending": f"Nous 读心：{title}无法撤销，需要管理员批准，已发到管理员的 Slack / 飞书。请告诉用户；批准后原样再执行这一条一次即可，不要换写法绕过。",
                                    "denied": "Nous 读心：管理员拒绝了这个操作，不要再尝试。"}.get(st, f"Nous 读心：{title}无法撤销，需要管理员确认，但现在连不上 Nous，已拒绝。"))
            else:
                dec = (call("/approve", {**card, "timeout": 120}, 135) or {}).get("decision")
                log(ev, dec or "unavailable", title)
                ledger("unauthorized_action", {"allow": "approved", "deny": "denied", "timeout": "timeout"}.get(dec, "asked_in_tool"), src, title, cmd_text)
                if dec == "allow": decide("allow", "Nous 读心：管理员已允许这一次。")
                elif dec in ("deny", "timeout"): decide("deny", "Nous 读心：管理员拒绝了这个操作。" if dec == "deny" else "Nous 读心：2 分钟内没有人确认，已自动拒绝。")
                else: decide("ask", f"Nous 读心：{title}，无法撤销。连不上 Nous，请你在这里确认。")
        return
    if ev == "Stop":
        reply = str(d.get("last_assistant_message") or "")
        if len(reply) >= 20:
            r = call("/read", {"source": src, "reply": reply[:8000], "context": None if codex else last_user_text(d.get("transcript_path") or "")}, 60)
            log(ev, "read", ",".join((r or {}).get("findings", [])) if r else "unavailable")


if __name__ == "__main__":
    try: main()
    except Exception as e:                  # never break the person's session because of Nous
        log("error", "none", repr(e)[:120])
