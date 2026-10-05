#!/usr/bin/env python3
"""Nous 读心 hook for Claude Code. Standard library only; reads the hook JSON on stdin.
  UserPromptSubmit : a prompt with a secret is blocked before it reaches the model (checked locally, works offline)
  PreToolUse       : outbound commands carrying secrets are denied; reading secret files, destructive commands, uploads,
                     piping downloads into a shell, sudo -> the owner approves or denies in Slack / Feishu
                     (no answer in time -> denied; Nous unreachable -> falls back to Claude Code's own local prompt)
  Stop (async)     : the final answer is read by Nous 读心; flagged -> the owner gets a DM
Decisions are logged to ~/.nous-colleague/hook.log without prompts, commands or secrets."""
import json, os, re, sys, time, urllib.request

HOME = os.path.expanduser("~/.nous-colleague")
API = os.environ.get("NOUS_LOCAL_API", "http://127.0.0.1:7870")
SAFETY_CORE = os.environ.get("NOUS_SAFETY_CORE", os.path.expanduser(
    "~/.codex/.chatgpt-projects/g-p-6abfea876b9c8191980bd8cb26029f18/nous-safety-core"))
URGENT = {"private_key", "api_key", "bearer_token", "credential", "cvv", "payment_card"}
NAME = {"private_key": "私钥", "api_key": "API 密钥", "bearer_token": "访问令牌", "credential": "密码或密钥", "cvv": "银行卡安全码", "payment_card": "银行卡号"}

SECRET_FILE = re.compile(r"""(?ix)(^|[\s/'"=])(
    \.env(?!\.(example|sample|template|dist))(\.[\w-]+)?\b | \.ssh/ | id_(rsa|ed25519|ecdsa|dsa)\b | \.(pem|p12|pfx|keystore|jks)\b |
    [\w.-]+\.key\b | \.aws/credentials | \.netrc\b | \.npmrc\b | \.pypirc\b | \.git-credentials | \.docker/config\.json |
    \.kube/config | secrets?\.(env|json|ya?ml|toml)\b | credentials\.json\b | \.nous-colleague/local\.token |\w+\.env\b(?!\.)
)""")
READERS = r"(cat|less|more|head|tail|bat|grep|rg|awk|sed|cp|mv|scp|rsync|base64|xxd|od|strings|source|\.|open|curl|zip|tar)"
RISKY = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*[rR])[a-zA-Z]*\s+(/|~|\$HOME|\*|\.\.?)(\s|/?$|/\*)"), "大范围删除文件"),
    (re.compile(r"\bgit\s+push\b.*\s(--force(-with-lease)?|-f)\b"), "强制推送，会覆盖远端历史"),
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f)"), "丢弃本地未提交的修改"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|python3?|node)\b"), "下载脚本并直接执行"),
    (re.compile(r"(^|[;&|]\s*)sudo\b"), "以管理员权限执行"),
    (re.compile(r"\bsecurity\s+(find|dump)-(generic|internet)-password|\bsecurity\s+dump-keychain"), "读取系统钥匙串"),
    (re.compile(r"\b(mkfs|diskutil\s+erase\w*|dd\s+if=)"), "格式化或直接写磁盘"),
    (re.compile(r"\bchmod\s+(-R\s+)?777\b"), "把权限开放给所有人"),
    (re.compile(r"\bcurl\b.*\s(-d|--data(-\w+)?|-F|--form|-T|--upload-file)\b|\bwget\b.*--post-(data|file)|\bscp\b.*\s[\w.-]+@?[\w.-]+:|\brsync\b.*\s[\w.-]+@[\w.-]+:|\bnc\s+[\w.-]+\s+\d+"), "把数据发到外部"),
]
OUTBOUND = re.compile(r"\b(curl|wget|http|nc|scp|rsync|ssh|ftp|git\s+push|gh\s+api|aws\s+s3\s+cp)\b")


def log(event, decision, why=""):
    try:
        os.makedirs(HOME, exist_ok=True)
        with open(os.path.join(HOME, "hook.log"), "a") as f:
            f.write(json.dumps({"ts": round(time.time()), "event": event, "decision": decision, "why": why}, ensure_ascii=False) + "\n")
    except OSError:
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


def pre_tool(d):
    """-> (decision, reason, ask_lines) ; decision in {None, 'deny', 'approve'}"""
    tool, ti = d.get("tool_name", ""), d.get("tool_input") or {}
    if tool == "Bash":
        cmd = str(ti.get("command", ""))
        found = secrets_in(cmd)
        if found and OUTBOUND.search(cmd):
            return "deny", f"命令要把 {NAME.get(found[0], '密钥')} 发到外部", []
        if SECRET_FILE.search(cmd) and re.search(r"(^|[\s;&|(])" + READERS + r"\b", cmd):
            return "approve", "读取密钥文件，内容会进入 AI 的上下文", [f"要执行的命令：`{cmd[:300]}`"]
        for rx, why in RISKY:
            if rx.search(cmd): return "approve", why, [f"要执行的命令：`{cmd[:300]}`"]
        return None, "", []
    if tool in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        p = str(ti.get("file_path") or ti.get("notebook_path") or "")
        if SECRET_FILE.search(p): return "approve", f"{'读取' if tool == 'Read' else '修改'}密钥文件，内容会进入 AI 的上下文", [f"文件：`{p}`"]
        return None, "", []
    if tool == "WebFetch" or tool.startswith("mcp__"):
        found = secrets_in(json.dumps(ti, ensure_ascii=False))
        if found: return "deny", f"要把 {NAME.get(found[0], '密钥')} 发给外部服务", []
    return None, "", []


def out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False)); sys.stdout.flush()


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
    ev = d.get("hook_event_name", "")
    src = "Claude Code · " + os.path.basename(d.get("cwd") or "") if d.get("cwd") else "Claude Code"
    if ev == "UserPromptSubmit":
        found = secrets_in(str(d.get("user_input") or d.get("prompt") or ""))
        if found:
            names = "、".join(dict.fromkeys(NAME[t] for t in found))
            log(ev, "block", ",".join(found))
            call("/notify", {"source": src, "title": "已拦下一条含密钥的消息", "level": "warn",
                             "lines": [f"你在 Claude Code 里发的消息像是有 **{names}**，Nous 在发给模型之前拦下了，模型没有收到。"]}, 3)
            out({"decision": "block", "reason": f"Nous 读心：这条消息里像是有 {names}，已拦下，没有发给模型。请删掉密钥后重发；需要用到时，让 AI 从环境变量或密钥文件读取。"})
        return
    if ev == "PreToolUse":
        decision, why, lines = pre_tool(d)
        if decision == "deny":
            log(ev, "deny", why)
            call("/notify", {"source": src, "title": "已拦下一个操作", "level": "warn", "lines": [f"Claude Code {why}，Nous 没有让它执行。"]}, 3)
            out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                        "permissionDecisionReason": f"Nous 读心：{why}，已拦下。请不要在命令或请求里直接带密钥。"}})
        elif decision == "approve":
            r = call("/approve", {"source": src, "title": f"Claude Code 想{why}" if not why.startswith(("读取", "修改")) else f"Claude Code 要{why}",
                                  "lines": [f"**{why}**"] + lines + [f"项目：{d.get('cwd', '')}"], "timeout": 120}, 135)
            dec = (r or {}).get("decision")
            log(ev, dec or "unavailable", why)
            if dec == "allow":
                out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow", "permissionDecisionReason": "Nous 读心：管理员已允许这一次。"}})
            elif dec in ("deny", "timeout"):
                out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                            "permissionDecisionReason": "Nous 读心：管理员拒绝了这个操作。" if dec == "deny" else "Nous 读心：2 分钟内没有人确认，已自动拒绝。"}})
            else:
                out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                            "permissionDecisionReason": f"Nous 读心：{why}。连不上 Nous，请你在这里确认。"}})
        return
    if ev == "Stop":
        reply = str(d.get("last_assistant_message") or "")
        if len(reply) >= 20:
            r = call("/read", {"source": src, "reply": reply[:8000], "context": last_user_text(d.get("transcript_path") or "")}, 60)
            log(ev, "read", ",".join((r or {}).get("findings", [])) if r else "unavailable")


if __name__ == "__main__":
    try: main()
    except Exception as e:                  # never break the person's session because of Nous
        log("error", "none", repr(e)[:120])
