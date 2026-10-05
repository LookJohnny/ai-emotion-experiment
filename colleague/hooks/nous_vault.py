#!/usr/bin/env python3
"""Nous 读心 key vault. Keys the person gives an AI are kept encrypted in the macOS login keychain
(service "center.nous.vault", account = the key's name) and are handed only to programs, never to the model:
  nous-run NAME [NAME...] -- command args     run a command with those keys as environment variables; any key value that
                                              appears in its output is replaced by ●●●
  nous-run --list                             names only
Standard library + the system `security` tool. The names (never the values) are listed in ~/.nous-colleague/vault.json."""
import json, os, re, subprocess, sys

SERVICE = os.environ.get("NOUS_VAULT_SERVICE", "center.nous.vault")          # overridable for tests
INDEX = os.path.expanduser(os.environ.get("NOUS_VAULT_INDEX", "~/.nous-colleague/vault.json"))

# (name hint, pattern for the secret value); order matters: specific before generic
VALUE_PATTERNS = [
    ("ANTHROPIC_API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("OPENAI_API_KEY", re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}")),
    ("API_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("GITHUB_TOKEN", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("AWS_ACCESS_KEY_ID", re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}")),
    ("GROQ_API_KEY", re.compile(r"\bgsk_[A-Za-z0-9]{20,}")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
]
ASSIGN = re.compile(r"(?i)\b([A-Za-z][A-Za-z0-9_]*(?:password|passwd|pwd|secret|token|api[_-]?key))\s*[:=]\s*[\"']?([^\s\"'，,;；]{6,})")
PROVIDERS = [("KIMI_API_KEY", r"kimi|moonshot|月之暗面"), ("DEEPSEEK_API_KEY", r"deepseek|深度求索"), ("OPENAI_API_KEY", r"openai|chatgpt|gpt"),
             ("ANTHROPIC_API_KEY", r"anthropic|claude"), ("DASHSCOPE_API_KEY", r"dashscope|通义|qwen|百炼"), ("ZHIPU_API_KEY", r"zhipu|智谱|glm"),
             ("SILICONFLOW_API_KEY", r"siliconflow|硅基"), ("VOLC_API_KEY", r"火山|doubao|豆包|volc")]


def index():
    try: return json.load(open(INDEX))
    except Exception: return {}


def save_index(d):
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    fd = os.open(INDEX + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f: json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(INDEX + ".tmp", INDEX)


def put(name, value):
    # -U updates an existing item; the value goes to `security` through argv of a short-lived local process only
    r = subprocess.run(["security", "add-generic-password", "-U", "-s", SERVICE, "-a", name, "-l", f"Nous 读心 · {name}", "-w", value],
                       capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError("钥匙串写入失败：" + r.stderr.strip()[:120])
    d = index(); d[name] = {"saved": __import__("time").strftime("%Y-%m-%d %H:%M")}; save_index(d)


def get(name):
    r = subprocess.run(["security", "find-generic-password", "-s", SERVICE, "-a", name, "-w"], capture_output=True, text=True)
    return r.stdout.rstrip("\n") if r.returncode == 0 else None


def find_secrets(text):
    """-> [(name_hint, value)] in order of appearance, deduplicated by value"""
    out, seen = [], set()
    for hint, rx in VALUE_PATTERNS:
        for m in rx.finditer(text):
            v = m.group(0)
            if v not in seen and not any(v in s or s in v for s in seen): seen.add(v); out.append((hint, v, m.start()))
    for m in ASSIGN.finditer(text):
        v = m.group(2)
        if v not in seen and not any(v in s or s in v for s in seen): seen.add(v); out.append((m.group(1).upper(), v, m.start(2)))
    return [(h, v) for h, v, _ in sorted(out, key=lambda x: x[2])]


def name_for(hint, value, text):
    """A readable environment-variable name: the provider named in the message wins over the generic hint."""
    name = hint
    if hint in ("API_KEY", "OPENAI_API_KEY"):
        for n, rx in PROVIDERS:
            if re.search(rx, text, re.I): name = n; break
    existing = index(); base, i = name, 2
    while name in existing and get(name) not in (None, value):
        name = f"{base}_{i}"; i += 1
    return name


def stash(text):
    """Store every secret found in `text`; -> (rewritten text with names instead of values, [names])"""
    names = []
    for hint, value in find_secrets(text):
        n = name_for(hint, value, text); put(n, value); names.append(n)
        text = text.replace(value, f"{n}（已存入 Nous 保险箱）")
    return text, names


def run(argv):
    if "--" not in argv: sys.exit("用法：nous-run 名字 [名字…] -- 命令 参数")
    i = argv.index("--"); names, cmd = argv[:i], argv[i + 1:]
    env, values = dict(os.environ), []
    for n in names:
        v = get(n)
        if v is None: sys.exit(f"Nous 保险箱里没有 {n}。已有：{', '.join(index()) or '（空）'}")
        env[n] = v; values.append(v)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:                       # the program's output may echo a key: never let it reach the model
        for v in values: line = line.replace(v, "●●●")
        sys.stdout.write(line); sys.stdout.flush()
    sys.exit(p.wait())


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--list"]:
        print("\n".join(f"{n}（保存于 {m.get('saved')}）" for n, m in index().items()) or "Nous 保险箱是空的。")
    else:
        run(a)
