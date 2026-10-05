#!/usr/bin/env python3
"""Mask secret values in text read by an AI agent, keeping the structure it needs (variable names, sections, comments).
  KEY=value            -> KEY=●●●（已隐藏 N 位）
  "key": "value"       -> "key": "●●●（已隐藏 N 位）"   (only for secret-looking key names)
  sk-… / xox… / ghp_… / Bearer … / private key blocks anywhere -> ●●●
Used as a filter (stdin -> stdout) and as a module. Standard library only."""
import re, sys

SECRETISH = re.compile(r"(?i)(pass(word|wd)?|pwd|secret|token|api[_-]?key|access[_-]?key|private|credential|auth|cookie|session|dsn|conn(ection)?_?str|database_url|webhook)")
PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|(AKIA|ASIA)[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{16,})\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*"),
    re.compile(r"[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@]+@"),            # user:password@ in URLs
]
ENV_LINE = re.compile(r"^(\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_.-]*\s*[=:]\s*)(.+?)\s*$")
JSON_PAIR = re.compile(r'("([^"]+)"\s*:\s*")([^"]*)(")')


def hide(v):
    return f"●●●（已隐藏 {len(v)} 位）"


def mask(text, env_file=True):
    """env_file=True: every KEY=value line is masked (the whole file is secrets). Otherwise only secret-looking keys."""
    out = []
    for line in text.splitlines(keepends=True):
        nl = "\n" if line.endswith("\n") else ""
        body = line[:-1] if nl else line
        m = ENV_LINE.match(body)
        if m and not body.lstrip().startswith("#") and (env_file or SECRETISH.search(m.group(1))):
            v = m.group(2).strip().strip('"').strip("'")
            body = m.group(1) + (hide(v) if v else "")
        else:
            body = JSON_PAIR.sub(lambda j: j.group(1) + (hide(j.group(3)) if SECRETISH.search(j.group(2)) and j.group(3) else j.group(3)) + j.group(4), body)
        out.append(body + nl)
    s = "".join(out)
    for p in PATTERNS:
        s = p.sub(lambda m: hide(m.group(0)), s)
    return s


if __name__ == "__main__":
    data = sys.stdin.read()
    sys.stdout.write(mask(data, env_file="--all" not in sys.argv))
    sys.stdout.write("\n[Nous 读心：密钥值已自动隐藏，只保留名称和结构。需要真实值时，请让程序从环境变量读取。]\n")
