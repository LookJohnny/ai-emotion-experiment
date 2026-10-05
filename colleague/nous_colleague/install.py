"""Turn Nous 读心 protection on or off for Claude Code (~/.claude/settings.json) or Codex, including the Codex app
(~/.codex/hooks.json). Only Nous's own hooks are added or removed; a backup is written first.
  python install.py claude|codex [--off]
Codex asks you to review and trust new hooks once (/hooks in the CLI, or the hooks prompt in the app)."""
import json, os, shutil, sys, time

HOOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hooks", "nous_hook.py"))
FILES = {"claude": os.path.expanduser("~/.claude/settings.json"), "codex": os.path.expanduser("~/.codex/hooks.json")}
MARK = "nous_hook.py"


def ours(target):
    cmd = f"python3 {HOOK}" + (" codex" if target == "codex" else "")
    if target == "codex":
        return {
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": cmd, "timeout": 10, "statusMessage": "Nous 读心：检查密钥"}]}],
            "PreToolUse": [{"matcher": "Bash|apply_patch|mcp__.*", "hooks": [{"type": "command", "command": cmd, "timeout": 150,
                                                                             "statusMessage": "Nous 读心：检查操作"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": cmd, "timeout": 60, "async": True}]}],
        }
    return {
        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": cmd, "timeout": 10}]}],
        "PreToolUse": [{"matcher": "Bash|Read|Edit|Write|MultiEdit|NotebookEdit|WebFetch|mcp__.*",
                        "hooks": [{"type": "command", "command": cmd, "timeout": 150}]}],
        "Stop": [{"hooks": [{"type": "command", "command": cmd, "timeout": 60, "async": True}]}],
    }


def strip(hooks):
    out = {}
    for ev, groups in (hooks or {}).items():
        keep = [g for g in groups if not any(MARK in h.get("command", "") for h in g.get("hooks", []))]
        if keep: out[ev] = keep
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in FILES: sys.exit(__doc__)
    target = sys.argv[1]; SETTINGS = FILES[target]
    s = json.load(open(SETTINGS)) if os.path.exists(SETTINGS) else {}
    if os.path.exists(SETTINGS): shutil.copy2(SETTINGS, SETTINGS + f".bak-nous-{time.strftime('%Y%m%d-%H%M%S')}")
    hooks = strip(s.get("hooks"))
    if "--off" not in sys.argv:
        for ev, groups in ours(target).items(): hooks.setdefault(ev, []).extend(groups)
    if hooks: s["hooks"] = hooks
    else: s.pop("hooks", None)
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    tmp = SETTINGS + ".tmp"; json.dump(s, open(tmp, "w"), ensure_ascii=False, indent=2); os.replace(tmp, SETTINGS)
    name = {"claude": "Claude Code", "codex": "Codex"}[target]
    if "--off" in sys.argv: print(f"已关闭 {name} 的 Nous 保护。")
    else: print(f"已为所有项目开启 {name} 的 Nous 保护（新开的会话生效）。" + ("Codex 会请你确认信任这些 hook 一次。" if target == "codex" else ""))


if __name__ == "__main__":
    main()
