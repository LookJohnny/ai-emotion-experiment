"""Turn Nous 读心 protection on or off for Claude Code (all projects), by merging hooks into ~/.claude/settings.json.
  python -m nous_colleague.install claude          # on  (a backup of the settings file is written first)
  python -m nous_colleague.install claude --off    # off (removes only Nous's hooks)"""
import json, os, shutil, sys, time

HOOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hooks", "nous_hook.py"))
SETTINGS = os.path.expanduser("~/.claude/settings.json")
MARK = "nous_hook.py"


def ours():
    cmd = f"python3 {HOOK}"
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
    if len(sys.argv) < 2 or sys.argv[1] != "claude": sys.exit(__doc__)
    s = json.load(open(SETTINGS)) if os.path.exists(SETTINGS) else {}
    if os.path.exists(SETTINGS): shutil.copy2(SETTINGS, SETTINGS + f".bak-nous-{time.strftime('%Y%m%d-%H%M%S')}")
    hooks = strip(s.get("hooks"))
    if "--off" not in sys.argv:
        for ev, groups in ours().items(): hooks.setdefault(ev, []).extend(groups)
    if hooks: s["hooks"] = hooks
    else: s.pop("hooks", None)
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    tmp = SETTINGS + ".tmp"; json.dump(s, open(tmp, "w"), ensure_ascii=False, indent=2); os.replace(tmp, SETTINGS)
    print("已关闭 Claude Code 的 Nous 保护。" if "--off" in sys.argv else "已为所有项目开启 Claude Code 的 Nous 保护（新开的会话生效）。")


if __name__ == "__main__":
    main()
