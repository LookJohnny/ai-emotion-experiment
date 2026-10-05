#!/bin/bash
# Run Nous 读心 in the background on macOS, started at login and restarted if it stops:
#   center.nous.lens       the 读心 engine (127.0.0.1:7869)
#   center.nous.colleague  the colleague (Slack / Feishu + 127.0.0.1:7870 for Claude Code hooks)
# The code is copied to ~/.nous-colleague/app first: background processes cannot read ~/Documents without extra
# macOS permissions. Re-run after changing the code. Remove with:  scripts/install_service.sh --off
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$HOME/.nous-colleague/app"; LOGS="$HOME/.nous-colleague/logs"; LA="$HOME/Library/LaunchAgents"
PLATFORMS="${NOUS_PLATFORMS:-slack}"
UIDN="$(id -u)"

unload() {   # bootout is asynchronous: wait until the jobs are really gone before bootstrapping again
  for l in center.nous.colleague center.nous.lens; do launchctl bootout "gui/$UIDN/$l" 2>/dev/null || true; done
  for _ in $(seq 30); do launchctl print "gui/$UIDN/center.nous.lens" >/dev/null 2>&1 || launchctl print "gui/$UIDN/center.nous.colleague" >/dev/null 2>&1 || return 0; sleep 1; done
}

if [ "${1:-}" = "--off" ]; then
  unload; rm -f "$LA/center.nous.colleague.plist" "$LA/center.nous.lens.plist"
  echo "已停止并移除 Nous 后台服务。"; exit 0
fi

mkdir -p "$APP/lens" "$LOGS" "$LA"
rsync -a --delete --exclude __pycache__ --exclude tests "$REPO/colleague/" "$APP/colleague/"
rsync -a --delete --exclude __pycache__ "$REPO/lens/nous_lens/" "$APP/lens/nous_lens/"

plist() {  # label workdir log program-args...
  local label=$1 wd=$2 log=$3; shift 3
  { echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict>'
    echo "<key>Label</key><string>$label</string>"
    echo '<key>ProgramArguments</key><array>'; for a in "$@"; do echo "<string>$a</string>"; done; echo '</array>'
    echo "<key>WorkingDirectory</key><string>$wd</string>"
    echo '<key>EnvironmentVariables</key><dict><key>PYTHONUNBUFFERED</key><string>1</string><key>PYTHONPATH</key><string>'"$wd"'</string></dict>'
    echo '<key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>30</integer>'
    echo "<key>StandardOutPath</key><string>$LOGS/$log.log</string><key>StandardErrorPath</key><string>$LOGS/$log.log</string>"
    echo '</dict></plist>'; } > "$LA/$label.plist"
}
plist center.nous.lens "$APP/lens" lens "$HOME/.nous-lens/venv/bin/python" -m nous_lens.server
# shellcheck disable=SC2086
plist center.nous.colleague "$APP/colleague" colleague "$HOME/.nous-colleague/venv/bin/python" -m nous_colleague $([ "$PLATFORMS" = "slack feishu" ] || [ "$PLATFORMS" = "all" ] && echo all || echo "$PLATFORMS")

unload
launchctl bootstrap "gui/$UIDN" "$LA/center.nous.lens.plist"
launchctl bootstrap "gui/$UIDN" "$LA/center.nous.colleague.plist"
# point Claude Code's hooks at the installed copy
"$HOME/.nous-colleague/venv/bin/python" "$APP/colleague/nous_colleague/install.py" claude >/dev/null
echo "Nous 后台服务已安装：开机自动启动，停止后自动重启。日志在 $LOGS"
