#!/bin/bash
# Run Nous 读心 in the background on macOS, started at login and restarted if it stops:
#   center.nous.lens       the 读心 engine (127.0.0.1:7869)
#   center.nous.colleague  the colleague (Slack / Feishu + 127.0.0.1:7870 for Claude Code hooks)
#   center.nous.ship       --member computers only: once a day, DM this computer's ledger bundle to the admin on Slack
#                          (Slack gives each event to one random connection, so only the admin's computer runs the colleague)
# NOUS_GATEWAY=1 also installs the local key gateway (opt-in; not recommended for Claude Code).
# The code is copied to ~/.nous-colleague/app first: background processes cannot read ~/Documents without extra
# macOS permissions. Re-run after changing the code. Remove with:  scripts/install_service.sh --off
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$HOME/.nous-colleague/app"; LOGS="$HOME/.nous-colleague/logs"; LA="$HOME/Library/LaunchAgents"
PLATFORMS="${NOUS_PLATFORMS:-slack}"
UIDN="$(id -u)"

unload() {   # bootout is asynchronous: wait until the jobs are really gone before bootstrapping again
  for l in center.nous.colleague center.nous.lens center.nous.ship; do launchctl bootout "gui/$UIDN/$l" 2>/dev/null || true; done
  for _ in $(seq 30); do launchctl print "gui/$UIDN/center.nous.lens" >/dev/null 2>&1 || launchctl print "gui/$UIDN/center.nous.colleague" >/dev/null 2>&1 || return 0; sleep 1; done
}

if [ "${1:-}" = "--off" ]; then
  "$HOME/.nous-colleague/venv/bin/python" "$APP/colleague/nous_colleague/install.py" claude --off >/dev/null 2>&1 || true   # Claude Code back to direct
  unload; for l in center.nous.reader center.nous.gateway; do launchctl bootout "gui/$UIDN/$l" 2>/dev/null || true; done
  rm -f "$LA/center.nous.colleague.plist" "$LA/center.nous.lens.plist" "$LA/center.nous.reader.plist" "$LA/center.nous.gateway.plist" "$LA/center.nous.ship.plist"
  echo "已停止并移除 Nous 后台服务。"; exit 0
fi

mkdir -p "$APP/lens" "$LOGS" "$LA"
rsync -a --delete --exclude __pycache__ --exclude tests "$REPO/colleague/" "$APP/colleague/"
rsync -a --delete --exclude __pycache__ "$REPO/lens/nous_lens/" "$APP/lens/nous_lens/"
# nous-run: lets an AI use a stored key without seeing it
mkdir -p "$HOME/.local/bin"
printf '#!/bin/sh\nexec python3 "%s/colleague/hooks/nous_vault.py" "$@"\n' "$APP" > "$HOME/.local/bin/nous-run"; chmod 755 "$HOME/.local/bin/nous-run"
# the browser extension, at an easy-to-find path for Chrome's "Load unpacked" (reload it in chrome://extensions after updates)
mkdir -p "$HOME/Applications/Nous 读心浏览器插件"
rsync -a --delete "$REPO/lens/extension/" "$HOME/Applications/Nous 读心浏览器插件/"

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
if [ "${1:-}" = "--member" ]; then   # a team member's computer: hooks + the daily ship job, nothing else
  { echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict><key>Label</key><string>center.nous.ship</string>'
    echo "<key>ProgramArguments</key><array><string>$HOME/.nous-colleague/venv/bin/python</string><string>-m</string><string>nous_colleague.ship</string></array>"
    echo "<key>WorkingDirectory</key><string>$APP/colleague</string>"
    echo "<key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>$APP/colleague</string></dict>"
    echo "<key>StartCalendarInterval</key><dict><key>Hour</key><integer>${NOUS_SHIP_HOUR:-18}</integer><key>Minute</key><integer>5</integer></dict>"
    echo "<key>StandardOutPath</key><string>$LOGS/ship.log</string><key>StandardErrorPath</key><string>$LOGS/ship.log</string>"
    echo '</dict></plist>'; } > "$LA/center.nous.ship.plist"
  unload
  launchctl bootstrap "gui/$UIDN" "$LA/center.nous.ship.plist"
  "$HOME/.nous-colleague/venv/bin/python" "$APP/colleague/nous_colleague/install.py" claude >/dev/null
  echo "成员电脑已安装：Claude Code 保护已开启，每天 ${NOUS_SHIP_HOUR:-18}:05 把证据包私信给管理员（需要 secrets.env 里的 SLACK_BOT_TOKEN 和 SLACK_ADMIN_USER）。"
  exit 0
fi

plist center.nous.lens "$APP/lens" lens "$HOME/.nous-lens/venv/bin/python" -m nous_lens.server
# shellcheck disable=SC2086
plist center.nous.colleague "$APP/colleague" colleague "$HOME/.nous-colleague/venv/bin/python" -m nous_colleague $([ "$PLATFORMS" = "slack feishu" ] || [ "$PLATFORMS" = "all" ] && echo all || echo "$PLATFORMS")

# the key gateway for Claude Code (127.0.0.1:7871): stdlib only, independent of the colleague so a colleague restart never
# interrupts Claude Code
[ "${NOUS_GATEWAY:-}" = "1" ] && plist center.nous.gateway "$APP/colleague" gateway /usr/bin/python3 "$APP/colleague/hooks/nous_gateway.py"
READER="$HOME/Applications/Nous 读心助手.app/Contents/MacOS/NousReader"   # built by colleague/desktop/build.sh
[ -x "$READER" ] && plist center.nous.reader "$HOME/.nous-colleague" reader "$READER"

unload
launchctl bootstrap "gui/$UIDN" "$LA/center.nous.lens.plist"
launchctl bootstrap "gui/$UIDN" "$LA/center.nous.colleague.plist"
launchctl bootout "gui/$UIDN/center.nous.gateway" 2>/dev/null || true
if [ "${NOUS_GATEWAY:-}" = "1" ]; then sleep 1; launchctl bootstrap "gui/$UIDN" "$LA/center.nous.gateway.plist"; else rm -f "$LA/center.nous.gateway.plist"; fi
if [ -f "$LA/center.nous.reader.plist" ]; then
  pkill -f "NousReader --dump" 2>/dev/null || true
  launchctl bootout "gui/$UIDN/center.nous.reader" 2>/dev/null || true; sleep 1
  launchctl bootstrap "gui/$UIDN" "$LA/center.nous.reader.plist"
fi
# point Claude Code's hooks at the installed copy
"$HOME/.nous-colleague/venv/bin/python" "$APP/colleague/nous_colleague/install.py" claude >/dev/null
echo "Nous 后台服务已安装：开机自动启动，停止后自动重启。日志在 $LOGS"
