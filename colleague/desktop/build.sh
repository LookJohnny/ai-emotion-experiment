#!/bin/bash
# Build "Nous 读心助手.app" into ~/Applications (ad-hoc signed). Rebuilding changes the signature, and macOS then asks for
# the Accessibility permission again, so rebuild only when NousReader.swift really changes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HOME/Applications/Nous 读心助手.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -o "$APP/Contents/MacOS/NousReader" "$HERE/NousReader.swift"
cp "$HERE/../assets/avatar-1024.png" "$APP/Contents/Resources/icon.png" 2>/dev/null || true
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>center.nous.reader</string>
  <key>CFBundleName</key><string>Nous 读心助手</string>
  <key>CFBundleDisplayName</key><string>Nous 读心助手</string>
  <key>CFBundleExecutable</key><string>NousReader</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSAccessibilityUsageDescription</key><string>读取 ChatGPT 桌面版窗口里的文字，交给本机的 Nous 读心检查。不会上传。</string>
</dict></plist>
EOF
codesign --force --sign - "$APP"
echo "$APP"
