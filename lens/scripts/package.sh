#!/bin/bash
# Build dist/NousLens-<version>.zip (code + probes + extension; the model is downloaded on first run)
set -e; cd "$(dirname "$0")/.."; V=$(python3 -c "import json;print(json.load(open('extension/manifest.json'))['version'])")
rm -rf dist/stage && mkdir -p dist/stage/NousLens && cp -R nous_lens extension requirements.txt start-mac.command start-windows.bat README.md dist/stage/NousLens/
find dist/stage -name __pycache__ -prune -exec rm -rf {} + ; find dist/stage -name .DS_Store -delete
(cd dist/stage && zip -qr ../NousLens-$V.zip NousLens) && rm -rf dist/stage && ls -la dist/NousLens-$V.zip
