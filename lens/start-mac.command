#!/bin/bash
# Nous 读心 · macOS 启动（Apple 芯片，内存 ≥ 16 GB）
cd "$(dirname "$0")"
D="$HOME/.nous-lens"; VENV="$D/venv"; MODEL="$D/models/Qwen3-4B-Instruct-2507"; MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
mkdir -p "$D"
if [ ! -x "$VENV/bin/python" ]; then
  PY=""; for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v $c >/dev/null && $c -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then PY=$c; break; fi; done
  if [ -z "$PY" ]; then echo "需要 Python 3.10 或更新版本：https://www.python.org/downloads/ （安装后重新双击本文件）"; read -p "按回车退出"; exit 1; fi
  echo "首次运行：正在准备运行环境（约 3–5 分钟）…"
  $PY -m venv "$VENV" && "$VENV/bin/python" -m pip install -q --upgrade pip -i "$MIRROR" 2>/dev/null
  if ! "$VENV/bin/pip" install -q -i "$MIRROR" -r requirements.txt 2>/dev/null; then
    echo "国内镜像暂不可用，改用官方源继续安装…"
    "$VENV/bin/pip" install -q -r requirements.txt || { rm -rf "$VENV"; echo "安装失败，请检查网络后重试"; read -p "按回车退出"; exit 1; }
  fi
fi
if [ ! -f "$MODEL/.complete" ]; then
  echo "首次运行：正在下载本地模型 Qwen3-4B（约 8 GB，只需一次）…"
  "$VENV/bin/modelscope" download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir "$MODEL" && touch "$MODEL/.complete" || { echo "模型下载失败，请重试"; read -p "按回车退出"; exit 1; }
fi
"$VENV/bin/python" -m nous_lens.server --model "$MODEL"
