#!/bin/bash
# B2 -> B3 -> B4 chain, runs entirely on the server. Stops at the first failed self-patch check or crash.
cd /root/autodl-tmp/E01
PY=/root/miniconda3/bin/python; M=models/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master; D=data/formal_960.jsonl; T="我的看法是："
check() { $PY -c "import json,sys;s=json.load(open('$1'))['self_patch_exact'];a,b=s.split('/');sys.exit(0 if a==b and b!='0' else 1)"; }
log() { echo "$(date '+%F %T') $*" >> logs/chain.log; }
rm -f runs/CHAIN_DONE runs/CHAIN_FAILED; log "chain start"
mkdir -p runs/B2 runs/B3 runs/B4
$PY data/b2_windows.py $M $D runs/B2 --tail "$T" --b1 runs/B1/b1_summary.json --primary anger > logs/B2.log 2>&1 \
  && check runs/B2/b2_summary.json || { log "B2 failed"; echo B2 > runs/CHAIN_FAILED; exit 1; }
log "B2 ok"
$PY data/b3_heads.py $M $D runs/B3 --tail "$T" --b2 runs/B2/b2_summary.json --primary anger > logs/B3.log 2>&1 \
  && check runs/B3/b3_summary.json || { log "B3 failed"; echo B3 > runs/CHAIN_FAILED; exit 1; }
log "B3 ok"
$PY data/b4_repr.py $M $D runs/B4 --tail "$T" --b1 runs/B1/b1_summary.json --b2 runs/B2/b2_summary.json --primary anger > logs/B4.log 2>&1 \
  && test -f runs/B4/b4_summary.json || { log "B4 failed"; echo B4 > runs/CHAIN_FAILED; exit 1; }
log "B4 ok"; touch runs/CHAIN_DONE
