#!/bin/bash
cd /root/autodl-tmp/E01; touch runs/BUSY_ablate; trap 'rm -f runs/BUSY_ablate' EXIT
echo "$(date '+%F %T') ablate start" >> logs/chain.log; mkdir -p runs/ablate
/root/miniconda3/bin/python data/ablate.py models/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master data/formal_960.jsonl runs/readout/probe_L19.npz runs/ablate --tail "我的看法是：" > logs/ablate.log 2>&1
echo "$(date '+%F %T') ablate exit $?" >> logs/chain.log
