#!/bin/bash
# Waits for the B2-B4 chain to finish (done or failed) and for the prompt set, then trains the pre-readout probe.
cd /root/autodl-tmp/E01
log() { echo "$(date '+%F %T') $*" >> logs/chain.log; }
while [ ! -f runs/CHAIN_DONE ] && [ ! -f runs/CHAIN_FAILED ]; do sleep 60; done
while [ ! -f data/prompts_pre.jsonl ]; do sleep 30; done
log "prereadout start (chain: $( [ -f runs/CHAIN_DONE ] && echo done || echo "failed at $(cat runs/CHAIN_FAILED)"))"
mkdir -p runs/pre
/root/miniconda3/bin/python data/prereadout.py models/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master data/prompts_pre.jsonl runs/readout/probe_L19.npz runs/pre > logs/pre.log 2>&1 \
  && { log "prereadout ok"; touch runs/PRE_DONE; } || { log "prereadout failed"; touch runs/PRE_FAILED; }
