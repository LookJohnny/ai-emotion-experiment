#!/bin/bash
# After the first census finishes, rerun it on paraphrase B only (leads without emotion-category words).
cd /root/autodl-tmp/E01
while pgrep -f "data/neuron_census.py .* runs/census " >/dev/null || [ ! -f runs/census/census.json ]; do
  pgrep -f "data/neuron_census.py" >/dev/null || break; sleep 30; done
echo "$(date '+%F %T') censusB start" >> logs/chain.log; mkdir -p runs/census_B
/root/miniconda3/bin/python data/neuron_census.py models/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master data/formal_960.jsonl data/prompts_pre.jsonl runs/census_B --tail "我的看法是：" --para B > logs/census_B.log 2>&1
echo "$(date '+%F %T') censusB exit $?" >> logs/chain.log; touch runs/CENSUSB_DONE
