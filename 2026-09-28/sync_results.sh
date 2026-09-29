#!/bin/bash
# Pull B1-B4 + readout results from the GPU server into the repo, then commit & push.
set -e
PORT=${PORT:-24140}; H="-o ConnectTimeout=20 -P $PORT root@connect.bjb2.seetacloud.com"
R=/root/autodl-tmp/E01; L=/Users/lovelyjoy/Documents/AI-Emotion-Experiment/2026-09-28
mkdir -p $L/stageB/results $L/product/results
for s in B1 B2 B3 B4; do
  mkdir -p $L/stageB/results/$s
  scp -q $H:"$R/runs/$s/*" $L/stageB/results/$s/ 2>/dev/null || true
  scp -q $H:"$R/logs/$s.log" $L/stageB/results/$s/ 2>/dev/null || true
done
scp -q $H:"$R/logs/chain.log" $L/stageB/results/ 2>/dev/null || true
scp -q $H:"$R/runs/readout/*" $L/product/results/ 2>/dev/null || true
mkdir -p $L/product/results/pre $L/stageB/results/census
scp -q $H:"$R/runs/pre/*" $L/product/results/pre/ 2>/dev/null || true
scp -q $H:"$R/logs/pre.log" $L/product/results/pre/ 2>/dev/null || true
scp -q $H:"$R/runs/census/*" $L/stageB/results/census/ 2>/dev/null || true
scp -q $H:"$R/logs/census.log" $L/stageB/results/census/ 2>/dev/null || true
mkdir -p $L/stageB/results/census_B
scp -q $H:"$R/runs/census_B/*" $L/stageB/results/census_B/ 2>/dev/null || true
scp -q $H:"$R/logs/census_B.log" $L/stageB/results/census_B/ 2>/dev/null || true
mkdir -p $L/stageB/results/ablate
scp -q $H:"$R/runs/ablate/*" $L/stageB/results/ablate/ 2>/dev/null || true
scp -q $H:"$R/logs/ablate.log" $L/stageB/results/ablate/ 2>/dev/null || true
for f in run_censusB.sh auto_shutdown.sh run_ablate.sh; do scp -q $H:"$R/data/$f" $L/stageB/ 2>/dev/null || true; done
scp -q $H:"$R/data/run_pre.sh" $L/product/ 2>/dev/null || true
scp -q $H:"$R/data/run_chain.sh" $L/stageB/ 2>/dev/null || true
cd /Users/lovelyjoy/Documents/AI-Emotion-Experiment
find 2026-09-28/stageB/results 2026-09-28/product/results -type f -size +50M -print | sed 's/^/TOO BIG (not added): /'
git add 2026-09-28/stageB 2026-09-28/product 2026-09-28/sync_results.sh
git diff --cached --stat | tail -3
