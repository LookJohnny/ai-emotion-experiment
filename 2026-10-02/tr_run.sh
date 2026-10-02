#!/bin/sh
# Teacher-forced features of one model on every dataset of PREREG_TRANSFER.md.
#   sh tr_run.sh NAME MODEL_DIR N_LAYERS   (reference: sh tr_run.sh ref <qwen3-4b> ref)
# Layers: native candidates at relative depths {.30,.40,.45,.50,.55,.60,.70} + the align layers round(ref_layer/36*N) for 16,19,20.
set -e
NAME=$1; M=$2; N=$3; P=/root/miniconda3/bin/python; D=/root/autodl-tmp/E01/tr; cd $D
if [ "$N" = "ref" ]; then LAY="16,19,20"; else
  LAY=$($P -c "import sys;N=int(sys.argv[1]);s=sorted({round(f*N) for f in (.30,.40,.45,.50,.55,.60,.70)}|{round(l/36*N) for l in (16,19,20)});print(','.join(map(str,s)))" $N); fi
echo "$NAME layers $LAY"; mkdir -p feats/$NAME
f() { $P gpu_feats.py $M data/$1.jsonl feats/$NAME/$1.npz --reply_field $2 --max_reply $3 --layers $LAY; }
f c1_dev target_text 120; f c1_test target_text 120; f c1_kimi content 120
f sy_dev target_text 120; f sy_test target_text 120; f sy_kimi content 120
f fa_dev target_text 64;  f fa_test target_text 64;  f fa_kimi content 64
echo "ALLDONE $NAME"
