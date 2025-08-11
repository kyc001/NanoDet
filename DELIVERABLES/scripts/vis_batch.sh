#!/usr/bin/env bash
set -euo pipefail
conda activate nano
OUT=DELIVERABLES/images/sample_dets
mkdir -p "$OUT" DELIVERABLES/logs
python nanodet-jittor/tools/vis_batch.py \
  --config nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml \
  --ckpt workspace/nanodet-plus-m_320_voc_bs64_50epochs/model_best/model_best.ckpt \
  --out_dir "$OUT" \
  --num 32 \
  --score_thr 0.35 | tee DELIVERABLES/logs/vis_batch.txt

