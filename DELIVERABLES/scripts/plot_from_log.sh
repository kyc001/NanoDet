#!/usr/bin/env bash
set -euo pipefail
conda activate nano
python nanodet-jittor/tools/parse_train_log_and_plot.py --log workspace/tiny20_overfit.txt --out_dir DELIVERABLES/images
cp -f workspace/full_val_final.txt DELIVERABLES/logs/full_val_final.txt 2>/dev/null || true
cp -f workspace/tiny20_overfit.txt DELIVERABLES/logs/tiny20_overfit.txt 2>/dev/null || true

