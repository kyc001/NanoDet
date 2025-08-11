#!/usr/bin/env bash
set -euo pipefail
conda activate nano
python nanodet-jittor/tools/train.py nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml --max_epochs=1 --max_train_batches=1 --warmup_steps=0 | tee workspace/full_val_final.txt

