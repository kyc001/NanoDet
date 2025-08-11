#!/usr/bin/env bash
set -euo pipefail
conda activate nano
mkdir -p workspace
python - <<'PY'
import json, os
os.makedirs('data/annotations', exist_ok=True)
SRC='data/annotations/voc_train.json'
DST='data/annotations/voc_train_tiny20.json'
if not os.path.exists(DST):
    with open(SRC,'r') as f: coco=json.load(f)
    ids=sorted([img['id'] for img in coco['images']])[:20]
    keep=set(ids)
    images=[img for img in coco['images'] if img['id'] in keep]
    anns=[ann for ann in coco['annotations'] if ann['image_id'] in keep]
    new={'images':images,'annotations':anns,'categories':coco['categories']}
    with open(DST,'w') as f: json.dump(new,f)
    print('Created', DST, 'images', len(images), 'anns', len(anns))
else:
    print('Found existing', DST)
PY
python nanodet-jittor/tools/train.py nanodet-jittor/config/nanodet-plus-m_320_voc_tiny20.yml --max_epochs=3 --warmup_steps=0 | tee workspace/tiny20_overfit.txt

