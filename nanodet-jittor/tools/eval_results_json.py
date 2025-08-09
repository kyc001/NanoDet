# -*- coding: utf-8 -*-
# Evaluate an existing COCO-format detection json against the val set defined in cfg
import os
import sys
import argparse
import copy
import io
import contextlib

import jittor as jt
from pycocotools.cocoeval import COCOeval

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)

from nanodet.util import load_config, cfg, Logger
from nanodet.data.dataset import build_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--json", required=True, help="path to COCO-format detections json")
    ap.add_argument("--save_dir", default="result/pytorch_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--auto_remap_cat", action="store_true", help="auto remap dt category_id to match gt if off-by-one (VOC->COCO)")
    return ap.parse_args()


def main():
    args = parse_args()
    jt.flags.use_cuda = 1 if args.device.startswith("cuda") else 0

    load_config(cfg, args.cfg)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(save_dir=args.save_dir, name="eval_json")

    # dataset with COCO API
    val_dataset = build_dataset(cfg.data.val, mode="val")
    coco_gt = copy.deepcopy(val_dataset.coco_api)

    # load detections json
    json_path = args.json
    assert os.path.isfile(json_path), f"json not found: {json_path}"
    # optionally auto-remap category ids (common VOC->COCO off-by-one)
    if args.auto_remap_cat:
        import json
        with open(json_path,'r') as f:
            data=json.load(f)
        gt_ids=sorted([c['id'] for c in coco_gt.dataset['categories']])
        dt_ids=sorted(list({d['category_id'] for d in data}))
        if min(gt_ids)==0 and min(dt_ids)==1 and len(gt_ids)==len(dt_ids):
            for d in data:
                d['category_id']-=1
            remap_path=os.path.join(args.save_dir,'remapped.json')
            with open(remap_path,'w') as f:
                json.dump(data,f)
            json_path=remap_path
            logger.info(f"Auto-remapped category_id by -1 and saved to {remap_path}")

    coco_dt = coco_gt.loadRes(json_path)

    # run COCOeval
    coco_eval = COCOeval(copy.deepcopy(coco_gt), copy.deepcopy(coco_dt), 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_eval.summarize()
    logger.info("\n" + redirect_string.getvalue())


if __name__ == "__main__":
    main()

