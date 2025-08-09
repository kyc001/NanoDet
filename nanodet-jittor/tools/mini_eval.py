# -*- coding: utf-8 -*-
"""
Mini evaluation script: run inference on a small subset of validation data
to quickly check mAP improvement after alignment fixes.
"""
import os, sys, argparse
import numpy as np
import jittor as jt
from jittor import nn

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)

from nanodet.util import load_config, cfg, Logger
from nanodet.model.arch import build_model
from tools.infer_from_pt_ckpt import pt_state_to_jt_checkpoint
from nanodet.util.check_point import load_model_weight
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True, help='PyTorch checkpoint path')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max_batches', type=int, default=50, help='Max batches to evaluate (for quick test)')
    args = ap.parse_args()

    jt.flags.use_cuda = 1 if args.device.startswith('cuda') else 0
    logger = Logger(save_dir='workspace')
    load_config(cfg, args.cfg)

    model = build_model(cfg.model)
    model.eval()

    # load PT ckpt into JT
    import torch
    pt_ckpt = torch.load(args.ckpt, map_location='cpu')
    jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True)
    load_model_weight(model, jt_ckpt, logger)

    # build dataset and evaluator
    val_dataset = build_dataset(cfg.data.val, 'val')
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    print(f"Running mini-eval on {args.max_batches} batches...")
    
    batch_count = 0
    for batch in val_dataset:
        if batch_count >= args.max_batches:
            break
        
        meta = batch
        img = meta['img']
        if isinstance(img, list):
            img = img[0]
        
        with jt.no_grad():
            preds = model(img.unsqueeze(0))
            
        # Convert to evaluation format
        results = model.head.post_process(preds, meta)
        evaluator.update(results, meta)
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processed {batch_count}/{args.max_batches} batches")
    
    # Get evaluation results
    eval_results = evaluator.evaluate()
    print(f"\nMini-eval results ({batch_count} batches):")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
