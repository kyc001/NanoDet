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

    # load checkpoint: support JT(.pkl) directly, or convert PT(.ckpt/.pth) on-the-fly
    ckpt_path = args.ckpt
    if ckpt_path.lower().endswith(('.pkl', '.jt', '.jittor')):
        ckpt = jt.load(ckpt_path)
        load_model_weight(model, ckpt, logger)
    else:
        import torch
        pt_ckpt = torch.load(ckpt_path, map_location='cpu')
        jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True)
        load_model_weight(model, jt_ckpt, logger)

    # build dataset and evaluator
    val_dataset = build_dataset(cfg.data.val, 'val')
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    print(f"Running mini-eval on {args.max_batches} batches...")

    results_all = {}
    batch_count = 0
    save_dir = os.path.join('workspace', 'mini_eval_from_pt')
    os.makedirs(save_dir, exist_ok=True)
    for batch in val_dataset:
        if batch_count >= args.max_batches:
            break

        meta = batch
        img = meta['img']
        # handle possible list/batched outputs from jittor Dataset
        if isinstance(img, list):
            img = img[0]
        # ensure jt.Var
        if not isinstance(img, jt.Var):
            img = jt.array(img)
        # if already batched (N,C,H,W), use directly; if CHW, add batch dim
        if len(img.shape) == 3:
            img_b = img.unsqueeze(0)
        elif len(img.shape) == 4:
            img_b = img
        else:
            raise ValueError(f"Unexpected img shape: {img.shape}")

        with jt.no_grad():
            preds = model(img_b)

        # Convert to evaluation format; ensure meta fields are python lists/ints, not jt.Var
        meta_eval = dict(meta)
        # ensure batched img inside meta for correct input size
        meta_eval['img'] = img_b
        # normalize meta content types
        if 'warp_matrix' in meta_eval and not isinstance(meta_eval['warp_matrix'], list):
            meta_eval['warp_matrix'] = [meta_eval['warp_matrix']]
        for subk in ['height','width','id']:
            v = meta_eval['img_info'].get(subk, 0)
            if isinstance(v, jt.Var):
                try:
                    v = int(v.item())
                except Exception:
                    v = int(np.array(v)[0])
            if not isinstance(v, (list, tuple, np.ndarray)):
                meta_eval['img_info'][subk] = [v]
        results = model.head.post_process(preds, meta_eval)
        # merge results
        for img_id, dets in results.items():
            # ensure python int as key
            if hasattr(img_id, 'item'):
                img_id = int(img_id.item())
            results_all[int(img_id)] = dets

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Processed {batch_count}/{args.max_batches} batches")

    # Get evaluation results
    eval_results = evaluator.evaluate(results_all, save_dir)
    print(f"\nMini-eval results ({batch_count} batches):")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
