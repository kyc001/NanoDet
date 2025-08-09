# -*- coding: utf-8 -*-
"""
Dump Jittor NanoDet head weights and BN-like stats to JSON for PT/JT comparison.
"""
import os, sys, json, argparse
import numpy as np
import jittor as jt
import jittor.nn as nn

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)

from nanodet.util import load_config, cfg, Logger
from nanodet.model.arch import build_model
from tools.infer_from_pt_ckpt import pt_state_to_jt_checkpoint
from nanodet.util.check_point import load_model_weight


def var_stats(v):
    if v is None:
        return None
    arr = v.numpy().astype(np.float32)
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()) if arr.size>0 else 0.0,
        "std": float(arr.std()) if arr.size>1 else 0.0,
        "min": float(arr.min()) if arr.size>0 else 0.0,
        "max": float(arr.max()) if arr.size>0 else 0.0,
        "nonzero": int((arr!=0).sum()),
        "numel": int(arr.size),
        "dtype": str(v.dtype),
    }


def dump_head_stats(model):
    head = model.head
    out = {"gfl_cls": {}, "bn": {}, "misc": {}}
    # gfl_cls conv weights
    if hasattr(head, "gfl_cls"):
        for i, layer in enumerate(head.gfl_cls):
            w = getattr(layer, 'weight', None)
            b = getattr(layer, 'bias', None)
            out["gfl_cls"][f"{i}.weight"] = var_stats(w)
            out["gfl_cls"][f"{i}.bias"] = var_stats(b)
    # BN-like layers (Jittor BatchNorm2d)
    bn_index = 0
    for name, m in head.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            out["bn"][f"{bn_index}:{name}.weight"] = var_stats(getattr(m, 'weight', None))
            out["bn"][f"{bn_index}:{name}.bias"] = var_stats(getattr(m, 'bias', None))
            out["bn"][f"{bn_index}:{name}.running_mean"] = var_stats(getattr(m, 'running_mean', None))
            out["bn"][f"{bn_index}:{name}.running_var"] = var_stats(getattr(m, 'running_var', None))
            out["bn"][f"{bn_index}:{name}.hyper"] = {
                "eps": float(getattr(m, 'eps', 1e-5)),
                "momentum": float(getattr(m, 'momentum', 0.1)),
                "affine": bool(getattr(m, 'affine', True)),
                "track_running_stats": bool(getattr(m, 'track_running_stats', True)),
            }
            bn_index += 1
    # misc
    out["misc"]["strides"] = getattr(head, 'strides', None)
    out["misc"]["reg_max"] = getattr(head, 'reg_max', None)
    out["misc"]["num_classes"] = getattr(head, 'num_classes', None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True, help='PyTorch checkpoint path')
    ap.add_argument('--out', default='workspace/jt_head_stats.json')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    jt.flags.use_cuda = 1 if args.device.startswith('cuda') else 0
    logger = Logger(save_dir=os.path.dirname(args.out) or '.')
    load_config(cfg, args.cfg)

    model = build_model(cfg.model)
    model.eval()

    # load PT ckpt into JT
    import torch
    pt_ckpt = torch.load(args.ckpt, map_location='cpu')
    jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True)
    load_model_weight(model, jt_ckpt, logger)

    stats = dump_head_stats(model)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"saved JT head stats to {args.out}")


if __name__ == '__main__':
    main()

