# -*- coding: utf-8 -*-
"""
Dump PyTorch NanoDet head weights and BN stats to JSON for PT/JT comparison.
"""
import os, sys, json, argparse
import numpy as np
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PT_ROOT not in sys.path:
    sys.path.insert(0, PT_ROOT)

from nanodet.util import load_config, cfg
from nanodet.util.logger import Logger
from nanodet.model.arch import build_model


def tensor_stats(t: torch.Tensor):
    if t is None:
        return None
    arr = t.detach().cpu().float()
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean().item()) if arr.numel()>0 else 0.0,
        "std": float(arr.std().item()) if arr.numel()>1 else 0.0,
        "min": float(arr.min().item()) if arr.numel()>0 else 0.0,
        "max": float(arr.max().item()) if arr.numel()>0 else 0.0,
        "nonzero": int((arr!=0).sum().item()),
        "numel": int(arr.numel()),
        "dtype": str(t.dtype).replace('torch.', ''),
    }


def dump_head_stats(model):
    head = model.head
    out = {"gfl_cls": {}, "bn": {}, "misc": {}}
    # gfl_cls conv weights/biases
    if hasattr(head, "gfl_cls"):
        for i, layer in enumerate(head.gfl_cls):
            w = getattr(layer, 'weight', None)
            b = getattr(layer, 'bias', None)
            out["gfl_cls"][f"{i}.weight"] = tensor_stats(w)
            out["gfl_cls"][f"{i}.bias"] = tensor_stats(b)
    # traverse BN layers in head
    from torch import nn
    bn_index = 0
    for name, m in head.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            out["bn"][f"{bn_index}:{name}.weight"] = tensor_stats(m.weight)
            out["bn"][f"{bn_index}:{name}.bias"] = tensor_stats(m.bias)
            # buffers
            rm = getattr(m, 'running_mean', None)
            rv = getattr(m, 'running_var', None)
            out["bn"][f"{bn_index}:{name}.running_mean"] = tensor_stats(rm)
            out["bn"][f"{bn_index}:{name}.running_var"] = tensor_stats(rv)
            # eps/momentum
            out["bn"][f"{bn_index}:{name}.hyper"] = {
                "eps": float(m.eps),
                "momentum": float(0.0 if m.momentum is None else m.momentum),
                "affine": bool(m.affine),
                "track_running_stats": bool(m.track_running_stats),
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
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='workspace/pt_head_stats.json')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device if args.device.startswith('cuda') else 'cpu')
    logger = Logger(local_rank=0, save_dir=os.path.dirname(args.out) or '.')
    load_config(cfg, args.cfg)

    model = build_model(cfg.model)
    model.to(device)
    model.eval()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    if any(k.startswith('avg_model.') for k in sd):
        sd = {k[len('avg_model.'):]: v for k,v in sd.items() if k.startswith('avg_model.')}
    model.load_state_dict(sd, strict=False)

    stats = dump_head_stats(model)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"saved PT head stats to {args.out}")


if __name__ == '__main__':
    main()

