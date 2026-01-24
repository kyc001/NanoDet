#!/usr/bin/env python3
"""
Convert Jittor checkpoint to PyTorch checkpoint.
"""

import argparse
import os
import sys

import numpy as np
import torch


def _strip_prefix_key(k: str) -> str:
    if k.startswith("module."):
        k = k[7:]
    if k.startswith("model."):
        k = k[6:]
    if k.startswith("avg_model."):
        k = k[10:]
    return k


def _to_torch(v):
    if isinstance(v, torch.Tensor):
        return v
    if hasattr(v, "numpy"):
        v = v.numpy()
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    return torch.tensor(v)


def load_jittor_checkpoint(path: str):
    try:
        import jittor as jt
    except Exception as e:
        raise RuntimeError("Jittor is required to load .pkl checkpoint") from e
    return jt.load(path)


def convert(jt_ckpt_path: str, pt_ckpt_path: str):
    if not os.path.exists(jt_ckpt_path):
        raise FileNotFoundError(f"Jittor checkpoint not found: {jt_ckpt_path}")

    ckpt = load_jittor_checkpoint(jt_ckpt_path)
    state_dict = ckpt.get("state_dict", ckpt)

    converted = {}
    for k, v in state_dict.items():
        nk = _strip_prefix_key(k)
        converted[nk] = _to_torch(v)

    out = {
        "state_dict": converted,
        "meta": {
            "converted_from": "Jittor",
            "original_path": jt_ckpt_path,
        },
    }

    os.makedirs(os.path.dirname(pt_ckpt_path), exist_ok=True)
    torch.save(out, pt_ckpt_path)
    return pt_ckpt_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jt_ckpt",
        default="workspace/jittor_model_best.pkl",
        help="Input Jittor checkpoint (.pkl)",
    )
    ap.add_argument(
        "--pt_ckpt",
        default="workspace/jt2pt_model_best.pth",
        help="Output PyTorch checkpoint (.pth)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        out_path = convert(args.jt_ckpt, args.pt_ckpt)
        print(f"Converted to PyTorch: {out_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)
