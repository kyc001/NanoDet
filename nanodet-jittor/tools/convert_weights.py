#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bidirectional weight conversion for NanoDet-Plus (PyTorch <-> Jittor).
- pt2jt: Convert PyTorch checkpoint to Jittor checkpoint with DW mapping.
- jt2pt: Convert Jittor checkpoint to PyTorch checkpoint with reverse DW mapping.

Usage examples:
  PT -> JT:
    python nanodet-jittor/tools/convert_weights.py \
      --mode pt2jt \
      --pt_ckpt nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64_50epochs/model_last.ckpt \
      --jt_config nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml \
      --output_jt_ckpt workspace/nanodet_plus_m_jittor_pretrained.pkl

  JT -> PT:
    python nanodet-jittor/tools/convert_weights.py \
      --mode jt2pt \
      --jt_ckpt workspace/nanodet_plus_m_jittor_pretrained.pkl \
      --output_pt_ckpt workspace/nanodet_plus_m_pytorch_from_jittor.ckpt
"""

import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import jittor as jt
import torch

# Ensure nanodet-jittor on path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)

from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model
from nanodet.util.check_point import load_model_weight, save_model


def _as_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)


def pt_state_to_jt_dict(pt_ckpt, model=None, prefer_avg=True):
    """Convert PT checkpoint (dict or path) to JT state_dict (numpy arrays).
    Optionally reconcile shapes using a Jittor model instance.
    """
    if isinstance(pt_ckpt, str):
        pt_ckpt = torch.load(pt_ckpt, map_location="cpu")
    state_dict = pt_ckpt.get("state_dict", pt_ckpt)

    # Prefer EMA weights if present
    if prefer_avg and any(k.startswith("avg_model.") for k in state_dict.keys()):
        state_dict = OrderedDict(
            (k[len("avg_model."):], v) for k, v in state_dict.items() if k.startswith("avg_model.")
        )

    # Strip common wrappers
    proc_sd = {}
    for k, v in state_dict.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[7:]
        if k2.startswith("model."):
            k2 = k2[6:]
        proc_sd[k2] = _as_numpy(v)

    if model is None:
        return proc_sd

    # Reconcile shapes against JT model
    model_sd = model.state_dict()
    reconciled = {}
    for k, v_np in proc_sd.items():
        if k not in model_sd:
            continue
        tgt = model_sd[k]
        tshape = tuple(tgt.shape)
        vshape = tuple(v_np.shape)
        if vshape == tshape:
            reconciled[k] = v_np
            continue
        # Map DW conv weights: PT (Cout,1,kh,kw) -> JT (Cout,Cin,kh,kw)
        if v_np.ndim == 4 and len(tshape) == 4 and v_np.shape[0] == tshape[0]:
            if v_np.shape[1] == 1:
                cout_t, cin_t, kh, kw = tshape
                neww = np.zeros(tshape, dtype=np.float32)
                for oc in range(cout_t):
                    ic = oc % cin_t
                    neww[oc, ic, :, :] = v_np[oc, 0, :, :]
                reconciled[k] = neww
                continue
        # Scalar -> vector expand (e.g., scales)
        if v_np.ndim == 0 and len(tshape) == 1 and tshape[0] == 1:
            reconciled[k] = np.array([v_np.item()], dtype=np.float32)
            continue
        # Fallback: skip mismatched
    return reconciled


def jt_state_to_pt_dict(jt_ckpt):
    """Convert JT checkpoint (dict or path) to PT state_dict (torch tensors).
    Reverse DW mapping: JT (Cout,Cin,kh,kw) -> PT (Cout,1,kh,kw) by diagonal pick.
    """
    if isinstance(jt_ckpt, str):
        jt_ckpt = jt.load(jt_ckpt)
    state_dict = jt_ckpt.get("state_dict", jt_ckpt)

    pt_sd = OrderedDict()
    for k, v in state_dict.items():
        v_np = np.array(v)
        # Reverse DW conv weight mapping if matches 4D conv with Cin>=1
        if v_np.ndim == 4 and v_np.shape[1] >= 1:
            cout, cin, kh, kw = v_np.shape
            # Heuristic: depthwise/group conv weights usually have cin equal to in_channels or groups
            # Collapse input channel dim by selecting diagonal index (oc % cin)
            if cin > 1:
                neww = np.zeros((cout, 1, kh, kw), dtype=np.float32)
                for oc in range(cout):
                    ic = oc % cin
                    neww[oc, 0, :, :] = v_np[oc, ic, :, :]
                pt_sd[k] = torch.from_numpy(neww)
                continue
            else:
                # Already (Cout,1,kh,kw)
                pt_sd[k] = torch.from_numpy(v_np.astype(np.float32))
                continue
        # Other params
        pt_sd[k] = torch.from_numpy(v_np.astype(np.float32)) if isinstance(v_np, np.ndarray) else torch.tensor(v_np)
    return pt_sd


def convert_pt2jt(pt_ckpt_path, jt_config_path, output_jt_ckpt):
    print(f"[convert] Loading JT config: {jt_config_path}")
    load_config(cfg, jt_config_path)
    print("[convert] Building JT model for shape reconciliation...")
    model = build_model(cfg.model)

    print(f"[convert] Loading PT checkpoint: {pt_ckpt_path}")
    jt_sd = pt_state_to_jt_dict(pt_ckpt_path, model=model, prefer_avg=True)

    print("[convert] Loading converted weights into JT model to validate...")
    ckpt = {"state_dict": jt_sd}
    load_model_weight(model, ckpt, logger=None)

    print(f"[convert] Saving JT checkpoint to: {output_jt_ckpt}")
    os.makedirs(os.path.dirname(output_jt_ckpt), exist_ok=True)
    # Save in standard JT checkpoint format
    save_model(model, output_jt_ckpt, epoch=0, iter=0, optimizer=None)
    print("[convert] PT -> JT conversion done.")
    return output_jt_ckpt


def convert_jt2pt(jt_ckpt_path, output_pt_ckpt):
    print(f"[convert] Loading JT checkpoint: {jt_ckpt_path}")
    pt_sd = jt_state_to_pt_dict(jt_ckpt_path)
    # Wrap into PT-like checkpoint dict
    pt_ckpt = {"state_dict": pt_sd}
    os.makedirs(os.path.dirname(output_pt_ckpt), exist_ok=True)
    torch.save(pt_ckpt, output_pt_ckpt)
    print(f"[convert] JT -> PT checkpoint saved to: {output_pt_ckpt}")
    return output_pt_ckpt


def main():
    parser = argparse.ArgumentParser(description="Bidirectional weight conversion (PT <-> JT)")
    parser.add_argument('--mode', choices=['pt2jt', 'jt2pt'], required=True)
    parser.add_argument('--pt_ckpt', type=str, help='Path to PyTorch checkpoint (.ckpt/.pth)')
    parser.add_argument('--jt_config', type=str, help='Path to Jittor config (.yml) for building JT model (required for pt2jt)')
    parser.add_argument('--jt_ckpt', type=str, help='Path to Jittor checkpoint (.pkl/.pkl2)')
    parser.add_argument('--output_jt_ckpt', type=str, help='Output path for converted Jittor checkpoint')
    parser.add_argument('--output_pt_ckpt', type=str, help='Output path for converted PyTorch checkpoint')
    args = parser.parse_args()

    if args.mode == 'pt2jt':
        if not args.pt_ckpt or not args.jt_config or not args.output_jt_ckpt:
            raise SystemExit("pt2jt requires --pt_ckpt, --jt_config, --output_jt_ckpt")
        if torch.cuda.is_available():
            jt.flags.use_cuda = 1
        else:
            jt.flags.use_cuda = 0
        convert_pt2jt(args.pt_ckpt, args.jt_config, args.output_jt_ckpt)
    else:  # jt2pt
        if not args.jt_ckpt or not args.output_pt_ckpt:
            raise SystemExit("jt2pt requires --jt_ckpt, --output_pt_ckpt")
        convert_jt2pt(args.jt_ckpt, args.output_pt_ckpt)


if __name__ == '__main__':
    main()

