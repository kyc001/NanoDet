#!/usr/bin/env python3
"""
NanoDet-Plus PyTorch to Jittor Weight Conversion Script
Converts PyTorch checkpoint to Jittor format with optimized depthwise convolution mapping.
"""

import os
import sys
import argparse
import torch
import numpy as np
from collections import OrderedDict

# Add nanodet-jittor to path
sys.path.insert(0, 'nanodet-jittor')

import jittor as jt
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model
from nanodet.util.check_point import load_model_weight, save_model

def pt_state_to_jt_checkpoint(pt_ckpt, model=None, prefer_avg=True):
    """
    Convert a PyTorch checkpoint dict to a Jittor-style checkpoint dict.
    Optimized for NanoDet-Plus with Ghost depthwise convolution mapping.
    """
    if isinstance(pt_ckpt, str):
        pt_ckpt = torch.load(pt_ckpt, map_location="cpu")
    state_dict = pt_ckpt.get("state_dict", pt_ckpt)

    # Prefer avg_model.* weights if present (EMA weights)
    if prefer_avg and any(k.startswith("avg_model.") for k in state_dict.keys()):
        state_dict = OrderedDict((k[len("avg_model."):], v) for k, v in state_dict.items() if k.startswith("avg_model."))

    # Strip wrappers
    proc_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        v_np = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
        proc_sd[k] = v_np

    # If a model is provided, reconcile shapes exactly to its parameters
    if model is not None:
        model_sd = model.state_dict()
        reconciled = {}
        for k, v_np in proc_sd.items():
            if k not in model_sd:
                print(f"[WARNING] Key {k} not found in Jittor model, skipping...")
                continue
            tgt = model_sd[k]
            tshape = tuple(tgt.shape)
            vshape = tuple(v_np.shape)
            if vshape == tshape:
                reconciled[k] = v_np
                continue
            
            # Handle depthwise/group conv mapping: pt (Cout,1,kh,kw) -> jt (Cout,Cin,kh,kw)
            if v_np.ndim == 4 and len(tshape) == 4 and v_np.shape[0] == tshape[0]:
                if v_np.shape[1] == 1:
                    cout_t, cin_t, kh, kw = tshape
                    # Ghost-aware mapping: for Ghost cheap_operation, map based on actual topology
                    if 'cheap_operation' in k and cin_t > 1:
                        # Ghost cheap DW: init_channels -> new_channels with groups=init_channels
                        # Each output channel maps to its corresponding input channel (1:1 for true DW)
                        neww = np.zeros(tshape, dtype=np.float32)
                        for oc in range(min(cout_t, cin_t)):
                            neww[oc, oc, :, :] = v_np[oc, 0, :, :]
                        # For extra output channels beyond cin_t, use modulo mapping
                        for oc in range(cin_t, cout_t):
                            ic = oc % cin_t
                            neww[oc, ic, :, :] = v_np[oc, 0, :, :]
                    else:
                        # Standard channel-multiplier mapping for other DW convs
                        neww = np.zeros(tshape, dtype=np.float32)
                        for oc in range(cout_t):
                            ic = oc % cin_t
                            neww[oc, ic, :, :] = v_np[oc, 0, :, :]
                    reconciled[k] = neww
                    print(f"[INFO] Mapped depthwise conv {k}: {vshape} -> {tshape}")
                    continue
            
            # Expand scalar -> vector
            if v_np.ndim == 0 and len(tshape) == 1 and tshape[0] == 1:
                reconciled[k] = np.array([v_np.item()], dtype=np.float32)
                continue
            
            print(f"[WARNING] Shape mismatch for {k}: PT {vshape} vs JT {tshape}, skipping...")
        
        return reconciled
    else:
        return proc_sd

def convert_weights(pt_ckpt_path, jt_config_path, output_path):
    """Convert PyTorch weights to Jittor format."""
    print(f"[INFO] Loading config from {jt_config_path}")
    load_config(cfg, jt_config_path)
    
    print(f"[INFO] Building Jittor model...")
    model = build_model(cfg.model)
    
    print(f"[INFO] Converting weights from {pt_ckpt_path}")
    jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt_path, model)
    
    print(f"[INFO] Loading converted weights to model...")
    # Create Jittor checkpoint format
    jt_checkpoint = {
        'state_dict': jt_ckpt,
        'epoch': 0,
        'save_dir': os.path.dirname(output_path)
    }
    load_model_weight(model, jt_checkpoint, logger=None)

    print(f"[INFO] Saving Jittor checkpoint to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_model(model, output_path)
    
    print(f"[INFO] Conversion completed successfully!")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch NanoDet weights to Jittor format')
    parser.add_argument('--pt_ckpt', required=True, help='PyTorch checkpoint path')
    parser.add_argument('--jt_config', required=True, help='Jittor config file path')
    parser.add_argument('--output', required=True, help='Output Jittor checkpoint path')
    
    args = parser.parse_args()
    
    # Set Jittor to use CUDA if available
    if torch.cuda.is_available():
        jt.flags.use_cuda = 1
        print("[INFO] Using CUDA for Jittor")
    else:
        jt.flags.use_cuda = 0
        print("[INFO] Using CPU for Jittor")
    
    convert_weights(args.pt_ckpt, args.jt_config, args.output)

if __name__ == '__main__':
    main()
