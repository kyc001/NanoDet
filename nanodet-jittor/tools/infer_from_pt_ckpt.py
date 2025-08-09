# -*- coding: utf-8 -*-
# Load a PyTorch NanoDet ckpt and run inference with the Jittor model
import os
import sys
import argparse
import time
import cv2
import numpy as np
import jittor as jt

# Ensure we import nanodet from the Jittor repo, not the PyTorch one
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
PT_ROOT = os.path.abspath(os.path.join(JT_ROOT, "../nanodet-pytorch"))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)
# Remove pytorch repo path if present to avoid name collision
if PT_ROOT in sys.path:
    sys.path.remove(PT_ROOT)

# torch is needed just to load the state_dict
import torch

from nanodet.util import load_config, cfg, Logger
from nanodet.model.arch import build_model
from nanodet.data.transform.pipeline import Pipeline
from nanodet.util.check_point import load_model_weight
from collections import OrderedDict


def build_logger():
    # Logger signature in Jittor version: Logger(save_dir="./", name="NanoDet")
    logger = Logger(save_dir="./", name="pt2jt_infer")
    return logger


def pt_state_to_jt_checkpoint(pt_ckpt, model=None, prefer_avg=True):
    """
    Convert a PyTorch checkpoint dict to a Jittor-style checkpoint dict that
    load_model_weight can consume. Prefer avg_model.* weights if present, and
    map DWConv depthwise weights from (out,1,kh,kw) -> (out,out,kh,kw) for Jittor.
    """
    if isinstance(pt_ckpt, str):
        pt_ckpt = torch.load(pt_ckpt, map_location="cpu")
    state_dict = pt_ckpt.get("state_dict", pt_ckpt)

    # if avg_model.* present, use it exclusively
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
                continue
            tgt = model_sd[k]
            tshape = tuple(tgt.shape)
            vshape = tuple(v_np.shape)
            if vshape == tshape:
                reconciled[k] = v_np
                continue
            # handle depthwise/group conv mapping: pt (Cout,1,kh,kw) -> jt (Cout,Cin,kh,kw)
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
                    continue
            # expand scalar -> vector
            if v_np.ndim == 0 and len(tshape) == 1 and tshape[0] == 1:
                reconciled[k] = np.array([v_np.item()], dtype=np.float32)
                continue
            # otherwise skip
        proc_sd = reconciled

    jt_ckpt = {"state_dict": proc_sd}
    return jt_ckpt


def prepare_meta(img_path, pipeline, input_size):
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    img_info = {
        "file_name": os.path.basename(img_path),
        "height": img.shape[0],
        "width": img.shape[1],
        "id": 0,
    }
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, input_size)
    # HWC -> NCHW, to jt.Var
    im = meta["img"].transpose(2, 0, 1)  # CHW
    im = np.ascontiguousarray(im)
    meta["img"] = jt.array(im).unsqueeze(0)  # 1,C,H,W

    # Ensure list-like fields for post_process
    if not isinstance(meta.get("warp_matrix"), list):
        meta["warp_matrix"] = [meta["warp_matrix"]]
    if "img_info" in meta:
        for key in ("height", "width", "id"):
            val = meta["img_info"].get(key, 0)
            if not isinstance(val, (list, tuple, np.ndarray)):
                meta["img_info"][key] = [val]
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="path to YAML config (PyTorch config is fine)")
    parser.add_argument("--ckpt", required=True, help="path to PyTorch ckpt, e.g., model_last.ckpt")
    parser.add_argument("--img", required=True, help="path to an image for inference")
    parser.add_argument("--score_thr", type=float, default=0.35)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # enable cuda if available
    jt.flags.use_cuda = 1 if args.device.startswith("cuda") else 0

    logger = build_logger()

    # Load config and build Jittor model
    load_config(cfg, args.cfg)
    model = build_model(cfg.model)
    model.eval()

    # Load PyTorch checkpoint into Jittor model
    pt_ckpt = torch.load(args.ckpt, map_location="cpu")
    jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True)
    load_model_weight(model, jt_ckpt, logger)

    # Build val pipeline
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    input_size = cfg.data.val.input_size

    # Prepare meta and run inference
    meta = prepare_meta(args.img, pipeline, input_size)

    jt.sync_all(True)
    t1 = time.time()
    results = model.inference(meta)
    jt.sync_all(True)
    dt = time.time() - t1

    print(f"Inference done in {dt:.3f}s")
    # results could be dict: {img_id: {class_idx: [[x1,y1,x2,y2,score], ...]}}
    if isinstance(results, dict) and len(results):
        img_id = list(results.keys())[0]
        dets = results[img_id]
        # count boxes above threshold
        total = 0
        for cls_idx, arr in dets.items():
            arr = np.array(arr, dtype=np.float32)
            if arr.size == 0:
                continue
            keep = arr[:, 4] >= args.score_thr
            total += int(keep.sum())
        print(f"Detections >= {args.score_thr}: {total}")
        # save visualization
        out_dir = os.path.join(os.getcwd(), "result", "infer")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "pt2jt_vis.jpg")
        try:
            model.head.show_result(meta["raw_img"], dets, cfg.class_names, score_thres=args.score_thr, show=False, save_path=save_path)
            print(f"Saved visualization to: {save_path}")
        except Exception as e:
            print("Visualization failed:", e)
    else:
        print("No detections returned.")


if __name__ == "__main__":
    main()

