#!/usr/bin/env python3
"""
Round-trip check: PyTorch -> Jittor -> PyTorch
Outputs per-parameter diff stats and a JSON report.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append("nanodet-jittor")

import jittor as jt
from nanodet.model.arch import build_model
from nanodet.util.check_point import pt_to_jt_checkpoint
from nanodet.util.config import load_config, cfg


def _strip_prefix_key(k: str) -> str:
    if k.startswith("avg_model."):
        k = k[10:]
    if k.startswith("module."):
        k = k[7:]
    if k.startswith("model."):
        k = k[6:]
    return k


def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, "numpy"):
        return v.numpy()
    return np.array(v)


def _to_torch(v):
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    if hasattr(v, "numpy"):
        return torch.from_numpy(v.numpy())
    return torch.tensor(v)


def load_pt_state_dict(pt_ckpt_path: str):
    ckpt = torch.load(pt_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[_strip_prefix_key(k)] = v
    return cleaned


def convert_pt_to_jt(pt_ckpt_path: str, jt_config: str, jt_ckpt_out: str):
    load_config(cfg, jt_config)
    model = build_model(cfg.model)
    pt_ckpt = torch.load(pt_ckpt_path, map_location="cpu")
    jt_ckpt = pt_to_jt_checkpoint(pt_ckpt, model)
    out = {
        "state_dict": jt_ckpt["state_dict"] if "state_dict" in jt_ckpt else jt_ckpt,
        "meta": {
            "converted_from": "PyTorch",
            "original_path": pt_ckpt_path,
            "config": jt_config,
        },
    }
    os.makedirs(os.path.dirname(jt_ckpt_out), exist_ok=True)
    jt.save(out, jt_ckpt_out)
    return jt_ckpt_out


def load_jt_state_dict(jt_ckpt_path: str):
    ckpt = jt.load(jt_ckpt_path)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[_strip_prefix_key(k)] = v
    return cleaned


def save_pt_checkpoint(state_dict, pt_ckpt_out: str, meta=None):
    out = {"state_dict": state_dict}
    if meta:
        out["meta"] = meta
    os.makedirs(os.path.dirname(pt_ckpt_out), exist_ok=True)
    torch.save(out, pt_ckpt_out)


def diff_state_dicts(sd_a, sd_b, topk=20):
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    total_elems = 0
    sum_abs = 0.0
    max_abs = 0.0
    max_key = None
    per_key = []
    shape_mismatch = []

    for k in common:
        a = _to_numpy(sd_a[k])
        b = _to_numpy(sd_b[k])
        if a.shape != b.shape:
            shape_mismatch.append({"key": k, "shape_a": list(a.shape), "shape_b": list(b.shape)})
            continue
        diff = np.abs(a - b)
        mean_abs = float(diff.mean())
        max_abs_k = float(diff.max())
        per_key.append({"key": k, "mean_abs": mean_abs, "max_abs": max_abs_k, "shape": list(a.shape)})
        total_elems += diff.size
        sum_abs += diff.sum()
        if max_abs_k > max_abs:
            max_abs = max_abs_k
            max_key = k

    per_key_sorted = sorted(per_key, key=lambda x: x["max_abs"], reverse=True)
    report = {
        "common_keys": len(common),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "shape_mismatch": shape_mismatch,
        "global_mean_abs": float(sum_abs / max(total_elems, 1)),
        "global_max_abs": float(max_abs),
        "global_max_key": max_key,
        "topk_by_max_abs": per_key_sorted[:topk],
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_ckpt", required=True, help="PyTorch ckpt path")
    ap.add_argument("--jt_config", default="nanodet-jittor/config/nanodet-plus-m_320_voc.yml")
    ap.add_argument("--jt_ckpt", default=None, help="Existing Jittor ckpt (optional)")
    ap.add_argument("--jt_ckpt_out", default="workspace/pt2jt_roundtrip.pkl")
    ap.add_argument("--pt_ckpt_out", default="workspace/jt2pt_roundtrip.pth")
    ap.add_argument("--report", default="workspace/roundtrip_report.json")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    # 1) load original PT
    pt_sd = load_pt_state_dict(args.pt_ckpt)

    # 2) PT -> JT (if not provided)
    if args.jt_ckpt is None:
        jt_ckpt_path = convert_pt_to_jt(args.pt_ckpt, args.jt_config, args.jt_ckpt_out)
    else:
        jt_ckpt_path = args.jt_ckpt

    # 3) JT -> PT (round-trip)
    jt_sd = load_jt_state_dict(jt_ckpt_path)
    jt2pt_sd = {k: _to_torch(v) for k, v in jt_sd.items()}
    save_pt_checkpoint(
        jt2pt_sd,
        args.pt_ckpt_out,
        meta={"converted_from": "Jittor", "original_path": jt_ckpt_path},
    )

    # 4) diff
    report = diff_state_dicts(pt_sd, jt2pt_sd, topk=args.topk)
    report.update({
        "pt_ckpt": args.pt_ckpt,
        "jt_ckpt": jt_ckpt_path,
        "pt_roundtrip": args.pt_ckpt_out,
    })

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Round-trip report saved: {args.report}")
    print(f"Global mean abs diff: {report['global_mean_abs']:.6e}")
    print(f"Global max abs diff:  {report['global_max_abs']:.6e} @ {report['global_max_key']}")
    print(f"Common keys: {report['common_keys']} | only_in_a: {report['only_in_a']} | only_in_b: {report['only_in_b']}")
    if report["shape_mismatch"]:
        print(f"Shape mismatches: {len(report['shape_mismatch'])}")


if __name__ == "__main__":
    main()
