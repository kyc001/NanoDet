# -*- coding: utf-8 -*-
"""
双向权重转换脚本：
- PyTorch -> Jittor (.pkl)
- Jittor  -> PyTorch (.pth)

用法示例：

1) PT -> JT
python tools/convert_checkpoint.py \
    --mode pt2jt \
    --cfg nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml \
    --in_ckpt /path/to/model_best.ckpt \
    --out_ckpt workspace/pt2jt_model_best.pkl \
    --prefer_avg

2) JT -> PT
python tools/convert_checkpoint.py \
    --mode jt2pt \
    --cfg nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml \
    --in_ckpt workspace/model_last.ckpt \
    --out_ckpt workspace/jt2pt_model_last.pth
"""
import os
import sys
import argparse
import numpy as np
import jittor as jt

# 把项目根目录加入 sys.path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)

from nanodet.util import load_config, cfg
from nanodet.model.arch import build_model
from collections import OrderedDict


def strip_prefix(k: str):
    # 去掉常见前缀
    if k.startswith("module."):
        k = k[7:]
    if k.startswith("model."):
        k = k[6:]
    return k


def pt_state_to_jt_checkpoint(pt_ckpt: dict, model=None, prefer_avg=True, logger=None):
    try:
        import torch  # only used when loading pt ckpt
    except Exception:
        pass
    state_dict = pt_ckpt.get("state_dict", pt_ckpt)
    # 优先使用 avg_model.* 分支
    if prefer_avg and any(k.startswith("avg_model.") for k in state_dict.keys()):
        state_dict = OrderedDict(
            (k[len("avg_model."):], v) for k, v in state_dict.items() if k.startswith("avg_model.")
        )
    # 统一转 numpy
    proc_sd = {}
    for k, v in state_dict.items():
        k = strip_prefix(k)
        try:
            import torch
            if isinstance(v, torch.Tensor):
                v_np = v.detach().cpu().numpy()
            else:
                v_np = np.array(v)
        except Exception:
            v_np = np.array(v)
        proc_sd[k] = v_np

    # 关键别名映射，避免丢失中间卷积权重
    proc_alias = {}
    for k, v_np in proc_sd.items():
        if k.startswith('head.gfl_reg_convs.'):
            nk = k.replace('head.gfl_reg_convs.', 'head.reg_convs.', 1)
            proc_alias[nk] = v_np
            continue
        if k.startswith('head.gfl_cls_convs.'):
            nk = k.replace('head.gfl_cls_convs.', 'head.cls_convs.', 1)
            proc_alias[nk] = v_np
            continue
        proc_alias[k] = v_np
    proc_sd = proc_alias

    # 形状对齐 + 合并头 -> 分离头拆分
    if model is not None:
        model_sd = model.state_dict()
        reconciled = {}
        for k, v_np in proc_sd.items():
            # 先尝试拆分 head（PT 的合并 head.gfl_cls.{i}.* -> JT 的 gfl_cls/gfl_reg）
            if k.startswith("head.gfl_cls.") and (k.endswith(".weight") or k.endswith(".bias")):
                try:
                    i = int(k.split(".")[2])
                    if k.endswith(".weight"):
                        cls_w_key = f"head.gfl_cls.{i}.weight"
                        reg_w_key = f"head.gfl_reg.{i}.weight"
                        if cls_w_key in model_sd and reg_w_key in model_sd:
                            cls_out = model_sd[cls_w_key].shape[0]
                            reg_out = model_sd[reg_w_key].shape[0]
                            if v_np.shape[0] == cls_out + reg_out:
                                reconciled[cls_w_key] = v_np[:cls_out].astype(np.float32)
                                reconciled[reg_w_key] = v_np[cls_out:cls_out+reg_out].astype(np.float32)
                                continue
                    else:  # bias
                        cls_b_key = f"head.gfl_cls.{i}.bias"
                        reg_b_key = f"head.gfl_reg.{i}.bias"
                        if cls_b_key in model_sd and reg_b_key in model_sd:
                            cls_out = model_sd[cls_b_key].shape[0]
                            reg_out = model_sd[reg_b_key].shape[0]
                            if v_np.shape[0] == cls_out + reg_out:
                                reconciled[cls_b_key] = v_np[:cls_out].astype(np.float32)
                                reconciled[reg_b_key] = v_np[cls_out:cls_out+reg_out].astype(np.float32)
                                continue
                except Exception:
                    pass
            # 常规对齐
            if k not in model_sd:
                continue
            tgt = model_sd[k]
            tshape = tuple(tgt.shape)
            vshape = tuple(v_np.shape)
            if vshape == tshape:
                reconciled[k] = v_np.astype(np.float32)
                continue
            # 处理 depthwise/grouped 的常见差异
            if v_np.ndim == 4 and len(tshape) == 4 and v_np.shape[0] == tshape[0]:
                if v_np.shape[1] == 1:
                    if tshape[1] == tshape[0]:
                        cout = tshape[0]
                        neww = np.zeros(tshape, dtype=np.float32)
                        for c in range(cout):
                            neww[c, c, :, :] = v_np[c, 0, :, :]
                        reconciled[k] = neww
                        continue
                    elif tshape[1] > 1:
                        reps = [1, tshape[1], 1, 1]
                        v_np_t = np.tile(v_np, reps)[:, :tshape[1], :, :]
                        reconciled[k] = v_np_t.astype(np.float32)
                        continue
            # 1D 标量扩展
            if v_np.ndim == 0 and len(tshape) == 1 and tshape[0] == 1:
                reconciled[k] = np.array([v_np.item()], dtype=np.float32)
                continue
            # 其他不匹配，跳过
        proc_sd = reconciled

    return {"state_dict": proc_sd}


def jt_checkpoint_to_pt(jt_ckpt: dict, model=None):
    """将 Jittor 的分离头合并为 PyTorch 风格的合并头 state_dict。"""
    if isinstance(jt_ckpt, dict) and "state_dict" in jt_ckpt:
        sd = jt_ckpt["state_dict"]
    else:
        sd = jt_ckpt
    out_sd = OrderedDict()

    # 先复制不涉及 head.gfl_* 的权重
    for k, v in sd.items():
        if k.startswith("head.gfl_cls.") or k.startswith("head.gfl_reg."):
            continue
        out_sd[k] = np.array(v)

    # 合并每个层级的 cls 和 reg
    for i in range(4):
        cls_w = sd.get(f"head.gfl_cls.{i}.weight", None)
        reg_w = sd.get(f"head.gfl_reg.{i}.weight", None)
        if cls_w is not None and reg_w is not None:
            out_sd[f"head.gfl_cls.{i}.weight"] = np.concatenate([np.array(cls_w), np.array(reg_w)], axis=0)
        cls_b = sd.get(f"head.gfl_cls.{i}.bias", None)
        reg_b = sd.get(f"head.gfl_reg.{i}.bias", None)
        if cls_b is not None and reg_b is not None:
            out_sd[f"head.gfl_cls.{i}.bias"] = np.concatenate([np.array(cls_b), np.array(reg_b)], axis=0)

    return {"state_dict": out_sd}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pt2jt", "jt2pt"], required=True)
    ap.add_argument("--cfg", required=True, help="用于构建模型以对齐形状")
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--prefer_avg", action="store_true", help="PT->JT 时优先使用 avg_model.* 分支")
    return ap.parse_args()


def main():
    args = parse_args()
    load_config(cfg, args.cfg)
    model = build_model(cfg.model)
    model.eval()

    if args.mode == "pt2jt":
        try:
            import torch
        except Exception as e:
            raise RuntimeError("PT->JT 需要安装 PyTorch 以读取 .ckpt/.pth") from e
        pt_ckpt = torch.load(args.in_ckpt, map_location="cpu")
        jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=args.prefer_avg)
        # 保存为 Jittor 可加载的 .pkl
        jt.save(jt_ckpt, args.out_ckpt)
        print(f"[OK] PT->JT: 保存到 {args.out_ckpt}，键数: {len(jt_ckpt['state_dict'])}")
    else:  # jt2pt
        # 读 Jittor ckpt（.pkl/.ckpt）
        jt_ckpt = jt.load(args.in_ckpt)
        pt_ckpt = jt_checkpoint_to_pt(jt_ckpt, model=model)
        # 保存为 PyTorch .pth
        try:
            import torch
            torch.save(pt_ckpt, args.out_ckpt)
        except Exception:
            # 退化为 numpy pickle（仍可被 torch.load 读取）
            import pickle
            with open(args.out_ckpt, 'wb') as f:
                pickle.dump(pt_ckpt, f)
        print(f"[OK] JT->PT: 保存到 {args.out_ckpt}，键数: {len(pt_ckpt['state_dict'])}")


if __name__ == "__main__":
    main()

