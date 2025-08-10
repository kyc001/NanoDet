# -*- coding: utf-8 -*-
"""
双向权重转换脚本：
- PyTorch ckpt  -> Jittor ckpt(.pkl)
- Jittor  ckpt(.pkl) -> PyTorch ckpt(.ckpt/.pth)

用法示例：
1) PT -> JT
   conda run -n nano python nanodet-jittor/tools/convert_pt_jt_weights.py \
       --direction pt2jt \
       --cfg nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml \
       --src /path/to/model_best.ckpt \
       --dst workspace/pt2jt_from_best.pkl 

2) JT -> PT
   conda run -n nano python nanodet-jittor/tools/convert_pt_jt_weights.py \
       --direction jt2pt \
       --cfg nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml \
       --src workspace/pt2jt_from_best.pkl \
       --dst workspace/jt2pt_from_best.ckpt
"""
import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import jittor as jt

# 仅在需要 PT 时导入 torch
try:
    import torch
except Exception:
    torch = None

# 将项目根目录加入路径
CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CUR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nanodet.util import load_config, cfg
from nanodet.model.arch import build_model
from nanodet.util.check_point import load_model_weight


def _strip_prefix_key(k: str) -> str:
    if k.startswith("module."):
        k = k[7:]
    if k.startswith("model."):
        k = k[6:]
    return k


def _pt2jt_state(model, pt_ckpt):
    """将 PT 的 state_dict 标准化为 JT 可直接 load 的 dict。
    额外处理：
    - 若存在 avg_model.* 分支，优先使用其权重；
    - 合并头(gfl_cls) -> 分离头(gfl_cls+gfl_reg) 切分映射；
    - 形状对齐与回退。
    """
    state_dict = pt_ckpt.get("state_dict", pt_ckpt)
    # 优先用 EMA 分支
    if any(k.startswith("avg_model.") for k in state_dict.keys()):
        state_dict = OrderedDict((k[len("avg_model."):], v) for k, v in state_dict.items() if k.startswith("avg_model."))
    # 去前缀
    proc = {}
    for k, v in state_dict.items():
        k = _strip_prefix_key(k)
        proc[k] = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.array(v)

    # 形状对齐 + 合并头拆分
    model_sd = build_model_state(model)
    reconciled = {}
    for k, v_np in proc.items():
        # 合并头 -> 分离头
        if k.startswith("head.gfl_cls.") and (k.endswith(".weight") or k.endswith(".bias")):
            try:
                i = int(k.split(".")[2])
                cls_key = f"head.gfl_cls.{i}.{k.split('.')[-1]}"
                reg_key = f"head.gfl_reg.{i}.{k.split('.')[-1]}"
                if cls_key in model_sd and reg_key in model_sd:
                    cls_out = model_sd[cls_key].shape[0]
                    reg_out = model_sd[reg_key].shape[0]
                    if v_np.shape[0] == cls_out + reg_out:
                        reconciled[cls_key] = v_np[:cls_out].astype(np.float32)
                        reconciled[reg_key] = v_np[cls_out:cls_out+reg_out].astype(np.float32)
                        continue
            except Exception:
                pass
        # 常规对齐
        if k not in model_sd:
            continue
        tshape = tuple(model_sd[k].shape)
        if tuple(v_np.shape) == tshape:
            reconciled[k] = v_np.astype(np.float32)
            continue
        # 其他不匹配，跳过，由默认初始化填充
    return {"state_dict": reconciled}


def build_model_state(model):
    """取出 Jittor 模型当前 state_dict 的 numpy 形状信息。"""
    sd = model.state_dict()
    info = {}
    for k, v in sd.items():
        info[k] = np.zeros(tuple(v.shape), dtype=np.float32)
    return info


def convert_pt2jt(cfg_path: str, src: str, dst: str):
    assert torch is not None, "需要安装 PyTorch 才能进行 pt2jt 转换"
    load_config(cfg, cfg_path)
    model = build_model(cfg.model)
    pt_ckpt = torch.load(src, map_location="cpu")
    # 先得到候选 jt_ckpt（只含匹配键），再通过框架内的 load_model_weight 做一次完整对齐和推断填充
    jt_ckpt = _pt2jt_state(model, pt_ckpt)
    # 将权重真正加载进模型，触发合并头切分、形状过滤与镜像填充
    load_model_weight(model, jt_ckpt, logger=None)
    # 保存为 JT 标准检查点（包含完整 state_dict）
    full_ckpt = {"state_dict": model.state_dict()}
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    jt.save(full_ckpt, dst)
    print(f"PT->JT 转换完成: {src} -> {dst}")


def convert_jt2pt(cfg_path: str, src: str, dst: str):
    load_config(cfg, cfg_path)
    model = build_model(cfg.model)
    # 读取 JT 权重
    ckpt = jt.load(src)
    # 由 load_model_weight 做一次映射+对齐，确保能正确加载
    load_model_weight(model, ckpt, logger=None)
    # 取出对齐后的权重，封装成 PT 风格
    sd = model.state_dict()
    pt_sd = OrderedDict()
    for k, v in sd.items():
        arr = np.array(v)
        if torch is not None:
            pt_sd[k] = torch.from_numpy(arr)
        else:
            pt_sd[k] = arr  # 退化为 numpy
    to_save = {"state_dict": pt_sd}
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if torch is not None:
        torch.save(to_save, dst)
    else:
        import pickle
        with open(dst, 'wb') as f:
            pickle.dump(to_save, f)
    print(f"JT->PT 转换完成: {src} -> {dst}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["pt2jt", "jt2pt"], required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.direction == "pt2jt":
        convert_pt2jt(args.cfg, args.src, args.dst)
    else:
        convert_jt2pt(args.cfg, args.src, args.dst)


if __name__ == "__main__":
    main()

