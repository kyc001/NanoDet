# -*- coding: utf-8 -*-
# Mini evaluation: load PyTorch ckpt -> Jittor model -> evaluate first N batches on val set
import os
import sys
import argparse
import random
import time
import numpy as np
import jittor as jt
import cv2

# prefer Jittor nanodet over pytorch repo
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
PT_ROOT = os.path.abspath(os.path.join(JT_ROOT, "../nanodet-pytorch"))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)
if PT_ROOT in sys.path:
    sys.path.remove(PT_ROOT)

import torch
from nanodet.util import load_config, cfg, Logger, mkdir
from nanodet.model.arch import build_model
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from nanodet.data.transform.pipeline import Pipeline
from nanodet.evaluator import build_evaluator
from nanodet.util.check_point import load_model_weight
from nanodet.util.box_transform import distance2bbox
from nanodet.model.module.nms import multiclass_nms
from collections import OrderedDict


def pt_state_to_jt_checkpoint(pt_ckpt, model=None, prefer_avg=True, logger=None):
    if isinstance(pt_ckpt, str):
        pt_ckpt = torch.load(pt_ckpt, map_location="cpu")
    state_dict = pt_ckpt.get("state_dict", pt_ckpt)
    if prefer_avg and any(k.startswith("avg_model.") for k in state_dict.keys()):
        state_dict = OrderedDict((k[len("avg_model."):], v) for k, v in state_dict.items() if k.startswith("avg_model."))
    # strip wrappers
    proc_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        v_np = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
        proc_sd[k] = v_np

    # shape-aware reconcile against model, with head mapping (merged->split)
    if model is not None:
        model_sd = model.state_dict()
        reconciled = {}
        for k, v_np in proc_sd.items():
            # 先做合并头 -> 分离头的映射（针对 gfl_cls.{i} 的 52 通道）
            # 权重
            if k.startswith("head.gfl_cls.") and k.endswith(".weight"):
                try:
                    i = int(k.split(".")[2])
                    cls_w_key = f"head.gfl_cls.{i}.weight"
                    reg_w_key = f"head.gfl_reg.{i}.weight"
                    if cls_w_key in model_sd and reg_w_key in model_sd:
                        cls_out = model_sd[cls_w_key].shape[0]
                        reg_out = model_sd[reg_w_key].shape[0]
                        if v_np.shape[0] == cls_out + reg_out:
                            reconciled[cls_w_key] = v_np[:cls_out].astype(np.float32)
                            reconciled[reg_w_key] = v_np[cls_out:cls_out+reg_out].astype(np.float32)
                            if logger:
                                logger.info(f"Split head weight {k} -> ({cls_w_key},{reg_w_key})")
                            continue
                except Exception:
                    pass
            # 偏置
            if k.startswith("head.gfl_cls.") and k.endswith(".bias"):
                try:
                    i = int(k.split(".")[2])
                    cls_b_key = f"head.gfl_cls.{i}.bias"
                    reg_b_key = f"head.gfl_reg.{i}.bias"
                    if cls_b_key in model_sd and reg_b_key in model_sd:
                        cls_out = model_sd[cls_b_key].shape[0]
                        reg_out = model_sd[reg_b_key].shape[0]
                        if v_np.shape[0] == cls_out + reg_out:
                            reconciled[cls_b_key] = v_np[:cls_out].astype(np.float32)
                            reconciled[reg_b_key] = v_np[cls_out:cls_out+reg_out].astype(np.float32)
                            if logger:
                                logger.info(f"Split head bias {k} -> ({cls_b_key},{reg_b_key})")
                            continue
                except Exception:
                    pass
            # 其他正常对齐：
            if k not in model_sd:
                continue
            tgt = model_sd[k]
            tshape = tuple(tgt.shape)
            vshape = tuple(v_np.shape)
            if vshape == tshape:
                reconciled[k] = v_np.astype(np.float32)
                continue
            # 4D conv weights（含 depthwise/grouped 适配）
            if v_np.ndim == 4 and len(tshape) == 4 and v_np.shape[0] == tshape[0]:
                if v_np.shape[1] == 1:
                    if tshape[1] == tshape[0]:
                        cout = tshape[0]
                        neww = np.zeros(tshape, dtype=np.float32)
                        for c in range(cout):
                            neww[c, c, :, :] = v_np[c, 0, :, :]
                        reconciled[k] = neww
                        if logger:
                            logger.info(f"Diagonal depthwise weight for {k}: {vshape} -> {tshape}")
                        continue
                    elif tshape[1] > 1:
                        reps = [1, tshape[1], 1, 1]
                        v_np_t = np.tile(v_np, reps)[:, :tshape[1], :, :]
                        reconciled[k] = v_np_t.astype(np.float32)
                        if logger:
                            logger.info(f"Tiled depthwise weight for {k}: {vshape} -> {tshape}")
                        continue
            # 1D 标量扩展
            if v_np.ndim == 0 and len(tshape) == 1 and tshape[0] == 1:
                reconciled[k] = np.array([v_np.item()], dtype=np.float32)
                if logger:
                    logger.info(f"Expanded scalar to vector for {k}: {vshape} -> {tshape}")
                continue
            # 其他不匹配：跳过，用模型默认初始化
            if logger:
                logger.info(f"Shape mismatch keep default for {k}: pt{vshape} vs jt{tshape}")
        proc_sd = reconciled

    return {"state_dict": proc_sd}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_val_batches", type=int, default=10)
    ap.add_argument("--save_dir", default="result/jittor_from_pt_mini")
    ap.add_argument('--dump_post', action='store_true', help='dump post-process intermediates for first few samples')
    ap.add_argument('--img', default=None, help='optional: specific image path used only for dump_post')

    ap.add_argument("--vis_num", type=int, default=2)
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    jt.flags.use_cuda = 1 if args.device.startswith("cuda") else 0

    load_config(cfg, args.cfg)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(save_dir=args.save_dir, name="mini_eval")

    # dataset & dataloader
    val_dataset = build_dataset(cfg.data.val, mode="val")
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    # model
    model = build_model(cfg.model)
    model.eval()

    # load pt ckpt
    pt_ckpt = torch.load(args.ckpt, map_location="cpu")
    jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True, logger=logger)
    load_model_weight(model, jt_ckpt, logger)

    # iterate first N batches
    total_batches = min(args.max_val_batches, len(val_dataset))
    results = {}
    vis_saved = 0

    # Build pipeline for visualization (need original raw img)
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    t0 = time.time()
    for idx in range(total_batches):
        # prefer a specific image for deterministic dump_post
        if args.img is not None and os.path.isfile(args.img):
            file_name = os.path.basename(args.img)
            raw_img = cv2.imread(args.img)
            meta = dict(img_info={"file_name": file_name, "height": raw_img.shape[0], "width": raw_img.shape[1], "id": idx}, img=raw_img)
            # run pipeline to build input tensor & warp_matrix
            pl = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
            meta = pl(val_dataset, meta, cfg.data.val.input_size)
        else:
            data = val_dataset[idx]
            meta = data
            file_name = meta["img_info"]["file_name"]
            raw_img = cv2.imread(os.path.join(val_dataset.img_path, file_name))
        # ensure batch dimension for model input
        if isinstance(meta["img"], jt.Var):
            meta["img"] = meta["img"].unsqueeze(0)
        else:
            im = meta["img"]
            if isinstance(im, np.ndarray) and im.ndim == 3 and im.shape[2] == 3:
                im = im.transpose(2, 0, 1)
            im = np.ascontiguousarray(im)
            meta["img"] = jt.array(im).unsqueeze(0)
        # ensure list-like fields
        if not isinstance(meta.get("warp_matrix"), list):
            meta["warp_matrix"] = [meta["warp_matrix"]]
        for key in ("height","width","id"):
            val = meta["img_info"].get(key, 0)
            if not isinstance(val, (list, tuple, np.ndarray)):
                meta["img_info"][key] = [val]

        if args.dump_post and idx < 3:
            # dump intermediates from head.get_bboxes pipeline
            # we re-run minimal parts here to capture tensors
            cls_reg = model(meta['img'])
            cls_pred, reg_pred = cls_reg.split([model.head.num_classes, 4*(model.head.reg_max+1)], dim=-1)
            b = cls_pred.shape[0]
            H, W = meta['img'].shape[2:]
            # 优先使用 head.execute 记录的实际各层特征图尺寸，避免用 ceil 推断导致不一致
            if hasattr(model.head, '_last_featmap_sizes') and len(model.head._last_featmap_sizes)==len(model.head.strides):
                featmap_sizes = list(model.head._last_featmap_sizes)
            else:
                featmap_sizes = [(int(np.ceil(H/s)), int(np.ceil(W/s))) for s in model.head.strides]
            priors = []
            for i,s in enumerate(model.head.strides):
                h,w = featmap_sizes[i]
                x = (jt.arange(w, dtype=jt.float32) * s).float32()
                y = (jt.arange(h, dtype=jt.float32) * s).float32()
                yy, xx = jt.meshgrid(y, x)
                pri = jt.stack([xx.flatten(), yy.flatten(), jt.full((xx.numel(),), s, dtype=jt.float32), jt.full((xx.numel(),), s, dtype=jt.float32)], dim=-1)
                priors.append(pri.unsqueeze(0).repeat(b,1,1))
            center_priors = jt.cat(priors, dim=1)
            prob = model.head.distribution_project(reg_pred)
            dis_only = prob
            dis_preds = prob * center_priors[...,2,None]
            xyxy = distance2bbox(center_priors[...,:2], dis_preds, max_shape=(H,W))
            scores = cls_pred.sigmoid()
            # NMS at input scale for alignment with PT
            score0 = scores[0]
            bbox0 = xyxy[0]
            padding = jt.zeros((score0.shape[0],1), dtype=score0.dtype)
            score0 = jt.concat([score0, padding], dim=1)
            dets0, labels0 = multiclass_nms(bbox0, score0, 0.05, dict(type='nms', iou_threshold=0.6), 100)

            npz = {
                'center_priors': center_priors.numpy(),
                'featmap_sizes': np.array(featmap_sizes, dtype=np.int32),
                'reg_logits': reg_pred.numpy(),
                'prob': prob.numpy(),
                'dis_only': dis_only.numpy(),
                'dis_preds': dis_preds.numpy(),
                'xyxy': xyxy.numpy(),
                'scores': scores.numpy(),
                'dets': dets0.numpy(),
                'labels': labels0.numpy(),
                'file_name': np.array([file_name]),
            }
            outp = os.path.join(args.save_dir, f'post_jt_{idx}.npz')
            np.savez_compressed(outp, **npz)

        # inference
        with jt.no_grad():
            det_dict = model.inference(meta)
        # det_dict is {img_id: {class_idx: [[x1,y1,x2,y2,score],...]}}
        results.update(det_dict)

        # visualization: save first few
        if vis_saved < args.vis_num:
            img_id = list(det_dict.keys())[0]
            save_path = os.path.join(args.save_dir, f"vis_{idx}_id{img_id}.jpg")
            try:
                model.head.show_result(raw_img, det_dict[img_id], cfg.class_names, score_thres=args.score_thr, show=False, save_path=save_path)
                vis_saved += 1
            except Exception as e:
                logger.info(f"vis failed on idx {idx}: {e}")

    # evaluate
    eval_res = evaluator.evaluate(results, args.save_dir)
    logger.info(f"Val_metrics: {eval_res}")
    # 保存评估结果到 JSON，便于自动化脚本读取
    try:
        import json
        with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
            json.dump(eval_res, f)
    except Exception as e:
        logger.info(f"save metrics.json failed: {e}")
    dt = time.time()-t0
    print(f"Mini-eval done in {dt:.1f}s; saved {vis_saved} visualizations to {args.save_dir}")


if __name__ == "__main__":
    main()

