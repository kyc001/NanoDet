# -*- coding: utf-8 -*-
# Dump Jittor post-process intermediates: center_priors, dis_preds, bboxes, scores, dets
import os, sys, argparse, json
import numpy as np
import jittor as jt
from jittor import nn
# robust image reader
def imread_bgr(path):
    try:
        import cv2
        img = cv2.imread(path)
        if img is not None:
            return img
    except Exception:
        pass
    from PIL import Image
    im = Image.open(path).convert('RGB')
    arr = np.array(im)[:, :, ::-1].copy()
    return arr

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
PT_ROOT = os.path.abspath(os.path.join(JT_ROOT, "../nanodet-pytorch"))
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)
if PT_ROOT in sys.path:
    sys.path.remove(PT_ROOT)

from nanodet.util import load_config, cfg, Logger
from nanodet.model.arch import build_model
from nanodet.data.transform.pipeline import Pipeline
from nanodet.model.module.nms import multiclass_nms
from nanodet.util.box_transform import distance2bbox

# reuse PT->JT converter
from tools.infer_from_pt_ckpt import pt_state_to_jt_checkpoint
from nanodet.util.check_point import load_model_weight

def prepare_meta(img_path, pipeline, input_size):
    img = imread_bgr(img_path)
    img_info = {"file_name": os.path.basename(img_path), "height": img.shape[0], "width": img.shape[1], "id": 0}
    meta = dict(img_info=img_info, raw_img=img, img=img)
    # 为严格对齐 PT，这里根据原图大小计算最小适配尺寸（如 320x232），再送入 pipeline
    try:
        from nanodet.data.transform.warp import get_minimum_dst_shape
        dst_size = get_minimum_dst_shape((img_info['width'], img_info['height']), tuple(input_size))
    except Exception:
        dst_size = tuple(input_size)
    meta = pipeline(None, meta, dst_size)
    im = meta["img"].transpose(2,0,1)
    im = np.ascontiguousarray(im)
    meta["img"] = jt.array(im).unsqueeze(0)
    if not isinstance(meta.get("warp_matrix"), list):
        meta["warp_matrix"] = [meta["warp_matrix"]]
    for key in ("height","width","id"):
        v = meta["img_info"].get(key, 0)
        if not isinstance(v,(list,tuple,np.ndarray)):
            meta["img_info"][key] = [v]
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img', required=True)
    ap.add_argument('--out', default='result/jt_post.npz')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    jt.flags.use_cuda = 1 if args.device.startswith('cuda') else 0
    logger = Logger(save_dir=os.path.dirname(args.out) or '.', name='dump_post_jt')
    load_config(cfg, args.cfg)

    model = build_model(cfg.model)
    model.eval()
    # load PT ckpt into JT
    import torch
    pt_ckpt = torch.load(args.ckpt, map_location='cpu')
    jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True)
    load_model_weight(model, jt_ckpt, logger)

    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    input_size = cfg.data.val.input_size
    meta = prepare_meta(args.img, pipeline, input_size)

    # forward to get preds
    img = meta['img']
    feats = model.backbone(img)
    fpn_feats = model.fpn(feats)
    preds = model.head(fpn_feats)

    # build center priors as get_bboxes
    b = preds.shape[0]
    input_h, input_w = img.shape[2:]
    # 固定到 PT 配置的三个尺度，H/8, W/8 等
    featmap_sizes = [ (int(np.ceil(input_h/ s)), int(np.ceil(input_w/ s))) for s in model.head.strides ]
    mlvl = [ model.head.get_single_level_center_priors(b, featmap_sizes[i], s, jt.float32, None) for i,s in enumerate(model.head.strides) ]
    center_priors = jt.cat(mlvl, dim=1)

    cls_preds, reg_preds = preds.split([model.head.num_classes, 4*(model.head.reg_max+1)], dim=-1)
    # 额外导出：reg_logits 与 softmax 概率 p（形状 [B,N,4,m+1]）
    B,N = cls_preds.shape[0], center_priors.shape[1]
    m = model.head.reg_max
    reg_logits = reg_preds.reshape(B, N, 4, m+1)
    p = nn.softmax(reg_logits, dim=-1)
    dis_only = (p * jt.arange(0, m+1, dtype=jt.float32)).sum(dim=-1)  # [B,N,4]

    dis_preds = model.head.distribution_project(reg_preds) * center_priors[...,2,None]
    bboxes = distance2bbox(center_priors[...,:2], dis_preds, max_shape=(input_h, input_w))
    scores = cls_preds.sigmoid()

    # nms for img 0
    score0 = scores[0]
    bbox0 = bboxes[0]
    padding = jt.zeros((score0.shape[0],1), dtype=score0.dtype)
    score0 = jt.concat([score0, padding], dim=1)
    dets0, labels0 = multiclass_nms(bbox0, score0, 0.05, dict(type='nms', iou_threshold=0.6), 100)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out,
        center_priors=center_priors.numpy(),
        reg_logits=reg_logits.numpy(),
        prob=p.numpy(),
        dis_only=dis_only.numpy(),
        dis_preds=dis_preds.numpy(),
        bboxes=bboxes.numpy(),
        scores=scores.numpy(),
        dets=dets0.numpy(),
        labels=labels0.numpy(),
        input_shape=np.array([input_h, input_w], dtype=np.int32),
    )
    print(f"saved JT post to {args.out}")

if __name__=='__main__':
    main()

