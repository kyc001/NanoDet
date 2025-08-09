# -*- coding: utf-8 -*-
# Dump PyTorch post-process intermediates: center_priors, dis_preds, bboxes, scores, dets
import os, sys, argparse, json
import numpy as np
import cv2
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PT_ROOT not in sys.path:
    sys.path.insert(0, PT_ROOT)

from nanodet.util import load_config, cfg
from nanodet.util.logger import Logger
from nanodet.model.arch import build_model
from nanodet.data.transform.pipeline import Pipeline
from nanodet.model.head.nanodet_plus_head import multiclass_nms
from nanodet.util.box_transform import distance2bbox

def prepare_meta(img_path, pipeline, input_size):
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"
    img_info = {"file_name": os.path.basename(img_path), "height": img.shape[0], "width": img.shape[1], "id": 0}
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, input_size)
    im = meta["img"].transpose(2,0,1)
    im = np.ascontiguousarray(im)
    meta["img"] = torch.from_numpy(im).unsqueeze(0).float()
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
    ap.add_argument('--out', default='workspace/pt_post.npz')
    ap.add_argument('--device', default='cuda:0')
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

    # 与官方评估一致：keep_ratio=True，最小适配尺寸（如 320x232）
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    input_size = cfg.data.val.input_size
    meta = prepare_meta(args.img, pipeline, input_size)
    meta['img'] = meta['img'].to(device)

    with torch.no_grad():
        feats = model.backbone(meta['img'])
        fpn_feats = model.fpn(feats)
        preds = model.head(fpn_feats)

    b = preds.shape[0]
    input_h, input_w = meta['img'].shape[2:]
    featmap_sizes = [ (int(np.ceil(input_h/s)), int(np.ceil(input_w))/ s) for s in model.head.strides ]
    mlvl = [ model.head.get_single_level_center_priors(b, featmap_sizes[i], s, torch.float32, device) for i,s in enumerate(model.head.strides) ]
    center_priors = torch.cat(mlvl, dim=1)
    cls_preds, reg_preds = preds.split([model.head.num_classes, 4*(model.head.reg_max+1)], dim=-1)
    dis_preds = model.head.distribution_project(reg_preds) * center_priors[...,2,None]
    bboxes = distance2bbox(center_priors[...,:2], dis_preds, max_shape=(input_h, input_w))
    scores = torch.sigmoid(cls_preds)

    score0 = scores[0]
    bbox0 = bboxes[0]
    padding = torch.zeros((score0.shape[0],1), device=device, dtype=score0.dtype)
    score0 = torch.cat([score0, padding], dim=1)
    dets_np = None; labels_np = None
    try:
        dets0, labels0 = multiclass_nms(bbox0, score0, 0.05, dict(type='nms', iou_threshold=0.6), 100)
        dets_np = dets0.cpu().numpy(); labels_np = labels0.cpu().numpy()
    except Exception as e:
        print('[warn] PT NMS failed, skip dets/labels dump:', e)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out,
        center_priors=center_priors.cpu().numpy(),
        dis_preds=dis_preds.cpu().numpy(),
        bboxes=bboxes.cpu().numpy(),
        scores=scores.cpu().numpy(),
        dets=dets_np,
        labels=labels_np,
        input_shape=np.array([input_h, input_w], dtype=np.int32),
    )
    print(f"saved PT post to {args.out}")

if __name__=='__main__':
    main()

