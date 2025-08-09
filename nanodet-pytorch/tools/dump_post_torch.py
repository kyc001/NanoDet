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
        print(f"[PT] FPN output feats count: {len(fpn_feats)}, head strides count: {len(model.head.strides)}")
        for i, f in enumerate(fpn_feats):
            print(f"  feat[{i}]: shape={f.shape}")
        preds = model.head(fpn_feats)

    b = preds.shape[0]
    input_h, input_w = meta['img'].shape[2:]
    featmap_sizes = [ (int(np.ceil(input_h / s)), int(np.ceil(input_w / s))) for s in model.head.strides ]
    mlvl = [ model.head.get_single_level_center_priors(b, featmap_sizes[i], s, torch.float32, device) for i,s in enumerate(model.head.strides) ]
    center_priors = torch.cat(mlvl, dim=1)
    # Debug: stride distribution and dtype
    cp_stride = center_priors[..., 2]
    try:
        uniq = torch.unique(cp_stride).tolist()
    except Exception:
        uniq = sorted(list(set(cp_stride.reshape(-1).tolist())))
    flat = cp_stride.reshape(-1)
    print(f"[PT] center_priors stride dtype={cp_stride.dtype}, unique={uniq}")
    print("[PT] stride head10=", flat[:10].tolist(), "tail10=", flat[-10:].tolist())

    # Layer-wise priors debug: print each level's range and count
    print("[PT] Layer-wise priors debug:")
    start_idx = 0
    for i, (s, fs) in enumerate(zip(model.head.strides, featmap_sizes)):
        h, w = fs
        count = h * w
        end_idx = start_idx + count
        level_priors = center_priors[0, start_idx:end_idx]
        xy_min = level_priors[:, :2].min(dim=0)[0]
        xy_max = level_priors[:, :2].max(dim=0)[0]
        print(f"  Level {i}: stride={s}, shape=({h},{w}), count={count}, idx=[{start_idx}:{end_idx})")
        print(f"    xy_range: min=({xy_min[0]:.1f},{xy_min[1]:.1f}), max=({xy_max[0]:.1f},{xy_max[1]:.1f})")
        start_idx = end_idx


    cls_preds, reg_preds = preds.split([model.head.num_classes, 4*(model.head.reg_max+1)], dim=-1)
    # 额外导出：reg_logits 与 softmax 概率 p（形状 [B,N,4,m+1]）
    B,N = cls_preds.shape[0], center_priors.shape[1]
    m = model.head.reg_max
    reg_logits = reg_preds.reshape(B, N, 4, m+1)
    p = torch.softmax(reg_logits, dim=-1)
    dis_only = (p * torch.arange(0, m+1, dtype=torch.float32, device=device)).sum(dim=-1)  # [B,N,4]

    # Use dis_only for dis_preds to match JT diagnostics path exactly
    dis_preds = dis_only * center_priors[...,2,None]

    # Layer-wise head output debug: print each level's reg_logits stats
    print("[PT] Layer-wise head output debug:")
    start_idx = 0
    for i, (s, fs) in enumerate(zip(model.head.strides, featmap_sizes)):
        h, w = fs
        count = h * w
        end_idx = start_idx + count
        level_reg = reg_logits[0, start_idx:end_idx]  # [count, 4, 8]
        level_dis = dis_only[0, start_idx:end_idx]    # [count, 4]
        print(f"  Level {i}: stride={s}, reg_logits mean={level_reg.mean().item():.6f}, std={level_reg.std().item():.6f}")
        print(f"    dis_only mean={level_dis.mean().item():.6f}, std={level_dis.std().item():.6f}")
        start_idx = end_idx

    # Detailed decode diagnostics: export xyxy_raw and xyxy for element-level comparison
    pts = center_priors[..., :2]
    ltrb = dis_preds
    x1_raw = pts[..., 0] - ltrb[..., 0]
    y1_raw = pts[..., 1] - ltrb[..., 1]
    x2_raw = pts[..., 0] + ltrb[..., 2]
    y2_raw = pts[..., 1] + ltrb[..., 3]
    xyxy_raw = torch.stack([x1_raw, y1_raw, x2_raw, y2_raw], dim=-1)

    x1 = x1_raw.clamp(min=0, max=input_w)
    y1 = y1_raw.clamp(min=0, max=input_h)
    x2 = x2_raw.clamp(min=0, max=input_w)
    y2 = y2_raw.clamp(min=0, max=input_h)
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    bboxes = xyxy
    print(f"[PT] input_shape=({input_h},{input_w}), xyxy_raw vs xyxy diff: mean={torch.abs(xyxy_raw-xyxy).mean().item():.6f}, max={torch.abs(xyxy_raw-xyxy).max().item():.6f}")
    # Diagnostics: l,t,r,b pre/post clamp via xyxy clamp
    pts = center_priors[..., :2]
    ltrb = dis_preds
    x1_raw = pts[..., 0] - ltrb[..., 0]
    y1_raw = pts[..., 1] - ltrb[..., 1]
    x2_raw = pts[..., 0] + ltrb[..., 2]
    y2_raw = pts[..., 1] + ltrb[..., 3]
    x1 = x1_raw.clamp(min=0, max=int(input_w))
    y1 = y1_raw.clamp(min=0, max=int(input_h))
    x2 = x2_raw.clamp(min=0, max=int(input_w))
    y2 = y2_raw.clamp(min=0, max=int(input_h))
    ltrb_after = torch.stack([pts[...,0]-x1, pts[...,1]-y1, x2-pts[...,0], y2-pts[...,1]], dim=-1)
    diff = (ltrb_after - ltrb).abs()
    print(f"[PT] ltrb clamp delta mean={diff.mean().item():.4e}, max={diff.max().item():.4e}")
    # Show top-5 deltas (flat)
    flat = diff.reshape(-1)
    if flat.numel() > 0:
        topk = torch.topk(flat, k=min(5, flat.numel()))
        print("[PT] top5 |Δ|:", [float(v) for v in topk.values.tolist()])

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
        reg_logits=reg_logits.cpu().numpy(),
        prob=p.cpu().numpy(),
        dis_only=dis_only.cpu().numpy(),
        dis_preds=dis_preds.cpu().numpy(),
        xyxy_raw=xyxy_raw.cpu().numpy(),
        xyxy=xyxy.cpu().numpy(),
        bboxes=bboxes.cpu().numpy(),
        scores=scores.cpu().numpy(),
        dets=dets_np,
        labels=labels_np,
        input_shape=np.array([input_h, input_w], dtype=np.int32),
    )
    print(f"saved PT post to {args.out}")

if __name__=='__main__':
    main()

