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
    # 与 PT 保持一致：不在此处预计算最小适配尺寸，直接将 input_size 交给 pipeline
    meta = pipeline(None, meta, input_size)
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
    ap.add_argument('--pt_npz', default=None, help='optional: path to PT dump npz to override reg_logits for pipeline check')
    args = ap.parse_args()

    jt.flags.use_cuda = 1 if args.device.startswith('cuda') else 0
    logger = Logger(save_dir=os.path.dirname(args.out) or '.', name='dump_post_jt')
    load_config(cfg, args.cfg)

    model = build_model(cfg.model)
    model.eval()
    # load checkpoint: support JT(.pkl) directly, or convert PT(.ckpt/.pth) on-the-fly
    ckpt_path = args.ckpt
    if ckpt_path.lower().endswith(('.pkl', '.jt', '.jittor')):
        ckpt = jt.load(ckpt_path)
        load_model_weight(model, ckpt, logger)
    else:
        import torch
        pt_ckpt = torch.load(ckpt_path, map_location='cpu')
        jt_ckpt = pt_state_to_jt_checkpoint(pt_ckpt, model=model, prefer_avg=True)
        load_model_weight(model, jt_ckpt, logger)

    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    input_size = cfg.data.val.input_size
    meta = prepare_meta(args.img, pipeline, input_size)

    # forward to get preds
    img = meta['img']
    feats = model.backbone(img)
    fpn_feats = model.fpn(feats)
    print(f"[JT] FPN output feats count: {len(fpn_feats)}, head strides count: {len(model.head.strides)}")
    for i, f in enumerate(fpn_feats):
        print(f"  feat[{i}]: shape={f.shape}")
    preds = model.head(fpn_feats)

    # Optional: override reg_logits with PT dump to isolate pipeline diffs
    pt_npz = None
    if args.pt_npz and os.path.isfile(args.pt_npz):
        try:
            pt_npz = np.load(args.pt_npz)
            if 'reg_logits' in pt_npz:
                print('[JT] Override reg_logits from PT npz for pipeline verification')
        except Exception as e:
            print('[JT] Failed loading pt_npz:', e)

    # build center priors as get_bboxes
    b = preds.shape[0]
    input_h, input_w = img.shape[2:]
    # 固定到 PT 配置的三个尺度，H/8, W/8 等
    featmap_sizes = [ (int(np.ceil(input_h/ s)), int(np.ceil(input_w/ s))) for s in model.head.strides ]
    mlvl = [ model.head.get_single_level_center_priors(b, featmap_sizes[i], s, jt.float32, None) for i,s in enumerate(model.head.strides) ]
    center_priors = jt.cat(mlvl, dim=1)
    # Debug: stride distribution and dtype
    cp_stride = center_priors[..., 2]
    uniq = np.unique(cp_stride.numpy().reshape(-1))
    flat = cp_stride.reshape(-1)
    print(f"[JT] center_priors stride dtype={cp_stride.dtype}, unique={uniq.tolist()}")
    print("[JT] stride head10=", flat[:10].numpy().tolist(), "tail10=", flat[-10:].numpy().tolist())

    # Layer-wise priors debug: print each level's range and count
    print("[JT] Layer-wise priors debug:")
    start_idx = 0
    for i, (s, fs) in enumerate(zip(model.head.strides, featmap_sizes)):
        h, w = fs
        count = h * w
        end_idx = start_idx + count
        level_priors = center_priors[0, start_idx:end_idx]
        xy_min = level_priors[:, :2].min(dim=0)
        xy_max = level_priors[:, :2].max(dim=0)
        print(f"  Level {i}: stride={s}, shape=({h},{w}), count={count}, idx=[{start_idx}:{end_idx})")
        print(f"    xy_range: min=({xy_min[0].item():.1f},{xy_min[1].item():.1f}), max=({xy_max[0].item():.1f},{xy_max[1].item():.1f})")
        start_idx = end_idx


    cls_preds, reg_preds = preds.split([model.head.num_classes, 4*(model.head.reg_max+1)], dim=-1)
    # 额外导出：reg_logits 与 softmax 概率 p（形状 [B,N,4,m+1]）
    B,N = cls_preds.shape[0], center_priors.shape[1]
    m = model.head.reg_max
    reg_logits = reg_preds.reshape(B, N, 4, m+1)
    # override from pt_npz if provided (only for diagnostics)
    if pt_npz is not None and 'reg_logits' in pt_npz and pt_npz['reg_logits'].shape == reg_logits.shape:
        reg_logits = jt.array(pt_npz['reg_logits'])
        print('[JT] reg_logits overridden by PT dump (diagnostics only)')
    p = nn.softmax(reg_logits, dim=-1)
    dis_only = (p * jt.arange(0, m+1, dtype=jt.float32)).sum(dim=-1)  # [B,N,4]

    # 强制 float32 精度避免 stride 乘法的累积误差
    stride_multiplier = center_priors[...,2,None].float32()
    # Use dis_only computed from reg_logits to form dis_preds, ensuring path parity with PT
    dis_preds = (dis_only.float32() * stride_multiplier).float32()

    # Layer-wise head output debug: print each level's reg_logits stats
    print("[JT] Layer-wise head output debug:")
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



    # Diagnostics: l,t,r,b pre/post clamp via xyxy clamp
    pts = center_priors[..., :2]
    ltrb = dis_preds
    x1_raw = pts[..., 0] - ltrb[..., 0]
    y1_raw = pts[..., 1] - ltrb[..., 1]
    x2_raw = pts[..., 0] + ltrb[..., 2]
    y2_raw = pts[..., 1] + ltrb[..., 3]
    x1 = x1_raw.clamp(min_v=0, max_v=int(input_w))
    y1 = y1_raw.clamp(min_v=0, max_v=int(input_h))
    x2 = x2_raw.clamp(min_v=0, max_v=int(input_w))
    y2 = y2_raw.clamp(min_v=0, max_v=int(input_h))
    ltrb_after = jt.stack([pts[...,0]-x1, pts[...,1]-y1, x2-pts[...,0], y2-pts[...,1]], dim=-1)
    diff = (ltrb_after - ltrb).abs()
    print(f"[JT] ltrb clamp delta mean={float(diff.mean()) :.4e}, max={float(diff.max()) :.4e}")
    # Show top-5 deltas (flat)
    flat = diff.reshape(-1)
    # print top-5 manually since Jittor lacks topk on 1D Var in some versions
    vals = flat.numpy()
    topk_idx = np.argpartition(vals, -5)[-5:]
    print("[JT] top5 |Δ|:", [float(vals[i]) for i in topk_idx])
    # Detailed decode diagnostics: export xyxy_raw and xyxy for element-level comparison
    pts = center_priors[..., :2]
    ltrb = dis_preds
    x1_raw = pts[..., 0] - ltrb[..., 0]
    y1_raw = pts[..., 1] - ltrb[..., 1]
    x2_raw = pts[..., 0] + ltrb[..., 2]
    y2_raw = pts[..., 1] + ltrb[..., 3]
    xyxy_raw = jt.stack([x1_raw, y1_raw, x2_raw, y2_raw], dim=-1)

    x1 = x1_raw.clamp(min_v=0, max_v=input_w)
    y1 = y1_raw.clamp(min_v=0, max_v=input_h)
    x2 = x2_raw.clamp(min_v=0, max_v=input_w)
    y2 = y2_raw.clamp(min_v=0, max_v=input_h)
    xyxy = jt.stack([x1, y1, x2, y2], dim=-1)

    bboxes = xyxy
    print(f"[JT] input_shape=({input_h},{input_w}), xyxy_raw vs xyxy diff: mean={float((xyxy_raw-xyxy).abs().mean()):.6f}, max={float((xyxy_raw-xyxy).abs().max()):.6f}")
    scores = cls_preds.sigmoid()

    # nms for img 0
    score0 = scores[0]
    bbox0 = bboxes[0]
    padding = jt.zeros((score0.shape[0],1), dtype=score0.dtype)
    score0 = jt.concat([score0, padding], dim=1)
    dets0, labels0 = multiclass_nms(bbox0, score0, 0.05, dict(type='nms', iou_threshold=0.6), 100)

    # also produce warped-to-original dets using inverse warp_matrix
    from nanodet.data.transform.warp import warp_boxes
    W = meta['warp_matrix'][0]
    W = np.array(W, dtype=np.float64)
    if W.ndim == 3 and W.shape[0] == 1:
        W = W[0]
    if W.shape == (2,3):
        W = np.vstack([W, [0.0, 0.0, 1.0]])
    invW = np.linalg.inv(W)
    ow = int(meta['img_info']['width'][0]); oh = int(meta['img_info']['height'][0])
    dets_warp = dets0.numpy().copy()
    if dets_warp.shape[0] > 0:
        dets_warp[:, :4] = warp_boxes(dets_warp[:, :4], invW, ow, oh)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out,
        center_priors=center_priors.numpy(),
        reg_logits=reg_logits.numpy(),
        prob=p.numpy(),
        dis_only=dis_only.numpy(),
        dis_preds=dis_preds.numpy(),
        xyxy_raw=xyxy_raw.numpy(),
        xyxy=xyxy.numpy(),
        bboxes=bboxes.numpy(),
        scores=scores.numpy(),
        dets=dets0.numpy(),
        labels=labels0.numpy(),
        dets_warped=dets_warp,
        labels_warped=labels0.numpy(),
        input_shape=np.array([input_h, input_w], dtype=np.int32),
    )
    print(f"saved JT post to {args.out}")

if __name__=='__main__':
    main()

