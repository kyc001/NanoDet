import os, sys, json, math
import numpy as np
import torch
import jittor as jt

# insert paths
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, '..'))
# 优先加载 Jittor 版本的 nanodet
sys.path.insert(0, os.path.join(PROJ, 'nanodet-jittor'))
# 安全导入：优先 Jittor 侧 box_transform，避免 PT 包冲突
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), 'nanodet-jittor'))
from nanodet.util.box_transform import distance2bbox

from nanodet.util.config import load_config as jt_load_cfg, cfg as jt_cfg
from nanodet.model.arch.nanodet_plus import NanoDetPlus as JT_NanoDetPlus
from nanodet.util.check_point import load_model_weight as jt_load_weight

# PyTorch 侧：延迟导入，且先清理已加载的 nanodet 模块以避免命名冲突
import importlib
pt_model = None
pt_cfg = None

def load_pt_model(cfg_path):
    global pt_model, pt_cfg
    # 清理已加载的 nanodet* 模块，切换到 PT 包
    to_del = [k for k in list(sys.modules.keys()) if k == 'nanodet' or k.startswith('nanodet.')]
    for k in to_del:
        sys.modules.pop(k, None)
    # 将 PT 工程路径置顶
    pt_root = os.path.join(PROJ, 'nanodet-pytorch')
    if pt_root in sys.path:
        sys.path.remove(pt_root)
    sys.path.insert(0, pt_root)
    # 导入 PT 版配置与模型
    cfg_mod = importlib.import_module('nanodet.util.config')
    load_config = getattr(cfg_mod, 'load_config')
    pt_cfg = getattr(cfg_mod, 'cfg')
    load_config(pt_cfg, cfg_path)
    arch_mod = importlib.import_module('nanodet.model.arch.nanodet_plus')
    PT_NanoDetPlus = getattr(arch_mod, 'NanoDetPlus')
    m = pt_cfg.model
    pt_model = PT_NanoDetPlus(m.arch.backbone, m.arch.fpn, m.arch.aux_head, m.arch.head, getattr(m.arch, 'detach_epoch', 0))
    pt_model.eval()
    return pt_model


def preprocess_image(img_path, cfg):
    # simple loader matching VOC eval pipeline: resize to input_size, keep_ratio
    import cv2
    img = cv2.imread(img_path)
    assert img is not None, f'fail to read {img_path}'
    d_w, d_h = cfg.model.arch.head.input_size if hasattr(cfg.model.arch.head, 'input_size') else cfg.data.val.input_size
    # keep_ratio resize like get_resize_matrix
    h, w = img.shape[:2]
    if w / h < d_w / d_h:
        ratio = d_h / h
    else:
        ratio = d_w / w
    nh, nw = int(round(h*ratio)), int(round(w*ratio))
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((d_h, d_w, 3), dtype=img.dtype)
    top = (d_h - nh)//2
    left = (d_w - nw)//2
    canvas[top:top+nh, left:left+nw] = img_resized
    # to CHW
    img_chw = canvas[:, :, ::-1].transpose(2,0,1).copy()  # BGR->RGB if needed; follow training norm
    img_chw = img_chw.astype(np.float32)
    # normalize like NanoDet (assume 1/255)
    img_chw /= 255.0
    return img_chw, (h, w), ratio, (top, left)

    # 诊断模式下为避免 GPU Fused Kernel 罕见崩溃，强制使用 CPU 更稳健（单图速度可接受）
    try:
        jt.flags.use_cuda = 0
    except Exception:
        pass


def jt_forward_dump(jt_cfg_path, jt_ckpt_path, img_chw):
    jt_load_cfg(jt_cfg, jt_cfg_path)
    # 构造与 PT 一致的 NanoDetPlus 签名（支持 aux_head / detach_epoch）
    arch = jt_cfg.model.arch
    model = JT_NanoDetPlus(arch.backbone, arch.fpn, arch.head, getattr(arch, 'aux_head', None), getattr(arch, 'detach_epoch', 0))
    # load converted or original
    if jt_ckpt_path and os.path.exists(jt_ckpt_path):
        # 轻量日志器（本地推理不需要复杂分布式上下文）
        class _DummyLogger:
            def info(self, *a, **k):
                pass
            def log(self, *a, **k):
                pass
        logger = _DummyLogger()
        ckpt = jt.load(jt_ckpt_path)
        jt_load_weight(model, ckpt, logger)
    model.eval()
    var = jt.array(img_chw[None, ...])
    feat = model.backbone(var)
    fpn_feat = model.fpn(feat)
    # ensure 3 levels according to strides
    if isinstance(fpn_feat, (list, tuple)) and len(fpn_feat) > len(model.head.strides):
        fpn_feat = fpn_feat[:len(model.head.strides)]
    # head forward manual to tap tensors
    cls_logits = []
    reg_logits = []
    featmap_sizes = []
    for idx, (feat, cls_convs, gfl_cls) in enumerate(
        zip(fpn_feat, model.head.cls_convs, model.head.gfl_cls)
    ):
        cf = feat
        for conv in cls_convs:
            cf = conv(cf)
        cls_l = gfl_cls(cf)  # note: not sigmoid
        cls_logits.append(cls_l)
        n, c, h, w = cls_l.shape
        featmap_sizes.append([int(h), int(w)])
        # Jittor 分离头：回归分支与 PT 不同结构，这里我们只做分类对齐；回归分支用 preds 切分统一方式
    # flatten to common shape [N, sum(HW), C]
    def _flatten(xs):
        ys = []
        shapes = []
        for x in xs:
            n, c, h, w = x.shape
            ys.append(x.reshape(n, c, h*w))
            shapes.append((int(h), int(w)))
        return jt.concat(ys, dim=2).permute(0,2,1), shapes
    cls_flat, cls_shapes = _flatten(cls_logits)
    # 与 PT 一致：通过 forward 输出 preds 再 split 得到 reg_flat
    # 额外：手动计算不含 scale 的回归分支（对齐 PT 合并头无 scale 行为）
    reg_logits_noscale = []
    for feat, reg_convs, gfl_reg in zip(
        fpn_feat, model.head.reg_convs, model.head.gfl_reg
    ):
        rf = feat
        for conv in reg_convs:
            rf = conv(rf)
        reg_l = gfl_reg(rf)  # 不乘 Scale
        reg_logits_noscale.append(reg_l)
    reg_flat_noscale, reg_shapes = _flatten(reg_logits_noscale)

    preds = model.head(fpn_feat)
    num_classes = model.head.num_classes
    reg_max = model.head.reg_max
    cls_flat2, reg_flat = preds.split([num_classes, 4*(reg_max+1)], dim=-1)
    assert cls_flat.shape == cls_flat2.shape, f"JT cls mismatch {cls_flat.shape} vs {cls_flat2.shape}"
    # DFL decode distances and bboxes
    dis_raw = model.head.distribution_project(reg_flat)
    dis = dis_raw
    # AB: 不含 scale 的距离与框
    dis_raw_noscale = model.head.distribution_project(reg_flat_noscale)
    dis_noscale = dis_raw_noscale
    mlvl_priors = [
        model.head.get_single_level_center_priors(
            batch_size=1, featmap_size=sz, stride=s, dtype=jt.float32, device=None
        ) for sz,s in zip(featmap_sizes, model.head.strides)
    ]
    priors = jt.concat(mlvl_priors, dim=1)
    input_h, input_w = img_chw.shape[1:]

    # noscale priors/bboxes
    priors_noscale = priors
    dis_noscale = dis_noscale * priors_noscale[...,2,None]
    bboxes_noscale = distance2bbox(priors_noscale[...,:2], dis_noscale, max_shape=(input_h, input_w))

    dis = dis * priors[...,2,None]
    input_h, input_w = img_chw.shape[1:]
    bboxes = distance2bbox(priors[...,:2], dis, max_shape=(input_h, input_w))
    scores = jt.sigmoid(cls_flat)
    return {
        'cls_flat': cls_flat.numpy(),
        'reg_flat': reg_flat.numpy(),
        'per_level': {
            'cls_shapes': cls_shapes,
            'reg_shapes': reg_shapes,
        },

        'reg_flat_noscale': reg_flat_noscale.numpy(),
        'dis_raw_noscale': dis_raw_noscale.numpy(),
        'bboxes_noscale': bboxes_noscale.numpy(),

        'dis_raw': dis_raw.numpy(),
        'dis': dis.numpy(),
        'bboxes': bboxes.numpy(),
        'scores': scores.numpy(),
        'priors': priors.numpy(),
        'feat_shapes': featmap_sizes,
    }


def pt_forward_dump(pt_cfg_path, pt_ckpt_path, img_chw):
    model = load_pt_model(pt_cfg_path)
    # load ckpt
    ckpt = torch.load(pt_ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    # strip prefixes
    def strip_key(k):
        for p in ('avg_model.', 'module.', 'model.'):
            if k.startswith(p):
                k = k[len(p):]
        return k
    sd = { strip_key(k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=False)
    model.eval()
    import torch.nn.functional as F
    x = torch.from_numpy(img_chw[None, ...]).float()
    with torch.no_grad():
        feat = model.backbone(x)
        fpn_feat = model.fpn(feat)
        if isinstance(fpn_feat, (list, tuple)) and len(fpn_feat) > len(model.head.strides):
            fpn_feat = fpn_feat[:len(model.head.strides)]
        # 直接用合并头前向，得到 [B, N, num_classes + 4*(reg_max+1)]
        preds = model.head(fpn_feat)
        num_classes = model.head.num_classes
        reg_max = model.head.reg_max
        cls_flat, reg_flat = preds.split([num_classes, 4*(reg_max+1)], dim=-1)
        # 使用 PT 端 Integral 做投影
        dis_raw = model.head.distribution_project(reg_flat)
        dis = dis_raw
        # 使用实际 FPN 特征图尺寸来构造 priors，避免 ceil/round 差异
        featmap_sizes = [ (int(f.shape[2]), int(f.shape[3])) for f in fpn_feat ]
        mlvl_priors = []
        device = cls_flat.device
        for (fh,fw), s in zip(featmap_sizes, model.head.strides):
            pri = model.head.get_single_level_center_priors(
                batch_size=1, featmap_size=(fh,fw), stride=s, dtype=torch.float32, device=device
            )
            mlvl_priors.append(pri)
        priors = torch.cat(mlvl_priors, dim=1)
        dis = dis * priors[...,2:3]
        # distance2bbox
        H, W = img_chw.shape[1:]
        l,t,r,b = dis[...,0],dis[...,1],dis[...,2],dis[...,3]
        cx, cy = priors[...,0], priors[...,1]
        x1 = (cx - l).clamp(min=0, max=W)
        y1 = (cy - t).clamp(min=0, max=H)
        x2 = (cx + r).clamp(min=0, max=W)
        y2 = (cy + b).clamp(min=0, max=H)
        bboxes = torch.stack([x1,y1,x2,y2], dim=-1)
        scores = torch.sigmoid(cls_flat)
    return {
        'cls_flat': cls_flat.numpy(),
        'reg_flat': reg_flat.numpy(),
        'reg_flat_noscale': reg_flat.numpy(),
        # noscale 分支（与 JT 对齐）：不乘 scale 的距离和 bbox
        'dis_raw_noscale': model.head.distribution_project(reg_flat).numpy(),
        'bboxes_noscale': torch.stack([
            (priors[...,0]-model.head.distribution_project(reg_flat)[...,0]).clamp(min=0, max=W),
            (priors[...,1]-model.head.distribution_project(reg_flat)[...,1]).clamp(min=0, max=H),
            (priors[...,0]+model.head.distribution_project(reg_flat)[...,2]).clamp(min=0, max=W),
            (priors[...,1]+model.head.distribution_project(reg_flat)[...,3]).clamp(min=0, max=H),
        ], dim=-1).numpy(),
        'dis_raw': dis_raw.numpy(),
        'dis': dis.numpy(),
        'bboxes': bboxes.numpy(),
        'scores': scores.numpy(),
        'priors': priors.numpy(),
        'feat_shapes': [ (int(f.shape[2]), int(f.shape[3])) for f in fpn_feat ],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True)
    ap.add_argument('--jt_cfg', default='nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    ap.add_argument('--pt_cfg', default='nanodet-pytorch/config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    ap.add_argument('--pt_ckpt', required=True)
    ap.add_argument('--jt_ckpt', default='workspace/pt2jt_model_best.pkl')
    ap.add_argument('--out', default='workspace/diag_pt_jt.json')
    args = ap.parse_args()

    # 解析 Jittor 配置，获取输入尺寸
    jt_load_cfg(jt_cfg, args.jt_cfg)
    d_w, d_h = jt_cfg.data.val.input_size
    img_chw, raw_hw, ratio, pad = preprocess_image(args.img, jt_cfg)
    jt_dump = jt_forward_dump(args.jt_cfg, args.jt_ckpt, img_chw)
    # 加载 PT 配置，与 jt 输入尺寸保持一致（我们的预处理已按 JT cfg 做过）
    pt_dump = pt_forward_dump(args.pt_cfg, args.pt_ckpt, img_chw)

    def summarize(d1, d2, key):
        a, b = d1[key], d2[key]
        diff = np.abs(a-b)
        return {
            'shape': a.shape,
            'mean_abs_diff': float(diff.mean()),
            'max_abs_diff': float(diff.max()),
        }

    # 若两侧总priors不一致（例如 ceil 差异），按共同长度截断比较
    jt_n = jt_dump['cls_flat'].shape[1]
    # 分层级误差统计（按每层HW切片）
    def per_level_metrics(jt_d, pt_d, shapes):
        res = {}
        start = 0
        for li, (h, w) in enumerate(shapes):
            cnt = h*w
            sl = slice(start, start+cnt)
            entry = {}
            for k in ['reg_flat','dis_raw','bboxes']:
                if k not in jt_d or k not in pt_d:
                    continue
                a = jt_d[k][:, sl]
                b = pt_d[k][:, sl]
                entry[k] = {
                    'mae': float(np.abs(a-b).mean()),
                    'max': float(np.abs(a-b).max()),
                    'shape': [int(h), int(w)]
                }
            res[f'P{li+3}'] = entry
            start += cnt
        return res
    # 初始化报告主要指标
    report = { k: summarize(jt_dump, pt_dump, k) for k in ['cls_flat','reg_flat','reg_flat_noscale','dis_raw_noscale','dis_raw','dis','bboxes_noscale','bboxes','scores'] }
    report['feat_shapes'] = {'jt': jt_dump.get('feat_shapes'), 'pt': pt_dump.get('feat_shapes')}
    report['per_level'] = per_level_metrics(jt_dump, pt_dump, jt_dump['feat_shapes'])

    per_level = per_level_metrics(jt_dump, pt_dump, jt_dump['feat_shapes'])

    pt_n = pt_dump['cls_flat'].shape[1]
    common = min(jt_n, pt_n)
    for d in (jt_dump, pt_dump):
        for k in ['cls_flat','reg_flat','reg_flat_noscale','dis_raw','dis','bboxes','scores','priors']:
            if k not in d:
                continue
            v = d[k]
            d[k] = v[:, :common, ...]
    report = { k: summarize(jt_dump, pt_dump, k) for k in ['cls_flat','reg_flat','reg_flat_noscale','dis_raw','dis','bboxes','scores'] }
    report['feat_shapes'] = {'jt': jt_dump.get('feat_shapes'), 'pt': pt_dump.get('feat_shapes')}

    # 额外分析：dis_raw 四个方向的维度误差、以及最优维度置换
    import itertools
    a = jt_dump['dis_raw']
    b = pt_dump['dis_raw']
    # 按方向维度分解误差
    per_dir = [float(np.abs(a[...,i]-b[...,i]).mean()) for i in range(4)]
    report['dis_raw_per_direction_mae'] = per_dir
    # 寻找最优维度置换
    perms = list(itertools.permutations([0,1,2,3]))
    best = None
    best_mae = 1e9
    for p in perms:
        mae = float(np.abs(a[...,list(p)] - b).mean())
        if mae < best_mae:
            best_mae = mae
            best = p
    report['dis_raw_best_perm'] = {'perm': best, 'mae': best_mae}

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print('[diag] saved', args.out)

if __name__ == '__main__':
    main()

