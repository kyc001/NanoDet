# -*- coding: utf-8 -*-
# Compare PT and JT post-process intermediates to locate first divergence
import os, argparse, numpy as np

KEYS = [
    'center_priors', 'reg_logits', 'prob', 'dis_only', 'dis_preds', 'xyxy_raw', 'xyxy', 'bboxes', 'scores'
]


def stat_diff(a, b):
    if a.shape != b.shape:
        return {'shape_pt': a.shape, 'shape_jt': b.shape, 'mean_abs': None, 'max_abs': None}
    d = np.abs(a - b)
    return {'shape': a.shape, 'mean_abs': float(d.mean()), 'max_abs': float(d.max())}


def compare_dets_labels(pt, jt, topk=10, xyxy_mae_thr=0.2, score_mae_thr=1e-3):
    print("\nNMS outputs (dets/labels) comparison:")
    if ('dets' not in pt) or ('dets' not in jt):
        print(' dets missing in one of files')
        return
    if ('labels' not in pt) or ('labels' not in jt):
        print(' labels missing in one of files')
        return
    dets_pt, dets_jt = pt['dets'], jt['dets']
    labels_pt, labels_jt = pt['labels'], jt['labels']

    if dets_pt is None or dets_jt is None:
        print(' dets is None in one of files, skip NMS compare')
        return
    if labels_pt is None or labels_jt is None:
        print(' labels is None in one of files, skip NMS compare')
        return

    # shapes
    print(' dets shapes: PT', dets_pt.shape, 'JT', dets_jt.shape)
    print(' labels shapes: PT', labels_pt.shape, 'JT', labels_jt.shape)
    # counts
    n_pt = dets_pt.shape[0]
    n_jt = dets_jt.shape[0]
    print(' num boxes: PT', n_pt, 'JT', n_jt)

    # top-k comparison (by current order)
    k = min(topk, n_pt, n_jt)
    if k == 0:
        print(' no boxes to compare')
        return
    pt_top = dets_pt[:k]
    jt_top = dets_jt[:k]
    d_xy = np.abs(pt_top[:, :4] - jt_top[:, :4])
    d_sc = np.abs(pt_top[:, 4] - jt_top[:, 4])
    mae_xy = float(d_xy.mean())
    max_xy = float(d_xy.max())
    mae_sc = float(d_sc.mean())
    max_sc = float(d_sc.max())
    print(f' top-{k} xyxy: MAE={mae_xy:.4f}, max={max_xy:.4f} (thr {xyxy_mae_thr})')
    print(f' top-{k} score: MAE={mae_sc:.6f}, max={max_sc:.6f} (thr {score_mae_thr})')

    lbl_top_eq = int((labels_pt[:k] == labels_jt[:k]).sum())
    print(f' top-{k} labels equal: {lbl_top_eq}/{k}')

    # verdicts
    ok_xy = mae_xy <= xyxy_mae_thr
    ok_sc = mae_sc <= score_mae_thr
    print(' verdict:', 'PASS' if (ok_xy and ok_sc and (lbl_top_eq == k) and (n_pt == n_jt)) else 'CHECK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', required=True)
    ap.add_argument('--jt', required=True)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--xy_thr', type=float, default=0.2)
    ap.add_argument('--sc_thr', type=float, default=1e-3)
    args = ap.parse_args()
    pt = np.load(args.pt, allow_pickle=True)
    jt = np.load(args.jt, allow_pickle=True)

    print('Compare files:')
    print(' PT:', args.pt)
    print(' JT:', args.jt)

    # include warped dets if present
    if 'dets_warped' in pt and 'dets_warped' in jt:
        print('\nCompare dets_warped on original image scale:')
        dets_pt = pt['dets_warped']; dets_jt = jt['dets_warped']
        if dets_pt is not None and dets_jt is not None and dets_pt.shape == dets_jt.shape and dets_pt.shape[0]>0:
            k2 = min(args.topk, dets_pt.shape[0])
            dxy = np.abs(dets_pt[:k2,:4] - dets_jt[:k2,:4])
            dsc = np.abs(dets_pt[:k2,4] - dets_jt[:k2,4])
            print(f' warped top-{k2} xyxy MAE={float(dxy.mean()):.4f}, max={float(dxy.max()):.4f}')
            print(f' warped top-{k2} score MAE={float(dsc.mean()):.6f}, max={float(dsc.max()):.6f}')
        else:
            print(' dets_warped missing or shape mismatch')

    for k in KEYS:
        if k not in pt or k not in jt:
            print(f' - {k}: missing in one of files')
            continue
        s = stat_diff(pt[k], jt[k])
        print(f' - {k}:', s)

    # Optional: show top-5 largest diffs for bboxes if shapes match
    if 'bboxes' in pt and 'bboxes' in jt and pt['bboxes'].shape == jt['bboxes'].shape:
        d = np.abs(pt['bboxes'] - jt['bboxes'])
        flat = d.reshape(-1)
        idx = np.argpartition(flat, -5)[-5:]
        top = sorted([(int(i), float(flat[i])) for i in idx], key=lambda x: -x[1])
        print(' top bbox diffs (flat idx, abs):', top)

    # Grouped stats per stride level if center_priors are present and match
    if 'center_priors' in pt and 'center_priors' in jt and pt['center_priors'].shape == jt['center_priors'].shape:
        strides = pt['center_priors'][..., 2].reshape(-1)
        uniq = sorted(list(set(strides.tolist())))
        print('\nPer-stride mean_abs diffs:')
        for s_val in uniq:
            mask = (strides == s_val)
            idxs = np.where(mask)[0]
            print(f' stride={s_val}: count={idxs.size}')
            for k in ['reg_logits','prob','dis_only','dis_preds','bboxes','scores']:
                if k in pt and k in jt and pt[k].shape == jt[k].shape:
                    a = pt[k].reshape((-1,)+pt[k].shape[2:])[idxs]
                    b = jt[k].reshape((-1,)+jt[k].shape[2:])[idxs]
                    d = np.abs(a-b)
                    print(f'  - {k}: mean_abs={float(d.mean()):.4e}, max_abs={float(d.max()):.4e}')

    # Compare NMS outputs
    compare_dets_labels(pt, jt, topk=args.topk, xyxy_mae_thr=args.xy_thr, score_mae_thr=args.sc_thr)


if __name__ == '__main__':
    main()
