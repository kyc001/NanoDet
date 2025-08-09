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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', required=True)
    ap.add_argument('--jt', required=True)
    args = ap.parse_args()
    pt = np.load(args.pt, allow_pickle=True)
    jt = np.load(args.jt, allow_pickle=True)

    print('Compare files:')
    print(' PT:', args.pt)
    print(' JT:', args.jt)

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

if __name__ == '__main__':
    main()

