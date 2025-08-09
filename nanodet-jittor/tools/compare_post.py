# -*- coding: utf-8 -*-
# Compare PT and JT post-process intermediates to locate first divergence
import os, argparse, numpy as np

KEYS = [
    'center_priors', 'dis_preds', 'bboxes', 'scores'
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

if __name__ == '__main__':
    main()

