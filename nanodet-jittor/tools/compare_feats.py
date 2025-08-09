# Compare dumped features between PyTorch and Jittor
import os, argparse, numpy as np

def stat(name, a, b):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape:
        return f"{name}: shape mismatch {a.shape} vs {b.shape}"
    diff=np.abs(a-b)
    return f"{name}: shape={a.shape}, mean|Δ|={diff.mean():.4e}, max|Δ|={diff.max():.4e}"

ap=argparse.ArgumentParser();
ap.add_argument('--pt', required=True);
ap.add_argument('--jt', required=True);
args=ap.parse_args()
pt=np.load(args.pt)
jt=np.load(args.jt)
keys=['bb0','bb1','bb2','fpn0','fpn1','fpn2','fpn3','head']
for k in keys:
    if k in pt.files and k in jt.files:
        print(stat(k, pt[k], jt[k]))
    else:
        print(f"missing {k}: pt has {k in pt.files}, jt has {k in jt.files}")

