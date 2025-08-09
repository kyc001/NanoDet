# -*- coding: utf-8 -*-
"""Compare PT vs JT head stats JSON files and print differences."""
import json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', required=True)
    ap.add_argument('--jt', required=True)
    args = ap.parse_args()
    with open(args.pt, 'r') as f:
        pt = json.load(f)
    with open(args.jt, 'r') as f:
        jt = json.load(f)

    def stat_diff(a, b, key):
        if a is None or b is None:
            return {"key": key, "pt": a, "jt": b}
        if a.get('shape') != b.get('shape'):
            return {"key": key, "shape_pt": a.get('shape'), "shape_jt": b.get('shape')}
        out = {"key": key}
        for k in ["mean", "std", "min", "max"]:
            out[k] = {
                "pt": a.get(k),
                "jt": b.get(k),
                "abs_diff": None if (a.get(k) is None or b.get(k) is None) else abs(a.get(k)-b.get(k)),
            }
        return out

    print("Compare gfl_cls weights/bias:")
    for name, a in pt.get('gfl_cls', {}).items():
        b = jt.get('gfl_cls', {}).get(name)
        print(stat_diff(a, b, f"gfl_cls.{name}"))

    print("\nCompare BN stats:")
    pt_bn = pt.get('bn', {})
    jt_bn = jt.get('bn', {})
    keys = sorted(set(list(pt_bn.keys()) + list(jt_bn.keys())))
    for k in keys:
        if k.endswith('.hyper'):
            print({"key": k, "pt": pt_bn.get(k), "jt": jt_bn.get(k)})
        else:
            print(stat_diff(pt_bn.get(k), jt_bn.get(k), k))

    print("\nCompare misc:")
    print({'pt': pt.get('misc'), 'jt': jt.get('misc')})

if __name__ == '__main__':
    main()

