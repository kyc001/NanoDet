# -*- coding: utf-8 -*-
import re
import os
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pat_epoch = re.compile(r"Epoch\s+(\d+)/(\d+)")
pat_loss = re.compile(r"Loss:\s+([0-9.]+)")
pat_map  = re.compile(r"mAP:\s+([0-9.]+)")


def parse_log(path):
    epochs = []
    losses = []
    maps = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m1 = pat_epoch.search(line)
            if m1:
                cur_epoch = int(m1.group(1))
            m2 = pat_loss.search(line)
            if m2 and 'Epoch' in line:
                losses.append((cur_epoch, float(m2.group(1))))
            m3 = pat_map.search(line)
            if m3:
                maps.append((cur_epoch, float(m3.group(1))))
    # 聚合到每个 epoch
    epoch_loss = {}
    for ep, l in losses:
        epoch_loss[ep] = l
    epoch_map = {}
    for ep, v in maps:
        epoch_map[ep] = v
    all_eps = sorted(set(list(epoch_loss.keys()) + list(epoch_map.keys())))
    rows = []
    for ep in all_eps:
        rows.append({
            'epoch': ep,
            'loss': epoch_loss.get(ep, None),
            'mAP': epoch_map.get(ep, None)
        })
    return rows


def save_csv(rows, out_csv):
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch','loss','mAP'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_curves(rows, out_png):
    xs = [r['epoch'] for r in rows]
    loss = [r['loss'] for r in rows]
    maps = [r['mAP'] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.set_title('Training Loss & mAP over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(xs, loss, '-o', color='tab:blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('mAP', color='tab:orange')
    ax2.plot(xs, maps, '-s', color='tab:orange', label='mAP')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--out_dir', default='workspace/train_analysis')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = parse_log(args.log)
    out_csv = os.path.join(args.out_dir, 'metrics.csv')
    out_png = os.path.join(args.out_dir, 'curves.png')
    save_csv(rows, out_csv)
    plot_curves(rows, out_png)
    print('Saved:', out_csv, out_png)

if __name__ == '__main__':
    main()

