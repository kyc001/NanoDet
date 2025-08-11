# -*- coding: utf-8 -*-
import re
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 日志解析正则
pat_epoch_any   = re.compile(r"Epoch\s+(\d+)/(\d+)")
pat_map         = re.compile(r"mAP:\s+([0-9.]+)")
# 训练阶段聚合：示例行
# Train|Epoch1/5|Iter123(4/200)| mem:0.00G| lr:1.00e-03| loss_qfl:0.1000| loss_dfl:0.0500| loss_bbox:1.2000|
pat_train_line  = re.compile(r"Train\|Epoch(\d+)/(\d+)\|.*?lr:([0-9eE+\-.]+)\|(?P<tail>.*)")
pat_kv          = re.compile(r"(loss_[a-zA-Z0-9_]+):\s*([0-9.]+)")
# 验证阶段可能输出 Loss: x.x 或 mAP: x.x
pat_val_loss    = re.compile(r"Loss:\s+([0-9.]+)")


def parse_log(path):
    # 按 epoch 聚合
    agg = {}
    def ensure(ep):
        if ep not in agg:
            agg[ep] = {
                'count': 0,
                'loss': 0.0,
                'loss_qfl': None,
                'loss_dfl': None,
                'loss_bbox': None,
                'lr': None,
                'mAP': None,
            }
        return agg[ep]

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        cur_epoch = None
        for line in f:
            # 训练行
            mt = pat_train_line.search(line)
            if mt:
                ep = int(mt.group(1))
                lr = float(mt.group(3)) if mt.group(3) not in (None, '') else None
                tail = mt.group('tail')
                epd = ensure(ep)
                # 解析 tail 中的各项 loss
                kvs = dict((k, float(v)) for k, v in pat_kv.findall(tail))
                # 统计：总 loss 用子项加总替代（如可用），否则忽略
                total = 0.0
                used = False
                for name in ('loss_qfl','loss_dfl','loss_bbox'):
                    if name in kvs:
                        epd[name] = kvs[name]
                        total += kvs[name]
                        used = True
                if used:
                    epd['loss'] = total
                # 学习率记录最后一次
                epd['lr'] = lr
                epd['count'] += 1
                continue
            # 验证行中的 mAP
            mm = pat_map.search(line)
            if mm:
                # 最近一次出现的 Epoch 作为该 mAP 的 epoch
                me = pat_epoch_any.search(line)
                ep = int(me.group(1)) if me else (cur_epoch if cur_epoch is not None else 0)
                epd = ensure(ep)
                epd['mAP'] = float(mm.group(1))
                continue
            # 记录最近出现的 Epoch 号（用于回退）
            me2 = pat_epoch_any.search(line)
            if me2:
                cur_epoch = int(me2.group(1))
                continue

    rows = []
    for ep in sorted(agg.keys()):
        v = agg[ep]
        rows.append({
            'epoch': ep,
            'loss': v['loss'],
            'loss_qfl': v['loss_qfl'],
            'loss_dfl': v['loss_dfl'],
            'loss_bbox': v['loss_bbox'],
            'lr': v['lr'],
            'mAP': v['mAP'],
        })
    return rows


def save_csv(rows, out_csv):
    import csv
    with open(out_csv, 'w', newline='') as f:
        fields = ['epoch','loss','loss_qfl','loss_dfl','loss_bbox','lr','mAP']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_loss_map(rows, out_png):
    xs = [r['epoch'] for r in rows]
    loss = [r['loss'] for r in rows]
    maps = [r['mAP'] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.set_title('Loss & mAP over Epochs')
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


def plot_loss_details(rows, out_png):
    xs = [r['epoch'] for r in rows]
    qfl = [r['loss_qfl'] for r in rows]
    dfl = [r['loss_dfl'] for r in rows]
    giou= [r['loss_bbox'] for r in rows]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_title('Sub-loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(xs, qfl, '-o', label='loss_qfl')
    ax.plot(xs, dfl, '-o', label='loss_dfl')
    ax.plot(xs, giou,'-o', label='loss_bbox')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def plot_lr(rows, out_png):
    xs = [r['epoch'] for r in rows]
    lr = [r['lr'] for r in rows]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_title('Learning Rate over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.plot(xs, lr, '-o', color='tab:green')
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
    save_csv(rows, out_csv)
    plot_loss_map(rows, os.path.join(args.out_dir, 'curves.png'))
    plot_loss_details(rows, os.path.join(args.out_dir, 'loss_details.png'))
    plot_lr(rows, os.path.join(args.out_dir, 'lr.png'))
    print('Saved:', out_csv, os.path.join(args.out_dir, 'curves.png'), os.path.join(args.out_dir, 'loss_details.png'), os.path.join(args.out_dir, 'lr.png'))

if __name__ == '__main__':
    main()
