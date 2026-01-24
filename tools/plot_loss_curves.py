#!/usr/bin/env python3
"""
绘制训练Loss曲线和mAP曲线
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import re
import os
import numpy as np

def parse_pytorch_log(log_path):
    """解析PyTorch训练日志"""
    iterations = []
    losses = {'loss_qfl': [], 'loss_bbox': [], 'loss_dfl': [],
              'aux_loss_qfl': [], 'aux_loss_bbox': [], 'aux_loss_dfl': []}
    maps = {'mAP': [], 'AP_50': [], 'AP_75': []}
    map_epochs = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 解析每个iteration的loss
    loss_pattern = r'Train\|Epoch(\d+)/\d+\|Iter(\d+).*?loss_qfl:([0-9.]+)\|\s*loss_bbox:([0-9.]+)\|\s*loss_dfl:([0-9.]+)\|\s*aux_loss_qfl:([0-9.]+)\|\s*aux_loss_bbox:([0-9.]+)\|\s*aux_loss_dfl:([0-9.]+)'
    for line in lines:
        match = re.search(loss_pattern, line)
        if match:
            iteration = int(match.group(2))
            if iteration not in iterations:
                iterations.append(iteration)
                losses['loss_qfl'].append(float(match.group(3)))
                losses['loss_bbox'].append(float(match.group(4)))
                losses['loss_dfl'].append(float(match.group(5)))
                losses['aux_loss_qfl'].append(float(match.group(6)))
                losses['aux_loss_bbox'].append(float(match.group(7)))
                losses['aux_loss_dfl'].append(float(match.group(8)))

    # 解析mAP - 按行提取，并去重
    mAP_pattern = r'Average Precision.*IoU=0\.50:0\.95.*area=\s*all.*= ([0-9.]+)'
    AP50_pattern = r'Average Precision.*IoU=0\.50\s+\|.*area=\s*all.*= ([0-9.]+)'
    AP75_pattern = r'Average Precision.*IoU=0\.75\s+\|.*area=\s*all.*= ([0-9.]+)'

    mAP_values = []
    AP50_values = []
    AP75_values = []

    for line in lines:
        m = re.search(mAP_pattern, line)
        if m:
            mAP_values.append(float(m.group(1)))
        m = re.search(AP50_pattern, line)
        if m:
            AP50_values.append(float(m.group(1)))
        m = re.search(AP75_pattern, line)
        if m:
            AP75_values.append(float(m.group(1)))

    # 每个验证周期输出两次，取第一次 (偶数索引: 0, 2, 4, 6, 8)
    val_epochs = [10, 20, 30, 40, 50]
    for i, e in enumerate(val_epochs):
        idx = i * 2
        if idx < len(mAP_values):
            map_epochs.append(e)
            maps['mAP'].append(mAP_values[idx])
            maps['AP_50'].append(AP50_values[idx] if idx < len(AP50_values) else 0)
            maps['AP_75'].append(AP75_values[idx] if idx < len(AP75_values) else 0)

    return iterations, losses, map_epochs, maps

def parse_jittor_log(log_path):
    """解析Jittor训练日志"""
    iterations = []
    losses = {'loss_qfl': [], 'loss_bbox': [], 'loss_dfl': [],
              'aux_loss_qfl': [], 'aux_loss_bbox': [], 'aux_loss_dfl': []}
    maps = {'mAP': [], 'AP_50': [], 'AP_75': []}
    map_epochs = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 解析loss
    loss_pattern = r'Train\|Epoch(\d+)/\d+\|Iter(\d+).*?loss_qfl:([0-9.]+)\|\s*loss_bbox:([0-9.]+)\|\s*loss_dfl:([0-9.]+)\|\s*aux_loss_qfl:([0-9.]+)\|\s*aux_loss_bbox:([0-9.]+)\|\s*aux_loss_dfl:([0-9.]+)'
    for line in lines:
        match = re.search(loss_pattern, line)
        if match:
            iteration = int(match.group(2))
            if iteration not in iterations:
                iterations.append(iteration)
                losses['loss_qfl'].append(float(match.group(3)))
                losses['loss_bbox'].append(float(match.group(4)))
                losses['loss_dfl'].append(float(match.group(5)))
                losses['aux_loss_qfl'].append(float(match.group(6)))
                losses['aux_loss_bbox'].append(float(match.group(7)))
                losses['aux_loss_dfl'].append(float(match.group(8)))

    # 解析mAP - 按行提取
    mAP_pattern = r'Average Precision.*IoU=0\.50:0\.95.*area=\s*all.*= ([0-9.]+)'
    AP50_pattern = r'Average Precision.*IoU=0\.50\s+\|.*area=\s*all.*= ([0-9.]+)'
    AP75_pattern = r'Average Precision.*IoU=0\.75\s+\|.*area=\s*all.*= ([0-9.]+)'

    mAP_values = []
    AP50_values = []
    AP75_values = []

    for line in lines:
        m = re.search(mAP_pattern, line)
        if m:
            mAP_values.append(float(m.group(1)))
        m = re.search(AP50_pattern, line)
        if m:
            AP50_values.append(float(m.group(1)))
        m = re.search(AP75_pattern, line)
        if m:
            AP75_values.append(float(m.group(1)))

    # Jittor日志：前2个是中途快速评估（iter=100），后面每个验证周期重复两次
    # 跳过前2个，然后每隔2个取一个
    val_epochs = [10, 20, 30, 40, 50]
    for i, e in enumerate(val_epochs):
        idx = 2 + i * 2  # 跳过中途评估的2个，然后每个epoch取第一个
        if idx < len(mAP_values):
            map_epochs.append(e)
            maps['mAP'].append(mAP_values[idx])
            maps['AP_50'].append(AP50_values[idx] if idx < len(AP50_values) else 0)
            maps['AP_75'].append(AP75_values[idx] if idx < len(AP75_values) else 0)

    return iterations, losses, map_epochs, maps

def plot_loss_curves(pt_iters, pt_losses, jt_iters, jt_losses, output_dir):
    """绘制Loss曲线对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Loss Curves: PyTorch vs Jittor', fontsize=14)

    loss_names = ['loss_qfl', 'loss_bbox', 'loss_dfl', 'aux_loss_qfl', 'aux_loss_bbox', 'aux_loss_dfl']
    titles = ['QFL Loss', 'BBox Loss', 'DFL Loss', 'Aux QFL Loss', 'Aux BBox Loss', 'Aux DFL Loss']

    for idx, (name, title) in enumerate(zip(loss_names, titles)):
        ax = axes[idx // 3, idx % 3]
        if pt_losses[name]:
            ax.plot(pt_iters[:len(pt_losses[name])], pt_losses[name], 'b-', label='PyTorch', linewidth=1.5, alpha=0.8)
        if jt_losses[name]:
            ax.plot(jt_iters[:len(jt_losses[name])], jt_losses[name], 'r-', label='Jittor', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss curves saved to: {output_path}")
    plt.close()
    return output_path

def plot_map_curves(pt_epochs, pt_maps, jt_epochs, jt_maps, output_dir):
    """绘制mAP曲线对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Validation mAP Curves: PyTorch vs Jittor', fontsize=14)

    map_names = ['mAP', 'AP_50', 'AP_75']
    titles = ['mAP (IoU=0.50:0.95)', 'AP@IoU=0.50', 'AP@IoU=0.75']

    for idx, (name, title) in enumerate(zip(map_names, titles)):
        ax = axes[idx]
        if pt_maps[name] and pt_epochs:
            ax.plot(pt_epochs[:len(pt_maps[name])], pt_maps[name], 'b-o', label='PyTorch', linewidth=2, markersize=8)
        if jt_maps[name] and jt_epochs:
            ax.plot(jt_epochs[:len(jt_maps[name])], jt_maps[name], 'r--s', label='Jittor', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'map_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"mAP curves saved to: {output_path}")
    plt.close()
    return output_path

def plot_fps_comparison(output_dir):
    """绘制FPS对比图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    frameworks = ['PyTorch', 'Jittor']
    fps = [109.0, 114.5]
    times = [9.18, 8.73]
    colors = ['#3498db', '#e74c3c']

    x = np.arange(len(frameworks))
    width = 0.35

    bars1 = ax.bar(x - width/2, fps, width, label='FPS', color=colors, alpha=0.8)
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_xlabel('Framework', fontsize=12)
    ax.set_title('Inference Performance Comparison (RTX 3090, 320x320)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks)
    ax.legend(loc='upper left')

    # 添加数值标签
    for bar, f, t in zip(bars1, fps, times):
        height = bar.get_height()
        ax.annotate(f'{f:.1f} FPS\n({t:.2f}ms)',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)

    ax.set_ylim(0, 140)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fps_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"FPS comparison saved to: {output_path}")
    plt.close()
    return output_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_log', type=str, default='workspace/pytorch_full_train.log')
    parser.add_argument('--jt_log', type=str, default='workspace/jittor_full_train.log')
    parser.add_argument('--output_dir', type=str, default='workspace/figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Parsing PyTorch log...")
    pt_iters, pt_losses, pt_map_epochs, pt_maps = parse_pytorch_log(args.pt_log)
    print(f"  Found {len(pt_iters)} iterations, {len(pt_map_epochs)} validation points")
    print(f"  PyTorch mAP values: {pt_maps['mAP']}")
    print(f"  PyTorch AP_50 values: {pt_maps['AP_50']}")

    print("Parsing Jittor log...")
    jt_iters, jt_losses, jt_map_epochs, jt_maps = parse_jittor_log(args.jt_log)
    print(f"  Found {len(jt_iters)} iterations, {len(jt_map_epochs)} validation points")
    print(f"  Jittor mAP values: {jt_maps['mAP']}")
    print(f"  Jittor AP_50 values: {jt_maps['AP_50']}")

    print("\nGenerating plots...")
    plot_loss_curves(pt_iters, pt_losses, jt_iters, jt_losses, args.output_dir)
    plot_map_curves(pt_map_epochs, pt_maps, jt_map_epochs, jt_maps, args.output_dir)
    plot_fps_comparison(args.output_dir)

    print("\nDone!")
