#!/usr/bin/env python3
"""
批量生成 PyTorch vs Jittor 真实可视化对比图
流程：
1) 将 PyTorch ckpt (.ckpt/.pth) 转为 Jittor 可加载的 .pkl（使用 tools/convert_checkpoint.py）
2) 分别用 vis_batch.py 生成两套可视化：
   - PT(经转换的权重) => workspace/vis_pt
   - JT(原生Jittor权重) => workspace/vis_jt
3) 将相同文件名的两张可视化图拼接成对比图，输出到 DELIVERABLES/images/pt_vs_jt_comparison
"""

import os
import subprocess
import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JT_ROOT = ROOT / 'nanodet-jittor'
DELIV = ROOT / 'DELIVERABLES'

CFG = JT_ROOT / 'config' / 'nanodet-plus-m_320_voc_bs64_50epochs.yml'
PT_CKPT = ROOT / 'nanodet-pytorch' / 'workspace' / 'nanodet-plus-m_320_voc_bs64_50epochs' / 'model_best' / 'model_best.ckpt'
JT_CKPT = ROOT / 'workspace' / 'jittor_50epochs_model_best.pkl'
PT2JT_OUT = ROOT / 'workspace' / 'pt2jt_model_best.pkl'

VIS_PT = ROOT / 'workspace' / 'vis_pt'
VIS_JT = ROOT / 'workspace' / 'vis_jt'
OUT_DIR = DELIV / 'images' / 'pt_vs_jt_comparison'


def run(cmd, cwd=None, timeout=600):
    print(f"$ {' '.join(map(str, cmd))}")
    return subprocess.run(list(map(str, cmd)), cwd=cwd, timeout=timeout, capture_output=True, text=True)


def ensure_conversion():
    if not PT_CKPT.exists():
        raise FileNotFoundError(f"PyTorch 权重不存在: {PT_CKPT}")
    if PT2JT_OUT.exists():
        print(f"[skip] 已存在转换结果: {PT2JT_OUT}")
        return
    cmd = [
        sys.executable, str(JT_ROOT / 'tools' / 'convert_checkpoint.py'),
        '--mode', 'pt2jt',
        '--cfg', str(CFG),
        '--in_ckpt', str(PT_CKPT),
        '--out_ckpt', str(PT2JT_OUT),
        '--prefer_avg'
    ]
    r = run(cmd, cwd=str(JT_ROOT))
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError('PT->JT 转换失败')
    print('[OK] PT->JT 转换完成')


def run_vis(ckpt_path: Path, out_dir: Path, num: int = 12, score_thr: float = 0.35):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")
    cmd = [
        sys.executable, str(JT_ROOT / 'tools' / 'vis_batch.py'),
        '--config', str(CFG),
        '--ckpt', str(ckpt_path),
        '--out_dir', str(out_dir),
        '--num', str(num),
        '--score_thr', str(score_thr)
    ]
    r = run(cmd, cwd=str(JT_ROOT))
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(f'vis_batch 失败: {ckpt_path}')


def stitch_pair(img_left_path: Path, img_right_path: Path, out_path: Path, left_title='PyTorch', right_title='Jittor'):
    img_l = cv2.imread(str(img_left_path))
    img_r = cv2.imread(str(img_right_path))
    if img_l is None or img_r is None:
        print(f"[warn] 跳过：读取失败 {img_left_path} 或 {img_right_path}")
        return False
    # 对齐高度
    h = max(img_l.shape[0], img_r.shape[0])
    w_l = int(img_l.shape[1] * h / img_l.shape[0])
    w_r = int(img_r.shape[1] * h / img_r.shape[0])
    img_lr = cv2.resize(img_l, (w_l, h))
    img_rr = cv2.resize(img_r, (w_r, h))
    gap = np.ones((h, 10, 3), dtype=np.uint8) * 255
    canvas = np.concatenate([img_lr, gap, img_rr], axis=1)
    # 简单标题条
    bar_h = 40
    bar = np.ones((bar_h, canvas.shape[1], 3), dtype=np.uint8) * 255
    canvas = np.concatenate([bar, canvas], axis=0)
    cv2.putText(canvas, left_title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.putText(canvas, right_title, (10 + w_l + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return True


def compose_comparisons():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    left_dir = VIS_PT
    right_dir = VIS_JT
    names_left = {p.stem.replace('_det',''): p for p in left_dir.glob('*_det.jpg')}
    names_right = {p.stem.replace('_det',''): p for p in right_dir.glob('*_det.jpg')}
    common = sorted(set(names_left.keys()) & set(names_right.keys()))
    if not common:
        print('[warn] 两侧没有重叠的可视化图片，无法拼接对比')
        return 0
    cnt = 0
    for name in common:
        out_path = OUT_DIR / f'{name}_pt_vs_jt.jpg'
        ok = stitch_pair(names_left[name], names_right[name], out_path)
        if ok:
            print(f'[OK] {out_path}')
            cnt += 1
    return cnt


def main():
    print('== 生成 PT(经转换) vs JT(训练) 真实检测对比图 ==')
    # 1) 转换
    ensure_conversion()
    # 2) 生成两套可视化
    run_vis(PT2JT_OUT, VIS_PT, num=12, score_thr=0.35)
    run_vis(JT_CKPT, VIS_JT, num=12, score_thr=0.35)
    # 3) 拼接
    made = compose_comparisons()
    print(f'== 完成，对比图数量: {made}, 输出目录: {OUT_DIR} ==')

if __name__ == '__main__':
    main()

