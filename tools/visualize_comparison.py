#!/usr/bin/env python3
"""
PyTorch vs Jittor 目标检测可视化对比脚本
对同一张图片分别用两个框架推理，并排展示检测结果
支持显示 Ground Truth (GT) 标注框
"""

import argparse
import os
import sys
import cv2
import numpy as np
import subprocess
import pickle
import xml.etree.ElementTree as ET

# VOC类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 类别名称到索引的映射
VOC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}

# 颜色表
_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125,
    0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933,
    0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600,
    1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000,
    0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000,
    0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000,
]).astype(np.float32).reshape(-1, 3)


def parse_voc_annotation(xml_path):
    """解析 VOC XML 标注文件，返回 GT 框列表"""
    gt_boxes = []
    if not os.path.exists(xml_path):
        return gt_boxes

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in VOC_CLASS_TO_IDX:
            continue

        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue  # 跳过 difficult 样本

        label = VOC_CLASS_TO_IDX[name]
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        gt_boxes.append({
            'label': label,
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return gt_boxes


def draw_gt_boxes(img, gt_boxes, class_names):
    """在图像上绘制 GT 框（绿色虚线）"""
    img = img.copy()
    for gt in gt_boxes:
        label = gt['label']
        x0, y0, x1, y1 = gt['bbox']
        name = gt['name']

        # GT 框用绿色
        color = (0, 255, 0)

        # 绘制虚线边框
        draw_dashed_rectangle(img, (x0, y0), (x1, y1), color, thickness=2, dash_length=10)

        # 绘制标签（右下角，与检测框区分）
        text = f"GT:{name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

        # 标签放在框的右下角
        txt_x = max(x1 - txt_size[0] - 4, x0)
        txt_y = y1

        cv2.rectangle(img, (txt_x, txt_y),
                      (txt_x + txt_size[0] + 4, txt_y + txt_size[1] + 4),
                      color, -1)
        cv2.putText(img, text, (txt_x + 2, txt_y + txt_size[1] + 2),
                    font, 0.4, (0, 0, 0), thickness=1)

    return img


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    """绘制虚线矩形"""
    x1, y1 = pt1
    x2, y2 = pt2

    # 绘制四条虚线边
    draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_length)  # 上
    draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_length)  # 下
    draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_length)  # 左
    draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_length)  # 右


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """绘制虚线"""
    x1, y1 = pt1
    x2, y2 = pt2

    # 计算线段长度和方向
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx*dx + dy*dy)

    if dist == 0:
        return

    # 单位向量
    ux = dx / dist
    uy = dy / dist

    # 绘制虚线段
    pos = 0
    draw = True
    while pos < dist:
        if draw:
            end_pos = min(pos + dash_length, dist)
            sx = int(x1 + ux * pos)
            sy = int(y1 + uy * pos)
            ex = int(x1 + ux * end_pos)
            ey = int(y1 + uy * end_pos)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        pos += dash_length
        draw = not draw


def draw_detections(img, dets, class_names, score_thresh=0.35):
    """在图像上绘制检测框"""
    img = img.copy()
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])

    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        color = (_COLORS[label % len(_COLORS)] * 255).astype(np.uint8).tolist()
        text = "{}:{:.0f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label % len(_COLORS)]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]

        # 绘制边框
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        # 绘制标签背景
        cv2.rectangle(img, (x0, y0 - txt_size[1] - 4),
                      (x0 + txt_size[0] + 4, y0), color, -1)
        # 绘制标签文字
        cv2.putText(img, text, (x0 + 2, y0 - 2), font, 0.5, txt_color, thickness=1)

    return img


def run_pytorch_inference(config_path, model_path, image_paths, output_path):
    """运行PyTorch推理并保存结果"""
    script = f'''
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/nanodet-pytorch")
import torch
import cv2
import pickle
import numpy as np
from nanodet.util import cfg, load_config, load_model_weight
from nanodet.model.arch import build_model

# 图像预处理（与训练时一致）
def preprocess(img, input_size=(320, 320)):
    """图像预处理，保持比例resize并居中padding"""
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    h, w = img.shape[:2]
    d_h, d_w = input_size

    # 计算缩放比例
    scale = min(d_h / h, d_w / w)

    # 计算warp_matrix（居中对齐）
    # 1. 将原点移到原图中心
    C = np.eye(3)
    C[0, 2] = -w / 2
    C[1, 2] = -h / 2

    # 2. 缩放
    S = np.eye(3)
    S[0, 0] = scale
    S[1, 1] = scale

    # 3. 将原点移到目标图像中心
    T = np.eye(3)
    T[0, 2] = d_w / 2
    T[1, 2] = d_h / 2

    warp_matrix = T @ S @ C

    # 使用warpPerspective进行变换
    padded = cv2.warpPerspective(img, warp_matrix, (d_w, d_h), borderValue=(114, 114, 114))

    # normalize
    padded = padded.astype(np.float32)
    padded = (padded - mean) / std

    return padded, warp_matrix

load_config(cfg, "{config_path}")
model = build_model(cfg.model)
ckpt = torch.load("{model_path}", map_location='cuda')
if 'state_dict' in ckpt:
    state_dict = {{k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}}
    if not state_dict:
        state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict, strict=False)
model = model.cuda().eval()

results = {{}}
image_paths = {image_paths}
for img_path in image_paths:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    preprocessed, warp_matrix = preprocess(img, (320, 320))
    img_tensor = torch.from_numpy(preprocessed.transpose(2, 0, 1)).unsqueeze(0).cuda()

    img_info = {{"height": [height], "width": [width], "id": [0]}}
    meta = {{"img_info": img_info, "raw_img": img, "img": img_tensor, "warp_matrix": [warp_matrix]}}

    with torch.no_grad():
        dets = model.inference(meta)

    # 转换为可序列化格式
    dets_serializable = {{}}
    for k, v in dets.items():
        if hasattr(v, 'numpy'):
            dets_serializable[k] = v.cpu().numpy().tolist()
        elif hasattr(v, 'tolist'):
            dets_serializable[k] = v.tolist()
        else:
            dets_serializable[k] = v
    results[img_path] = dets_serializable
    print(f"PyTorch processed: {{os.path.basename(img_path)}}")

with open("{output_path}", "wb") as f:
    pickle.dump(results, f)
print("PyTorch inference done!")
'''

    script_path = "/tmp/pt_inference_tmp.py"
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run([sys.executable, script_path],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("PyTorch Error:", result.stderr)
        return False
    print(result.stdout)
    return True


def run_jittor_inference(config_path, model_path, image_paths, output_path):
    """运行Jittor推理并保存结果"""
    script = f'''
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/nanodet-jittor")
import jittor as jt
jt.flags.use_cuda = 1
import cv2
import pickle
import numpy as np
from nanodet.util import cfg, load_config, load_model_weight
from nanodet.model.arch import build_model

# 图像预处理（与训练时一致）
def preprocess(img, input_size=(320, 320)):
    """图像预处理，保持比例resize并居中padding"""
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)

    h, w = img.shape[:2]
    d_h, d_w = input_size

    # 计算缩放比例
    scale = min(d_h / h, d_w / w)

    # 计算warp_matrix（居中对齐）
    # 1. 将原点移到原图中心
    C = np.eye(3)
    C[0, 2] = -w / 2
    C[1, 2] = -h / 2

    # 2. 缩放
    S = np.eye(3)
    S[0, 0] = scale
    S[1, 1] = scale

    # 3. 将原点移到目标图像中心
    T = np.eye(3)
    T[0, 2] = d_w / 2
    T[1, 2] = d_h / 2

    warp_matrix = T @ S @ C

    # 使用warpPerspective进行变换
    padded = cv2.warpPerspective(img, warp_matrix, (d_w, d_h), borderValue=(114, 114, 114))

    # normalize
    padded = padded.astype(np.float32)
    padded = (padded - mean) / std

    return padded, warp_matrix

load_config(cfg, "{config_path}")
model = build_model(cfg.model)
ckpt = jt.load("{model_path}")
if 'state_dict' not in ckpt:
    ckpt = {{'state_dict': ckpt}}
load_model_weight(model, ckpt, None)
model.eval()

results = {{}}
image_paths = {image_paths}
for img_path in image_paths:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    preprocessed, warp_matrix = preprocess(img, (320, 320))
    img_tensor = jt.array(preprocessed.transpose(2, 0, 1)).unsqueeze(0)

    img_info = {{"height": [height], "width": [width], "id": [0]}}
    meta = {{"img_info": img_info, "raw_img": img, "img": img_tensor, "warp_matrix": [warp_matrix]}}

    with jt.no_grad():
        dets = model.inference(meta)

    # 转换为可序列化格式
    dets_serializable = {{}}
    for k, v in dets.items():
        if hasattr(v, 'numpy'):
            dets_serializable[k] = v.numpy().tolist()
        elif hasattr(v, 'tolist'):
            dets_serializable[k] = v.tolist()
        else:
            dets_serializable[k] = v
    results[img_path] = dets_serializable
    print(f"Jittor processed: {{os.path.basename(img_path)}}")

with open("{output_path}", "wb") as f:
    pickle.dump(results, f)
print("Jittor inference done!")
'''

    script_path = "/tmp/jt_inference_tmp.py"
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run([sys.executable, script_path],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("Jittor Error:", result.stderr)
        return False
    print(result.stdout)
    return True


def create_comparison_image(img_path, pt_dets, jt_dets, gt_boxes, output_path, score_thresh=0.35):
    """创建对比图像：左边PyTorch，右边Jittor，带GT框"""
    img = cv2.imread(img_path)

    # 先绘制 GT 框（绿色虚线），再绘制检测框
    if gt_boxes:
        pt_img = draw_gt_boxes(img, gt_boxes, VOC_CLASSES)
        jt_img = draw_gt_boxes(img, gt_boxes, VOC_CLASSES)
    else:
        pt_img = img.copy()
        jt_img = img.copy()

    # 绘制检测结果
    pt_img = draw_detections(pt_img, pt_dets, VOC_CLASSES, score_thresh)
    jt_img = draw_detections(jt_img, jt_dets, VOC_CLASSES, score_thresh)

    # 添加标题
    h, w = img.shape[:2]
    title_h = 40

    # 创建带标题的图像
    pt_with_title = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    pt_with_title[title_h:, :] = pt_img
    cv2.rectangle(pt_with_title, (0, 0), (w, title_h), (50, 50, 50), -1)
    cv2.putText(pt_with_title, "PyTorch", (w//2 - 50, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    jt_with_title = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    jt_with_title[title_h:, :] = jt_img
    cv2.rectangle(jt_with_title, (0, 0), (w, title_h), (50, 50, 50), -1)
    cv2.putText(jt_with_title, "Jittor", (w//2 - 40, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 水平拼接
    comparison = np.hstack([pt_with_title, jt_with_title])

    cv2.imwrite(output_path, comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_config', type=str, required=True)
    parser.add_argument('--jt_config', type=str, required=True)
    parser.add_argument('--pt_model', type=str, required=True)
    parser.add_argument('--jt_model', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--image_list', type=str, required=True)
    parser.add_argument('--annotation_dir', type=str, default=None,
                        help='VOC Annotations directory for GT boxes')
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='workspace/figures/detection_comparison')
    parser.add_argument('--score_thresh', type=float, default=0.35)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取图片列表
    with open(args.image_list, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    # 选择有代表性的图片（均匀采样）
    step = max(1, len(image_ids) // args.num_images)
    selected_ids = image_ids[::step][:args.num_images]
    image_paths = [os.path.join(args.image_dir, f"{img_id}.jpg") for img_id in selected_ids]

    print(f"Selected {len(image_paths)} images for comparison:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")

    # 运行推理
    pt_results_path = "/tmp/pt_results.pkl"
    jt_results_path = "/tmp/jt_results.pkl"

    print("\n" + "="*50)
    print("Running PyTorch inference...")
    print("="*50)
    if not run_pytorch_inference(args.pt_config, args.pt_model, image_paths, pt_results_path):
        print("PyTorch inference failed!")
        return

    print("\n" + "="*50)
    print("Running Jittor inference...")
    print("="*50)
    if not run_jittor_inference(args.jt_config, args.jt_model, image_paths, jt_results_path):
        print("Jittor inference failed!")
        return

    # 加载结果
    with open(pt_results_path, "rb") as f:
        pt_results = pickle.load(f)
    with open(jt_results_path, "rb") as f:
        jt_results = pickle.load(f)

    # 生成对比图
    print("\n" + "="*50)
    print("Generating comparison images...")
    if args.annotation_dir:
        print(f"GT annotations from: {args.annotation_dir}")
    print("="*50)

    for img_path in image_paths:
        pt_dets_raw = pt_results.get(img_path, {})
        jt_dets_raw = jt_results.get(img_path, {})

        # 结果格式是 {img_id: {class_id: [bboxes]}}，需要取出img_id=0的结果
        pt_dets = pt_dets_raw.get(0, {}) if isinstance(pt_dets_raw, dict) and 0 in pt_dets_raw else pt_dets_raw
        jt_dets = jt_dets_raw.get(0, {}) if isinstance(jt_dets_raw, dict) and 0 in jt_dets_raw else jt_dets_raw

        # 转换为正确的格式
        pt_dets_converted = {}
        jt_dets_converted = {}
        for k, v in pt_dets.items():
            pt_dets_converted[k] = np.array(v) if isinstance(v, list) else v
        for k, v in jt_dets.items():
            jt_dets_converted[k] = np.array(v) if isinstance(v, list) else v

        # 读取 GT 标注
        gt_boxes = []
        if args.annotation_dir:
            img_name = os.path.basename(img_path).replace('.jpg', '')
            xml_path = os.path.join(args.annotation_dir, f"{img_name}.xml")
            gt_boxes = parse_voc_annotation(xml_path)

        img_name = os.path.basename(img_path).replace('.jpg', '')
        output_path = os.path.join(args.output_dir, f"comparison_{img_name}.jpg")

        create_comparison_image(img_path, pt_dets_converted, jt_dets_converted,
                               gt_boxes, output_path, args.score_thresh)
        print(f"Saved: {output_path} (GT: {len(gt_boxes)} boxes)")

    # 生成汇总大图
    print("\n" + "="*50)
    print("Generating summary grid...")
    print("="*50)

    comparison_images = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path).replace('.jpg', '')
        comp_path = os.path.join(args.output_dir, f"comparison_{img_name}.jpg")
        if os.path.exists(comp_path):
            comparison_images.append(cv2.imread(comp_path))

    if comparison_images:
        # 创建网格图 (5行2列)
        n_cols = 2
        n_rows = (len(comparison_images) + n_cols - 1) // n_cols

        # 调整所有图像到相同大小
        target_w = 800
        resized_images = []
        for img in comparison_images:
            h, w = img.shape[:2]
            scale = target_w / w
            new_h = int(h * scale)
            resized = cv2.resize(img, (target_w, new_h))
            resized_images.append(resized)

        # 找到最大高度
        max_h = max(img.shape[0] for img in resized_images)

        # 填充到相同高度
        padded_images = []
        for img in resized_images:
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            padded_images.append(img)

        # 创建网格
        rows = []
        for i in range(n_rows):
            row_imgs = padded_images[i*n_cols:(i+1)*n_cols]
            if len(row_imgs) < n_cols:
                for _ in range(n_cols - len(row_imgs)):
                    row_imgs.append(np.zeros_like(padded_images[0]))
            rows.append(np.hstack(row_imgs))

        grid = np.vstack(rows)
        grid_path = os.path.join(args.output_dir, "detection_comparison_grid.jpg")
        cv2.imwrite(grid_path, grid)
        print(f"Saved grid: {grid_path}")

    print("\n" + "="*50)
    print("Done! All comparison images saved to:", args.output_dir)
    print("="*50)


if __name__ == '__main__':
    main()
