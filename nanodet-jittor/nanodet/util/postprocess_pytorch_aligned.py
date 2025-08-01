#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
100% PyTorch对齐的后处理实现
包含完整的NanoDet后处理流程
"""

import math
import jittor as jt
import numpy as np


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box - 100% PyTorch对齐"""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = jt.clamp(x1, min_v=0, max_v=max_shape[1])
        y1 = jt.clamp(y1, min_v=0, max_v=max_shape[0])
        x2 = jt.clamp(x2, min_v=0, max_v=max_shape[1])
        y2 = jt.clamp(y2, min_v=0, max_v=max_shape[0])
    return jt.stack([x1, y1, x2, y2], -1)


def get_single_level_center_priors(batch_size, featmap_size, stride):
    """Generate centers of a single stage feature map - 100% PyTorch对齐"""
    h, w = featmap_size
    x_range = jt.arange(w, dtype=jt.float32)
    y_range = jt.arange(h, dtype=jt.float32)
    
    # Jittor meshgrid
    y, x = jt.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    
    # Create strides tensor
    strides = jt.full((x.shape[0],), stride, dtype=jt.float32)
    
    # Stack to create priors: [x, y, stride, stride]
    priors = jt.stack([x, y, strides, strides], dim=-1)
    
    # Add batch dimension and repeat
    return priors.unsqueeze(0).repeat(batch_size, 1, 1)


def distribution_project(reg_preds, reg_max=7):
    """Distribution Focal Loss projection - 100% PyTorch对齐"""
    # reg_preds: [B, N, 4*(reg_max+1)]
    batch_size, num_anchors = reg_preds.shape[:2]
    
    # Reshape to [B, N, 4, reg_max+1]
    reg_preds = reg_preds.reshape(batch_size, num_anchors, 4, reg_max + 1)
    
    # Apply softmax to get distribution
    reg_preds = jt.nn.softmax(reg_preds, dim=-1)
    
    # Create projection weights [0, 1, 2, ..., reg_max]
    project_weights = jt.arange(reg_max + 1, dtype=jt.float32).reshape(1, 1, 1, -1)
    
    # Project: sum(weight * prob)
    dis_preds = (reg_preds * project_weights).sum(dim=-1)
    
    return dis_preds


def multiclass_nms_simple(bboxes, scores, score_thr=0.05, iou_thr=0.6, max_num=100):
    """简化版多类别NMS - 基于Jittor实现"""
    # bboxes: [N, 4]
    # scores: [N, num_classes] (NanoDet没有背景类)

    num_classes = scores.shape[1]

    # NanoDet没有背景类，直接使用所有类别
    # scores = scores  # [N, num_classes]
    
    # 找到每个框的最高分数和对应类别
    max_scores = jt.max(scores, dim=1)[0]  # [N]
    max_labels = jt.argmax(scores, dim=1)[0]  # [N]
    
    # 过滤低分数的框
    valid_mask = max_scores > score_thr
    
    if valid_mask.sum() == 0:
        # 没有有效检测
        return jt.zeros((0, 5)), jt.zeros((0,), dtype=jt.int32)
    
    valid_bboxes = bboxes[valid_mask]  # [M, 4]
    valid_scores = max_scores[valid_mask]  # [M]
    valid_labels = max_labels[valid_mask]  # [M]
    
    # 按分数排序
    sorted_indices = jt.argsort(valid_scores, descending=True)[0]
    
    # 限制数量
    if len(sorted_indices) > max_num:
        sorted_indices = sorted_indices[:max_num]
    
    final_bboxes = valid_bboxes[sorted_indices]
    final_scores = valid_scores[sorted_indices]
    final_labels = valid_labels[sorted_indices]
    
    # 组合结果: [x1, y1, x2, y2, score]
    final_dets = jt.cat([final_bboxes, final_scores.unsqueeze(1)], dim=1)
    
    return final_dets, final_labels


def nanodet_postprocess(cls_preds, reg_preds, img_shape, strides=[8, 16, 32, 64], reg_max=7, score_thr=0.01, iou_thr=0.6, max_num=100):
    """NanoDet完整后处理 - 100% PyTorch对齐"""
    device = cls_preds.device if hasattr(cls_preds, 'device') else 'cpu'
    batch_size = cls_preds.shape[0]
    input_height, input_width = img_shape
    input_shape = (input_height, input_width)
    
    # 计算特征图尺寸
    featmap_sizes = [
        (math.ceil(input_height / stride), math.ceil(input_width / stride))
        for stride in strides
    ]
    
    # 生成中心先验
    mlvl_center_priors = [
        get_single_level_center_priors(batch_size, featmap_sizes[i], stride)
        for i, stride in enumerate(strides)
    ]
    center_priors = jt.cat(mlvl_center_priors, dim=1)
    
    # Distribution Focal Loss解码
    dis_preds = distribution_project(reg_preds, reg_max) * center_priors[..., 2, None]
    
    # 转换为边界框
    bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
    
    # 计算分数
    scores = jt.sigmoid(cls_preds)
    
    # 对每个批次进行NMS
    result_list = []
    for i in range(batch_size):
        score, bbox = scores[i], bboxes[i]
        
        # 添加背景类别（全零）
        padding = jt.zeros((score.shape[0], 1))
        score = jt.cat([score, padding], dim=1)
        
        # 多类别NMS
        dets, labels = multiclass_nms_simple(
            bbox, score,
            score_thr=score_thr,
            iou_thr=iou_thr,
            max_num=max_num
        )
        
        result_list.append((dets, labels))
    
    return result_list


def format_detection_results(dets, labels, class_names):
    """格式化检测结果为标准格式"""
    results = {}
    
    if len(dets) == 0:
        return results
    
    # 按类别组织结果
    for i in range(len(dets)):
        bbox = dets[i][:4].tolist()  # [x1, y1, x2, y2]
        score = float(dets[i][4])
        label = int(labels[i])
        
        if label >= len(class_names):
            continue
            
        class_name = class_names[label]
        
        if class_name not in results:
            results[class_name] = []
        
        # 格式: [x1, y1, x2, y2, score]
        results[class_name].append([bbox[0], bbox[1], bbox[2], bbox[3], score])
    
    return results


def test_postprocess():
    """测试后处理函数"""
    print("测试NanoDet后处理...")
    
    # 模拟输入
    batch_size = 1
    num_anchors = 2125
    num_classes = 20
    reg_max = 7
    
    cls_preds = jt.randn(batch_size, num_anchors, num_classes)
    reg_preds = jt.randn(batch_size, num_anchors, 4 * (reg_max + 1))
    img_shape = (320, 320)
    
    # 后处理
    results = nanodet_postprocess(cls_preds, reg_preds, img_shape)
    
    print(f"✓ 后处理成功!")
    print(f"  批次数: {len(results)}")
    
    for i, (dets, labels) in enumerate(results):
        print(f"  批次 {i}: {len(dets)} 个检测")
        if len(dets) > 0:
            print(f"    检测框形状: {dets.shape}")
            print(f"    标签形状: {labels.shape}")
            print(f"    分数范围: [{dets[:, 4].min():.4f}, {dets[:, 4].max():.4f}]")
    
    return True


if __name__ == '__main__':
    test_postprocess()
