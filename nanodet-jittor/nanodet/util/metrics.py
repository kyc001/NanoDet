#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估指标模块
实现mAP计算和COCO评估器
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    logger.warning("pycocotools not available, COCO evaluation will be disabled")
    COCO_AVAILABLE = False


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # box格式: [x1, y1, x2, y2]
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(detections, ground_truths, iou_threshold=0.5):
    """计算单个类别的AP"""
    if len(ground_truths) == 0:
        return 0.0
    
    if len(detections) == 0:
        return 0.0
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    
    gt_matched = [False] * len(ground_truths)
    
    for i, det in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            if gt_matched[j]:
                continue
            
            iou = calculate_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # 计算precision和recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # 计算AP (使用11点插值)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def calculate_map(detections_by_class, ground_truths_by_class, num_classes, iou_threshold=0.5):
    """计算mAP"""
    aps = []
    
    for class_id in range(num_classes):
        class_detections = detections_by_class.get(class_id, [])
        class_ground_truths = ground_truths_by_class.get(class_id, [])
        
        ap = calculate_ap(class_detections, class_ground_truths, iou_threshold)
        aps.append(ap)
    
    map_score = np.mean(aps)
    return map_score, aps


class COCOEvaluator:
    """COCO格式评估器"""
    
    def __init__(self, ann_file, img_prefix=''):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.results = []
        
        if COCO_AVAILABLE:
            self.coco_gt = COCO(ann_file)
        else:
            logger.error("pycocotools not available, cannot use COCOEvaluator")
            self.coco_gt = None
    
    def add_result(self, image_id, detections):
        """添加检测结果"""
        for det in detections:
            result = {
                'image_id': image_id,
                'category_id': det['category_id'],
                'bbox': det['bbox'],  # [x, y, w, h]
                'score': det['score']
            }
            self.results.append(result)
    
    def evaluate(self):
        """执行评估"""
        if not COCO_AVAILABLE or self.coco_gt is None:
            logger.error("Cannot evaluate: pycocotools not available")
            return {}
        
        if len(self.results) == 0:
            logger.warning("No detection results to evaluate")
            return {}
        
        # 保存结果到临时文件
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.results, f)
            results_file = f.name
        
        try:
            # 加载检测结果
            coco_dt = self.coco_gt.loadRes(results_file)
            
            # 执行评估
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # 提取指标
            metrics = {
                'mAP': coco_eval.stats[0],           # mAP@0.5:0.95
                'mAP_50': coco_eval.stats[1],        # mAP@0.5
                'mAP_75': coco_eval.stats[2],        # mAP@0.75
                'mAP_small': coco_eval.stats[3],     # mAP for small objects
                'mAP_medium': coco_eval.stats[4],    # mAP for medium objects
                'mAP_large': coco_eval.stats[5],     # mAP for large objects
                'AR_1': coco_eval.stats[6],          # AR@1
                'AR_10': coco_eval.stats[7],         # AR@10
                'AR_100': coco_eval.stats[8],        # AR@100
                'AR_small': coco_eval.stats[9],      # AR for small objects
                'AR_medium': coco_eval.stats[10],    # AR for medium objects
                'AR_large': coco_eval.stats[11],     # AR for large objects
            }
            
            return metrics
            
        finally:
            # 清理临时文件
            import os
            if os.path.exists(results_file):
                os.unlink(results_file)
    
    def reset(self):
        """重置结果"""
        self.results = []


class SimpleEvaluator:
    """简单评估器（不依赖pycocotools）"""
    
    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.detections_by_class = {i: [] for i in range(num_classes)}
        self.ground_truths_by_class = {i: [] for i in range(num_classes)}
    
    def add_result(self, image_id, detections, ground_truths):
        """添加检测结果和真值"""
        for det in detections:
            class_id = det['category_id'] - 1  # 转换为0-based
            if 0 <= class_id < self.num_classes:
                self.detections_by_class[class_id].append({
                    'image_id': image_id,
                    'bbox': det['bbox'],
                    'score': det['score']
                })
        
        for gt in ground_truths:
            class_id = gt['category_id'] - 1  # 转换为0-based
            if 0 <= class_id < self.num_classes:
                self.ground_truths_by_class[class_id].append({
                    'image_id': image_id,
                    'bbox': gt['bbox']
                })
    
    def evaluate(self, iou_threshold=0.5):
        """执行评估"""
        map_score, aps = calculate_map(
            self.detections_by_class,
            self.ground_truths_by_class,
            self.num_classes,
            iou_threshold
        )
        
        metrics = {
            'mAP': map_score,
            'APs': aps
        }
        
        return metrics
    
    def reset(self):
        """重置结果"""
        self.detections_by_class = {i: [] for i in range(self.num_classes)}
        self.ground_truths_by_class = {i: [] for i in range(self.num_classes)}
