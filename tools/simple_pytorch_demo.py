#!/usr/bin/env python3
"""
简单的PyTorch NanoDet推理脚本
直接使用PyTorch进行推理，不依赖lightning
"""

import sys
import os
import cv2
import torch
import numpy as np

# 添加PyTorch NanoDet路径
sys.path.insert(0, 'nanodet-pytorch')

from nanodet.util.config import load_config, cfg
from nanodet.util.logger import Logger
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight

class SimplePredictor:
    def __init__(self, config_path, model_path, device="cuda:0"):
        print(f"🔧 初始化PyTorch预测器...")
        print(f"   配置文件: {config_path}")
        print(f"   模型文件: {model_path}")
        
        # 加载配置
        load_config(cfg, config_path)
        
        # 创建logger
        logger = Logger(-1, use_tensorboard=False)
        
        # 构建模型
        model = build_model(cfg.model)
        
        # 加载权重
        if model_path.endswith('.ckpt'):
            # Lightning checkpoint
            ckpt = torch.load(model_path, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                # 处理不同的前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('avg_model.'):
                        # EMA模型权重，移除'avg_model.'前缀
                        new_state_dict[k[10:]] = v
                    elif k.startswith('model.'):
                        # 普通模型权重，移除'model.'前缀
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(ckpt)
        else:
            # 普通PyTorch checkpoint
            ckpt = torch.load(model_path, map_location='cpu')
            load_model_weight(model, ckpt, logger)
        
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.device = device
        self.cfg = cfg
        
        # VOC类别名
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        print(f"✅ PyTorch预测器初始化完成")
    
    def inference(self, img_path):
        """推理单张图片"""
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 无法读取图片: {img_path}")
            return None, None
        
        height, width = img.shape[:2]
        img_info = {
            "file_name": os.path.basename(img_path),
            "height": height,
            "width": width
        }
        
        # 预处理
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size, self.cfg.data.val.input_size)
        meta["img"] = (
            torch.from_numpy(meta["img"].transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        
        # 推理
        with torch.no_grad():
            results = self.model.inference(meta)
        
        return meta, results
    
    def visualize_and_save(self, img_path, output_path, score_thres=0.35):
        """推理并保存可视化结果"""
        meta, results = self.inference(img_path)
        
        if meta is None or results is None:
            return False
        
        # 获取检测结果
        dets = results[0]
        
        # 在原图上绘制检测结果
        img = meta["raw_img"].copy()
        
        detection_count = 0
        for det in dets:
            if len(det) >= 6:
                bbox = det[:4]
                score = det[4]
                class_id = int(det[5])
                
                if score > score_thres and class_id < len(self.class_names):
                    detection_count += 1
                    
                    # 绘制边界框
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # 绘制标签
                    label = f"{self.class_names[class_id]}: {score:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    cv2.rectangle(img, (x1, y1 - label_size[1] - 15), 
                                 (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1 + 5, y1 - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存结果
        cv2.imwrite(output_path, img)
        print(f"✅ PyTorch推理完成，检测到 {detection_count} 个目标")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple PyTorch NanoDet Inference')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--model', required=True, help='模型权重路径')
    parser.add_argument('--img', required=True, help='输入图片路径')
    parser.add_argument('--output', required=True, help='输出图片路径')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--score_thres', type=float, default=0.35, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.img):
        print(f"❌ 输入图片不存在: {args.img}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # 创建预测器
        predictor = SimplePredictor(args.config, args.model, args.device)
        
        # 推理并保存结果
        success = predictor.visualize_and_save(args.img, args.output, args.score_thres)
        
        if success:
            print("🎉 PyTorch推理成功完成！")
            print(f"📄 结果保存到: {args.output}")
        else:
            print("❌ PyTorch推理失败")
            
    except Exception as e:
        print(f"❌ PyTorch推理异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
