#!/usr/bin/env python3
"""
创建真正的PyTorch vs Jittor检测结果对比
使用Jittor加载PyTorch权重来模拟PyTorch结果，与Jittor独立训练结果对比
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加Jittor路径
sys.path.append('nanodet-jittor')

import jittor as jt
from nanodet.util.config import load_config, cfg
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline
from nanodet.util.check_point import pt_to_jt_checkpoint
import torch

def load_jittor_with_pytorch_weights():
    """使用Jittor加载PyTorch权重"""
    print("🔧 使用Jittor加载PyTorch权重...")
    
    # 加载配置
    config_path = "nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml"
    load_config(cfg, config_path)
    
    # 构建模型
    model = build_model(cfg.model)
    
    # 加载PyTorch权重
    pt_model_path = "nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64_50epochs/model_best/model_best.ckpt"
    
    if not os.path.exists(pt_model_path):
        print(f"❌ PyTorch权重文件不存在: {pt_model_path}")
        return None
    
    print(f"📥 加载PyTorch权重: {pt_model_path}")
    
    # 加载PyTorch检查点
    pt_ckpt = torch.load(pt_model_path, map_location='cpu')
    
    # 提取state_dict
    if 'state_dict' in pt_ckpt:
        state_dict = pt_ckpt['state_dict']
        # 处理avg_model前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('avg_model.'):
                new_state_dict[k[10:]] = v
            elif k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        pt_ckpt['state_dict'] = new_state_dict
    
    print(f"✅ PyTorch权重加载成功，键数: {len(pt_ckpt.get('state_dict', pt_ckpt))}")
    
    # 转换为Jittor格式
    print("🔄 转换权重格式...")
    jt_ckpt = pt_to_jt_checkpoint(pt_ckpt, model)
    
    # 加载权重到模型
    model.load_state_dict(jt_ckpt['state_dict'])
    model.eval()
    
    print("✅ Jittor模型加载PyTorch权重成功")
    return model

def load_jittor_trained_model():
    """加载Jittor独立训练的模型"""
    print("🔧 加载Jittor独立训练模型...")
    
    # 加载配置
    config_path = "nanodet-jittor/config/nanodet-plus-m_320_voc_bs64_50epochs.yml"
    load_config(cfg, config_path)
    
    # 构建模型
    model = build_model(cfg.model)
    
    # 加载Jittor权重
    jt_model_path = "workspace/jittor_50epochs_model_best.pkl"
    
    if not os.path.exists(jt_model_path):
        print(f"❌ Jittor权重文件不存在: {jt_model_path}")
        return None
    
    print(f"📥 加载Jittor权重: {jt_model_path}")
    
    checkpoint = jt.load(jt_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print("✅ Jittor独立训练模型加载成功")
    return model

def inference_with_model(model, image_path):
    """使用模型进行推理"""
    try:
        # 数据预处理
        pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

        # 加载图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 无法加载图片: {image_path}")
            return []

        # 正确构造 meta 并调用 Pipeline
        meta = {"img": img}
        meta = pipeline(None, meta, cfg.data.val.input_size)
        proc_img = meta["img"]

        # 推理
        with jt.no_grad():
            results = model.inference([proc_img])
        
        # 处理结果
        processed_results = []
        if results and len(results) > 0 and len(results[0]) > 0:
            for det in results[0]:
                if len(det) >= 6:  # bbox + score + class
                    bbox = det[:4].tolist()
                    score = float(det[4])
                    class_id = int(det[5])
                    
                    # VOC类别名
                    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                                 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                                 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor']
                    
                    if class_id < len(voc_classes):
                        class_name = voc_classes[class_id]
                        
                        processed_results.append({
                            "bbox": bbox,
                            "score": score,
                            "class": class_name,
                            "class_id": class_id
                        })
        
        return processed_results
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return []

def draw_detections(img, detections, title, color=(0, 255, 0)):
    """绘制检测结果"""
    img_draw = img.copy()
    
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        score = det["score"]
        class_name = det["class"]
        
        # 绘制边界框
        cv2.rectangle(img_draw, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     color, 3)
        
        # 绘制标签背景
        label = f"{class_name}: {score:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        cv2.rectangle(img_draw,
                     (int(bbox[0]), int(bbox[1]) - label_size[1] - 15),
                     (int(bbox[0]) + label_size[0] + 10, int(bbox[1])),
                     color, -1)
        
        # 绘制标签文字
        cv2.putText(img_draw, label,
                   (int(bbox[0]) + 5, int(bbox[1]) - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制检测序号
        cv2.circle(img_draw, (int(bbox[0]) + 15, int(bbox[1]) + 25), 18, color, -1)
        cv2.putText(img_draw, str(i+1),
                   (int(bbox[0]) + 8, int(bbox[1]) + 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_draw

def create_three_way_comparison(image_path, pytorch_results, jittor_results, output_path, img_name):
    """创建三图对比"""
    
    # 加载原图
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 绘制PyTorch结果 (红色)
    pytorch_img = draw_detections(img_rgb, pytorch_results, "PyTorch", (255, 0, 0))
    
    # 绘制Jittor结果 (蓝色)
    jittor_img = draw_detections(img_rgb, jittor_results, "Jittor", (0, 0, 255))
    
    # 创建对比图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # 原图
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # PyTorch结果
    ax2.imshow(pytorch_img)
    ax2.set_title(f'PyTorch (via Jittor)\n{len(pytorch_results)} detections', 
                 fontsize=16, fontweight='bold', color='red')
    ax2.axis('off')
    
    # Jittor结果
    ax3.imshow(jittor_img)
    ax3.set_title(f'Jittor (Independent)\n{len(jittor_results)} detections', 
                 fontsize=16, fontweight='bold', color='blue')
    ax3.axis('off')
    
    # 添加图片名称
    fig.suptitle(f'Real Detection Comparison: {img_name}', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 真实对比图已保存: {output_path}")

def main():
    """主函数"""
    print("🎯 开始创建真正的PyTorch vs Jittor检测结果对比...")
    
    # 创建输出目录
    os.makedirs("DELIVERABLES/images/real_pytorch_jittor_comparison", exist_ok=True)
    
    # 1. 加载两个模型
    print("\n" + "="*60)
    print("📥 加载模型...")
    print("="*60)
    
    pytorch_model = load_jittor_with_pytorch_weights()
    jittor_model = load_jittor_trained_model()
    
    if pytorch_model is None or jittor_model is None:
        print("❌ 模型加载失败")
        return
    
    # 2. 获取测试图片
    test_images = [
        "data/VOCdevkit/VOC2007/JPEGImages/000003.jpg",
        "data/VOCdevkit/VOC2007/JPEGImages/000011.jpg",
        "data/VOCdevkit/VOC2007/JPEGImages/000014.jpg",
        "data/VOCdevkit/VOC2007/JPEGImages/000015.jpg"
    ]
    
    # 过滤存在的图片
    valid_images = [img for img in test_images if os.path.exists(img)]
    
    if not valid_images:
        print("❌ 未找到测试图片")
        return
    
    print(f"\n📋 找到 {len(valid_images)} 张测试图片")
    
    # 3. 处理每张图片
    for i, image_path in enumerate(valid_images):
        img_name = os.path.basename(image_path).split('.')[0]
        print(f"\n🖼️ 处理图片 {i+1}/{len(valid_images)}: {img_name}")
        
        # PyTorch推理（使用Jittor加载PyTorch权重）
        print("   🔍 PyTorch推理（via Jittor）...")
        pytorch_results = inference_with_model(pytorch_model, image_path)
        print(f"      检测到 {len(pytorch_results)} 个目标")
        
        # Jittor推理（独立训练）
        print("   🔍 Jittor推理（独立训练）...")
        jittor_results = inference_with_model(jittor_model, image_path)
        print(f"      检测到 {len(jittor_results)} 个目标")
        
        # 创建对比图
        comparison_path = f"DELIVERABLES/images/real_pytorch_jittor_comparison/{img_name}_real_comparison.png"
        create_three_way_comparison(image_path, pytorch_results, jittor_results, comparison_path, img_name)
        
        # 打印检测结果对比
        print(f"   📊 对比结果:")
        print(f"      PyTorch: {len(pytorch_results)} 个检测")
        print(f"      Jittor:  {len(jittor_results)} 个检测")
        print(f"      差异:    {abs(len(pytorch_results) - len(jittor_results))} 个检测")
    
    print(f"\n🎉 真实PyTorch vs Jittor对比完成！")
    print(f"📁 输出目录: DELIVERABLES/images/real_pytorch_jittor_comparison/")
    print(f"📊 生成了 {len(valid_images)} 张真实对比图")
    
    print(f"\n💡 说明:")
    print(f"   - PyTorch结果: 使用Jittor框架加载PyTorch权重进行推理")
    print(f"   - Jittor结果: 使用Jittor独立训练的权重进行推理")
    print(f"   - 这样可以看到权重转换的效果和两种训练方式的差异")

if __name__ == "__main__":
    main()
