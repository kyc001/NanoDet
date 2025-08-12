#!/usr/bin/env python3
"""
运行真实的PyTorch推理并与Jittor结果对比
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json

def run_pytorch_demo(image_path):
    """运行PyTorch demo进行推理"""
    try:
        print(f"🔍 运行PyTorch推理: {os.path.basename(image_path)}")
        
        # 检查PyTorch推理脚本
        pytorch_inference = "tools/pytorch_inference_demo.py"
        if not os.path.exists(pytorch_inference):
            print(f"❌ PyTorch推理脚本不存在: {pytorch_inference}")
            return None

        # 检查配置文件
        config_path = "nanodet-pytorch/config/nanodet-plus-m_320_voc_bs64_50epochs.yml"
        if not os.path.exists(config_path):
            print(f"❌ PyTorch配置文件不存在: {config_path}")
            return None

        # 检查模型权重
        model_path = "nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64_50epochs/model_best/nanodet_model_best.pth"
        if not os.path.exists(model_path):
            print(f"❌ PyTorch模型权重不存在: {model_path}")
            return None

        # 创建输出目录
        output_dir = "temp_pytorch_results"
        os.makedirs(output_dir, exist_ok=True)

        # 输出文件路径
        output_img = os.path.join(output_dir, os.path.basename(image_path))

        # 构建推理命令
        cmd = [
            "python", pytorch_inference,
            "--config", config_path,
            "--model", model_path,
            "--img", image_path,
            "--output", output_img,
            "--device", "cuda:0"
        ]
        
        print(f"   执行命令: {' '.join(cmd)}")

        # 执行推理命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"✅ PyTorch推理成功")
            print(f"   输出: {result.stdout}")

            # 检查输出图片是否存在
            if os.path.exists(output_img):
                print(f"   输出图片: {output_img}")
                return output_img
            else:
                print(f"   未找到输出图片: {output_img}")
                return None
        else:
            print(f"❌ PyTorch推理失败:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ PyTorch推理超时")
        return None
    except Exception as e:
        print(f"❌ PyTorch推理异常: {e}")
        return None

def create_real_side_by_side_comparison():
    """创建真实的并排对比图"""
    
    # 获取测试图片
    test_images = []
    sample_dets_dir = "DELIVERABLES/images/sample_dets"
    
    if os.path.exists(sample_dets_dir):
        det_files = [f for f in os.listdir(sample_dets_dir) if f.endswith('_det.jpg')]
        for det_file in det_files[:4]:  # 只处理前4张
            img_name = det_file.replace('_det.jpg', '.jpg')
            img_path = f"data/VOCdevkit/VOC2007/JPEGImages/{img_name}"
            if os.path.exists(img_path):
                test_images.append((img_path, det_file))
    
    if not test_images:
        print("❌ 未找到测试图片")
        return
    
    print(f"📋 找到 {len(test_images)} 张测试图片")
    
    # 创建输出目录
    os.makedirs("DELIVERABLES/images/real_comparisons", exist_ok=True)
    
    comparison_results = []
    
    for i, (image_path, jittor_det_file) in enumerate(test_images):
        img_name = os.path.basename(image_path).split('.')[0]
        print(f"\n🖼️ 处理图片 {i+1}/{len(test_images)}: {img_name}")
        
        # 运行PyTorch推理
        pytorch_result_img = run_pytorch_demo(image_path)
        
        # 获取Jittor结果
        jittor_result_img = os.path.join(sample_dets_dir, jittor_det_file)
        
        # 创建对比图
        create_comparison_image(image_path, pytorch_result_img, jittor_result_img, img_name)
        
        # 记录结果
        comparison_results.append({
            "image": img_name,
            "original": image_path,
            "pytorch_result": pytorch_result_img,
            "jittor_result": jittor_result_img,
            "pytorch_success": pytorch_result_img is not None
        })
    
    # 保存对比结果
    results_path = "DELIVERABLES/images/real_comparisons/real_comparison_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 对比结果已保存: {results_path}")
    
    # 生成统计报告
    successful_pytorch = sum(1 for r in comparison_results if r["pytorch_success"])
    print(f"\n📊 统计报告:")
    print(f"   总测试图片: {len(comparison_results)}")
    print(f"   PyTorch推理成功: {successful_pytorch}")
    print(f"   Jittor结果可用: {len(comparison_results)}")

def create_comparison_image(original_path, pytorch_result_path, jittor_result_path, img_name):
    """创建三图对比"""
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 原图
    if os.path.exists(original_path):
        original_img = cv2.imread(original_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_img_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center', fontsize=16)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # PyTorch结果
    if pytorch_result_path and os.path.exists(pytorch_result_path):
        pytorch_img = cv2.imread(pytorch_result_path)
        pytorch_img_rgb = cv2.cvtColor(pytorch_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(pytorch_img_rgb)
        axes[1].set_title('PyTorch Detection\n(Real Result)', fontsize=14, fontweight='bold', color='red')
    else:
        axes[1].text(0.5, 0.5, 'PyTorch\nInference Failed', ha='center', va='center', fontsize=16, color='red')
        axes[1].set_title('PyTorch Detection\n(Failed)', fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # Jittor结果
    if os.path.exists(jittor_result_path):
        jittor_img = cv2.imread(jittor_result_path)
        jittor_img_rgb = cv2.cvtColor(jittor_img, cv2.COLOR_BGR2RGB)
        axes[2].imshow(jittor_img_rgb)
        axes[2].set_title('Jittor Detection\n(Real Result)', fontsize=14, fontweight='bold', color='blue')
    else:
        axes[2].text(0.5, 0.5, 'Jittor\nResult Not Found', ha='center', va='center', fontsize=16, color='blue')
        axes[2].set_title('Jittor Detection\n(Not Found)', fontsize=14, fontweight='bold', color='blue')
    axes[2].axis('off')
    
    # 设置总标题
    fig.suptitle(f'Real Detection Comparison: {img_name}.jpg', fontsize=18, fontweight='bold')
    
    # 保存对比图
    output_path = f"DELIVERABLES/images/real_comparisons/{img_name}_real_side_by_side.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 真实对比图已保存: {output_path}")

def create_detection_analysis():
    """创建检测分析图"""
    
    # 分析sample_dets中的Jittor结果
    sample_dets_dir = "DELIVERABLES/images/sample_dets"
    if not os.path.exists(sample_dets_dir):
        print("❌ sample_dets目录不存在")
        return
    
    det_files = [f for f in os.listdir(sample_dets_dir) if f.endswith('_det.jpg')]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Jittor Real Detection Results Analysis', fontsize=18, fontweight='bold')
    
    for i, det_file in enumerate(det_files[:8]):
        row = i // 4
        col = i % 4
        
        det_img_path = os.path.join(sample_dets_dir, det_file)
        det_img = cv2.imread(det_img_path)
        det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(det_img_rgb)
        axes[row, col].set_title(det_file.replace('_det.jpg', ''), fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('DELIVERABLES/images/real_comparisons/jittor_real_detection_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Jittor真实检测分析图已生成: jittor_real_detection_analysis.png")

def main():
    """主函数"""
    print("🎯 开始运行真实的PyTorch推理并与Jittor对比...")
    
    # 1. 运行真实的PyTorch vs Jittor对比
    create_real_side_by_side_comparison()
    
    # 2. 创建检测分析图
    create_detection_analysis()
    
    print("\n🎉 真实推理对比完成！")
    print("📁 输出目录: DELIVERABLES/images/real_comparisons/")
    
    # 清理临时文件
    if os.path.exists("temp_pytorch_results"):
        import shutil
        shutil.rmtree("temp_pytorch_results")
        print("🧹 临时文件已清理")

if __name__ == "__main__":
    main()
