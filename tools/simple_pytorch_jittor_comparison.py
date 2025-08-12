#!/usr/bin/env python3
"""
简单的PyTorch vs Jittor检测结果对比
使用sample_dets中的原图，用PyTorch推理，然后与Jittor结果对比
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

def get_original_images():
    """获取sample_dets对应的原图"""
    sample_dets_dir = "DELIVERABLES/images/sample_dets"
    original_images = []
    
    if os.path.exists(sample_dets_dir):
        det_files = sorted([f for f in os.listdir(sample_dets_dir) if f.endswith('_det.jpg')])
        
        for det_file in det_files:
            img_name = det_file.replace('_det.jpg', '.jpg')
            original_path = f"data/VOCdevkit/VOC2007/JPEGImages/{img_name}"
            jittor_result_path = os.path.join(sample_dets_dir, det_file)
            
            if os.path.exists(original_path):
                original_images.append({
                    'name': img_name.replace('.jpg', ''),
                    'original': original_path,
                    'jittor_result': jittor_result_path
                })
    
    return original_images

def run_pytorch_inference_simple(image_path, output_path):
    """使用PyTorch进行简单推理"""
    try:
        print(f"🔍 PyTorch推理: {os.path.basename(image_path)}")
        
        # 检查必要文件
        config_path = "nanodet-pytorch/config/nanodet-plus-m_320_voc_bs64_50epochs.yml"
        model_path = "nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64_50epochs/model_best/model_best.ckpt"
        demo_script = "tools/simple_pytorch_demo.py"

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False

        if not os.path.exists(demo_script):
            print(f"❌ 推理脚本不存在: {demo_script}")
            return False

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 构建推理命令
        cmd = [
            "python", demo_script,
            "--config", config_path,
            "--model", model_path,
            "--img", image_path,
            "--output", output_path,
            "--device", "cuda:0"
        ]
        
        print(f"   执行命令: {' '.join(cmd)}")
        
        # 执行推理命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✅ PyTorch推理成功")
            if os.path.exists(output_path):
                print(f"   输出文件: {output_path}")
                return True
            else:
                print(f"   输出文件未生成: {output_path}")
                return False
        else:
            print(f"❌ PyTorch推理失败:")
            print(f"   stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ PyTorch推理超时")
        return False
    except Exception as e:
        print(f"❌ PyTorch推理异常: {e}")
        return False

def create_three_way_comparison(original_path, pytorch_result_path, jittor_result_path, output_path, img_name):
    """创建三图对比"""
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 原图
    if os.path.exists(original_path):
        original_img = cv2.imread(original_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_img_rgb)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'Original\nNot Found', ha='center', va='center', fontsize=16)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # PyTorch结果
    if pytorch_result_path and os.path.exists(pytorch_result_path):
        pytorch_img = cv2.imread(pytorch_result_path)
        pytorch_img_rgb = cv2.cvtColor(pytorch_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(pytorch_img_rgb)
        axes[1].set_title('PyTorch Detection\n(Real Result)', fontsize=16, fontweight='bold', color='red')
    else:
        axes[1].text(0.5, 0.5, 'PyTorch\nInference Failed', ha='center', va='center', fontsize=16, color='red')
        axes[1].set_title('PyTorch Detection\n(Failed)', fontsize=16, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # Jittor结果
    if os.path.exists(jittor_result_path):
        jittor_img = cv2.imread(jittor_result_path)
        jittor_img_rgb = cv2.cvtColor(jittor_img, cv2.COLOR_BGR2RGB)
        axes[2].imshow(jittor_img_rgb)
        axes[2].set_title('Jittor Detection\n(Real Result)', fontsize=16, fontweight='bold', color='blue')
    else:
        axes[2].text(0.5, 0.5, 'Jittor\nResult Not Found', ha='center', va='center', fontsize=16, color='blue')
        axes[2].set_title('Jittor Detection\n(Not Found)', fontsize=16, fontweight='bold', color='blue')
    axes[2].axis('off')
    
    # 设置总标题
    fig.suptitle(f'Real Detection Comparison: {img_name}', fontsize=20, fontweight='bold')
    
    # 保存对比图
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 三图对比已保存: {output_path}")

def create_comparison_grid():
    """创建对比网格图"""
    
    # 获取原图列表
    images = get_original_images()
    
    if len(images) < 4:
        print("❌ 图片数量不足")
        return
    
    # 选择前8张图片
    selected_images = images[:8]
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('PyTorch vs Jittor Detection Results Comparison', fontsize=20, fontweight='bold')
    
    for i, img_info in enumerate(selected_images):
        row = i // 4
        col = i % 4
        
        # 尝试加载PyTorch结果
        pytorch_result_path = f"temp_pytorch_results/{img_info['name']}_det.jpg"
        
        if os.path.exists(pytorch_result_path):
            # 如果有PyTorch结果，显示对比
            pytorch_img = cv2.imread(pytorch_result_path)
            pytorch_img_rgb = cv2.cvtColor(pytorch_img, cv2.COLOR_BGR2RGB)
            
            # 上半部分显示PyTorch结果
            if row == 0:
                axes[row, col].imshow(pytorch_img_rgb)
                axes[row, col].set_title(f'PyTorch: {img_info["name"]}', fontweight='bold', color='red')
            else:
                # 下半部分显示Jittor结果
                jittor_img = cv2.imread(img_info['jittor_result'])
                jittor_img_rgb = cv2.cvtColor(jittor_img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(jittor_img_rgb)
                axes[row, col].set_title(f'Jittor: {img_info["name"]}', fontweight='bold', color='blue')
        else:
            # 如果没有PyTorch结果，显示Jittor结果
            jittor_img = cv2.imread(img_info['jittor_result'])
            jittor_img_rgb = cv2.cvtColor(jittor_img, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(jittor_img_rgb)
            axes[row, col].set_title(f'Jittor: {img_info["name"]}', fontweight='bold', color='blue')
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('DELIVERABLES/images/real_comparisons/pytorch_jittor_comparison_grid.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 对比网格图已生成: pytorch_jittor_comparison_grid.png")

def main():
    """主函数"""
    print("🎯 开始简单的PyTorch vs Jittor检测结果对比...")
    
    # 创建输出目录
    os.makedirs("DELIVERABLES/images/real_comparisons", exist_ok=True)
    os.makedirs("temp_pytorch_results", exist_ok=True)
    
    # 获取原图列表
    images = get_original_images()
    
    if not images:
        print("❌ 未找到原图")
        return
    
    print(f"📋 找到 {len(images)} 张图片")
    
    successful_pytorch = 0
    
    # 处理每张图片
    for i, img_info in enumerate(images[:4]):  # 只处理前4张
        print(f"\n🖼️ 处理图片 {i+1}/{min(4, len(images))}: {img_info['name']}")
        
        # PyTorch推理
        pytorch_output = f"temp_pytorch_results/{img_info['name']}_det.jpg"
        pytorch_success = run_pytorch_inference_simple(img_info['original'], pytorch_output)
        
        if pytorch_success:
            successful_pytorch += 1
        
        # 创建三图对比
        comparison_output = f"DELIVERABLES/images/real_comparisons/{img_info['name']}_three_way_comparison.png"
        create_three_way_comparison(
            img_info['original'],
            pytorch_output if pytorch_success else None,
            img_info['jittor_result'],
            comparison_output,
            img_info['name']
        )
    
    # 创建对比网格
    create_comparison_grid()
    
    print(f"\n📊 处理结果:")
    print(f"   总图片数: {min(4, len(images))}")
    print(f"   PyTorch推理成功: {successful_pytorch}")
    print(f"   Jittor结果可用: {min(4, len(images))}")
    
    # 清理临时文件
    if os.path.exists("temp_pytorch_results"):
        import shutil
        # 不删除，保留结果
        print(f"   PyTorch结果保存在: temp_pytorch_results/")
    
    print("\n🎉 简单对比完成！")
    print("📁 输出目录: DELIVERABLES/images/real_comparisons/")

if __name__ == "__main__":
    main()
