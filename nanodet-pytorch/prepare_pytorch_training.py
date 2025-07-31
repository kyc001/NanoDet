#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch版本训练准备脚本
检查环境、数据集，并提供训练指导
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_pytorch_environment():
    """检查PyTorch环境"""
    print("=" * 60)
    print("PyTorch Environment Check")
    print("=" * 60)
    
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*NumPy.*")

        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        try:
            import torchvision
            print(f"✓ TorchVision version: {torchvision.__version__}")
        except Exception as e:
            print(f"⚠ TorchVision warning (but available): {str(e)[:50]}...")

        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠ CUDA not available, will use CPU training")

        return True
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False


def check_dependencies():
    """检查依赖包（修复了包名与模块名映射问题）"""
    print("\n" + "=" * 60)
    print("Dependencies Check")
    print("=" * 60)
    
    # 关键修复：建立「安装包名→导入模块名」的映射关系
    required_packages = {
        'pytorch_lightning': 'pytorch_lightning',
        'opencv-python': 'cv2',          # 安装名≠导入名
        'pycocotools': 'pycocotools',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'Pillow': 'PIL',                 # 安装名≠导入名
        'tqdm': 'tqdm',
        'tensorboard': 'tensorboard'
    }
    
    missing_packages = []
    
    for install_name, import_name in required_packages.items():
        try:
            import warnings
            warnings.filterwarnings("ignore")
            __import__(import_name)  # 使用正确的导入名
            print(f"✓ {install_name}")
        except ImportError:
            print(f"✗ {install_name} - Missing")
            missing_packages.append(install_name)  # 提示时用安装名
        except Exception as e:
            print(f"⚠ {install_name} - Available but with warnings")
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def check_voc_dataset():
    """检查VOC数据集"""
    print("\n" + "=" * 60)
    print("VOC Dataset Check")
    print("=" * 60)
    
    # 检查数据集路径
    data_paths = [
        'data/VOC_mini',
        '../nanodet-jittor/data/VOC_mini',
        'data/VOCdevkit'
    ]
    
    dataset_path = None
    for path in data_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("✗ VOC dataset not found")
        print("Please prepare VOC dataset using one of these methods:")
        print("1. cd ../nanodet-jittor && python tools/create_mini_voc_dataset.py")
        print("2. cd ../nanodet-jittor && python tools/download_voc_dataset.py --download")
        return False
    
    print(f"✓ Dataset found at: {dataset_path}")
    
    # 检查标注文件
    ann_files = [
        os.path.join(dataset_path, 'annotations/voc_train.json'),
        os.path.join(dataset_path, 'annotations/voc_val.json')
    ]
    
    for ann_file in ann_files:
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                data = json.load(f)
            print(f"✓ {os.path.basename(ann_file)}: {len(data['images'])} images, {len(data['annotations'])} annotations")
        else:
            print(f"✗ {ann_file} not found")
            return False
    
    # 检查图片目录
    img_dir = os.path.join(dataset_path, 'images')
    if os.path.exists(img_dir):
        img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Images directory: {img_count} images")
    else:
        print(f"✗ Images directory not found: {img_dir}")
        return False
    
    return True, dataset_path


def update_config_paths(dataset_path):
    """更新配置文件中的数据集路径"""
    print("\n" + "=" * 60)
    print("Config Update")
    print("=" * 60)
    
    config_file = 'config/nanodet-plus-m_320_voc.yml'
    
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        return False
    
    # 读取配置文件
    with open(config_file, 'r') as f:
        content = f.read()
    
    # 更新路径
    abs_dataset_path = os.path.abspath(dataset_path)
    img_path = os.path.join(abs_dataset_path, 'images')
    train_ann_path = os.path.join(abs_dataset_path, 'annotations/voc_train.json')
    val_ann_path = os.path.join(abs_dataset_path, 'annotations/voc_val.json')
    
    # 替换路径
    content = content.replace('data/VOC_mini/images', img_path)
    content = content.replace('data/VOC_mini/annotations/voc_train.json', train_ann_path)
    content = content.replace('data/VOC_mini/annotations/voc_val.json', val_ann_path)
    
    # 写回配置文件
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated config file: {config_file}")
    print(f"  Image path: {img_path}")
    print(f"  Train annotations: {train_ann_path}")
    print(f"  Val annotations: {val_ann_path}")
    
    return True


def install_nanodet():
    """安装NanoDet包"""
    print("\n" + "=" * 60)
    print("NanoDet Installation")
    print("=" * 60)
    
    try:
        # 检查是否已安装
        import nanodet
        print("✓ NanoDet already installed")
        return True
    except ImportError:
        print("Installing NanoDet...")
        try:
            subprocess.run([sys.executable, 'setup.py', 'develop'], check=True)
            print("✓ NanoDet installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install NanoDet: {e}")
            return False


def main():
    """主函数"""
    print("NanoDet PyTorch Training Preparation")
    print("Preparing PyTorch version training for baseline results...")
    
    # 检查环境
    if not check_pytorch_environment():
        print("\n❌ PyTorch environment check failed!")
        return False
    
    if not check_dependencies():
        print("\n❌ Dependencies check failed!")
        return False
    
    # 检查数据集
    dataset_check = check_voc_dataset()
    if isinstance(dataset_check, tuple):
        success, dataset_path = dataset_check
        if not success:
            print("\n❌ VOC dataset check failed!")
            return False
    else:
        print("\n❌ VOC dataset check failed!")
        return False
    
    # 更新配置
    if not update_config_paths(dataset_path):
        print("\n❌ Config update failed!")
        return False
    
    # 安装NanoDet
    if not install_nanodet():
        print("\n❌ NanoDet installation failed!")
        return False
    
    # 显示训练命令
    print("\n" + "=" * 60)
    print("Ready to Train!")
    print("=" * 60)
    print("\n🎉 All checks passed! Ready to start PyTorch training.")
    print("\nTraining commands:")
    print("1. Quick test (few epochs):")
    print("   python tools/train.py config/nanodet-plus-m_320_voc.yml")
    print("\n2. Full training:")
    print("   python tools/train.py config/nanodet-plus-m_320_voc.yml")
    print("\n3. Resume training:")
    print("   python tools/train.py config/nanodet-plus-m_320_voc.yml --resume")
    print("\n4. Test model:")
    print("   python tools/test.py config/nanodet-plus-m_320_voc.yml --checkpoint workspace/nanodet-plus-m_320_voc/model_best.ckpt")
    
    print("\n📊 Expected results:")
    print("- Training time: ~2-4 hours for 100 epochs")
    print("- Memory usage: ~6-7GB on RTX4060")
    print("- mAP target: 40-50% on VOC dataset")
    print("- Model size: ~1.2MB")
    
    print("\n📝 Training logs will be saved to:")
    print("- Checkpoints: workspace/nanodet-plus-m_320_voc/")
    print("- TensorBoard: workspace/nanodet-plus-m_320_voc/tb_logs/")
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        print("\n❌ Preparation failed! Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n✅ Preparation completed successfully!")
        sys.exit(0)
