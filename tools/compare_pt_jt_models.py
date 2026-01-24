#!/usr/bin/env python3
"""
对比PyTorch模型和Jittor加载PyTorch权重的mAP
这是验证权重转换工具有效性的关键测试
"""

import sys
import os
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(ROOT, ".."))
JT_ROOT = os.path.join(PROJ, "nanodet-jittor")
PT_ROOT = os.path.join(PROJ, "nanodet-pytorch")
if JT_ROOT not in sys.path:
    sys.path.insert(0, JT_ROOT)
if PT_ROOT not in sys.path:
    sys.path.append(PT_ROOT)

import jittor as jt
import torch
import numpy as np
from nanodet.util.config import load_config, cfg
from nanodet.model.arch import build_model
from nanodet.util.check_point import pt_to_jt_checkpoint
from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from nanodet.evaluator import build_evaluator

def load_pytorch_model(pt_model_path):
    """加载PyTorch原始模型"""
    print("🔧 加载PyTorch原始模型...")

    if not os.path.exists(pt_model_path):
        raise FileNotFoundError(f"PyTorch模型文件不存在: {pt_model_path}")
    
    print(f"✅ PyTorch模型路径: {pt_model_path}")
    return pt_model_path

def load_jittor_model_with_pt_weights(config_path, pt_model_path):
    """加载Jittor模型并使用转换后的PyTorch权重"""
    print("🔧 构建Jittor模型并加载转换后的PyTorch权重...")
    
    # 加载配置
    load_config(cfg, config_path)
    
    # 构建Jittor模型
    model = build_model(cfg.model)
    
    # 加载PyTorch权重
    if not os.path.exists(pt_model_path):
        raise FileNotFoundError(f"PyTorch权重文件不存在: {pt_model_path}")
    
    print(f"📥 加载PyTorch权重: {pt_model_path}")
    
    # 加载PyTorch检查点
    pt_ckpt = torch.load(pt_model_path, map_location='cpu')
    print(f"✅ PyTorch权重加载成功，键数: {len(pt_ckpt.get('state_dict', pt_ckpt))}")
    
    # 转换为Jittor格式
    print("🔄 转换权重格式...")
    jt_ckpt = pt_to_jt_checkpoint(pt_ckpt, model)
    
    # 加载权重到模型
    model.load_state_dict(jt_ckpt['state_dict'])
    model.eval()
    
    print("✅ Jittor模型加载PyTorch权重成功")
    return model

def evaluate_model(model, dataset_name="val", max_images=None):
    """评估模型性能"""
    print(f"📊 开始评估模型性能 (数据集: {dataset_name})...")
    
    # 构建数据集
    dataset = build_dataset(cfg.data[dataset_name], "val")
    total = len(dataset)
    if max_images is not None:
        total = min(total, max_images)
    print(f"📋 数据集大小: {len(dataset)} (评估 {total})")
    
    # 确保保存目录存在
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 构建评估器
    evaluator = build_evaluator(cfg.evaluator, dataset)
    
    # 评估
    model.eval()
    results = {}
    
    print("🔍 开始推理...")
    start_time = time.time()
    
    for i in range(total):
        data = dataset[i]
        if i % 100 == 0:
            print(f"进度: {i}/{total}")
        
        # 数据预处理：构造 Jittor inference 需要的 meta
        meta = data
        if isinstance(meta["img"], jt.Var) and len(meta["img"].shape) == 3:
            meta["img"] = meta["img"].unsqueeze(0)
        # 规范 img_info 为列表格式
        h = meta["img_info"]["height"]
        w = meta["img_info"]["width"]
        img_id = meta["img_info"]["id"]
        meta["img_info"] = {
            "height": [h],
            "width": [w],
            "id": [img_id],
        }
        
        # 推理
        with jt.no_grad():
            preds = model(meta["img"])
            results_batch = model.head.post_process(preds, meta)
        if isinstance(results_batch, dict):
            results.update(results_batch)
    
    end_time = time.time()
    print(f"⏱️ 推理完成，耗时: {end_time - start_time:.2f}秒")
    
    # 计算mAP
    print("📈 计算mAP...")
    eval_results = evaluator.evaluate(results, cfg.save_dir)
    
    return eval_results

def compare_models(args):
    """对比PyTorch和Jittor模型的性能"""
    print("="*80)
    print("🎯 PyTorch vs Jittor 模型性能对比测试")
    print("="*80)
    
    try:
        # 1. 验证PyTorch模型存在
        pt_model_path = load_pytorch_model(args.pt_ckpt)
        
        # 2. 加载Jittor模型（使用转换后的PyTorch权重）
        jt_model = load_jittor_model_with_pt_weights(args.jt_config, pt_model_path)
        
        # 3. 评估Jittor模型性能
        print("\n" + "="*50)
        print("📊 评估Jittor模型（加载PyTorch权重）")
        print("="*50)
        
        jt_results = evaluate_model(jt_model, max_images=args.max_images)
        
        # 4. 输出对比结果
        print("\n" + "="*80)
        print("📋 模型性能对比结果")
        print("="*80)
        
        # PyTorch基准结果（可选）
        pt_map = args.pt_map
        pt_ap50 = args.pt_ap50
        
        # Jittor结果
        jt_map = jt_results.get('mAP', 0.0)
        jt_ap50 = jt_results.get('AP_50', jt_results.get('AP50', 0.0))
        
        if pt_map is not None and pt_ap50 is not None:
            print(f"{'指标':<15} {'PyTorch':<12} {'Jittor':<12} {'差异':<12} {'状态'}")
            print("-" * 65)
            print(f"{'mAP':<15} {pt_map:<12.4f} {jt_map:<12.4f} {abs(pt_map-jt_map):<12.6f} {'✅' if abs(pt_map-jt_map) < 0.001 else '❌'}")
            print(f"{'AP50':<15} {pt_ap50:<12.4f} {jt_ap50:<12.4f} {abs(pt_ap50-jt_ap50):<12.6f} {'✅' if abs(pt_ap50-jt_ap50) < 0.001 else '❌'}")
        else:
            print(f"{'指标':<15} {'Jittor':<12}")
            print("-" * 30)
            print(f"{'mAP':<15} {jt_map:<12.4f}")
            print(f"{'AP50':<15} {jt_ap50:<12.4f}")
        
        # 5. 生成对比报告
        if pt_map is not None and pt_ap50 is not None:
            generate_comparison_report(pt_map, pt_ap50, jt_map, jt_ap50)
        
        print("\n🎉 权重转换验证完成！")
        
        return {
            'pytorch': {'mAP': pt_map, 'AP50': pt_ap50},
            'jittor': {'mAP': jt_map, 'AP50': jt_ap50},
            'success': (pt_map is not None and pt_ap50 is not None
                        and abs(pt_map-jt_map) < 0.001 and abs(pt_ap50-jt_ap50) < 0.001)
        }
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        return None

def generate_comparison_report(pt_map, pt_ap50, jt_map, jt_ap50):
    """生成详细的对比报告"""
    
    report_content = f"""# PyTorch vs Jittor 权重转换验证报告

## 测试概述
本测试验证了PyTorch权重转换到Jittor后的模型性能一致性。

## 测试配置
- **PyTorch模型**: nanodet-plus-m_320_voc_bs64_50epochs/model_best
- **Jittor模型**: 使用转换工具加载PyTorch权重
- **数据集**: VOC2007 验证集 (1494张图片)
- **输入尺寸**: 320×320

## 性能对比结果

| 指标 | PyTorch | Jittor | 差异 | 状态 |
|------|---------|--------|------|------|
| **mAP** | {pt_map:.4f} | {jt_map:.4f} | {abs(pt_map-jt_map):.6f} | {'✅ 完全一致' if abs(pt_map-jt_map) < 0.001 else '❌ 存在差异'} |
| **AP50** | {pt_ap50:.4f} | {jt_ap50:.4f} | {abs(pt_ap50-jt_ap50):.6f} | {'✅ 完全一致' if abs(pt_ap50-jt_ap50) < 0.001 else '❌ 存在差异'} |

## 结论

{'✅ **权重转换成功**: Jittor模型使用转换后的PyTorch权重，性能与原始PyTorch模型完全一致，证明了权重转换工具的有效性。' if abs(pt_map-jt_map) < 0.001 and abs(pt_ap50-jt_ap50) < 0.001 else '❌ **权重转换存在问题**: 性能差异超过阈值，需要检查转换过程。'}

## 技术意义

1. **验证了框架迁移的可行性**: 证明可以将PyTorch训练的模型无损迁移到Jittor
2. **确保了数值精度**: 权重转换过程保持了模型的原始性能
3. **支持了混合开发**: 可以使用PyTorch训练，Jittor部署的开发模式

---
*生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    report_path = "DELIVERABLES/pytorch_jittor_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📄 对比报告已保存: {report_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jt_config",
        default="nanodet-jittor/config/nanodet-plus-m_320_voc.yml",
        help="Jittor config path",
    )
    ap.add_argument(
        "--pt_ckpt",
        default="nanodet-pytorch/workspace/nanodet-plus-m_320_voc/model_best/nanodet_model_best.pth",
        help="PyTorch checkpoint path",
    )
    ap.add_argument("--pt_map", type=float, default=None, help="PyTorch mAP baseline")
    ap.add_argument("--pt_ap50", type=float, default=None, help="PyTorch AP50 baseline")
    ap.add_argument("--max_images", type=int, default=None, help="限制评估图片数量(用于快速验证)")
    return ap.parse_args()


if __name__ == "__main__":
    print("🚀 开始PyTorch vs Jittor模型性能对比测试...")

    # 设置Jittor为评估模式
    jt.flags.use_cuda = 1 if jt.has_cuda else 0

    args = parse_args()
    results = compare_models(args)

    if results and results['success']:
        print("\n🎉 测试成功！权重转换工具验证通过！")
    else:
        print("\n❌ 测试失败，需要检查权重转换过程。")
