#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度分析PyTorch训练好的模型架构
精确到每个参数的形状、通道数、层级结构
"""

import os
import sys
import torch
import json
from collections import OrderedDict

def analyze_pytorch_model():
    """深度分析PyTorch模型"""
    
    # 加载PyTorch模型
    model_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    
    print("=" * 80)
    print("🔍 深度分析PyTorch NanoDet模型架构")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    print(f"✓ 成功加载模型: {model_path}")
    print(f"✓ 总参数数量: {len(state_dict)}")
    
    # 分析参数结构
    analysis_result = {
        "model_info": {
            "total_params": len(state_dict),
            "model_path": model_path
        },
        "backbone": {},
        "fpn": {},
        "head": {},
        "aux_head": {},
        "other": {}
    }
    
    # 按模块分类分析
    for param_name, param_tensor in state_dict.items():
        param_shape = list(param_tensor.shape)
        param_numel = param_tensor.numel()
        param_dtype = str(param_tensor.dtype)
        
        # 处理不同数据类型
        if param_tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            param_info = {
                "shape": param_shape,
                "numel": param_numel,
                "dtype": param_dtype,
                "mean": float(param_tensor.mean()),
                "std": float(param_tensor.std()),
                "min": float(param_tensor.min()),
                "max": float(param_tensor.max())
            }
        else:
            param_info = {
                "shape": param_shape,
                "numel": param_numel,
                "dtype": param_dtype,
                "mean": "N/A",
                "std": "N/A",
                "min": float(param_tensor.min()),
                "max": float(param_tensor.max())
            }
        
        # 分类到不同模块
        if "backbone" in param_name:
            analysis_result["backbone"][param_name] = param_info
        elif "fpn" in param_name:
            analysis_result["fpn"][param_name] = param_info
        elif "head" in param_name and "aux_head" not in param_name:
            analysis_result["head"][param_name] = param_info
        elif "aux_head" in param_name:
            analysis_result["aux_head"][param_name] = param_info
        else:
            analysis_result["other"][param_name] = param_info
    
    # 打印详细分析
    print("\n" + "=" * 80)
    print("📊 模块参数统计")
    print("=" * 80)
    
    for module_name, module_params in analysis_result.items():
        if module_name == "model_info":
            continue
            
        if module_params:
            total_params = sum(p["numel"] for p in module_params.values())
            print(f"\n🔹 {module_name.upper()}:")
            print(f"   参数数量: {len(module_params)}")
            print(f"   总参数量: {total_params:,}")
            
            # 显示前5个参数的详细信息
            for i, (param_name, param_info) in enumerate(module_params.items()):
                if i < 5:  # 只显示前5个
                    print(f"   {param_name}:")
                    print(f"     形状: {param_info['shape']}")
                    print(f"     参数量: {param_info['numel']:,}")
                    if isinstance(param_info['min'], float):
                        print(f"     数值范围: [{param_info['min']:.6f}, {param_info['max']:.6f}]")
                    else:
                        print(f"     数值范围: [{param_info['min']}, {param_info['max']}]")
                elif i == 5:
                    print(f"   ... 还有 {len(module_params) - 5} 个参数")
                    break
    
    # 保存详细分析结果
    output_file = "pytorch_model_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 详细分析结果已保存到: {output_file}")
    
    # 生成参数检查清单
    generate_parameter_checklist(analysis_result)
    
    return True


def generate_parameter_checklist(analysis_result):
    """生成参数检查清单"""
    
    print("\n" + "=" * 80)
    print("📋 生成参数检查清单")
    print("=" * 80)
    
    checklist = []
    
    for module_name, module_params in analysis_result.items():
        if module_name == "model_info":
            continue
            
        if module_params:
            checklist.append(f"\n## {module_name.upper()} 模块参数检查")
            checklist.append(f"总参数数: {len(module_params)}")
            checklist.append("")
            
            for param_name, param_info in module_params.items():
                status = "[ ]"  # 未检查
                checklist.append(f"{status} **{param_name}**")
                checklist.append(f"   - 形状: {param_info['shape']}")
                checklist.append(f"   - 参数量: {param_info['numel']:,}")
                if isinstance(param_info['min'], float):
                    checklist.append(f"   - 数值范围: [{param_info['min']:.6f}, {param_info['max']:.6f}]")
                else:
                    checklist.append(f"   - 数值范围: [{param_info['min']}, {param_info['max']}]")
                checklist.append(f"   - Jittor对应: _待检查_")
                checklist.append("")
    
    # 保存检查清单
    checklist_file = "parameter_checklist.md"
    with open(checklist_file, 'w', encoding='utf-8') as f:
        f.write("# NanoDet参数检查清单\n\n")
        f.write("## 检查说明\n")
        f.write("- [ ] 未检查\n")
        f.write("- [/] 检查中\n") 
        f.write("- [x] 检查通过\n")
        f.write("- [-] 检查失败\n\n")
        f.write("\n".join(checklist))
    
    print(f"✓ 参数检查清单已生成: {checklist_file}")


def analyze_specific_layers():
    """分析特定层的详细结构"""
    
    print("\n" + "=" * 80)
    print("🔬 分析关键层结构")
    print("=" * 80)
    
    model_path = "/home/kyc/project/nanodet/nanodet-pytorch/workspace/nanodet-plus-m_320_voc_bs64/model_best/model_best.ckpt"
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 分析backbone第一层
    print("\n🔹 Backbone第一层分析:")
    for name, param in state_dict.items():
        if "backbone.conv1" in name:
            print(f"   {name}: {list(param.shape)}")
    
    # 分析head层
    print("\n🔹 Head层分析:")
    head_params = {}
    for name, param in state_dict.items():
        if "head." in name and "aux_head" not in name:
            head_params[name] = param.shape
    
    for name, shape in sorted(head_params.items()):
        print(f"   {name}: {list(shape)}")
    
    # 分析输出层
    print("\n🔹 输出层分析:")
    for name, param in state_dict.items():
        if "gfl_cls" in name or "gfl_reg" in name:
            print(f"   {name}: {list(param.shape)}")
            if len(param.shape) >= 2:
                print(f"     输入通道: {param.shape[1]}")
                print(f"     输出通道: {param.shape[0]}")


def main():
    """主函数"""
    print("🚀 开始深度分析PyTorch NanoDet模型")
    
    success = analyze_pytorch_model()
    
    if success:
        analyze_specific_layers()
        print("\n🎉 模型分析完成!")
        print("📋 下一步: 使用parameter_checklist.md逐个检查Jittor参数")
    else:
        print("\n❌ 模型分析失败")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
