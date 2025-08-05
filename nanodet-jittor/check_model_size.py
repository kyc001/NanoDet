#!/usr/bin/env python3
"""
🔍 检查模型参数数量和内存使用
"""

import sys
sys.path.insert(0, '.')

import jittor as jt
from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model

def count_parameters(model):
    """统计模型参数数量"""
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        # 显示大参数层
        if param_count > 100000:  # 超过10万参数的层
            print(f"大参数层: {name} - {param_count:,} 参数, 形状: {param.shape}")
    
    return total_params, trainable_params

def analyze_depthwise_modules(model):
    """分析 DepthwiseConvModule 的参数使用"""
    depthwise_count = 0
    depthwise_params = 0
    
    def count_depthwise(module, prefix=""):
        nonlocal depthwise_count, depthwise_params
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if "DepthwiseConvModule" in str(type(child)):
                depthwise_count += 1
                module_params = sum(p.numel() for p in child.parameters())
                depthwise_params += module_params
                
                print(f"DepthwiseConvModule #{depthwise_count}: {full_name}")
                print(f"  参数数量: {module_params:,}")
                
                # 检查 depthwise_convs
                if hasattr(child, 'depthwise_convs'):
                    conv_count = len(child.depthwise_convs)
                    conv_params = sum(sum(p.numel() for p in conv.parameters()) 
                                    for conv in child.depthwise_convs)
                    print(f"  独立卷积数量: {conv_count}")
                    print(f"  独立卷积参数: {conv_params:,}")
                
                print()
            else:
                count_depthwise(child, full_name)
    
    count_depthwise(model)
    return depthwise_count, depthwise_params

def main():
    print("🔍 开始检查模型大小和内存使用...")
    
    # 设置 Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    
    # 加载配置
    load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
    print(f"✅ 配置加载成功")
    
    # 创建模型
    print("🔍 创建模型...")
    model = build_model(cfg.model)
    print(f"✅ 模型创建成功")
    
    # 统计总参数
    print("\n🔍 统计模型参数...")
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n📊 模型参数统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 分析 DepthwiseConvModule
    print("\n🔍 分析 DepthwiseConvModule...")
    dw_count, dw_params = analyze_depthwise_modules(model)
    
    print(f"📊 DepthwiseConvModule 统计:")
    print(f"  模块数量: {dw_count}")
    print(f"  总参数数量: {dw_params:,}")
    print(f"  占总参数比例: {dw_params/total_params*100:.1f}%")
    
    # 估算内存使用
    print(f"\n💾 内存使用估算:")
    
    # 参数内存
    param_memory = total_params * 4 / 1024 / 1024  # MB
    print(f"  参数内存: {param_memory:.2f} MB")
    
    # 梯度内存（与参数相同）
    grad_memory = param_memory
    print(f"  梯度内存: {grad_memory:.2f} MB")
    
    # 优化器状态内存（AdamW 需要 2x 参数内存）
    optimizer_memory = param_memory * 2
    print(f"  优化器内存: {optimizer_memory:.2f} MB")
    
    # 激活内存（估算）
    batch_size = cfg.device.batchsize_per_gpu
    input_size = 320  # 从配置中获取
    # 估算激活内存：batch_size * channels * height * width * 4 bytes * 层数
    activation_memory = batch_size * 96 * (input_size//8) * (input_size//8) * 4 * 20 / 1024 / 1024  # MB
    print(f"  激活内存估算: {activation_memory:.2f} MB")
    
    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
    print(f"  总内存估算: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)")
    
    # 检查是否超出 GPU 内存
    gpu_memory = 8000  # 8GB GPU
    if total_memory > gpu_memory:
        print(f"  ❌ 估算内存 ({total_memory:.0f}MB) 超出 GPU 内存 ({gpu_memory}MB)!")
        print(f"  超出: {total_memory - gpu_memory:.0f}MB")
    else:
        print(f"  ✅ 估算内存在 GPU 内存范围内")
    
    print("\n🎯 分析完成！")

if __name__ == "__main__":
    main()
