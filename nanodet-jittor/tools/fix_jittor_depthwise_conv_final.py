#!/usr/bin/env python3
"""
修复 Jittor depthwise_conv.py 中的 jt.code 调用问题 - 最终版本
使用正确的多输出语法：jt.code([shape1, shape2], [dtype1, dtype2], inputs, ...)
"""

import os
import sys
import shutil
from pathlib import Path

def fix_depthwise_conv_grad(depthwise_conv_path):
    """修复 depthwise_conv.py 中的 grad 方法"""
    print(f"正在修复 {depthwise_conv_path}...")
    
    # 读取原始文件
    with open(depthwise_conv_path, 'r') as f:
        content = f.read()
    
    # 查找需要修复的代码段 - 找到修复后的 jt.code_ 调用
    code_start = content.find('input_grad, weight_grad = jt.code_([x.shape, weight.shape], [x.dtype, weight.dtype], [x, weight, grad],')
    if code_start == -1:
        print("❌ 未找到需要修复的 jt.code_ 调用")
        return False
    
    # 找到这个 jt.code_ 调用的结束位置
    bracket_count = 0
    end_idx = code_start
    in_string = False
    escape_next = False
    
    for i, char in enumerate(content[code_start:], code_start):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '(':
                bracket_count += 1
            elif char == ')':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
    
    if bracket_count != 0:
        print("❌ 无法找到 jt.code_ 调用的结束位置")
        return False
    
    # 提取原始的 jt.code_ 调用
    original_call = content[code_start:end_idx]
    print(f"找到原始调用，长度: {len(original_call)} 字符")
    
    # 提取 cuda_header 和 cuda_src 内容
    header_start = original_call.find('cuda_header')
    if header_start == -1:
        print("❌ 未找到 cuda_header")
        return False
    
    cuda_content = original_call[header_start:]
    
    # 创建修复后的调用
    # 根据 jt.code 文档 Example-3，对于多个输出应该这样调用：
    # b,c = jt.code([(1,), (1,)], [a.dtype, a.dtype], [a], ...)
    fixed_call = f"""input_grad, weight_grad = jt.code([x.shape, weight.shape], [x.dtype, weight.dtype], [x, weight, grad],
        {cuda_content}"""
    
    # 替换内容
    new_content = content[:code_start] + fixed_call + content[end_idx:]
    
    # 写入修复后的文件
    with open(depthwise_conv_path, 'w') as f:
        f.write(new_content)
    
    print("✅ 修复完成")
    return True


def test_fix():
    """测试修复是否成功"""
    print("正在测试修复...")
    
    try:
        import jittor as jt
        from jittor import nn
        
        # 创建一个简单的 depthwise 卷积测试
        conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)
        x = jt.randn(1, 3, 8, 8)
        
        # 前向传播
        y = conv(x)
        print(f"✅ 前向传播成功: {x.shape} -> {y.shape}")
        
        # 测试梯度计算
        loss = y.sum()
        grads = jt.grad(loss, conv.parameters())
        print(f"✅ 梯度计算成功: {len(grads)} 个参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=== Jittor DepthwiseConv 修复工具 - 最终版本 ===")
    
    # 获取 Jittor 安装路径
    try:
        import jittor as jt
        jittor_path = os.path.dirname(jt.__file__)
        print(f"Jittor 安装路径: {jittor_path}")
    except ImportError:
        print("❌ 无法导入 Jittor")
        return 1
    
    depthwise_conv_path = os.path.join(jittor_path, 'depthwise_conv.py')
    backup_path = depthwise_conv_path + '.backup'
    
    # 检查文件是否存在
    if not os.path.exists(depthwise_conv_path):
        print(f"❌ 文件不存在: {depthwise_conv_path}")
        return 1
    
    # 修复文件
    if not fix_depthwise_conv_grad(depthwise_conv_path):
        print("❌ 修复失败")
        return 1
    
    # 测试修复
    if not test_fix():
        print("❌ 修复测试失败，恢复原始文件")
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, depthwise_conv_path)
        return 1
    
    print("🎉 修复成功！")
    return 0


if __name__ == '__main__':
    sys.exit(main())
