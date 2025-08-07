#!/usr/bin/env python3
"""
🔍 Jittor 基础功能测试
验证 loss 定义和优化器是否正常工作
"""

import sys
import os
sys.path.insert(0, '.')

import jittor as jt

def test_basic_functionality():
    print("🔍 测试 Jittor 基础功能")
    print("=" * 50)
    
    # 设置环境变量
    os.environ['DISABLE_MULTIPROCESSING'] = '1'
    
    # 启用 CUDA
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    print(f"✅ CUDA: {'启用' if jt.flags.use_cuda else '禁用'}")
    
    try:
        print("\n🧪 测试1: 简单线性模型")
        
        # 简单模型和数据
        model = jt.nn.Linear(10, 2)
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
        x = jt.randn(2, 10)
        label = jt.array([0, 1])
        
        print(f"   模型: Linear(10, 2)")
        print(f"   输入形状: {x.shape}")
        print(f"   标签形状: {label.shape}")

        # 前向计算与loss
        pred = model(x)
        loss = jt.nn.cross_entropy_loss(pred, label)

        print(f"   预测形状: {pred.shape}")
        print(f"   Loss类型: {type(loss)}")
        print(f"   Loss形状: {loss.shape}")
        
        # 验证 loss 是 Jittor Var 类型
        assert isinstance(loss, jt.Var), f"Loss 不是 Jittor Var 类型: {type(loss)}"
        print("   ✅ Loss 是正确的 Jittor Var 类型")
        
        # 反向传播与优化
        optimizer.zero_grad()
        optimizer.step(loss)
        
        print("   ✅ 基础训练步骤成功！")
        
    except Exception as e:
        print(f"   ❌ 基础测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("\n🧪 测试2: 复杂一点的模型")
        
        # 稍微复杂的模型
        model = jt.nn.Sequential(
            jt.nn.Linear(10, 20),
            jt.nn.ReLU(),
            jt.nn.Linear(20, 5),
            jt.nn.ReLU(),
            jt.nn.Linear(5, 2)
        )
        
        optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
        
        # 多个批次测试
        for i in range(3):
            x = jt.randn(4, 10)  # 批次大小为4
            label = jt.randint(0, 2, (4,))
            
            pred = model(x)
            loss = jt.nn.cross_entropy_loss(pred, label)
            
            optimizer.zero_grad()
            optimizer.step(loss)
            
            print(f"   批次 {i+1}: Loss形状 {loss.shape}")
        
        print("   ✅ 复杂模型测试成功！")
        
    except Exception as e:
        print(f"   ❌ 复杂模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_nanodet_loss():
    print("\n🧪 测试3: NanoDet 相关的 Loss 计算")
    print("-" * 50)
    
    try:
        # 模拟 NanoDet 的损失计算
        from nanodet.util import cfg, load_config
        
        # 加载配置
        load_config(cfg, 'config/nanodet-plus-m_320_voc_bs64_50epochs.yml')
        
        # 创建模拟的损失数据
        batch_size = 2
        num_classes = 20
        
        # 模拟分类损失
        pred_scores = jt.randn(batch_size, 1000, num_classes)  # 模拟预测分数
        labels = jt.randint(0, num_classes, (batch_size, 1000))  # 模拟标签
        
        # 计算交叉熵损失
        loss_cls = jt.nn.cross_entropy_loss(pred_scores.view(-1, num_classes), labels.view(-1))
        
        print(f"   分类损失形状: {loss_cls.shape}")
        print(f"   分类损失类型: {type(loss_cls)}")
        
        # 模拟回归损失
        pred_bbox = jt.randn(batch_size, 1000, 4)  # 模拟边界框预测
        target_bbox = jt.randn(batch_size, 1000, 4)  # 模拟目标边界框
        
        loss_bbox = jt.nn.mse_loss(pred_bbox, target_bbox)
        
        print(f"   回归损失形状: {loss_bbox.shape}")
        print(f"   回归损失类型: {type(loss_bbox)}")
        
        # 总损失
        total_loss = loss_cls + loss_bbox
        
        print(f"   总损失形状: {total_loss.shape}")
        print(f"   总损失类型: {type(total_loss)}")
        
        # 验证损失是正确的 Jittor Var 类型
        assert isinstance(total_loss, jt.Var), f"总损失不是 Jittor Var 类型: {type(total_loss)}"
        print("   ✅ NanoDet 风格的损失计算成功！")
        
        return True
        
    except Exception as e:
        print(f"   ❌ NanoDet 损失测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 Jittor 基础功能验证")
    print("根据错误分析，验证 loss 定义是否符合 Jittor 规范")
    print("=" * 60)
    
    # 测试基础功能
    basic_ok = test_basic_functionality()
    
    if basic_ok:
        print("\n✅ 基础功能测试通过！")
        
        # 测试 NanoDet 相关功能
        nanodet_ok = test_nanodet_loss()
        
        if nanodet_ok:
            print("\n🎉 所有测试通过！")
            print("✅ Jittor 基础功能正常")
            print("✅ Loss 定义符合规范")
            print("✅ 可以进行下一步调试")
        else:
            print("\n⚠️ NanoDet 损失测试失败")
            print("问题可能出现在复杂的损失计算逻辑中")
    else:
        print("\n❌ 基础功能测试失败")
        print("问题出现在 Jittor 环境或基础配置中")
        print("建议检查 CUDA 环境和 Jittor 安装")

if __name__ == "__main__":
    main()
