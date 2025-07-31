#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检测头功能测试脚本
测试NanoDetPlusHead是否正常工作
"""

import jittor as jt
import time
import traceback


def test_loss_functions():
    """测试损失函数"""
    print("Testing loss functions...")
    
    try:
        from nanodet.model.loss import QualityFocalLoss, DistributionFocalLoss, GIoULoss
        
        # 测试QualityFocalLoss
        qfl = QualityFocalLoss(beta=2.0, loss_weight=1.0)
        pred = jt.randn(10, 20)  # (N, num_classes)
        target_labels = jt.randint(0, 20, (10,))  # (N,)
        target_scores = jt.rand(10)  # (N,)
        target = (target_labels, target_scores)
        
        loss_qfl = qfl(pred, target)
        print(f"✓ QualityFocalLoss works, loss: {loss_qfl.item():.4f}")
        
        # 测试DistributionFocalLoss
        dfl = DistributionFocalLoss(loss_weight=0.25)
        pred_dist = jt.randn(100, 8)  # (N, reg_max+1)
        target_dist = jt.rand(100) * 7  # (N,) in range [0, reg_max]
        
        loss_dfl = dfl(pred_dist, target_dist)
        print(f"✓ DistributionFocalLoss works, loss: {loss_dfl.item():.4f}")
        
        # 测试GIoULoss
        giou_loss = GIoULoss(loss_weight=2.0)
        pred_bbox = jt.rand(50, 4) * 100  # (N, 4) in xyxy format
        target_bbox = jt.rand(50, 4) * 100  # (N, 4) in xyxy format
        
        loss_giou = giou_loss(pred_bbox, target_bbox)
        print(f"✓ GIoULoss works, loss: {loss_giou.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss functions test failed: {e}")
        traceback.print_exc()
        return False


def test_integral_module():
    """测试Integral模块"""
    print("\nTesting Integral module...")
    
    try:
        from nanodet.model.head.gfl_head import Integral
        
        # 测试不同reg_max值
        reg_max_values = [7, 15, 31]
        
        for reg_max in reg_max_values:
            integral = Integral(reg_max=reg_max)
            
            # 输入形状: (N, 4*(reg_max+1))
            batch_size = 8
            input_dim = 4 * (reg_max + 1)
            x = jt.randn(batch_size, input_dim)
            
            output = integral(x)
            
            print(f"✓ Integral(reg_max={reg_max}) works")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            
            # 验证输出形状
            expected_shape = (batch_size, 4)
            if output.shape != expected_shape:
                print(f"✗ Output shape mismatch: got {output.shape}, expected {expected_shape}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Integral module test failed: {e}")
        traceback.print_exc()
        return False


def test_nanodet_plus_head():
    """测试NanoDetPlusHead"""
    print("\nTesting NanoDetPlusHead...")
    
    try:
        from nanodet.model.head import build_head
        
        # 创建损失配置
        loss_cfg = type('LossCfg', (), {
            'loss_qfl': type('QFL', (), {'beta': 2.0, 'loss_weight': 1.0})(),
            'loss_dfl': type('DFL', (), {'loss_weight': 0.25})(),
            'loss_bbox': type('BBOX', (), {'loss_weight': 2.0})()
        })()
        
        # 创建检测头配置
        head_cfg = {
            'name': 'NanoDetPlusHead',
            'num_classes': 20,  # VOC数据集
            'loss': loss_cfg,
            'input_channel': 96,
            'feat_channels': 96,
            'stacked_convs': 2,
            'kernel_size': 5,
            'strides': [8, 16, 32, 64],
            'conv_type': 'DWConv',
            'norm_cfg': dict(type='BN'),
            'reg_max': 7,
            'activation': 'LeakyReLU'
        }
        
        head = build_head(head_cfg)
        print(f"✓ NanoDetPlusHead created")
        
        # 创建输入特征 (来自FPN的输出)
        feats = [
            jt.randn(1, 96, 40, 40),  # P3: stride 8
            jt.randn(1, 96, 20, 20),  # P4: stride 16
            jt.randn(1, 96, 10, 10),  # P5: stride 32
            jt.randn(1, 96, 5, 5),    # P6: stride 64
        ]
        
        print(f"Input feature shapes: {[f.shape for f in feats]}")
        
        # 前向传播
        start_time = time.time()
        with jt.no_grad():
            outputs = head(feats)
        end_time = time.time()
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Inference time: {(end_time - start_time)*1000:.2f}ms")
        
        # 验证输出形状
        # 输出应该是 (batch_size, total_points, num_classes + 4*(reg_max+1))
        batch_size = 1
        total_points = 40*40 + 20*20 + 10*10 + 5*5  # 1600 + 400 + 100 + 25 = 2125
        expected_channels = 20 + 4 * (7 + 1)  # num_classes + 4*(reg_max+1) = 20 + 32 = 52
        expected_shape = (batch_size, total_points, expected_channels)
        
        if outputs.shape != expected_shape:
            print(f"✗ Output shape mismatch: got {outputs.shape}, expected {expected_shape}")
            return False
        
        print(f"✓ Output shape correct: {outputs.shape}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in head.parameters())
        print(f"✓ Total parameters: {total_params/1e6:.2f}M")
        
        return True
    except Exception as e:
        print(f"✗ NanoDetPlusHead test failed: {e}")
        traceback.print_exc()
        return False


def test_full_model_integration():
    """测试完整模型集成"""
    print("\nTesting full model integration...")
    
    try:
        from nanodet.model.backbone import build_backbone
        from nanodet.model.fpn import build_fpn
        from nanodet.model.head import build_head
        
        # 创建backbone
        backbone_cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': False
        }
        backbone = build_backbone(backbone_cfg)
        
        # 创建FPN
        fpn_cfg = {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        }
        fpn = build_fpn(fpn_cfg)
        
        # 创建检测头
        loss_cfg = type('LossCfg', (), {
            'loss_qfl': type('QFL', (), {'beta': 2.0, 'loss_weight': 1.0})(),
            'loss_dfl': type('DFL', (), {'loss_weight': 0.25})(),
            'loss_bbox': type('BBOX', (), {'loss_weight': 2.0})()
        })()
        
        head_cfg = {
            'name': 'NanoDetPlusHead',
            'num_classes': 20,
            'loss': loss_cfg,
            'input_channel': 96,
            'feat_channels': 96,
            'stacked_convs': 2,
            'kernel_size': 5,
            'strides': [8, 16, 32, 64],
            'conv_type': 'DWConv',
            'norm_cfg': dict(type='BN'),
            'reg_max': 7,
            'activation': 'LeakyReLU'
        }
        head = build_head(head_cfg)
        
        # 端到端测试
        x = jt.randn(1, 3, 320, 320)
        
        start_time = time.time()
        with jt.no_grad():
            backbone_outputs = backbone(x)
            fpn_outputs = fpn(backbone_outputs)
            head_outputs = head(fpn_outputs)
        end_time = time.time()
        
        print(f"✓ Full model integration successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Backbone outputs: {[o.shape for o in backbone_outputs]}")
        print(f"  FPN outputs: {[o.shape for o in fpn_outputs]}")
        print(f"  Head output: {head_outputs.shape}")
        print(f"  Total inference time: {(end_time - start_time)*1000:.2f}ms")
        
        # 计算总参数量
        backbone_params = sum(p.numel() for p in backbone.parameters())
        fpn_params = sum(p.numel() for p in fpn.parameters())
        head_params = sum(p.numel() for p in head.parameters())
        total_params = backbone_params + fpn_params + head_params
        
        print(f"✓ Parameter count:")
        print(f"  Backbone: {backbone_params/1e6:.2f}M")
        print(f"  FPN: {fpn_params/1e6:.2f}M")
        print(f"  Head: {head_params/1e6:.2f}M")
        print(f"  Total: {total_params/1e6:.2f}M")
        
        return True
    except Exception as e:
        print(f"✗ Full model integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("NanoDetPlusHead Functionality Test")
    print("=" * 60)
    
    # 显示系统信息
    print(f"Jittor version: {jt.__version__}")
    print(f"CUDA available: {jt.has_cuda}")
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print(f"Using CUDA: {jt.flags.use_cuda}")
    
    # 运行测试
    tests = [
        ("Loss Functions", test_loss_functions),
        ("Integral Module", test_integral_module),
        ("NanoDetPlusHead", test_nanodet_plus_head),
        ("Full Model Integration", test_full_model_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All head tests passed! NanoDetPlusHead is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the error messages above.")
    
    return passed == len(results)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
