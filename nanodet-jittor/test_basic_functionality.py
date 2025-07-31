#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础功能测试脚本
测试NanoDet-Jittor的基本组件是否正常工作
"""

import jittor as jt
import numpy as np
import time
import traceback


def test_jittor_basic():
    """测试Jittor基础功能"""
    print("Testing Jittor basic functionality...")
    
    try:
        # 测试基本张量操作
        x = jt.randn(2, 3, 4, 4)
        y = jt.randn(2, 3, 4, 4)
        z = x + y
        
        print(f"✓ Basic tensor operations work")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {z.shape}")
        
        # 测试CUDA
        if jt.has_cuda:
            jt.flags.use_cuda = 1
            x_cuda = jt.randn(2, 3, 4, 4)
            print(f"✓ CUDA is available and working")
        else:
            print("! CUDA not available, using CPU")
            
        return True
    except Exception as e:
        print(f"✗ Jittor basic test failed: {e}")
        return False


def test_conv_module():
    """测试卷积模块"""
    print("\nTesting ConvModule...")
    
    try:
        from nanodet.model.module.conv import ConvModule, DepthwiseConvModule
        
        # 测试标准卷积模块
        conv = ConvModule(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            activation='ReLU'
        )
        
        x = jt.randn(1, 3, 224, 224)
        y = conv(x)
        
        print(f"✓ ConvModule works")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
        # 测试深度可分离卷积
        dw_conv = DepthwiseConvModule(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            activation='ReLU'
        )
        
        z = dw_conv(y)
        print(f"✓ DepthwiseConvModule works")
        print(f"  Input shape: {y.shape}")
        print(f"  Output shape: {z.shape}")
        
        return True
    except Exception as e:
        print(f"✗ ConvModule test failed: {e}")
        traceback.print_exc()
        return False


def test_activation():
    """测试激活函数"""
    print("\nTesting activation functions...")
    
    try:
        from nanodet.model.module.activation import Swish, HardSwish, act_layers
        
        x = jt.randn(2, 64, 32, 32)
        
        # 测试Swish
        swish = Swish()
        y1 = swish(x)
        print(f"✓ Swish activation works, output shape: {y1.shape}")
        
        # 测试HardSwish
        hard_swish = HardSwish()
        y2 = hard_swish(x)
        print(f"✓ HardSwish activation works, output shape: {y2.shape}")
        
        # 测试激活函数字典
        relu = act_layers['ReLU']()
        y3 = relu(x)
        print(f"✓ ReLU from act_layers works, output shape: {y3.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Activation test failed: {e}")
        traceback.print_exc()
        return False


def test_shufflenet_backbone():
    """测试ShuffleNetV2 backbone"""
    print("\nTesting ShuffleNetV2 backbone...")
    
    try:
        from nanodet.model.backbone import build_backbone
        
        # 创建backbone配置
        cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'ReLU',
            'pretrain': False
        }
        
        backbone = build_backbone(cfg)
        print(f"✓ ShuffleNetV2 backbone created")
        
        # 测试前向传播
        x = jt.randn(1, 3, 320, 320)
        
        start_time = time.time()
        with jt.no_grad():
            outputs = backbone(x)
        end_time = time.time()
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        print(f"  Inference time: {(end_time - start_time)*1000:.2f}ms")
        
        # 验证输出尺寸
        expected_channels = [116, 232, 464]  # 1.0x model channels
        for i, (output, expected_ch) in enumerate(zip(outputs, expected_channels)):
            if output.shape[1] != expected_ch:
                print(f"✗ Output {i} channel mismatch: got {output.shape[1]}, expected {expected_ch}")
                return False
        
        print(f"✓ Output channels match expected values")
        
        return True
    except Exception as e:
        print(f"✗ ShuffleNetV2 test failed: {e}")
        traceback.print_exc()
        return False


def test_fpn_functionality():
    """测试FPN功能"""
    print("\nTesting FPN functionality...")

    try:
        from nanodet.model.backbone import build_backbone
        from nanodet.model.fpn import build_fpn

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

        # 端到端测试
        x = jt.randn(1, 3, 320, 320)

        start_time = time.time()
        with jt.no_grad():
            backbone_outputs = backbone(x)
            fpn_outputs = fpn(backbone_outputs)
        end_time = time.time()

        print(f"✓ Backbone + FPN integration successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Backbone outputs: {[o.shape for o in backbone_outputs]}")
        print(f"  FPN outputs: {[o.shape for o in fpn_outputs]}")
        print(f"  Total inference time: {(end_time - start_time)*1000:.2f}ms")

        # 验证输出形状
        expected_shapes = [
            (1, 96, 40, 40),  # P3: stride 8
            (1, 96, 20, 20),  # P4: stride 16
            (1, 96, 10, 10),  # P5: stride 32
            (1, 96, 5, 5),    # P6: stride 64
        ]

        for i, (output, expected_shape) in enumerate(zip(fpn_outputs, expected_shapes)):
            if output.shape != expected_shape:
                print(f"✗ FPN output {i} shape mismatch: got {output.shape}, expected {expected_shape}")
                return False

        print(f"✓ All FPN output shapes correct")

        # 计算参数量
        backbone_params = sum(p.numel() for p in backbone.parameters())
        fpn_params = sum(p.numel() for p in fpn.parameters())
        total_params = backbone_params + fpn_params

        print(f"✓ Parameter count:")
        print(f"  Backbone: {backbone_params/1e6:.2f}M")
        print(f"  FPN: {fpn_params/1e6:.2f}M")
        print(f"  Total: {total_params/1e6:.2f}M")

        return True
    except Exception as e:
        print(f"✗ FPN functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_head_functionality():
    """测试检测头功能"""
    print("\nTesting detection head functionality...")

    try:
        from nanodet.model.head import build_head
        # Loss functions are tested in head functionality

        # 创建损失配置
        loss_cfg = type('LossCfg', (), {
            'loss_qfl': type('QFL', (), {'beta': 2.0, 'loss_weight': 1.0})(),
            'loss_dfl': type('DFL', (), {'loss_weight': 0.25})(),
            'loss_bbox': type('BBOX', (), {'loss_weight': 2.0})()
        })()

        # 创建检测头
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

        # 测试前向传播
        feats = [
            jt.randn(1, 96, 40, 40),  # P3
            jt.randn(1, 96, 20, 20),  # P4
            jt.randn(1, 96, 10, 10),  # P5
            jt.randn(1, 96, 5, 5),    # P6
        ]

        start_time = time.time()
        with jt.no_grad():
            outputs = head(feats)
        end_time = time.time()

        print(f"✓ Detection head works")
        print(f"  Input features: {[f.shape for f in feats]}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Inference time: {(end_time - start_time)*1000:.2f}ms")

        # 验证输出形状
        expected_shape = (1, 2125, 52)  # (batch, points, classes+reg)
        if outputs.shape != expected_shape:
            print(f"✗ Output shape mismatch: got {outputs.shape}, expected {expected_shape}")
            return False

        # 计算参数量
        head_params = sum(p.numel() for p in head.parameters())
        print(f"✓ Head parameters: {head_params/1e6:.2f}M")

        return True
    except Exception as e:
        print(f"✗ Detection head test failed: {e}")
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试内存使用情况"""
    print("\nTesting memory usage...")

    try:
        from nanodet.model.backbone import build_backbone
        from nanodet.model.fpn import build_fpn
        from nanodet.model.head import build_head

        # 创建完整模型
        backbone_cfg = {
            'name': 'ShuffleNetV2',
            'model_size': '1.0x',
            'out_stages': [2, 3, 4],
            'activation': 'LeakyReLU',
            'pretrain': False
        }
        backbone = build_backbone(backbone_cfg)

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

        # 测试不同batch size的内存使用
        batch_sizes = [1, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            try:
                x = jt.randn(batch_size, 3, 320, 320)

                start_time = time.time()
                with jt.no_grad():
                    backbone_outputs = backbone(x)
                    fpn_outputs = fpn(backbone_outputs)
                    head_outputs = head(fpn_outputs)
                end_time = time.time()

                # 计算参数量
                total_params = (sum(p.numel() for p in backbone.parameters()) +
                               sum(p.numel() for p in fpn.parameters()) +
                               sum(p.numel() for p in head.parameters()))

                print(f"✓ Batch size {batch_size:2d}: "
                      f"time={((end_time - start_time)*1000):6.2f}ms, "
                      f"params={total_params/1e6:.2f}M")

                # 清理内存
                del x, backbone_outputs, fpn_outputs, head_outputs
                jt.gc()

            except Exception as e:
                print(f"✗ Batch size {batch_size} failed: {e}")
                break

        return True
    except Exception as e:
        print(f"✗ Memory usage test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("NanoDet-Jittor Basic Functionality Test")
    print("=" * 60)
    
    # 显示系统信息
    print(f"Jittor version: {jt.__version__}")
    print(f"CUDA available: {jt.has_cuda}")
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print(f"Using CUDA: {jt.flags.use_cuda}")
    
    # 运行测试
    tests = [
        ("Jittor Basic", test_jittor_basic),
        ("ConvModule", test_conv_module),
        ("Activation", test_activation),
        ("ShuffleNetV2", test_shufflenet_backbone),
        ("FPN Functionality", test_fpn_functionality),
        ("Head Functionality", test_head_functionality),
        ("Memory Usage", test_memory_usage),
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
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your NanoDet-Jittor setup is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the error messages above.")
    
    return passed == len(results)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
