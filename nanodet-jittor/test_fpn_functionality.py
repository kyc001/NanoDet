#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FPN功能测试脚本
测试GhostPAN FPN网络是否正常工作
"""

import jittor as jt
import time
import traceback


def test_ghost_module():
    """测试GhostModule"""
    print("Testing GhostModule...")
    
    try:
        from nanodet.model.backbone.ghostnet import GhostModule
        
        # 创建GhostModule
        ghost = GhostModule(
            inp=64,
            oup=128,
            kernel_size=1,
            ratio=2,
            activation='ReLU'
        )
        
        x = jt.randn(1, 64, 32, 32)
        y = ghost(x)
        
        print(f"✓ GhostModule works")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
        # 验证输出通道数
        if y.shape[1] != 128:
            print(f"✗ Output channels mismatch: got {y.shape[1]}, expected 128")
            return False
        
        return True
    except Exception as e:
        print(f"✗ GhostModule test failed: {e}")
        traceback.print_exc()
        return False


def test_ghost_bottleneck():
    """测试GhostBottleneck"""
    print("\nTesting GhostBottleneck...")
    
    try:
        from nanodet.model.backbone.ghostnet import GhostBottleneck
        
        # 测试不同配置的GhostBottleneck
        configs = [
            # (in_chs, mid_chs, out_chs, stride, se_ratio)
            (64, 128, 64, 1, 0.0),    # 残差连接
            (64, 128, 128, 2, 0.25),  # 下采样 + SE
            (128, 256, 128, 1, 0.0),  # 标准配置
        ]
        
        for i, (in_chs, mid_chs, out_chs, stride, se_ratio) in enumerate(configs):
            bottleneck = GhostBottleneck(
                in_chs=in_chs,
                mid_chs=mid_chs,
                out_chs=out_chs,
                stride=stride,
                se_ratio=se_ratio,
                activation='ReLU'
            )
            
            input_size = 32 // stride
            x = jt.randn(1, in_chs, input_size, input_size)
            y = bottleneck(x)
            
            expected_size = input_size // stride
            
            print(f"✓ GhostBottleneck config {i+1} works")
            print(f"  Input: {x.shape}")
            print(f"  Output: {y.shape}")
            print(f"  Stride: {stride}, SE: {se_ratio > 0}")
            
            # 验证输出形状
            if y.shape != (1, out_chs, expected_size, expected_size):
                print(f"✗ Output shape mismatch: got {y.shape}, expected {(1, out_chs, expected_size, expected_size)}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ GhostBottleneck test failed: {e}")
        traceback.print_exc()
        return False


def test_ghost_blocks():
    """测试GhostBlocks"""
    print("\nTesting GhostBlocks...")
    
    try:
        from nanodet.model.fpn.ghost_pan import GhostBlocks
        
        # 测试不同配置
        configs = [
            # (in_channels, out_channels, num_blocks, use_res)
            (128, 128, 1, False),
            (128, 96, 2, True),
            (256, 128, 3, False),
        ]
        
        for i, (in_channels, out_channels, num_blocks, use_res) in enumerate(configs):
            blocks = GhostBlocks(
                in_channels=in_channels,
                out_channels=out_channels,
                expand=1,
                kernel_size=5,
                num_blocks=num_blocks,
                use_res=use_res,
                activation='LeakyReLU'
            )
            
            x = jt.randn(1, in_channels, 32, 32)
            y = blocks(x)
            
            print(f"✓ GhostBlocks config {i+1} works")
            print(f"  Input: {x.shape}")
            print(f"  Output: {y.shape}")
            print(f"  Blocks: {num_blocks}, Residual: {use_res}")
            
            # 验证输出通道数
            if y.shape[1] != out_channels:
                print(f"✗ Output channels mismatch: got {y.shape[1]}, expected {out_channels}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ GhostBlocks test failed: {e}")
        traceback.print_exc()
        return False


def test_ghost_pan():
    """测试GhostPAN"""
    print("\nTesting GhostPAN...")
    
    try:
        from nanodet.model.fpn import build_fpn
        
        # 创建GhostPAN配置
        cfg = {
            'name': 'GhostPAN',
            'in_channels': [116, 232, 464],  # ShuffleNetV2 1.0x输出通道
            'out_channels': 96,
            'kernel_size': 5,
            'num_extra_level': 1,
            'use_depthwise': True,
            'activation': 'LeakyReLU'
        }
        
        fpn = build_fpn(cfg)
        print(f"✓ GhostPAN created")
        
        # 创建输入特征
        inputs = [
            jt.randn(1, 116, 40, 40),  # stage2: 320/8 = 40
            jt.randn(1, 232, 20, 20),  # stage3: 320/16 = 20  
            jt.randn(1, 464, 10, 10),  # stage4: 320/32 = 10
        ]
        
        print(f"Input shapes: {[x.shape for x in inputs]}")
        
        # 前向传播
        start_time = time.time()
        with jt.no_grad():
            outputs = fpn(inputs)
        end_time = time.time()
        
        print(f"✓ Forward pass successful")
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        print(f"  Inference time: {(end_time - start_time)*1000:.2f}ms")
        
        # 验证输出
        expected_shapes = [
            (1, 96, 40, 40),  # P3: stride 8
            (1, 96, 20, 20),  # P4: stride 16
            (1, 96, 10, 10),  # P5: stride 32
            (1, 96, 5, 5),    # P6: stride 64 (extra level)
        ]
        
        if len(outputs) != len(expected_shapes):
            print(f"✗ Output count mismatch: got {len(outputs)}, expected {len(expected_shapes)}")
            return False
        
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            if output.shape != expected_shape:
                print(f"✗ Output {i} shape mismatch: got {output.shape}, expected {expected_shape}")
                return False
        
        print(f"✓ All output shapes match expected values")
        
        # 计算参数量
        total_params = sum(p.numel() for p in fpn.parameters())
        print(f"✓ Total parameters: {total_params/1e6:.2f}M")
        
        return True
    except Exception as e:
        print(f"✗ GhostPAN test failed: {e}")
        traceback.print_exc()
        return False


def test_backbone_fpn_integration():
    """测试Backbone + FPN集成"""
    print("\nTesting Backbone + FPN integration...")
    
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
        
        # 计算总参数量
        backbone_params = sum(p.numel() for p in backbone.parameters())
        fpn_params = sum(p.numel() for p in fpn.parameters())
        total_params = backbone_params + fpn_params
        
        print(f"✓ Parameter count:")
        print(f"  Backbone: {backbone_params/1e6:.2f}M")
        print(f"  FPN: {fpn_params/1e6:.2f}M")
        print(f"  Total: {total_params/1e6:.2f}M")
        
        return True
    except Exception as e:
        print(f"✗ Backbone + FPN integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("GhostPAN FPN Functionality Test")
    print("=" * 60)
    
    # 显示系统信息
    print(f"Jittor version: {jt.__version__}")
    print(f"CUDA available: {jt.has_cuda}")
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print(f"Using CUDA: {jt.flags.use_cuda}")
    
    # 运行测试
    tests = [
        ("GhostModule", test_ghost_module),
        ("GhostBottleneck", test_ghost_bottleneck),
        ("GhostBlocks", test_ghost_blocks),
        ("GhostPAN", test_ghost_pan),
        ("Backbone+FPN Integration", test_backbone_fpn_integration),
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
        print("\n🎉 All FPN tests passed! GhostPAN is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the error messages above.")
    
    return passed == len(results)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
