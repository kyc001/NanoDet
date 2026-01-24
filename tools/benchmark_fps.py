#!/usr/bin/env python3
"""
推理性能FPS测试脚本
测试 PyTorch 和 Jittor 的推理速度
"""

import argparse
import time
import os
import sys
import numpy as np

def benchmark_pytorch(config_path, model_path, num_images=100, warmup=10):
    """测试PyTorch推理FPS"""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanodet-pytorch'))
    from nanodet.util import cfg, load_config, load_model_weight
    from nanodet.model.arch import build_model
    
    load_config(cfg, config_path)
    model = build_model(cfg.model)
    
    ckpt = torch.load(model_path, map_location='cuda')
    if 'state_dict' in ckpt:
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        if not state_dict:
            state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model = model.cuda().eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 320, 320).cuda()
    
    # Warmup
    print(f"PyTorch Warmup ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"PyTorch Benchmark ({num_images} iterations)...")
    times = []
    with torch.no_grad():
        for _ in range(num_images):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    avg_time = times.mean() * 1000  # ms
    std_time = times.std() * 1000
    fps = 1000 / avg_time
    
    return {
        'framework': 'PyTorch',
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'fps': fps,
        'num_images': num_images
    }

def benchmark_jittor(config_path, model_path, num_images=100, warmup=10):
    """测试Jittor推理FPS"""
    import jittor as jt
    jt.flags.use_cuda = 1
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanodet-jittor'))
    from nanodet.util import cfg, load_config, load_model_weight
    from nanodet.model.arch import build_model
    
    load_config(cfg, config_path)
    model = build_model(cfg.model)
    
    ckpt = jt.load(model_path)
    if 'state_dict' not in ckpt:
        ckpt = {'state_dict': ckpt}
    load_model_weight(model, ckpt, None)
    model.eval()
    
    # 创建虚拟输入
    dummy_input = jt.randn(1, 3, 320, 320)
    
    # Warmup
    print(f"Jittor Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model(dummy_input)
        jt.sync_all()
    
    # Benchmark
    print(f"Jittor Benchmark ({num_images} iterations)...")
    times = []
    for _ in range(num_images):
        jt.sync_all()
        start = time.perf_counter()
        _ = model(dummy_input)
        jt.sync_all()
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    avg_time = times.mean() * 1000  # ms
    std_time = times.std() * 1000
    fps = 1000 / avg_time
    
    return {
        'framework': 'Jittor',
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'fps': fps,
        'num_images': num_images
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, choices=['pytorch', 'jittor'], required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    
    if args.framework == 'pytorch':
        result = benchmark_pytorch(args.config, args.model, args.num_images, args.warmup)
    else:
        result = benchmark_jittor(args.config, args.model, args.num_images, args.warmup)
    
    print("\n" + "="*50)
    print(f"Framework: {result['framework']}")
    print(f"Avg inference time: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
    print(f"FPS: {result['fps']:.1f}")
    print(f"Test images: {result['num_images']}")
    print("="*50)
