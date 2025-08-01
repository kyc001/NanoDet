#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Scale参数的创建方式
"""

import jittor as jt
from jittor import nn

# 测试不同的标量创建方式
print("测试Jittor标量参数创建方式:")

# 方式1: jt.array(scale)
scale1 = jt.array(1.0)
print(f"jt.array(1.0): shape={scale1.shape}, value={scale1}")

# 方式2: jt.float32(scale)
scale2 = jt.float32(1.0)
print(f"jt.float32(1.0): shape={scale2.shape}, value={scale2}")

# 方式3: jt.Var([scale])
scale3 = jt.Var([1.0])
print(f"jt.Var([1.0]): shape={scale3.shape}, value={scale3}")

# 方式4: 直接使用float
scale4 = 1.0
print(f"float(1.0): type={type(scale4)}, value={scale4}")

# 测试哪种方式创建的参数形状是[]
class TestScale1(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = jt.array(1.0)

class TestScale2(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = jt.float32(1.0)

class TestScale3(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = jt.Var(1.0)

class TestScale4(nn.Module):
    def __init__(self):
        super().__init__()
        # 尝试创建真正的标量
        self.scale = jt.array(1.0).squeeze()

class TestScale5(nn.Module):
    def __init__(self):
        super().__init__()
        # 尝试使用0维张量
        import numpy as np
        self.scale = jt.array(np.array(1.0))

print("\n测试模型参数形状:")
model1 = TestScale1()
model2 = TestScale2()
model3 = TestScale3()
model4 = TestScale4()
model5 = TestScale5()

for name, param in model1.named_parameters():
    print(f"TestScale1.{name}: shape={param.shape}")

for name, param in model2.named_parameters():
    print(f"TestScale2.{name}: shape={param.shape}")

for name, param in model3.named_parameters():
    print(f"TestScale3.{name}: shape={param.shape}")

for name, param in model4.named_parameters():
    print(f"TestScale4.{name}: shape={param.shape}")

for name, param in model5.named_parameters():
    print(f"TestScale5.{name}: shape={param.shape}")
