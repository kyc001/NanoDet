#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Integral类的参数
确保project不被计入named_parameters()
"""

import sys
sys.path.append('/home/kyc/project/nanodet/nanodet-jittor')

from nanodet.model.head.gfl_head import Integral


def test_integral_parameters():
    """测试Integral类的参数"""
    print("🔍 测试Integral类的参数")
    
    # 创建Integral实例
    integral = Integral(reg_max=7)
    
    print(f"Integral实例创建成功")
    
    # 检查所有属性
    print(f"\nIntegral所有属性:")
    for name in dir(integral):
        if not name.startswith('__'):
            attr = getattr(integral, name)
            if hasattr(attr, 'shape'):
                print(f"  {name}: {attr.shape} - {type(attr)}")
    
    # 检查named_parameters
    print(f"\nIntegral named_parameters():")
    params = list(integral.named_parameters())
    for name, param in params:
        print(f"  {name}: {param.shape}")
    
    print(f"\n参数总数: {len(params)}")
    
    # 检查project属性
    if hasattr(integral, 'project'):
        project = integral.project
        print(f"\nproject属性:")
        print(f"  类型: {type(project)}")
        print(f"  形状: {project.shape}")
        print(f"  值: {project}")
    
    # 检查_project_data属性
    if hasattr(integral, '_project_data'):
        project_data = integral._project_data
        print(f"\n_project_data属性:")
        print(f"  类型: {type(project_data)}")
        print(f"  值: {project_data}")
    
    return len(params) == 0


def main():
    """主函数"""
    print("🚀 开始测试Integral参数")
    
    success = test_integral_parameters()
    
    if success:
        print("\n✅ Integral参数测试成功！project不被计入参数")
    else:
        print("\n❌ Integral参数测试失败！project仍被计入参数")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
