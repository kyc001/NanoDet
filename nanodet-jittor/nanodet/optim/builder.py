import copy
import logging

import jittor as jt
from jittor import nn

# 定义所有归一化层的类型元组
NORMS = (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm)

def build_optimizer(model, config):
    """
    从配置构建优化器 (Jittor 版本)。
    此版本修复了 `TypeError` 并正确实现了参数分组。
    """
    config = copy.deepcopy(config)
    logger = logging.getLogger("NanoDet")

    # 获取基础超参数
    base_lr = config.get("lr", 0)
    base_wd = config.get("weight_decay", 0)

    # 获取参数分组规则
    no_norm_decay = config.pop("no_norm_decay", False)
    no_bias_decay = config.pop("no_bias_decay", False)
    param_level_cfg = config.pop("param_level_cfg", {})

    # 获取优化器类别
    name = config.pop("name")
    optim_cls = getattr(jt.optim, name)

    # 初始化参数组的字典
    # 使用集合来跟踪已分配的参数名，避免重复处理
    param_groups = {}
    assigned_params = set()

    # 1. 优先处理有特殊配置的参数 (param_level_cfg)
    for key, val in param_level_cfg.items():
        params_in_group = []
        for name, p in model.named_parameters():
            if not p.requires_grad or name in assigned_params:
                continue
            if key in name:
                params_in_group.append(p)
                assigned_params.add(name)
        
        if params_in_group:
            group_config = {
                "lr": base_lr * val.get("lr_mult", 1.0),
                "weight_decay": base_wd * val.get("decay_mult", 1.0),
            }
            logger.info(f"为参数组 '{key}' 设置特殊超参数: {group_config}")
            param_groups[f"special_{key}"] = {
                "params": params_in_group,
                **group_config
            }

    # 2. 处理需要特殊衰减规则的参数 (norm, bias)
    no_decay_params = []
    bias_params = []
    
    for m_name, m in model.named_modules():
        # 处理归一化层的参数
        if no_norm_decay and isinstance(m, NORMS):
            for p_name, p in m.named_parameters(recurse=False):
                full_p_name = f"{m_name}.{p_name}" if m_name else p_name
                if not p.requires_grad or full_p_name in assigned_params:
                    continue
                no_decay_params.append(p)
                assigned_params.add(full_p_name)
        # 处理偏置项
        elif no_bias_decay and hasattr(m, 'bias') and m.bias is not None:
            full_b_name = f"{m_name}.bias" if m_name else "bias"
            if full_b_name not in assigned_params:
                bias_params.append(m.bias)
                assigned_params.add(full_b_name)

    if no_decay_params:
        param_groups["no_norm_decay"] = {"params": no_decay_params, "weight_decay": 0.0}
        logger.info(f"为 {len(no_decay_params)} 个归一化层参数设置 weight_decay=0")
    
    if bias_params:
        param_groups["no_bias_decay"] = {"params": bias_params, "weight_decay": 0.0}
        logger.info(f"为 {len(bias_params)} 个偏置参数设置 weight_decay=0")

    # 3. 处理所有剩余的、使用默认配置的参数
    default_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and name not in assigned_params:
            default_params.append(p)
    
    if default_params:
        param_groups["default"] = {"params": default_params}

    # 将字典转换为优化器所需的列表格式
    final_param_groups = list(param_groups.values())
    
    optimizer = optim_cls(final_param_groups, **config)
    return optimizer
