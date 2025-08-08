# JITTOR MIGRATION: 导入 jittor 和其他所需库
import jittor as jt
from collections import OrderedDict
from typing import Any, Dict
from .rank_filter import rank_filter


def load_model_weight(model, checkpoint, logger):
    """
    加载模型权重。此函数已从 PyTorch 迁移到 Jittor。
    """
    # JITTOR MIGRATION: Jittor 的 checkpoint 通常也是字典，直接操作
    state_dict = checkpoint["state_dict"].copy()
    
    # 转换平均模型权重 (EMA)
    for k in list(state_dict.keys()): # 使用 list(keys()) 来安全地在循环中修改字典
        if k.startswith("avg_model."):
            v = state_dict.pop(k)
            state_dict[k[10:]] = v # 移除 'avg_model.' 前缀

    # 移除分布式训练时可能添加的前缀
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in state_dict.items()}

    model_state_dict = model.state_dict()

    # 检查加载的参数和模型参数的匹配情况
    for k in list(state_dict.keys()):
        if k in model_state_dict:
            # JITTOR MIGRATION: shape 比较逻辑在 Jittor 中保持不变
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info(
                    f"Skip loading parameter {k}, required shape{model_state_dict[k].shape}, "
                    f"loaded shape{state_dict[k].shape}."
                )
                state_dict[k] = model_state_dict[k]
        else:
            logger.info(f"Drop parameter {k}.")
            state_dict.pop(k) # 移除模型中不存在的参数

    for k in model_state_dict:
        if k not in state_dict:
            logger.info(f"No param {k} in checkpoint.")
            state_dict[k] = model_state_dict[k]
            
    # JITTOR MIGRATION: 使用 model.load_state_dict，Jittor 会自动处理不严格匹配
    model.load_state_dict(state_dict)


@rank_filter
def save_model(model, path, epoch, iter, optimizer=None):
    """
    保存模型。此函数已从 PyTorch 迁移到 Jittor。
    只在 rank 0 进程上执行。
    """
    model_state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": model_state_dict, "iter": iter}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()

    # JITTOR MIGRATION: 使用 jt.save 替换 jt.save
    jt.save(data, path)


def convert_old_model(old_model_dict):
    """
    转换旧格式的模型检查点。
    JITTOR MIGRATION: 此函数移除了 PyTorch Lightning 特定的逻辑，
    并将其转换为一个通用的 Jittor 检查点格式。
    """
    if "pytorch-lightning_version" in old_model_dict:
        raise ValueError("This model is not old format. No need to convert!")
        
    epoch = old_model_dict["epoch"]
    global_step = old_model_dict["iter"]
    state_dict = old_model_dict["state_dict"]
    
    # Jittor 模型不需要 'model.' 前缀，直接使用原始 state_dict
    new_state_dict = state_dict

    new_checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": new_state_dict,
    }

    if "optimizer" in old_model_dict:
        new_checkpoint["optimizer_states"] = [old_model_dict["optimizer"]]

    return new_checkpoint


def convert_avg_params(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    从检查点中提取平均（EMA）模型权重。
    此函数是框架无关的，因为它只操作字典，所以无需修改。
    Args:
        checkpoint: 模型检查点字典。
    Returns:
        转换后的平均模型状态字典。
    """
    state_dict = checkpoint["state_dict"]
    avg_weights = {}
    for k, v in state_dict.items():
        if "avg_model" in k:
            avg_weights[k[10:]] = v # 移除 'avg_model.' 前缀
    return avg_weights
