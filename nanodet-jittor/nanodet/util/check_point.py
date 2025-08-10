# JITTOR MIGRATION: 导入 jittor 和其他所需库
import jittor as jt
from collections import OrderedDict
from typing import Any, Dict
from .rank_filter import rank_filter

# 可选导入 PyTorch（用于直接加载 PT 权重）
try:
    import torch
except Exception:
    torch = None


def _strip_prefix_key(k: str) -> str:
    if k.startswith('module.'):
        k = k[7:]
    if k.startswith('model.'):
        k = k[6:]
    return k


def pt_to_jt_checkpoint(pt_ckpt, model):
    """将 PyTorch ckpt/state_dict 转为 Jittor 检查点（含合并头->分离头切分）。"""
    # 允许 pt_ckpt 是路径
    if torch is not None and isinstance(pt_ckpt, (str, bytes)):
        pt_ckpt = torch.load(pt_ckpt, map_location='cpu')
    # state_dict 抽取，并优先 avg_model.*
    state_dict = pt_ckpt.get('state_dict', pt_ckpt)
    if any(k.startswith('avg_model.') for k in state_dict.keys()):
        state_dict = {k[len('avg_model.'):]: v for k, v in state_dict.items() if k.startswith('avg_model.')}
    # 去前缀+转 numpy
    proc = {}
    for k, v in state_dict.items():
        k = _strip_prefix_key(k)
        try:
            import numpy as np
            import torch as _t
            v_np = v.detach().cpu().numpy() if isinstance(v, _t.Tensor) else np.array(v)
        except Exception:
            v_np = v
        proc[k] = v_np

    # 关键别名映射（对齐 JT 命名）：
    # - PT 里可能使用 head.gfl_reg_convs.* / head.gfl_cls_convs.*
    # - JT 使用 head.reg_convs.* / head.cls_convs.*
    proc_alias = {}
    for k, v_np in proc.items():
        if k.startswith('head.gfl_reg_convs.'):
            nk = k.replace('head.gfl_reg_convs.', 'head.reg_convs.', 1)
            proc_alias[nk] = v_np
            continue
        if k.startswith('head.gfl_cls_convs.'):
            nk = k.replace('head.gfl_cls_convs.', 'head.cls_convs.', 1)
            proc_alias[nk] = v_np
            continue
        proc_alias[k] = v_np
    proc = proc_alias

    # 形状对齐与合并头切分（增强版，含 depthwise/group conv 自适应）
    model_sd = model.state_dict()
    reconciled = {}
    for k, v_np in proc.items():
        # 合并头 -> 分离头映射（逐层 i=0..N-1，全覆盖）
        if k.startswith('head.gfl_cls.') and (k.endswith('.weight') or k.endswith('.bias')):
            try:
                parts = k.split('.')
                i = int(parts[2])
                suffix = parts[-1]  # weight or bias
                cls_key = f'head.gfl_cls.{i}.{suffix}'
                reg_key = f'head.gfl_reg.{i}.{suffix}'
                if cls_key in model_sd and reg_key in model_sd:
                    cls_out = model_sd[cls_key].shape[0]
                    reg_out = model_sd[reg_key].shape[0]
                    total_out = cls_out + reg_out
                    if getattr(v_np, 'shape', None) and v_np.shape[0] >= total_out:
                        import numpy as np
                        reconciled[cls_key] = v_np[:cls_out].astype(np.float32)
                        reconciled[reg_key] = v_np[cls_out:cls_out+reg_out].astype(np.float32)
                        continue
            except Exception:
                pass
        # 普通一一对齐
        if k in model_sd and getattr(v_np, 'shape', None) == model_sd[k].shape:
            import numpy as np
            reconciled[k] = v_np.astype(np.float32)
            continue
        # 4D conv 权重适配（特别是 depthwise/grouped 差异）
        if k in model_sd:
            tgt = model_sd[k]
            tshape = tuple(tgt.shape)
            vshape = getattr(v_np, 'shape', None)
            if vshape and len(vshape) == 4 and len(tshape) == 4 and v_np.shape[0] == tshape[0]:
                import numpy as np
                # 情况A：PT 是 depthwise (in_ch=1)，JT 需要 grouped/标准 conv -> 对角复制或通道平铺
                if v_np.shape[1] == 1:
                    # 目标是 depthwise grouped: (C_out, C_out, k, k)
                    if tshape[1] == tshape[0]:
                        cout = tshape[0]
                        neww = np.zeros(tshape, dtype=np.float32)
                        for c in range(cout):
                            neww[c, c, :, :] = v_np[c, 0, :, :]
                        reconciled[k] = neww
                        continue
                    # 目标是一般 conv: (C_out, C_in, k, k) 且 C_in>1 -> 平铺截断
                    elif tshape[1] > 1:
                        reps = [1, tshape[1], 1, 1]
                        v_np_t = np.tile(v_np, reps)[:, :tshape[1], :, :]
                        reconciled[k] = v_np_t.astype(np.float32)
                        continue
    return {'state_dict': reconciled}



def load_model_weight(model, checkpoint, logger):
    """支持三种输入：
    - JT 检查点: {'state_dict': ...}
    - 直接的 state_dict: {k: v}
    - PT 路径/检查点: 传入 dict 且带有 'pytorch-lightning_version' 或一个 .ckpt/.pth 文件路径
    """
    # 如果是 PT 风格对象，先转为 JT 检查点
    if isinstance(checkpoint, str) and (checkpoint.endswith('.ckpt') or checkpoint.endswith('.pth')):
        if torch is None:
            raise RuntimeError('需要安装 PyTorch 才能从 .ckpt/.pth 加载权重')
        pt_ckpt = torch.load(checkpoint, map_location='cpu')
        checkpoint = pt_to_jt_checkpoint(pt_ckpt, model)
    elif isinstance(checkpoint, dict) and 'pytorch-lightning_version' in checkpoint:
        if torch is None:
            raise RuntimeError('需要安装 PyTorch 才能从 PyTorch Lightning ckpt 加载权重')
        checkpoint = pt_to_jt_checkpoint(checkpoint, model)

    """
    加载模型权重，并做头部兼容性映射：
    - 兼容旧版合并头(gfl_cls输出= num_classes + 4*(reg_max+1)) -> 新版分离头(gfl_cls & gfl_reg)
    - 若无 reg_convs.* 权重，则从 cls_convs.* 对应拷贝一份以加速收敛
    """
    log = (logger.info if logger is not None else (lambda *a, **k: None))

    # Jittor 的 checkpoint 通常也是字典，直接操作
    state_dict = checkpoint["state_dict"].copy()

    # 转换平均模型权重 (EMA)
    for k in list(state_dict.keys()):  # 使用 list(keys()) 来安全地在循环中修改字典
        if k.startswith("avg_model."):
            v = state_dict.pop(k)
            state_dict[k[10:]] = v  # 移除 'avg_model.' 前缀

    # 移除分布式训练时可能添加的前缀
    def _strip_prefix(sd):
        if not sd:
            return sd
        keys = list(sd.keys())
        if keys and keys[0].startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
            keys = list(sd.keys())
        if keys and keys[0].startswith("model."):
            sd = {k[6:]: v for k, v in sd.items()}
        return sd

    state_dict = _strip_prefix(state_dict)

    model_state_dict = model.state_dict()

    # 预审计：统计原始命中情况（未做任何映射前）
    def _audit(prefix_list):
        hit = {p: 0 for p in prefix_list}
        tot = {p: 0 for p in prefix_list}
        for k, v in state_dict.items():
            for p in prefix_list:
                if k.startswith(p):
                    tot[p] += 1
                    if k in model_state_dict and getattr(v, 'shape', None) == model_state_dict[k].shape:
                        hit[p] += 1
                    break
        return hit, tot
    prefixes = ['backbone.', 'fpn.', 'head.']
    pre_hit, pre_tot = _audit(prefixes)
    log(f"[Audit:pre] matches/total -> " + ", ".join([f"{p}{pre_hit[p]}/{pre_tot[p]}" for p in prefixes]))

    # 1) 合并头 -> 分离头 映射：切分 head.gfl_cls.{i}.{weight,bias} 的 [cls|reg]
    import re
    for i in range(0, 8):  # 最多支持到 P10，实际以存在键为准
        w_key = f"head.gfl_cls.{i}.weight"
        b_key = f"head.gfl_cls.{i}.bias"
        # weight 拆分
        if w_key in state_dict and f"head.gfl_reg.{i}.weight" not in state_dict:
            w = state_dict[w_key]
            if w_key in model_state_dict and f"head.gfl_reg.{i}.weight" in model_state_dict:
                cls_out = model_state_dict[w_key].shape[0]
                reg_out = model_state_dict[f"head.gfl_reg.{i}.weight"].shape[0]
                total_out = cls_out + reg_out
                if hasattr(w, 'shape') and w.shape[0] >= total_out:
                    state_dict[w_key] = w[:cls_out]
                    state_dict[f"head.gfl_reg.{i}.weight"] = w[cls_out:cls_out+reg_out]
                    log(f"Mapped {w_key} -> split to gfl_cls/gfl_reg")
        # bias 拆分
        if b_key in state_dict and f"head.gfl_reg.{i}.bias" not in state_dict:
            b = state_dict[b_key]
            if b_key in model_state_dict and f"head.gfl_reg.{i}.bias" in model_state_dict:
                cls_out = model_state_dict[b_key].shape[0]
                reg_out = model_state_dict[f"head.gfl_reg.{i}.bias"].shape[0]
                total_out = cls_out + reg_out
                if hasattr(b, 'shape') and b.shape[0] >= total_out:
                    state_dict[b_key] = b[:cls_out]
                    state_dict[f"head.gfl_reg.{i}.bias"] = b[cls_out:cls_out+reg_out]
                    log(f"Mapped {b_key} -> split to gfl_cls/gfl_reg")

    # 2) 若 reg_convs.* 缺失：
    #    情况A：模型配置允许共享塔（share_cls_reg_tower=True）时，直接复用 cls_convs 的引用
    #    情况B：否则精确拷贝对应权重，保持与 cls_convs 相同的初始化分布
    share_tower = getattr(getattr(getattr(model, 'head', None), 'share_cls_reg_tower', False), 'item', lambda: False)()
    for k in list(model_state_dict.keys()):
        if k.startswith("head.reg_convs.") and k not in state_dict:
            mirror_k = k.replace("head.reg_convs.", "head.cls_convs.")
            if mirror_k in state_dict and state_dict.get(mirror_k, None) is not None:
                state_dict[k] = state_dict[mirror_k]
                log(f"Filled {k} from {mirror_k}")

    # 3) 过滤/对齐形状并加载
    for k in list(state_dict.keys()):
        if k in model_state_dict:
            if getattr(state_dict[k], 'shape', None) != model_state_dict[k].shape:
                log(
                    f"Skip loading parameter {k}, required shape{model_state_dict[k].shape}, "
                    f"loaded shape{getattr(state_dict[k], 'shape', None)}."
                )
                state_dict[k] = model_state_dict[k]
        else:
            log(f"Drop parameter {k}.")
            state_dict.pop(k)

    for k in model_state_dict:
        if k not in state_dict:
            log(f"No param {k} in checkpoint.")
            state_dict[k] = model_state_dict[k]

    # 加载前再次审计
    post_hit, post_tot = _audit(prefixes)
    log(f"[Audit:post] matches/total -> " + ", ".join([f"{p}{post_hit[p]}/{post_tot[p]}" for p in prefixes]))

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
