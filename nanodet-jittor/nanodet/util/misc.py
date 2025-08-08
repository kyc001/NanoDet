from functools import partial

# JITTOR MIGRATION: 导入 jittor 库
import jittor as jt


def multi_apply(func, *args, **kwargs):
    """对一系列输入参数应用一个函数。

    此函数是框架无关的，因为它使用标准的 Python 功能。
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def images_to_levels(target, num_level_anchors):
    """将按图像组织的目标转换为按特征层级组织的目标。

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    # JITTOR MIGRATION: jt.stack -> jt.stack
    target = jt.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def unmap(data, count, inds, fill=0):
    """将一个项目的子集（数据）反向映射回原始项目集（大小为 count）。"""
    if data.ndim == 1:
        # JITTOR MIGRATION: data.new_full -> jt.full, 并指定 dtype
        ret = jt.full((count,), fill, dtype=data.dtype)
        # JITTOR MIGRATION: inds.type(jt.bool) -> inds.bool()
        ret[inds.bool()] = data
    else:
        new_size = (count,) + data.shape[1:]
        # JITTOR MIGRATION: data.new_full -> jt.full, 并指定 dtype
        ret = jt.full(new_size, fill, dtype=data.dtype)
        # JITTOR MIGRATION: inds.type(jt.bool) -> inds.bool()
        ret[inds.bool(), :] = data
    return ret
