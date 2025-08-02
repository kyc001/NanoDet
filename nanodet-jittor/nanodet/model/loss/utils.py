import functools
import jittor as jt
from jittor import nn

def reduce_loss(loss, reduction):
    """
    按照指定方式 reduce 損失 (Jittor 版本)。
    """
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"Unsupported reduction type: {reduction}")


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """
    應用逐元素的權重並 reduce 損失 (Jittor 版本)。
    """
    # 如果指定了權重，則應用逐元素的權重
    if weight is not None:
        loss = loss * weight

    # 如果未指定 avg_factor，則直接 reduce 損失
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # 如果 reduction 是 'mean'，則用 avg_factor 對損失進行平均
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # 如果 reduction 不是 'none'，則拋出錯誤
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """
    為給定的損失函數創建一個帶權重的版本 (Jittor 版本)。
    """
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        # 獲取逐元素的損失
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
