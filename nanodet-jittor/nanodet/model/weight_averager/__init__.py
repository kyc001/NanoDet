import copy

from .ema import ExpMovingAverager


def build_weight_averager(cfg):
    cfg = copy.deepcopy(cfg)
    name = cfg.pop("name")
    if name == "ExpMovingAverager":
        return ExpMovingAverager(**cfg)
    else:
        raise NotImplementedError(f"{name} is not implemented")
