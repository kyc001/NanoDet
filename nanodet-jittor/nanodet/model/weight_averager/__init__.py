import copy

from .ema import ExpMovingAverager


def build_weight_averager(cfg, device="cpu"):
    cfg = copy.deepcopy(cfg)
    name = cfg.pop("name")
    if name == "ExpMovingAverager":
        return ExpMovingAverager(**cfg, device=device)
    else:
        raise NotImplementedError(f"{name} is not implemented")
