import copy

from .fpn import FPN
from .ghost_pan import GhostPAN
from .pan import PAN
from .tan import TAN


def build_fpn(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop("name")
    if name == "FPN":
        return FPN(**fpn_cfg)
    elif name == "PAN":
        return PAN(**fpn_cfg)
    elif name == "TAN":
        return TAN(**fpn_cfg)
    elif name == "GhostPAN":
        return GhostPAN(**fpn_cfg)
    else:
        raise NotImplementedError
