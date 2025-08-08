import itertools
import math
from typing import Any, Dict, Optional

import jittor as jt
from jittor import nn


class ExpMovingAverager(object):
    """Exponential Moving Averager (Jittor Version)."""
    def __init__(self, decay: float = 0.9998):
        if decay < 0 or decay > 1.0:
            raise ValueError(f"Decay should be in [0, 1], {decay} was given.")
        self.decay: float = decay
        self.state: Dict[str, Any] = {}

    def load_from(self, model: nn.Module) -> None:
        """从模型加载状态。"""
        self.state.clear()
        for name, val in self._get_model_state_iterator(model):
            # [迁移] .detach().clone() -> .clone()
            self.state[name] = val.clone()

    def has_inited(self) -> bool:
        return len(self.state) > 0

    def apply_to(self, model: nn.Module) -> None:
        """将 EMA 状态应用于模型。"""
        # [迁移] @jt.no_grad() -> @jt.no_grad()
        with jt.no_grad():
            for name, val in self._get_model_state_iterator(model):
                assert name in self.state, f"Name {name} not exist"
                # [迁移] val.copy_() -> val.assign()
                val.assign(self.state[name])

    def state_dict(self) -> Dict[str, Any]:
        return self.state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.state.clear()
        self.state.update(state_dict)

    def _get_model_state_iterator(self, model: nn.Module):
        # Jittor 的 named_parameters() 和 named_buffers() 与 PyTorch 兼容
        param_iter = model.named_parameters()
        buffer_iter = model.named_buffers()
        return itertools.chain(param_iter, buffer_iter)

    def calculate_dacay(self, iteration: int) -> float:
        # 此方法与框架无关
        decay = (self.decay) * math.exp(-(1 + iteration) / 2000) + (1 - self.decay)
        return decay

    def update(self, model: nn.Module, iteration: int) -> None:
        """使用模型的新权重更新 EMA 状态。"""
        
        # ---------- FIX START ----------
        # [修复] 增加延迟初始化逻辑，防止在状态未初始化时调用 update 导致 KeyError
        if not self.has_inited():
            self.load_from(model)
        # ---------- FIX END ----------

        decay = self.calculate_dacay(iteration)
        with jt.no_grad():
            for name, val in self._get_model_state_iterator(model):
                ema_val = self.state[name]
                # [迁移] ema_val.copy_() -> ema_val.assign()
                ema_val.assign(ema_val * (1 - decay) + val * decay)