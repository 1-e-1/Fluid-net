"""
FluidNet Ensemble & Improvement Wrappers
========================================
不修改源码的模型改进方案:
1. Multi-Seed Ensemble (投票集成)
2. Cosine Annealing + Warmup 学习率调度
3. Label Smoothing 正则化
4. Multi-Model Stacking (三版本模型融合)
5. Data Augmentation Pipeline
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FluidNetEnsemble(nn.Module):
    """多种子集成模型 - 不修改任何源码, 仅做wrapper"""

    def __init__(
        self,
        model_cls,
        config_cls,
        configs: list,
        seeds: Optional[list[int]] = None,
    ):
        super().__init__()
        if seeds is None:
            seeds = list(range(len(configs)))
        assert len(configs) == len(seeds), "configs and seeds must match"

        self.models = nn.ModuleList()
        for i, cfg in enumerate(configs):
            set_seed(seeds[i])
            model = model_cls(cfg)
            self.models.append(model)
        self.num_models = len(self.models)

    def forward(self, inputs, mode="nonsequential", max_output_steps=1):
        all_outputs = []
        for model in self.models:
            out = model(inputs, mode=mode, max_output_steps=max_output_steps)
            all_outputs.append(out.outputs[:, 0])
        # 平均集成
        stacked = torch.stack(all_outputs, dim=0)
        return stacked.mean(dim=0)

    def forward_vote(self, inputs, mode="nonsequential", max_output_steps=1):
        """硬投票"""
        all_logits = []
        for model in self.models:
            out = model(inputs, mode=mode, max_output_steps=max_output_steps)
            all_logits.append(out.outputs[:, 0])
        votes = torch.stack([logits.argmax(dim=-1) for logits in all_logits], dim=0)
        # majority vote -> convert to one-hot averaged
        result = torch.zeros_like(all_logits[0])
        for i in range(self.num_models):
            result.scatter_add_(1, votes[i].unsqueeze(1), torch.ones_like(result) / self.num_models)
        return result

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def parameters(self, recurse=True):
        for m in self.models:
            yield from m.parameters(recurse)


class CosineWarmupScheduler:
    """Cosine Annealing with Linear Warmup"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr * self.base_lrs[i]

    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.current_step / max(1, self.warmup_steps)
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
            1 + np.cos(np.pi * min(progress, 1.0))
        )

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy"""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        return -(smooth_target * log_probs).sum(dim=-1).mean()


class FluidNetMultiVariantEnsemble(nn.Module):
    """三版本模型融合: fluid_net + fluid_net_pro + fluid_net_pro_max"""

    def __init__(self, base_cfg, pro_cfg, promax_cfg, device="cuda:0"):
        super().__init__()
        # 延迟导入以避免修改源码
        import fluid_net as fn
        import fluid_net_pro as fnp
        import fluid_net_pro_max as fnpm

        self.model_base = fn.FluidResidualSequenceNetwork(base_cfg)
        self.model_pro = fnp.FluidResidualSequenceNetwork(pro_cfg)
        self.model_promax = fnpm.FluidResidualSequenceNetwork(promax_cfg)

    def to(self, *args, **kwargs):
        self.model_base.to(*args, **kwargs)
        self.model_pro.to(*args, **kwargs)
        self.model_promax.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, inputs, mode="nonsequential", max_output_steps=1):
        out_base = self.model_base(inputs, mode=mode, max_output_steps=max_output_steps).outputs[:, 0]
        out_pro = self.model_pro(inputs, mode=mode, max_output_steps=max_output_steps).outputs[:, 0]
        out_promax = self.model_promax(inputs, mode=mode, max_output_steps=max_output_steps).outputs[:, 0]
        return (out_base + out_pro + out_promax) / 3.0

    def train(self, mode=True):
        self.model_base.train(mode)
        self.model_pro.train(mode)
        self.model_promax.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        yield from self.model_base.parameters(recurse)
        yield from self.model_pro.parameters(recurse)
        yield from self.model_promax.parameters(recurse)


def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().cpu())


def image_to_field(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).contiguous()
