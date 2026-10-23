# -*- coding: utf-8 -*-
"""
experiments_v2/models/deep_fluid.py
深层 FluidNet 残差块 - 利用 final_field 做残差连接
"""
from __future__ import annotations

import math, random, sys, time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

import fluid_net as fn

# ============ Utils ============
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def img2field(x): return x.permute(0, 2, 3, 1).contiguous()

def make_subset(ds, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()
    return Subset(ds, idx)

def acc_fn(logits, y):
    return float((logits.argmax(dim=-1) == y).float().mean().cpu())

# ============ FourierGlobalMix ============
class FourierGlobalMix(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n, n // 2 + 1, h, 2) * 0.02)

    def forward(self, x):
        xf = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        xf = torch.view_as_real(xf)
        xf = xf * (1.0 + self.weight)
        xf = torch.view_as_complex(xf)
        return torch.fft.irfft2(xf, dim=(1, 2), norm="ortho", s=(x.shape[1], x.shape[2])).to(x.dtype)

# ============ FieldFluidBlock ============
class FieldFluidBlock(nn.Module):
    """用 FluidNet 的 final_field 做残差, 保留完整 Q 压缩-重建循环"""
    def __init__(self, n, h, static_adr_steps=3, adr_hidden=32):
        super().__init__()
        self.fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=h,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                adr_after_subtraction=True,
                q_channels=16, q_depth=2, q_height=2, q_width=1,
                extractor_hidden_channels=adr_hidden,
                reconstructor_hidden_channels=adr_hidden,
                output_hidden_dim=adr_hidden * 2,
                adr_hidden_channels=adr_hidden,
            ))
        self.norm = nn.LayerNorm([n, n, h])

    def forward(self, x):
        out = self.fluid(self.norm(x), mode="nonsequential", max_output_steps=1)
        return x + out.final_field.squeeze(1)

# ============ DeepFluidFormer ============
class DeepFluidFormer(nn.Module):
    def __init__(self, n=28, h=8, num_blocks=3, num_classes=10, static_adr_steps=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "fluid": FieldFluidBlock(n, h, static_adr_steps),
                "ffn": nn.Sequential(
                    nn.LayerNorm([n, n, h]),
                    nn.Linear(h, h * 3), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(h * 3, h), nn.Dropout(dropout),
                ),
            }) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, h * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes),
        )

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.fourier_norm(self.fourier(h))
        for block in self.blocks:
            h = block["fluid"](h)
            h = h + block["ffn"](h)
        return self.head(self.final_norm(h).permute(0, 3, 1, 2))

# ============ WideFluidFormer ============
class WideFluidFormer(nn.Module):
    def __init__(self, n=28, h=16, num_blocks=4, num_classes=10, static_adr_steps=4, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "fluid": FieldFluidBlock(n, h, static_adr_steps, adr_hidden=48),
                "ffn": nn.Sequential(
                    nn.LayerNorm([n, n, h]),
                    nn.Linear(h, h * 3), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(h * 3, h), nn.Dropout(dropout),
                ),
            }) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, h * 3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 3, h * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes),
        )

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x) + self.pos_embed
        h = h + self.fourier_norm(self.fourier(h))
        for block in self.blocks:
            h = block["fluid"](h)
            h = h + block["ffn"](h)
        return self.head(self.final_norm(h).permute(0, 3, 1, 2))

# ============ Builders ============
def build_baseline():
    return fn.FluidResidualSequenceNetwork(
        fn.FluidResidualSequenceConfig(
            n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
            q_channels=32, q_depth=7, q_height=7, q_width=1,
            extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))

# ============ Training ============
def train_eval(model, tr_ldr, vl_ldr, device, steps=300, lr=1e-3, wd=1e-4, name=""):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()
    ll, la = 0.0, 0.0
    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        fld = img2field(x)
        logits = model(fld, mode="nonsequential", max_output_steps=1)
        if hasattr(logits, 'outputs'):
            logits = logits.outputs[:, 0]
        F.cross_entropy(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        ll = float(F.cross_entropy(logits.detach(), y).cpu())
        la = acc_fn(logits.detach(), y)
    sec = time.perf_counter() - t0

    model.eval(); vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            fld = img2field(x)
            logits = model(fld, mode="nonsequential", max_output_steps=1)
            if hasattr(logits, 'outputs'):
                logits = logits.outputs[:, 0]
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc / vt,
            "val_loss": vs / vt, "seconds": sec}

# ============ Validation ============
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device} | PyTorch: {torch.__version__}")

    x = torch.randn(2, 28, 28, 1).to(device)
    for name, builder in [
        ("Baseline", build_baseline),
        ("Deep(h=8,nb=3)", lambda: DeepFluidFormer(28, 8, 3, 10)),
        ("Deep(h=8,nb=4)", lambda: DeepFluidFormer(28, 8, 4, 10)),
        ("Wide(h=16,nb=4)", lambda: WideFluidFormer(28, 16, 4, 10)),
    ]:
        m = builder().to(device)
        out = m(x, mode="nonsequential", max_output_steps=1)
        logits = out if isinstance(out, torch.Tensor) else out.outputs[:, 0]
        p = sum(p.numel() for p in m.parameters())
        print(f"  {name}: P={p:,}, output={logits.shape} OK")
    print("All models validated!")
