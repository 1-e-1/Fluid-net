# -*- coding: utf-8 -*-
"""
experiments_v3/models/improved_fluid.py (v2 - 修复版)
======================================================
核心改进:
1. EnhancedFluidFormer: Fourier + ChannelMix + SpatialMix → FluidNet(Q压缩) → outputs
2. MultiScaleFluidFormer: 多尺度 + 完整FluidNet管线
3. 增强训练: warmup+cosine, label smoothing, RandAugment-lite
"""
from __future__ import annotations

import math, random, sys, time, json
from itertools import cycle
from pathlib import Path
from typing import Optional

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
import fluid_net as fn

# ===================== Utils =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def img2field(x):
    return x.permute(0, 2, 3, 1).contiguous()

def make_subset(ds, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()
    return Subset(ds, idx)

def acc_fn(logits, y):
    return float((logits.argmax(dim=-1) == y).float().mean().cpu())

# ===================== FourierGlobalMix =====================
class FourierGlobalMix(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n, n // 2 + 1, h, 2) * 0.02)

    def forward(self, x):
        xf = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        xf = torch.view_as_real(xf)
        xf = xf * (1.0 + self.weight)
        xf = torch.view_as_complex(xf)
        return torch.fft.irfft2(xf, dim=(1, 2), norm="ortho",
                                s=(x.shape[1], x.shape[2])).to(x.dtype)

# ===================== ChannelMix =====================
class ChannelMix(nn.Module):
    def __init__(self, n, h, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm([n, n, h])
        self.mlp = nn.Sequential(
            nn.Linear(h, h * expansion),
            nn.GELU(),
            nn.Linear(h * expansion, h),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))

# ===================== SpatialMix =====================
class SpatialMix(nn.Module):
    def __init__(self, n, h, expansion=2):
        super().__init__()
        # 安全地选择 groups
        g = 1
        for candidate in [4, 2]:
            if h % candidate == 0:
                g = candidate
                break
        self.norm = nn.LayerNorm([n, n, h])
        self.mlp = nn.Sequential(
            nn.Conv2d(h, h * expansion, 3, padding=1, groups=g),
            nn.GELU(),
            nn.Conv2d(h * expansion, h, 1),
        )

    def forward(self, x):
        r = x.permute(0, 3, 1, 2)
        r = self.mlp(r)
        r = r.permute(0, 2, 3, 1)
        return x + r

# ===================== EnhancedFluidFormer (核心模型) =====================
class EnhancedFluidFormer(nn.Module):
    """完整FluidNet管线: Fourier + Mixes + FluidNet(含Q压缩)"""
    def __init__(self, n=28, h=12, num_classes=10, static_adr_steps=6,
                 adr_hidden=48, dropout=0.15, q_channels=32):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])

        # 预处理: ChannelMix + SpatialMix
        self.channel_mix = ChannelMix(n, h, expansion=3)
        self.spatial_mix = SpatialMix(n, h, expansion=2)

        # FluidNet (完整管线, 含Q压缩)
        self.fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=q_channels, q_depth=4, q_height=4, q_width=1,
                extractor_hidden_channels=adr_hidden,
                reconstructor_hidden_channels=adr_hidden,
                output_hidden_dim=adr_hidden * 2,
                adr_hidden_channels=adr_hidden,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x) + self.pos_embed
        h = h + self.fourier_norm(self.fourier(h))
        h = self.channel_mix(h)
        h = self.spatial_mix(h)
        return self.fluid(h, mode=mode, max_output_steps=max_output_steps)

# ===================== MultiScaleFluidFormer =====================
class MultiScaleFluidFormer(nn.Module):
    """多尺度: 在28x28和14x14上分别运行完整FluidNet, 融合输出"""
    def __init__(self, n=28, h=12, num_classes=10, static_adr_steps=5,
                 dropout=0.15):
        super().__init__()
        self.h = h
        self.h_hi = h * 3 // 4  # 高分辨率通道数
        self.h_lo = h - self.h_hi  # 低分辨率通道数

        self.proj = nn.Linear(1, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)

        # 高分辨率流 (28x28)
        self.fourier_hi = FourierGlobalMix(n, self.h_hi)
        self.norm_hi = nn.LayerNorm([n, n, self.h_hi])
        self.channel_mix_hi = ChannelMix(n, self.h_hi, expansion=3)
        self.fluid_hi = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=self.h_hi, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=28, q_depth=4, q_height=4, q_width=1,
                extractor_hidden_channels=40,
                reconstructor_hidden_channels=40,
                output_hidden_dim=200,
                adr_hidden_channels=40,
            ))

        # 低分辨率流 (14x14)
        self.n14 = n // 2
        self.fourier_lo = FourierGlobalMix(self.n14, self.h_lo)
        self.norm_lo = nn.LayerNorm([self.n14, self.n14, self.h_lo])
        self.channel_mix_lo = ChannelMix(self.n14, self.h_lo, expansion=3)
        self.fluid_lo = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=self.n14, h=self.h_lo, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=24, q_depth=3, q_height=3, q_width=1,
                extractor_hidden_channels=32,
                reconstructor_hidden_channels=32,
                output_hidden_dim=160,
                adr_hidden_channels=32,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        f = self.proj(x) + self.pos_embed  # [B, 28, 28, h]

        # 高分辨率流
        f_hi = f[..., :self.h_hi]
        f_hi = f_hi + self.norm_hi(self.fourier_hi(f_hi))
        f_hi = self.channel_mix_hi(f_hi)
        out_hi = self.fluid_hi(f_hi, mode=mode, max_output_steps=max_output_steps)

        # 低分辨率流 (下采样)
        f_lo = f[..., self.h_hi:]
        f_lo = F.interpolate(f_lo.permute(0, 3, 1, 2), size=(self.n14, self.n14),
                             mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        f_lo = f_lo + self.norm_lo(self.fourier_lo(f_lo))
        f_lo = self.channel_mix_lo(f_lo)
        out_lo = self.fluid_lo(f_lo, mode=mode, max_output_steps=max_output_steps)

        # 融合logits
        logits_hi = out_hi.outputs[:, 0]
        logits_lo = out_lo.outputs[:, 0]
        logits = (logits_hi + logits_lo) / 2.0

        # 返回兼容格式
        return type("obj", (object,), {"outputs": logits.unsqueeze(1)})()

# ===================== NLP Improved =====================
class NLPImprovedFormer(nn.Module):
    def __init__(self, input_dim, n=16, h=12, num_classes=4, static_adr_steps=4,
                 dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, n * n * h),
        )
        self.n, self.h = n, h

        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.channel_mix = ChannelMix(n, h, expansion=3)

        self.fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=2, q_height=2, q_width=2,
                extractor_hidden_channels=48,
                reconstructor_hidden_channels=48,
                output_hidden_dim=256,
                adr_hidden_channels=48,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        B = x.shape[0]
        field = self.encoder(x).view(B, self.n, self.n, self.h)
        field = field + self.pos_embed
        field = field + self.fourier_norm(self.fourier(field))
        field = self.channel_mix(field)
        return self.fluid(field, mode=mode, max_output_steps=max_output_steps)

# ===================== MM Improved =====================
class MMImprovedFormer(nn.Module):
    def __init__(self, n_img=8, n_txt=8, h=8, num_classes=10, static_adr_steps=4,
                 dropout=0.2):
        super().__init__()
        self.h = h
        self.n_img, self.n_txt = n_img, n_txt

        self.img_proj = nn.Linear(1, h)
        self.img_fourier = FourierGlobalMix(n_img, h)
        self.img_norm = nn.LayerNorm([n_img, n_img, h])
        self.img_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_img, h=h, output_dim=h * 2,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=24, q_depth=1, q_height=1, q_width=1,
                extractor_hidden_channels=32,
                reconstructor_hidden_channels=32,
                output_hidden_dim=128, adr_hidden_channels=32,
            ))

        self.txt_proj = nn.Linear(4, h)
        self.txt_fourier = FourierGlobalMix(n_txt, h)
        self.txt_norm = nn.LayerNorm([n_txt, n_txt, h])
        self.txt_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_txt, h=h, output_dim=h * 2,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=24, q_depth=1, q_height=1, q_width=2,
                extractor_hidden_channels=32,
                reconstructor_hidden_channels=32,
                output_hidden_dim=128, adr_hidden_channels=32,
            ))

        self.fuse = nn.Sequential(
            nn.Linear(h * 4, h * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h * 4, h * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes),
        )

    def forward(self, img, txt, mode="nonsequential", max_output_steps=1):
        B = img.shape[0]
        img_f = img2field(img)
        img_f = self.img_proj(img_f)
        img_f = img_f + self.img_norm(self.img_fourier(img_f))
        img_out = self.img_fluid(img_f, mode=mode, max_output_steps=max_output_steps)
        img_feat = img_out.outputs[:, 0]  # [B, h*2]

        txt_f = self.txt_proj(txt)
        txt_f = txt_f.view(B, self.n_txt, self.n_txt, self.h)
        txt_f = txt_f + self.txt_norm(self.txt_fourier(txt_f))
        txt_out = self.txt_fluid(txt_f, mode=mode, max_output_steps=max_output_steps)
        txt_feat = txt_out.outputs[:, 0]  # [B, h*2]

        fused = torch.cat([img_feat, txt_feat], dim=-1)
        return self.fuse(fused)

# ===================== Training Utils =====================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min_factor=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = optimizer.param_groups[0]["lr"] * eta_min_factor
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

def rand_augment_lite(x, n_ops=1, magnitude=5):
    ops = []
    x_min = x.amin(dim=(-1, -2), keepdim=True)
    x_max = x.amax(dim=(-1, -2), keepdim=True)
    if (x_max - x_min).mean() > 1e-6:
        x = (x - x_min) / (x_max - x_min)

    for _ in range(n_ops):
        op = random.choice(["rotate", "translate_x", "translate_y", "shear_x",
                            "shear_y", "solarize", "invert", "sharpness"])
        mag = magnitude / 10.0

        if op == "rotate":
            deg = mag * 15 * random.uniform(-1, 1)
            theta = math.radians(deg)
            grid = F.affine_grid(
                torch.tensor([[[math.cos(theta), -math.sin(theta), 0],
                              [math.sin(theta), math.cos(theta), 0]]],
                             device=x.device).expand(x.size(0), -1, -1),
                x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")

        elif op in ("translate_x", "translate_y"):
            tx = mag * 0.2 * random.uniform(-1, 1) if op == "translate_x" else 0
            ty = mag * 0.2 * random.uniform(-1, 1) if op == "translate_y" else 0
            grid = F.affine_grid(
                torch.tensor([[[1, 0, tx], [0, 1, ty]]], device=x.device).float().expand(x.size(0), -1, -1),
                x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")

        elif op in ("shear_x", "shear_y"):
            sx = mag * 0.2 * random.uniform(-1, 1) if op == "shear_x" else 0
            sy = mag * 0.2 * random.uniform(-1, 1) if op == "shear_y" else 0
            grid = F.affine_grid(
                torch.tensor([[[1, sx, 0], [sy, 1, 0]]], device=x.device).float().expand(x.size(0), -1, -1),
                x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")

        elif op == "solarize":
            threshold = 0.5 + mag * 0.3 * random.uniform(-1, 1)
            x = torch.where(x < threshold, x, 1.0 - x)

        elif op == "invert":
            if random.random() < mag:
                x = 1.0 - x

        elif op == "sharpness":
            blur_kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
            blurred = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'),
                              blur_kernel.expand(x.size(1), -1, -1, -1), groups=x.size(1))
            x = x + mag * (x - blurred)

    return x

# ===================== Trainer =====================
def train_eval_improved(model, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3,
                        wd=1e-4, name="", label_smoothing=0.05, warmup_ratio=0.1,
                        use_augment=False, num_classes=10, is_image=True,
                        mm_mode=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * warmup_ratio))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    criterion = LabelSmoothingLoss(num_classes, label_smoothing)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for step_i in range(steps):
        model.train()
        batch = next(tr_iter)

        if mm_mode:
            x_img, x_txt, y = [b.to(device) for b in batch]
            if use_augment:
                x_img = rand_augment_lite(x_img)
            opt.zero_grad(set_to_none=True)
            logits = model(x_img, x_txt, mode="nonsequential", max_output_steps=1)
        else:
            x, y = [b.to(device) for b in batch[:2]]
            if use_augment and is_image:
                x = rand_augment_lite(x)
            opt.zero_grad(set_to_none=True)
            if is_image:
                fld = img2field(x)
            else:
                fld = x
            logits = model(fld, mode="nonsequential", max_output_steps=1)

        if hasattr(logits, 'outputs'):
            logits = logits.outputs[:, 0]
        if logits.dim() > 2:
            logits = logits.squeeze(1)

        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    sec = time.perf_counter() - t0

    model.eval()
    vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for batch in vl_ldr:
            if mm_mode:
                x_img, x_txt, y = [b.to(device) for b in batch]
                logits = model(x_img, x_txt, mode="nonsequential", max_output_steps=1)
            else:
                x, y = [b.to(device) for b in batch[:2]]
                if is_image:
                    fld = img2field(x)
                else:
                    fld = x
                logits = model(fld, mode="nonsequential", max_output_steps=1)

            if hasattr(logits, 'outputs'):
                logits = logits.outputs[:, 0]
            if logits.dim() > 2:
                logits = logits.squeeze(1)

            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()

    params = sum(p.numel() for p in model.parameters())
    return {
        "model": name, "params": params, "val_acc": vc / vt,
        "val_loss": vs / vt, "seconds": sec
    }

print("models/improved_fluid.py v2 loaded successfully!")
print(f"Available: EnhancedFluidFormer, MultiScaleFluidFormer")
print(f"           NLPImprovedFormer, MMImprovedFormer")
