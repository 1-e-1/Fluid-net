# -*- coding: utf-8 -*-
"""
experiments_v3/tests/nocompress_test.py
========================================
NoCompressFluidNet: 取消Q压缩层, 同维度ADR变换 + 大输出维度

核心创新 (基于用户想法):
  1. 无Q压缩: 场演化保持 [B,1,n,n,h] 同维度, 不经过 extractor/reconstructor
  2. 深度堆叠: ADRBlock3D 可以叠 K 层 (打破2层限制)
  3. 大输出维度: 每层输出大向量, 或最终场池化 → 大MLP
"""
from __future__ import annotations

import sys, time, json, math, random
from pathlib import Path
from itertools import cycle
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import (
    set_seed, img2field, make_subset, FourierGlobalMix,
    CosineWarmupScheduler, LabelSmoothingLoss,
)

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")


def make_adr_block(n, h, adr_hidden=48, adr_steps_internal=1):
    """构造一个可独立使用的同维度ADR块"""
    cfg = fn.FluidResidualSequenceConfig(
        n=n, h=h, output_dim=h,
        max_output_steps=1, static_adr_steps=1,
        adr_hidden_channels=adr_hidden,
        q_channels=16, q_depth=1, q_height=1, q_width=1,
        extractor_hidden_channels=16, reconstructor_hidden_channels=16,
        output_hidden_dim=128,
    )
    return fn.ADRBlock3D(cfg)


# ============================================================
# V1: 同维度ADR深度堆叠 + 最终场池化 → 大MLP
# ============================================================
class NoCompressFluidNet_V1(nn.Module):
    """K层同维度ADR + 残差 + 最终场全局池化 → 大维度MLP"""
    def __init__(self, n=28, h=8, num_layers=6, num_classes=10,
                 adr_hidden=48, out_dim=256, dropout=0.15):
        super().__init__()
        self.h = h
        self.proj = nn.Linear(1, h)  # [B,n,n,1] -> [B,n,n,h]
        self.pos = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)

        # Fourier 预处理 (全局交互)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])

        # K层同维度ADR (每个都独立, 带残差+LayerNorm)
        self.adr_layers = nn.ModuleList()
        self.adr_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.adr_layers.append(make_adr_block(n, h, adr_hidden))
            self.adr_norms.append(nn.LayerNorm([n, n, h]))

        # 大输出头: 全局池化 → 大MLP
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),          # h
            nn.Linear(h, out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes),
        )

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        f = self.proj(x) + self.pos           # [B,n,n,h]
        f = f + self.fourier_norm(self.fourier(f))

        f5 = f.unsqueeze(1)  # [B,1,n,n,h]

        for layer_i, (adr, norm) in enumerate(zip(self.adr_layers, self.adr_norms)):
            step_idx = torch.full((f5.shape[0], 1), layer_i, dtype=torch.long, device=f5.device)
            delta = adr(f5, step_idx)
            # 残差 + 归一化 (Transformer风格)
            f5 = norm((f5 + delta).squeeze(1)).unsqueeze(1)

        f_out = f5.squeeze(1)  # [B,n,n,h]
        return self.head(f_out.permute(0, 3, 1, 2))


# ============================================================
# V2: 同维度ADR + 每层输出大向量 (累积特征金字塔)
# ============================================================
class NoCompressFluidNet_V2(nn.Module):
    """K层ADR, 每层输出一个大向量, 拼接后大MLP"""
    def __init__(self, n=28, h=8, num_layers=6, num_classes=10,
                 adr_hidden=48, per_layer_out=128, dropout=0.15):
        super().__init__()
        self.h = h
        self.num_layers = num_layers
        self.proj = nn.Linear(1, h)
        self.pos = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)

        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])

        # K层ADR + 每层输出头
        self.adr_layers = nn.ModuleList()
        self.adr_norms = nn.ModuleList()
        self.layer_heads = nn.ModuleList()
        for _ in range(num_layers):
            self.adr_layers.append(make_adr_block(n, h, adr_hidden))
            self.adr_norms.append(nn.LayerNorm([n, n, h]))
            self.layer_heads.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(h, per_layer_out), nn.GELU(),
            ))

        # 拼接所有层输出 → 大MLP
        total_out = per_layer_out * num_layers
        self.head = nn.Sequential(
            nn.Linear(total_out, total_out // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(total_out // 2, total_out // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(total_out // 4, num_classes),
        )

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        f = self.proj(x) + self.pos
        f = f + self.fourier_norm(self.fourier(f))

        f5 = f.unsqueeze(1)  # [B,1,n,n,h]
        layer_outs = []

        for layer_i, (adr, norm, head) in enumerate(
            zip(self.adr_layers, self.adr_norms, self.layer_heads)):
            step_idx = torch.full((f5.shape[0], 1), layer_i, dtype=torch.long, device=f5.device)
            delta = adr(f5, step_idx)
            f5 = norm((f5 + delta).squeeze(1)).unsqueeze(1)

            # 每层输出一个大向量
            layer_outs.append(head(f5.squeeze(1).permute(0, 3, 1, 2)))

        fused = torch.cat(layer_outs, dim=-1)  # [B, per_layer_out * num_layers]
        return self.head(fused)


# ============================================================
# V3: 加深+加宽, 无残差直接ADR (验证纯堆叠能力)
# ============================================================
class NoCompressFluidNet_V3(nn.Module):
    """纯ADR堆叠 (无残差), 验证深层数值稳定性 + 大输出"""
    def __init__(self, n=28, h=8, num_layers=8, num_classes=10,
                 adr_hidden=48, out_dim=256, dropout=0.15):
        super().__init__()
        self.h = h
        self.proj = nn.Linear(1, h)
        self.pos = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])

        self.adr_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.adr_layers.append(make_adr_block(n, h, adr_hidden))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes),
        )

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        f = self.proj(x) + self.pos
        f = f + self.fourier_norm(self.fourier(f))
        f5 = f.unsqueeze(1)  # [B,1,n,n,h]

        for layer_i, adr in enumerate(self.adr_layers):
            step_idx = torch.full((f5.shape[0], 1), layer_i, dtype=torch.long, device=f5.device)
            f5 = adr(f5, step_idx)

        f_out = f5.squeeze(1)
        return self.head(f_out.permute(0, 3, 1, 2))


# ============================================================
# Training
# ============================================================
def train_eval_nc(model, tr_ldr, vl_ldr, device, steps=600, lr=1.2e-3, wd=1e-4,
                  name="", num_classes=10):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    crit = LabelSmoothingLoss(num_classes, 0.05)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        fld = img2field(x)
        out = model(fld, mode="nonsequential", max_output_steps=1)
        logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
        if logits.dim() > 2: logits = logits.squeeze(1)
        crit(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    sec = time.perf_counter() - t0
    model.eval()
    vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            fld = img2field(x)
            out = model(fld, mode="nonsequential", max_output_steps=1)
            logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
            if logits.dim() > 2: logits = logits.squeeze(1)
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc / vt,
            "val_loss": vs / vt, "seconds": sec}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    set_seed(42)
    print("=" * 70)
    print("  NO-COMPRESS FLUIDNET: 同维度ADR + 大输出维度")
    print("=" * 70)

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    all_results = []

    for ds_name, ds_cls in [("MNIST", datasets.MNIST), ("FashionMNIST", datasets.FashionMNIST)]:
        tr_size = 3000 if ds_name == "MNIST" else 5000
        tr = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=True, download=False, transform=t), tr_size)
        vl = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=False, download=False, transform=t), 1000)
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)

        print(f"\n{'='*60} {ds_name}")

        # Baseline (含Q压缩)
        set_seed(42)
        b = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(
            n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
            q_channels=32, q_depth=7, q_height=7, q_width=1,
            extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))
        r = train_eval_nc(b, tr_ldr, vl_ldr, device, steps=600, lr=1e-3, wd=1e-4,
                          name=f"{ds_name[:4]}-Baseline-Q", num_classes=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  Baseline (Q压缩):   acc={r['val_acc']:.4f} P={r['params']:,}")

        # V1: 6层同维度ADR + 残差 + 大输出
        set_seed(42)
        m1 = NoCompressFluidNet_V1(n=28, h=8, num_layers=6, num_classes=10, out_dim=256)
        r = train_eval_nc(m1, tr_ldr, vl_ldr, device, steps=600, lr=1.2e-3, wd=1e-4,
                          name=f"{ds_name[:4]}-NC-V1(6L)", num_classes=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  NC-V1 6层:          acc={r['val_acc']:.4f} P={r['params']:,}")

        # V2: 6层 + 每层输出头 (特征金字塔)
        set_seed(42)
        m2 = NoCompressFluidNet_V2(n=28, h=8, num_layers=6, num_classes=10, per_layer_out=128)
        r = train_eval_nc(m2, tr_ldr, vl_ldr, device, steps=600, lr=1.2e-3, wd=1e-4,
                          name=f"{ds_name[:4]}-NC-V2(6L-pyramid)", num_classes=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  NC-V2 6层金字塔:   acc={r['val_acc']:.4f} P={r['params']:,}")

        # V3: 8层纯ADR (无残差, 验证深度)
        set_seed(42)
        m3 = NoCompressFluidNet_V3(n=28, h=8, num_layers=8, num_classes=10, out_dim=256)
        r = train_eval_nc(m3, tr_ldr, vl_ldr, device, steps=600, lr=1.2e-3, wd=1e-4,
                          name=f"{ds_name[:4]}-NC-V3(8L-nores)", num_classes=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  NC-V3 8层无残差:   acc={r['val_acc']:.4f} P={r['params']:,}")

        # V1-Deep: 16层同维度ADR (打破2层限制)
        if ds_name == "MNIST":
            set_seed(42)
            m4 = NoCompressFluidNet_V1(n=28, h=8, num_layers=16, num_classes=10, out_dim=256)
            r = train_eval_nc(m4, tr_ldr, vl_ldr, device, steps=600, lr=1.2e-3, wd=1e-4,
                              name=f"{ds_name[:4]}-NC-V1(16L)", num_classes=10)
            r["dataset"] = ds_name; all_results.append(r)
            print(f"  NC-V1 16层:         acc={r['val_acc']:.4f} P={r['params']:,}")

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("  NO-COMPRESS FLUIDNET SUMMARY")
    print("=" * 70)
    print(f"{'Model':<26s} {'MNIST':>8s} {'FMNIST':>8s} {'Params':>10s}")
    print("-" * 56)
    mnist = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fmnist = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    for name in sorted(mnist.keys()):
        m = mnist[name]; f = fmnist.get(name, {})
        print(f"{name:<26s} {m['val_acc']:.4f}     {f.get('val_acc',0):.4f}     {m['params']:>10,}")

    out = ROOT / "experiments_v3" / "results" / "nocompress_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
