# -*- coding: utf-8 -*-
"""
Phase 1: FluidFormerV2 - 修复三大根因
=======================================
根因1修复: ADRBlock3D + LayerNorm (控制范数)
根因2修复: 加法残差 h + ADR(h) (不用减法)
根因3修复: Skip Connection 跨层直达
"""
import sys, json, time, random, math
from itertools import cycle
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))

import fluid_net as fn
from fluid_net import ADRBlock3D, FluidResidualSequenceConfig

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

# ============ NormedADR: LayerNorm + ADRBlock3D ============
class NormedADR(nn.Module):
    """根因1修复: 每次ADR迭代前后加LayerNorm, 阻止范数爆炸"""
    def __init__(self, n, h, num_steps=4, enable_step_cond=True):
        super().__init__()
        cfg = FluidResidualSequenceConfig(
            n=n, h=h, output_dim=10,
            enable_step_conditioned_adr=enable_step_cond,
            step_encoder_hidden=16 if enable_step_cond else None,
            step_modulation_range=0.5,
            adr_hidden_channels=32,
        )
        self.adr = ADRBlock3D(cfg)
        self.norm = nn.LayerNorm([n, n, h])
        self.num_steps = num_steps
        self.n, self.h = n, h

    def forward(self, x):
        # x: [B, n, n, h]
        B = x.shape[0]
        out = self.norm(x).unsqueeze(1)  # [B, 1, n, n, h], 预归一化
        for s in range(self.num_steps):
            step_idx = torch.full((B, 1), float(s + 1), device=x.device, dtype=x.dtype)
            out = self.adr(out, step_idx)
            # 每次ADR后归一化 → 根因1修复
            out = self.norm(out.squeeze(1)).unsqueeze(1) if s < self.num_steps - 1 else out
        return out.squeeze(1)

# ============ FluidFormerV2 Block ============
class FluidFormerV2Block(nn.Module):
    """
    根因2修复: h + ADR(h) 加法残差
    根因3修复: Skip connection
    Transformer式 Pre-LN + ADR + FFN + Residual
    """
    def __init__(self, n, h, adr_steps=4, dropout=0.1, expansion=3):
        super().__init__()
        # ADR sublayer
        self.norm1 = nn.LayerNorm([n, n, h])
        self.adr = NormedADR(n, h, adr_steps, enable_step_cond=True)
        # FFN sublayer (类似 Transformer FFN)
        self.norm2 = nn.LayerNorm([n, n, h])
        self.ffn = nn.Sequential(
            nn.Linear(h, h * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * expansion, h), nn.Dropout(dropout),
        )

    def forward(self, x):
        # 加法残差: h + ADR(LN(h)) — 根因2修复
        x = x + self.adr(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ============ FluidFormerV2 完整架构 ============
class FluidFormerV2(nn.Module):
    def __init__(self, n=28, h=16, num_blocks=4, num_classes=10, 
                 adr_steps=4, dropout=0.1, use_fourier=True):
        super().__init__()
        self.use_fourier = use_fourier
        # 输入投影
        self.proj = nn.Linear(1, h)
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        # 全局 Fourier 混合
        if use_fourier:
            self.fourier = FourierGlobalMix(n, h)
            self.fourier_norm = nn.LayerNorm([n, n, h])
        # 深层块
        self.blocks = nn.ModuleList([
            FluidFormerV2Block(n, h, adr_steps, dropout) for _ in range(num_blocks)
        ])
        # 最终归一化 + 分类头
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, h * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        """
        x: [B, n, n, 1] field format
        returns: [B, num_classes]
        """
        h = self.proj(x) + self.pos_embed
        
        if self.use_fourier:
            h = h + self.fourier_norm(self.fourier(h))
        
        for block in self.blocks:
            h = block(h)
        
        return self.head(self.final_norm(h).permute(0, 3, 1, 2))

# ============ Builders ============
def build_v2_small(n=28, nc=10):
    return FluidFormerV2(n=n, h=8, num_blocks=3, num_classes=nc, adr_steps=4, dropout=0.1)

def build_v2_base(n=28, nc=10):
    return FluidFormerV2(n=n, h=16, num_blocks=4, num_classes=nc, adr_steps=6, dropout=0.15)

def build_v2_large(n=28, nc=10):
    return FluidFormerV2(n=n, h=24, num_blocks=6, num_classes=nc, adr_steps=6, dropout=0.15)

def build_baseline():
    return fn.FluidResidualSequenceNetwork(FluidResidualSequenceConfig(
        n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
        q_channels=32, q_depth=7, q_height=7, q_width=1,
        extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))

# ============ Training ============
def train_eval(model, tr_ldr, vl_ldr, device, steps, lr, wd, name, is_img=True):
    model.to(device)
    # Use SGD with momentum for deeper models (better than Adam for ViT-like archs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    # Warmup + Cosine schedule
    warmup = max(1, steps // 10)
    def lr_lambda(s):
        if s < warmup: return s / warmup
        progress = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()
    for s in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if is_img:
            out = model(img2field(x))
        else:
            out = model(x)
        logits = out if isinstance(out, torch.Tensor) else out.outputs[:, 0]
        F.cross_entropy(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    sec = time.perf_counter() - t0

    model.eval(); vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            if is_img:
                out = model(img2field(x))
            else:
                out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out.outputs[:, 0]
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc/vt,
            "val_loss": vs/vt, "seconds": sec, "steps": steps}

# ============ Main Test ============
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    print(f"Phase 1: FluidFormerV2 | Device: {device}")

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_root = str(ROOT / "data")
    mnist_tr = make_subset(datasets.MNIST(data_root, train=True, download=False, transform=t), 3000)
    mnist_vl = make_subset(datasets.MNIST(data_root, train=False, download=False, transform=t), 1000)
    fmnist_tr = make_subset(datasets.FashionMNIST(data_root, train=True, download=False, transform=t), 3000)
    fmnist_vl = make_subset(datasets.FashionMNIST(data_root, train=False, download=False, transform=t), 1000)

    all_results = []

    for ds_name, tr, vl in [("MNIST", mnist_tr, mnist_vl), ("FashionMNIST", fmnist_tr, fmnist_vl)]:
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)
        steps = 600 if ds_name == "MNIST" else 500
        print(f"\n{'='*55} {ds_name}")

        models = [
            ("Baseline", build_baseline(), steps, 1e-3, 1e-4),
            ("V2-Small(h=8,nb=3)", build_v2_small(), steps, 1.5e-3, 2e-4),
            ("V2-Base(h=16,nb=4)", build_v2_base(), steps, 1.5e-3, 3e-4),
        ]
        if ds_name == "MNIST":
            models.append(("V2-Large(h=24,nb=6)", build_v2_large(), steps, 2e-3, 3e-4))

        for name, model, st, lr, wd in models:
            set_seed(42)
            r = train_eval(model, tr_ldr, vl_ldr, device, st, lr, wd, name)
            r["dataset"] = ds_name
            all_results.append(r)
            print(f"  {name:30s} P={r['params']:>8,} acc={r['val_acc']:.4f} loss={r['val_loss']:.4f} {r['seconds']:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  PHASE 1 RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<30s} {'Params':>8s} {'MNIST':>8s} {'FMNIST':>8s}")
    print("-" * 58)
    mnist_r = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fmnist_r = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    for name in ["Baseline", "V2-Small(h=8,nb=3)", "V2-Base(h=16,nb=4)", "V2-Large(h=24,nb=6)"]:
        m = mnist_r.get(name, {})
        f = fmnist_r.get(name, {})
        p = m.get("params", f.get("params", 0))
        print(f"{name:<30s} {p:>8,} {m.get('val_acc',0):.4f}     {f.get('val_acc',0):.4f}")

    # Save
    out_path = ROOT / "experiments_v2" / "results" / "phase1_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
