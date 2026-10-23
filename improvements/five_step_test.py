"""
FluidFormer 五步渐进提升 - 完整对比测试 (v3 - ADRBlock3D)
============================================================
关键修复: Step3+ 使用 ADRBlock3D 直接场演化，绕开 Q 压缩瓶颈
"""
from __future__ import annotations

import json, math, random, sys, time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fluid_net as fn
from fluid_net import ADRBlock3D, FluidResidualSequenceConfig

# ========== Utils ==========
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_logits(out):
    return out.outputs[:, 0] if hasattr(out, "outputs") else out

def img2field(x):
    return x.permute(0, 2, 3, 1).contiguous()

def make_subset(ds, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()
    return Subset(ds, idx)

def acc_fn(logits, y):
    return float((logits.argmax(dim=-1) == y).float().mean().cpu())

def wrap_output(logits):
    return type('obj', (object,), {'outputs': logits.unsqueeze(1)})()

# ========== Step 0: Baseline ==========
def build_baseline(n=28, h=1, nc=10):
    return fn.FluidResidualSequenceNetwork(FluidResidualSequenceConfig(
        n=n, h=h, output_dim=nc, max_output_steps=1, static_adr_steps=4,
        q_channels=32, q_depth=7, q_height=7, q_width=1,
        extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))

# ========== FourierGlobalMix ==========
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

# ========== ADRFieldBlock (使用 ADRBlock3D 直接场演化) ==========
class ADRFieldBlock(nn.Module):
    """直接 ADR 场演化，输出 [B, n, n, h]，无 Q 压缩瓶颈"""
    def __init__(self, n, h):
        super().__init__()
        cfg = FluidResidualSequenceConfig(n=n, h=h, output_dim=10)  # output_dim dummy
        self.adr = ADRBlock3D(cfg)
        self.n, self.h = n, h

    def forward(self, x):
        # x: [B, n, n, h] -> need [B, 1, n, n, h]
        B = x.shape[0]
        step_idx = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        out = self.adr(x.unsqueeze(1), step_idx)  # [B, 1, n, n, h]
        return out.squeeze(1)  # [B, n, n, h]

# ========== DeepADRBlock (Pre-LN + ADR + FFN) ==========
class DeepADRBlock(nn.Module):
    def __init__(self, n, h, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([n, n, h])
        self.adr_field = ADRFieldBlock(n, h)
        self.norm2 = nn.LayerNorm([n, n, h])
        self.ffn = nn.Sequential(
            nn.Linear(h, h * 3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 3, h), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.adr_field(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ========== ContentGate ==========
class ContentGate(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, h), nn.Sigmoid())

    def forward(self, x):
        g = self.net(x.permute(0, 3, 1, 2)).unsqueeze(1).unsqueeze(1)
        return x * g

# ========== ClsHead ==========
class ClsHead(nn.Module):
    def __init__(self, h, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, h * 2), nn.GELU(),
            nn.Linear(h * 2, nc))

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2))

# ========== Step 1: +FourierGlobalMix ==========
class FluidFormerStep1(nn.Module):
    def __init__(self, n=28, h=4, nc=10):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.fourier = FourierGlobalMix(n, h)
        self.norm_f = nn.LayerNorm([n, n, h])
        self.adr = fn.FluidResidualSequenceNetwork(FluidResidualSequenceConfig(
            n=n, h=h, output_dim=nc, max_output_steps=1, static_adr_steps=3,
            q_channels=24, q_depth=4, q_height=4, q_width=1,
            extractor_hidden_channels=36, reconstructor_hidden_channels=36, output_hidden_dim=160))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.norm_f(self.fourier(h))
        return self.adr(h, mode=mode, max_output_steps=max_output_steps)

# ========== Step 2: +MultiHead ADR ==========
class MultiHeadADR(nn.Module):
    def __init__(self, n, h, num_heads=4, nc=10):
        super().__init__()
        self.num_heads = num_heads
        head_h = max(1, h // num_heads)
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            r = i / max(1, num_heads - 1)
            self.heads.append(fn.FluidResidualSequenceNetwork(FluidResidualSequenceConfig(
                n=n, h=head_h, output_dim=nc, max_output_steps=1, static_adr_steps=2,
                q_channels=12, q_depth=2, q_height=2, q_width=1,
                extractor_hidden_channels=20, reconstructor_hidden_channels=20, output_hidden_dim=64,
                max_velocity=0.3 + 1.4 * r, max_diffusion=0.02 + 0.16 * r)))
        self.proj = nn.Linear(num_heads * nc, nc)

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        head_h = x.shape[-1] // self.num_heads
        outs = []
        for i, head in enumerate(self.heads):
            xi = x[:, :, :, i * head_h:(i + 1) * head_h]
            outs.append(head(xi, mode=mode, max_output_steps=max_output_steps).outputs[:, 0])
        return self.proj(torch.cat(outs, dim=-1))

class FluidFormerStep2(nn.Module):
    def __init__(self, n=28, h=4, nh=4, nc=10):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.fourier = FourierGlobalMix(n, h)
        self.norm_f = nn.LayerNorm([n, n, h])
        self.mha = MultiHeadADR(n, h, nh, nc)

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.norm_f(self.fourier(h))
        return wrap_output(self.mha(h, mode, max_output_steps))

# ========== Step 3: +Deep Stack (直接ADR场演化) ==========
class FluidFormerStep3(nn.Module):
    def __init__(self, n=28, h=4, nb=3, nc=10):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.fourier = FourierGlobalMix(n, h)
        self.norm_f = nn.LayerNorm([n, n, h])
        self.blocks = nn.ModuleList([DeepADRBlock(n, h, 0.1) for _ in range(nb)])
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = ClsHead(h, nc)

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.norm_f(self.fourier(h))
        for blk in self.blocks:
            h = blk(h)
        return wrap_output(self.head(self.final_norm(h)))

# ========== Step 4: +ContentGate + CrossScale ==========
class FluidFormerStep4(nn.Module):
    def __init__(self, n=28, h=4, nb=3, nc=10):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.fourier = FourierGlobalMix(n, h)
        self.norm_f = nn.LayerNorm([n, n, h])
        self.pool = nn.AvgPool2d(2, 2)
        self.upsample = nn.Upsample(size=(n, n), mode='bilinear')
        self.coarse_adr = ADRFieldBlock(n // 2, h)
        self.gate = ContentGate(h)
        self.blocks = nn.ModuleList([DeepADRBlock(n, h, 0.1) for _ in range(nb)])
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = ClsHead(h, nc)

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.norm_f(self.fourier(h))
        # Coarse cross-scale
        c = self.pool(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.coarse_adr(c)
        h = h + self.upsample(c.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        h = self.gate(h)
        for blk in self.blocks:
            h = blk(h)
        return wrap_output(self.head(self.final_norm(h)))

# ========== Step 5: ScaleUp ==========
class FluidFormerStep5(nn.Module):
    def __init__(self, n=28, h=16, nb=6, nc=10):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.norm_f = nn.LayerNorm([n, n, h])
        self.blocks = nn.ModuleList([DeepADRBlock(n, h, 0.15) for _ in range(nb)])
        self.gate = ContentGate(h)
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, h * 3), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(h * 3, h * 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(h * 2, nc))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x) + self.pos_embed
        h = h + self.norm_f(self.fourier(h))
        h = self.gate(h)
        for blk in self.blocks:
            h = blk(h)
        return wrap_output(self.head(self.final_norm(h).permute(0, 3, 1, 2)))

# ========== Training ==========
def train_eval(model, tr_ldr, vl_ldr, device, steps=300, lr=1e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()
    ll, la = 0.0, 0.0
    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        fld = img2field(x)
        out = model(fld, mode="nonsequential", max_output_steps=1)
        logits = get_logits(out)
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
            out = model(fld, mode="nonsequential", max_output_steps=1)
            logits = get_logits(out)
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()
    return {"params": sum(p.numel() for p in model.parameters()),
            "train_acc": la, "val_acc": vc / vt, "val_loss": vs / vt, "seconds": sec}

# ========== Main ==========
def main():
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_root = str(ROOT / "data")

    def loaders(ds_cls, n_tr, n_vl):
        tr = make_subset(ds_cls(data_root, train=True, download=False, transform=t), n_tr)
        vl = make_subset(ds_cls(data_root, train=False, download=False, transform=t), n_vl)
        return DataLoader(tr, 64, shuffle=True, drop_last=True), DataLoader(vl, 64)

    mnist_tr, mnist_vl = loaders(datasets.MNIST, 3000, 1000)
    fmnist_tr, fmnist_vl = loaders(datasets.FashionMNIST, 3000, 1000)

    print(f"Device: {device} | Torch: {torch.__version__}")

    models = [
        ("Step0 Baseline FluidNet  ", build_baseline(28, 1, 10)),
        ("Step1 +FourierGlobalMix  ", FluidFormerStep1(28, 4, 10)),
        ("Step2 +MultiHead (4h)    ", FluidFormerStep2(28, 4, 4, 10)),
        ("Step3 +Deep Stack (3blk) ", FluidFormerStep3(28, 4, 3, 10)),
        ("Step4 +Gate+CrossScale   ", FluidFormerStep4(28, 4, 3, 10)),
        ("Step5 ScaleUp(h=16,6blk) ", FluidFormerStep5(28, 16, 6, 10)),
    ]

    all_results = []
    for ds_name, tr_ldr, vl_ldr, steps in [
        ("MNIST", mnist_tr, mnist_vl, 400),
        ("FashionMNIST", fmnist_tr, fmnist_vl, 400),
    ]:
        print(f"\n{'='*70}")
        print(f"  {ds_name} ({steps} steps)")
        print(f"{'='*70}")
        for name, model in models:
            if "Step5" in name and ds_name == "FashionMNIST":
                continue  # too slow
            set_seed(42)
            r = train_eval(model, tr_ldr, vl_ldr, device, steps, lr=1e-3)
            r["model"] = name.strip(); r["dataset"] = ds_name; r["steps"] = steps
            all_results.append(r)
            print(f"  {name} | P={r['params']:>8,} | acc={r['val_acc']:.4f} | loss={r['val_loss']:.4f} | {r['seconds']:.1f}s")

    # Summary
    print(f"\n{'='*75}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*75}")
    print(f"{'Model':<30s} {'Params':>8s} {'MNIST':>8s} {'F-MNIST':>8s} {'Δ MNIST':>10s}")
    print("-" * 70)
    mnist_map = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fmnist_map = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    base_mnist = mnist_map.get("Step0 Baseline FluidNet", {}).get("val_acc", 0)

    for name, _ in models:
        m = mnist_map.get(name.strip(), {})
        f = fmnist_map.get(name.strip(), {})
        ma = m.get("val_acc", 0); fa = f.get("val_acc", 0)
        mp = f'{m.get("params", 0):,}'
        delta = f"{ma-base_mnist:+.4f}" if base_mnist > 0 else "N/A"
        fmnist_str = f"{fa:.4f}" if fa else "N/A"
        print(f"{name:<30s} {mp:>8s} {ma:.4f}     {fmnist_str:>8s} {delta:>10s}")

    # Analysis
    print(f"\n{'='*75}")
    print("  ANALYSIS")
    print(f"{'='*75}")
    s1 = mnist_map.get("Step1 +FourierGlobalMix", {})
    s0 = mnist_map.get("Step0 Baseline FluidNet", {})
    if s0 and s1:
        ratio = s1["params"] / s0["params"] * 100
        print(f"  Step1 param efficiency: {ratio:.0f}% params, {-abs(s1['val_acc']-s0['val_acc']):.4f} acc gap")
    
    s3 = mnist_map.get("Step3 +Deep Stack (3blk)", {})
    if s3:
        print(f"  Step3 deep stack:       {s3.get('val_acc',0):.4f} acc with {s3.get('params',0):,} params")
    
    s5 = mnist_map.get("Step5 ScaleUp(h=16,6blk)", {})
    if s5:
        print(f"  Step5 scale-up:         {s5.get('val_acc',0):.4f} acc with {s5.get('params',0):,} params")

if __name__ == "__main__":
    main()
