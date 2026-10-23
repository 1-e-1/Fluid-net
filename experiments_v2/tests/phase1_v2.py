# -*- coding: utf-8 -*-
"""Phase 1: FluidFormerV2 - 修复三大根因 (Fixed)"""
import sys, json, time, random, math
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
from fluid_net import ADRBlock3D, FluidResidualSequenceConfig

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def img2field(x): return x.permute(0, 2, 3, 1).contiguous()
def make_subset(ds, n, seed=42):
    rng = np.random.default_rng(seed)
    return Subset(ds, rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist())

class FourierGlobalMix(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n, n // 2 + 1, h, 2) * 0.02)
    def forward(self, x):
        xf = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        xf = torch.view_as_real(xf) * (1.0 + self.weight)
        return torch.fft.irfft2(torch.view_as_complex(xf), dim=(1, 2), norm="ortho", s=(x.shape[1], x.shape[2])).to(x.dtype)

class NormedADR(nn.Module):
    def __init__(self, n, h, num_steps=4):
        super().__init__()
        cfg = FluidResidualSequenceConfig(n=n, h=h, output_dim=10,
            enable_step_conditioned_adr=True, step_encoder_hidden=16,
            step_modulation_range=0.5, adr_hidden_channels=32)
        self.adr = ADRBlock3D(cfg)
        self.norm = nn.LayerNorm([n, n, h])
        self.num_steps = num_steps
    def forward(self, x):
        B = x.shape[0]
        out = self.norm(x).unsqueeze(1)
        for s in range(self.num_steps):
            step_idx = torch.full((B, 1), float(s + 1), device=x.device, dtype=x.dtype)
            out = self.adr(out, step_idx)
        return self.norm(out.squeeze(1))

class V2Block(nn.Module):
    def __init__(self, n, h, adr_steps=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([n, n, h])
        self.adr = NormedADR(n, h, adr_steps)
        self.norm2 = nn.LayerNorm([n, n, h])
        self.ffn = nn.Sequential(
            nn.Linear(h, h * 3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 3, h), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.adr(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FluidFormerV2(nn.Module):
    def __init__(self, n=28, h=16, num_blocks=4, num_classes=10, adr_steps=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.blocks = nn.ModuleList([V2Block(n, h, adr_steps, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(h, h * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x) + self.pos_embed
        h = h + self.fourier_norm(self.fourier(h))
        for block in self.blocks: h = block(h)
        return self.head(self.final_norm(h).permute(0, 3, 1, 2))

def build_baseline():
    return fn.FluidResidualSequenceNetwork(FluidResidualSequenceConfig(
        n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
        q_channels=32, q_depth=7, q_height=7, q_width=1,
        extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))

def train_eval(model, tr_ldr, vl_ldr, device, steps, lr, wd, name):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    warmup = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt,
        lambda s: s/warmup if s < warmup else 0.5*(1+math.cos(math.pi*(s-warmup)/max(1,steps-warmup))))
    tr_iter = cycle(tr_ldr); t0 = time.perf_counter()
    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(img2field(x), mode="nonsequential", max_output_steps=1)
        logits = out if isinstance(out, torch.Tensor) else out.outputs[:, 0]
        F.cross_entropy(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    sec = time.perf_counter() - t0
    model.eval(); vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            out = model(img2field(x), mode="nonsequential", max_output_steps=1)
            logits = out if isinstance(out, torch.Tensor) else out.outputs[:, 0]
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu()); vt += y.numel()
    return {"model": name, "params": sum(p.numel() for p in model.parameters()),
            "val_acc": vc/vt, "val_loss": vs/vt, "seconds": sec}

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    print(f"FluidFormerV2 | Device: {device}")

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dr = str(ROOT / "data")
    mnist_tr = make_subset(datasets.MNIST(dr, train=True, download=False, transform=t), 3000)
    mnist_vl = make_subset(datasets.MNIST(dr, train=False, download=False, transform=t), 1000)
    fmnist_tr = make_subset(datasets.FashionMNIST(dr, train=True, download=False, transform=t), 3000)
    fmnist_vl = make_subset(datasets.FashionMNIST(dr, train=False, download=False, transform=t), 1000)

    all_results = []
    for ds_name, tr, vl in [("MNIST", mnist_tr, mnist_vl), ("FashionMNIST", fmnist_tr, fmnist_vl)]:
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)
        steps = 600 if ds_name == "MNIST" else 500
        print(f"\n{'='*50} {ds_name}")

        configs = [
            ("Baseline", build_baseline, steps, 1e-3, 1e-4),
            ("V2-S(h=8,nb=3)", lambda: FluidFormerV2(28, 8, 3, 10, 4, 0.1), steps, 1.5e-3, 2e-4),
            ("V2-B(h=16,nb=4)", lambda: FluidFormerV2(28, 16, 4, 10, 6, 0.15), steps, 1.5e-3, 3e-4),
        ]
        if ds_name == "MNIST":
            configs.append(("V2-L(h=24,nb=6)", lambda: FluidFormerV2(28, 24, 6, 10, 6, 0.15), steps, 2e-3, 3e-4))

        for name, builder, st, lr, wd in configs:
            set_seed(42)
            r = train_eval(builder(), tr_ldr, vl_ldr, device, st, lr, wd, name)
            r["dataset"] = ds_name; all_results.append(r)
            print(f"  {name:25s} P={r['params']:>8,} acc={r['val_acc']:.4f} {r['seconds']:.0f}s")

    print(f"\n{'='*55}")
    print(f"  PHASE 1 RESULTS")
    print(f"{'='*55}")
    print(f"{'Model':<25s} {'Params':>8s} {'MNIST':>8s} {'FMNIST':>8s}")
    print("-" * 53)
    mr = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fr = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    for n in ["Baseline", "V2-S(h=8,nb=3)", "V2-B(h=16,nb=4)", "V2-L(h=24,nb=6)"]:
        m = mr.get(n, {}); f = fr.get(n, {})
        print(f"{n:<25s} {m.get('params',f.get('params',0)):>8,} {m.get('val_acc',0):.4f}     {f.get('val_acc',0):.4f}")

    json.dump(all_results, open(str(ROOT / "experiments_v2/results/phase1_v2.json"), "w"), indent=2)
