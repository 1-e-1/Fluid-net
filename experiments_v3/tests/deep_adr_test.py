import sys, time, json, random
from pathlib import Path
from itertools import cycle
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import *

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

# ============================================================
# ★ 新架构: 无Q压缩, 纯ADR场演化, 同维度变换, 可多层堆叠
# ============================================================

class DeepADRNet(nn.Module):
    """
    移除Q压缩, 每层都是同维度 ADR 场变换:
      Input → Fourier → [ADR(n×n×h)]×N → GlobalPool → MLP Head
    
    关键: 不再使用 FluidResidualSequenceNetwork 的完整管线,
          只用 encode_nonsequential (纯ADR) + 自定义head
    """
    def __init__(self, n=28, h=8, num_layers=3, num_classes=10, adr_steps=6, dropout=0.15):
        super().__init__()
        self.n, self.h = n, h
        
        # 输入投影
        self.proj = nn.Linear(1, h)
        self.pos = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        
        # ★ 多层ADR (无Q压缩!)
        # 每层创建一个独立的 FluidNet 实例, 只用它的 encode_nonsequential
        self.adr_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.channel_mixes = nn.ModuleList()
        
        for i in range(num_layers):
            # 每层独立的FluidNet(只用ADR)
            fluid = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=n, h=h, output_dim=h,  # output_dim 不用,但必须设置
                    max_output_steps=1, static_adr_steps=adr_steps,
                    q_channels=16, q_depth=2, q_height=2, q_width=1,
                    extractor_hidden_channels=32, reconstructor_hidden_channels=32,
                    output_hidden_dim=64, adr_hidden_channels=48,
                ))
            self.adr_layers.append(fluid)
            self.layer_norms.append(nn.LayerNorm([n, n, h]))
            self.channel_mixes.append(ChannelMix(n, h, expansion=2))
        
        # 最终输出头 (宽MLP)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h, h * 6), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 6, h * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 4, h * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes),
        )
    
    def forward(self, x):
        # 预处理
        f = self.proj(x) + self.pos
        f = f + self.fourier_norm(self.fourier(f))
        
        # ★ 多层ADR: 每层同维度变换, 无信息瓶颈
        for adr, norm, cm in zip(self.adr_layers, self.layer_norms, self.channel_mixes):
            # ADR 场演化 (无Q压缩!)
            f = adr.encode_nonsequential(f).squeeze(1)  # [B, n, n, h]
            # 归一化 + 通道混合 (同维度)
            f = norm(f)
            f = cm(f)
        
        # 输出
        f = f.permute(0, 3, 1, 2)  # [B, h, n, n]
        return self.head(self.pool(f))


class DeepADRNet_Wide(nn.Module):
    """宽通道版: h=12, 适合更复杂任务"""
    def __init__(self, n=28, h=12, num_layers=3, num_classes=10, adr_steps=5, dropout=0.12):
        super().__init__()
        self.n, self.h = n, h
        
        self.proj = nn.Linear(1, h)
        self.pos = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        
        self.adr_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.channel_mixes = nn.ModuleList()
        
        for i in range(num_layers):
            fluid = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=n, h=h, output_dim=h,
                    max_output_steps=1, static_adr_steps=adr_steps,
                    q_channels=16, q_depth=2, q_height=2, q_width=1,
                    extractor_hidden_channels=32, reconstructor_hidden_channels=32,
                    output_hidden_dim=64, adr_hidden_channels=64,
                ))
            self.adr_layers.append(fluid)
            self.layer_norms.append(nn.LayerNorm([n, n, h]))
            self.channel_mixes.append(ChannelMix(n, h, expansion=3))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h, h * 6), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 6, h * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 4, h * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes),
        )

    def forward(self, x):
        f = self.proj(x) + self.pos
        f = f + self.fourier_norm(self.fourier(f))
        for adr, norm, cm in zip(self.adr_layers, self.layer_norms, self.channel_mixes):
            f = adr.encode_nonsequential(f).squeeze(1)
            f = norm(f)
            f = cm(f)
        f = f.permute(0, 3, 1, 2)
        return self.head(self.pool(f))


# ============================================================
# Training
# ============================================================
def train_eval_dadr(model, tr_ldr, vl_ldr, device, steps=600, lr=1e-3, wd=1e-4, name="", nc=10):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    crit = LabelSmoothingLoss(nc, 0.05)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()
    
    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        fld = img2field(x)
        if hasattr(model, 'encode_nonsequential'):
            logits = model(fld, mode="nonsequential", max_output_steps=1)
            logits = logits.outputs[:, 0] if hasattr(logits, 'outputs') else logits
        else:
            logits = model(fld)
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
            if hasattr(model, 'encode_nonsequential'):
                logits = model(fld, mode="nonsequential", max_output_steps=1)
                logits = logits.outputs[:, 0] if hasattr(logits, 'outputs') else logits
            else:
                logits = model(fld)
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(-1) == y).sum().cpu())
            vt += y.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc/vt, "val_loss": vs/vt, "seconds": sec}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    set_seed(42)
    print("=" * 65)
    print("  DeepADRNet: NO Q-COMPRESSION, MULTI-LAYER ADR")
    print("=" * 65)
    
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    all_results = []
    
    for ds_name, ds_cls in [("MNIST", datasets.MNIST), ("FashionMNIST", datasets.FashionMNIST)]:
        tr_size = 3000 if ds_name == "MNIST" else 5000
        tr = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=t), tr_size)
        vl = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=False, download=False, transform=t), 1000)
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)
        
        print(f"\n{'='*55} {ds_name}")
        
        # Baseline
        set_seed(42)
        b = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(
            n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
            q_channels=32, q_depth=7, q_height=7, q_width=1,
            extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))
        r = train_eval_dadr(b, tr_ldr, vl_ldr, device, steps=700, lr=1e-3, wd=1e-4,
                            name=f"{ds_name[:4]}-Baseline(Q)", nc=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  Baseline(Q-compress):  acc={r['val_acc']:.4f} P={r['params']:,}")
        
        # DeepADR h=8, 1 layer
        set_seed(42)
        m = DeepADRNet(n=28, h=8, num_layers=1, num_classes=10, adr_steps=6)
        r = train_eval_dadr(m, tr_ldr, vl_ldr, device, steps=700, lr=1.5e-3, wd=2e-4,
                            name=f"{ds_name[:4]}-DeepADR(1L,h=8)", nc=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  DeepADR(1L,h=8):       acc={r['val_acc']:.4f} P={r['params']:,}")
        
        # DeepADR h=8, 3 layers
        set_seed(42)
        m = DeepADRNet(n=28, h=8, num_layers=3, num_classes=10, adr_steps=6)
        r = train_eval_dadr(m, tr_ldr, vl_ldr, device, steps=700, lr=1.5e-3, wd=2e-4,
                            name=f"{ds_name[:4]}-DeepADR(3L,h=8)", nc=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  DeepADR(3L,h=8):       acc={r['val_acc']:.4f} P={r['params']:,}")
        
        # DeepADR h=12, 3 layers
        set_seed(42)
        m = DeepADRNet_Wide(n=28, h=12, num_layers=3, num_classes=10, adr_steps=5)
        r = train_eval_dadr(m, tr_ldr, vl_ldr, device, steps=700, lr=1.5e-3, wd=2e-4,
                            name=f"{ds_name[:4]}-DeepADR(3L,h=12)", nc=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  DeepADR(3L,h=12):      acc={r['val_acc']:.4f} P={r['params']:,}")
        
        # DeepADR h=12, 5 layers
        if ds_name == "MNIST":
            set_seed(42)
            m = DeepADRNet_Wide(n=28, h=12, num_layers=5, num_classes=10, adr_steps=5)
            r = train_eval_dadr(m, tr_ldr, vl_ldr, device, steps=700, lr=1.5e-3, wd=3e-4,
                                name=f"{ds_name[:4]}-DeepADR(5L,h=12)", nc=10)
            r["dataset"] = ds_name; all_results.append(r)
            print(f"  DeepADR(5L,h=12):      acc={r['val_acc']:.4f} P={r['params']:,}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  DeepADRNet RESULTS (NO Q-COMPRESSION!)")
    print("=" * 70)
    print(f"{'Model':<25s} {'Layers':>6s} {'MNIST':>8s} {'FMNIST':>8s} {'Params':>10s}")
    print("-" * 62)
    mnist = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fmnist = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    for name in sorted(mnist.keys()):
        m = mnist[name]; f = fmnist.get(name, {})
        nlayers = name.count("L,")
        layers_str = name.split("(")[1].split(",")[0] if "(" in name else "1"
        print(f"{name:<25s} {layers_str:>6s} {m['val_acc']:.4f}     {f.get('val_acc',0):.4f}     {m['params']:>10,}")
    
    out = ROOT / "experiments_v3" / "results" / "deep_adr_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
