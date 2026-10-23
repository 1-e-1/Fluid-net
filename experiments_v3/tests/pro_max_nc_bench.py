# -*- coding: utf-8 -*-
"""
experiments_v3/tests/pro_max_nc_bench.py
=========================================
Lightweight performance check for ProMaxNC (merged architecture).

Safety (avoid blue screen / OOM):
  - cap this process GPU memory to ~25% of the device
  - small batch sizes, short training steps
  - monitor peak CUDA memory

Tasks:
  1. MNIST classification (field n=28)
  2. Multimodal digit recognition (image stream + text stream, fused)

Compare with recorded baselines:
  MNIST : Baseline-Q 98.6% | NC-L4 98.4% | NC-L8 98.7%
  MM    : Baseline-Q 89.44% | NC-L4 99.17% | NC-L8 99.17%
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path
from itertools import cycle

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

from improved_fluid import set_seed, CosineWarmupScheduler
from pro_max_nc import ProMaxNC, ProMaxNCStacked, make_config

# ---------- memory safety ----------
torch.cuda.set_per_process_memory_fraction(0.25)
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

device = torch.device("cuda:0")


# ==================== MNIST wrapper ====================
class ProMaxNCImage(nn.Module):
    """image [B,1,n,n] -> linear proj -> field -> ProMaxNC -> logits"""
    def __init__(self, config, pool_size=4, fourier=True):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(1, config.h)
        self.pos = nn.Parameter(torch.randn(1, config.n, config.n, config.h) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, config.n, config.n, config.h))
        self.net = ProMaxNC(config, fourier=fourier, pool_size=pool_size)

    def forward(self, img):
        B = img.shape[0]
        emb = self.proj(img.reshape(B, -1, 1))           # [B,T,h]
        field = self.bias.expand(B, -1, -1, -1).clone()
        T = emb.shape[1]
        rows = torch.arange(T, device=img.device) // self.config.n
        cols = torch.arange(T, device=img.device) % self.config.n
        field[:, rows, cols, :] = emb + self.pos[:, rows, cols, :]
        return self.net(field, mode="nonsequential", max_output_steps=1).outputs[:, 0]


# ==================== Multimodal wrapper ====================
class ProMaxNCMM(nn.Module):
    """image stream + text stream -> two ProMaxNC -> fuse -> logits"""
    def __init__(self, n=8, h=8, steps=4, num_classes=10, pool_size=2, out_feat=128):
        super().__init__()
        self.img_proj = nn.Linear(1, h)
        self.txt_proj = nn.Linear(4, h)
        cfg_img = make_config(n=n, h=h, output_dim=out_feat, static_adr_steps=steps,
                              max_output_steps=1, output_hidden_dim=out_feat)
        cfg_txt = make_config(n=n, h=h, output_dim=out_feat, static_adr_steps=steps,
                              max_output_steps=1, output_hidden_dim=out_feat)
        self.img_net = ProMaxNC(cfg_img, fourier=True, pool_size=pool_size)
        self.txt_net = ProMaxNC(cfg_txt, fourier=True, pool_size=pool_size)
        self.fuse = nn.Sequential(
            nn.Linear(out_feat * 2, out_feat * 2), nn.GELU(),
            nn.Linear(out_feat * 2, out_feat), nn.GELU(),
            nn.Linear(out_feat, num_classes),
        )

    def forward(self, img, txt):
        B = img.shape[0]
        img_f = self.img_proj(img.reshape(B, 8, 8, 1))
        txt_f = self.txt_proj(txt.view(B, 8, 8, 4))
        fi = self.img_net(img_f, mode="nonsequential", max_output_steps=1).outputs[:, 0]
        ft = self.txt_net(txt_f, mode="nonsequential", max_output_steps=1).outputs[:, 0]
        return self.fuse(torch.cat([fi, ft], dim=-1))


class StackedNCImage(nn.Module):
    """stacked PMNC: image -> field -> L independent pro_max ADR blocks -> readout"""
    def __init__(self, config, num_layers=4, pool_size=4, fourier=True):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(1, config.h)
        self.pos = nn.Parameter(torch.randn(1, config.n, config.n, config.h) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, config.n, config.n, config.h))
        self.net = ProMaxNCStacked(config, num_layers=num_layers,
                                   fourier=fourier, pool_size=pool_size)

    def forward(self, img):
        B = img.shape[0]
        emb = self.proj(img.reshape(B, -1, 1))
        field = self.bias.expand(B, -1, -1, -1).clone()
        T = emb.shape[1]
        rows = torch.arange(T, device=img.device) // self.config.n
        cols = torch.arange(T, device=img.device) % self.config.n
        field[:, rows, cols, :] = emb + self.pos[:, rows, cols, :]
        return self.net(field, mode="nonsequential", max_output_steps=1).outputs[:, 0]


class StackedNCMM(nn.Module):
    """stacked PMNC multimodal: image stream + text stream -> fuse"""
    def __init__(self, n=8, h=8, num_layers=4, num_classes=10, pool_size=2, out_feat=128):
        super().__init__()
        self.img_proj = nn.Linear(1, h)
        self.txt_proj = nn.Linear(4, h)
        cfg_img = make_config(n=n, h=h, output_dim=out_feat, static_adr_steps=1,
                              max_output_steps=1, output_hidden_dim=out_feat)
        cfg_txt = make_config(n=n, h=h, output_dim=out_feat, static_adr_steps=1,
                              max_output_steps=1, output_hidden_dim=out_feat)
        self.img_net = ProMaxNCStacked(cfg_img, num_layers=num_layers, fourier=True, pool_size=pool_size)
        self.txt_net = ProMaxNCStacked(cfg_txt, num_layers=num_layers, fourier=True, pool_size=pool_size)
        self.fuse = nn.Sequential(
            nn.Linear(out_feat * 2, out_feat * 2), nn.GELU(),
            nn.Linear(out_feat * 2, out_feat), nn.GELU(),
            nn.Linear(out_feat, num_classes),
        )

    def forward(self, img, txt):
        B = img.shape[0]
        img_f = self.img_proj(img.reshape(B, 8, 8, 1))
        txt_f = self.txt_proj(txt.view(B, 8, 8, 4))
        fi = self.img_net(img_f, mode="nonsequential", max_output_steps=1).outputs[:, 0]
        ft = self.txt_net(txt_f, mode="nonsequential", max_output_steps=1).outputs[:, 0]
        return self.fuse(torch.cat([fi, ft], dim=-1))


# ==================== training helpers ====================
def train_cls(model, dtr, dva, steps, lr, name=""):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    it = cycle(dtr)
    t0 = time.perf_counter()
    for _ in range(steps):
        model.train()
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    ok, cnt = 0, 0
    with torch.no_grad():
        for x, y in dva:
            x, y = x.to(device), y.to(device)
            ok += int((model(x).argmax(-1) == y).sum().cpu())
            cnt += y.numel()
    return {"acc": ok / cnt, "seconds": time.perf_counter() - t0,
            "params": sum(p.numel() for p in model.parameters()),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}


def train_mm(model, dtr, dva, steps, lr, name=""):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    it = cycle(dtr)
    t0 = time.perf_counter()
    for _ in range(steps):
        model.train()
        xi, xt, y = next(it)
        xi, xt, y = xi.to(device), xt.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xi, xt), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    ok, cnt = 0, 0
    with torch.no_grad():
        for xi, xt, y in dva:
            xi, xt, y = xi.to(device), xt.to(device), y.to(device)
            ok += int((model(xi, xt).argmax(-1) == y).sum().cpu())
            cnt += y.numel()
    return {"acc": ok / cnt, "seconds": time.perf_counter() - t0,
            "params": sum(p.numel() for p in model.parameters()),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}


# ==================== data ====================
def mnist_loaders(batch=32):
    tr = datasets.MNIST(ROOT / "data", train=True, download=False,
                        transform=transforms.ToTensor())
    va = datasets.MNIST(ROOT / "data", train=False, download=False,
                        transform=transforms.ToTensor())
    return DataLoader(tr, batch, shuffle=True, drop_last=True), DataLoader(va, 256)


def mm_loaders(batch=32):
    from sklearn.datasets import load_digits
    data = load_digits()
    X_img = data.images.astype(np.float32) / 16.0 - 0.5
    y = data.target.astype(np.int64)
    n = len(y)
    txt_dim = 4
    X_txt = np.zeros((n, 64, txt_dim), dtype=np.float32)
    rng = np.random.default_rng(0)
    for i in range(n):
        for j in range(64):
            X_txt[i, j, int(y[i]) % txt_dim] = rng.normal() * 0.1 + 0.5
    idx = np.random.default_rng(42).permutation(n)
    split = int(n * 0.8)
    tr_idx, vl_idx = idx[:split], idx[split:]
    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_img[tr_idx]), torch.from_numpy(X_txt[tr_idx]),
        torch.from_numpy(y[tr_idx]))
    vl_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_img[vl_idx]), torch.from_numpy(X_txt[vl_idx]),
        torch.from_numpy(y[vl_idx]))
    return DataLoader(tr_ds, batch, shuffle=True, drop_last=True), DataLoader(vl_ds, 64)


def _save(report):
    out = ROOT / "experiments_v3" / "results" / "pro_max_nc_bench.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)


# ==================== main ====================
def main():
    only_stacked = "--only-stacked" in sys.argv
    set_seed(42)
    report = {}
    if (ROOT / "experiments_v3" / "results" / "pro_max_nc_bench.json").exists():
        with open(ROOT / "experiments_v3" / "results" / "pro_max_nc_bench.json", encoding="utf-8") as f:
            report = json.load(f)

    print("=" * 66)
    print("  ProMaxNC light benchmark  (GPU mem cap 25%, short steps)")
    print("=" * 66)

    # ---------- MNIST ----------
    print("\n----- MNIST (field 28x28, batch 32, 600 steps) -----")
    dtr, dva = mnist_loaders(32)
    if not only_stacked:
        for steps, name in [(4, "PMNC-L4"), (8, "PMNC-L8")]:
            torch.cuda.reset_peak_memory_stats()
            set_seed(42)
            cfg = make_config(n=28, h=8, output_dim=10, static_adr_steps=steps,
                              max_output_steps=1, adr_hidden_channels=48)
            m = ProMaxNCImage(cfg)
            r = train_cls(m, dtr, dva, 600, 1.2e-3, name)
            report.setdefault("mnist", {})[name] = r
            _save(report)
            print(f"  {name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
                  f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
            del m; torch.cuda.empty_cache()

    for layers, name in [(4, "PMNCStacked-L4"), (8, "PMNCStacked-L8")]:
        torch.cuda.reset_peak_memory_stats()
        set_seed(42)
        cfg = make_config(n=28, h=8, output_dim=10, static_adr_steps=1,
                          max_output_steps=1, adr_hidden_channels=48)
        m = StackedNCImage(cfg, num_layers=layers)
        r = train_cls(m, dtr, dva, 600, 1.2e-3, name)
        report.setdefault("mnist_stacked", {})[name] = r
        _save(report)
        print(f"  {name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
              f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
        del m; torch.cuda.empty_cache()

    # ---------- Multimodal ----------
    print("\n----- Multimodal digits (image+text, batch 32, 500 steps) -----")
    dtr, dva = mm_loaders(32)
    if not only_stacked:
        for steps, name in [(4, "PMNC-MM-L4"), (8, "PMNC-MM-L8")]:
            torch.cuda.reset_peak_memory_stats()
            set_seed(42)
            m = ProMaxNCMM(n=8, h=8, steps=steps)
            r = train_mm(m, dtr, dva, 500, 1.2e-3, name)
            report.setdefault("mm", {})[name] = r
            _save(report)
            print(f"  {name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
                  f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
            del m; torch.cuda.empty_cache()

    for layers, name in [(4, "PMNCStacked-MM-L4"), (8, "PMNCStacked-MM-L8")]:
        torch.cuda.reset_peak_memory_stats()
        set_seed(42)
        m = StackedNCMM(n=8, h=8, num_layers=layers)
        r = train_mm(m, dtr, dva, 500, 1.2e-3, name)
        report.setdefault("mm_stacked", {})[name] = r
        _save(report)
        print(f"  {name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
              f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
        del m; torch.cuda.empty_cache()

    # ---------- comparison ----------
    print("\n" + "=" * 66)
    print("  Comparison (recorded baselines)")
    print("=" * 66)
    mn = report["mnist"]; mm = report["mm"]
    rows = [
        ("MNIST", "Baseline-Q", 0.986, mn.get("PMNC-L4", {}).get("acc"), mn.get("PMNC-L8", {}).get("acc")),
        ("MNIST", "NC (no-compress)", 0.984, 0.987, None),
        ("MM",    "Baseline-Q", 0.8944, mm.get("PMNC-MM-L4", {}).get("acc"), mm.get("PMNC-MM-L8", {}).get("acc")),
        ("MM",    "NC (no-compress)", 0.9917, 0.9917, None),
    ]
    print(f"  {'Task':<5s} {'baseline':<20s} {'Base-Q':>8s} {'NC-L4':>8s} {'NC-L8':>8s}")
    for task, label, bq, n4, n8 in rows:
        p4 = report["mnist"]["PMNC-L4"]["acc"] if task == "MNIST" and label.startswith("Baseline") else (n4 if label.startswith("NC") else None)
        fmt = lambda v: f"{v*100:6.2f}%" if v is not None else "   n/a "
        if task == "MNIST":
            if label.startswith("Baseline"):
                print(f"  {task:<5s} {label:<20s} {fmt(bq):>8s} {fmt(p4):>8s} {fmt(mn['PMNC-L8']['acc']):>8s}")
            else:
                print(f"  {task:<5s} {label:<20s} {fmt(0.986):>8s} {fmt(0.984):>8s} {fmt(0.987):>8s}")
        else:
            if label.startswith("Baseline"):
                print(f"  {task:<5s} {label:<20s} {fmt(bq):>8s} {fmt(mm['PMNC-MM-L4']['acc']):>8s} {fmt(mm['PMNC-MM-L8']['acc']):>8s}")
            else:
                print(f"  {task:<5s} {label:<20s} {fmt(0.8944):>8s} {fmt(0.9917):>8s} {fmt(0.9917):>8s}")

    out = ROOT / "experiments_v3" / "results" / "pro_max_nc_bench.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()