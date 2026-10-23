# -*- coding: utf-8 -*-
"""
experiments_v3/tests/temporal_suite.py
=======================================
Complex temporal task suite: NC (no-compress) vs LSTM vs Transformer

Tasks:
  1. Adding problem (T=500)          - long-range selection / memory (regression)
  2. Exchange-rate forecast          - real data, lookback 96 -> horizon 24 (regression)
  3. Sequential FashionMNIST (784)   - long-sequence classification
"""
from __future__ import annotations

import sys, time, json, math
from pathlib import Path
from itertools import cycle
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import set_seed, FourierGlobalMix, CosineWarmupScheduler

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")


def make_block(n, h, adr_hidden=32, step_size=0.30):
    cfg = fn.FluidResidualSequenceConfig(
        n=n, h=h, output_dim=h, max_output_steps=1, static_adr_steps=1,
        adr_hidden_channels=adr_hidden, adr_step_size=step_size,
        q_channels=8, q_depth=1, q_height=1, q_width=1,
        extractor_hidden_channels=8, reconstructor_hidden_channels=8, output_hidden_dim=64)
    return fn.ADRBlock3D(cfg)


# ==================== Models ====================
class LSTMSeq(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class TransSeq(nn.Module):
    def __init__(self, in_dim, d=64, nhead=4, nlayers=3, out_dim=1, max_len=1024, pool="mean"):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=256, dropout=0.1,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.pool = pool
        self.head = nn.Linear(d, out_dim)
    def forward(self, x):
        z = self.proj(x) + self.pos[:, :x.shape[1]]
        z = self.enc(z)
        if self.pool == "mean":
            z = z.mean(1)
        else:
            z = z[:, -1]
        return self.head(z)


class NCSeq(nn.Module):
    """Sequence -> field (row-major) -> Fourier + same-dim ADR stack -> global pool -> head"""
    def __init__(self, in_dim, n, h=8, num_layers=4, out_dim=1,
                 pool_size=4, out_feat=128, adr_hidden=32, step_size=0.30, dropout=0.1):
        super().__init__()
        self.n = n
        self.embed = nn.Linear(in_dim, h)
        self.pos = nn.Parameter(torch.randn(1, n, n, h) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, n, n, h))
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.adr_layers = nn.ModuleList([make_block(n, h, adr_hidden, step_size) for _ in range(num_layers)])
        self.adr_norms = nn.ModuleList([nn.LayerNorm([n, n, h]) for _ in range(num_layers)])
        flat = pool_size * pool_size * h
        self.pool_size = pool_size
        self.head = nn.Sequential(
            nn.Linear(flat, out_feat), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_feat, out_dim))
    def forward(self, x):
        B, T, C = x.shape
        emb = self.embed(x)                                    # [B,T,h]
        field = self.bias.expand(B, -1, -1, -1).clone()
        rows = torch.arange(T, device=x.device) // self.n
        cols = torch.arange(T, device=x.device) % self.n
        field[:, rows, cols, :] = emb + self.pos[:, rows, cols, :]
        f = field + self.fourier_norm(self.fourier(field))
        f5 = f.unsqueeze(1)
        for li, (adr, norm) in enumerate(zip(self.adr_layers, self.adr_norms)):
            step_idx = torch.full((B, 1), li, dtype=torch.long, device=x.device)
            f5 = norm((f5 + adr(f5, step_idx)).squeeze(1)).unsqueeze(1)
        pooled = F.adaptive_avg_pool2d(
            f5.squeeze(1).permute(0, 3, 1, 2), (self.pool_size, self.pool_size))
        return self.head(pooled.flatten(1))


# ==================== Train helpers ====================
def run_reg(model, dl_tr, dl_va, steps, lr, name="", succ_thr=None):
    """regression: MSE (+ optional success rate for adding problem)"""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    it = cycle(dl_tr)
    t0 = time.perf_counter()
    for _ in range(steps):
        model.train()
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x).view(*y.shape), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    mse, mae, succ, cnt = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for x, y in dl_va:
            x, y = x.to(device), y.to(device)
            p = model(x).view(*y.shape)
            mse += float(F.mse_loss(p, y).item()) * y.shape[0]
            mae += float(F.l1_loss(p, y).item()) * y.shape[0]
            if succ_thr is not None:
                succ += int(((p - y).abs() < succ_thr).all(dim=tuple(range(1, p.ndim))).sum().cpu())
            cnt += y.shape[0]
    return {"mse": mse / cnt, "mae": mae / cnt, "succ": succ / cnt,
            "seconds": time.perf_counter() - t0}


def run_cls(model, dl_tr, dl_va, steps, lr, name=""):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    it = cycle(dl_tr)
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
        for x, y in dl_va:
            x, y = x.to(device), y.to(device)
            ok += int((model(x).argmax(-1) == y).sum().cpu())
            cnt += y.numel()
    return {"acc": ok / cnt, "seconds": time.perf_counter() - t0}


def params_of(m):
    return sum(p.numel() for p in m.parameters())


# ==================== Task 1: Adding problem ====================
def make_adding(n_tr, n_va, T=500, seed=42):
    rng = np.random.default_rng(seed)
    def gen(n):
        X = np.zeros((n, T, 2), dtype=np.float32)
        v = rng.uniform(-1, 1, size=(n, T)).astype(np.float32)
        X[:, :, 0] = v
        y = np.zeros(n, dtype=np.float32)
        for i in range(n):
            j = rng.choice(T, size=2, replace=False)
            X[i, j, 1] = 1.0
            y[i] = v[i, j].sum()
        return X, y
    X1, y1 = gen(n_tr); X2, y2 = gen(n_va)
    tr = TensorDataset(torch.from_numpy(X1), torch.from_numpy(y1))
    va = TensorDataset(torch.from_numpy(X2), torch.from_numpy(y2))
    return tr, va


# ==================== Task 2: Exchange rate ====================
def make_exchange(lookback=96, horizon=24, stride=4, seed=42):
    raw = np.genfromtxt(ROOT / "data" / "exchange_rate" / "exchange_rate.csv",
                        delimiter=",", skip_header=1)
    raw = np.asarray(raw, dtype=np.float32)          # [N, 8]
    N = raw.shape[0]
    tr_end, va_end = int(N * 0.7), int(N * 0.8)
    def windows(a, b):
        xs, ys = [], []
        for s in range(a, b - lookback - horizon + 1, stride):
            xs.append(raw[s:s + lookback]); ys.append(raw[s + lookback:s + lookback + horizon])
        return np.stack(xs), np.stack(ys)
    X_tr, Y_tr = windows(0, tr_end)
    X_va, Y_va = windows(tr_end, va_end)
    # normalize by train stats per series
    mu = X_tr.reshape(-1, 8).mean(0, keepdims=True)
    sd = X_tr.reshape(-1, 8).std(0, keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sd; Y_tr = (Y_tr - mu) / sd
    X_va = (X_va - mu) / sd; Y_va = (Y_va - mu) / sd
    tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va))
    return tr, va, 8, horizon


# ==================== Task 3: Sequential FashionMNIST ====================
def load_fmnist_idx(root):
    raw = root / "FashionMNIST" / "raw"
    def rd(img, lbl):
        with open(raw / img, "rb") as f:
            d = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        with open(raw / lbl, "rb") as f:
            l = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return d, l
    Xtr, ytr = rd("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    Xva, yva = rd("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    return Xtr, ytr, Xva, yva


def make_sfmnist(n_tr=12000, n_va=2000, seed=42):
    Xtr, ytr, Xva, yva = load_fmnist_idx(ROOT / "data")
    rng = np.random.default_rng(seed)
    i_tr = rng.choice(len(Xtr), n_tr, replace=False)
    i_va = rng.choice(len(Xva), n_va, replace=False)
    Xtr = (Xtr[i_tr].astype(np.float32) / 255.0).reshape(n_tr, 784, 1)
    Xva = (Xva[i_va].astype(np.float32) / 255.0).reshape(n_va, 784, 1)
    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr[i_tr].astype(np.int64)))
    va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva[i_va].astype(np.int64)))
    return tr, va


# ==================== Main ====================
def main():
    set_seed(42)
    report = {}

    # ---- Task 1: Adding problem ----
    print("=" * 68)
    print("  TASK 1: Adding problem (T=500, regression, target in [-2,2])")
    print("=" * 68)
    tr, va = make_adding(6000, 2000, T=500)
    dtr = DataLoader(tr, 128, shuffle=True, drop_last=True)
    dva = DataLoader(va, 256)
    models = [
        ("LSTM",      LSTMSeq(2, 64, 1)),
        ("Transformer", TransSeq(2, 64, 4, 3, 1)),
        ("NC-L4",     NCSeq(2, 24, 8, 4, 1, pool_size=4)),
        ("NC-L8",     NCSeq(2, 24, 8, 8, 1, pool_size=4)),
    ]
    for name, m in models:
        set_seed(42)
        r = run_reg(m, dtr, dva, 1000, 1.2e-3, name)
        r["params"] = params_of(m)
        report.setdefault("adding", {})[name] = r
        print(f"  {name:<12s} MSE={r['mse']:.5f} MAE={r['mae']:.4f} succ(<0.04)={r['succ']*100:5.1f}% P={r['params']:,} {r['seconds']:.0f}s", flush=True)

    with open(ROOT / "experiments_v3" / "results" / "temporal_suite.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  [incremental saved after adding]", flush=True)

    # ---- Task 2: Exchange rate ----
    print()
    print("=" * 68)
    print("  TASK 2: Exchange-rate forecast (lookback 96 -> horizon 24, 8 series)")
    print("=" * 68)
    tr, va, n_series, horizon = make_exchange()
    dtr = DataLoader(tr, 128, shuffle=True, drop_last=True)
    dva = DataLoader(va, 256)
    out_dim = n_series * horizon
    models = [
        ("LSTM",      LSTMSeq(n_series, 64, out_dim)),
        ("Transformer", TransSeq(n_series, 64, 4, 3, out_dim)),
        ("NC-L4",     NCSeq(n_series, 10, 8, 4, out_dim, pool_size=4)),
        ("NC-L8",     NCSeq(n_series, 10, 8, 8, out_dim, pool_size=4)),
    ]
    for name, m in models:
        set_seed(42)
        r = run_reg(m, dtr, dva, 1000, 1.2e-3, name)
        r["params"] = params_of(m)
        report.setdefault("exchange", {})[name] = r
        print(f"  {name:<12s} MSE={r['mse']:.5f} MAE={r['mae']:.4f} P={r['params']:,} {r['seconds']:.0f}s", flush=True)

    with open(ROOT / "experiments_v3" / "results" / "temporal_suite.json", "w") as f:
        json.dump(report, f, indent=2)
    print("  [incremental saved after exchange]", flush=True)

    # ---- Task 3: Sequential FashionMNIST ----
    print()
    print("=" * 68)
    print("  TASK 3: Sequential FashionMNIST (784 steps, 10 classes)")
    print("=" * 68)
    tr, va = make_sfmnist()
    dtr = DataLoader(tr, 64, shuffle=True, drop_last=True)
    dva = DataLoader(va, 256)
    models = [
        ("LSTM",      LSTMSeq(1, 64, 10)),
        ("Transformer", TransSeq(1, 64, 4, 3, 10)),
        ("NC-L4",     NCSeq(1, 28, 8, 4, 10, pool_size=4)),
        ("NC-L8",     NCSeq(1, 28, 8, 8, 10, pool_size=4)),
    ]
    for name, m in models:
        set_seed(42)
        r = run_cls(m, dtr, dva, 1500, 1.2e-3, name)
        r["params"] = params_of(m)
        report.setdefault("sfmnist", {})[name] = r
        print(f"  {name:<12s} Acc={r['acc']*100:6.2f}% P={r['params']:,} {r['seconds']:.0f}s", flush=True)

    out = ROOT / "experiments_v3" / "results" / "temporal_suite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
