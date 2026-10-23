# -*- coding: utf-8 -*-
"""
experiments_v3/tests/strict_compare.py
=======================================
Strict head-to-head: base-ADR stack (NC) vs pro_max-pure-ADR stack.

Everything identical (wrapper, Fourier, readout, config, steps, batch, seed);
the ONLY difference is the ADR block source:
  NC   : fluid_net.ADRBlock3D            (base)
  PURE : fluid_net_pro_max.ADRBlock3D    (enhancers disabled)
"""
from __future__ import annotations

import sys, json
from pathlib import Path

import torch, torch.nn as nn
import torch.nn.functional as F

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
sys.path.insert(0, str(ROOT / "experiments_v3" / "tests"))

import fluid_net as fn
import fluid_net_pro_max as pmax
from improved_fluid import set_seed, FourierGlobalMix
from pro_max_nc_bench import mnist_loaders, mm_loaders, train_cls, train_mm

device = torch.device("cuda:0")


# ---------------- configs (matched) ----------------
def base_cfg(n, h, out_dim):
    return fn.FluidResidualSequenceConfig(
        n=n, h=h, output_dim=out_dim, adr_hidden_channels=64, adr_step_size=0.30,
        static_adr_steps=1, max_output_steps=1, q_channels=16, q_depth=1,
        q_height=1, q_width=1, extractor_hidden_channels=32,
        reconstructor_hidden_channels=32, output_hidden_dim=256)


def pmax_cfg(n, h, out_dim):
    return pmax.FluidResidualSequenceConfig(
        n=n, h=h, output_dim=out_dim, adr_hidden_channels=64, adr_step_size=0.30,
        static_adr_steps=1, max_output_steps=1, q_channels=16, q_depth=1,
        q_height=1, q_width=1, extractor_hidden_channels=32,
        reconstructor_hidden_channels=32, output_hidden_dim=256,
        enable_step_conditioned_adr=False, enable_container_velocity=False,
        max_viscosity=0.0)


# ---------------- shared wrapper ----------------
class StrictImage(nn.Module):
    def __init__(self, cfg, adr_factory, num_layers=8, pool_size=4):
        super().__init__()
        self.n = cfg.n; self.h = cfg.h
        self.proj = nn.Linear(1, cfg.h)
        self.pos = nn.Parameter(torch.randn(1, cfg.n, cfg.n, cfg.h) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, cfg.n, cfg.n, cfg.h))
        self.fourier = FourierGlobalMix(cfg.n, cfg.h)
        self.fourier_norm = nn.LayerNorm([cfg.n, cfg.n, cfg.h])
        self.blocks = nn.ModuleList([adr_factory(cfg) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm([cfg.n, cfg.n, cfg.h]) for _ in range(num_layers)])
        self.pool_size = pool_size
        flat = pool_size * pool_size * cfg.h
        self.readout = pmax.ZeroPreservingMLP(flat, cfg.output_hidden_dim, cfg.output_dim)

    def forward(self, img):
        B = img.shape[0]
        emb = self.proj(img.reshape(B, -1, 1))
        field = self.bias.expand(B, -1, -1, -1).clone()
        T = emb.shape[1]
        rows = torch.arange(T, device=img.device) // self.n
        cols = torch.arange(T, device=img.device) % self.n
        field[:, rows, cols, :] = emb + self.pos[:, rows, cols, :]
        f4 = field + self.fourier_norm(self.fourier(field))
        for li, (adr, norm) in enumerate(zip(self.blocks, self.norms)):
            step_idx = torch.full((B, 1), float(li), dtype=torch.float32, device=img.device)
            f4 = norm(f4 + adr(f4.unsqueeze(1), step_idx).squeeze(1))
        pooled = F.adaptive_avg_pool2d(f4.permute(0, 3, 1, 2), (self.pool_size, self.pool_size))
        return self.readout(pooled.flatten(1))


class StrictMM(nn.Module):
    def __init__(self, cfg, adr_factory, num_layers=8, out_feat=128):
        super().__init__()
        self.img_proj = nn.Linear(1, cfg.h)
        self.txt_proj = nn.Linear(4, cfg.h)
        self.img_net = StrictImage(cfg, adr_factory, num_layers, pool_size=2)
        self.txt_net = StrictImage(cfg, adr_factory, num_layers, pool_size=2)
        self.fuse = nn.Sequential(
            nn.Linear(out_feat * 2, out_feat * 2), nn.GELU(),
            nn.Linear(out_feat * 2, out_feat), nn.GELU(),
            nn.Linear(out_feat, 10),
        )
        # align output dim of inner nets to out_feat
        self.img_net.readout = pmax.ZeroPreservingMLP(2 * 2 * cfg.h, out_feat, out_feat)
        self.txt_net.readout = pmax.ZeroPreservingMLP(2 * 2 * cfg.h, out_feat, out_feat)

    def forward(self, img, txt):
        B = img.shape[0]
        img_f = self.img_proj(img.reshape(B, 8, 8, 1))
        txt_f = self.txt_proj(txt.view(B, 8, 8, 4))
        # reuse StrictImage forward on already-projected fields
        fi = self._run_stream(self.img_net, img_f)
        ft = self._run_stream(self.txt_net, txt_f)
        return self.fuse(torch.cat([fi, ft], dim=-1))

    def _run_stream(self, net, f4):
        B = f4.shape[0]
        f4 = f4 + net.fourier_norm(net.fourier(f4))
        for li, (adr, norm) in enumerate(zip(net.blocks, net.norms)):
            step_idx = torch.full((B, 1), float(li), dtype=torch.float32, device=f4.device)
            f4 = norm(f4 + adr(f4.unsqueeze(1), step_idx).squeeze(1))
        pooled = F.adaptive_avg_pool2d(f4.permute(0, 3, 1, 2), (net.pool_size, net.pool_size))
        return net.readout(pooled.flatten(1))


# ---------------- main ----------------
def save(report):
    out = ROOT / "experiments_v3" / "results" / "strict_compare.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)


def main():
    only_mm = "--only-mm" in sys.argv
    set_seed(42)
    report = {}

    if not only_mm:
        print("===== MNIST strict compare (L=8, h=8, 1200 steps) =====", flush=True)
        dtr, dva = mnist_loaders(32)
        for name, cfg_fn, adr_fn in [("NC-base", base_cfg, fn.ADRBlock3D),
                                     ("PMNC-pure", pmax_cfg, pmax.ADRBlock3D)]:
            torch.cuda.reset_peak_memory_stats()
            set_seed(42)
            m = StrictImage(cfg_fn(28, 8, 10), adr_fn, num_layers=8)
            r = train_cls(m, dtr, dva, 1200, 1.2e-3, name)
            r["params"] = sum(p.numel() for p in m.parameters())
            r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            report.setdefault("mnist", {})[name] = r
            save(report)
            print(f"  {name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
                  f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
            del m; torch.cuda.empty_cache()

    print("===== Multimodal strict compare (L=8, h=16, 1000 steps) =====", flush=True)
    dtr, dva = mm_loaders(32)
    for name, cfg_fn, adr_fn in [("NC-base", base_cfg, fn.ADRBlock3D),
                                 ("PMNC-pure", pmax_cfg, pmax.ADRBlock3D)]:
        torch.cuda.reset_peak_memory_stats()
        set_seed(42)
        m = StrictMM(cfg_fn(8, 16, 128), adr_fn, num_layers=8)
        r = train_mm(m, dtr, dva, 1000, 1.2e-3, name)
        r["params"] = sum(p.numel() for p in m.parameters())
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        report.setdefault("mm", {})[name] = r
        save(report)
        print(f"  {name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
              f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
        del m; torch.cuda.empty_cache()

    print("\n===== Summary =====", flush=True)
    for task in ["mnist", "mm"]:
        for name, r in report.get(task, {}).items():
            print(f"  {task}-{name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
                  f"peak={r['peak_gb']:.2f}GB", flush=True)
    save(report)
    print(f"Saved: {ROOT / 'experiments_v3' / 'results' / 'strict_compare.json'}", flush=True)


if __name__ == "__main__":
    main()