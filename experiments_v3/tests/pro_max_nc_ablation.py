# -*- coding: utf-8 -*-
"""
experiments_v3/tests/pro_max_nc_ablation.py
============================================
Capacity-matched + kernel ablation training for the STACKED ProMaxNC.

Kernels (pro_max inner enhancements inside the NC stacked shell):
  full : step-conditioned ADR + container velocity + viscosity
  pure : all enhancers OFF  (identical to base ADR in NC shell)
  visc : only viscosity ON

Tasks: MNIST (L=8, h=8, adr_hidden=64, 1200 steps) and Multimodal (L=8, h=16, adr_hidden=64, 1000 steps).
Reference (recorded): NC-L8 MNIST 98.70% | NC-L4/L8 MM 99.17%.
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path

import torch, torch.nn as nn

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
sys.path.insert(0, str(ROOT / "experiments_v3" / "tests"))

from improved_fluid import set_seed
from pro_max_nc import ProMaxNCStacked, make_config
from pro_max_nc_bench import mnist_loaders, mm_loaders, train_cls, train_mm

device = torch.device("cuda:0")


def make_kernel_config(n, h, output_dim, kernel, adr_hidden=64, out_hidden=256):
    kw = dict(n=n, h=h, output_dim=output_dim, static_adr_steps=1,
              max_output_steps=1, adr_hidden_channels=adr_hidden,
              output_hidden_dim=out_hidden)
    if kernel == "full":
        kw.update(enable_step_conditioned_adr=True, enable_container_velocity=True, max_viscosity=0.1)
    elif kernel == "pure":
        kw.update(enable_step_conditioned_adr=False, enable_container_velocity=False, max_viscosity=0.0)
    elif kernel == "visc":
        kw.update(enable_step_conditioned_adr=False, enable_container_velocity=False, max_viscosity=0.1)
    else:
        raise ValueError(kernel)
    return make_config(**kw)


class AbImage(nn.Module):
    """image -> field -> stacked ProMaxNC -> logits"""
    def __init__(self, config, num_layers=8):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(1, config.h)
        self.pos = nn.Parameter(torch.randn(1, config.n, config.n, config.h) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, config.n, config.n, config.h))
        self.net = ProMaxNCStacked(config, num_layers=num_layers, fourier=True, pool_size=4)

    def forward(self, img):
        B = img.shape[0]
        emb = self.proj(img.reshape(B, -1, 1))
        field = self.bias.expand(B, -1, -1, -1).clone()
        T = emb.shape[1]
        rows = torch.arange(T, device=img.device) // self.config.n
        cols = torch.arange(T, device=img.device) % self.config.n
        field[:, rows, cols, :] = emb + self.pos[:, rows, cols, :]
        return self.net(field, mode="nonsequential", max_output_steps=1).outputs[:, 0]


class AbMM(nn.Module):
    """two stacked ProMaxNC streams -> fuse -> logits"""
    def __init__(self, cfg_img, cfg_txt, num_layers=8, out_feat=128):
        super().__init__()
        self.img_proj = nn.Linear(1, cfg_img.h)
        self.txt_proj = nn.Linear(4, cfg_txt.h)
        self.img_net = ProMaxNCStacked(cfg_img, num_layers=num_layers, fourier=True, pool_size=2)
        self.txt_net = ProMaxNCStacked(cfg_txt, num_layers=num_layers, fourier=True, pool_size=2)
        self.fuse = nn.Sequential(
            nn.Linear(out_feat * 2, out_feat * 2), nn.GELU(),
            nn.Linear(out_feat * 2, out_feat), nn.GELU(),
            nn.Linear(out_feat, 10),
        )

    def forward(self, img, txt):
        B = img.shape[0]
        img_f = self.img_proj(img.reshape(B, 8, 8, 1))
        txt_f = self.txt_proj(txt.view(B, 8, 8, 4))
        fi = self.img_net(img_f, mode="nonsequential", max_output_steps=1).outputs[:, 0]
        ft = self.txt_net(txt_f, mode="nonsequential", max_output_steps=1).outputs[:, 0]
        return self.fuse(torch.cat([fi, ft], dim=-1))


def save(report):
    out = ROOT / "experiments_v3" / "results" / "pro_max_nc_ablation.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)


def main():
    only_mm = "--only-mm" in sys.argv
    set_seed(42)
    report = {}

    # ---------- MNIST ----------
    if not only_mm:
        print("===== MNIST (L=8, h=8, adr_hidden=64, 1200 steps) =====", flush=True)
        dtr, dva = mnist_loaders(32)
        for kernel in ["full", "pure", "visc"]:
            torch.cuda.reset_peak_memory_stats()
            set_seed(42)
            cfg = make_kernel_config(28, 8, 10, kernel)
            m = AbImage(cfg, num_layers=8)
            r = train_cls(m, dtr, dva, 1200, 1.2e-3, f"MNIST-{kernel}")
            r["params"] = sum(p.numel() for p in m.parameters())
            r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            report.setdefault("mnist", {})[kernel] = r
            save(report)
            print(f"  MNIST-{kernel}: acc={r['acc']*100:.2f}% P={r['params']:,} "
                  f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
            del m; torch.cuda.empty_cache()

    # ---------- Multimodal ----------
    print("===== Multimodal (L=8, h=16, adr_hidden=64, 1000 steps) =====", flush=True)
    dtr, dva = mm_loaders(32)
    for kernel in ["full", "pure"]:
        torch.cuda.reset_peak_memory_stats()
        set_seed(42)
        cfg_img = make_kernel_config(8, 16, 128, kernel, out_hidden=128)
        cfg_txt = make_kernel_config(8, 16, 128, kernel, out_hidden=128)
        m = AbMM(cfg_img, cfg_txt, num_layers=8)
        r = train_mm(m, dtr, dva, 1000, 1.2e-3, f"MM-{kernel}")
        r["params"] = sum(p.numel() for p in m.parameters())
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        report.setdefault("mm", {})[kernel] = r
        save(report)
        print(f"  MM-{kernel}: acc={r['acc']*100:.2f}% P={r['params']:,} "
              f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
        del m; torch.cuda.empty_cache()

    # ---------- summary ----------
    print("\n===== Summary (reference: NC-L8 MNIST 98.70%, NC MM 99.17%) =====", flush=True)
    for task, ref in [("mnist", 0.9870), ("mm", 0.9917)]:
        for kernel, r in report.get(task, {}).items():
            print(f"  {task}-{kernel}: acc={r['acc']*100:.2f}%  (ref {ref*100:.2f}%)", flush=True)
    save(report)
    print(f"\nSaved: {ROOT / 'experiments_v3' / 'results' / 'pro_max_nc_ablation.json'}", flush=True)


if __name__ == "__main__":
    main()