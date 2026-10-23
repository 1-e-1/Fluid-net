# -*- coding: utf-8 -*-
"""experiments_v2/tests/benchmark.py - 隔离实验基准测试"""
import json, sys, time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v2" / "models"))

from deep_fluid import (
    set_seed, img2field, make_subset, acc_fn,
    build_baseline, DeepFluidFormer, WideFluidFormer, train_eval,
)

def run_all():
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_root = str(ROOT / "data")

    # MNIST
    mnist_tr = make_subset(datasets.MNIST(data_root, train=True, download=False, transform=t), 3000)
    mnist_vl = make_subset(datasets.MNIST(data_root, train=False, download=False, transform=t), 1000)
    tr_ldr = DataLoader(mnist_tr, 64, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(mnist_vl, 64)

    # FashionMNIST
    fmnist_tr = make_subset(datasets.FashionMNIST(data_root, train=True, download=False, transform=t), 3000)
    fmnist_vl = make_subset(datasets.FashionMNIST(data_root, train=False, download=False, transform=t), 1000)
    ftr_ldr = DataLoader(fmnist_tr, 64, shuffle=True, drop_last=True)
    fvl_ldr = DataLoader(fmnist_vl, 64)

    models = [
        ("Baseline FluidNet", build_baseline()),
        ("Deep(h=8,nb=3)", DeepFluidFormer(28, 8, 3, 10)),
        ("Deep(h=8,nb=4)", DeepFluidFormer(28, 8, 4, 10)),
        ("Wide(h=16,nb=4)", WideFluidFormer(28, 16, 4, 10)),
    ]

    all_results = []
    for ds_name, tr, vl, steps in [
        ("MNIST", tr_ldr, vl_ldr, 400),
        ("FashionMNIST", ftr_ldr, fvl_ldr, 400),
    ]:
        print(f"\n{'='*60}")
        print(f"  {ds_name} ({steps} steps)")
        print(f"{'='*60}")
        for name, model in models:
            if "Wide" in name and ds_name == "FashionMNIST":
                continue  # skip heaviest for FMNIST
            set_seed(42)
            r = train_eval(model, tr, vl, device, steps, lr=1e-3, name=name)
            r["dataset"] = ds_name
            all_results.append(r)
            print(f"  {name:25s} | P={r['params']:>8,} | acc={r['val_acc']:.4f} | loss={r['val_loss']:.4f} | {r['seconds']:.1f}s")

    # Save results
    out_path = ROOT / "experiments_v2" / "results" / "benchmark_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25s} {'Params':>8s} {'MNIST':>8s} {'FMNIST':>8s}")
    print("-" * 55)
    mnist_r = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fmnist_r = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    for name, _ in models:
        m = mnist_r.get(name, {})
        f = fmnist_r.get(name, {})
        print(f"{name:<25s} {m.get('params',0):>8,} {m.get('val_acc',0):.4f}     {f.get('val_acc',0):.4f}")

    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    run_all()
