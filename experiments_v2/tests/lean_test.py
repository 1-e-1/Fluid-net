# -*- coding: utf-8 -*-
"""experiments_v2/tests/lean_test.py - Lean FluidFormer + Ensemble"""
import sys, json, time
from itertools import cycle
sys.path.insert(0, ".")
sys.path.insert(0, "experiments_v2/models")
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import fluid_net as fn
from deep_fluid import FourierGlobalMix, set_seed, img2field, make_subset, acc_fn

# ============ LeanFluidFormer: 单层宽通道+傅里叶 ============
class LeanFluidFormer(nn.Module):
    """单层宽通道 FluidNet + Fourier, 不做深层堆叠"""
    def __init__(self, n=28, h=8, num_classes=10, static_adr_steps=6):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=4, q_height=4, q_width=1,
                extractor_hidden_channels=48,
                reconstructor_hidden_channels=48,
                output_hidden_dim=256,
                adr_hidden_channels=48,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.fourier_norm(self.fourier(h))
        return self.adr(h, mode=mode, max_output_steps=max_output_steps)

# ============ Ensemble wrapper ============
class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    def forward(self, x, mode="nonsequential", max_output_steps=1):
        outs = []
        for m in self.models:
            o = m(x, mode=mode, max_output_steps=max_output_steps)
            outs.append(o.outputs[:, 0])
        return type("obj", (object,), {"outputs": torch.stack(outs, dim=0).mean(dim=0).unsqueeze(1)})()

# ============ Training ============
def train_eval(model, tr_ldr, vl_ldr, device, steps, lr, wd, name):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr*0.01)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()
    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        fld = img2field(x)
        out = model(fld, mode="nonsequential", max_output_steps=1)
        logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
        F.cross_entropy(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    sec = time.perf_counter() - t0

    model.eval(); vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            fld = img2field(x)
            out = model(fld, mode="nonsequential", max_output_steps=1)
            logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc/vt, "val_loss": vs/vt, "seconds": sec}

# ============ Main ============
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    all_results = []
    for ds_name, ds_cls in [("MNIST", datasets.MNIST), ("FashionMNIST", datasets.FashionMNIST)]:
        tr = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=t), 3000)
        vl = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=False, download=False, transform=t), 1000)
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)

        print(f"\n{'='*55} {ds_name}")

        # Baseline (h=1)
        cfg = fn.FluidResidualSequenceConfig(
            n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
            q_channels=32, q_depth=7, q_height=7, q_width=1,
            extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192)
        b = fn.FluidResidualSequenceNetwork(cfg)
        r = train_eval(b, tr_ldr, vl_ldr, device, 400, 1e-3, 1e-4, "Baseline")
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  Baseline: acc={r['val_acc']:.4f}")

        # Lean(h=8)
        l8 = LeanFluidFormer(28, 8, 10, static_adr_steps=6)
        r = train_eval(l8, tr_ldr, vl_ldr, device, 400, 1e-3, 1e-4, "Lean(h=8)")
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  Lean(h=8): acc={r['val_acc']:.4f} P={r['params']:,}")

        # Lean(h=12)
        l12 = LeanFluidFormer(28, 12, 10, static_adr_steps=6)
        r = train_eval(l12, tr_ldr, vl_ldr, device, 400, 1.5e-3, 2e-4, "Lean(h=12)")
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  Lean(h=12): acc={r['val_acc']:.4f} P={r['params']:,}")

        # Ensemble of 3 Lean(h=8) with different seeds
        if ds_name == "MNIST":
            models_ens = []
            for s in [42, 123, 456]:
                set_seed(s)
                models_ens.append(LeanFluidFormer(28, 8, 10, static_adr_steps=6))
            ens = Ensemble(models_ens)
            r = train_eval(ens, tr_ldr, vl_ldr, device, 400, 1e-3, 1e-4, "Ensemble(3xLean-h8)")
            r["dataset"] = ds_name; all_results.append(r)
            print(f"  Ensemble(3xLean-h8): acc={r['val_acc']:.4f} P={r['params']:,}")

    # Summary
    print(f"\n{'='*55}")
    print(f"{'Model':<25s} {'Params':>9s} {'MNIST':>8s} {'FMNIST':>8s}")
    print("-" * 55)
    mnist = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fmnist = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    for name in ["Baseline", "Lean(h=8)", "Lean(h=12)", "Ensemble(3xLean-h8)"]:
        m = mnist.get(name, {})
        f = fmnist.get(name, {})
        print(f"{name:<25s} {m.get('params',0):>9,} {m.get('val_acc',0):.4f}     {f.get('val_acc',0):.4f}")

    with open(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'experiments_v2/results/lean_results.json'), "w") as fh:
        json.dump(all_results, fh, indent=2)
