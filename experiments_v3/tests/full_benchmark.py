# -*- coding: utf-8 -*-
"""
experiments_v3/tests/full_benchmark.py v2
全面基准测试: CV (MNIST + FashionMNIST) + NLP + Multimodal
"""
import sys, json, time, random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import (
    set_seed, img2field, make_subset, acc_fn,
    EnhancedFluidFormer, MultiScaleFluidFormer,
    NLPImprovedFormer, MMImprovedFormer,
    train_eval_improved, rand_augment_lite, ChannelMix, SpatialMix,
)

def run_cv_benchmark(device):
    results = []
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    for ds_name, ds_cls in [("MNIST", datasets.MNIST), ("FashionMNIST", datasets.FashionMNIST)]:
        tr_size = 5000 if ds_name == "FashionMNIST" else 3000
        tr = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=True, download=False, transform=t), tr_size)
        vl = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=False, download=False, transform=t), 1000)
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)

        print(f"\n{'='*60}")
        print(f"  {ds_name} | Train={tr_size} Val=1000")
        print(f"{'='*60}")

        # Baseline
        set_seed(42)
        b = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(
            n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
            q_channels=32, q_depth=7, q_height=7, q_width=1,
            extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))
        r = train_eval_improved(b, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
                                name="Baseline(h=1)", num_classes=10)
        r["dataset"] = ds_name; r["domain"] = "CV"
        results.append(r)
        print(f"  Baseline:             acc={r['val_acc']:.4f} P={r['params']:,}")

        # EnhancedFluidFormer h=12 (核心改进)
        set_seed(42)
        m1 = EnhancedFluidFormer(28, 12, 10, static_adr_steps=6)
        r = train_eval_improved(m1, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                                name="Enhanced(h=12)", num_classes=10, label_smoothing=0.05)
        r["dataset"] = ds_name; r["domain"] = "CV"
        results.append(r)
        print(f"  Enhanced(h=12):       acc={r['val_acc']:.4f} P={r['params']:,}")

        # EnhancedFluidFormer h=14
        set_seed(42)
        m2 = EnhancedFluidFormer(28, 14, 10, static_adr_steps=6, dropout=0.12)
        r = train_eval_improved(m2, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                                name="Enhanced(h=14)", num_classes=10, label_smoothing=0.05)
        r["dataset"] = ds_name; r["domain"] = "CV"
        results.append(r)
        print(f"  Enhanced(h=14):       acc={r['val_acc']:.4f} P={r['params']:,}")

        # MultiScaleFluidFormer
        set_seed(42)
        m3 = MultiScaleFluidFormer(28, 12, 10, static_adr_steps=5)
        r = train_eval_improved(m3, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                                name="MultiScale(h=12)", num_classes=10, label_smoothing=0.05)
        r["dataset"] = ds_name; r["domain"] = "CV"
        results.append(r)
        print(f"  MultiScale(h=12):     acc={r['val_acc']:.4f} P={r['params']:,}")

        # FashionMNIST专用: +Augmentation +更多训练
        if ds_name == "FashionMNIST":
            # Enhanced + Aug
            set_seed(42)
            m5 = EnhancedFluidFormer(28, 12, 10, static_adr_steps=6)
            r = train_eval_improved(m5, tr_ldr, vl_ldr, device, steps=700, lr=1.5e-3, wd=2e-4,
                                    name="Enhanced(h=12)+Aug", num_classes=10,
                                    label_smoothing=0.1, use_augment=True, warmup_ratio=0.15)
            r["dataset"] = ds_name; r["domain"] = "CV"
            results.append(r)
            print(f"  Enhanced+Aug:         acc={r['val_acc']:.4f} P={r['params']:,}")

            # Enhanced h=16 + Aug
            set_seed(42)
            m6 = EnhancedFluidFormer(28, 16, 10, static_adr_steps=6, dropout=0.1, q_channels=40)
            r = train_eval_improved(m6, tr_ldr, vl_ldr, device, steps=700, lr=2e-3, wd=3e-4,
                                    name="Enhanced(h=16)+Aug", num_classes=10,
                                    label_smoothing=0.1, use_augment=True, warmup_ratio=0.15)
            r["dataset"] = ds_name; r["domain"] = "CV"
            results.append(r)
            print(f"  Enhanced(h=16)+Aug:   acc={r['val_acc']:.4f} P={r['params']:,}")

    return results

def run_nlp_benchmark(device):
    results = []
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    print(f"\n{'='*60}")
    print("  NLP: 20Newsgroups (4-class)")
    print(f"{'='*60}")

    cats = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.guns']
    ng = fetch_20newsgroups(subset='all', categories=cats, remove=('headers', 'footers', 'quotes'))
    vec = TfidfVectorizer(max_features=2048, stop_words='english')
    X = vec.fit_transform(ng.data).toarray().astype(np.float32)
    y = np.array([cats.index(ng.target_names[t]) for t in ng.target], dtype=np.int64)

    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    vl_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(y_vl))
    tr_ldr = DataLoader(tr_ds, 32, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl_ds, 32)

    class NLPBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2048, 256)
            self.adr = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=16, h=1, output_dim=4, max_output_steps=1, static_adr_steps=3,
                    q_channels=32, q_depth=4, q_height=4, q_width=1,
                    extractor_hidden_channels=40, reconstructor_hidden_channels=40,
                    output_hidden_dim=192))
        def forward(self, x, mode="nonsequential", max_output_steps=1):
            f = self.proj(x).view(-1, 16, 16, 1)
            return self.adr(f, mode="nonsequential", max_output_steps=1)

    set_seed(42)
    m = NLPBaseline()
    r = train_eval_improved(m, tr_ldr, vl_ldr, device, steps=400, lr=1e-3, wd=1e-4,
                            name="NLP-Baseline", num_classes=4, is_image=False,
                            label_smoothing=0.05)
    r["domain"] = "NLP"; results.append(r)
    print(f"  NLP-Baseline:         acc={r['val_acc']:.4f} P={r['params']:,}")

    set_seed(42)
    m2 = NLPImprovedFormer(2048, n=16, h=12, num_classes=4, static_adr_steps=4, dropout=0.3)
    r = train_eval_improved(m2, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                            name="NLP-Improved(h=12)", num_classes=4, is_image=False,
                            label_smoothing=0.1)
    r["domain"] = "NLP"; results.append(r)
    print(f"  NLP-Improved(h=12):   acc={r['val_acc']:.4f} P={r['params']:,}")

    set_seed(42)
    m3 = NLPImprovedFormer(2048, n=16, h=16, num_classes=4, static_adr_steps=4, dropout=0.25)
    r = train_eval_improved(m3, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=3e-4,
                            name="NLP-Improved(h=16)", num_classes=4, is_image=False,
                            label_smoothing=0.1)
    r["domain"] = "NLP"; results.append(r)
    print(f"  NLP-Improved(h=16):   acc={r['val_acc']:.4f} P={r['params']:,}")

    return results

def run_mm_benchmark(device):
    results = []
    print(f"\n{'='*60}")
    print("  Multimodal: Digits (img+txt)")
    print(f"{'='*60}")

    from sklearn.datasets import load_digits
    data = load_digits()
    X_img = data.images.astype(np.float32) / 16.0 - 0.5
    X_img = X_img[:, None, :, :]
    y = data.target.astype(np.int64)
    n_samples = len(y)

    txt_dim = 4
    X_txt_flat = np.zeros((n_samples, 64, txt_dim), dtype=np.float32)
    for i in range(n_samples):
        for j in range(8):
            for k in range(8):
                X_txt_flat[i, j * 8 + k, int(y[i]) % txt_dim] = np.random.randn() * 0.1 + 0.5

    idx = np.random.default_rng(42).permutation(n_samples)
    split = int(n_samples * 0.8)
    tr_idx, vl_idx = idx[:split], idx[split:]

    class MMDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, txts, lbls):
            self.imgs = imgs; self.txts = txts; self.lbls = lbls
        def __len__(self): return len(self.lbls)
        def __getitem__(self, i): return self.imgs[i], self.txts[i], self.lbls[i]

    tr_ds = MMDataset(
        torch.from_numpy(X_img[tr_idx]),
        torch.from_numpy(X_txt_flat[tr_idx]),
        torch.from_numpy(y[tr_idx]))
    vl_ds = MMDataset(
        torch.from_numpy(X_img[vl_idx]),
        torch.from_numpy(X_txt_flat[vl_idx]),
        torch.from_numpy(y[vl_idx]))
    tr_ldr = DataLoader(tr_ds, 32, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl_ds, 32)

    class MMBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.cv = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=8, h=1, output_dim=32, max_output_steps=1, static_adr_steps=2,
                    q_channels=8, q_depth=1, q_height=1, q_width=1,
                    extractor_hidden_channels=16, reconstructor_hidden_channels=16,
                    output_hidden_dim=64))
            self.nl = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=8, h=4, output_dim=32, max_output_steps=1, static_adr_steps=2,
                    q_channels=16, q_depth=1, q_height=1, q_width=2,
                    extractor_hidden_channels=16, reconstructor_hidden_channels=16,
                    output_hidden_dim=64))
            self.fuse = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
        def forward(self, img, txt, mode="nonsequential", max_output_steps=1):
            co = self.cv(img2field(img), mode="nonsequential", max_output_steps=1).outputs[:, 0]
            no = self.nl(txt, mode="sequential", max_output_steps=1).outputs[:, 0]
            return self.fuse(torch.cat([co, no], dim=-1))

    set_seed(42)
    m = MMBaseline()
    r = train_eval_improved(m, tr_ldr, vl_ldr, device, steps=400, lr=1e-3, wd=1e-4,
                            name="MM-Baseline", num_classes=10, mm_mode=True,
                            label_smoothing=0.05)
    r["domain"] = "Multimodal"; results.append(r)
    print(f"  MM-Baseline:          acc={r['val_acc']:.4f} P={r['params']:,}")

    set_seed(42)
    m2 = MMImprovedFormer(n_img=8, n_txt=8, h=8, num_classes=10, static_adr_steps=4)
    r = train_eval_improved(m2, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                            name="MM-Improved(h=8)", num_classes=10, mm_mode=True,
                            label_smoothing=0.1)
    r["domain"] = "Multimodal"; results.append(r)
    print(f"  MM-Improved(h=8):     acc={r['val_acc']:.4f} P={r['params']:,}")

    set_seed(42)
    m3 = MMImprovedFormer(n_img=8, n_txt=8, h=12, num_classes=10, static_adr_steps=4)
    r = train_eval_improved(m3, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                            name="MM-Improved(h=12)", num_classes=10, mm_mode=True,
                            label_smoothing=0.1)
    r["domain"] = "Multimodal"; results.append(r)
    print(f"  MM-Improved(h=12):    acc={r['val_acc']:.4f} P={r['params']:,}")

    return results

# ==================== Main ====================
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("=" * 65)
    print("  FLUIDNET v3 FULL BENCHMARK")
    print("=" * 65)

    all_results = []
    all_results.extend(run_cv_benchmark(device))
    all_results.extend(run_nlp_benchmark(device))
    all_results.extend(run_mm_benchmark(device))

    print("\n" + "=" * 80)
    print("  THREE-DOMAIN COMPARISON")
    print("=" * 80)
    print(f"{'Model':<30s} {'Domain':<12s} {'Params':>9s} {'Acc':>8s} {'Time':>7s}")
    print("-" * 72)
    for r in sorted(all_results, key=lambda x: (x.get("domain", ""), -x.get("val_acc", 0))):
        print(f"{r['model']:<30s} {r.get('domain','?'):<12s} {r['params']:>9,} "
              f"{r['val_acc']:>7.4f} {r['seconds']:>6.0f}s")

    print("\n--- Per-Domain Best ---")
    for domain in ["CV", "NLP", "Multimodal"]:
        dom = [r for r in all_results if r.get("domain") == domain]
        if dom:
            best = max(dom, key=lambda x: x["val_acc"])
            print(f"  {domain:<12s}: {best['model']:<30s} acc={best['val_acc']:.4f}")

    out = ROOT / "experiments_v3" / "results" / "full_benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
