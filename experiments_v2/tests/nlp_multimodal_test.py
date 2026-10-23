# -*- coding: utf-8 -*-
"""
experiments_v2/tests/nlp_multimodal_test.py
LeanFluidFormer - NLP & 多模态 基准测试
"""
import sys, json, time, random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v2" / "models"))

import fluid_net as fn
from deep_fluid import FourierGlobalMix

# ============ Utils ============
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def img2field(x): return x.permute(0, 2, 3, 1).contiguous()

def make_subset(ds, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()
    return Subset(ds, idx)

def acc_fn(logits, y):
    return float((logits.argmax(dim=-1) == y).float().mean().cpu())

def get_logits(out):
    return out.outputs[:, 0] if hasattr(out, "outputs") else out

# ============ CV LeanFluidFormer (复用已验证架构) ============
class LeanFluidFormerCV(nn.Module):
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
                output_hidden_dim=256, adr_hidden_channels=48,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.fourier_norm(self.fourier(h))
        return self.adr(h, mode=mode, max_output_steps=max_output_steps)

# ============ NLP LeanFluidFormer ============
class LeanFluidFormerNLP(nn.Module):
    """NLP: BoW特征 → 投影 → 2D Field → Fourier → ADR"""
    def __init__(self, input_dim, n=16, h=8, num_classes=4, static_adr_steps=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n * n * h),
        )
        self.n, self.h = n, h
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=24, q_depth=2, q_height=2, q_width=2,
                extractor_hidden_channels=40,
                reconstructor_hidden_channels=40,
                output_hidden_dim=192, adr_hidden_channels=40,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        B = x.shape[0]
        field = self.proj(x).view(B, self.n, self.n, self.h)
        field = field + self.fourier_norm(self.fourier(field))
        return self.adr(field, mode=mode, max_output_steps=max_output_steps)

# ============ 多模态 LeanFluidFormer ============
class LeanFluidFormerMM(nn.Module):
    """多模态: 双流 (图像+文本) → 融合 → 分类"""
    def __init__(self, n_img=8, n_txt=8, h_img=4, h_txt=4, num_classes=10):
        super().__init__()
        # 图像流
        self.img_proj = nn.Linear(1, h_img)
        self.img_fourier = FourierGlobalMix(n_img, h_img)
        self.img_norm = nn.LayerNorm([n_img, n_img, h_img])
        self.img_adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_img, h=h_img, output_dim=32,
                max_output_steps=1, static_adr_steps=3,
                q_channels=16, q_depth=1, q_height=1, q_width=1,
                extractor_hidden_channels=24,
                reconstructor_hidden_channels=24,
                output_hidden_dim=96, adr_hidden_channels=24,
            ))
        # 文本流
        self.txt_proj = nn.Linear(4, h_txt)  # 4-dim text features
        self.txt_fourier = FourierGlobalMix(n_txt, h_txt)
        self.txt_norm = nn.LayerNorm([n_txt, n_txt, h_txt])
        self.txt_adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_txt, h=h_txt, output_dim=32,
                max_output_steps=1, static_adr_steps=3,
                q_channels=16, q_depth=1, q_height=1, q_width=1,
                extractor_hidden_channels=24,
                reconstructor_hidden_channels=24,
                output_hidden_dim=96, adr_hidden_channels=24,
            ))
        # 融合头
        self.fusion = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, img, txt, mode="nonsequential", max_output_steps=1):
        # 图像处理
        hi = self.img_proj(img)
        hi = hi + self.img_norm(self.img_fourier(hi))
        img_out = self.img_adr(hi, mode=mode, max_output_steps=max_output_steps)

        # 文本处理  
        ht = self.txt_proj(txt)
        ht = ht + self.txt_norm(self.txt_fourier(ht))
        txt_out = self.txt_adr(ht, mode=mode, max_output_steps=max_output_steps)

        # 融合
        fused = torch.cat([img_out.outputs[:, 0], txt_out.outputs[:, 0]], dim=-1)
        logits = self.fusion(fused)
        return type("obj", (object,), {"outputs": logits.unsqueeze(1)})()

# ============ 统一训练函数 ============
def train_eval(model, tr_ldr, vl_ldr, device, steps, lr, wd, name,
               is_image=True, mm_mode=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr*0.01)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for _ in range(steps):
        model.train()
        batch = next(tr_iter)
        if mm_mode:
            img, txt, y = batch
            img, txt, y = img.to(device), txt.to(device), y.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)

        opt.zero_grad(set_to_none=True)

        if mm_mode:
            out = model(img, txt)
        elif is_image:
            out = model(img2field(x), mode="nonsequential", max_output_steps=1)
        else:
            out = model(x, mode="nonsequential", max_output_steps=1)

        logits = get_logits(out)
        F.cross_entropy(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    sec = time.perf_counter() - t0

    model.eval(); vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for batch in vl_ldr:
            if mm_mode:
                img, txt, y = batch
                img, txt, y = img.to(device), txt.to(device), y.to(device)
                out = model(img, txt)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                if is_image:
                    out = model(img2field(x), mode="nonsequential", max_output_steps=1)
                else:
                    out = model(x, mode="nonsequential", max_output_steps=1)

            logits = get_logits(out)
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()

    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc/vt,
            "val_loss": vs/vt, "seconds": sec}

# ============ CV 基准 ============
def run_cv_baseline(device):
    """快速CV基线: MNIST"""
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    tr = make_subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=t), 3000)
    vl = make_subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=False, download=False, transform=t), 1000)
    tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl, 64)

    results = []

    # Baseline (h=1)
    cfg = fn.FluidResidualSequenceConfig(
        n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
        q_channels=32, q_depth=7, q_height=7, q_width=1,
        extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192)
    m = fn.FluidResidualSequenceNetwork(cfg)
    r = train_eval(m, tr_ldr, vl_ldr, device, 400, 1e-3, 1e-4, "CV-Baseline(h=1)", is_image=True)
    r["domain"] = "CV"; results.append(r)

    # Lean(h=8)
    m = LeanFluidFormerCV(28, 8, 10, static_adr_steps=6)
    r = train_eval(m, tr_ldr, vl_ldr, device, 400, 1e-3, 1e-4, "CV-Lean(h=8)", is_image=True)
    r["domain"] = "CV"; results.append(r)

    return results

# ============ NLP 基准 ============
def run_nlp_benchmark(device):
    """NLP: 20 Newsgroups 4-class"""
    results = []
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer

        cats = ["comp.graphics", "rec.sport.baseball", "sci.med", "talk.politics.misc"]
        tr_raw = fetch_20newsgroups(subset="train", categories=cats,
                                     remove=("headers", "footers", "quotes"))
        vl_raw = fetch_20newsgroups(subset="test", categories=cats,
                                     remove=("headers", "footers", "quotes"))
        vec = CountVectorizer(max_features=5000, stop_words="english")
        tr_x = torch.tensor(vec.fit_transform(tr_raw.data).toarray(), dtype=torch.float32)
        tr_y = torch.tensor(tr_raw.target, dtype=torch.long)
        vl_x = torch.tensor(vec.transform(vl_raw.data).toarray(), dtype=torch.float32)
        vl_y = torch.tensor(vl_raw.target, dtype=torch.long)

        # Subset
        n_tr = min(2000, len(tr_x))
        n_vl = min(800, len(vl_x))
        idx = torch.randperm(len(tr_x))[:n_tr]; tr_x, tr_y = tr_x[idx], tr_y[idx]
        idx = torch.randperm(len(vl_x))[:n_vl]; vl_x, vl_y = vl_x[idx], vl_y[idx]

        tr_ldr = DataLoader(TensorDataset(tr_x, tr_y), 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(TensorDataset(vl_x, vl_y), 64)

        nc = len(cats); input_dim = tr_x.shape[1]

        # NLP Baseline: BoW → FluidNet (h=8, 无Fourier)
        class NLPBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 16*16*8))
                self.adr = fn.FluidResidualSequenceNetwork(
                    fn.FluidResidualSequenceConfig(
                        n=16, h=8, output_dim=nc,
                        max_output_steps=1, static_adr_steps=4,
                        q_channels=32, q_depth=2, q_height=2, q_width=8,
                        extractor_hidden_channels=48,
                        reconstructor_hidden_channels=48,
                        output_hidden_dim=192))
                self.n, self.h = 16, 8
            def forward(self, x, mode="nonsequential", max_output_steps=1):
                B = x.shape[0]
                f = self.proj(x).view(B, self.n, self.n, self.h)
                return self.adr(f, mode=mode, max_output_steps=max_output_steps)

        m = NLPBaseline()
        r = train_eval(m, tr_ldr, vl_ldr, device, 500, 1.5e-3, 2e-4,
                       "NLP-Baseline(BoW→ADR)", is_image=False)
        r["domain"] = "NLP"; results.append(r)

        # NLP Lean: BoW → Fourier → ADR
        m2 = LeanFluidFormerNLP(input_dim, n=16, h=8, num_classes=nc, static_adr_steps=4)
        r2 = train_eval(m2, tr_ldr, vl_ldr, device, 500, 1.5e-3, 3e-4,
                        "NLP-Lean(BoW→Fourier→ADR)", is_image=False)
        r2["domain"] = "NLP"; results.append(r2)

        # NLP Lean wider: h=12
        m3 = LeanFluidFormerNLP(input_dim, n=16, h=12, num_classes=nc, static_adr_steps=5)
        r3 = train_eval(m3, tr_ldr, vl_ldr, device, 500, 1.5e-3, 3e-4,
                        "NLP-Lean(h=12)", is_image=False)
        r3["domain"] = "NLP"; results.append(r3)

    except Exception as e:
        results.append({"model": "NLP-*", "domain": "NLP", "status": "error", "reason": str(e)[:200]})

    return results

# ============ 多模态基准 ============
def run_multimodal_benchmark(device):
    """多模态: Digits图像 + 标签序列"""
    results = []
    try:
        from sklearn.datasets import load_digits
        d = load_digits()
        imgs = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16.0  # [N,1,8,8]
        lbls = torch.tensor(d.target, dtype=torch.long)

        n_tr = min(1200, len(imgs) - 300)
        perm = torch.randperm(len(imgs))
        tr_img, tr_lbl = imgs[perm[:n_tr]], lbls[perm[:n_tr]]
        vl_img, vl_lbl = imgs[perm[n_tr:n_tr+300]], lbls[perm[n_tr:n_tr+300]]

        nc = 10

        def make_txt(lbs, sl=4):
            b = lbs.shape[0]
            s = torch.zeros(b, sl, 4)
            for i in range(b):
                for j in range(sl):
                    s[i, j, int(lbs[i]) % 4] = 1.0
            return s

        tr_txt = make_txt(tr_lbl)
        vl_txt = make_txt(vl_lbl)

        # Dataset: (img, txt, label)
        class MMDataset(torch.utils.data.Dataset):
            def __init__(self, imgs, txts, lbls):
                self.imgs = imgs; self.txts = txts; self.lbls = lbls
            def __len__(self): return len(self.lbls)
            def __getitem__(self, i): return self.imgs[i], self.txts[i], self.lbls[i]

        tr_ds = MMDataset(tr_img, tr_txt, tr_lbl)
        vl_ds = MMDataset(vl_img, vl_txt, vl_lbl)
        tr_ldr = DataLoader(tr_ds, 32, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl_ds, 32)

        # MM Baseline: 双流 FluidNet (无Fourier)
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
                self.fuse = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, nc))
            def forward(self, img, txt):
                co = self.cv(img2field(img), mode="nonsequential", max_output_steps=1).outputs[:, 0]
                no = self.nl(txt, mode="sequential", max_output_steps=1).outputs[:, 0]
                return self.fuse(torch.cat([co, no], dim=-1))

        m = MMBaseline()
        r = train_eval(m, tr_ldr, vl_ldr, device, 300, 1e-3, 1e-4,
                       "MM-Baseline", is_image=False, mm_mode=True)
        r["domain"] = "Multimodal"; results.append(r)

        # MM Lean: 双流 + Fourier
        m2 = LeanFluidFormerMM(n_img=8, n_txt=8, h_img=4, h_txt=4, num_classes=nc)
        r2 = train_eval(m2, tr_ldr, vl_ldr, device, 300, 1.5e-3, 2e-4,
                        "MM-Lean(h=4)", is_image=False, mm_mode=True)
        r2["domain"] = "Multimodal"; results.append(r2)

        # MM Lean wider: h_img=6, h_txt=6
        m3 = LeanFluidFormerMM(n_img=8, n_txt=8, h_img=6, h_txt=6, num_classes=nc)
        r3 = train_eval(m3, tr_ldr, vl_ldr, device, 300, 1.5e-3, 2e-4,
                        "MM-Lean(h=6)", is_image=False, mm_mode=True)
        r3["domain"] = "Multimodal"; results.append(r3)

    except Exception as e:
        import traceback
        results.append({"model": "MM-*", "domain": "Multimodal", "status": "error",
                        "reason": str(e)[:300], "trace": traceback.format_exc()[-300:]})

    return results

# ============ Main ============
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("=" * 65)

    all_results = []

    # CV
    print("\n--- CV (MNIST) ---")
    for r in run_cv_baseline(device):
        print(f"  {r['model']:30s} | acc={r['val_acc']:.4f} | P={r['params']:,}")
        all_results.append(r)

    # NLP
    print("\n--- NLP (20Newsgroups 4-class) ---")
    for r in run_nlp_benchmark(device):
        if r.get("status") == "error":
            print(f"  ERROR: {r['reason'][:100]}")
        else:
            print(f"  {r['model']:30s} | acc={r['val_acc']:.4f} | P={r['params']:,}")
        all_results.append(r)

    # Multimodal
    print("\n--- Multimodal (Digits img+txt) ---")
    for r in run_multimodal_benchmark(device):
        if r.get("status") == "error":
            print(f"  ERROR: {r['reason'][:100]}")
        else:
            print(f"  {r['model']:30s} | acc={r['val_acc']:.4f} | P={r['params']:,}")
        all_results.append(r)

    # Final Summary
    print("\n" + "=" * 65)
    print("  THREE-DOMAIN COMPARISON")
    print("=" * 65)
    print(f"{'Model':<30s} {'Domain':<12s} {'Params':>8s} {'Acc':>8s}")
    print("-" * 62)
    for r in all_results:
        if r.get("status") == "error": continue
        print(f"{r['model']:<30s} {r['domain']:<12s} {r['params']:>8,} {r['val_acc']:>7.4f}")

    # Save
    out = ROOT / "experiments_v2" / "results" / "nlp_mm_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
