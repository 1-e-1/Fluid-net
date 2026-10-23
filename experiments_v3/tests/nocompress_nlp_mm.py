# -*- coding: utf-8 -*-
"""
experiments_v3/tests/nocompress_nlp_mm.py
==========================================
NoCompressFluidNet (无Q压缩同维度ADR) 在 NLP 和 多模态 上的测试

NLP:    20Newsgroups 4-class
        文本 → BoW(2048) → 投影场 [16,16,16] → L层ADR → 池化4x4 → 大MLP
MM:     Digits img+txt 10-class
        图像流 [8,8,8] + 文本流 [8,8,8] → 各自L层ADR → 池化 → 融合 → MLP
"""
from __future__ import annotations

import sys, time, json, math, random
from pathlib import Path
from itertools import cycle
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import (
    set_seed, img2field, make_subset, FourierGlobalMix,
    CosineWarmupScheduler, LabelSmoothingLoss,
)

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")


def make_block(n, h, adr_hidden=48, step_size=0.30):
    cfg = fn.FluidResidualSequenceConfig(
        n=n, h=h, output_dim=h, max_output_steps=1, static_adr_steps=1,
        adr_hidden_channels=adr_hidden, adr_step_size=step_size,
        q_channels=16, q_depth=1, q_height=1, q_width=1,
        extractor_hidden_channels=16, reconstructor_hidden_channels=16, output_hidden_dim=128)
    return fn.ADRBlock3D(cfg)


class NCFieldStack(nn.Module):
    """通用同维度ADR堆叠模块: 输入[B,n,n,h]场 → Fourier → L层ADR → 池化 → 特征"""
    def __init__(self, n, h, num_layers, pool_size, step_size=0.30,
                 adr_hidden=48, out_feat=256):
        super().__init__()
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.adr_layers = nn.ModuleList([make_block(n, h, adr_hidden, step_size) for _ in range(num_layers)])
        self.adr_norms = nn.ModuleList([nn.LayerNorm([n, n, h]) for _ in range(num_layers)])
        flat = pool_size * pool_size * h
        self.pool_size = pool_size
        self.proj_out = nn.Linear(flat, out_feat)

    def forward(self, field, mode="nonsequential", max_output_steps=1):
        # field: [B,n,n,h]
        f = field + self.fourier_norm(self.fourier(field))
        f5 = f.unsqueeze(1)  # [B,1,n,n,h]
        for li, (adr, norm) in enumerate(zip(self.adr_layers, self.adr_norms)):
            step_idx = torch.full((f5.shape[0], 1), li, dtype=torch.long, device=f5.device)
            f5 = norm((f5 + adr(f5, step_idx)).squeeze(1)).unsqueeze(1)
        pooled = F.adaptive_avg_pool2d(
            f5.squeeze(1).permute(0, 3, 1, 2), (self.pool_size, self.pool_size))
        return self.proj_out(pooled.flatten(1))


# ==================== NLP ====================
class NC_NLP(nn.Module):
    """NLP: BoW → 投影场 → NCFieldStack → 大MLP"""
    def __init__(self, input_dim=2048, n=16, h=16, num_layers=6, num_classes=4,
                 pool_size=4, out_feat=256, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(1024, n * n * h),
        )
        self.stack = NCFieldStack(n, h, num_layers, pool_size, out_feat=out_feat)
        self.head = nn.Sequential(
            nn.Linear(out_feat, out_feat // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_feat // 2, num_classes),
        )

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        B = x.shape[0]
        n = self.stack.adr_layers[0].context_net[0].weight.shape[-2] if hasattr(self.stack, "n") else 16
        # 场尺寸从第一层ADR推断不可靠, 直接用16
        field = self.encoder(x).view(B, 16, 16, self.stack.fourier.weight.shape[-2])
        feat = self.stack(field, mode=mode, max_output_steps=max_output_steps)
        return self.head(feat)


# ==================== Multimodal ====================
class NC_MM(nn.Module):
    """MM: 图像流 + 文本流 → 各自NCStack → 融合 → 大MLP"""
    def __init__(self, n_img=8, n_txt=8, h=8, num_layers=4, num_classes=10,
                 pool_size=2, out_feat=128, dropout=0.2):
        super().__init__()
        self.img_proj = nn.Linear(1, h)   # 8x8x1 -> 8x8xh
        self.txt_proj = nn.Linear(4, h)   # 64x4 -> 8x8xh
        self.img_stack = NCFieldStack(n_img, h, num_layers, pool_size, out_feat=out_feat)
        self.txt_stack = NCFieldStack(n_txt, h, num_layers, pool_size, out_feat=out_feat)
        self.fuse = nn.Sequential(
            nn.Linear(out_feat * 2, out_feat * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_feat * 2, out_feat), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(out_feat, num_classes),
        )

    def forward(self, img, txt, mode="nonsequential", max_output_steps=1):
        B = img.shape[0]
        img_f = img2field(img)                       # [B,8,8,1]
        img_f = self.img_proj(img_f)                 # [B,8,8,h]
        txt_f = txt.view(B, 8, 8, 4)
        txt_f = self.txt_proj(txt_f)                 # [B,8,8,h]

        img_feat = self.img_stack(img_f, mode=mode, max_output_steps=max_output_steps)
        txt_feat = self.txt_stack(txt_f, mode=mode, max_output_steps=max_output_steps)
        return self.fuse(torch.cat([img_feat, txt_feat], dim=-1))


# ==================== Training ====================
def train_eval(model, tr_ldr, vl_ldr, device, steps=500, lr=1.2e-3, wd=1e-4,
               name="", num_classes=10, is_image=False, mm_mode=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    crit = LabelSmoothingLoss(num_classes, 0.05)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for _ in range(steps):
        model.train()
        if mm_mode:
            x_img, x_txt, y = [b.to(device) for b in next(tr_iter)]
            opt.zero_grad(set_to_none=True)
            logits = model(x_img, x_txt, mode="nonsequential", max_output_steps=1)
        else:
            x, y = [b.to(device) for b in next(tr_iter)[:2]]
            opt.zero_grad(set_to_none=True)
            if is_image:
                logits = model(img2field(x), mode="nonsequential", max_output_steps=1)
            else:
                logits = model(x, mode="nonsequential", max_output_steps=1)
        if hasattr(logits, "outputs"): logits = logits.outputs[:, 0]
        if logits.dim() > 2: logits = logits.squeeze(1)
        crit(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    sec = time.perf_counter() - t0
    model.eval()
    vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for batch in vl_ldr:
            if mm_mode:
                x_img, x_txt, y = [b.to(device) for b in batch]
                logits = model(x_img, x_txt, mode="nonsequential", max_output_steps=1)
            else:
                x, y = [b.to(device) for b in batch[:2]]
                if is_image:
                    logits = model(img2field(x), mode="nonsequential", max_output_steps=1)
                else:
                    logits = model(x, mode="nonsequential", max_output_steps=1)
            if hasattr(logits, "outputs"): logits = logits.outputs[:, 0]
            if logits.dim() > 2: logits = logits.squeeze(1)
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(-1) == y).sum().cpu()); vt += y.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc / vt,
            "val_loss": vs / vt, "seconds": sec}


# ==================== NLP Data ====================
def build_nlp_data():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    cats = ["comp.graphics", "rec.sport.baseball", "sci.med", "talk.politics.guns"]
    ng = fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=2048, stop_words="english")
    X = vec.fit_transform(ng.data).toarray().astype(np.float32)
    y = np.array([cats.index(ng.target_names[t]) for t in ng.target], dtype=np.int64)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    vl_ds = TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(y_vl))
    return DataLoader(tr_ds, 32, shuffle=True, drop_last=True), DataLoader(vl_ds, 32)


# ==================== MM Data ====================
def build_mm_data():
    from sklearn.datasets import load_digits
    data = load_digits()
    X_img = data.images.astype(np.float32) / 16.0 - 0.5
    X_img = X_img[:, None, :, :]
    y = data.target.astype(np.int64)
    n = len(y)

    txt_dim = 4
    X_txt = np.zeros((n, 64, txt_dim), dtype=np.float32)
    for i in range(n):
        for j in range(64):
            X_txt[i, j, int(y[i]) % txt_dim] = np.random.randn() * 0.1 + 0.5

    idx = np.random.default_rng(42).permutation(n)
    split = int(n * 0.8)
    tr_idx, vl_idx = idx[:split], idx[split:]

    class MMDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, txts, lbls):
            self.imgs = imgs; self.txts = txts; self.lbls = lbls
        def __len__(self): return len(self.lbls)
        def __getitem__(self, i): return self.imgs[i], self.txts[i], self.lbls[i]

    tr_ds = MMDataset(torch.from_numpy(X_img[tr_idx]), torch.from_numpy(X_txt[tr_idx]), torch.from_numpy(y[tr_idx]))
    vl_ds = MMDataset(torch.from_numpy(X_img[vl_idx]), torch.from_numpy(X_txt[vl_idx]), torch.from_numpy(y[vl_idx]))
    return DataLoader(tr_ds, 32, shuffle=True, drop_last=True), DataLoader(vl_ds, 32)


# ==================== Main ====================
if __name__ == "__main__":
    set_seed(42)
    print("=" * 65)
    print("  NO-COMPRESS: NLP + 多模态 测试")
    print("=" * 65)

    all_results = []

    # ---------- NLP ----------
    print(f"\n{'='*55} NLP: 20Newsgroups (4-class)")
    tr_ldr, vl_ldr = build_nlp_data()

    class NLPBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2048, 256)
            self.adr = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=16, h=1, output_dim=4, max_output_steps=1, static_adr_steps=3,
                    q_channels=32, q_depth=4, q_height=4, q_width=1,
                    extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))
        def forward(self, x, mode="nonsequential", max_output_steps=1):
            f = self.proj(x).view(-1, 16, 16, 1)
            return self.adr(f, mode=mode, max_output_steps=max_output_steps)

    set_seed(42)
    r = train_eval(NLPBaseline(), tr_ldr, vl_ldr, device, steps=500, lr=1e-3,
                   name="NLP-Baseline(Q)", num_classes=4)
    r["domain"] = "NLP"; all_results.append(r)
    print(f"  NLP-Baseline(Q):   acc={r['val_acc']:.4f} P={r['params']:,}")

    # NC-NLP L4
    set_seed(42)
    m = NC_NLP(input_dim=2048, n=16, h=16, num_layers=4, num_classes=4, out_feat=256)
    r = train_eval(m, tr_ldr, vl_ldr, device, steps=500, lr=1.2e-3, name="NLP-NC(L4)")
    r["domain"] = "NLP"; all_results.append(r)
    print(f"  NLP-NC(L4):        acc={r['val_acc']:.4f} P={r['params']:,}")

    # NC-NLP L8
    set_seed(42)
    m = NC_NLP(input_dim=2048, n=16, h=16, num_layers=8, num_classes=4, out_feat=256)
    r = train_eval(m, tr_ldr, vl_ldr, device, steps=500, lr=1.2e-3, name="NLP-NC(L8)")
    r["domain"] = "NLP"; all_results.append(r)
    print(f"  NLP-NC(L8):        acc={r['val_acc']:.4f} P={r['params']:,}")

    # ---------- Multimodal ----------
    print(f"\n{'='*55} Multimodal: Digits (img+txt)")
    tr_ldr2, vl_ldr2 = build_mm_data()

    class MMBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.cv = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=8, h=1, output_dim=32, max_output_steps=1, static_adr_steps=2,
                    q_channels=8, q_depth=1, q_height=1, q_width=1,
                    extractor_hidden_channels=16, reconstructor_hidden_channels=16, output_hidden_dim=64))
            self.nl = fn.FluidResidualSequenceNetwork(
                fn.FluidResidualSequenceConfig(
                    n=8, h=4, output_dim=32, max_output_steps=1, static_adr_steps=2,
                    q_channels=16, q_depth=1, q_height=1, q_width=2,
                    extractor_hidden_channels=16, reconstructor_hidden_channels=16, output_hidden_dim=64))
            self.fuse = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
        def forward(self, img, txt, mode="nonsequential", max_output_steps=1):
            co = self.cv(img2field(img), mode=mode, max_output_steps=max_output_steps).outputs[:, 0]
            no = self.nl(txt, mode="sequential", max_output_steps=max_output_steps).outputs[:, 0]
            return self.fuse(torch.cat([co, no], dim=-1))

    set_seed(42)
    r = train_eval(MMBaseline(), tr_ldr2, vl_ldr2, device, steps=400, lr=1e-3,
                   name="MM-Baseline(Q)", num_classes=10, mm_mode=True)
    r["domain"] = "Multimodal"; all_results.append(r)
    print(f"  MM-Baseline(Q):    acc={r['val_acc']:.4f} P={r['params']:,}")

    # NC-MM L4
    set_seed(42)
    m = NC_MM(n_img=8, n_txt=8, h=8, num_layers=4, num_classes=10, out_feat=128)
    r = train_eval(m, tr_ldr2, vl_ldr2, device, steps=400, lr=1.2e-3,
                   name="MM-NC(L4)", num_classes=10, mm_mode=True)
    r["domain"] = "Multimodal"; all_results.append(r)
    print(f"  MM-NC(L4):         acc={r['val_acc']:.4f} P={r['params']:,}")

    # NC-MM L8
    set_seed(42)
    m = NC_MM(n_img=8, n_txt=8, h=8, num_layers=8, num_classes=10, out_feat=128)
    r = train_eval(m, tr_ldr2, vl_ldr2, device, steps=400, lr=1.2e-3,
                   name="MM-NC(L8)", num_classes=10, mm_mode=True)
    r["domain"] = "Multimodal"; all_results.append(r)
    print(f"  MM-NC(L8):         acc={r['val_acc']:.4f} P={r['params']:,}")

    # ==================== Summary ====================
    print("\n" + "=" * 65)
    print(f"{'Model':<20s} {'Domain':<12s} {'Acc':>8s} {'Params':>10s}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: (x["domain"], -x["val_acc"])):
        print(f"{r['model']:<20s} {r['domain']:<12s} {r['val_acc']:>7.4f} {r['params']:>10,}")

    with open(ROOT / "experiments_v3" / "results" / "nocompress_nlp_mm.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved!")
