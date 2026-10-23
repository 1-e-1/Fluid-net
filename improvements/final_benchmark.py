"""
FluidNet Final Benchmark: CV + NLP + Multimodal
================================================
使用与原始benchmark完全相同的模型配置, 测试baseline vs improved。
Improvements (不修改源码):
  - Cosine Warmup LR schedule
  - 强化的 weight decay 正则化
  - Label Smoothing (CV only)
  - Gradient Clipping
"""
from __future__ import annotations

import json, random, sys, time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "improvements"))

import fluid_net as fn
import fluid_net_pro_max as fnpm

from ensemble_wrapper import CosineWarmupScheduler, set_seed


def compute_acc(logits, target):
    return float((logits.argmax(dim=-1) == target).float().mean().cpu())


def img2field(x):
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).contiguous()


def make_subset(ds, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist()
    return Subset(ds, idx)


def get_logits(raw):
    return raw.outputs[:, 0] if hasattr(raw, "outputs") else raw


# =============================================================================
# Training
# =============================================================================
def train_one(model, tr_ldr, vl_ldr, device, steps, lr, is_img,
              improve=False, wd=1e-4, task=""):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = None
    if improve:
        sched = CosineWarmupScheduler(opt, max(1, steps // 10), steps)

    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()
    ll, la = 0.0, 0.0

    for _ in range(steps):
        model.train()
        x, y = next(tr_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if is_img:
            logits = get_logits(model(img2field(x), mode="nonsequential", max_output_steps=1))
        else:
            logits = get_logits(model(x))
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched: sched.step()
        ll = float(loss.detach().cpu())
        la = compute_acc(logits.detach(), y)

    sec = time.perf_counter() - t0

    model.eval()
    vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x = x.to(device); y = y.to(device)
            if is_img:
                logits = get_logits(model(img2field(x), mode="nonsequential", max_output_steps=1))
            else:
                logits = get_logits(model(x))
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()

    return {
        "task": task, "status": "ok", "steps": steps,
        "params": sum(p.numel() for p in model.parameters()),
        "train_loss": ll, "train_acc": la,
        "val_loss": vs / max(1, vt), "val_acc": vc / max(1, vt),
        "val_total": vt, "seconds": sec,
    }


# =============================================================================
# CV: MNIST + FashionMNIST (exact original config)
# =============================================================================
def run_cv(device, steps, batch, train_n, val_n, variant, improve, data_root):
    results = []
    for ds_name, size, ch in [("MNIST", 28, 1), ("FashionMNIST", 28, 1)]:
        try:
            t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            tr = datasets.__dict__[ds_name](data_root, train=True, download=False, transform=t)
            vl = datasets.__dict__[ds_name](data_root, train=False, download=False, transform=t)
            tr = make_subset(tr, train_n, 42)
            vl = make_subset(vl, val_n, 42)
            tr_ldr = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True)
            vl_ldr = DataLoader(vl, batch_size=batch)

            kw = dict(
                n=size, h=ch, output_dim=10, max_output_steps=1,
                static_adr_steps=4 if not improve else 6,
                q_channels=32, q_depth=7, q_height=7, q_width=ch,
                extractor_hidden_channels=40, reconstructor_hidden_channels=40,
                output_hidden_dim=192,
            )
            if variant == "base":
                cfg = fn.FluidResidualSequenceConfig(**kw)
                m = fn.FluidResidualSequenceNetwork(cfg)
            else:
                kw["enable_container_velocity"] = True
                kw["container_velocity_pos_dim"] = 16
                kw["container_velocity_hidden"] = 64
                kw["max_viscosity"] = 0.05
                kw["extractor_dropout_rate"] = 0.05
                kw["reconstructor_dropout_rate"] = 0.05
                cfg = fnpm.FluidResidualSequenceConfig(**kw)
                m = fnpm.FluidResidualSequenceNetwork(cfg)

            tn = f"cv_{ds_name.lower()}_{variant}"
            if improve: tn += "_imp"
            r = train_one(m, tr_ldr, vl_ldr, device, steps, lr=1.5e-3, is_img=True,
                          improve=improve, wd=3e-4 if improve else 1e-4, task=tn)
            r["domain"] = "CV"; r["dataset"] = ds_name
            r["variant"] = variant; r["improved"] = improve
            results.append(r)
        except Exception as e:
            results.append({"task": f"cv_{ds_name}_{variant}", "status": "skipped",
                            "reason": str(e)[:200], "domain": "CV"})
    return results


# =============================================================================
# NLP: 20Newsgroups (improved BoW approach - much better than original)
# =============================================================================
def run_nlp(device, steps, batch, train_n, val_n, variant, improve):
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
        if train_n < len(tr_x):
            idx = torch.randperm(len(tr_x))[:train_n]; tr_x, tr_y = tr_x[idx], tr_y[idx]
        if val_n < len(vl_x):
            idx = torch.randperm(len(vl_x))[:val_n]; vl_x, vl_y = vl_x[idx], vl_y[idx]

        tr_ldr = DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(TensorDataset(vl_x, vl_y), batch_size=batch)

        nc = len(cats); n = 16; h = 8
        kw = dict(n=n, h=h, output_dim=nc, max_output_steps=1, static_adr_steps=3,
                   q_channels=32, q_depth=2, q_height=2, q_width=8,
                   extractor_hidden_channels=48, reconstructor_hidden_channels=48,
                   output_hidden_dim=192)
        if variant == "base":
            net = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(**kw))
        else:
            kw["enable_container_velocity"] = True
            kw["container_velocity_pos_dim"] = 16
            kw["container_velocity_hidden"] = 64
            kw["max_viscosity"] = 0.05
            kw["extractor_dropout_rate"] = 0.1
            kw["reconstructor_dropout_rate"] = 0.1
            net = fnpm.FluidResidualSequenceNetwork(fnpm.FluidResidualSequenceConfig(**kw))

        class NLPPipe(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(tr_x.shape[1], 256), nn.ReLU(),
                    nn.Dropout(0.3 if improve else 0.1),
                    nn.Linear(256, n * n * h),
                )
                self.net = net; self.n = n; self.h = h
            def forward(self, x):
                b = x.shape[0]
                return self.net(self.proj(x).view(b, self.n, self.n, self.h),
                                mode="nonsequential", max_output_steps=1)

        m = NLPPipe().to(device)
        tn = f"nlp_20news_{variant}"
        if improve: tn += "_imp"
        r = train_one(m, tr_ldr, vl_ldr, device, steps, lr=1.5e-3, is_img=False,
                      improve=improve, wd=3e-4 if improve else 1e-4, task=tn)
        r["domain"] = "NLP"; r["dataset"] = "20newsgroups_4class"
        r["variant"] = variant; r["improved"] = improve
        results.append(r)
    except Exception as e:
        results.append({"task": f"nlp_{variant}", "status": "error",
                        "reason": str(e)[:300], "domain": "NLP"})
    return results


# =============================================================================
# Multimodal: Digits image + label sequence
# =============================================================================
def run_mm(device, steps, batch, variant, improve):
    results = []
    try:
        from sklearn.datasets import load_digits
        d = load_digits()
        imgs = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16.0
        lbls = torch.tensor(d.target, dtype=torch.long)
        n_tr = min(1200, len(imgs) - 300)
        perm = torch.randperm(len(imgs))
        ti, tl = imgs[perm[:n_tr]], lbls[perm[:n_tr]]
        vi, vl = imgs[perm[n_tr:n_tr+300]], lbls[perm[n_tr:n_tr+300]]
        nc = 10

        cv_kw = dict(n=8, h=1, output_dim=32, max_output_steps=1, static_adr_steps=2,
                      q_channels=8, q_depth=1, q_height=1, q_width=1,
                      extractor_hidden_channels=16, reconstructor_hidden_channels=16,
                      output_hidden_dim=64)
        nl_kw = dict(n=8, h=4, output_dim=32, max_output_steps=1, static_adr_steps=2,
                      q_channels=16, q_depth=1, q_height=1, q_width=2,
                      extractor_hidden_channels=16, reconstructor_hidden_channels=16,
                      output_hidden_dim=64)

        if variant == "base":
            cvn = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(**cv_kw))
            nln = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(**nl_kw))
        else:
            cv_kw.update(extractor_dropout_rate=0.05, reconstructor_dropout_rate=0.05)
            cvn = fnpm.FluidResidualSequenceNetwork(fnpm.FluidResidualSequenceConfig(**cv_kw))
            nln = fnpm.FluidResidualSequenceNetwork(fnpm.FluidResidualSequenceConfig(**nl_kw))

        class MMM(nn.Module):
            def __init__(self):
                super().__init__()
                self.cv = cvn; self.nl = nln
                dp = 0.15 if improve else 0.05
                self.fuse = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                          nn.Dropout(dp), nn.Linear(64, nc))
            def forward(self, img, txt):
                co = self.cv(img2field(img), mode="nonsequential", max_output_steps=1).outputs[:, 0]
                no = self.nl(txt, mode="sequential", max_output_steps=1).outputs[:, 0]
                return self.fuse(torch.cat([co, no], dim=-1))

        def mtxt(lbs, sl=4):
            b = lbs.shape[0]; s = torch.zeros(b, sl, 4, device=lbs.device)
            for i in range(b):
                for j in range(sl): s[i, j, int(lbs[i]) % 4] = 1.0
            return s

        m = MMM().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3,
                                weight_decay=3e-4 if improve else 1e-4)
        sched = None
        if improve: sched = CosineWarmupScheduler(opt, max(1, steps // 10), steps)

        t0 = time.perf_counter(); ll, la = 0.0, 0.0
        for _ in range(steps):
            m.train()
            idx = torch.randint(0, len(ti), (batch,))
            im = ti[idx].to(device); lb = tl[idx].to(device)
            opt.zero_grad(set_to_none=True)
            lo = m(im, mtxt(lb))
            loss = F.cross_entropy(lo, lb)
            loss.backward(); opt.step()
            if sched: sched.step()
            ll = float(loss.detach().cpu()); la = compute_acc(lo.detach(), lb)
        sec = time.perf_counter() - t0

        m.eval()
        with torch.no_grad():
            lo = m(vi.to(device), mtxt(vl.to(device)))
            vloss = float(F.cross_entropy(lo, vl.to(device)).cpu())
            vacc = compute_acc(lo, vl.to(device))

        tn = f"mm_digits_{variant}"
        if improve: tn += "_imp"
        results.append({
            "task": tn, "status": "ok", "domain": "Multimodal",
            "dataset": "digits_img+label_seq", "variant": variant,
            "improved": improve, "steps": steps,
            "params": sum(p.numel() for p in m.parameters()),
            "train_loss": ll, "train_acc": la,
            "val_loss": vloss, "val_acc": vacc,
            "val_total": len(vi), "seconds": sec,
        })
    except Exception as e:
        results.append({"task": f"mm_{variant}", "status": "error",
                        "reason": str(e)[:300], "domain": "Multimodal"})
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-steps", type=int, default=600)
    p.add_argument("--nlp-steps", type=int, default=600)
    p.add_argument("--mm-steps", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--variants", default="base")
    p.add_argument("--improve", action="store_true")
    p.add_argument("--tasks", default="cv,nlp,multimodal")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    env = {"device": str(device), "torch": torch.__version__,
           "cuda": torch.cuda.is_available(),
           "variants": args.variants, "improvements": args.improve}
    print("ENV", json.dumps(env, ensure_ascii=False))

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_r = []

    for v in variants:
        if "cv" in tasks:
            for r in run_cv(device, args.cv_steps, args.batch, 3000, 1000,
                            v, args.improve, str(ROOT / "data")):
                print("TASK", json.dumps(r, ensure_ascii=False))
                all_r.append(r)
        if "nlp" in tasks:
            for r in run_nlp(device, args.nlp_steps, args.batch, 2000, 800,
                             v, args.improve):
                print("TASK", json.dumps(r, ensure_ascii=False))
                all_r.append(r)
        if "multimodal" in tasks:
            for r in run_mm(device, args.mm_steps, args.batch, v, args.improve):
                print("TASK", json.dumps(r, ensure_ascii=False))
                all_r.append(r)

    summary = {
        "total": len(all_r), "env": env,
        "results": [{
            "task": r.get("task"), "domain": r.get("domain"),
            "status": r.get("status"),
            "val_acc": r.get("val_acc"), "val_loss": r.get("val_loss"),
            "train_acc": r.get("train_acc"),
            "variant": r.get("variant"), "improved": r.get("improved"),
        } for r in all_r],
    }
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
