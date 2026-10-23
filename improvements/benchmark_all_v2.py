"""
FluidNet Comprehensive Benchmark: CV + NLP + Multimodal (Fixed)
================================================================
修复: CV输入维度匹配, 多模态设备bug
"""
from __future__ import annotations

import json
import random
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "improvements"))

import fluid_net as fn
import fluid_net_pro as fnp
import fluid_net_pro_max as fnpm

from ensemble_wrapper import (
    CosineWarmupScheduler,
    LabelSmoothingLoss,
    set_seed,
    compute_accuracy,
    image_to_field,
)


def make_subset(dataset, max_items: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    n = min(max_items, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, indices)


def train_model(model, train_loader, val_loader, device, steps, lr=1e-3,
                is_image=True, label_smoothing=0.0, cosine_schedule=False,
                warmup_ratio=0.1, weight_decay=1e-4, task_name="task"):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if cosine_schedule:
        warmup = max(1, int(steps * warmup_ratio))
        scheduler = CosineWarmupScheduler(opt, warmup, steps)

    train_iter = cycle(train_loader)
    t0 = time.perf_counter()
    last_loss, last_acc = 0.0, 0.0

    for step_idx in range(steps):
        model.train()
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if is_image:
            raw_out = model(image_to_field(x), mode="nonsequential", max_output_steps=1)
        else:
            raw_out = model(x)
        logits = raw_out.outputs[:, 0] if hasattr(raw_out, 'outputs') else raw_out
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if scheduler:
            scheduler.step()

        last_loss = float(loss.detach().cpu())
        last_acc = compute_accuracy(logits.detach(), y)

    train_seconds = time.perf_counter() - t0

    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_image:
                raw_out = model(image_to_field(x), mode="nonsequential", max_output_steps=1)
            else:
                raw_out = model(x)
            logits = raw_out.outputs[:, 0] if hasattr(raw_out, 'outputs') else raw_out
            loss = F.cross_entropy(logits, y, reduction="sum")
            val_loss_sum += float(loss.cpu())
            val_correct += int((logits.argmax(dim=-1) == y).sum().cpu())
            val_total += y.numel()

    return {
        "task": task_name, "status": "ok",
        "steps": steps, "params": sum(p.numel() for p in model.parameters()),
        "train_last_loss": last_loss, "train_last_acc": last_acc,
        "val_loss": val_loss_sum / max(1, val_total),
        "val_acc": val_correct / max(1, val_total),
        "val_total": val_total, "train_seconds": train_seconds,
    }


# ==============================================================================
# CV Benchmark
# ==============================================================================
def run_cv_tasks(device, steps=300, batch_size=64, train_subset=3000, val_subset=1000,
                 variant="base", improve=False, data_root=""):
    if not data_root:
        data_root = str(ROOT / "data")
    results = []

    for ds_name, nc, img_size in [
        ("MNIST", 10, 28),
        ("FashionMNIST", 10, 28),
    ]:
        try:
            n = img_size  # 使用图像尺寸作为容器大小
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            train_ds = datasets.__dict__[ds_name](
                data_root, train=True, download=False, transform=transform
            )
            val_ds = datasets.__dict__[ds_name](
                data_root, train=False, download=False, transform=transform
            )
            train_ds = make_subset(train_ds, train_subset, 42)
            val_ds = make_subset(val_ds, val_subset, 42)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)

            cfg_kwargs = dict(
                n=n, h=1, output_dim=nc,
                max_output_steps=1, static_adr_steps=2,
                q_channels=16, q_depth=1, q_height=1, q_width=1,
                extractor_hidden_channels=32, reconstructor_hidden_channels=32,
                output_hidden_dim=128,
            )

            if variant == "base":
                cfg = fn.FluidResidualSequenceConfig(**cfg_kwargs)
                model = fn.FluidResidualSequenceNetwork(cfg)
            elif variant == "pro":
                cfg = fnp.FluidResidualSequenceConfig(
                    **cfg_kwargs,
                    enable_container_velocity=True,
                    container_velocity_pos_dim=16,
                    container_velocity_hidden=64,
                )
                model = fnp.FluidResidualSequenceNetwork(cfg)
            else:
                cfg = fnpm.FluidResidualSequenceConfig(
                    **cfg_kwargs,
                    enable_container_velocity=True,
                    container_velocity_pos_dim=16,
                    container_velocity_hidden=64,
                    max_viscosity=0.05,
                    extractor_dropout_rate=0.05,
                    reconstructor_dropout_rate=0.05,
                )
                model = fnpm.FluidResidualSequenceNetwork(cfg)

            task_name = f"cv_{ds_name.lower()}_{variant}"
            if improve:
                task_name += "_improved"

            result = train_model(
                model, train_loader, val_loader, device, steps,
                lr=1e-3, is_image=True, task_name=task_name,
                cosine_schedule=improve, weight_decay=3e-4 if improve else 1e-4,
            )
            result["domain"] = "CV"
            result["dataset"] = ds_name
            result["model_variant"] = variant
            result["improvements"] = improve
            results.append(result)

        except Exception as e:
            results.append({
                "task": f"cv_{ds_name.lower()}_{variant}",
                "status": "skipped", "reason": str(e)[:200], "domain": "CV",
            })

    return results


# ==============================================================================
# NLP Benchmark
# ==============================================================================
def run_nlp_tasks(device, steps=500, batch_size=64, train_subset=2000, val_subset=800,
                  variant="base", improve=False):
    results = []
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer

        categories = ["comp.graphics", "rec.sport.baseball", "sci.med", "talk.politics.misc"]
        train_raw = fetch_20newsgroups(subset="train", categories=categories,
                                       remove=("headers", "footers", "quotes"))
        val_raw = fetch_20newsgroups(subset="test", categories=categories,
                                     remove=("headers", "footers", "quotes"))

        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        train_vec = vectorizer.fit_transform(train_raw.data)
        val_vec = vectorizer.transform(val_raw.data)

        train_x = torch.tensor(train_vec.toarray(), dtype=torch.float32)
        train_y = torch.tensor(train_raw.target, dtype=torch.long)
        val_x = torch.tensor(val_vec.toarray(), dtype=torch.float32)
        val_y = torch.tensor(val_raw.target, dtype=torch.long)

        if train_subset < len(train_x):
            idx = torch.randperm(len(train_x))[:train_subset]
            train_x, train_y = train_x[idx], train_y[idx]
        if val_subset < len(val_x):
            idx = torch.randperm(len(val_x))[:val_subset]
            val_x, val_y = val_x[idx], val_y[idx]

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y), batch_size=batch_size,
        )

        num_classes = len(categories)
        input_dim = train_x.shape[1]
        n = 16
        h = 8

        cfg_kwargs = dict(
            n=n, h=h, output_dim=num_classes,
            max_output_steps=1, static_adr_steps=3,
            q_channels=32, q_depth=2, q_height=2, q_width=8,
            extractor_hidden_channels=48, reconstructor_hidden_channels=48,
            output_hidden_dim=192,
        )

        if variant == "base":
            cfg = fn.FluidResidualSequenceConfig(**cfg_kwargs)
            net = fn.FluidResidualSequenceNetwork(cfg)
        elif variant == "pro":
            cfg = fnp.FluidResidualSequenceConfig(
                **cfg_kwargs, enable_container_velocity=True,
                container_velocity_pos_dim=16, container_velocity_hidden=64,
            )
            net = fnp.FluidResidualSequenceNetwork(cfg)
        else:
            cfg = fnpm.FluidResidualSequenceConfig(
                **cfg_kwargs,
                enable_container_velocity=True,
                container_velocity_pos_dim=16, container_velocity_hidden=64,
                max_viscosity=0.05,
                extractor_dropout_rate=0.1, reconstructor_dropout_rate=0.1,
            )
            net = fnpm.FluidResidualSequenceNetwork(cfg)

        class NLPPipeline(nn.Module):
            def __init__(self, fluid_net, in_dim, n, h):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(in_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3 if improve else 0.1),
                    nn.Linear(256, n * n * h),
                )
                self.fluid_net = fluid_net
                self.n, self.h = n, h

            def forward(self, x):
                b = x.shape[0]
                field = self.proj(x).view(b, self.n, self.n, self.h)
                return self.fluid_net(field, mode="nonsequential", max_output_steps=1).outputs[:, 0]

        model = NLPPipeline(net, input_dim, n, h).to(device)

        task_name = f"nlp_20newsgroups_{variant}"
        if improve:
            task_name += "_improved"

        result = train_model(
            model, train_loader, val_loader, device, steps,
            lr=1.5e-3, is_image=False, task_name=task_name,
            cosine_schedule=improve,
            weight_decay=3e-4 if improve else 1e-4,
        )
        result["domain"] = "NLP"
        result["dataset"] = "20newsgroups_4class"
        result["model_variant"] = variant
        result["improvements"] = improve
        result["categories"] = categories
        results.append(result)

    except Exception as e:
        import traceback
        results.append({
            "task": f"nlp_20newsgroups_{variant}",
            "status": "error", "reason": str(e)[:300],
            "traceback": traceback.format_exc()[-400:], "domain": "NLP",
        })

    return results


# ==============================================================================
# Multimodal Benchmark
# ==============================================================================
def run_mm_tasks(device, steps=200, batch_size=32, variant="promax", improve=False):
    results = []
    try:
        from sklearn.datasets import load_digits
        digits = load_digits()
        images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
        labels = torch.tensor(digits.target, dtype=torch.long)

        n_train = min(1200, len(images) - 300)
        perm = torch.randperm(len(images))
        train_img = images[perm[:n_train]]
        train_lbl = labels[perm[:n_train]]
        val_img = images[perm[n_train:n_train + 300]]
        val_lbl = labels[perm[n_train:n_train + 300]]

        nc = 10
        n_img = 8

        cfg_cv = dict(
            n=n_img, h=1, output_dim=32,
            max_output_steps=1, static_adr_steps=2,
            q_channels=8, q_depth=1, q_height=1, q_width=1,
            extractor_hidden_channels=16, reconstructor_hidden_channels=16,
            output_hidden_dim=64,
        )
        cfg_nlp = dict(
            n=8, h=4, output_dim=32,
            max_output_steps=1, static_adr_steps=2,
            q_channels=16, q_depth=1, q_height=1, q_width=2,
            extractor_hidden_channels=16, reconstructor_hidden_channels=16,
            output_hidden_dim=64,
        )

        if variant == "base":
            cv_net = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(**cfg_cv))
            nlp_net = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(**cfg_nlp))
        else:
            cfg_cv.update(extractor_dropout_rate=0.05, reconstructor_dropout_rate=0.05)
            cv_net = fnpm.FluidResidualSequenceNetwork(fnpm.FluidResidualSequenceConfig(**cfg_cv))
            nlp_net = fnpm.FluidResidualSequenceNetwork(fnpm.FluidResidualSequenceConfig(**cfg_nlp))

        class MMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cv_net = cv_net
                self.nlp_net = nlp_net
                self.fusion = nn.Sequential(
                    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1 if improve else 0.0),
                    nn.Linear(64, nc),
                )

            def forward(self, img, text_seq):
                cv_out = self.cv_net(image_to_field(img), mode="nonsequential", max_output_steps=1).outputs[:, 0]
                nlp_out = self.nlp_net(text_seq, mode="sequential", max_output_steps=1).outputs[:, 0]
                return self.fusion(torch.cat([cv_out, nlp_out], dim=-1))

        model = MMModel().to(device)

        def make_text(lbls, seq_len=4):
            b = lbls.shape[0]
            seq = torch.zeros(b, seq_len, 4, device=lbls.device)
            for i in range(b):
                for s in range(seq_len):
                    seq[i, s, int(lbls[i]) % 4] = 1.0
            return seq

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=2e-4 if improve else 1e-4)
        scheduler = None
        if improve:
            scheduler = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)

        t0 = time.perf_counter()
        last_loss, last_acc = 0.0, 0.0
        for step_idx in range(steps):
            model.train()
            # Fix: generate indices on CPU
            idx = torch.randint(0, len(train_img), (batch_size,))
            imgs = train_img[idx].to(device)
            lbls = train_lbl[idx].to(device)
            texts = make_text(lbls)

            opt.zero_grad(set_to_none=True)
            logits = model(imgs, texts)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            opt.step()
            if scheduler:
                scheduler.step()
            last_loss = float(loss.detach().cpu())
            last_acc = compute_accuracy(logits.detach(), lbls)

        train_seconds = time.perf_counter() - t0

        model.eval()
        val_imgs = val_img.to(device)
        val_lbls = val_lbl.to(device)
        val_texts = make_text(val_lbls)
        with torch.no_grad():
            logits = model(val_imgs, val_texts)
            val_loss = float(F.cross_entropy(logits, val_lbls).cpu())
            val_acc = compute_accuracy(logits, val_lbls)

        task_name = f"multimodal_digits_{variant}"
        if improve:
            task_name += "_improved"

        results.append({
            "task": task_name, "status": "ok", "domain": "Multimodal",
            "dataset": "digits_img+label_seq",
            "model_variant": variant, "improvements": improve,
            "steps": steps, "params": sum(p.numel() for p in model.parameters()),
            "train_last_loss": last_loss, "train_last_acc": last_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_total": len(val_img), "train_seconds": train_seconds,
        })

    except Exception as e:
        import traceback
        results.append({
            "task": f"multimodal_digits_{variant}",
            "status": "error", "reason": str(e)[:300],
            "traceback": traceback.format_exc()[-500:], "domain": "Multimodal",
        })

    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-steps", type=int, default=300)
    parser.add_argument("--nlp-steps", type=int, default=500)
    parser.add_argument("--mm-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variants", default="base")
    parser.add_argument("--improve", action="store_true")
    parser.add_argument("--tasks", default="cv,nlp,multimodal")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    env_info = {
        "device": str(device), "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "variants": args.variants,
        "improvements": args.improve,
    }
    print("ENV", json.dumps(env_info, ensure_ascii=False))

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_results = []

    for variant in variants:
        if "cv" in task_list:
            print(f"\n--- CV: {variant} (improve={args.improve}) ---")
            for r in run_cv_tasks(device, args.cv_steps, args.batch_size,
                                  variant=variant, improve=args.improve):
                print("TASK_RESULT", json.dumps(r, ensure_ascii=False))
                all_results.append(r)

        if "nlp" in task_list:
            print(f"\n--- NLP: {variant} (improve={args.improve}) ---")
            for r in run_nlp_tasks(device, args.nlp_steps, args.batch_size,
                                   variant=variant, improve=args.improve):
                print("TASK_RESULT", json.dumps(r, ensure_ascii=False))
                all_results.append(r)

        if "multimodal" in task_list:
            print(f"\n--- Multimodal: {variant} (improve={args.improve}) ---")
            for r in run_mm_tasks(device, args.mm_steps, args.batch_size,
                                  variant=variant, improve=args.improve):
                print("TASK_RESULT", json.dumps(r, ensure_ascii=False))
                all_results.append(r)

    summary = {
        "total": len(all_results),
        "env": env_info,
        "results": [{
            "task": r.get("task"), "domain": r.get("domain"),
            "status": r.get("status"),
            "val_acc": r.get("val_acc"), "val_loss": r.get("val_loss"),
            "train_acc": r.get("train_last_acc"),
            "variant": r.get("model_variant"),
            "improved": r.get("improvements"),
        } for r in all_results],
    }
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))
    return all_results


if __name__ == "__main__":
    main()
