"""
FluidNet Comprehensive Benchmark: CV + NLP + Multimodal
========================================================
不修改任何源码，通过wrapper实现改进方案测试。
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

# 导入原始模型 (不修改源码)
import fluid_net as fn
import fluid_net_pro as fnp
import fluid_net_pro_max as fnpm

from ensemble_wrapper import (
    FluidNetEnsemble,
    CosineWarmupScheduler,
    LabelSmoothingLoss,
    FluidNetMultiVariantEnsemble,
    set_seed,
    compute_accuracy,
    image_to_field,
)


def make_subset(dataset, max_items: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    n = min(max_items, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, indices)


# ==============================================================================
# 1. 训练工具函数 (带改进方案)
# ==============================================================================

def train_single_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    steps: int,
    lr: float = 1e-3,
    is_image: bool = True,
    use_label_smoothing: float = 0.0,
    use_cosine_schedule: bool = False,
    warmup_ratio: float = 0.1,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    task_name: str = "task",
) -> dict:
    """统一训练函数，支持多种改进策略"""
    model.to(device)

    if use_label_smoothing > 0 and is_image:
        # Find num_classes from model config
        num_classes = None
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "output_dim"):
                num_classes = m.config.output_dim
                break
        if num_classes:
            loss_fn = LabelSmoothingLoss(num_classes, smoothing=use_label_smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_cosine_schedule:
        warmup_steps = max(1, int(steps * warmup_ratio))
        scheduler = CosineWarmupScheduler(opt, warmup_steps, steps)

    train_iter = cycle(train_loader)
    t0 = time.perf_counter()
    last_loss = 0.0
    last_acc = 0.0

    for step_idx in range(steps):
        model.train()
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if is_image:
            logits = model(image_to_field(x), mode="nonsequential", max_output_steps=1)
            if isinstance(logits, tuple):
                logits = logits[0]
        else:
            logits = model(x)

        loss = loss_fn(logits, y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        opt.step()
        if scheduler:
            scheduler.step()

        last_loss = float(loss.detach().cpu())
        last_acc = compute_accuracy(logits.detach(), y)

    train_seconds = time.perf_counter() - t0

    # Validation
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_image:
                logits = model(image_to_field(x), mode="nonsequential", max_output_steps=1)
                if isinstance(logits, tuple):
                    logits = logits[0]
            else:
                logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            val_loss_sum += float(loss.cpu())
            val_correct += int((logits.argmax(dim=-1) == y).sum().cpu())
            val_total += y.numel()

    return {
        "task": task_name,
        "status": "ok",
        "steps": steps,
        "lr": lr,
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_loss": last_loss,
        "train_last_acc": last_acc,
        "val_loss": val_loss_sum / max(1, val_total),
        "val_acc": val_correct / max(1, val_total),
        "val_total": val_total,
        "train_seconds": train_seconds,
    }


# ==============================================================================
# 2. CV 任务
# ==============================================================================

def build_cv_config(n=16, h=1, output_dim=10, model_variant="base"):
    """构建CV模型配置"""
    common = dict(
        n=n, h=h, output_dim=output_dim,
        max_output_steps=1,
        static_adr_steps=4,
        q_channels=16,
        q_depth=1, q_height=1, q_width=1,
        extractor_hidden_channels=32,
        reconstructor_hidden_channels=32,
        output_hidden_dim=128,
    )
    if model_variant == "base":
        return fn.FluidResidualSequenceConfig(**common)
    elif model_variant == "pro":
        return fnp.FluidResidualSequenceConfig(
            **common,
            enable_container_velocity=True,
            container_velocity_pos_dim=16,
            container_velocity_hidden=64,
        )
    elif model_variant == "promax":
        return fnpm.FluidResidualSequenceConfig(
            **common,
            enable_container_velocity=True,
            container_velocity_pos_dim=16,
            container_velocity_hidden=64,
            max_viscosity=0.1,
            extractor_dropout_rate=0.05,
            reconstructor_dropout_rate=0.05,
        )
    else:
        raise ValueError(f"Unknown variant: {model_variant}")


def run_cv_benchmark(
    device: torch.device,
    steps: int = 300,
    batch_size: int = 64,
    train_subset: int = 3000,
    val_subset: int = 1000,
    data_root: str = "",
    model_variant: str = "base",
    use_improvements: bool = False,
):
    """CV全面基准测试: MNIST + FashionMNIST"""
    if not data_root:
        data_root = str(ROOT / "data")
    results = []

    for dataset_name, num_classes in [("MNIST", 10), ("FashionMNIST", 10)]:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            train_ds = datasets.__dict__[dataset_name](
                data_root, train=True, download=False, transform=transform
            )
            val_ds = datasets.__dict__[dataset_name](
                data_root, train=False, download=False, transform=transform
            )

            train_ds = make_subset(train_ds, train_subset, 42)
            val_ds = make_subset(val_ds, val_subset, 42)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, drop_last=True
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size)

            # Build model
            cfg = build_cv_config(
                n=16, h=1,
                output_dim=num_classes,
                model_variant=model_variant,
            )
            if model_variant == "base":
                model = fn.FluidResidualSequenceNetwork(cfg)
            elif model_variant == "pro":
                model = fnp.FluidResidualSequenceNetwork(cfg)
            else:
                model = fnpm.FluidResidualSequenceNetwork(cfg)

            task_name = f"cv_{dataset_name.lower()}_{model_variant}"
            if use_improvements:
                task_name += "_improved"

            kwargs = dict(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, steps=steps, lr=1e-3, is_image=True,
                task_name=task_name,
            )
            if use_improvements:
                kwargs.update(
                    use_label_smoothing=0.05,
                    use_cosine_schedule=True,
                    warmup_ratio=0.15,
                    weight_decay=2e-4,
                )

            result = train_single_model(**kwargs)
            result["domain"] = "CV"
            result["dataset"] = dataset_name
            result["model_variant"] = model_variant
            result["improvements"] = use_improvements
            results.append(result)

        except Exception as e:
            results.append({
                "task": f"cv_{dataset_name.lower()}_{model_variant}",
                "status": "skipped",
                "reason": str(e)[:200],
                "domain": "CV",
            })

    return results


# ==============================================================================
# 3. NLP 任务
# ==============================================================================

def run_nlp_benchmark(
    device: torch.device,
    steps: int = 400,
    batch_size: int = 64,
    train_subset: int = 2000,
    val_subset: int = 800,
    model_variant: str = "base",
    use_improvements: bool = False,
):
    """NLP基准测试: 20 Newsgroups"""
    results = []
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        return [{"task": "nlp_20newsgroups", "status": "skipped", "reason": "sklearn not available", "domain": "NLP"}]

    try:
        categories = ["comp.graphics", "rec.sport.baseball", "sci.med", "talk.politics.misc"]
        train_raw = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"))
        val_raw = fetch_20newsgroups(subset="test", categories=categories, remove=("headers", "footers", "quotes"))

        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        train_vec = vectorizer.fit_transform(train_raw.data)
        val_vec = vectorizer.transform(val_raw.data)

        # 转为tensor
        train_x = torch.tensor(train_vec.toarray(), dtype=torch.float32)
        train_y = torch.tensor(train_raw.target, dtype=torch.long)
        val_x = torch.tensor(val_vec.toarray(), dtype=torch.float32)
        val_y = torch.tensor(val_raw.target, dtype=torch.long)

        # Subset
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

        seq_len = train_x.shape[1]
        num_classes = len(categories)
        embed_dim = 8

        # Build simpler NLP model: treat features as field
        n = 8
        cfg_kwargs = dict(
            n=n, h=embed_dim, output_dim=num_classes,
            max_output_steps=1,
            static_adr_steps=4,
            q_channels=32,
            q_depth=2, q_height=2, q_width=8,
            extractor_hidden_channels=40,
            reconstructor_hidden_channels=40,
            output_hidden_dim=192,
        )

        if model_variant == "base":
            cfg = fn.FluidResidualSequenceConfig(**cfg_kwargs)
            model = fn.FluidResidualSequenceNetwork(cfg)
        elif model_variant == "pro":
            cfg = fnp.FluidResidualSequenceConfig(**cfg_kwargs, enable_container_velocity=True)
            model = fnp.FluidResidualSequenceNetwork(cfg)
        else:
            cfg = fnpm.FluidResidualSequenceConfig(
                **cfg_kwargs,
                enable_container_velocity=True,
                max_viscosity=0.05,
                extractor_dropout_rate=0.1,
                reconstructor_dropout_rate=0.1,
            )
            model = fnpm.FluidResidualSequenceNetwork(cfg)

        # For NLP, we process as nonsequential with image_to_field reshape
        # Actually, for BoW this is a bit tricky - let me reshape properly
        class NLPWrapper(nn.Module):
            def __init__(self, fluid_net_model, input_dim, n, h):
                super().__init__()
                self.proj = nn.Linear(input_dim, n * n * h)
                self.fluid_net = fluid_net_model
                self.n = n
                self.h = h

            def forward(self, x):
                # x: [B, D] -> [B, n, n, h]
                b = x.shape[0]
                field = self.proj(x).view(b, self.n, self.n, self.h)
                out = self.fluid_net(field, mode="nonsequential", max_output_steps=1)
                return out.outputs[:, 0]

        wrapped = NLPWrapper(model, seq_len, n, embed_dim)
        wrapped.to(device)

        opt = torch.optim.AdamW(wrapped.parameters(), lr=1.5e-3, weight_decay=1e-4)
        if use_improvements:
            opt = torch.optim.AdamW(wrapped.parameters(), lr=1.5e-3, weight_decay=3e-4)
            scheduler = CosineWarmupScheduler(opt, max(1, int(steps * 0.15)), steps)

        train_iter = cycle(train_loader)
        t0 = time.perf_counter()
        last_loss, last_acc = 0.0, 0.0
        for step_idx in range(steps):
            wrapped.train()
            x, y = next(train_iter)
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = wrapped(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            opt.step()
            if use_improvements:
                scheduler.step()
            last_loss = float(loss.detach().cpu())
            last_acc = compute_accuracy(logits.detach(), y)

        train_seconds = time.perf_counter() - t0

        # Validation
        wrapped.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = wrapped(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
                val_loss_sum += float(loss.cpu())
                val_correct += int((logits.argmax(dim=-1) == y).sum().cpu())
                val_total += y.numel()

        task_name = f"nlp_20newsgroups_{model_variant}"
        if use_improvements:
            task_name += "_improved"

        results.append({
            "task": task_name,
            "status": "ok",
            "domain": "NLP",
            "dataset": "20newsgroups_4class",
            "model_variant": model_variant,
            "improvements": use_improvements,
            "steps": steps,
            "params": sum(p.numel() for p in wrapped.parameters()),
            "train_last_loss": last_loss,
            "train_last_acc": last_acc,
            "val_loss": val_loss_sum / max(1, val_total),
            "val_acc": val_correct / max(1, val_total),
            "val_total": val_total,
            "train_seconds": train_seconds,
            "categories": categories,
        })

    except Exception as e:
        import traceback
        results.append({
            "task": f"nlp_20newsgroups_{model_variant}",
            "status": "error",
            "reason": str(e)[:300],
            "traceback": traceback.format_exc()[-500:],
            "domain": "NLP",
        })

    return results


# ==============================================================================
# 4. 多模态任务
# ==============================================================================

def run_multimodal_benchmark(
    device: torch.device,
    steps: int = 200,
    batch_size: int = 32,
    model_variant: str = "promax",
    use_improvements: bool = False,
):
    """多模态基准测试: 图像+文本联合分类"""
    results = []

    try:
        # 构建合成多模态数据: 手写数字图像 + 数值标签作为"文本"特征
        from sklearn.datasets import load_digits
        digits = load_digits()
        images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0  # [N, 1, 8, 8]
        labels = torch.tensor(digits.target, dtype=torch.long)

        # Split
        n_train = min(1200, len(images) - 300)
        perm = torch.randperm(len(images))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + 300]

        train_img = images[train_idx]
        train_labels = labels[train_idx]
        val_img = images[val_idx]
        val_labels = labels[val_idx]

        num_classes = 10

        # Multimodal model: processes image via FluidNet nonsequential + text via FluidNet sequential
        class MultimodalFluidNet(nn.Module):
            def __init__(self, variant, num_classes):
                super().__init__()
                cv_cfg_kwargs = dict(
                    n=8, h=1, output_dim=32,
                    max_output_steps=1, static_adr_steps=2,
                    q_channels=8, q_depth=1, q_height=1, q_width=1,
                    extractor_hidden_channels=16, reconstructor_hidden_channels=16,
                    output_hidden_dim=64,
                )
                nlp_cfg_kwargs = dict(
                    n=8, h=4, output_dim=32,
                    max_output_steps=1, static_adr_steps=2,
                    q_channels=16, q_depth=1, q_height=1, q_width=2,
                    extractor_hidden_channels=16, reconstructor_hidden_channels=16,
                    output_hidden_dim=64,
                )

                if variant == "base":
                    self.cv_net = fn.FluidResidualSequenceNetwork(
                        fn.FluidResidualSequenceConfig(**cv_cfg_kwargs)
                    )
                    self.nlp_net = fn.FluidResidualSequenceNetwork(
                        fn.FluidResidualSequenceConfig(**nlp_cfg_kwargs)
                    )
                elif variant == "promax":
                    cv_cfg_kwargs.update(
                        extractor_dropout_rate=0.05, reconstructor_dropout_rate=0.05
                    )
                    self.cv_net = fnpm.FluidResidualSequenceNetwork(
                        fnpm.FluidResidualSequenceConfig(**cv_cfg_kwargs)
                    )
                    self.nlp_net = fnpm.FluidResidualSequenceNetwork(
                        fnpm.FluidResidualSequenceConfig(
                            **nlp_cfg_kwargs,
                            extractor_dropout_rate=0.05, reconstructor_dropout_rate=0.05,
                        )
                    )
                else:
                    self.cv_net = fn.FluidResidualSequenceNetwork(
                        fn.FluidResidualSequenceConfig(**cv_cfg_kwargs)
                    )
                    self.nlp_net = fn.FluidResidualSequenceNetwork(
                        fn.FluidResidualSequenceConfig(**nlp_cfg_kwargs)
                    )

                self.fusion_head = nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes),
                )

            def forward(self, img, text_seq):
                cv_out = self.cv_net(
                    image_to_field(img), mode="nonsequential", max_output_steps=1
                ).outputs[:, 0]
                nlp_out = self.nlp_net(
                    text_seq, mode="sequential", max_output_steps=1
                ).outputs[:, 0]
                fused = torch.cat([cv_out, nlp_out], dim=-1)
                return self.fusion_head(fused)

        model = MultimodalFluidNet(model_variant, num_classes).to(device)

        # Generate text features: use label as a "word embedding" sequence
        def make_text_seq(labels_tensor, seq_len=4):
            b = labels_tensor.shape[0]
            seq = torch.zeros(b, seq_len, 4, device=labels_tensor.device)
            for i in range(b):
                lbl = int(labels_tensor[i])
                for s in range(seq_len):
                    seq[i, s, lbl % 4] = 1.0
            return seq

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if use_improvements:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
            scheduler = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)

        t0 = time.perf_counter()
        last_loss, last_acc = 0.0, 0.0
        for step_idx in range(steps):
            model.train()
            idx = torch.randint(0, len(train_img), (batch_size,), device=device)
            imgs = train_img[idx].to(device)
            lbls = train_labels[idx].to(device)
            texts = make_text_seq(lbls)

            opt.zero_grad(set_to_none=True)
            logits = model(imgs, texts)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            opt.step()
            if use_improvements:
                scheduler.step()
            last_loss = float(loss.detach().cpu())
            last_acc = compute_accuracy(logits.detach(), lbls)

        train_seconds = time.perf_counter() - t0

        # Validation
        model.eval()
        val_imgs = val_img.to(device)
        val_lbls = val_labels.to(device)
        val_texts = make_text_seq(val_lbls)
        with torch.no_grad():
            logits = model(val_imgs, val_texts)
            val_loss = float(F.cross_entropy(logits, val_lbls).cpu())
            val_acc = compute_accuracy(logits, val_lbls)

        task_name = f"multimodal_digits_{model_variant}"
        if use_improvements:
            task_name += "_improved"

        results.append({
            "task": task_name,
            "status": "ok",
            "domain": "Multimodal",
            "dataset": "sklearn_digits + label_sequence",
            "model_variant": model_variant,
            "improvements": use_improvements,
            "steps": steps,
            "params": sum(p.numel() for p in model.parameters()),
            "train_last_loss": last_loss,
            "train_last_acc": last_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_total": len(val_img),
            "train_seconds": train_seconds,
        })

    except Exception as e:
        import traceback
        results.append({
            "task": f"multimodal_digits_{model_variant}",
            "status": "error",
            "reason": str(e)[:300],
            "traceback": traceback.format_exc()[-500:],
            "domain": "Multimodal",
        })

    return results


# ==============================================================================
# 5. 主入口
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FluidNet Comprehensive Benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-steps", type=int, default=300)
    parser.add_argument("--nlp-steps", type=int, default=500)
    parser.add_argument("--mm-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variants", default="base,pro,promax",
                       help="comma-separated: base,pro,promax")
    parser.add_argument("--improve", action="store_true", help="Enable improvement strategies")
    parser.add_argument("--tasks", default="cv,nlp,multimodal",
                       help="comma-separated: cv,nlp,multimodal")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    env_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "model_variants": args.variants,
        "improvements_enabled": args.improve,
    }
    print("ENV", json.dumps(env_info, ensure_ascii=False))

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_results = []

    for variant in variants:
        # CV tasks
        if "cv" in task_list:
            print(f"\n{'='*60}")
            print(f"CV Benchmark: variant={variant}, improve={args.improve}")
            print(f"{'='*60}")
            cv_results = run_cv_benchmark(
                device=device,
                steps=args.cv_steps,
                batch_size=args.batch_size,
                model_variant=variant,
                use_improvements=args.improve,
            )
            for r in cv_results:
                print("TASK_RESULT", json.dumps(r, ensure_ascii=False))
            all_results.extend(cv_results)

        # NLP tasks
        if "nlp" in task_list:
            print(f"\n{'='*60}")
            print(f"NLP Benchmark: variant={variant}, improve={args.improve}")
            print(f"{'='*60}")
            nlp_results = run_nlp_benchmark(
                device=device,
                steps=args.nlp_steps,
                batch_size=args.batch_size,
                model_variant=variant,
                use_improvements=args.improve,
            )
            for r in nlp_results:
                print("TASK_RESULT", json.dumps(r, ensure_ascii=False))
            all_results.extend(nlp_results)

        # Multimodal tasks
        if "multimodal" in task_list:
            print(f"\n{'='*60}")
            print(f"Multimodal Benchmark: variant={variant}, improve={args.improve}")
            print(f"{'='*60}")
            mm_results = run_multimodal_benchmark(
                device=device,
                steps=args.mm_steps,
                batch_size=args.batch_size,
                model_variant=variant,
                use_improvements=args.improve,
            )
            for r in mm_results:
                print("TASK_RESULT", json.dumps(r, ensure_ascii=False))
            all_results.extend(mm_results)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    summary = {
        "total_tasks": len(all_results),
        "env": env_info,
        "results": [
            {
                "task": r.get("task"),
                "domain": r.get("domain"),
                "status": r.get("status"),
                "val_acc": r.get("val_acc"),
                "val_loss": r.get("val_loss"),
                "train_acc": r.get("train_last_acc"),
                "model_variant": r.get("model_variant"),
                "improvements": r.get("improvements"),
            }
            for r in all_results
        ],
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))

    return all_results


if __name__ == "__main__":
    main()
