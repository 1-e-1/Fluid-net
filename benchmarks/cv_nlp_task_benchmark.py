from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "fluid_net.py"
sys.path.insert(0, str(ROOT))

from fluid_net import FluidResidualSequenceConfig, FluidResidualSequenceNetwork


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def r2_score(pred: Tensor, target: Tensor) -> float:
    ss_res = (pred - target).square().sum()
    ss_tot = (target - target.mean()).square().sum().clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def accuracy(logits: Tensor, target: Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().cpu())


def make_shape_images(
    batch_size: int,
    size: int,
    device: torch.device,
    noise_std: float = 0.18,
) -> tuple[Tensor, Tensor]:
    labels = torch.randint(0, 4, (batch_size,), device=device)
    images = noise_std * torch.randn(batch_size, size, size, device=device)

    for i, label in enumerate(labels.tolist()):
        if label == 0:
            col = random.randint(3, size - 4)
            images[i, :, col - 1 : col + 2] += 1.0
        elif label == 1:
            row = random.randint(3, size - 4)
            images[i, row - 1 : row + 2, :] += 1.0
        elif label == 2:
            offset = random.randint(-3, 3)
            for r in range(size):
                c = min(max(r + offset, 0), size - 1)
                images[i, r, c] += 1.0
                if c + 1 < size:
                    images[i, r, c + 1] += 0.7
        else:
            top = random.randint(2, size // 2 - 1)
            left = random.randint(2, size // 2 - 1)
            extent = random.randint(5, 8)
            bottom = min(top + extent, size - 3)
            right = min(left + extent, size - 3)
            images[i, top:bottom, left] += 1.0
            images[i, top:bottom, right] += 1.0
            images[i, top, left:right] += 1.0
            images[i, bottom, left:right] += 1.0

    images = images.clamp(-0.5, 1.5).unsqueeze(-1)
    return images, labels


def make_blob_images(
    batch_size: int,
    size: int,
    device: torch.device,
    noise_std: float = 0.08,
) -> tuple[Tensor, Tensor]:
    ys = torch.arange(size, device=device).view(1, size, 1)
    xs = torch.arange(size, device=device).view(1, 1, size)
    cx = torch.randint(2, size - 2, (batch_size,), device=device).float()
    cy = torch.randint(2, size - 2, (batch_size,), device=device).float()
    sigma = torch.empty(batch_size, device=device).uniform_(1.0, 2.2)
    dist2 = (xs - cx[:, None, None]).square() + (ys - cy[:, None, None]).square()
    clean = torch.exp(-dist2 / (2.0 * sigma[:, None, None].square()))
    noisy = (clean + noise_std * torch.randn_like(clean)).clamp(0.0, 1.2)
    target = torch.stack(
        [
            (cx / (size - 1)) * 2.0 - 1.0,
            (cy / (size - 1)) * 2.0 - 1.0,
        ],
        dim=-1,
    )
    return noisy.unsqueeze(-1), target.unsqueeze(1)


def make_denoise_images(
    batch_size: int,
    size: int,
    device: torch.device,
    noise_std: float = 0.20,
) -> tuple[Tensor, Tensor]:
    clean, _ = make_blob_images(batch_size, size, device, noise_std=0.0)
    clean = clean.squeeze(-1)
    for i in range(batch_size):
        if random.random() < 0.5:
            row = random.randint(3, size - 4)
            clean[i, row - 1 : row + 2, :] = torch.maximum(
                clean[i, row - 1 : row + 2, :],
                torch.full_like(clean[i, row - 1 : row + 2, :], 0.8),
            )
        else:
            col = random.randint(3, size - 4)
            clean[i, :, col - 1 : col + 2] = torch.maximum(
                clean[i, :, col - 1 : col + 2],
                torch.full_like(clean[i, :, col - 1 : col + 2], 0.8),
            )
    noisy = (clean + noise_std * torch.randn_like(clean)).clamp(0.0, 1.2)
    return noisy.unsqueeze(-1), clean


def make_sentiment_tokens(
    batch_size: int,
    length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tokens = torch.randint(17, vocab_size, (batch_size, length), device=device)
    pos_mask = torch.rand(batch_size, length, device=device) < 0.18
    neg_mask = torch.rand(batch_size, length, device=device) < 0.18
    tokens[pos_mask] = torch.randint(1, 9, (int(pos_mask.sum()),), device=device)
    tokens[neg_mask] = torch.randint(9, 17, (int(neg_mask.sum()),), device=device)
    pos = ((tokens >= 1) & (tokens <= 8)).sum(dim=1)
    neg = ((tokens >= 9) & (tokens <= 16)).sum(dim=1)
    labels = (pos > neg).long()
    ties = pos == neg
    if bool(ties.any()):
        labels[ties] = torch.randint(0, 2, (int(ties.sum()),), device=device)
    return tokens, labels


def make_topic_tokens(
    batch_size: int,
    length: int,
    vocab_size: int,
    num_topics: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    labels = torch.randint(0, num_topics, (batch_size,), device=device)
    tokens = torch.randint(num_topics * 8, vocab_size, (batch_size, length), device=device)
    topic_mask = torch.rand(batch_size, length, device=device) < 0.55
    rows, cols = topic_mask.nonzero(as_tuple=True)
    if rows.numel() > 0:
        base = labels[rows] * 8
        tokens[rows, cols] = base + torch.randint(0, 8, (rows.numel(),), device=device)
    return tokens, labels


def make_count_tokens(
    batch_size: int,
    length: int,
    vocab_size: int,
    num_groups: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tokens = torch.randint(0, vocab_size, (batch_size, length), device=device)
    counts = []
    for g in range(num_groups):
        lo = g * 8
        hi = lo + 8
        counts.append(((tokens >= lo) & (tokens < hi)).float().sum(dim=1))
    target = torch.stack(counts, dim=-1) / math.sqrt(length)
    return tokens, target.unsqueeze(1)


class FluidTextTask(nn.Module):
    def __init__(
        self,
        cfg: FluidResidualSequenceConfig,
        vocab_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.model = FluidResidualSequenceNetwork(cfg)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence = self.embedding(tokens)
        mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        return self.model(sequence, mode="sequential", sequence_mask=mask).outputs


def train_cv_classification(
    device: torch.device,
    steps: int,
    batch_size: int,
    val_batch_size: int,
) -> dict[str, object]:
    set_seed(1101)
    size = 16
    cfg = FluidResidualSequenceConfig(
        n=size,
        h=1,
        output_dim=4,
        max_output_steps=1,
        static_adr_steps=8,
        q_channels=24,
        q_depth=4,
        q_height=4,
        q_width=1,
    )
    model = FluidResidualSequenceNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        model.train()
        x, y = make_shape_images(batch_size, size, device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, mode="nonsequential").outputs[:, 0, :]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    sync(device)
    train_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        x, y = make_shape_images(val_batch_size, size, device)
        logits = model(x, mode="nonsequential").outputs[:, 0, :]
        val_loss = float(F.cross_entropy(logits, y).cpu())
        val_acc = accuracy(logits, y)
    return {
        "task": "cv_shape_classification",
        "domain": "CV",
        "metric": "accuracy",
        "config": asdict(cfg),
        "params": sum(p.numel() for p in model.parameters()),
        "steps": steps,
        "batch_size": batch_size,
        "train_last_loss": last_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_seconds": train_seconds,
    }


def train_cv_localization(
    device: torch.device,
    steps: int,
    batch_size: int,
    val_batch_size: int,
) -> dict[str, object]:
    set_seed(1201)
    size = 16
    cfg = FluidResidualSequenceConfig(
        n=size,
        h=1,
        output_dim=2,
        max_output_steps=1,
        static_adr_steps=8,
        q_channels=24,
        q_depth=4,
        q_height=4,
        q_width=1,
    )
    model = FluidResidualSequenceNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        model.train()
        x, target = make_blob_images(batch_size, size, device)
        opt.zero_grad(set_to_none=True)
        pred = model(x, mode="nonsequential").outputs[:, :1, :]
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    sync(device)
    train_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        x, target = make_blob_images(val_batch_size, size, device)
        pred = model(x, mode="nonsequential").outputs[:, :1, :]
        val_mse = float(F.mse_loss(pred, target).cpu())
        norm_mae = float((pred - target).abs().mean().cpu())
        pixel_mae = norm_mae * (size - 1) / 2.0
    return {
        "task": "cv_blob_localization",
        "domain": "CV",
        "metric": "pixel_mae",
        "config": asdict(cfg),
        "params": sum(p.numel() for p in model.parameters()),
        "steps": steps,
        "batch_size": batch_size,
        "train_last_loss": last_loss,
        "val_mse": val_mse,
        "norm_mae": norm_mae,
        "pixel_mae": pixel_mae,
        "val_r2": r2_score(pred, target),
        "train_seconds": train_seconds,
    }


def train_cv_denoising(
    device: torch.device,
    steps: int,
    batch_size: int,
    val_batch_size: int,
) -> dict[str, object]:
    set_seed(1301)
    size = 16
    cfg = FluidResidualSequenceConfig(
        n=size,
        h=1,
        output_dim=size,
        max_output_steps=size,
        static_adr_steps=8,
        q_channels=32,
        q_depth=8,
        q_height=8,
        q_width=1,
        extractor_hidden_channels=40,
        reconstructor_hidden_channels=40,
        output_hidden_dim=160,
    )
    model = FluidResidualSequenceNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        model.train()
        x, target = make_denoise_images(batch_size, size, device)
        opt.zero_grad(set_to_none=True)
        pred = model(x, mode="nonsequential").outputs[:, :size, :]
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    sync(device)
    train_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        x, target = make_denoise_images(val_batch_size, size, device)
        pred = model(x, mode="nonsequential").outputs[:, :size, :].clamp(0.0, 1.2)
        val_mse = float(F.mse_loss(pred, target).cpu())
        val_mae = float((pred - target).abs().mean().cpu())
        psnr = 10.0 * math.log10(1.0 / max(val_mse, 1e-12))
    return {
        "task": "cv_denoise_reconstruct",
        "domain": "CV",
        "metric": "psnr",
        "config": asdict(cfg),
        "params": sum(p.numel() for p in model.parameters()),
        "steps": steps,
        "batch_size": batch_size,
        "train_last_loss": last_loss,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "psnr_db": psnr,
        "val_r2": r2_score(pred, target),
        "train_seconds": train_seconds,
    }


def train_text_classification(
    device: torch.device,
    task_name: str,
    data_fn: Callable[[], tuple[Tensor, Tensor]],
    val_data_fn: Callable[[], tuple[Tensor, Tensor]],
    num_classes: int,
    vocab_size: int,
    length: int,
    steps: int,
    batch_size: int,
) -> dict[str, object]:
    set_seed(2101 if "sentiment" in task_name else 2201)
    embed_dim = 16
    cfg = FluidResidualSequenceConfig(
        n=5,
        h=embed_dim,
        output_dim=num_classes,
        max_output_steps=1,
        static_adr_steps=4,
        q_channels=24,
        q_depth=2,
        q_height=2,
        q_width=4,
    )
    model = FluidTextTask(cfg, vocab_size, embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        model.train()
        tokens, target = data_fn()
        opt.zero_grad(set_to_none=True)
        logits = model(tokens)[:, 0, :]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    sync(device)
    train_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        tokens, target = val_data_fn()
        logits = model(tokens)[:, 0, :]
        val_loss = float(F.cross_entropy(logits, target).cpu())
        val_acc = accuracy(logits, target)
    return {
        "task": task_name,
        "domain": "NLP",
        "metric": "accuracy",
        "config": asdict(cfg),
        "vocab_size": vocab_size,
        "sequence_length": length,
        "params": sum(p.numel() for p in model.parameters()),
        "steps": steps,
        "batch_size": batch_size,
        "train_last_loss": last_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_seconds": train_seconds,
    }


def train_text_count_regression(
    device: torch.device,
    steps: int,
    batch_size: int,
    val_batch_size: int,
) -> dict[str, object]:
    set_seed(2301)
    vocab_size = 64
    length = 24
    num_groups = 4
    embed_dim = 16
    cfg = FluidResidualSequenceConfig(
        n=5,
        h=embed_dim,
        output_dim=num_groups,
        max_output_steps=1,
        static_adr_steps=4,
        q_channels=24,
        q_depth=2,
        q_height=2,
        q_width=4,
    )
    model = FluidTextTask(cfg, vocab_size, embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        model.train()
        tokens, target = make_count_tokens(batch_size, length, vocab_size, num_groups, device)
        opt.zero_grad(set_to_none=True)
        pred = model(tokens)[:, :1, :]
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    sync(device)
    train_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        tokens, target = make_count_tokens(val_batch_size, length, vocab_size, num_groups, device)
        pred = model(tokens)[:, :1, :]
        val_mse = float(F.mse_loss(pred, target).cpu())
        val_mae = float((pred - target).abs().mean().cpu())
    return {
        "task": "nlp_token_group_count_regression",
        "domain": "NLP",
        "metric": "mae",
        "config": asdict(cfg),
        "vocab_size": vocab_size,
        "sequence_length": length,
        "params": sum(p.numel() for p in model.parameters()),
        "steps": steps,
        "batch_size": batch_size,
        "train_last_loss": last_loss,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "val_rmse": math.sqrt(val_mse),
        "val_r2": r2_score(pred, target),
        "train_seconds": train_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    start_hash = sha256(MODEL_PATH)
    print(
        "ENV",
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "fluid_net_sha256_start": start_hash,
            },
            ensure_ascii=False,
        ),
    )

    def scaled(base: int) -> int:
        return max(1, int(round(base * args.steps_scale)))

    results: list[dict[str, object]] = []
    task_specs = [
        lambda: train_cv_classification(device, scaled(600), args.batch_size, args.val_batch_size),
        lambda: train_cv_localization(device, scaled(700), args.batch_size, args.val_batch_size),
        lambda: train_cv_denoising(device, scaled(700), 64, 512),
        lambda: train_text_classification(
            device,
            "nlp_synthetic_sentiment",
            lambda: make_sentiment_tokens(args.batch_size, 24, 64, device),
            lambda: make_sentiment_tokens(args.val_batch_size, 24, 64, device),
            2,
            64,
            24,
            scaled(650),
            args.batch_size,
        ),
        lambda: train_text_classification(
            device,
            "nlp_topic_classification",
            lambda: make_topic_tokens(args.batch_size, 24, 80, 4, device),
            lambda: make_topic_tokens(args.val_batch_size, 24, 80, 4, device),
            4,
            80,
            24,
            scaled(650),
            args.batch_size,
        ),
        lambda: train_text_count_regression(device, scaled(700), args.batch_size, args.val_batch_size),
    ]

    for run_task in task_specs:
        result = run_task()
        results.append(result)
        print("TASK_RESULT", json.dumps(result, ensure_ascii=False))

    end_hash = sha256(MODEL_PATH)
    summary = {
        "num_tasks": len(results),
        "fluid_net_sha256_start": start_hash,
        "fluid_net_sha256_end": end_hash,
        "model_file_unchanged": start_hash == end_hash,
        "results": [
            {
                "task": r["task"],
                "domain": r["domain"],
                "primary_metric": r["metric"],
                "value": r.get("val_acc", r.get("pixel_mae", r.get("psnr_db", r.get("val_mae")))),
            }
            for r in results
        ],
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
