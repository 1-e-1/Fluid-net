from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def image_to_field(x: Tensor) -> Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def make_subset(dataset, max_items: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    n = min(max_items, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, indices)


def corrupt(x: Tensor, kind: str) -> Tensor:
    if kind == "clean":
        return x
    if kind == "gaussian_noise_0p25":
        return (x + 0.25 * torch.randn_like(x)).clamp(0.0, 1.0)
    if kind == "shift_2px":
        shifted = torch.roll(x, shifts=(2, 2), dims=(-2, -1))
        shifted[:, :, :2, :] = 0.0
        shifted[:, :, :, :2] = 0.0
        return shifted
    if kind == "center_occlusion_8x8":
        y = x.clone()
        y[:, :, 10:18, 10:18] = 0.0
        return y
    raise ValueError(kind)


def evaluate(model, loader, device: torch.device, kind: str) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = corrupt(x.to(device), kind)
            y = y.to(device)
            logits = model(image_to_field(x), mode="nonsequential").outputs[:, 0]
            loss = F.cross_entropy(logits, y)
            loss_sum += float(loss.cpu()) * y.numel()
            correct += int((logits.argmax(dim=-1) == y).sum().cpu())
            total += y.numel()
    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset", type=int, default=6000)
    parser.add_argument("--val-subset", type=int, default=2000)
    args = parser.parse_args()

    set_seed(6201)
    device = torch.device(args.device)
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

    tfm = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tfm)
    val_dataset = datasets.MNIST(root=args.data_root, train=False, download=True, transform=tfm)
    train_ds = make_subset(train_dataset, args.train_subset, 6202)
    val_ds = make_subset(val_dataset, args.val_subset, 6203)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    cfg = FluidResidualSequenceConfig(
        n=28,
        h=1,
        output_dim=10,
        max_output_steps=1,
        static_adr_steps=4,
        q_channels=32,
        q_depth=7,
        q_height=7,
        q_width=1,
        extractor_hidden_channels=40,
        reconstructor_hidden_channels=40,
        output_hidden_dim=192,
    )
    model = FluidResidualSequenceNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    train_iter = cycle(train_loader)
    t0 = time.perf_counter()
    last_loss = 0.0
    last_acc = 0.0
    for _ in range(args.steps):
        model.train()
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(image_to_field(x), mode="nonsequential").outputs[:, 0]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
        last_acc = float((logits.argmax(dim=-1) == y).float().mean().cpu())
    sync(device)

    corruption_results = {
        kind: evaluate(model, val_loader, device, kind)
        for kind in [
            "clean",
            "gaussian_noise_0p25",
            "shift_2px",
            "center_occlusion_8x8",
        ]
    }
    end_hash = sha256(MODEL_PATH)
    result = {
        "task": "mnist_robustness",
        "status": "ok",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_subset": len(train_ds),
        "val_subset": len(val_ds),
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_loss": last_loss,
        "train_last_acc": last_acc,
        "train_seconds": time.perf_counter() - t0,
        "corruptions": corruption_results,
        "model_file_unchanged": start_hash == end_hash,
        "fluid_net_sha256_start": start_hash,
        "fluid_net_sha256_end": end_hash,
    }
    print("TASK_RESULT", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
