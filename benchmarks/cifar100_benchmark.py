from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

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


class Cifar100PickleDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        label_kind: str,
        augment: bool = False,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if label_kind not in ("coarse", "fine"):
            raise ValueError("label_kind must be coarse or fine")

        self.root = root
        self.split = split
        self.label_kind = label_kind
        self.augment = augment
        with (root / split).open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        data = obj["data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
        labels = obj[f"{label_kind}_labels"]
        self.images = torch.from_numpy(data)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = self.images[index]
        y = self.labels[index]
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=(-1,))
            pad = 4
            padded = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
            top = random.randint(0, 2 * pad)
            left = random.randint(0, 2 * pad)
            x = padded[:, top : top + 32, left : left + 32]
        return x, y


def make_subset(dataset: Dataset, max_items: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    n = min(max_items, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, indices)


def build_model(num_classes: int, static_adr_steps: int, q_channels: int) -> FluidResidualSequenceNetwork:
    cfg = FluidResidualSequenceConfig(
        n=32,
        h=3,
        output_dim=num_classes,
        max_output_steps=1,
        static_adr_steps=static_adr_steps,
        q_channels=q_channels,
        q_depth=8,
        q_height=8,
        q_width=3,
        extractor_hidden_channels=48,
        reconstructor_hidden_channels=40,
        output_hidden_dim=256,
    )
    return FluidResidualSequenceNetwork(cfg)


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(image_to_field(x), mode="nonsequential").outputs[:, 0]
            loss = F.cross_entropy(logits, y)
            loss_sum += float(loss.cpu()) * y.numel()
            pred = logits.argmax(dim=-1)
            correct1 += int((pred == y).sum().cpu())
            k = min(5, logits.shape[-1])
            topk = logits.topk(k, dim=-1).indices
            correct5 += int((topk == y[:, None]).any(dim=-1).sum().cpu())
            total += y.numel()
    return {
        "loss": loss_sum / max(1, total),
        "top1_acc": correct1 / max(1, total),
        "top5_acc": correct5 / max(1, total),
    }


def run_task(
    *,
    root: Path,
    label_kind: str,
    device: torch.device,
    steps: int,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    augment: bool,
    static_adr_steps: int,
    q_channels: int,
) -> dict[str, object]:
    set_seed(7301 if label_kind == "coarse" else 7302)
    num_classes = 20 if label_kind == "coarse" else 100
    train_ds = make_subset(
        Cifar100PickleDataset(root, "train", label_kind, augment=augment),
        train_subset,
        7401,
    )
    val_ds = make_subset(
        Cifar100PickleDataset(root, "test", label_kind, augment=False),
        val_subset,
        7402,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    model = build_model(num_classes, static_adr_steps, q_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    train_iter = cycle(train_loader)
    t0 = time.perf_counter()
    last_loss = 0.0
    last_acc = 0.0
    for _ in range(steps):
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
    train_seconds = time.perf_counter() - t0
    val = evaluate(model, val_loader, device)
    return {
        "task": f"cifar100_{label_kind}",
        "status": "ok",
        "domain": "CV",
        "dataset": "local cifar-100-python",
        "label_kind": label_kind,
        "num_classes": num_classes,
        "input_shape": "[B,3,32,32] -> [B,32,32,3]",
        "steps": steps,
        "batch_size": batch_size,
        "train_subset": len(train_ds),
        "val_subset": len(val_ds),
        "augment": augment,
        "static_adr_steps": static_adr_steps,
        "q_channels": q_channels,
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_loss": last_loss,
        "train_last_acc": last_acc,
        "val_loss": val["loss"],
        "val_top1_acc": val["top1_acc"],
        "val_top5_acc": val["top5_acc"],
        "train_seconds": train_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "cifar-100-python"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tasks", default="coarse,fine")
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--val-subset", type=int, default=2000)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--static-adr-steps", type=int, default=4)
    parser.add_argument("--q-channels", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root)
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
                "root": str(root),
                "fluid_net_sha256_start": start_hash,
            },
            ensure_ascii=False,
        ),
    )

    selected = {x.strip() for x in args.tasks.split(",") if x.strip()}
    results = []
    for label_kind, base_steps in [("coarse", 1200), ("fine", 1600)]:
        if label_kind not in selected:
            continue
        result = run_task(
            root=root,
            label_kind=label_kind,
            device=device,
            steps=max(1, int(round(base_steps * args.steps_scale))),
            batch_size=args.batch_size,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            augment=args.augment,
            static_adr_steps=args.static_adr_steps,
            q_channels=args.q_channels,
        )
        results.append(result)
        print("TASK_RESULT", json.dumps(result, ensure_ascii=False))

    end_hash = sha256(MODEL_PATH)
    print(
        "SUMMARY",
        json.dumps(
            {
                "num_tasks": len(results),
                "model_file_unchanged": start_hash == end_hash,
                "fluid_net_sha256_start": start_hash,
                "fluid_net_sha256_end": end_hash,
                "results": [
                    {
                        "task": r["task"],
                        "top1": r["val_top1_acc"],
                        "top5": r["val_top5_acc"],
                        "loss": r["val_loss"],
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
