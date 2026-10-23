from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

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


def accuracy(logits: Tensor, target: Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().cpu())


def make_subset(dataset, max_items: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    n = min(max_items, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return Subset(dataset, indices)


def image_to_field(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).contiguous()


class FluidTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        cfg = FluidResidualSequenceConfig(
            n=8,
            h=embed_dim,
            output_dim=num_classes,
            max_output_steps=1,
            static_adr_steps=4,
            q_channels=32,
            q_depth=2,
            q_height=2,
            q_width=8,
            extractor_hidden_channels=40,
            reconstructor_hidden_channels=40,
            output_hidden_dim=192,
        )
        if seq_len > cfg.capacity:
            raise ValueError(f"seq_len={seq_len} exceeds n*n={cfg.capacity}")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.net = FluidResidualSequenceNetwork(cfg)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence = self.embedding(tokens)
        mask = tokens.ne(0)
        return self.net(sequence, mode="sequential", sequence_mask=mask).outputs[:, 0]


def train_loader_classifier(
    *,
    task_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    is_image: bool,
) -> dict[str, object]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        if is_image:
            logits = model(image_to_field(x), mode="nonsequential").outputs[:, 0]
        else:
            logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
        last_acc = accuracy(logits.detach(), y)

    sync(device)
    train_seconds = time.perf_counter() - t0

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_image:
                logits = model(image_to_field(x), mode="nonsequential").outputs[:, 0]
            else:
                logits = model(x)
            loss = F.cross_entropy(logits, y)
            val_loss_sum += float(loss.cpu()) * y.numel()
            val_correct += int((logits.argmax(dim=-1) == y).sum().cpu())
            val_total += y.numel()

    return {
        "task": task_name,
        "status": "ok",
        "steps": steps,
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_loss": last_loss,
        "train_last_acc": last_acc,
        "val_loss": val_loss_sum / max(1, val_total),
        "val_acc": val_correct / max(1, val_total),
        "val_total": val_total,
        "train_seconds": train_seconds,
    }


def run_sklearn_digits(
    device: torch.device,
    steps: int,
    batch_size: int,
) -> dict[str, object]:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    set_seed(3101)
    data = load_digits()
    x = torch.tensor(data.images[:, None, :, :], dtype=torch.float32) / 16.0
    y = torch.tensor(data.target, dtype=torch.long)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=3101,
        stratify=y.numpy(),
    )
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256)
    cfg = FluidResidualSequenceConfig(
        n=8,
        h=1,
        output_dim=10,
        max_output_steps=1,
        static_adr_steps=4,
        q_channels=24,
        q_depth=4,
        q_height=4,
        q_width=1,
        output_hidden_dim=160,
    )
    model = FluidResidualSequenceNetwork(cfg)
    result = train_loader_classifier(
        task_name="sklearn_digits_8x8",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        steps=steps,
        lr=2e-3,
        is_image=True,
    )
    result["domain"] = "CV"
    result["dataset"] = "sklearn.load_digits"
    result["input_shape"] = "[B,1,8,8]"
    return result


def run_torchvision_dataset(
    *,
    name: str,
    device: torch.device,
    steps: int,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    data_root: Path,
) -> dict[str, object]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        return {"task": name, "status": "skipped", "reason": f"torchvision import failed: {exc}"}

    set_seed(3201 + abs(hash(name)) % 1000)
    if name == "mnist":
        dataset_cls = datasets.MNIST
        channels = 1
        size = 28
        tfm = transforms.Compose([transforms.ToTensor()])
    elif name == "fashion_mnist":
        dataset_cls = datasets.FashionMNIST
        channels = 1
        size = 28
        tfm = transforms.Compose([transforms.ToTensor()])
    elif name == "cifar10":
        dataset_cls = datasets.CIFAR10
        channels = 3
        size = 32
        tfm = transforms.Compose([transforms.ToTensor()])
    else:
        raise ValueError(name)

    try:
        train_dataset = dataset_cls(root=str(data_root), train=True, download=True, transform=tfm)
        val_dataset = dataset_cls(root=str(data_root), train=False, download=True, transform=tfm)
    except Exception as exc:
        return {"task": name, "status": "skipped", "reason": f"download/load failed: {exc}"}

    train_ds = make_subset(train_dataset, train_subset, 4101)
    val_ds = make_subset(val_dataset, val_subset, 4102)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    cfg = FluidResidualSequenceConfig(
        n=size,
        h=channels,
        output_dim=10,
        max_output_steps=1,
        static_adr_steps=4,
        q_channels=32,
        q_depth=7 if size == 28 else 8,
        q_height=7 if size == 28 else 8,
        q_width=channels,
        extractor_hidden_channels=40,
        reconstructor_hidden_channels=40,
        output_hidden_dim=192,
    )
    model = FluidResidualSequenceNetwork(cfg)
    result = train_loader_classifier(
        task_name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        steps=steps,
        lr=1.5e-3,
        is_image=True,
    )
    result["domain"] = "CV"
    result["dataset"] = f"torchvision.datasets.{dataset_cls.__name__}"
    result["input_shape"] = f"[B,{channels},{size},{size}]"
    result["train_subset"] = len(train_ds)
    result["val_subset"] = len(val_ds)
    return result


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def encode_texts(texts: Iterable[str], vocab: dict[str, int], seq_len: int) -> Tensor:
    encoded = []
    for text in texts:
        ids = [vocab.get(tok, 1) for tok in tokenize(text)[:seq_len]]
        ids = ids + [0] * (seq_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


def run_20newsgroups(
    device: torch.device,
    steps: int,
    batch_size: int,
    train_subset: int,
    val_subset: int,
) -> dict[str, object]:
    try:
        from sklearn.datasets import fetch_20newsgroups
    except Exception as exc:
        return {"task": "20newsgroups_4class", "status": "skipped", "reason": f"sklearn import failed: {exc}"}

    set_seed(5101)
    categories = [
        "comp.graphics",
        "rec.sport.baseball",
        "sci.med",
        "talk.politics.misc",
    ]
    try:
        train_raw = fetch_20newsgroups(
            subset="train",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
        val_raw = fetch_20newsgroups(
            subset="test",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
    except Exception as exc:
        return {"task": "20newsgroups_4class", "status": "skipped", "reason": f"download/load failed: {exc}"}

    train_idx = np.random.default_rng(5101).choice(
        len(train_raw.data),
        size=min(train_subset, len(train_raw.data)),
        replace=False,
    )
    val_idx = np.random.default_rng(5102).choice(
        len(val_raw.data),
        size=min(val_subset, len(val_raw.data)),
        replace=False,
    )
    train_texts = [train_raw.data[i] for i in train_idx]
    train_y = torch.tensor([train_raw.target[i] for i in train_idx], dtype=torch.long)
    val_texts = [val_raw.data[i] for i in val_idx]
    val_y = torch.tensor([val_raw.target[i] for i in val_idx], dtype=torch.long)

    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))
    vocab_size = 8000
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)

    seq_len = 64
    train_x = encode_texts(train_texts, vocab, seq_len)
    val_x = encode_texts(val_texts, vocab, seq_len)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    model = FluidTextClassifier(
        vocab_size=len(vocab),
        embed_dim=32,
        num_classes=len(categories),
        seq_len=seq_len,
    )
    result = train_loader_classifier(
        task_name="20newsgroups_4class",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        steps=steps,
        lr=1.5e-3,
        is_image=False,
    )
    result["domain"] = "NLP"
    result["dataset"] = "sklearn.fetch_20newsgroups"
    result["categories"] = categories
    result["sequence_length"] = seq_len
    result["vocab_size"] = len(vocab)
    result["train_subset"] = len(train_x)
    result["val_subset"] = len(val_x)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset", type=int, default=6000)
    parser.add_argument("--val-subset", type=int, default=1500)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument(
        "--tasks",
        default="digits,mnist,fashion_mnist,cifar10,20newsgroups",
        help="comma-separated: digits,mnist,fashion_mnist,cifar10,20newsgroups",
    )
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

    selected = {task.strip() for task in args.tasks.split(",") if task.strip()}
    results: list[dict[str, object]] = []

    if "digits" in selected:
        results.append(run_sklearn_digits(device, scaled(700), args.batch_size))
        print("TASK_RESULT", json.dumps(results[-1], ensure_ascii=False))

    for name, base_steps in [
        ("mnist", 900),
        ("fashion_mnist", 1000),
        ("cifar10", 900),
    ]:
        if name in selected:
            results.append(
                run_torchvision_dataset(
                    name=name,
                    device=device,
                    steps=scaled(base_steps),
                    batch_size=args.batch_size,
                    train_subset=args.train_subset,
                    val_subset=args.val_subset,
                    data_root=Path(args.data_root),
                )
            )
            print("TASK_RESULT", json.dumps(results[-1], ensure_ascii=False))

    if "20newsgroups" in selected:
        results.append(
            run_20newsgroups(
                device=device,
                steps=scaled(1100),
                batch_size=args.batch_size,
                train_subset=args.train_subset,
                val_subset=args.val_subset,
            )
        )
        print("TASK_RESULT", json.dumps(results[-1], ensure_ascii=False))

    end_hash = sha256(MODEL_PATH)
    summary = {
        "num_tasks": len(results),
        "model_file_unchanged": start_hash == end_hash,
        "fluid_net_sha256_start": start_hash,
        "fluid_net_sha256_end": end_hash,
        "results": [
            {
                "task": r.get("task"),
                "status": r.get("status"),
                "domain": r.get("domain"),
                "val_acc": r.get("val_acc"),
                "val_loss": r.get("val_loss"),
            }
            for r in results
        ],
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
