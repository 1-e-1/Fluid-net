import argparse
import hashlib
import json
import pickle
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fluid_net_pro_max import (
    FluidResidualSequenceConfig,
    FluidResidualSequenceNetwork,
)


MODEL_PATH = ROOT / "fluid_net_pro_max.py"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def output_tensor(out):
    return out.outputs[:, 0]


def train_classifier(model, make_batch, steps, batch_size, device, lr=1e-3):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    last_loss = 0.0
    last_acc = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y, mode = make_batch(batch_size, device)
        opt.zero_grad(set_to_none=True)
        logits = output_tensor(model(x, mode=mode, max_output_steps=1))
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        last_acc = float((logits.argmax(dim=1) == y).float().mean().detach().cpu())
    return last_loss, last_acc, time.perf_counter() - t0


@torch.no_grad()
def eval_classifier(model, make_batch, batches, batch_size, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for _ in range(batches):
        x, y, mode = make_batch(batch_size, device)
        logits = output_tensor(model(x, mode=mode, max_output_steps=1))
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.cpu())
        correct += int((logits.argmax(dim=1) == y).sum().cpu())
        total += int(y.numel())
    return loss_sum / total, correct / total


def make_pro_max_config(
    *,
    n,
    h,
    output_dim,
    q_channels=16,
    q_depth=2,
    q_height=2,
    q_width=4,
    static_adr_steps=2,
    max_viscosity=0.1,
    enable_step_conditioned_viscosity=False,
):
    return FluidResidualSequenceConfig(
        n=n,
        h=h,
        output_dim=output_dim,
        adr_hidden_channels=24,
        extractor_hidden_channels=24,
        reconstructor_hidden_channels=24,
        output_hidden_dim=96,
        max_output_steps=1,
        static_adr_steps=static_adr_steps,
        q_channels=q_channels,
        q_depth=q_depth,
        q_height=q_height,
        q_width=q_width,
        max_viscosity=max_viscosity,
        enable_step_conditioned_viscosity=enable_step_conditioned_viscosity,
        extractor_dropout_rate=0.02,
        reconstructor_dropout_rate=0.02,
    )


def synthetic_cv_batch(batch_size, device):
    n = 16
    images = torch.zeros(batch_size, n, n, 1, device=device)
    labels = torch.randint(0, 4, (batch_size,), device=device)
    for i in range(batch_size):
        cls = int(labels[i])
        if cls == 0:
            images[i, 3:13, 7:9, 0] = 1.0
        elif cls == 1:
            images[i, 7:9, 3:13, 0] = 1.0
        elif cls == 2:
            images[i, 4:12, 4:12, 0] = 1.0
        else:
            yy, xx = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device),
                indexing="ij",
            )
            images[i, ((xx - 8) ** 2 + (yy - 8) ** 2) <= 16, 0] = 1.0
    noise = 0.15 * torch.randn_like(images)
    return torch.clamp(images + noise, 0.0, 1.0), labels, "nonsequential"


def run_synthetic_cv(args, device):
    cfg = make_pro_max_config(n=16, h=1, output_dim=4, q_width=1)
    model = FluidResidualSequenceNetwork(cfg).to(device)
    loss, acc, seconds = train_classifier(
        model,
        synthetic_cv_batch,
        args.synthetic_steps,
        args.batch_size,
        device,
    )
    val_loss, val_acc = eval_classifier(
        model,
        synthetic_cv_batch,
        args.eval_batches,
        args.val_batch_size,
        device,
    )
    return {
        "task": "pro_max_synthetic_cv_shape",
        "domain": "CV",
        "status": "ok",
        "steps": args.synthetic_steps,
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_loss": loss,
        "train_last_acc": acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_seconds": seconds,
        "input_shape": "[B,16,16,1]",
    }


def run_digits(args, device):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    data = load_digits()
    x = (data.images.astype("float32") / 16.0)[..., None]
    y = data.target.astype("int64")
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=args.seed,
        stratify=y,
    )
    x_train = torch.tensor(x_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    x_val = torch.tensor(x_val, device=device)
    y_val = torch.tensor(y_val, device=device)

    cfg = make_pro_max_config(n=8, h=1, output_dim=10, q_width=1)
    model = FluidResidualSequenceNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(args.real_steps):
        idx = torch.randint(0, x_train.shape[0], (args.batch_size,), device=device)
        opt.zero_grad(set_to_none=True)
        logits = output_tensor(model(x_train[idx], mode="nonsequential", max_output_steps=1))
        loss = F.cross_entropy(logits, y_train[idx])
        loss.backward()
        opt.step()
    seconds = time.perf_counter() - t0
    with torch.no_grad():
        logits = output_tensor(model(x_val, mode="nonsequential", max_output_steps=1))
        val_loss = float(F.cross_entropy(logits, y_val).cpu())
        val_acc = float((logits.argmax(dim=1) == y_val).float().mean().cpu())
    return {
        "task": "pro_max_sklearn_digits",
        "domain": "CV",
        "status": "ok",
        "dataset": "sklearn.load_digits",
        "steps": args.real_steps,
        "params": sum(p.numel() for p in model.parameters()),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_seconds": seconds,
        "input_shape": "[B,8,8,1]",
    }


class SyntheticNLP(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.h)
        self.net = FluidResidualSequenceNetwork(cfg)

    def forward(self, tokens):
        x = self.embedding(tokens)
        return output_tensor(self.net(x, mode="sequential", max_output_steps=1))


def synthetic_nlp_tokens(batch_size, device, vocab_size=64, length=24):
    tokens = torch.randint(0, vocab_size, (batch_size, length), device=device)
    pos = ((tokens >= 8) & (tokens < 16)).sum(dim=1)
    neg = ((tokens >= 16) & (tokens < 24)).sum(dim=1)
    y = (pos >= neg).long()
    return tokens, y


def run_synthetic_nlp(args, device):
    cfg = make_pro_max_config(n=5, h=16, output_dim=2, q_width=4)
    model = SyntheticNLP(cfg, vocab_size=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(args.synthetic_steps):
        x, y = synthetic_nlp_tokens(args.batch_size, device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    seconds = time.perf_counter() - t0
    with torch.no_grad():
        x, y = synthetic_nlp_tokens(args.val_batch_size * args.eval_batches, device)
        logits = model(x)
        val_loss = float(F.cross_entropy(logits, y).cpu())
        val_acc = float((logits.argmax(dim=1) == y).float().mean().cpu())
    return {
        "task": "pro_max_synthetic_nlp_sentiment",
        "domain": "NLP",
        "status": "ok",
        "steps": args.synthetic_steps,
        "params": sum(p.numel() for p in model.parameters()),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_seconds": seconds,
        "sequence_shape": "[B,24,16]",
    }


def run_20newsgroups_bow(args, device):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    categories = [
        "comp.graphics",
        "rec.sport.baseball",
        "sci.med",
        "talk.politics.misc",
    ]
    raw = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    idx_train, idx_val = train_test_split(
        np.arange(len(raw.data)),
        test_size=min(300, max(80, len(raw.data) // 8)),
        random_state=args.seed,
        stratify=raw.target,
    )
    idx_train = idx_train[: args.real_text_train]
    idx_val = idx_val[: args.real_text_val]

    vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
    x_train_np = vectorizer.fit_transform([raw.data[i] for i in idx_train]).toarray().astype("float32")
    x_val_np = vectorizer.transform([raw.data[i] for i in idx_val]).toarray().astype("float32")
    y_train_np = raw.target[idx_train].astype("int64")
    y_val_np = raw.target[idx_val].astype("int64")

    x_train = torch.tensor(x_train_np.reshape(-1, 16, 16, 1), device=device)
    x_val = torch.tensor(x_val_np.reshape(-1, 16, 16, 1), device=device)
    y_train = torch.tensor(y_train_np, device=device)
    y_val = torch.tensor(y_val_np, device=device)

    cfg = make_pro_max_config(n=16, h=1, output_dim=4, q_width=1)
    model = FluidResidualSequenceNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(args.real_text_steps):
        idx = torch.randint(0, x_train.shape[0], (args.batch_size,), device=device)
        opt.zero_grad(set_to_none=True)
        logits = output_tensor(model(x_train[idx], mode="nonsequential", max_output_steps=1))
        loss = F.cross_entropy(logits, y_train[idx])
        loss.backward()
        opt.step()
    seconds = time.perf_counter() - t0
    with torch.no_grad():
        logits = output_tensor(model(x_val, mode="nonsequential", max_output_steps=1))
        val_loss = float(F.cross_entropy(logits, y_val).cpu())
        val_acc = float((logits.argmax(dim=1) == y_val).float().mean().cpu())
    return {
        "task": "pro_max_20newsgroups_tfidf_4class",
        "domain": "NLP",
        "status": "ok",
        "dataset": "sklearn.fetch_20newsgroups",
        "features": "TF-IDF 256 packed as [B,16,16,1]",
        "steps": args.real_text_steps,
        "params": sum(p.numel() for p in model.parameters()),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_seconds": seconds,
        "train_subset": int(x_train.shape[0]),
        "val_subset": int(x_val.shape[0]),
    }


def load_cifar100_split(root: Path, split: str):
    with (root / split).open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    x = obj["data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    x = np.transpose(x, (0, 2, 3, 1))
    coarse = np.asarray(obj["coarse_labels"], dtype="int64")
    fine = np.asarray(obj["fine_labels"], dtype="int64")
    return x, coarse, fine


@torch.no_grad()
def eval_topk(model, x_val, y_val, batch_size, device):
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    top5 = 0
    for start in range(0, x_val.shape[0], batch_size):
        x = x_val[start : start + batch_size]
        y = y_val[start : start + batch_size]
        logits = output_tensor(model(x, mode="nonsequential", max_output_steps=1))
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        total += int(y.numel())
        pred = logits.topk(min(5, logits.shape[1]), dim=1).indices
        top1 += int((pred[:, 0] == y).sum().cpu())
        top5 += int((pred == y[:, None]).any(dim=1).sum().cpu())
    return loss_sum / total, top1 / total, top5 / total


def run_cifar100(args, device, label_kind: str):
    root = Path(args.cifar_root)
    x_train_np, coarse_train_np, fine_train_np = load_cifar100_split(root, "train")
    x_val_np, coarse_val_np, fine_val_np = load_cifar100_split(root, "test")

    y_train_np = coarse_train_np if label_kind == "coarse" else fine_train_np
    y_val_np = coarse_val_np if label_kind == "coarse" else fine_val_np
    num_classes = 20 if label_kind == "coarse" else 100

    rng = np.random.default_rng(args.seed)
    train_idx = rng.permutation(len(x_train_np))[: args.cifar_train_subset]
    val_idx = rng.permutation(len(x_val_np))[: args.cifar_val_subset]

    x_train = torch.tensor(x_train_np[train_idx], device=device)
    y_train = torch.tensor(y_train_np[train_idx], device=device)
    x_val = torch.tensor(x_val_np[val_idx], device=device)
    y_val = torch.tensor(y_val_np[val_idx], device=device)

    mean = x_train.mean(dim=(0, 1, 2), keepdim=True)
    std = x_train.std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-4)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    cfg = make_pro_max_config(
        n=32,
        h=3,
        output_dim=num_classes,
        q_channels=args.cifar_q_channels,
        q_depth=2,
        q_height=2,
        q_width=3,
        static_adr_steps=args.cifar_static_adr_steps,
    )
    model = FluidResidualSequenceNetwork(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.perf_counter()
    last_loss = 0.0
    last_acc = 0.0
    for _ in range(args.cifar_steps):
        idx = torch.randint(0, x_train.shape[0], (args.batch_size,), device=device)
        opt.zero_grad(set_to_none=True)
        logits = output_tensor(model(x_train[idx], mode="nonsequential", max_output_steps=1))
        loss = F.cross_entropy(logits, y_train[idx])
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        last_acc = float((logits.argmax(dim=1) == y_train[idx]).float().mean().detach().cpu())
    seconds = time.perf_counter() - t0

    val_loss, top1, top5 = eval_topk(model, x_val, y_val, args.val_batch_size, device)
    return {
        "task": f"pro_max_cifar100_{label_kind}",
        "domain": "CV",
        "status": "ok",
        "dataset": "local cifar-100-python",
        "label_kind": label_kind,
        "num_classes": num_classes,
        "input_shape": "[B,32,32,3]",
        "steps": args.cifar_steps,
        "batch_size": args.batch_size,
        "train_subset": int(x_train.shape[0]),
        "val_subset": int(x_val.shape[0]),
        "static_adr_steps": args.cifar_static_adr_steps,
        "q_channels": args.cifar_q_channels,
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_loss": last_loss,
        "train_last_acc": last_acc,
        "val_loss": val_loss,
        "val_top1_acc": top1,
        "val_top5_acc": top5,
        "train_seconds": seconds,
    }


def run_cifar100_coarse(args, device):
    return run_cifar100(args, device, "coarse")


def run_cifar100_fine(args, device):
    return run_cifar100(args, device, "fine")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--synthetic-steps", type=int, default=60)
    parser.add_argument("--real-steps", type=int, default=80)
    parser.add_argument("--real-text-steps", type=int, default=60)
    parser.add_argument("--real-text-train", type=int, default=500)
    parser.add_argument("--real-text-val", type=int, default=160)
    parser.add_argument("--cifar-root", default=str(ROOT / "cifar-100-python"))
    parser.add_argument("--cifar-steps", type=int, default=80)
    parser.add_argument("--cifar-train-subset", type=int, default=1200)
    parser.add_argument("--cifar-val-subset", type=int, default=400)
    parser.add_argument("--cifar-static-adr-steps", type=int, default=2)
    parser.add_argument("--cifar-q-channels", type=int, default=16)
    parser.add_argument(
        "--tasks",
        default="synthetic_cv,digits,synthetic_nlp,20newsgroups",
        help=(
            "comma-separated: synthetic_cv,digits,synthetic_nlp,"
            "20newsgroups,cifar100_coarse,cifar100_fine"
        ),
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    seed_all(args.seed)
    device = pick_device(args.device)
    sha_start = file_sha256(MODEL_PATH)
    print(
        "ENV",
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "model_path": str(MODEL_PATH),
                "model_sha256_start": sha_start,
                "torch_threads": torch.get_num_threads(),
            },
            ensure_ascii=False,
        ),
    )

    runners = {
        "synthetic_cv": run_synthetic_cv,
        "digits": run_digits,
        "synthetic_nlp": run_synthetic_nlp,
        "20newsgroups": run_20newsgroups_bow,
        "cifar100_coarse": run_cifar100_coarse,
        "cifar100_fine": run_cifar100_fine,
    }
    results = []
    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        result = runners[task](args, device)
        results.append(result)
        print("TASK_RESULT", json.dumps(result, ensure_ascii=False))

    sha_end = file_sha256(MODEL_PATH)
    print(
        "SUMMARY",
        json.dumps(
            {
                "num_tasks": len(results),
                "model_file_unchanged": sha_start == sha_end,
                "model_sha256_start": sha_start,
                "model_sha256_end": sha_end,
                "results": results,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
