from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "fluid_net.py"
DATA_PATH = ROOT / "data" / "exchange_rate" / "exchange_rate.csv"
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


def load_exchange_rate(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected exchange_rate.csv with 8 columns."
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1).astype("float32")
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"Expected [T,8] data, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Data contains NaN/Inf values.")
    return data


class WindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        seq_len: int,
        pred_len: int,
    ) -> None:
        self.data = torch.from_numpy(data)
        self.indices = torch.from_numpy(indices.astype("int64"))
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        start = int(self.indices[item])
        mid = start + self.seq_len
        end = mid + self.pred_len
        return self.data[start:mid], self.data[mid:end]


def chronological_indices(
    total_len: int,
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_start = total_len - seq_len - pred_len + 1
    starts = np.arange(max_start, dtype=np.int64)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.8)

    target_start = starts + seq_len
    train = starts[target_start < train_end]
    val = starts[(target_start >= train_end) & (target_start < val_end)]
    test = starts[target_start >= val_end]
    return train, val, test


def build_model(
    num_vars: int,
    seq_len: int,
    pred_len: int,
    n: int,
    q_channels: int,
    step_conditioned: bool,
) -> FluidResidualSequenceNetwork:
    if seq_len > n * n:
        raise ValueError(f"seq_len={seq_len} exceeds capacity n*n={n*n}")
    cfg = FluidResidualSequenceConfig(
        n=n,
        h=num_vars,
        output_dim=num_vars,
        max_output_steps=pred_len,
        static_adr_steps=4,
        q_channels=q_channels,
        q_depth=2,
        q_height=2,
        q_width=num_vars,
        extractor_hidden_channels=48,
        reconstructor_hidden_channels=48,
        output_hidden_dim=192,
        enable_step_conditioned_adr=step_conditioned,
    )
    return FluidResidualSequenceNetwork(cfg)


def metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    err = pred - target
    mse = float(err.square().mean().cpu())
    mae = float(err.abs().mean().cpu())
    rmse = math.sqrt(mse)
    denom = (pred.abs() + target.abs()).clamp_min(1e-6)
    smape = float((2.0 * err.abs() / denom).mean().cpu())
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
    }


def evaluate_model(
    model: FluidResidualSequenceNetwork,
    loader: DataLoader,
    device: torch.device,
    residual_target: bool,
) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x, mode="sequential").outputs[:, : y.shape[1], :]
            if residual_target:
                out = out + x[:, -1:, :].repeat(1, y.shape[1], 1)
            preds.append(out.cpu())
            targets.append(y)
    return metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def evaluate_naive(loader: DataLoader) -> dict[str, float]:
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            pred = x[:, -1:, :].repeat(1, y.shape[1], 1)
            preds.append(pred)
            targets.append(y)
    return metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def run_forecast(
    *,
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    n: int,
    q_channels: int,
    batch_size: int,
    steps: int,
    device: torch.device,
    step_conditioned: bool,
    residual_target: bool,
) -> dict[str, object]:
    set_seed(8101 + pred_len + (1000 if step_conditioned else 0))
    train_idx, val_idx, test_idx = chronological_indices(len(data), seq_len, pred_len)

    train_mean = data[: int(len(data) * 0.7)].mean(axis=0, keepdims=True)
    train_std = data[: int(len(data) * 0.7)].std(axis=0, keepdims=True).clip(min=1e-6)
    normalized = (data - train_mean) / train_std

    train_ds = WindowDataset(normalized, train_idx, seq_len, pred_len)
    val_ds = WindowDataset(normalized, val_idx, seq_len, pred_len)
    test_ds = WindowDataset(normalized, test_idx, seq_len, pred_len)

    generator = torch.Generator()
    generator.manual_seed(9101)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(
        num_vars=data.shape[1],
        seq_len=seq_len,
        pred_len=pred_len,
        n=n,
        q_channels=q_channels,
        step_conditioned=step_conditioned,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_iter = iter(train_loader)
    t0 = time.perf_counter()
    last_loss = 0.0
    best_val = float("inf")
    best_state = None
    for step in range(1, steps + 1):
        model.train()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = model(x, mode="sequential").outputs[:, :pred_len, :]
        target = y
        if residual_target:
            target = y - x[:, -1:, :].repeat(1, pred_len, 1)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())

        if step == steps or step % max(50, steps // 5) == 0:
            val_metrics = evaluate_model(model, val_loader, device, residual_target)
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    sync(device)
    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device, residual_target)
    val_metrics = evaluate_model(model, val_loader, device, residual_target)
    naive_test = evaluate_naive(test_loader)
    return {
        "task": f"exchange_rate_forecast_{pred_len}",
        "status": "ok",
        "domain": "time_series",
        "dataset": "Exchange-Rate",
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n": n,
        "num_vars": data.shape[1],
        "q_channels": q_channels,
        "step_conditioned": step_conditioned,
        "residual_target": residual_target,
        "steps": steps,
        "batch_size": batch_size,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_mse": last_loss,
        "best_val_mse": best_val,
        "val": val_metrics,
        "test": test_metrics,
        "naive_last_value_test": naive_test,
        "mse_vs_naive_ratio": test_metrics["mse"] / max(naive_test["mse"], 1e-12),
        "mae_vs_naive_ratio": test_metrics["mae"] / max(naive_test["mae"], 1e-12),
        "train_seconds": train_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-lens", default="24,96")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--q-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--step-conditioned", action="store_true")
    parser.add_argument("--residual-target", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_path = Path(args.data_path)
    data = load_exchange_rate(data_path)
    start_hash = sha256(MODEL_PATH)
    print(
        "ENV",
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "data_path": str(data_path),
                "data_shape": list(data.shape),
                "fluid_net_sha256_start": start_hash,
            },
            ensure_ascii=False,
        ),
    )

    results = []
    for pred_len in [int(x) for x in args.pred_lens.split(",") if x.strip()]:
        base_steps = 1200 if pred_len <= 24 else 1600
        result = run_forecast(
            data=data,
            seq_len=args.seq_len,
            pred_len=pred_len,
            n=args.n,
            q_channels=args.q_channels,
            batch_size=args.batch_size,
            steps=max(1, int(round(base_steps * args.steps_scale))),
            device=device,
            step_conditioned=args.step_conditioned,
            residual_target=args.residual_target,
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
                        "pred_len": r["pred_len"],
                        "test_mse": r["test"]["mse"],
                        "test_mae": r["test"]["mae"],
                        "naive_mse": r["naive_last_value_test"]["mse"],
                        "mse_vs_naive_ratio": r["mse_vs_naive_ratio"],
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
