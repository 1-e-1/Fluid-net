# -*- coding: utf-8 -*-
"""
experiments_v3/vla/vla_benchmark.py
====================================
Vision-Language-Action (VLA) Benchmark for FluidNet

Tests:
  1. 2D Reaching Task: image(target+arm) + text → action(dx, dy, gripper)
  2. Pick-and-Place Task: image(objects) + text → pick(x,y) + place(x,y)

Models:
  - FluidNet-VLA (dual-stream: image FluidNet + text FluidNet)
  - FluidNet-VLA-Enhanced (with Fourier + ChannelMix)
  - CNN+MLP Baseline
  - Vanilla MLP Baseline
"""
from __future__ import annotations

import sys, time, json, math, random
from itertools import cycle
from pathlib import Path
from typing import Optional

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import (
    set_seed, FourierGlobalMix, ChannelMix, SpatialMix,
    CosineWarmupScheduler, img2field,
)

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

# ==================== VLA Dataset Generators ====================

def generate_reaching_dataset(n_samples=5000, img_size=28, noise_level=0.05):
    """2D Reaching Task: 图像中有target点和arm起点, 预测到达target的连续动作"""
    images = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    texts = np.zeros((n_samples, 8), dtype=np.float32)  # 8-dim text embedding
    actions = np.zeros((n_samples, 3), dtype=np.float32)  # dx, dy, gripper(0/1)

    for i in range(n_samples):
        # 随机target位置
        tx, ty = np.random.uniform(4, img_size - 4, 2)
        # 随机arm起点
        ax, ay = np.random.uniform(4, img_size - 4, 2)
        # 动作: 从arm到target的偏移
        dx, dy = (tx - ax) / img_size, (ty - ay) / img_size
        gripper = np.random.choice([0.0, 1.0])  # 0=close, 1=open

        # 绘制图像
        img = np.zeros((img_size, img_size), dtype=np.float32)
        # target: 圆形
        for r in range(-3, 4):
            for c in range(-3, 4):
                if r*r + c*c <= 9:
                    py, px = int(ty) + r, int(tx) + c
                    if 0 <= py < img_size and 0 <= px < img_size:
                        img[py, px] = 1.0
        # arm: 十字
        for d in range(-2, 3):
            py, px = int(ay) + d, int(ax)
            if 0 <= py < img_size and 0 <= px < img_size:
                img[py, px] = 0.5
            py, px = int(ay), int(ax) + d
            if 0 <= py < img_size and 0 <= px < img_size:
                img[py, px] = 0.5

        # 添加噪声
        img += np.random.randn(img_size, img_size) * noise_level
        img = np.clip(img, 0, 1)

        images[i, 0] = img
        # 文本: 将target位置编码为8-dim feature
        texts[i, 0] = tx / img_size
        texts[i, 1] = ty / img_size
        texts[i, 2] = ax / img_size
        texts[i, 3] = ay / img_size
        texts[i, 4] = gripper
        texts[i, 5:] = np.random.randn(3) * 0.1  # noise features
        actions[i] = [dx, dy, gripper]

    # 标准化图像
    images = (images - 0.5) * 2.0
    return images, texts, actions


def generate_pick_place_dataset(n_samples=5000, img_size=28, noise_level=0.03):
    """Pick-and-Place: 图像中有两个物体, 文本指定pick哪个, 输出pick+place位置"""
    images = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    texts = np.zeros((n_samples, 8), dtype=np.float32)
    actions = np.zeros((n_samples, 4), dtype=np.float32)  # pick_x, pick_y, place_x, place_y

    for i in range(n_samples):
        # 两个物体位置
        obj1_x, obj1_y = np.random.uniform(4, img_size // 2 - 2, 2)
        obj2_x, obj2_y = np.random.uniform(img_size // 2 + 2, img_size - 4, 2)

        # 随机选择pick哪个, place到另一个位置附近
        pick_obj = np.random.choice([0, 1])  # 0=obj1, 1=obj2
        if pick_obj == 0:
            px, py = obj1_x, obj1_y
            qx, qy = obj2_x + np.random.uniform(-3, 3), obj2_y + np.random.uniform(-3, 3)
        else:
            px, py = obj2_x, obj2_y
            qx, qy = obj1_x + np.random.uniform(-3, 3), obj1_y + np.random.uniform(-3, 3)

        # 约束place位置
        qx = np.clip(qx, 2, img_size - 2)
        qy = np.clip(qy, 2, img_size - 2)

        # 绘制图像: obj1=方形, obj2=圆形
        img = np.zeros((img_size, img_size), dtype=np.float32)

        # obj1: 3x3方块
        for r in range(-2, 3):
            for c in range(-2, 3):
                py_i, px_i = int(obj1_y) + r, int(obj1_x) + c
                if 0 <= py_i < img_size and 0 <= px_i < img_size:
                    img[py_i, px_i] = 0.7

        # obj2: 圆形
        for r in range(-3, 4):
            for c in range(-3, 4):
                if r*r + c*c <= 9:
                    py_i, px_i = int(obj2_y) + r, int(obj2_x) + c
                    if 0 <= py_i < img_size and 0 <= px_i < img_size:
                        img[py_i, px_i] = 0.9

        img += np.random.randn(img_size, img_size) * noise_level
        img = np.clip(img, 0, 1)
        images[i, 0] = img

        # 文本编码
        texts[i, 0] = pick_obj  # which object to pick
        texts[i, 1] = obj1_x / img_size
        texts[i, 2] = obj1_y / img_size
        texts[i, 3] = obj2_x / img_size
        texts[i, 4] = obj2_y / img_size
        texts[i, 5:] = np.random.randn(3) * 0.1

        actions[i] = [px / img_size, py / img_size, qx / img_size, qy / img_size]

    images = (images - 0.5) * 2.0
    return images, texts, actions


# ==================== VLA Models ====================

class VLAMLP(nn.Module):
    """Baseline: pure MLP"""
    def __init__(self, img_size=28, text_dim=8, action_dim=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size + text_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, img, txt, **kwargs):
        x = torch.cat([img.flatten(1), txt], dim=-1)
        return self.net(x)


class VLACNN(nn.Module):
    """Baseline: CNN + MLP"""
    def __init__(self, img_size=28, text_dim=8, action_dim=3, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + text_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, img, txt, **kwargs):
        img_feat = self.cnn(img)
        return self.fusion(torch.cat([img_feat, txt], dim=-1))


class VLAFluidNet(nn.Module):
    """FluidNet VLA: 双流(image+text) FluidNet → 融合 → action head"""
    def __init__(self, n_img=28, n_txt=8, h_img=1, h_txt=4, action_dim=3,
                 static_adr_steps=4):
        super().__init__()
        self.n_img, self.n_txt = n_img, n_txt
        self.h_txt = h_txt

        # 图像流: h=1, 28x28 field
        self.img_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_img, h=h_img, output_dim=64,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=5, q_height=5, q_width=1,
                extractor_hidden_channels=40,
                reconstructor_hidden_channels=40,
                output_hidden_dim=192,
                adr_hidden_channels=40,
            ))

        # 文本流: h=4, 8x8 field (文本reshaped)
        self.txt_proj = nn.Linear(8, n_txt * n_txt * h_txt)
        self.txt_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_txt, h=h_txt, output_dim=32,
                max_output_steps=1, static_adr_steps=3,
                q_channels=16, q_depth=2, q_height=2, q_width=1,
                extractor_hidden_channels=24,
                reconstructor_hidden_channels=24,
                output_hidden_dim=96,
                adr_hidden_channels=24,
            ))

        # 融合 + Action Head
        self.action_head = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, img, txt, mode="nonsequential", max_output_steps=1):
        B = img.shape[0]

        # 图像流
        img_f = img2field(img)
        img_out = self.img_fluid(img_f, mode=mode, max_output_steps=max_output_steps)
        img_feat = img_out.outputs[:, 0]  # [B, 64]

        # 文本流
        txt_f = self.txt_proj(txt).view(B, self.n_txt, self.n_txt, self.h_txt)
        txt_out = self.txt_fluid(txt_f, mode=mode, max_output_steps=max_output_steps)
        txt_feat = txt_out.outputs[:, 0]  # [B, 32]

        # 融合
        fused = torch.cat([img_feat, txt_feat], dim=-1)
        return self.action_head(fused)


class VLAFluidNetEnhanced(nn.Module):
    """增强版 VLA: 双流Fourier+FluidNet → action"""
    def __init__(self, n_img=28, n_txt=8, h_img=1, h_txt=4, action_dim=3,
                 static_adr_steps=4):
        super().__init__()
        self.n_img, self.n_txt = n_img, n_txt
        self.h_txt = h_txt

        # 图像流: h=1 + Fourier (1D FFT on 28x28 field works fine)
        self.img_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_img, h=h_img, output_dim=64,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=5, q_height=5, q_width=1,
                extractor_hidden_channels=48,
                reconstructor_hidden_channels=48,
                output_hidden_dim=256,
                adr_hidden_channels=48,
            ))

        # 文本流: h=4 with Fourier
        self.txt_proj = nn.Linear(8, n_txt * n_txt * h_txt)
        self.txt_fourier = FourierGlobalMix(n_txt, h_txt)
        self.txt_norm = nn.LayerNorm([n_txt, n_txt, h_txt])
        self.txt_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n_txt, h=h_txt, output_dim=32,
                max_output_steps=1, static_adr_steps=3,
                q_channels=16, q_depth=2, q_height=2, q_width=1,
                extractor_hidden_channels=32,
                reconstructor_hidden_channels=32,
                output_hidden_dim=128,
                adr_hidden_channels=32,
            ))

        self.action_head = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_dim),
        )

    def forward(self, img, txt, mode="nonsequential", max_output_steps=1):
        B = img.shape[0]

        img_f = img2field(img)
        img_out = self.img_fluid(img_f, mode=mode, max_output_steps=max_output_steps)
        img_feat = img_out.outputs[:, 0]

        txt_f = self.txt_proj(txt).view(B, self.n_txt, self.n_txt, self.h_txt)
        txt_f = txt_f + self.txt_norm(self.txt_fourier(txt_f))
        txt_out = self.txt_fluid(txt_f, mode=mode, max_output_steps=max_output_steps)
        txt_feat = txt_out.outputs[:, 0]

        return self.action_head(torch.cat([img_feat, txt_feat], dim=-1))


# ==================== Training & Evaluation ====================

def train_eval_vla(model, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
                   name="", label_smoothing=0.0):
    """VLA专用训练: MSE Loss + CosineWarmup"""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    criterion = nn.MSELoss()
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for _ in range(steps):
        model.train()
        img, txt, act = next(tr_iter)
        img, txt, act = img.to(device), txt.to(device), act.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(img, txt)
        loss = criterion(pred, act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    sec = time.perf_counter() - t0

    # 评估: MSE + MAE
    model.eval()
    total_mse, total_mae, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for img, txt, act in vl_ldr:
            img, txt, act = img.to(device), txt.to(device), act.to(device)
            pred = model(img, txt)
            total_mse += float(F.mse_loss(pred, act, reduction="sum").cpu())
            total_mae += float(F.l1_loss(pred, act, reduction="sum").cpu())
            n_samples += act.numel()

    params = sum(p.numel() for p in model.parameters())
    return {
        "model": name, "params": params,
        "val_mse": total_mse / n_samples,
        "val_mae": total_mae / n_samples,
        "seconds": sec,
    }


# ==================== Main Benchmark ====================

def run_vla_benchmark(task_name, data_generator, action_dim, device):
    print(f"\n{'='*60}")
    print(f"  VLA Task: {task_name} (action_dim={action_dim})")
    print(f"{'='*60}")

    images, texts, actions = data_generator()
    n_total = len(images)
    n_train = int(n_total * 0.8)

    idx = np.random.default_rng(42).permutation(n_total)
    tr_idx, vl_idx = idx[:n_train], idx[n_train:]

    tr_ds = TensorDataset(
        torch.from_numpy(images[tr_idx]),
        torch.from_numpy(texts[tr_idx]),
        torch.from_numpy(actions[tr_idx]))
    vl_ds = TensorDataset(
        torch.from_numpy(images[vl_idx]),
        torch.from_numpy(texts[vl_idx]),
        torch.from_numpy(actions[vl_idx]))

    tr_ldr = DataLoader(tr_ds, 64, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl_ds, 64)
    print(f"  Train: {len(tr_ds)} | Val: {len(vl_ds)}")

    results = []

    # 1. Vanilla MLP
    set_seed(42)
    m = VLAMLP(img_size=28, text_dim=8, action_dim=action_dim)
    r = train_eval_vla(m, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
                       name=f"{task_name[:8]}-MLP")
    r["task"] = task_name; results.append(r)
    print(f"  MLP:          MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f} P={r['params']:,}")

    # 2. CNN+MLP
    set_seed(42)
    m = VLACNN(img_size=28, text_dim=8, action_dim=action_dim)
    r = train_eval_vla(m, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
                       name=f"{task_name[:8]}-CNN")
    r["task"] = task_name; results.append(r)
    print(f"  CNN+MLP:      MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f} P={r['params']:,}")

    # 3. FluidNet-VLA
    set_seed(42)
    m = VLAFluidNet(n_img=28, n_txt=8, h_img=1, h_txt=4, action_dim=action_dim)
    r = train_eval_vla(m, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
                       name=f"{task_name[:8]}-Fluid")
    r["task"] = task_name; results.append(r)
    print(f"  FluidNet-VLA: MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f} P={r['params']:,}")

    # 4. FluidNet-VLA-Enhanced
    set_seed(42)
    m = VLAFluidNetEnhanced(n_img=28, n_txt=8, h_img=1, h_txt=4, action_dim=action_dim)
    r = train_eval_vla(m, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
                       name=f"{task_name[:8]}-Fluid+")
    r["task"] = task_name; results.append(r)
    print(f"  Fluid-Enh:    MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f} P={r['params']:,}")

    return results


# ==================== Run ====================
if __name__ == "__main__":
    set_seed(42)
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("=" * 65)
    print("  FLUIDNET VLA BENCHMARK")
    print("=" * 65)

    all_results = []

    # Task 1: Reaching (action_dim=3: dx, dy, gripper)
    r1 = run_vla_benchmark(
        "2D Reaching",
        lambda: generate_reaching_dataset(5000),
        action_dim=3,
        device=device,
    )
    all_results.extend(r1)

    # Task 2: Pick-and-Place (action_dim=4: pick_x, pick_y, place_x, place_y)
    r2 = run_vla_benchmark(
        "Pick&Place",
        lambda: generate_pick_place_dataset(5000),
        action_dim=4,
        device=device,
    )
    all_results.extend(r2)

    # ==================== Summary ====================
    print("\n" + "=" * 75)
    print("  VLA BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Model':<22s} {'Task':<14s} {'Params':>9s} {'MSE':>12s} {'MAE':>12s} {'Time':>6s}")
    print("-" * 72)

    for r in sorted(all_results, key=lambda x: (x["task"], x["val_mse"])):
        print(f"{r['model']:<22s} {r['task']:<14s} {r['params']:>9,} "
              f"{r['val_mse']:>10.6f} {r['val_mae']:>10.6f} {r['seconds']:>5.0f}s")

    # Per-task best
    print("\n--- Per-Task Best Model ---")
    for task in ["2D Reaching", "Pick&Place"]:
        task_results = [r for r in all_results if r["task"] == task]
        if task_results:
            best = min(task_results, key=lambda x: x["val_mse"])
            print(f"  {task:<14s}: {best['model']:<22s} MSE={best['val_mse']:.6f} MAE={best['val_mae']:.6f}")

    # Save
    out = ROOT / "experiments_v3" / "vla" / "vla_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
