# -*- coding: utf-8 -*-
"""
experiments_v3/vla/benchmark_quad_modal.py
===========================================
四模态 FluidNet VLA 完整 Benchmark

测试:
  1. QuadModalFluidVLA_Lite (完整4模态)
  2. Ablation: 去掉世界模型 (3模态)
  3. Ablation: 去掉文本 (3模态)
  4. Ablation: 仅视觉+状态 (2模态)
  5. Baseline: MLP (4模态拼接)
  6. Baseline: CNN+MLP

数据集: 仿真机器人Pick-and-Place
  - 视觉: 渲染的3通道场景图 (模拟)
  - 文本: 指令embedding (预计算768d)
  - 状态: 7-DoF关节角度+角速度
  - 世界: 预测的物体未来位姿
  - 动作: 7-DoF关节增量
"""
from __future__ import annotations

import sys, time, json, random, math
from pathlib import Path
from itertools import cycle
from typing import Optional

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
sys.path.insert(0, str(ROOT / "experiments_v3" / "vla"))

import fluid_net as fn
from improved_fluid import (
    set_seed, FourierGlobalMix, ChannelMix,
    CosineWarmupScheduler, img2field,
)
from quad_modal_vla import (
    QuadModalFluidVLA_Lite, QuadModalFusion,
    RobotStateEncoder, WorldModelEncoder,
)

# ============================================================================
# 仿真机器人VLA数据集
# ============================================================================

class SimRobotVLADataset(Dataset):
    """
    仿真Pick-and-Place任务数据集

    场景描述:
      - 桌面有若干物体 (方块/球体)
      - 机器人需要根据指令抓取指定物体并放到指定位置
      - 每个样本包含4个模态 + ground truth动作
    """
    def __init__(self, n_samples: int = 5000,
                 img_size: int = 224,
                 state_dim: int = 14,
                 world_dim: int = 20,
                 text_dim: int = 768,
                 action_dim: int = 7,
                 noise_level: float = 0.05):
        self.n_samples = n_samples
        self.img_size = img_size
        self.action_dim = action_dim

        rng = np.random.default_rng(42)

        # 生成数据
        self.images = np.zeros((n_samples, 3, img_size, img_size), dtype=np.float32)
        self.text_features = np.zeros((n_samples, text_dim), dtype=np.float32)
        self.robot_states = np.zeros((n_samples, state_dim), dtype=np.float32)
        self.world_states = np.zeros((n_samples, world_dim), dtype=np.float32)
        self.actions = np.zeros((n_samples, action_dim), dtype=np.float32)

        for i in range(n_samples):
            # --- 视觉: 模拟场景 (3通道: 物体位置 + 机器人 + 桌面) ---
            img = self._render_scene(rng, img_size)
            self.images[i] = (img - 0.5) * 2.0

            # --- 文本: 预计算的指令embedding ---
            # 模拟不同类型的指令:
            # "pick the [cube/sphere] and place it on the [left/right]"
            obj_type = rng.integers(0, 3)   # 0=cube, 1=sphere, 2=cylinder
            place_loc = rng.integers(0, 3)  # 0=left, 1=right, 2=center
            # 用确定性方式编码 (模拟预训练embedding)
            txt_embed = np.zeros(text_dim, dtype=np.float32)
            txt_embed[obj_type * 64 + rng.integers(0, 64)] = 1.0
            txt_embed[place_loc * 64 + 256 + rng.integers(0, 64)] = 1.0
            txt_embed += rng.normal(0, 0.02, text_dim)
            txt_embed = txt_embed / (np.linalg.norm(txt_embed) + 1e-8)
            self.text_features[i] = txt_embed

            # --- 机器人状态: 7-DoF关节状态 ---
            joint_angles = rng.normal(0, 0.5, 7)  # 当前关节角度
            joint_vels = rng.normal(0, 0.1, 7)    # 当前角速度
            self.robot_states[i] = np.concatenate([joint_angles, joint_vels])

            # --- 世界模型: 预测的未来状态 ---
            # 包含: 物体位姿(6D) + 物体速度(6D) + 接触力(3D) + 不确定性(5D)
            obj_pose = rng.uniform(-1, 1, 6)
            obj_vel = rng.normal(0, 0.1, 6)
            contact = rng.uniform(0, 1, 3)
            uncertainty = rng.uniform(0, 0.5, 5)
            self.world_states[i] = np.concatenate([obj_pose, obj_vel, contact, uncertainty])

            # --- 动作: 目标关节增量 (7-DoF) ---
            # 动作应该依赖于场景: 不同物体需要不同的关节轨迹
            target_pose = np.array([
                place_loc * 0.3 - 0.3,   # x target
                obj_type * 0.2 - 0.2,    # y target
                rng.uniform(0.1, 0.5),   # z target
            ])
            # 用IK-like映射生成关节增量
            base_action = np.zeros(7)
            base_action[0] = target_pose[0] * 0.5
            base_action[1] = target_pose[1] * 0.3
            base_action[2] = target_pose[2] * 0.4
            base_action[3] = obj_type * 0.15
            base_action[4] = place_loc * 0.1
            base_action[5] = rng.normal(0, 0.05)
            base_action[6] = rng.normal(0, 0.05)
            base_action += rng.normal(0, noise_level, 7)
            self.actions[i] = np.clip(base_action, -1, 1)

    def _render_scene(self, rng, size):
        """渲染模拟场景: 3通道图像"""
        img = np.zeros((3, size, size), dtype=np.float32)

        # 通道0: 物体位置 (R)
        obj_x = int(rng.uniform(size * 0.2, size * 0.8))
        obj_y = int(rng.uniform(size * 0.2, size * 0.8))
        obj_r = int(rng.uniform(8, 20))
        for dy in range(-obj_r, obj_r + 1):
            for dx in range(-obj_r, obj_r + 1):
                if dx*dx + dy*dy <= obj_r*obj_r:
                    py, px = obj_y + dy, obj_x + dx
                    if 0 <= py < size and 0 <= px < size:
                        img[0, py, px] = 0.8

        # 通道1: 机器人末端 (G)
        robot_x = int(rng.uniform(size * 0.3, size * 0.7))
        robot_y = int(rng.uniform(size * 0.3, size * 0.7))
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                py, px = robot_y + dy, robot_x + dx
                if 0 <= py < size and 0 <= px < size:
                    img[1, py, px] = 0.6

        # 通道2: 目标位置 (B)
        target_x = int(rng.uniform(size * 0.1, size * 0.9))
        target_y = int(rng.uniform(size * 0.1, size * 0.9))
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                py, px = target_y + dy, target_x + dx
                if 0 <= py < size and 0 <= px < size:
                    img[2, py, px] = 0.7

        # 噪声
        img += rng.normal(0, 0.02, img.shape)

        return np.clip(img, 0, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.images[idx]),
            torch.from_numpy(self.text_features[idx]),
            torch.from_numpy(self.robot_states[idx]),
            torch.from_numpy(self.world_states[idx]),
            torch.from_numpy(self.actions[idx]),
        )


# ============================================================================
# Baseline Models
# ============================================================================

class MLPBaseline(nn.Module):
    """4模态拼接 → MLP → 动作"""
    def __init__(self, img_size=224, text_dim=768, state_dim=14,
                 world_dim=20, action_dim=7, hidden=512):
        super().__init__()
        # 轻量图像编码
        self.img_enc = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 64),
        )
        input_dim = 64 + text_dim + state_dim + world_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, action_dim),
        )

    def forward(self, img, txt, state, world):
        feat = self.img_enc(img)
        x = torch.cat([feat, txt, state, world], dim=-1)
        return self.net(x)


class CNNBaseline(nn.Module):
    """纯CNN + 模态拼接 → 动作"""
    def __init__(self, img_size=224, text_dim=768, state_dim=14,
                 world_dim=20, action_dim=7, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        input_dim = 256 + text_dim + state_dim + world_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, img, txt, state, world):
        feat = self.cnn(img)
        x = torch.cat([feat, txt, state, world], dim=-1)
        return self.head(x)


# ============================================================================
# Training
# ============================================================================

def train_vla(model, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
              name="", is_fluid=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    criterion = nn.MSELoss()
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for _ in range(steps):
        model.train()
        img, txt, state, world, action = next(tr_iter)
        img = img.to(device); txt = txt.to(device)
        state = state.to(device); world = world.to(device)
        action = action.to(device)

        opt.zero_grad(set_to_none=True)
        if is_fluid:
            pred = model(img, txt, state, world, mode="nonsequential", max_output_steps=1)
        else:
            pred = model(img, txt, state, world)
        criterion(pred, action).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    sec = time.perf_counter() - t0

    # Evaluate
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    n_action_dims = 0
    with torch.no_grad():
        for img, txt, state, world, action in vl_ldr:
            img = img.to(device); txt = txt.to(device)
            state = state.to(device); world = world.to(device)
            action = action.to(device)

            if is_fluid:
                pred = model(img, txt, state, world, mode="nonsequential", max_output_steps=1)
            else:
                pred = model(img, txt, state, world)

            total_mse += float(F.mse_loss(pred, action, reduction="sum").cpu())
            total_mae += float(F.l1_loss(pred, action, reduction="sum").cpu())
            n_action_dims += action.numel()

    params = sum(p.numel() for p in model.parameters())
    return {
        "model": name, "params": params,
        "val_mse": total_mse / n_action_dims,
        "val_mae": total_mae / n_action_dims,
        "seconds": sec,
    }


# ============================================================================
# Main Benchmark
# ============================================================================

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    print("=" * 70)
    print("  QUAD-MODAL FLUIDNET VLA BENCHMARK")
    print("=" * 70)

    # 数据集
    print("\nGenerating dataset...")
    ds = SimRobotVLADataset(n_samples=5000)
    n_train = 4000
    idx = np.random.default_rng(42).permutation(len(ds))
    tr_idx, vl_idx = idx[:n_train], idx[n_train:]

    tr_ds = torch.utils.data.Subset(ds, tr_idx.tolist())
    vl_ds = torch.utils.data.Subset(ds, vl_idx.tolist())
    tr_ldr = DataLoader(tr_ds, 32, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl_ds, 32)
    print(f"  Train: {len(tr_ds)} | Val: {len(vl_ds)}")
    print(f"  Action dim: 7 (7-DoF joint deltas)")

    all_results = []

    # --- 1. MLP Baseline ---
    print("\n--- MLP Baseline ---")
    set_seed(42)
    m = MLPBaseline()
    r = train_vla(m, tr_ldr, vl_ldr, device, steps=400, lr=1e-3, name="MLP-Baseline")
    all_results.append(r)
    print(f"  MLP:           MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f} P={r['params']:,}")

    # --- 2. CNN Baseline ---
    print("\n--- CNN Baseline ---")
    set_seed(42)
    m = CNNBaseline()
    r = train_vla(m, tr_ldr, vl_ldr, device, steps=400, lr=1e-3, name="CNN-Baseline")
    all_results.append(r)
    print(f"  CNN:           MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f} P={r['params']:,}")

    # --- 3. QuadModalFluidVLA_Lite (完整4模态) ---
    print("\n--- QuadModalFluidVLA (4-modal) ---")
    set_seed(42)
    m = QuadModalFluidVLA_Lite(state_dim=14, world_dim=20, action_dim=7)
    r = train_vla(m, tr_ldr, vl_ldr, device, steps=400, lr=1.5e-3, wd=2e-4,
                  name="FluidVLA-4Modal", is_fluid=True)
    all_results.append(r)
    print(f"  FluidVLA-4:    MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f} P={r['params']:,}")

    # --- 4. Ablation: 无世界模型 (3模态) ---
    print("\n--- Ablation: No World Model ---")
    class FluidVLA_NoWorld(QuadModalFluidVLA_Lite):
        def forward(self, img, txt, state, world=None, **kwargs):
            B, device = img.shape[0], img.device
            img_feat = self.vision_cnn(img)
            txt_feat = self.text_mlp(txt) if txt is not None else torch.zeros(B, 256, device=device)
            state_feat = self.state_enc(state) if state is not None else torch.zeros(B, 256, device=device)
            world_feat = torch.zeros(B, 256, device=device)  # ZERO OUT
            fused, _ = self.fusion(img_feat, txt_feat, state_feat, world_feat)
            field = self.field_proj(fused).view(B, self.fluid_n, self.fluid_n, self.fluid_h)
            field = field + self.fourier_norm(self.fourier(field))
            field = self.channel_mix(field)
            out = self.adr(field, mode=kwargs.get("mode","nonsequential"),
                          max_output_steps=kwargs.get("max_output_steps",1))
            return out.outputs[:, 0]

    set_seed(42)
    m = FluidVLA_NoWorld(state_dim=14, world_dim=20, action_dim=7)
    r = train_vla(m, tr_ldr, vl_ldr, device, steps=400, lr=1.5e-3, wd=2e-4,
                  name="FluidVLA-NoWorld", is_fluid=True)
    all_results.append(r)
    print(f"  FluidVLA-NoW:  MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f}")

    # --- 5. Ablation: 无文本 ---
    print("\n--- Ablation: No Text ---")
    class FluidVLA_NoText(QuadModalFluidVLA_Lite):
        def forward(self, img, txt=None, state=None, world=None, **kwargs):
            B, device = img.shape[0], img.device
            img_feat = self.vision_cnn(img)
            txt_feat = torch.zeros(B, 256, device=device)
            state_feat = self.state_enc(state) if state is not None else torch.zeros(B, 256, device=device)
            world_feat = self.world_enc(world) if world is not None else torch.zeros(B, 256, device=device)
            fused, _ = self.fusion(img_feat, txt_feat, state_feat, world_feat)
            field = self.field_proj(fused).view(B, self.fluid_n, self.fluid_n, self.fluid_h)
            field = field + self.fourier_norm(self.fourier(field))
            field = self.channel_mix(field)
            out = self.adr(field, mode=kwargs.get("mode","nonsequential"),
                          max_output_steps=kwargs.get("max_output_steps",1))
            return out.outputs[:, 0]

    set_seed(42)
    m = FluidVLA_NoText(state_dim=14, world_dim=20, action_dim=7)
    r = train_vla(m, tr_ldr, vl_ldr, device, steps=400, lr=1.5e-3, wd=2e-4,
                  name="FluidVLA-NoText", is_fluid=True)
    all_results.append(r)
    print(f"  FluidVLA-NoT:  MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f}")

    # --- 6. Ablation: 仅视觉+状态 ---
    print("\n--- Ablation: Vision+State only ---")
    class FluidVLA_VS(QuadModalFluidVLA_Lite):
        def forward(self, img, txt=None, state=None, world=None, **kwargs):
            B, device = img.shape[0], img.device
            img_feat = self.vision_cnn(img)
            txt_feat = torch.zeros(B, 256, device=device)
            state_feat = self.state_enc(state) if state is not None else torch.zeros(B, 256, device=device)
            world_feat = torch.zeros(B, 256, device=device)
            fused, _ = self.fusion(img_feat, txt_feat, state_feat, world_feat)
            field = self.field_proj(fused).view(B, self.fluid_n, self.fluid_n, self.fluid_h)
            field = field + self.fourier_norm(self.fourier(field))
            field = self.channel_mix(field)
            out = self.adr(field, mode=kwargs.get("mode","nonsequential"),
                          max_output_steps=kwargs.get("max_output_steps",1))
            return out.outputs[:, 0]

    set_seed(42)
    m = FluidVLA_VS(state_dim=14, world_dim=20, action_dim=7)
    r = train_vla(m, tr_ldr, vl_ldr, device, steps=400, lr=1.5e-3, wd=2e-4,
                  name="FluidVLA-Vis+State", is_fluid=True)
    all_results.append(r)
    print(f"  FluidVLA-VS:   MSE={r['val_mse']:.6f} MAE={r['val_mae']:.6f}")

    # ==================== 汇总 ====================
    print("\n" + "=" * 75)
    print("  QUAD-MODAL VLA BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Model':<20s} {'Params':>9s} {'MSE':>10s} {'MAE':>10s} {'Time':>6s}")
    print("-" * 62)

    for r in sorted(all_results, key=lambda x: x["val_mse"]):
        print(f"{r['model']:<20s} {r['params']:>9,} {r['val_mse']:>10.6f} "
              f"{r['val_mae']:>10.6f} {r['seconds']:>5.0f}s")

    # 模态贡献分析
    print("\n--- Modality Contribution Analysis ---")
    base = next(r for r in all_results if r["model"] == "FluidVLA-Vis+State")
    full_4 = next(r for r in all_results if r["model"] == "FluidVLA-4Modal")
    no_world = next(r for r in all_results if r["model"] == "FluidVLA-NoWorld")
    no_text = next(r for r in all_results if r["model"] == "FluidVLA-NoText")

    print(f"  Base (Vis+State):    MSE={base['val_mse']:.6f}")
    print(f"  + Text:              MSE={no_world['val_mse']:.6f}  (delta={no_world['val_mse']-base['val_mse']:+.6f})")
    print(f"  + World:             MSE={no_text['val_mse']:.6f}  (delta={no_text['val_mse']-base['val_mse']:+.6f})")
    print(f"  + All 4:             MSE={full_4['val_mse']:.6f}  (delta={full_4['val_mse']-base['val_mse']:+.6f})")

    # Best vs Baseline
    best = min(all_results, key=lambda x: x["val_mse"])
    mlp = next(r for r in all_results if r["model"] == "MLP-Baseline")
    print(f"\n  Best FluidVLA: {best['val_mse']:.6f} vs MLP: {mlp['val_mse']:.6f} "
          f"({(1-best['val_mse']/mlp['val_mse'])*100:+.1f}% improvement)")

    # Save
    out = ROOT / "experiments_v3" / "vla" / "quad_modal_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
