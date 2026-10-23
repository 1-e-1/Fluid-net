# -*- coding: utf-8 -*-
"""
experiments_v3/vla/test_pusht.py
================================
在真实 PushT 数据集上测试 QuadModalFluidVLA

PushT 数据集:
  - 206 episodes, 25650 frames
  - State: [x, y]  T型块位姿
  - Action: [x, y]  推杆目标位置
  - 任务: 将T型块推到目标区域

测试模型:
  1. QuadModalFluidVLA_Lite (state+world+text→action)
  2. MLP Baseline
  3. 模态消融
"""
from __future__ import annotations
import sys, time, json, random
from pathlib import Path
from itertools import cycle
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
sys.path.insert(0, str(ROOT / "experiments_v3" / "vla"))

import fluid_net as fn
from improved_fluid import set_seed, CosineWarmupScheduler
from quad_modal_vla import QuadModalFluidVLA_Lite

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

# ===================== PushT Dataset =====================
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

class PushTDataset(Dataset):
    """PushT → QuadModal VLA 格式转换"""

    def __init__(self, split="train", train_ratio=0.8, seq_len=1,
                 pred_horizon=1):
        # 下载/加载数据
        data_path = hf_hub_download(
            "lerobot/pusht", "data/chunk-000/file-000.parquet",
            repo_type="dataset"
        )
        df = pq.read_table(data_path).to_pandas()

        # 按 episode 分割
        episodes = df.groupby("episode_index")
        ep_indices = sorted(df["episode_index"].unique())
        n_ep = len(ep_indices)
        n_train = int(n_ep * train_ratio)

        rng = np.random.default_rng(42)
        rng.shuffle(ep_indices)
        self.ep_indices = ep_indices[:n_train] if split == "train" else ep_indices[n_train:]

        # 构建样本: (state_t, action_t) 序列
        self.samples = []
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        for ep_idx in self.ep_indices:
            ep_data = episodes.get_group(ep_idx)
            states = np.stack(ep_data["observation.state"].values)  # [T, 2]
            actions = np.stack(ep_data["action"].values)            # [T, 2]

            T = len(states)
            for t in range(T - seq_len - pred_horizon + 1):
                s = states[t:t+seq_len].copy()    # [seq_len, 2]
                a = actions[t+seq_len:t+seq_len+pred_horizon].copy()  # [pred_horizon, 2]
                self.samples.append((s, a))

        # 归一化参数
        all_states = np.concatenate([s for s, _ in self.samples], axis=0)
        all_actions = np.concatenate([a for _, a in self.samples], axis=0)
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-6
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0) + 1e-6

        print(f"  PushT-{split}: {len(self.samples)} samples, {len(self.ep_indices)} episodes")
        print(f"  State  range: [{self.state_mean[0]:.0f},{self.state_mean[1]:.0f}] +- [{self.state_std[0]:.0f},{self.state_std[1]:.0f}]")
        print(f"  Action range: [{self.action_mean[0]:.0f},{self.action_mean[1]:.0f}] +- [{self.action_std[0]:.0f},{self.action_std[1]:.0f}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, action = self.samples[idx]

        # 归一化
        state_norm = (state - self.state_mean) / self.state_std  # [seq_len, 2]
        action_norm = (action - self.action_mean) / self.action_std  # [pred_horizon, 2]

        # 构建 QuadModal 输入:
        # - robot_state: 当前状态展开为14维 (填零)
        robot_state = np.zeros(14, dtype=np.float32)
        robot_state[:2] = state_norm[-1]  # 最新状态

        # - world_state: 预测未来（模拟）
        world_state = np.zeros(20, dtype=np.float32)
        # 简单世界模型: 用速度估计未来位置
        if len(state_norm) >= 2:
            vel = state_norm[-1] - state_norm[-2]
            world_state[:2] = state_norm[-1] + vel * 3  # 预测3步后
            world_state[2:4] = vel
        world_state[4] = np.linalg.norm(vel)  # 速度大小

        # - 模拟文本特征 (768维, 基于状态编码)
        text_feat = np.zeros(768, dtype=np.float32)
        # 将状态位置离散化编码到文本空间
        x_bin = int(np.clip(state_norm[-1, 0] + 3, 0, 6) * 42)
        y_bin = int(np.clip(state_norm[-1, 1] + 3, 0, 6) * 42)
        text_feat[min(x_bin, 767)] = 1.0
        text_feat[min(256 + y_bin, 767)] = 1.0
        text_feat += np.random.randn(768) * 0.01

        # - 模拟图像 (3×224×224, 基于状态渲染)
        img = np.zeros((3, 224, 224), dtype=np.float32)
        sx = int(np.clip(state_norm[-1, 0] * 30 + 112, 10, 214))
        sy = int(np.clip(state_norm[-1, 1] * 30 + 112, 10, 214))
        # 绘制T型块
        for dy in range(-8, 9):
            for dx in range(-4, 5):
                py, px = sy + dy, sx + dx
                if 0 <= py < 224 and 0 <= px < 224:
                    img[0, py, px] = 0.8
        for dy in range(-3, 4):
            for dx in range(-8, 9):
                py, px = sy + dy + 10, sx + dx
                if 0 <= py < 224 and 0 <= px < 224:
                    img[0, py, px] = 0.8
        # 目标区域
        for dy in range(-12, 13):
            for dx in range(-12, 13):
                if dx*dx + dy*dy <= 144:
                    py, px = 112 + dy, 160 + dx
                    if 0 <= py < 224 and 0 <= px < 224:
                        img[1, py, px] = 0.4
        img = (img - 0.5) * 2.0

        return (
            torch.from_numpy(img),
            torch.from_numpy(text_feat),
            torch.from_numpy(robot_state),
            torch.from_numpy(world_state),
            torch.from_numpy(action_norm.flatten()),  # [pred_horizon*2]
        )


# ===================== Baseline Models =====================

class MLPBaseline(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 64))
        self.net = nn.Sequential(
            nn.Linear(64 + 768 + 14 + 20, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim))

    def forward(self, img, txt, state, world):
        return self.net(torch.cat([self.cnn(img), txt, state, world], -1))


# ===================== Training =====================

def train_push(model, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
               name="", is_fluid=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    crit = nn.MSELoss()
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
        crit(pred, action).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    sec = time.perf_counter() - t0

    model.eval()
    tmse, tmae, nd = 0.0, 0.0, 0
    with torch.no_grad():
        for img, txt, state, world, action in vl_ldr:
            img = img.to(device); txt = txt.to(device)
            state = state.to(device); world = world.to(device)
            action = action.to(device)
            if is_fluid:
                pred = model(img, txt, state, world, mode="nonsequential", max_output_steps=1)
            else:
                pred = model(img, txt, state, world)
            tmse += float(F.mse_loss(pred, action, reduction="sum").cpu())
            tmae += float(F.l1_loss(pred, action, reduction="sum").cpu())
            nd += action.numel()

    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_mse": tmse/nd,
            "val_mae": tmae/nd, "seconds": sec}


# ===================== Main =====================

if __name__ == "__main__":
    set_seed(42)
    print("=" * 65)
    print("  PushT REAL DATASET TEST — QuadModal FluidVLA")
    print("=" * 65)

    # 加载数据
    print("\nLoading PushT...")
    tr_ds = PushTDataset("train", seq_len=2)
    vl_ds = PushTDataset("val", seq_len=2)

    tr_ldr = DataLoader(tr_ds, 64, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl_ds, 64)

    results = []

    # --- MLP Baseline ---
    print("\n--- MLP Baseline ---")
    set_seed(42)
    m = MLPBaseline(action_dim=2)
    r = train_push(m, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, name="MLP-Baseline")
    results.append(r)
    print(f"  MLP:            MSE={r['val_mse']:.6f}  MAE={r['val_mae']:.6f}  P={r['params']:,}")

    # --- FluidVLA 4-modal ---
    print("\n--- FluidVLA 4-modal ---")
    set_seed(42)
    m = QuadModalFluidVLA_Lite(state_dim=14, world_dim=20, action_dim=2)
    r = train_push(m, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                   name="FluidVLA-4Modal", is_fluid=True)
    results.append(r)
    print(f"  FluidVLA-4:     MSE={r['val_mse']:.6f}  MAE={r['val_mae']:.6f}  P={r['params']:,}")

    # --- Ablation: no world model ---
    print("\n--- Ablation: No World ---")
    class FluidNoWorld(QuadModalFluidVLA_Lite):
        def forward(self, img, txt, state, world=None, **kw):
            B, d = img.shape[0], img.device
            img_f = self.vision_cnn(img)
            txt_f = self.text_mlp(txt) if txt is not None else torch.zeros(B, 256, device=d)
            st_f = self.state_enc(state) if state is not None else torch.zeros(B, 256, device=d)
            w_f = torch.zeros(B, 256, device=d)
            fused, _ = self.fusion(img_f, txt_f, st_f, w_f)
            field = self.field_proj(fused).view(B, self.fluid_n, self.fluid_n, self.fluid_h)
            field = field + self.fourier_norm(self.fourier(field))
            field = self.channel_mix(field)
            out = self.adr(field, mode=kw.get("mode","nonsequential"),
                          max_output_steps=kw.get("max_output_steps",1))
            return out.outputs[:, 0]

    set_seed(42)
    m = FluidNoWorld(state_dim=14, world_dim=20, action_dim=2)
    r = train_push(m, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                   name="FluidVLA-NoWorld", is_fluid=True)
    results.append(r)
    print(f"  FluidVLA-NoW:   MSE={r['val_mse']:.6f}  MAE={r['val_mae']:.6f}")

    # --- Ablation: no text ---
    print("\n--- Ablation: No Text ---")
    class FluidNoText(QuadModalFluidVLA_Lite):
        def forward(self, img, txt=None, state=None, world=None, **kw):
            B, d = img.shape[0], img.device
            img_f = self.vision_cnn(img)
            txt_f = torch.zeros(B, 256, device=d)
            st_f = self.state_enc(state) if state is not None else torch.zeros(B, 256, device=d)
            w_f = self.world_enc(world) if world is not None else torch.zeros(B, 256, device=d)
            fused, _ = self.fusion(img_f, txt_f, st_f, w_f)
            field = self.field_proj(fused).view(B, self.fluid_n, self.fluid_n, self.fluid_h)
            field = field + self.fourier_norm(self.fourier(field))
            field = self.channel_mix(field)
            out = self.adr(field, mode=kw.get("mode","nonsequential"),
                          max_output_steps=kw.get("max_output_steps",1))
            return out.outputs[:, 0]

    set_seed(42)
    m = FluidNoText(state_dim=14, world_dim=20, action_dim=2)
    r = train_push(m, tr_ldr, vl_ldr, device, steps=500, lr=1.5e-3, wd=2e-4,
                   name="FluidVLA-NoText", is_fluid=True)
    results.append(r)
    print(f"  FluidVLA-NoT:   MSE={r['val_mse']:.6f}  MAE={r['val_mae']:.6f}")

    # ==================== Summary ====================
    print("\n" + "=" * 65)
    print("  PushT RESULTS")
    print("=" * 65)
    print(f"{'Model':<18s} {'Params':>9s} {'MSE':>10s} {'MAE':>10s} {'Time':>6s}")
    print("-" * 58)
    for r in sorted(results, key=lambda x: x["val_mse"]):
        print(f"{r['model']:<18s} {r['params']:>9,} {r['val_mse']:>10.6f} "
              f"{r['val_mae']:>10.6f} {r['seconds']:>5.0f}s")

    # Modality contribution
    base = next(r for r in results if r["model"] == "FluidVLA-NoText")
    full = next(r for r in results if r["model"] == "FluidVLA-4Modal")
    noworld = next(r for r in results if r["model"] == "FluidVLA-NoWorld")
    print(f"\n  Modality impact:")
    print(f"    +World:  MSE {noworld['val_mse']:.6f} -> {base['val_mse']:.6f}  ({((noworld['val_mse']-base['val_mse'])/noworld['val_mse']*100):+.1f}%)")
    print(f"    +Text:   MSE {base['val_mse']:.6f} -> {full['val_mse']:.6f}  ({((base['val_mse']-full['val_mse'])/base['val_mse']*100):+.1f}%)")

    mlp = next(r for r in results if r["model"] == "MLP-Baseline")
    best = min(results, key=lambda x: x["val_mse"])
    print(f"\n  Best vs MLP: {best['val_mse']:.6f} vs {mlp['val_mse']:.6f} ({((mlp['val_mse']-best['val_mse'])/mlp['val_mse']*100):+.1f}%)")

    out = ROOT / "experiments_v3" / "vla" / "pusht_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
