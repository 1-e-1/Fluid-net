import sys, time, json, random
from pathlib import Path
from itertools import cycle
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import (
    set_seed, img2field, make_subset, FourierGlobalMix, ChannelMix,
    CosineWarmupScheduler, LabelSmoothingLoss, acc_fn,
)

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

# ============================================================
# 新架构: 多路并行提取 → 二次融合 → 最终萃取
# ============================================================

class HierarchicalFluidNet(nn.Module):
    """
    层级化FluidNet: 
      Stage 1: N路并行提取 (不同h/不同scale/不同初始化)
      Stage 2: 跨路交叉注意力融合
      Stage 3: 融合特征 → 最终FluidNet萃取 → 输出
    """
    def __init__(self, n=28, h_per_branch=4, num_branches=3, num_classes=10,
                 static_adr_steps=4):
        super().__init__()
        self.num_branches = num_branches
        self.h_per_branch = h_per_branch
        total_h = h_per_branch * num_branches

        # 输入投影: 1ch → total_h channels
        self.proj = nn.Linear(1, total_h)
        self.pos = nn.Parameter(torch.zeros(1, n, n, total_h) * 0.02)
        self.fourier = FourierGlobalMix(n, total_h)
        self.fourier_norm = nn.LayerNorm([n, n, total_h])

        # Stage 1: N路并行FluidNet分支
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = nn.ModuleDict({
                "channel_mix": ChannelMix(n, h_per_branch, expansion=2),
                "fluid": fn.FluidResidualSequenceNetwork(
                    fn.FluidResidualSequenceConfig(
                        n=n, h=h_per_branch, output_dim=64,  # 每路输出64维特征
                        max_output_steps=1, static_adr_steps=static_adr_steps,
                        q_channels=24, q_depth=3, q_height=3, q_width=1,
                        extractor_hidden_channels=32,
                        reconstructor_hidden_channels=32,
                        output_hidden_dim=128,
                        adr_hidden_channels=32,
                    ))
            })
            self.branches.append(branch)

        # Stage 2: 跨路融合 (交叉注意力)
        self.cross_fusion = nn.MultiheadAttention(
            64, num_heads=4, dropout=0.1, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(64)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64)
        )
        self.fusion_ffn_norm = nn.LayerNorm(64)

        # Stage 3: 最终萃取FluidNet
        # 将融合特征投影到2D field
        self.final_proj = nn.Linear(64, n * n * 4)
        self.final_n = n
        self.final_h = 4
        self.final_fourier = FourierGlobalMix(n, 4)
        self.final_fourier_norm = nn.LayerNorm([n, n, 4])
        self.final_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=4, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=4, q_height=4, q_width=1,
                extractor_hidden_channels=48,
                reconstructor_hidden_channels=48,
                output_hidden_dim=256,
                adr_hidden_channels=48,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        B = x.shape[0]

        # 输入预处理
        h = self.proj(x) + self.pos
        h = h + self.fourier_norm(self.fourier(h))

        # Stage 1: 将total_h通道切分给N路
        branch_outputs = []
        branch_fields = []
        for i, branch in enumerate(self.branches):
            start_c = i * self.h_per_branch
            end_c = start_c + self.h_per_branch
            branch_input = h[:, :, :, start_c:end_c]

            # 通道混合预处理
            branch_input = branch["channel_mix"](branch_input)

            # FluidNet处理
            out = branch["fluid"](branch_input, mode=mode, max_output_steps=max_output_steps)
            branch_outputs.append(out.outputs[:, 0])  # [B, 64]
            branch_fields.append(out.final_field.squeeze(1))  # [B, n, n, h_per_branch]

        # Stage 2: 跨路融合
        # [B, num_branches, 64]
        branch_tokens = torch.stack(branch_outputs, dim=1)

        # 自注意力: 各路之间互相查询
        attn_out, attn_weights = self.cross_fusion(
            branch_tokens, branch_tokens, branch_tokens
        )
        branch_tokens = self.fusion_norm(branch_tokens + attn_out)
        branch_tokens = self.fusion_ffn_norm(
            branch_tokens + self.fusion_ffn(branch_tokens)
        )

        # 池化: 取各路均值作为融合特征
        fused = branch_tokens.mean(dim=1)  # [B, 64]

        # Stage 3: 最终萃取
        field = self.final_proj(fused).view(B, self.final_n, self.final_n, self.final_h)
        field = field + self.final_fourier_norm(self.final_fourier(field))
        return self.final_fluid(field, mode=mode, max_output_steps=max_output_steps)


class HierarchicalFluidNet_V2(nn.Module):
    """V2: 多尺度分支 + final_field融合"""
    def __init__(self, n=28, num_classes=10, static_adr_steps=4):
        super().__init__()

        # 3个尺度的分支
        self.scales = [
            {"n": 28, "h": 4, "q": (4,4), "name": "fine"},    # 细粒度
            {"n": 14, "h": 4, "q": (3,3), "name": "mid"},     # 中等
            {"n": 7,  "h": 4, "q": (2,2), "name": "coarse"},  # 粗粒度
        ]

        self.proj = nn.Linear(1, 4)  # 统一投影到h=4
        self.pos_28 = nn.Parameter(torch.zeros(1, 28, 28, 4) * 0.02)

        # 各尺度分支
        self.scale_branches = nn.ModuleDict()
        for sc in self.scales:
            sn, sh = sc["n"], sc["h"]
            sq_d, sq_h = sc["q"]
            branch = nn.ModuleDict({
                "fourier": FourierGlobalMix(sn, sh),
                "fourier_norm": nn.LayerNorm([sn, sn, sh]),
                "channel_mix": ChannelMix(sn, sh, expansion=2),
                "fluid": fn.FluidResidualSequenceNetwork(
                    fn.FluidResidualSequenceConfig(
                        n=sn, h=sh, output_dim=64,
                        max_output_steps=1, static_adr_steps=static_adr_steps,
                        q_channels=24, q_depth=sq_d, q_height=sq_h, q_width=1,
                        extractor_hidden_channels=32,
                        reconstructor_hidden_channels=32,
                        output_hidden_dim=128,
                        adr_hidden_channels=32,
                    ))
            })
            self.scale_branches[sc["name"]] = branch

        # 跨尺度融合
        self.cross_fusion = nn.MultiheadAttention(64, num_heads=4, dropout=0.1, batch_first=True)
        self.fusion_norm = nn.LayerNorm(64)

        # 最终萃取
        self.final_proj = nn.Linear(64, 14 * 14 * 4)
        self.final_fourier = FourierGlobalMix(14, 4)
        self.final_fourier_norm = nn.LayerNorm([14, 14, 4])
        self.final_fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=14, h=4, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=3, q_height=3, q_width=1,
                extractor_hidden_channels=48, reconstructor_hidden_channels=48,
                output_hidden_dim=256, adr_hidden_channels=48,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        B = x.shape[0]
        f_28 = self.proj(x) + self.pos_28  # [B, 28, 28, 4]

        branch_outputs = []
        for sc in self.scales:
            branch = self.scale_branches[sc["name"]]
            sn = sc["n"]

            if sn < 28:
                f_scaled = F.interpolate(
                    f_28.permute(0, 3, 1, 2), size=(sn, sn), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                f_scaled = f_28

            f_scaled = f_scaled + branch["fourier_norm"](branch["fourier"](f_scaled))
            f_scaled = branch["channel_mix"](f_scaled)

            out = branch["fluid"](f_scaled, mode=mode, max_output_steps=max_output_steps)
            branch_outputs.append(out.outputs[:, 0])

        # 跨尺度融合
        tokens = torch.stack(branch_outputs, dim=1)  # [B, 3, 64]
        attn_out, _ = self.cross_fusion(tokens, tokens, tokens)
        tokens = self.fusion_norm(tokens + attn_out)
        fused = tokens.mean(dim=1)  # [B, 64]

        # 最终萃取
        field = self.final_proj(fused).view(B, 14, 14, 4)
        field = field + self.final_fourier_norm(self.final_fourier(field))
        return self.final_fluid(field, mode=mode, max_output_steps=max_output_steps)


# ============================================================
# Training & Eval
# ============================================================

def train_eval_h(model, tr_ldr, vl_ldr, device, steps=500, lr=1e-3, wd=1e-4,
                 name="", num_classes=10):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    crit = LabelSmoothingLoss(num_classes, 0.05)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        fld = img2field(x)
        out = model(fld, mode="nonsequential", max_output_steps=1)
        logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
        if logits.dim() > 2: logits = logits.squeeze(1)
        crit(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    sec = time.perf_counter() - t0
    model.eval()
    vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            fld = img2field(x)
            out = model(fld, mode="nonsequential", max_output_steps=1)
            logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
            if logits.dim() > 2: logits = logits.squeeze(1)
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu())
            vt += y.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc/vt,
            "val_loss": vs/vt, "seconds": sec}


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    set_seed(42)
    print("=" * 65)
    print("  HIERARCHICAL FLUIDNET: 多路并行提取 → 融合 → 萃取")
    print("=" * 65)

    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    all_results = []

    for ds_name, ds_cls in [("MNIST", datasets.MNIST), ("FashionMNIST", datasets.FashionMNIST)]:
        tr_size = 3000 if ds_name == "MNIST" else 5000
        tr = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=t), tr_size)
        vl = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=False, download=False, transform=t), 1000)
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)

        print(f"\n{'='*55} {ds_name}")

        # Baseline
        set_seed(42)
        b = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(
            n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
            q_channels=32, q_depth=7, q_height=7, q_width=1,
            extractor_hidden_channels=40, reconstructor_hidden_channels=40, output_hidden_dim=192))
        r = train_eval_h(b, tr_ldr, vl_ldr, device, steps=600, lr=1e-3, wd=1e-4,
                         name=f"{ds_name[:4]}-Baseline", num_classes=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  Baseline(h=1):         acc={r['val_acc']:.4f} P={r['params']:,}")

        # Hierarchical V1: 3 parallel branches
        set_seed(42)
        m1 = HierarchicalFluidNet(n=28, h_per_branch=4, num_branches=3, num_classes=10, static_adr_steps=4)
        r = train_eval_h(m1, tr_ldr, vl_ldr, device, steps=600, lr=1.5e-3, wd=2e-4,
                         name=f"{ds_name[:4]}-HierV1(3x4)", num_classes=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  HierV1(3branchx4h):    acc={r['val_acc']:.4f} P={r['params']:,}")

        # Hierarchical V2: multi-scale branches
        set_seed(42)
        m2 = HierarchicalFluidNet_V2(n=28, num_classes=10, static_adr_steps=4)
        r = train_eval_h(m2, tr_ldr, vl_ldr, device, steps=600, lr=1.5e-3, wd=2e-4,
                         name=f"{ds_name[:4]}-HierV2(MultiScale)", num_classes=10)
        r["dataset"] = ds_name; all_results.append(r)
        print(f"  HierV2(MultiScale):    acc={r['val_acc']:.4f} P={r['params']:,}")

        # Hierarchical V1: 5 parallel branches
        if ds_name == "MNIST":
            set_seed(42)
            m3 = HierarchicalFluidNet(n=28, h_per_branch=3, num_branches=5, num_classes=10, static_adr_steps=4)
            r = train_eval_h(m3, tr_ldr, vl_ldr, device, steps=600, lr=1.5e-3, wd=2e-4,
                             name=f"{ds_name[:4]}-HierV1(5x3)", num_classes=10)
            r["dataset"] = ds_name; all_results.append(r)
            print(f"  HierV1(5branchx3h):    acc={r['val_acc']:.4f} P={r['params']:,}")

    # Summary
    print("\n" + "=" * 65)
    print("  HIERARCHICAL FLUIDNET SUMMARY")
    print("=" * 65)
    print(f"{'Model':<25s} {'MNIST':>8s} {'FMNIST':>8s} {'Params':>10s}")
    print("-" * 55)
    mnist = {r["model"]: r for r in all_results if r["dataset"] == "MNIST"}
    fmnist = {r["model"]: r for r in all_results if r["dataset"] == "FashionMNIST"}
    for name in sorted(mnist.keys()):
        m = mnist[name]; f = fmnist.get(name, {})
        print(f"{name:<25s} {m['val_acc']:.4f}     {f.get('val_acc',0):.4f}     {m['params']:>10,}")

    out = ROOT / "experiments_v3" / "results" / "hierarchical_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")
