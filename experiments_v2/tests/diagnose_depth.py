# -*- coding: utf-8 -*-
"""诊断: 为什么FluidNet不能突破2层"""
import sys, torch, time, random
from itertools import cycle
sys.path.insert(0, ".")
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import fluid_net as fn

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
device = torch.device("cuda:0")

def img2field(x): return x.permute(0, 2, 3, 1).contiguous()

class TwoLayerFluidNet(nn.Module):
    """两层的FluidNet - 带诊断钩子"""
    def __init__(self, n=28, h=4, num_classes=10):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.layer1 = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=4,
                q_channels=16, q_depth=4, q_height=4, q_width=1,
                extractor_hidden_channels=32, reconstructor_hidden_channels=32,
                output_hidden_dim=128, adr_hidden_channels=32))
        self.layer2 = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=4,
                q_channels=16, q_depth=4, q_height=4, q_width=1,
                extractor_hidden_channels=32, reconstructor_hidden_channels=32,
                output_hidden_dim=128, adr_hidden_channels=32))
        self.n, self.h = n, h

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        
        # Layer 1
        out1 = self.layer1(h, mode=mode, max_output_steps=max_output_steps)
        logits1 = out1.outputs[:, 0]
        
        # Get the processed field from layer1 Q reconstruction
        q1 = out1.q_tensors[:, 0]  # [B, C, D, H, W]
        reconstructed1 = out1.reconstructed_components[:, 0]  # [B, n, n, h]
        final_field1 = out1.final_field.squeeze(1)  # [B, n, n, h]
        
        # Layer 2: process the residual field
        out2 = self.layer2(final_field1, mode=mode, max_output_steps=max_output_steps)
        logits2 = out2.outputs[:, 0]
        
        final_field2 = out2.final_field.squeeze(1)
        q2 = out2.q_tensors[:, 0]
        reconstructed2 = out2.reconstructed_components[:, 0]
        
        # Return both logits and diagnostics
        return {
            "logits1": logits1,
            "logits2": logits2,
            "field1": final_field1,
            "field2": final_field2,
            "q1": q1, "q2": q2,
            "recon1": reconstructed1,
            "recon2": reconstructed2,
        }

# Data
t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
tr = Subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=True, download=False, transform=t),
            np.random.default_rng(42).choice(60000, 2000, replace=False).tolist())
vl = Subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data'), train=False, download=False, transform=t),
            np.random.default_rng(42).choice(10000, 500, replace=False).tolist())
tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
vl_ldr = DataLoader(vl, 64)

model = TwoLayerFluidNet(28, 4, 10).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
tr_iter = cycle(tr_ldr)

print("=" * 70)
print("  深层FluidNet诊断实验")
print("=" * 70)
print(f"{'Step':>5s} | {'Loss1':>7s} {'Acc1':>6s} | {'Loss2':>7s} {'Acc2':>6s} | "
      f"{'|F1|':>8s} {'|F2|':>8s} | {'|G_L1|':>8s} {'|G_L2|':>8s} | {'Qdiff':>6s}")
print("-" * 95)

for step in range(300):
    model.train()
    x, y = next(tr_iter); x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)
    
    fld = img2field(x)
    out = model(fld, mode="nonsequential", max_output_steps=1)
    
    # Loss from layer1 only (encourage layer1 to learn)
    loss1 = F.cross_entropy(out["logits1"], y)
    # Loss from layer2 only (encourage layer2 to learn from layer1's residual)
    loss2 = F.cross_entropy(out["logits2"], y)
    # Combined
    loss = loss1 + loss2
    loss.backward()
    
    # Measure gradient norms BEFORE clipping
    g_norm_l1 = 0.0
    g_norm_l2 = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            if "layer1" in name:
                g_norm_l1 += gn
            elif "layer2" in name:
                g_norm_l2 += gn
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    # Diagnostics every 20 steps
    if step % 20 == 0:
        acc1 = float((out["logits1"].argmax(dim=-1) == y).float().mean().cpu())
        acc2 = float((out["logits2"].argmax(dim=-1) == y).float().mean().cpu())
        f1_norm = out["field1"].flatten(1).norm(dim=1).mean().item()
        f2_norm = out["field2"].flatten(1).norm(dim=1).mean().item()
        q_diff = (out["q1"].flatten(1) - out["q2"].flatten(1)).norm(dim=1).mean().item()
        
        print(f"{step:>5d} | {loss1.item():>7.4f} {acc1:>6.4f} | "
              f"{loss2.item():>7.4f} {acc2:>6.4f} | "
              f"{f1_norm:>8.4f} {f2_norm:>8.4f} | "
              f"{g_norm_l1:>8.4f} {g_norm_l2:>8.4f} | {q_diff:>6.4f}")

# Final validation
model.eval()
with torch.no_grad():
    total, correct1, correct2 = 0, 0, 0
    for x, y in vl_ldr:
        x, y = x.to(device), y.to(device)
        out = model(img2field(x), mode="nonsequential", max_output_steps=1)
        correct1 += int((out["logits1"].argmax(dim=-1) == y).sum().cpu())
        correct2 += int((out["logits2"].argmax(dim=-1) == y).sum().cpu())
        total += y.numel()
    print(f"\nFinal Val: Layer1={correct1/total:.4f} Layer2={correct2/total:.4f}")

# Analyze field statistics
print("\n" + "=" * 70)
print("  深层ADR失败根因分析")
print("=" * 70)

# Measure field value range
model.eval()
with torch.no_grad():
    x_batch, _ = next(iter(vl_ldr)); x_batch = x_batch[:32].to(device)
    out = model(img2field(x_batch), mode="nonsequential", max_output_steps=1)
    
    f1 = out["field1"]  # [B, n, n, h]
    f2 = out["field2"]
    
    print(f"\n  Layer1 field: mean={f1.mean():.6f} std={f1.std():.6f} min={f1.min():.6f} max={f1.max():.6f}")
    print(f"  Layer2 field: mean={f2.mean():.6f} std={f2.std():.6f} min={f2.min():.6f} max={f2.max():.6f}")
    
    # Check if fields are collapsing
    collapse = (f2.std() / max(f1.std(), 1e-8)).item()
    print(f"\n  Field std ratio (L2/L1): {collapse:.4f}  (< 0.1 = collapsing)")
    
    # Check reconstruction quality
    recon_err1 = (f1 - out["recon1"]).norm() / f1.norm()
    recon_err2 = (f2 - out["recon2"]).norm() / max(f2.norm(), 1e-8)
    print(f"  Reconstruction relative error: L1={recon_err1:.4f} L2={recon_err2:.4f}")
    
    # Check Q tensor sparsity
    q1_sparsity = (out["q1"].abs() < 1e-4).float().mean().item()
    q2_sparsity = (out["q2"].abs() < 1e-4).float().mean().item()
    print(f"  Q tensor sparsity: L1={q1_sparsity:.4f} L2={q2_sparsity:.4f} (>0.5 = Q is dying)")
    
    if collapse < 0.1:
        print("\n  >>> 诊断结论: 场数值退化 (Field Collapse)")
        print("  >>> 原因: Layer1的ADR处理后, residual field 信号被大幅衰减")
        print("  >>> Layer2 收到的几乎是纯噪声, 无法学习")
    elif q2_sparsity > 0.5:
        print("\n  >>> 诊断结论: Q压缩崩溃 (Q Collapse)")
        print("  >>> 原因: Layer2的Q张量过于稀疏, 信息瓶颈失效")
    elif recon_err2 > 1.0:
        print("\n  >>> 诊断结论: 重建失败")
        print("  >>> 原因: 残差场中的信号太弱, reconstructor无法恢复")
    else:
        print("\n  >>> 其他原因: 可能是梯度流问题或训练动态问题")
