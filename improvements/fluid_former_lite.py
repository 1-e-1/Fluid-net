"""
FluidFormer-Lite: 最小可行改进版
=================================
只做三件事:
1. FourierGlobalMix - FFT全局谱混合 (取代self-attention的全局性)
2. 增加通道维度 h (类似Transformer的d_model)
3. Pre-LayerNorm + Residual

不引入多头、内容门控等复杂机制，保持简洁可验证。
"""
import sys
from pathlib import Path

import torch, torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fluid_net as fn


class FourierGlobalMix(nn.Module):
    """FFT-based global mixing: O(L log L) vs Attention O(L^2)"""
    def __init__(self, n: int, h: int):
        super().__init__()
        self.n = n
        # 低频滤波器: 只调制低频成分(长距离), 保留高频(局部细节)
        cutoff = n // 4
        mask = torch.zeros(n, n)
        mask[:cutoff, :cutoff] = 1.0
        mask[:cutoff, -cutoff:] = 1.0
        mask[-cutoff:, :cutoff] = 1.0
        mask[-cutoff:, -cutoff:] = 1.0
        self.freq_bias = nn.Parameter(mask.view(1, n, n, 1, 1) * 0.05)

    def forward(self, x: torch.Tensor, mode: str = "nonsequential", max_output_steps: int = 1) -> torch.Tensor:
        # x: [B, n, n, h]
        B = x.shape[0]
        x_freq = torch.fft.fft2(x.float(), dim=(1, 2), norm="ortho")
        x_freq = torch.view_as_real(x_freq)
        # 仅调制低频
        x_freq = x_freq * (1.0 + self.freq_bias.to(x.device))
        x_freq = torch.view_as_complex(x_freq)
        return torch.fft.ifft2(x_freq, dim=(1, 2), norm="ortho").real.to(x.dtype)


class FluidFormerLite(nn.Module):
    """Lite version: 单层 Fourier + ADR, 增加通道维度"""
    def __init__(self, n=28, h=4, num_classes=10):
        super().__init__()
        self.n = n
        self.h = h
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(1, h),
            nn.LayerNorm(h),
        )
        
        # 全局混合
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        
        # ADR with wider channels
        self.adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=3,
                q_channels=32, q_depth=4, q_height=4, q_width=1,
                extractor_hidden_channels=48,
                reconstructor_hidden_channels=48,
                output_hidden_dim=192,
            )
        )
        self.adr_norm = nn.LayerNorm([n, n, h])
        
    def forward(self, x: torch.Tensor, mode: str = "nonsequential", max_output_steps: int = 1) -> torch.Tensor:
        # x: [B, n, n, 1]  field format
        B = x.shape[0]
        
        # 投影到更宽的通道
        h = self.input_proj(x)  # [B, n, n, h_dim]
        
        # 全局Fourier混合 + residual
        h = h + self.fourier_norm(self.fourier(h))
        
        # ADR处理
        out = self.adr(h, mode="nonsequential", max_output_steps=1)
        return out


def train_and_compare():
    """Compare FluidFormerLite vs Baseline FluidNet"""
    import json, time, random
    from itertools import cycle
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    tr_ds = datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[0] / 'data'), train=True, download=False, transform=t)
    vl_ds = datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[0] / 'data'), train=False, download=False, transform=t)
    tr = Subset(tr_ds, np.random.default_rng(42).choice(60000, 3000, replace=False).tolist())
    vl = Subset(vl_ds, np.random.default_rng(42).choice(10000, 1000, replace=False).tolist())
    tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl, 64)
    
    def img2f(x): return x.permute(0, 2, 3, 1).contiguous()
    
    def run(model, name, steps):
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tr_iter = cycle(tr_ldr)
        t0 = time.perf_counter()
        for _ in range(steps):
            model.train()
            x, y = next(tr_iter); x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(img2f(x), mode="nonsequential", max_output_steps=1)
            logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
            F.cross_entropy(logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sec = time.perf_counter() - t0
        
        model.eval()
        vs, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for x, y in vl_ldr:
                x, y = x.to(device), y.to(device)
                out = model(img2f(x), mode="nonsequential", max_output_steps=1)
                logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
                vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
                vc += int((logits.argmax(dim=-1) == y).sum().cpu())
                vt += y.numel()
        params = sum(p.numel() for p in model.parameters())
        return {"model": name, "params": params, "steps": steps,
                "val_acc": vc / vt, "val_loss": vs / vt, "seconds": sec}

    # Baseline: Original FluidNet with h=1
    cfg_base = fn.FluidResidualSequenceConfig(
        n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
        q_channels=32, q_depth=7, q_height=7, q_width=1,
        extractor_hidden_channels=40, reconstructor_hidden_channels=40,
        output_hidden_dim=192)
    m_base = fn.FluidResidualSequenceNetwork(cfg_base)
    r1 = run(m_base, "FluidNet-Base(h=1)", 300)
    
    # FluidNet with h=4 (wider channels, no Fourier)
    cfg_wide = fn.FluidResidualSequenceConfig(
        n=28, h=4, output_dim=10, max_output_steps=1, static_adr_steps=4,
        q_channels=32, q_depth=7, q_height=7, q_width=4,
        extractor_hidden_channels=40, reconstructor_hidden_channels=40,
        output_hidden_dim=192)
    m_wide = fn.FluidResidualSequenceNetwork(cfg_wide)
    r2 = run(m_wide, "FluidNet-Wide(h=4)", 300)
    
    # FluidFormerLite: Wider + Fourier Global Mix
    m_ffl = FluidFormerLite(n=28, h=4, num_classes=10)
    r3 = run(m_ffl, "FluidFormerLite(h=4+Fourier)", 300)
    
    all_r = [r1, r2, r3]
    print(json.dumps(all_r, indent=2, ensure_ascii=False))
    return all_r

if __name__ == "__main__":
    train_and_compare()
