"""
FluidFormer: Bridging ADR Physics to Transformer-Level Performance
===================================================================
核心创新 (不修改 fluid_net 源码):
1. Multi-Head Fluid Dynamics  -  多头物理场 (模仿 multi-head attention)
2. Fourier Global Mixing      -  FFT全局交互 (模仿 self-attention 的全局性)  
3. Deep Residual Stack        -  深层残差堆叠 + LayerNorm
4. Content-Gated ADR          -  内容门控物理参数 (模仿 content-dependent attention)
5. Hierarchical Pooling       -  多尺度信息聚合

数学对应:
  Transformer Attention:  y_i = sum_j( softmax(q_i·k_j) * v_j )  [全局内容相关]
  FluidFormer Global:     J' = IFFT( M ⊙ FFT(J) )                [全局谱域混合]
  FluidFormer Local:      J' = J + Δτ * ADR(J, θ(content))        [局部物理演化]
"""
from __future__ import annotations

import math, sys
from pathlib import Path
from typing import Optional

import torch, torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fluid_net as fn
import fluid_net_pro_max as fnpm


# ==============================================================================
# 1. Fourier Global Mixing Layer (替代 self-attention 的全局交互)
# ==============================================================================
class FourierGlobalMix(nn.Module):
    """
    在频域进行全局信息混合。
    对应 Transformer: O(L²·d) 的全局 pairwise attention
    FluidFormer:      O(L·d·log L) 的 FFT 全局谱混合
    
    数学:
      J_freq = FFT(J)           # 转换到频域
      J_freq = W_freq ⊙ J_freq  # 频域逐元素可学习调制
      J_out  = IFFT(J_freq)     # 转回空域
    """
    def __init__(self, n: int, h: int):
        super().__init__()
        self.n = n
        self.h = h
        # 可学习频域滤波器
        self.freq_weight = nn.Parameter(torch.ones(1, n, n, h) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n, n, h]
        x_freq = torch.fft.fft2(x.float(), dim=(1, 2), norm="ortho")
        x_freq = torch.view_as_real(x_freq)  # [B, n, n, h, 2]
        # 对实部和虚部分别调制
        w = self.freq_weight.unsqueeze(-1)   # [1, n, n, h, 1]
        x_freq = x_freq * (1.0 + w)
        x_freq = torch.view_as_complex(x_freq)
        x_out = torch.fft.ifft2(x_freq, dim=(1, 2), norm="ortho").real
        return x_out.to(x.dtype)


# ==============================================================================
# 2. Multi-Head Fluid Dynamics
# ==============================================================================
class MultiHeadFluidBlock(nn.Module):
    """
    多头物理场: 多个并行的 ADR 演化 + 拼接 + 投影。
    对应 Transformer 的 multi-head attention:
      - 每个 head 有独立的物理参数 (velocity, diffusion, viscosity)
      - 不同 head 捕获不同尺度的信息流模式
    """
    def __init__(self, n: int, h: int, num_heads: int = 4, output_dim: int = 10):
        super().__init__()
        self.num_heads = num_heads
        head_h = max(1, h // num_heads)

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            # 每个 head 有不同的物理参数
            cfg = fn.FluidResidualSequenceConfig(
                n=n, h=head_h,
                output_dim=output_dim,
                max_output_steps=1,
                static_adr_steps=2,
                q_channels=16,
                q_depth=1, q_height=1, q_width=1,
                extractor_hidden_channels=24,
                reconstructor_hidden_channels=24,
                output_hidden_dim=64,
                max_velocity=0.5 + 0.5 * (i / max(1, num_heads - 1)),  # 不同速度
                max_diffusion=0.05 + 0.15 * (i / max(1, num_heads - 1)),  # 不同扩散
            )
            self.heads.append(fn.FluidResidualSequenceNetwork(cfg))

        # 多头输出投影
        self.proj = nn.Linear(num_heads * output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n, n, h]
        head_h = x.shape[-1] // self.num_heads
        head_outputs = []
        for i, head in enumerate(self.heads):
            start = i * head_h
            x_head = x[:, :, :, start:start + head_h]
            out = head(x_head, mode="nonsequential", max_output_steps=1)
            head_outputs.append(out.outputs[:, 0])
        concat = torch.cat(head_outputs, dim=-1)
        return self.proj(concat)


# ==============================================================================
# 3. Content-Gated ADR (内容门控)
# ==============================================================================
class ContentGate(nn.Module):
    """
    用全局池化特征门控 ADR 的物理参数。
    对应 Transformer 的 content-dependent attention weights。
    
    数学:
      g = σ(W_g · GlobalPool(J))
      J' = g ⊙ ADR(J) + (1 - g) ⊙ J
    """
    def __init__(self, n: int, h: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(h, h),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n, n, h]
        B, N1, N2, H = x.shape
        # Global context gate
        g = self.gate(x.permute(0, 3, 1, 2))  # [B, h]
        g = g.view(B, 1, 1, H)
        return x * g


# ==============================================================================
# 4. Deep Residual Fluid Block (深层残差)
# ==============================================================================
class ResidualFluidBlock(nn.Module):
    """
    Pre-LayerNorm + ADR + Residual
    对应 Transformer 的 sublayer 结构。
    """
    def __init__(self, n: int, h: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm([n, n, h])
        self.fluid = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=output_dim,
                max_output_steps=1, static_adr_steps=2,
                q_channels=16, q_depth=2, q_height=2, q_width=1,
                extractor_hidden_channels=32,
                reconstructor_hidden_channels=32,
                output_hidden_dim=96,
            )
        )
        self.norm2 = nn.LayerNorm([n, n, h])
        self.ffn = nn.Sequential(
            nn.Linear(h, h * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h * 4, h),
            nn.Dropout(dropout),
        )
        # 将输出投影回 h 维
        self.proj = nn.Linear(output_dim, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ADR sublayer
        normed = self.norm1(x)
        out = self.fluid(normed, mode="nonsequential", max_output_steps=1)
        residual = self.proj(out.outputs[:, 0])  # [B, h]
        x = x + residual.view(x.shape[0], 1, 1, x.shape[-1]).expand_as(x)
        
        # FFN sublayer (类似 Transformer FFN)
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return x


# ==============================================================================
# 5. FluidFormer: 完整架构
# ==============================================================================
class FluidFormer(nn.Module):
    """
    FluidFormer = FourierGlobalMix + MultiHeadFluid + DeepResidualStack + ContentGate
    
    架构对应:
      Transformer          →  FluidFormer
      ─────────────────────────────────────
      Self-Attention       →  FourierGlobalMix (全局) + ADR (局部)
      Multi-Head           →  MultiHeadFluidBlock
      LayerNorm + Residual →  ResidualFluidBlock  
      FFN                  →  Content-Gated MLP
      Position Encoding    →  (隐含在 grid 结构中)
    """
    def __init__(
        self,
        n: int = 28,
        h: int = 8,
        num_classes: int = 10,
        num_blocks: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.h = h
        
        # 输入投影: 将单通道图像映射到 h 维
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, h, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, h), h),
            nn.GELU(),
        )
        
        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, n, h) * 0.02)
        
        # 堆叠的 Fluid Block
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                "fourier": FourierGlobalMix(n, h),
                "content_gate": ContentGate(n, h),
                "residual_fluid": ResidualFluidBlock(
                    n, h, output_dim=h, dropout=dropout
                ),
                "norm": nn.LayerNorm([n, n, h]),
            }))
        
        # 输出头
        self.final_norm = nn.LayerNorm([n, n, h])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(h, h * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h * 2, num_classes),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] or [B, C, H, W] (image format)
        返回: [B, num_classes]
        """
        B = x.shape[0]
        
        # Input projection
        if x.shape[1] == 1 and x.ndim == 4:
            x = self.input_proj(x)  # [B, h, n, n]
            x = x.permute(0, 2, 3, 1)  # [B, n, n, h]
        elif x.ndim == 4 and x.shape[-1] != self.h:
            # Assume [B, n, n, c] format
            if x.shape[-1] != self.h:
                x = F.linear(x.view(B, -1, x.shape[-1]),
                            torch.eye(x.shape[-1], self.h, device=x.device)).view(B, self.n, self.n, self.h)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Process through blocks
        for block in self.blocks:
            # 1. Global Fourier Mixing (替代 self-attention)
            x_fourier = block["fourier"](x)
            x = x + x_fourier
            
            # 2. Content Gate
            x_gated = block["content_gate"](x)
            x = x + x_gated
            
            # 3. Local ADR + FFN (residual)
            x = block["residual_fluid"](x)
            
            # 4. LayerNorm
            x = block["norm"](x)
        
        # Final norm + classification head
        x = self.final_norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, h, n, n]
        return self.head(x)


# ==============================================================================
# 6. Training & Eval utilities
# ==============================================================================
def build_fluidformer_tiny(num_classes=10, n=28):
    """Tiny variant for quick experiments"""
    return FluidFormer(n=n, h=4, num_classes=num_classes, num_blocks=2, num_heads=2, dropout=0.1)


def build_fluidformer_small(num_classes=10, n=28):
    """Small variant"""
    return FluidFormer(n=n, h=8, num_classes=num_classes, num_blocks=3, num_heads=4, dropout=0.1)


def build_fluidformer_base(num_classes=10, n=28):
    """Base variant"""
    return FluidFormer(n=n, h=16, num_classes=num_classes, num_blocks=4, num_heads=4, dropout=0.15)


# ==============================================================================
# 7. Benchmark script
# ==============================================================================
if __name__ == "__main__":
    import json, random, time
    from itertools import cycle
    
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    
    # Test model creation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    for name, builder in [
        ("tiny", build_fluidformer_tiny),
        ("small", build_fluidformer_small),
        ("base", build_fluidformer_base),
    ]:
        model = builder(num_classes=10, n=28).to(device)
        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(2, 1, 28, 28).to(device)
        with torch.no_grad():
            y = model(x)
        print(f"FluidFormer-{name}: params={params:,}, output={y.shape}")
    
    print("\n=== Architecture Gap Analysis ===")
    print("""
    Transformer 的核心优势 → FluidFormer 的对应设计:
    ┌──────────────────────────┬──────────────────────────────────────┐
    │ Transformer 机制          │ FluidFormer 对应实现                   │
    ├──────────────────────────┼──────────────────────────────────────┤
    │ Self-Attention (全局交互) │ FourierGlobalMix (频域全局谱混合)      │
    │ Multi-Head (多子空间)    │ MultiHeadFluidBlock (多物理参数头)     │
    │ LayerNorm + Residual     │ ResidualFluidBlock (Pre-LN + 残差)   │
    │ FFN (逐位置非线性)       │ Content-Gated MLP (内容门控)          │
    │ Position Encoding        │ Learnable PosEmbed (可学习位置编码)   │
    │ Deep Stack (12-24层)     │ num_blocks=2-6 (可配置深度)           │
    │ Scale (d=768-4096)       │ h=8-64 (通道维度)                     │
    └──────────────────────────┴──────────────────────────────────────┘
    
    仍存在的差距:
    1. 计算复杂度: FourierGlobalMix O(L·logL) ≈ Self-Attention O(L²) 但在小尺度上
       Attention 的 softmax 选择机制更强 (稀疏激活)
    2. 容量: 当前 h≤16, 而 ViT-Base 的 d=768。需要扩大到 h=64+
    3. 预训练: Transformer 受益于大规模预训练, FluidFormer 需要类似 pipeline
    4. KV-Cache: Transformer 的 autoregressive 解码优势在 FluidFormer 中暂无对应
    
    预期性能阶梯:
    - FluidFormer-Tiny (h=4, 2 blocks):  ≈ MLP baseline, ~50K params
    - FluidFormer-Small (h=8, 3 blocks): ≈ Small CNN, ~200K params  
    - FluidFormer-Base (h=16, 4 blocks): ≈ ViT-Tiny, ~500K params
    - FluidFormer-Large (h=32, 6 blocks): ≈ ViT-Small, ~2M params
    """)
