# -*- coding: utf-8 -*-
"""
experiments_v3/vla/quad_modal_vla.py
=====================================
四模态 FluidNet VLA 架构:

  视觉 (ViT-Tiny, 预训练)  ──┐
  语言 (DistilBERT, 预训练) ──┤
  机器人状态 (MLP)          ──┼──→ Cross-Modal Fusion ──→ FluidNet ADR Field ──→ Action
  世界模型预测 (MLP)        ──┘

不修改 fluid_net.py 源码, 完全通过 wrapper 实现。
"""
from __future__ import annotations

import sys, time, json, math, random
from pathlib import Path
from itertools import cycle
from typing import Optional, Dict, Tuple

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import (
    set_seed, img2field, FourierGlobalMix, ChannelMix,
    CosineWarmupScheduler, LabelSmoothingLoss,
)

# ============================================================================
# 1. 预训练视觉编码器 (ViT)
# ============================================================================

class ViTEncoder(nn.Module):
    """Vision Transformer 编码器 (使用 timm 预训练权重)"""
    def __init__(self, model_name: str = "vit_tiny_patch16_224",
                 output_dim: int = 256, freeze: bool = True):
        super().__init__()
        import timm
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.vit.eval()

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        vit_dim = self._get_vit_dim(model_name)
        self.proj = nn.Sequential(
            nn.Linear(vit_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def _get_vit_dim(self, name: str) -> int:
        dims = {
            "vit_tiny_patch16_224": 192,
            "vit_small_patch16_224": 384,
            "vit_base_patch16_224": 768,
            "vit_small_patch16_224_dino": 384,
        }
        return dims.get(name, 192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.vit.parameters())):
            feat = self.vit(x)
        return self.proj(feat)


# ============================================================================
# 2. 预训练语言编码器 (DistilBERT)
# ============================================================================

class LanguageEncoder(nn.Module):
    """DistilBERT 文本编码器 (使用 transformers 预训练权重)"""
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 output_dim: int = 256, freeze: bool = True, max_length: int = 64):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.eval()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = self.bert.config.dim  # 768
        self.max_length = max_length
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def tokenize(self, texts: list) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )

    def forward(self, texts: list = None, input_ids=None, attention_mask=None):
        if texts is not None:
            encoded = self.tokenize(texts)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        device = next(self.bert.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.bert.parameters())):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # CLS token embedding
            cls_embed = outputs.last_hidden_state[:, 0, :]

        return self.proj(cls_embed)


# ============================================================================
# 3. 机器人状态编码器
# ============================================================================

class RobotStateEncoder(nn.Module):
    """机器人本体感知编码器: 关节角度 + 角速度 + 末端位姿 → embedding"""
    def __init__(self, state_dim: int = 14, output_dim: int = 256, hidden: int = 256):
        """
        state_dim: 典型7-DoF机械臂: 7(joint_angles) + 7(joint_vels) = 14
                   或者: 7(joints) + 7(vels) + 6(eff_pose) = 20
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 4. 世界模型编码器
# ============================================================================

class WorldModelEncoder(nn.Module):
    """世界模型预测编码器: 预测的未来状态 + 不确定性 → embedding"""
    def __init__(self, world_dim: int = 20, output_dim: int = 256, hidden: int = 256):
        """
        world_dim: 可以是预测的未来图像特征 + 物体位姿 + 接触力等
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 5. 四模态交叉注意力融合
# ============================================================================

class QuadModalFusion(nn.Module):
    """4模态 → 交叉注意力 → 融合特征"""
    def __init__(self, modal_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # 可学习的模态查询向量
        self.modal_query = nn.Parameter(torch.randn(1, 4, modal_dim) * 0.02)

        # 自注意力: 4个模态之间的交互
        self.self_attn = nn.MultiheadAttention(
            modal_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(modal_dim)

        # 交叉注意力到输出
        self.ffn = nn.Sequential(
            nn.Linear(modal_dim, modal_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(modal_dim * 2, modal_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(modal_dim)

        self.output_dim = modal_dim

    def forward(self, img_feat, txt_feat, state_feat, world_feat):
        """
        Args:
            img_feat:   [B, modal_dim]
            txt_feat:   [B, modal_dim]
            state_feat: [B, modal_dim]
            world_feat: [B, modal_dim]
        Returns:
            fused: [B, modal_dim]  融合后的特征
        """
        B = img_feat.shape[0]

        # 堆叠4个模态 [B, 4, modal_dim]
        tokens = torch.stack([img_feat, txt_feat, state_feat, world_feat], dim=1)

        # 添加可学习的模态查询
        query = self.modal_query.expand(B, -1, -1)  # [B, 4, modal_dim]

        # 自注意力
        attn_out, attn_weights = self.self_attn(query, tokens, tokens)
        tokens = self.norm1(query + attn_out)

        # FFN
        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        # 池化: 取4个模态token的均值作为融合特征
        fused = tokens.mean(dim=1)  # [B, modal_dim]

        return fused, attn_weights


# ============================================================================
# 6. FluidNet 融合 → 动作 主模型
# ============================================================================

class QuadModalFluidVLA(nn.Module):
    """
    四模态 FluidNet VLA 完整架构:

    Vision (ViT)   ──┐
    Language (BERT) ──┤
    Robot State      ──┼──→ QuadModalFusion ──→ FluidNet ADR Field ──→ Action
    World Model      ──┘
    """
    def __init__(self,
                 # 视觉
                 vit_model: str = "vit_tiny_patch16_224",
                 freeze_vit: bool = True,
                 # 语言
                 lm_model: str = "distilbert-base-uncased",
                 freeze_lm: bool = True,
                 # 状态
                 state_dim: int = 14,   # 7 joints + 7 velocities
                 # 世界模型
                 world_dim: int = 20,   # predicted future features
                 # 融合
                 modal_dim: int = 256,
                 num_heads: int = 8,
                 # FluidNet
                 fluid_n: int = 16,
                 fluid_h: int = 4,
                 static_adr_steps: int = 6,
                 # 动作
                 action_dim: int = 7,   # 7-DoF joint deltas
                 dropout: float = 0.1,
                 ):
        super().__init__()

        # --- 四个编码器 ---
        self.vision = ViTEncoder(model_name=vit_model, output_dim=modal_dim, freeze=freeze_vit)
        self.language = LanguageEncoder(model_name=lm_model, output_dim=modal_dim, freeze=freeze_lm)
        self.robot_state = RobotStateEncoder(state_dim=state_dim, output_dim=modal_dim)
        self.world_model = WorldModelEncoder(world_dim=world_dim, output_dim=modal_dim)

        # --- 四模态融合 ---
        self.fusion = QuadModalFusion(modal_dim=modal_dim, num_heads=num_heads, dropout=dropout)

        # --- 融合特征 → 2D Field 投影 ---
        self.field_proj = nn.Sequential(
            nn.Linear(modal_dim, fluid_n * fluid_n * fluid_h),
            nn.LayerNorm(fluid_n * fluid_n * fluid_h),
        )
        self.fluid_n = fluid_n
        self.fluid_h = fluid_h

        # --- Fourier + Channel 预处理 ---
        self.fourier = FourierGlobalMix(fluid_n, fluid_h)
        self.fourier_norm = nn.LayerNorm([fluid_n, fluid_n, fluid_h])
        self.channel_mix = ChannelMix(fluid_n, fluid_h, expansion=3)

        # --- FluidNet ADR ---
        self.adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=fluid_n, h=fluid_h, output_dim=action_dim,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=2, q_height=2, q_width=2,
                extractor_hidden_channels=48,
                reconstructor_hidden_channels=48,
                output_hidden_dim=256,
                adr_hidden_channels=48,
            ))

    def forward(self, images, texts=None, text_ids=None, text_mask=None,
                robot_states=None, world_states=None,
                mode="nonsequential", max_output_steps=1):
        """
        Args:
            images:       [B, 3, 224, 224]  视觉输入
            texts:        List[str] 或 None  文本指令
            text_ids:     [B, max_len]   tokenized text (备选)
            text_mask:    [B, max_len]   attention mask (备选)
            robot_states: [B, state_dim]  机器人关节状态
            world_states: [B, world_dim]  世界模型预测状态
        Returns:
            action: [B, action_dim]  连续动作输出
        """
        B = images.shape[0]
        device = images.device

        # --- 1. 视觉编码 ---
        img_feat = self.vision(images)  # [B, modal_dim]

        # --- 2. 语言编码 ---
        if texts is not None:
            txt_feat = self.language(texts=texts)
        else:
            txt_feat = self.language(input_ids=text_ids, attention_mask=text_mask)

        # --- 3. 机器人状态编码 ---
        if robot_states is not None:
            state_feat = self.robot_state(robot_states)
        else:
            state_feat = torch.zeros(B, self.fusion.output_dim, device=device)

        # --- 4. 世界模型编码 ---
        if world_states is not None:
            world_feat = self.world_model(world_states)
        else:
            world_feat = torch.zeros(B, self.fusion.output_dim, device=device)

        # --- 5. 四模态融合 ---
        fused, attn_weights = self.fusion(img_feat, txt_feat, state_feat, world_feat)
        # fused: [B, modal_dim]

        # --- 6. 投影到2D Field ---
        field = self.field_proj(fused).view(B, self.fluid_n, self.fluid_n, self.fluid_h)

        # --- 7. Fourier + Channel 预处理 ---
        field = field + self.fourier_norm(self.fourier(field))
        field = self.channel_mix(field)

        # --- 8. FluidNet ADR → Action ---
        out = self.adr(field, mode=mode, max_output_steps=max_output_steps)
        action = out.outputs[:, 0]  # [B, action_dim]

        return action, attn_weights


# ============================================================================
# 7. 简化版 (纯融合, 无文本时用向量替代)
# ============================================================================

class QuadModalFluidVLA_Lite(nn.Module):
    """轻量版: 用CNN替代ViT, 用固定文本embedding替代BERT (快速实验用)"""

    def __init__(self, state_dim=14, world_dim=20, action_dim=7,
                 fluid_n=16, fluid_h=4, static_adr_steps=4):
        super().__init__()

        # CNN 视觉编码器
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(),
        )

        # 文本编码器 (简化: 直接用可学习embedding或预计算特征)
        self.text_mlp = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(),   # 假设输入是预计算的768d特征
            nn.Linear(256, 256), nn.ReLU(),
        )

        # 状态&世界编码 (同主版)
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.world_enc = nn.Sequential(
            nn.Linear(world_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        # 融合
        self.fusion = QuadModalFusion(modal_dim=256, num_heads=4)

        # Field投影 + FluidNet
        self.field_proj = nn.Linear(256, fluid_n * fluid_n * fluid_h)
        self.fluid_n, self.fluid_h = fluid_n, fluid_h
        self.fourier = FourierGlobalMix(fluid_n, fluid_h)
        self.fourier_norm = nn.LayerNorm([fluid_n, fluid_n, fluid_h])
        self.channel_mix = ChannelMix(fluid_n, fluid_h, expansion=3)

        self.adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=fluid_n, h=fluid_h, output_dim=action_dim,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=32, q_depth=2, q_height=2, q_width=2,
                extractor_hidden_channels=48, reconstructor_hidden_channels=48,
                output_hidden_dim=256, adr_hidden_channels=48,
            ))

    def forward(self, images, text_features=None, robot_states=None, world_states=None,
                mode="nonsequential", max_output_steps=1):
        B, device = images.shape[0], images.device

        img_feat = self.vision_cnn(images)

        if text_features is not None:
            txt_feat = self.text_mlp(text_features)
        else:
            txt_feat = torch.zeros(B, 256, device=device)

        if robot_states is not None:
            state_feat = self.state_enc(robot_states)
        else:
            state_feat = torch.zeros(B, 256, device=device)

        if world_states is not None:
            world_feat = self.world_enc(world_states)
        else:
            world_feat = torch.zeros(B, 256, device=device)

        fused, _ = self.fusion(img_feat, txt_feat, state_feat, world_feat)

        field = self.field_proj(fused).view(B, self.fluid_n, self.fluid_n, self.fluid_h)
        field = field + self.fourier_norm(self.fourier(field))
        field = self.channel_mix(field)

        out = self.adr(field, mode=mode, max_output_steps=max_output_steps)
        return out.outputs[:, 0]


print("""
 quad_modal_vla.py loaded!
 =========================
 QuadModalFluidVLA:       ViT + DistilBERT + State + World → FluidNet → Action
 QuadModalFluidVLA_Lite:  CNN + TextMLP + State + World → FluidNet → Action (faster)
""")
