"""
Fluid Residual Sequence Network (Full Step-Conditioned ADR + Container Velocity + Content-Conditioned Viscosity + Dropout)
=========================================================================================================================

完整增强版本，包含：
1. 步数条件 ADR 全量调制（A/D/R 随 step_idx 自适应）
2. 容器速度场（速度由步数和位置决定，与场内容解耦）
3. 内容耦合粘度（Viscosity 由场内容 J 动态编码，而非全局常数）
4. Dropout（防止过拟合，增强泛化）

数学形式（最终版）：
    J_{k+1} = J_k + Δτ [
        M_A(t) * A(J_k, V(t, pos)) +
        M_D(t) * D(J_k) +
        M_R(t) * R(J_k) +
        ν(J_k) * ∇²(J_k)
    ]

其中 ν(J_k) 是由场内容动态编码的粘度场，实现“信息流动阻力自适应”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

AdvectionScheme = Literal["central", "upwind"]


@dataclass
class FluidResidualSequenceConfig:
    """网络结构配置。"""

    # 核心容器 J ∈ R^{n × n × h}
    n: int
    h: int

    # 最终每一步输出向量维度
    output_dim: int

    # ADR 参数（基础值，会被步数调制）
    adr_hidden_channels: int = 32
    adr_step_size: float = 0.05
    advection_scheme: AdvectionScheme = "upwind"
    max_velocity: float = 1.0
    max_diffusion: float = 0.1
    reaction_scale: float = 1.0

    # ========== 步数条件 ADR 全量调制 ==========
    enable_step_conditioned_adr: bool = False
    step_encoder_hidden: Optional[int] = None
    step_modulation_range: float = 0.5
    # ==========================================

    # ========== 容器速度场 ==========
    enable_container_velocity: bool = False
    container_velocity_pos_dim: int = 16
    container_velocity_hidden: int = 64
    # =================================

    # ========== NEW: 内容耦合粘度 ==========
    # 最大粘度系数（实际粘度由网络从场内容学习，0 ~ max_viscosity）
    max_viscosity: float = 0.1
    # 是否启用步数依赖的粘度微调（不改变内容决定性，仅在边界微调）
    enable_step_conditioned_viscosity: bool = False
    # ======================================

    # ========== Dropout ==========
    extractor_dropout_rate: float = 0.0
    reconstructor_dropout_rate: float = 0.0
    # ==============================

    # 非时序输入默认执行 n*n 次 ADR
    static_adr_steps: Optional[int] = None

    # 压缩后的 Q 张量形状
    q_channels: int = 16
    q_depth: int = 1
    q_height: int = 1
    q_width: int = 1
    extractor_hidden_channels: int = 32

    # 重构器
    reconstructor_hidden_channels: int = 32

    # 输出 FFN
    output_hidden_dim: int = 128

    # 输出序列控制
    max_output_steps: int = 16
    stop_threshold: float = 1e-4

    # 是否在每次残差消去后执行一次 ADR
    adr_after_subtraction: bool = False

    def __post_init__(self) -> None:
        if self.n <= 0 or self.h <= 0:
            raise ValueError("n 和 h 必须为正整数。")
        if self.output_dim <= 0:
            raise ValueError("output_dim 必须为正整数。")
        if self.max_output_steps <= 0:
            raise ValueError("max_output_steps 必须为正整数。")
        if self.adr_step_size <= 0:
            raise ValueError("adr_step_size 必须大于 0。")
        if self.max_velocity < 0 or self.max_diffusion < 0 or self.max_viscosity < 0:
            raise ValueError("速度、扩散、粘度的上界不能为负。")
        if self.advection_scheme not in ("central", "upwind"):
            raise ValueError(
                "advection_scheme 只能是 'central' 或 'upwind'。"
            )
        if self.step_modulation_range < 0 or self.step_modulation_range > 1:
            raise ValueError(
                "step_modulation_range 必须在 [0, 1] 范围内。"
            )
        if self.extractor_dropout_rate < 0 or self.extractor_dropout_rate >= 1:
            raise ValueError("extractor_dropout_rate 必须在 [0, 1) 范围内。")
        if self.reconstructor_dropout_rate < 0 or self.reconstructor_dropout_rate >= 1:
            raise ValueError("reconstructor_dropout_rate 必须在 [0, 1) 范围内。")

        q_shape = (self.q_depth, self.q_height, self.q_width)
        field_shape = (self.n, self.n, self.h)
        for q_size, field_size, axis in zip(
            q_shape, field_shape, ("depth", "height", "width")
        ):
            if q_size <= 0:
                raise ValueError(f"q_{axis} 必须为正整数。")
            if q_size > field_size:
                raise ValueError(
                    f"q_{axis}={q_size} 不能大于对应容器尺寸 {field_size}。"
                )

    @property
    def field_shape(self) -> tuple[int, int, int]:
        return self.n, self.n, self.h

    @property
    def capacity(self) -> int:
        return self.n * self.n

    @property
    def resolved_static_adr_steps(self) -> int:
        return (
            self.capacity
            if self.static_adr_steps is None
            else self.static_adr_steps
        )

    @property
    def q_shape(self) -> tuple[int, int, int]:
        return self.q_depth, self.q_height, self.q_width

    @property
    def q_flat_dim(self) -> int:
        return (
            self.q_channels
            * self.q_depth
            * self.q_height
            * self.q_width
        )


@dataclass
class FluidResidualSequenceOutput:
    """前向传播结果。"""
    outputs: Tensor
    q_tensors: Tensor
    reconstructed_components: Tensor
    residual_fields: Tensor
    stop_steps: Tensor
    final_field: Tensor


class ZeroPreservingMLP(nn.Module):
    """满足 f(0)=0 的 MLP。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ContainerVelocityField(nn.Module):
    """
    容器速度场生成器。

    输入：当前步数 step_idx [B, 1]
    输出：v [B, 3, n, n, h]（三个方向的速度）

    速度场只由“当前演化步数”和“空间位置”决定，
    完全不依赖于当前场内容（J）。
    """

    def __init__(self, config: FluidResidualSequenceConfig) -> None:
        super().__init__()
        self.config = config

        step_hidden = config.container_velocity_hidden
        self.step_encoder = nn.Sequential(
            nn.Linear(1, step_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(step_hidden, step_hidden, bias=False),
            nn.SiLU(),
        )

        pos_dim = config.container_velocity_pos_dim
        self.register_buffer(
            "position_encoding",
            self._generate_position_encoding(pos_dim)
        )

        total_dim = step_hidden + pos_dim
        self.velocity_decoder = nn.Sequential(
            nn.Linear(total_dim, step_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(step_hidden, 3, bias=False),
            nn.Tanh(),
        )

    def _generate_position_encoding(self, pos_dim: int) -> Tensor:
        n, h = self.config.n, self.config.h

        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        z = torch.linspace(-1, 1, h)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        positions = torch.stack([xx, yy, zz], dim=-1)

        encoding = []
        for i in range(pos_dim):
            freq = 1.0 / (10000 ** (i / pos_dim))
            for coord_idx in range(3):
                if i % 2 == 0:
                    encoding.append(torch.sin(positions[..., coord_idx:coord_idx+1] * freq))
                else:
                    encoding.append(torch.cos(positions[..., coord_idx:coord_idx+1] * freq))

        encoding = torch.cat(encoding, dim=-1)[..., :pos_dim]
        return encoding

    def forward(self, step_idx: Tensor) -> Tensor:
        batch_size = step_idx.shape[0]
        device = step_idx.device

        step_embed = self.step_encoder(step_idx)

        pos_embed = self.position_encoding.to(device)
        pos_embed = pos_embed.unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1
        )

        step_expanded = step_embed[:, None, None, None, :].expand(
            -1, self.config.n, self.config.n, self.config.h, -1
        )

        combined = torch.cat([step_expanded, pos_embed], dim=-1)
        v = self.velocity_decoder(combined)
        v = v.permute(0, 4, 1, 2, 3)
        v = v * self.config.max_velocity

        return v


class ADRBlock3D(nn.Module):
    """
    三维 Advection-Diffusion-Reaction 残差更新块。

    最终数学形式：
        J_{k+1} = J_k + Δτ [
            M_A(t) * A(J_k, V(t, pos)) +
            M_D(t) * D(J_k) +
            M_R(t) * R(J_k) +
            ν(J_k) * ∇²(J_k)
        ]

    其中 ν(J_k) 是由场内容动态编码的粘度场，实现自适应耗散。
    """

    def __init__(self, config: FluidResidualSequenceConfig) -> None:
        super().__init__()
        self.config = config

        hidden = config.adr_hidden_channels

        # ---------- 场上下文编码器 ----------
        self.context_net = nn.Sequential(
            nn.Conv3d(
                1,
                hidden,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv3d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            nn.SiLU(),
        )

        # ---------- 场依赖的算子 ----------
        self.velocity_head = nn.Conv3d(
            hidden, 3, kernel_size=1, bias=False
        )
        self.diffusion_head = nn.Conv3d(
            hidden, 1, kernel_size=1, bias=False
        )
        self.reaction_head = nn.Conv3d(
            hidden, 1, kernel_size=1, bias=False
        )

        # ========== NEW: 内容耦合粘度头 ==========
        # 粘度完全由场内容（context）决定
        # 输出 [0, 1] 范围，再乘以 max_viscosity
        self.viscosity_head = nn.Sequential(
            nn.Conv3d(hidden, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # =========================================

        # ========== 容器速度场 ==========
        if config.enable_container_velocity:
            self.container_velocity = ContainerVelocityField(config)
        # =================================

        # ========== 步数依赖的调制器（A/D/R） ==========
        if config.enable_step_conditioned_adr:
            step_hidden = (
                config.step_encoder_hidden
                if config.step_encoder_hidden is not None
                else hidden
            )

            self.step_encoder = nn.Sequential(
                nn.Linear(1, step_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(step_hidden, hidden, bias=False),
                nn.SiLU(),
            )

            self.modulator_head = nn.Linear(hidden, 3, bias=False)
            nn.init.zeros_(self.modulator_head.weight)
        # =================================

        # ========== NEW: 步数依赖的粘度微调 ==========
        # 注意：这只是对内容决定粘度的“微调”，而非替代
        if config.enable_step_conditioned_viscosity:
            if config.enable_step_conditioned_adr:
                # 复用已有的 step_encoder
                self.step_encoder_visc = self.step_encoder
            else:
                step_hidden = (
                    config.step_encoder_hidden
                    if config.step_encoder_hidden is not None
                    else hidden
                )
                self.step_encoder_visc = nn.Sequential(
                    nn.Linear(1, step_hidden, bias=False),
                    nn.SiLU(),
                    nn.Linear(step_hidden, hidden, bias=False),
                    nn.SiLU(),
                )

            self.viscosity_modulator = nn.Linear(hidden, 1, bias=False)
            nn.init.zeros_(self.viscosity_modulator.weight)
        # =============================================

    @staticmethod
    def _neighbor_values(
        field: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        padded = F.pad(
            field,
            pad=(1, 1, 1, 1, 1, 1),
            mode="replicate",
        )

        d_size, h_size, w_size = field.shape[-3:]

        d_minus = padded[:, :, 0:d_size, 1:h_size + 1, 1:w_size + 1]
        d_plus = padded[:, :, 2:d_size + 2, 1:h_size + 1, 1:w_size + 1]

        h_minus = padded[:, :, 1:d_size + 1, 0:h_size, 1:w_size + 1]
        h_plus = padded[:, :, 1:d_size + 1, 2:h_size + 2, 1:w_size + 1]

        w_minus = padded[:, :, 1:d_size + 1, 1:h_size + 1, 0:w_size]
        w_plus = padded[:, :, 1:d_size + 1, 1:h_size + 1, 2:w_size + 2]

        return (
            d_minus,
            d_plus,
            h_minus,
            h_plus,
            w_minus,
            w_plus,
        )

    def _advection(
        self,
        field: Tensor,
        velocity: Tensor,
    ) -> Tensor:
        (
            d_minus,
            d_plus,
            h_minus,
            h_plus,
            w_minus,
            w_plus,
        ) = self._neighbor_values(field)

        v_d = velocity[:, 0:1]
        v_h = velocity[:, 1:2]
        v_w = velocity[:, 2:3]

        if self.config.advection_scheme == "central":
            grad_d = 0.5 * (d_plus - d_minus)
            grad_h = 0.5 * (h_plus - h_minus)
            grad_w = 0.5 * (w_plus - w_minus)
        else:
            backward_d = field - d_minus
            forward_d = d_plus - field
            grad_d = torch.where(v_d >= 0, backward_d, forward_d)

            backward_h = field - h_minus
            forward_h = h_plus - field
            grad_h = torch.where(v_h >= 0, backward_h, forward_h)

            backward_w = field - w_minus
            forward_w = w_plus - field
            grad_w = torch.where(v_w >= 0, backward_w, forward_w)

        return -(v_d * grad_d + v_h * grad_h + v_w * grad_w)

    def _laplacian(self, field: Tensor) -> Tensor:
        (
            d_minus,
            d_plus,
            h_minus,
            h_plus,
            w_minus,
            w_plus,
        ) = self._neighbor_values(field)

        return (
            d_minus
            + d_plus
            + h_minus
            + h_plus
            + w_minus
            + w_plus
            - 6.0 * field
        )

    def forward(
        self,
        field: Tensor,
        step_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """
        输入：
            field: [B, 1, n, n, h]
            step_idx: [B, 1] 或 None，表示当前 ADR 迭代步数

        输出：
            [B, 1, n, n, h]
        """
        if field.ndim != 5 or field.shape[1] != 1:
            raise ValueError(
                "ADRBlock3D 需要形状 [B, 1, n, n, h] 的输入。"
            )

        # ---------- 1. 计算场上下文 ----------
        context = self.context_net(field)

        # ---------- 2. 速度场：两种模式 ----------
        if self.config.enable_container_velocity and step_idx is not None:
            velocity = self.container_velocity(step_idx)
        else:
            velocity = (
                self.config.max_velocity
                * torch.tanh(self.velocity_head(context))
            )

        # ---------- 3. 扩散和反应 ----------
        diffusion = (
            self.config.max_diffusion
            * torch.sigmoid(self.diffusion_head(context))
        )

        reaction = (
            self.config.reaction_scale
            * torch.tanh(self.reaction_head(context))
        )

        # ---------- 4. 计算差分项 ----------
        advection_term = self._advection(field, velocity)
        diffusion_term = diffusion * self._laplacian(field)

        # ========== 5. NEW: 内容耦合粘度项 ==========
        # 粘度完全由场内容（context）动态编码
        viscosity_raw = self.viscosity_head(context)  # [B, 1, n, n, h]
        viscosity = viscosity_raw * self.config.max_viscosity

        # 可选：步数微调（不改变内容决定性，仅作整体尺度调节）
        if (
            self.config.enable_step_conditioned_viscosity
            and step_idx is not None
        ):
            step_embed = self.step_encoder_visc(step_idx)  # [B, hidden]
            visc_mod = self.viscosity_modulator(step_embed)  # [B, 1]
              # 修复：3 次 unsqueeze 变成 5D
            visc_mod = visc_mod.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1, 1]
        # 可选断言（建议保留）
            assert visc_mod.ndim == 5, f"visc_mod 应为5D，实际为 {visc_mod.ndim}D"
            visc_mod = 1.0 + 0.5 * torch.tanh(visc_mod)      # 范围 [0.5, 1.5]
            viscosity = viscosity * visc_mod                  # [B, 1, n, n, h]


        # 粘度项：二阶耗散，作用于场的高频分量
        viscous_term = viscosity * self._laplacian(field)
        # ==============================================

        # ---------- 6. 步数调制（A/D/R 全量调制） ----------
        if (
            self.config.enable_step_conditioned_adr
            and step_idx is not None
        ):
            step_embed = self.step_encoder(step_idx)
            raw_mod = self.modulator_head(step_embed)

            delta = (
                self.config.step_modulation_range
                * torch.tanh(raw_mod)
            )
            modulator = 1.0 + delta

            m_adv = modulator[:, 0:1, None, None, None]
            m_diff = modulator[:, 1:2, None, None, None]
            m_react = modulator[:, 2:3, None, None, None]

            advection_term = advection_term * m_adv
            diffusion_term = diffusion_term * m_diff
            reaction = reaction * m_react

        # ---------- 7. 更新 ----------
        update = advection_term + diffusion_term + reaction + viscous_term
        return field + self.config.adr_step_size * update


class FieldExtractor3D(nn.Module):
    """从剩余场 J_r 中压缩提取小张量 Q_r。"""

    def __init__(self, config: FluidResidualSequenceConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.extractor_hidden_channels

        self.features = nn.Sequential(
            nn.Conv3d(
                1,
                hidden,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv3d(
                hidden,
                config.q_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            nn.SiLU(),
        )

        self.dropout = nn.Dropout3d(config.extractor_dropout_rate)
        self.pool = nn.AdaptiveAvgPool3d(config.q_shape)

    def forward(self, field: Tensor) -> Tensor:
        features = self.features(field)
        if self.training:
            features = self.dropout(features)
        return self.pool(features)


class FieldReconstructor3D(nn.Module):
    """将 Q_r 放大并重构为原始场大小的 L_r。"""

    def __init__(self, config: FluidResidualSequenceConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.reconstructor_hidden_channels
        self.target_shape = config.field_shape

        self.input_projection = nn.Conv3d(
            config.q_channels,
            hidden,
            kernel_size=1,
            bias=False,
        )

        self.base_projection = nn.Conv3d(
            hidden,
            1,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            bias=False,
        )

        self.residual_branch = nn.Sequential(
            nn.Conv3d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv3d(
                hidden,
                1,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
        )

        self.dropout = nn.Dropout3d(config.reconstructor_dropout_rate)

    def forward(self, q_tensor: Tensor) -> Tensor:
        hidden = self.input_projection(q_tensor)

        enlarged = F.interpolate(
            hidden,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False,
        )

        base = self.base_projection(enlarged)
        residual = self.residual_branch(enlarged)

        out = base + residual
        if self.training:
            out = self.dropout(out)

        return out


class FluidResidualSequenceNetwork(nn.Module):
    """
    完整封装。

    支持两种输入方式:
    1. mode="nonsequential": 输入完整场 [B, n, n, h] 或 [B, 1, n, n, h]
    2. mode="sequential": 输入序列 [B, T, h]，T <= n*n

    核心增强：
        1. ADR 中速度场可由容器速度场提供（与场内容解耦）
        2. A/D/R 三个算子均随步数自适应调制
        3. 粘度（Viscosity）由场内容动态编码，实现自适应耗散
        4. Dropout 防止过拟合
    """

    def __init__(self, config: FluidResidualSequenceConfig) -> None:
        super().__init__()
        self.config = config

        self.adr = ADRBlock3D(config)
        self.extractor = FieldExtractor3D(config)
        self.output_head = ZeroPreservingMLP(
            input_dim=config.q_flat_dim,
            hidden_dim=config.output_hidden_dim,
            output_dim=config.output_dim,
        )
        self.reconstructor = FieldReconstructor3D(config)

    def _step_tensor(
        self,
        step_value: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        return torch.full(
            (batch_size, 1),
            fill_value=float(step_value),
            device=device,
            dtype=dtype,
        )

    def _validate_field(self, field: Tensor) -> Tensor:
        if field.ndim == 4:
            field = field.unsqueeze(1)

        expected = (1, self.config.n, self.config.n, self.config.h)

        if field.ndim != 5 or tuple(field.shape[1:]) != expected:
            raise ValueError(
                "非时序输入必须是 "
                f"[B, {self.config.n}, {self.config.n}, {self.config.h}] "
                "或 "
                f"[B, 1, {self.config.n}, {self.config.n}, {self.config.h}]，"
                f"实际得到 {tuple(field.shape)}。"
            )

        return field

    def encode_nonsequential(self, field: Tensor) -> Tensor:
        current = self._validate_field(field)
        batch_size = current.shape[0]
        device = current.device
        dtype = current.dtype

        total_steps = self.config.resolved_static_adr_steps

        for t in range(total_steps):
            step_idx = self._step_tensor(t, batch_size, device, dtype)
            current = self.adr(current, step_idx)

        return current

    def _make_slot_injection(
        self,
        value: Tensor,
        slot_index: int,
    ) -> Tensor:
        if value.ndim != 2 or value.shape[-1] != self.config.h:
            raise ValueError(
                f"单步时序向量必须是 [B, {self.config.h}]，"
                f"实际得到 {tuple(value.shape)}。"
            )

        row = slot_index // self.config.n
        col = slot_index % self.config.n

        batch_size = value.shape[0]
        injection = value.new_zeros(
            batch_size,
            1,
            self.config.n,
            self.config.n,
            self.config.h,
        )

        slot_mask = value.new_zeros(
            1,
            1,
            self.config.n,
            self.config.n,
            1,
        )
        slot_mask[:, :, row, col, :] = 1.0

        expanded_value = value[:, None, None, None, :]
        injection = injection + slot_mask * expanded_value

        return injection

    def encode_sequential(
        self,
        sequence: Tensor,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.config.h:
            raise ValueError(
                "时序输入必须是 "
                f"[B, T, {self.config.h}]，"
                f"实际得到 {tuple(sequence.shape)}。"
            )

        batch_size, length, _ = sequence.shape

        if length > self.config.capacity:
            raise ValueError(
                f"序列长度 {length} 超过容器容量 "
                f"n*n={self.config.capacity}。"
            )

        if sequence_mask is None:
            sequence_mask = torch.ones(
                batch_size,
                length,
                dtype=torch.bool,
                device=sequence.device,
            )
        else:
            if sequence_mask.shape != (batch_size, length):
                raise ValueError(
                    "sequence_mask 必须是 [B, T]，"
                    f"实际得到 {tuple(sequence_mask.shape)}。"
                )
            sequence_mask = sequence_mask.to(
                device=sequence.device,
                dtype=torch.bool,
            )

        device = sequence.device
        dtype = sequence.dtype

        current = sequence.new_zeros(
            batch_size,
            1,
            self.config.n,
            self.config.n,
            self.config.h,
        )

        for step_index in range(length):
            valid = sequence_mask[:, step_index]
            valid_field_mask = valid[:, None, None, None, None]

            injection = self._make_slot_injection(
                sequence[:, step_index],
                slot_index=step_index,
            )

            written = current + injection

            step_idx = self._step_tensor(step_index, batch_size, device, dtype)
            evolved = self.adr(written, step_idx)

            current = torch.where(
                valid_field_mask,
                evolved,
                current,
            )

        return current

    @staticmethod
    def q_as_matrix(q_tensor: Tensor) -> Tensor:
        return q_tensor.flatten(start_dim=2)

    def q_is_zero(self, q_tensor: Tensor) -> Tensor:
        q_matrix = self.q_as_matrix(q_tensor)
        column_norms = torch.linalg.vector_norm(
            q_matrix,
            ord=2,
            dim=1,
        )
        return (column_norms <= self.config.stop_threshold).all(dim=1)

    def decode_sequence(
        self,
        initial_field: Tensor,
        max_output_steps: Optional[int] = None,
    ) -> FluidResidualSequenceOutput:
        current = self._validate_field(initial_field)

        batch_size = current.shape[0]
        device = current.device
        dtype = current.dtype

        max_steps = (
            self.config.max_output_steps
            if max_output_steps is None
            else max_output_steps
        )
        if max_steps <= 0:
            raise ValueError("max_output_steps 必须大于 0。")

        active = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        stop_steps = torch.full(
            (batch_size,),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        outputs: list[Tensor] = []
        q_history: list[Tensor] = []
        reconstruction_history: list[Tensor] = []
        field_history: list[Tensor] = [current]

        for output_step in range(max_steps):
            active_field_mask = active[:, None, None, None, None]

            q_tensor = self.extractor(current)
            q_tensor = q_tensor * active_field_mask.to(q_tensor.dtype)

            output_vector = self.output_head(
                q_tensor.flatten(start_dim=1)
            )

            stop_now = self.q_is_zero(q_tensor)
            newly_stopped = active & stop_now

            stop_steps = torch.where(
                newly_stopped,
                torch.full_like(stop_steps, output_step),
                stop_steps,
            )

            continuing = active & ~stop_now
            continuing_field_mask = continuing[
                :, None, None, None, None
            ]

            reconstruction = self.reconstructor(q_tensor)
            reconstruction = (
                reconstruction
                * continuing_field_mask.to(reconstruction.dtype)
            )

            next_field_candidate = current - reconstruction

            if self.config.adr_after_subtraction:
                step_idx = self._step_tensor(
                    output_step, batch_size, device, dtype
                )
                next_field_candidate = self.adr(
                    next_field_candidate, step_idx
                )

            current = torch.where(
                continuing_field_mask,
                next_field_candidate,
                current,
            )

            outputs.append(output_vector)
            q_history.append(q_tensor)
            reconstruction_history.append(reconstruction)
            field_history.append(current)

            active = continuing

            if not bool(active.any()):
                break

        return FluidResidualSequenceOutput(
            outputs=torch.stack(outputs, dim=1),
            q_tensors=torch.stack(q_history, dim=1),
            reconstructed_components=torch.stack(
                reconstruction_history, dim=1
            ),
            residual_fields=torch.stack(field_history, dim=1),
            stop_steps=stop_steps,
            final_field=current,
        )

    def forward(
        self,
        inputs: Tensor,
        mode: Literal["nonsequential", "sequential"],
        *,
        sequence_mask: Optional[Tensor] = None,
        max_output_steps: Optional[int] = None,
    ) -> FluidResidualSequenceOutput:
        if mode == "nonsequential":
            if sequence_mask is not None:
                raise ValueError("非时序模式不接受 sequence_mask。")
            initial_field = self.encode_nonsequential(inputs)

        elif mode == "sequential":
            initial_field = self.encode_sequential(
                inputs,
                sequence_mask=sequence_mask,
            )

        else:
            raise ValueError(
                "mode 必须是 'nonsequential' 或 'sequential'。"
            )

        return self.decode_sequence(
            initial_field,
            max_output_steps=max_output_steps,
        )