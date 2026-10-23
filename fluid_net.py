"""
Fluid Residual Sequence Network (Full Step-Conditioned ADR)
============================================================

基于 fluid_residual_sequence_network.py 的完整增强版本。

核心改进
--------
将 ADR 中的 A（对流）、D（扩散）、R（反应）三个算子的超参数
全部与当前演化步数（step_idx）挂钩。

数学形式（改进后）：
    J_{k+1} = J_k + Δτ [
        M_A(t) * A(J_k) +
        M_D(t) * D(J_k) +
        M_R(t) * R(J_k)
    ]

其中 M_A、M_D、M_R 是由当前步数 t 学习出的自适应调制因子。

设计哲学
--------
将“历史阶段论”完整编码进 ADR 物理场：
- 早期（步数小）：强对流（快速传播）、强扩散（充分混合）、强反应（剧烈交互）
- 中期（步数适中）：各项趋于平稳
- 后期（步数大）：弱对流（精细调整）、弱扩散（防止过平滑）、弱反应（稳定收敛）

这使得网络能够根据当前处于演化的哪个阶段，自主调整“混合策略”，
而不是用一套固定的物理参数走完全程。
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

    # ========== NEW: 步数条件 ADR 全量调制 ==========
    # 是否启用步数依赖的 A/D/R 全量调制
    enable_step_conditioned_adr: bool = False
    # 步数编码器的隐藏维度
    step_encoder_hidden: Optional[int] = None
    # 调制因子的范围: [1 - mod_range, 1 + mod_range]
    # 默认 0.5，即调制范围 [0.5, 1.5]
    step_modulation_range: float = 0.5
    # ==============================================

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
        if self.max_velocity < 0 or self.max_diffusion < 0:
            raise ValueError("速度和扩散上界不能为负。")
        if self.advection_scheme not in ("central", "upwind"):
            raise ValueError(
                "advection_scheme 只能是 'central' 或 'upwind'。"
            )
        if self.step_modulation_range < 0 or self.step_modulation_range > 1:
            raise ValueError(
                "step_modulation_range 必须在 [0, 1] 范围内。"
            )

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


class ADRBlock3D(nn.Module):
    """
    三维 Advection-Diffusion-Reaction 残差更新块。

    改进后数学形式：
        J_{k+1} = J_k + Δτ [
            M_A(t) * A(J_k) +
            M_D(t) * D(J_k) +
            M_R(t) * R(J_k)
        ]

    其中 M_A、M_D、M_R 是由当前步数 t 学习出的自适应调制因子。

    关键性质：
        1. 当 enable_step_conditioned_adr=False 时，行为与原版完全一致。
        2. 当 enable_step_conditioned_adr=True 时，调制因子初始值为 1.0，
           因此训练初期与原版行为一致，不会引入突变。
        3. 调制因子范围被限制在 [1 - range, 1 + range] 内，保证数值稳定性。
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

        # ---------- 场依赖的原始算子 ----------
        self.velocity_head = nn.Conv3d(
            hidden, 3, kernel_size=1, bias=False
        )
        self.diffusion_head = nn.Conv3d(
            hidden, 1, kernel_size=1, bias=False
        )
        self.reaction_head = nn.Conv3d(
            hidden, 1, kernel_size=1, bias=False
        )

        # ========== NEW: 步数依赖的调制器（同时调制 A、D、R） ==========
        if config.enable_step_conditioned_adr:
            step_hidden = (
                config.step_encoder_hidden
                if config.step_encoder_hidden is not None
                else hidden
            )

            # 步数编码器：标量 -> 特征向量
            self.step_encoder = nn.Sequential(
                nn.Linear(1, step_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(step_hidden, hidden, bias=False),
                nn.SiLU(),
            )

            # 调制头：输出 3 维调制因子 [M_A, M_D, M_R]
            self.modulator_head = nn.Linear(hidden, 3, bias=False)

            # 关键初始化：权重为 0，确保初始调制因子 = 1.0
            # 这样在训练开始时，网络行为与原版严格一致
            nn.init.zeros_(self.modulator_head.weight)
        # =============================================================

    @staticmethod
    def _neighbor_values(
        field: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """使用 replicate 边界获得三个轴正负方向的相邻值。"""
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
            # 一阶迎风差分
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

        # ---------- 1. 计算场依赖的基础项 ----------
        context = self.context_net(field)

        # 速度场，限幅到 [-max_velocity, max_velocity]
        velocity = (
            self.config.max_velocity
            * torch.tanh(self.velocity_head(context))
        )

        # 扩散系数，限幅到 [0, max_diffusion]
        diffusion = (
            self.config.max_diffusion
            * torch.sigmoid(self.diffusion_head(context))
        )

        # 反应项，限幅到 [-reaction_scale, reaction_scale]
        reaction = (
            self.config.reaction_scale
            * torch.tanh(self.reaction_head(context))
        )

        # ---------- 2. 计算差分项 ----------
        advection_term = self._advection(field, velocity)
        diffusion_term = diffusion * self._laplacian(field)

        # ========== 3. NEW: 应用步数调制 ==========
        if (
            self.config.enable_step_conditioned_adr
            and step_idx is not None
        ):
            # 编码步数
            step_embed = self.step_encoder(step_idx)  # [B, hidden]

            # 生成原始调制因子（无界）
            raw_mod = self.modulator_head(step_embed)  # [B, 3]

            # 使用 tanh 将调制限制在 [-range, +range] 范围内
            # 然后加 1，使得调制因子在 [1-range, 1+range] 范围内
            # 这样确保了调制因子始终为正，且初始值为 1.0
            delta = (
                self.config.step_modulation_range
                * torch.tanh(raw_mod)
            )  # [B, 3]
            modulator = 1.0 + delta  # [B, 3]

            # 拆分三个调制因子，并扩展到空间维度
            m_adv = modulator[:, 0:1, None, None, None]    # [B, 1, 1, 1, 1]
            m_diff = modulator[:, 1:2, None, None, None]   # [B, 1, 1, 1, 1]
            m_react = modulator[:, 2:3, None, None, None]  # [B, 1, 1, 1, 1]

            # 应用调制
            # 注意：velocity 是 3 通道，调制因子是标量乘整个速度场
            # 这保持了三个方向速度的相对比例，只改变整体强度
            advection_term = advection_term * m_adv
            diffusion_term = diffusion_term * m_diff
            reaction = reaction * m_react
        # ==========================================

        # ---------- 4. 更新 ----------
        update = advection_term + diffusion_term + reaction
        return field + self.config.adr_step_size * update


class FieldExtractor3D(nn.Module):
    """从剩余场 J_r 中压缩提取小张量 Q_r。"""

    def __init__(self, config: FluidResidualSequenceConfig) -> None:
        super().__init__()
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

        self.pool = nn.AdaptiveAvgPool3d(config.q_shape)

    def forward(self, field: Tensor) -> Tensor:
        return self.pool(self.features(field))


class FieldReconstructor3D(nn.Module):
    """将 Q_r 放大并重构为原始场大小的 L_r。"""

    def __init__(self, config: FluidResidualSequenceConfig) -> None:
        super().__init__()
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

        return base + residual


class FluidResidualSequenceNetwork(nn.Module):
    """
    完整封装。

    支持两种输入方式:
    1. mode="nonsequential":
         输入完整场 [B, n, n, h] 或 [B, 1, n, n, h]

    2. mode="sequential":
         输入序列 [B, T, h]，T <= n*n
         第 a 个向量写入展平后的第 a 个槽位，
         每次写入后执行一次 ADR。

    两种输入最终都得到 J_0，并共享同一个输出递推:
        Q_r = Extractor(J_r)
        y_r = OutputHead(Q_r)
        L_r = Reconstructor(Q_r)
        J_{r+1} = J_r - L_r

    核心增强：所有 ADR 调用都会传入当前步数索引，
            使得 A、D、R 三个算子全部自适应调整。
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
        """将整型步数值转换为 [B, 1] 的张量。"""
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
        """
        非时序输入：完整场 -> 多次 ADR -> J_0

        NEW: 每次 ADR 调用都传入当前步数索引。
        """
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
        """把 value [B, h] 放入展平后的第 slot_index 个 n×n 槽位。"""
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
        """
        时序输入：逐事件写入 + 每次写入后 ADR。

        NEW: 每次 ADR 调用都传入当前序列位置作为步数索引。
        """
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
        """Q 张量 -> 列向量矩阵。"""
        return q_tensor.flatten(start_dim=2)

    def q_is_zero(self, q_tensor: Tensor) -> Tensor:
        """判断 Q 的每一个列向量模长是否都接近 0。"""
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
        """
        从 J_0 开始执行压缩—输出—重构—相减递推。

        NEW: 如果 adr_after_subtraction=True，
             每次 ADR 调用都会传入当前输出步数。
        """
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
        """
        统一前向接口。

        参数
        ----
        inputs:
            mode="nonsequential": [B, n, n, h] 或 [B, 1, n, n, h]
            mode="sequential": [B, T, h]

        mode: "nonsequential" 或 "sequential"

        sequence_mask:
            仅时序输入使用，可选 [B, T]。

        max_output_steps:
            可临时覆盖配置中的最大输出迭代次数。
        """
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