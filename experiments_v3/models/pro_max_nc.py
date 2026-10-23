# -*- coding: utf-8 -*-
"""
experiments_v3/models/pro_max_nc.py
====================================
ProMax-NC: merge pro_max architecture with the NC (no-compress) recipe.

User design direction:
  1. Apply Fourier global mixing (FourierGlobalMix)
  2. Remove the Q compression process (extractor/reconstructor loop)
  3. Large ADR step size (adr_step_size=0.30)
  4. Wide channels (h, adr_hidden_channels)
  5. Residual connection + LayerNorm around every ADR step
  6. KEEP pro_max architecture & encode/decode style:
       - container field [B,1,n,n,h]
       - nonsequential encode (static ADR steps) / sequential encode (slot injection)
       - iterative multi-step decode with max_output_steps + stop_threshold + active mask
       - step-conditioned ADR modulation / container velocity / content viscosity

Decode replacement (the one necessary change):
  pro_max decode = extract Q -> output -> reconstruct -> subtract (peel loop)
  pro_max-NC decode = evolve-read loop:
      J <- LayerNorm(J + ADR(J))
      out <- readout(pool(J))
      stop when |out_t - out_{t-1}| < stop_threshold (per sample)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net_pro_max as pmax
from improved_fluid import FourierGlobalMix


@dataclass
class ProMaxNCOutput:
    """Forward result of ProMaxNC."""
    outputs: Tensor        # [B, steps, output_dim]
    stop_steps: Tensor     # [B]
    final_field: Tensor    # [B,1,n,n,h]


class ProMaxNC(pmax.FluidResidualSequenceNetwork):
    """
    pro_max encode/decode skeleton without Q compression.

    Inherits: config, ADRBlock3D (with step modulation / container
    velocity / viscosity), _step_tensor, _validate_field.
    Removes: extractor / reconstructor (Q compression modules).
    Adds: FourierGlobalMix + residual/LayerNorm around each ADR step,
    and a pooled readout head replacing the Q-flatten output head.
    """

    def __init__(
        self,
        config: pmax.FluidResidualSequenceConfig,
        fourier: bool = True,
        fourier_every_decode_step: bool = False,
        pool_size: int = 4,
        decode_readout_hidden: Optional[int] = None,
    ) -> None:
        super().__init__(config)

        # ---- remove Q compression modules (unused in this design) ----
        self.extractor = None
        self.reconstructor = None

        # ---- Fourier global mixing ----
        self.fourier = FourierGlobalMix(config.n, config.h) if fourier else None
        self.fourier_norm = nn.LayerNorm([config.n, config.n, config.h])
        self.fourier_every_decode_step = fourier_every_decode_step

        # ---- residual + LayerNorm around each ADR step ----
        self.block_norm = nn.LayerNorm([config.n, config.n, config.h])

        # ---- pooled readout head (replaces Q-flatten -> ZeroPreservingMLP) ----
        self.pool_size = pool_size
        hidden = decode_readout_hidden or config.output_hidden_dim
        flat = pool_size * pool_size * config.h
        self.readout = pmax.ZeroPreservingMLP(flat, hidden, config.output_dim)

    # ------------------------------------------------------------------
    #  encode: pro_max structure + residual/LN around every ADR step
    # ------------------------------------------------------------------
    def _evolve(self, field: Tensor, step_idx: Tensor) -> Tensor:
        """One same-dim ADR step with the NC residual+LN recipe:
        out = LayerNorm(field + ADR(field)), shape [B,1,n,n,h] -> same."""
        f4 = field.squeeze(1)                          # [B,n,n,h]
        out = self.adr(f4.unsqueeze(1), step_idx)      # [B,1,n,n,h]
        merged = f4 + out.squeeze(1)                   # residual (NC recipe)
        return self.block_norm(merged).unsqueeze(1)

    def encode_nonsequential(self, field: Tensor) -> Tensor:
        current = self._validate_field(field)
        batch = current.shape[0]
        device, dtype = current.device, current.dtype
        total = self.config.resolved_static_adr_steps
        for t in range(total):
            current = self._evolve(current, self._step_tensor(t, batch, device, dtype))
        return current

    def encode_sequential(
        self,
        sequence: Tensor,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """pro_max slot-injection sequential encode, with residual+LN evolution."""
        if sequence.ndim != 3 or sequence.shape[-1] != self.config.h:
            raise ValueError(
                "sequential input must be [B, T, h], got "
                f"{tuple(sequence.shape)}."
            )
        batch, length, _ = sequence.shape
        if length > self.config.capacity:
            raise ValueError(
                f"sequence length {length} exceeds container capacity "
                f"n*n={self.config.capacity}."
            )
        if sequence_mask is None:
            sequence_mask = torch.ones(batch, length, dtype=torch.bool, device=sequence.device)
        else:
            sequence_mask = sequence_mask.to(device=sequence.device, dtype=torch.bool)

        device, dtype = sequence.device, sequence.dtype
        current = sequence.new_zeros(batch, 1, self.config.n, self.config.n, self.config.h)

        for step_index in range(length):
            valid = sequence_mask[:, step_index]
            valid_field_mask = valid[:, None, None, None, None]
            injection = self._make_slot_injection(sequence[:, step_index], slot_index=step_index)
            written = current + injection
            step_idx = self._step_tensor(step_index, batch, device, dtype)
            evolved = self._evolve(written, step_idx)
            current = torch.where(valid_field_mask, evolved, current)

        return current

    # ------------------------------------------------------------------
    #  decode: evolve-read loop (no Q compression)
    # ------------------------------------------------------------------
    def decode_sequence(
        self,
        initial_field: Tensor,
        max_output_steps: Optional[int] = None,
    ) -> ProMaxNCOutput:
        current = self._validate_field(initial_field)
        batch = current.shape[0]
        device, dtype = current.device, current.dtype

        # Fourier global mix once after encode (residual + LN)
        if self.fourier is not None:
            f4 = current.squeeze(1)
            current = self.fourier_norm(f4 + self.fourier(f4)).unsqueeze(1)

        max_steps = (
            self.config.max_output_steps
            if max_output_steps is None
            else max_output_steps
        )
        if max_steps <= 0:
            raise ValueError("max_output_steps must be > 0.")

        active = torch.ones(batch, dtype=torch.bool, device=device)
        stop_steps = torch.full((batch,), -1, dtype=torch.long, device=device)
        outputs: list[Tensor] = []
        prev_out: Optional[Tensor] = None

        for step in range(max_steps):
            # evolve one step (same-dim, big step, residual + LN)
            current = self._evolve(current, self._step_tensor(step, batch, device, dtype))

            if self.fourier is not None and self.fourier_every_decode_step:
                f4 = current.squeeze(1)
                current = self.fourier_norm(f4 + self.fourier(f4)).unsqueeze(1)

            # pooled readout (replaces Q flatten)
            pooled = F.adaptive_avg_pool2d(
                current.squeeze(1).permute(0, 3, 1, 2),
                (self.pool_size, self.pool_size),
            )
            out = self.readout(pooled.flatten(1))            # [B, output_dim]

            # stop when output stabilizes (replaces Q ~ 0 criterion)
            if prev_out is not None:
                delta = (out - prev_out).abs().amax(dim=-1)  # [B]
                newly_stopped = active & (delta < self.config.stop_threshold)
                stop_steps = torch.where(
                    newly_stopped,
                    torch.full_like(stop_steps, step),
                    stop_steps,
                )
                active = active & ~newly_stopped

            prev_out = out
            outputs.append(out)

            if not bool(active.any()):
                break

        return ProMaxNCOutput(
            outputs=torch.stack(outputs, dim=1),
            stop_steps=stop_steps,
            final_field=current,
        )

    def forward(
        self,
        inputs: Tensor,
        mode: Literal["nonsequential", "sequential"] = "nonsequential",
        *,
        sequence_mask: Optional[Tensor] = None,
        max_output_steps: Optional[int] = None,
    ) -> ProMaxNCOutput:
        if mode == "nonsequential":
            if sequence_mask is not None:
                raise ValueError("nonsequential mode does not accept sequence_mask.")
            field = self.encode_nonsequential(inputs)
        else:
            field = self.encode_sequential(inputs, sequence_mask=sequence_mask)
        return self.decode_sequence(field, max_output_steps=max_output_steps)


def make_config(
    n: int,
    h: int,
    output_dim: int,
    adr_hidden_channels: int = 64,
    adr_step_size: float = 0.30,
    static_adr_steps: int = 4,
    max_output_steps: int = 1,
    stop_threshold: float = 1e-4,
    enable_step_conditioned_adr: bool = True,
    enable_container_velocity: bool = True,
    max_viscosity: float = 0.1,
    output_hidden_dim: int = 256,
    **kwargs,
) -> pmax.FluidResidualSequenceConfig:
    """Wide-channel, big-step pro_max config for the ProMaxNC network."""
    return pmax.FluidResidualSequenceConfig(
        n=n,
        h=h,
        output_dim=output_dim,
        adr_hidden_channels=adr_hidden_channels,
        adr_step_size=adr_step_size,
        static_adr_steps=static_adr_steps,
        max_output_steps=max_output_steps,
        stop_threshold=stop_threshold,
        enable_step_conditioned_adr=enable_step_conditioned_adr,
        enable_container_velocity=enable_container_velocity,
        max_viscosity=max_viscosity,
        output_hidden_dim=output_hidden_dim,
        **kwargs,
    )


# ==================== Smoke test (CPU only, tiny) ====================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = make_config(n=6, h=8, output_dim=4, static_adr_steps=2, max_output_steps=3)
    net = ProMaxNC(cfg)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ProMaxNC params: {n_params:,}")

    # nonsequential
    field = torch.randn(2, 6, 6, 8)
    out = net(field, mode="nonsequential")
    print("nonsequential outputs:", tuple(out.outputs.shape), "stop:", out.stop_steps.tolist())

    # sequential
    seq = torch.randn(2, 6, 8)
    out2 = net(seq, mode="sequential")
    print("sequential   outputs:", tuple(out2.outputs.shape), "stop:", out2.stop_steps.tolist())

class ProMaxNCStacked(pmax.FluidResidualSequenceNetwork):
    """
    Stacked version: L independent pro_max ADR blocks (each with its own
    viscosity / container velocity / step-conditioned modulation),
    wrapped with residual + LayerNorm (NC recipe).

    Difference vs ProMaxNC:
      - ProMaxNC loops ONE shared ADR block N times (params do not grow with L)
      - ProMaxNCStacked stacks L independent blocks (capacity grows with L)

    Flow (nonsequential):
      field -> FourierGlobalMix (+LN) -> L stacked ADR blocks -> readout
    Flow (sequential):
      sequence written into field slots -> Fourier -> L stacked blocks -> readout
    Decode (iterative, pro_max flavor):
      step 0 reads the encoded field; steps >0 evolve the whole stack once
      then read again; stop when output stabilizes (stop_threshold).
    """

    def __init__(
        self,
        config: pmax.FluidResidualSequenceConfig,
        num_layers: int = 4,
        fourier: bool = True,
        fourier_every_decode_step: bool = False,
        pool_size: int = 4,
        decode_readout_hidden: Optional[int] = None,
    ) -> None:
        super().__init__(config)

        # Q compression modules are not used
        self.extractor = None
        self.reconstructor = None

        # L independent pro_max ADR blocks + residual/LN
        self.num_layers = num_layers
        self.adr_blocks = nn.ModuleList(
            [pmax.ADRBlock3D(config) for _ in range(num_layers)]
        )
        self.block_norms = nn.ModuleList(
            [nn.LayerNorm([config.n, config.n, config.h]) for _ in range(num_layers)]
        )

        # Fourier global mixing
        self.fourier = FourierGlobalMix(config.n, config.h) if fourier else None
        self.fourier_norm = nn.LayerNorm([config.n, config.n, config.h])
        self.fourier_every_decode_step = fourier_every_decode_step

        # pooled readout head
        self.pool_size = pool_size
        hidden = decode_readout_hidden or config.output_hidden_dim
        flat = pool_size * pool_size * config.h
        self.readout = pmax.ZeroPreservingMLP(flat, hidden, config.output_dim)

    # ---------------- field -> Fourier -> stack ----------------
    def _fourier_mix(self, field: Tensor) -> Tensor:
        if self.fourier is None:
            return field
        f4 = field.squeeze(1)
        return self.fourier_norm(f4 + self.fourier(f4)).unsqueeze(1)

    def _evolve_stack(self, field: Tensor) -> Tensor:
        """L stacked same-dim ADR blocks, each wrapped with residual + LN."""
        B = field.shape[0]
        device, dtype = field.device, field.dtype
        f4 = field.squeeze(1)
        for li, (adr, norm) in enumerate(zip(self.adr_blocks, self.block_norms)):
            step_idx = torch.full((B, 1), float(li), dtype=torch.float32, device=device)
            out = adr(f4.unsqueeze(1), step_idx)      # [B,1,n,n,h]
            f4 = norm(f4 + out.squeeze(1))            # [B,n,n,h]
        return f4.unsqueeze(1)

    # ---------------- encode ----------------
    def encode_nonsequential(self, field: Tensor) -> Tensor:
        current = self._validate_field(field)
        current = self._fourier_mix(current)
        return self._evolve_stack(current)

    def encode_sequential(
        self,
        sequence: Tensor,
        sequence_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """NC-style: write all slots (row-major), then Fourier + stack."""
        if sequence.ndim != 3 or sequence.shape[-1] != self.config.h:
            raise ValueError(
                "sequential input must be [B, T, h], got "
                f"{tuple(sequence.shape)}."
            )
        batch, length, _ = sequence.shape
        if length > self.config.capacity:
            raise ValueError(
                f"sequence length {length} exceeds container capacity "
                f"n*n={self.config.capacity}."
            )
        n = self.config.n
        field = sequence.new_zeros(batch, 1, n, n, self.config.h)
        rows = torch.arange(length, device=sequence.device) // n
        cols = torch.arange(length, device=sequence.device) % n
        field[:, 0, rows, cols, :] = sequence
        return self._evolve_stack(self._fourier_mix(field))

    # ---------------- decode (iterative evolve-read) ----------------
    def decode_sequence(
        self,
        initial_field: Tensor,
        max_output_steps: Optional[int] = None,
    ) -> ProMaxNCOutput:
        current = self._validate_field(initial_field)
        batch = current.shape[0]
        device, dtype = current.device, current.dtype

        max_steps = (
            self.config.max_output_steps
            if max_output_steps is None
            else max_output_steps
        )
        if max_steps <= 0:
            raise ValueError("max_output_steps must be > 0.")

        active = torch.ones(batch, dtype=torch.bool, device=device)
        stop_steps = torch.full((batch,), -1, dtype=torch.long, device=device)
        outputs: list[Tensor] = []
        prev_out: Optional[Tensor] = None

        for step in range(max_steps):
            if step > 0:
                current = self._evolve_stack(current)
                if self.fourier is not None and self.fourier_every_decode_step:
                    current = self._fourier_mix(current)

            pooled = F.adaptive_avg_pool2d(
                current.squeeze(1).permute(0, 3, 1, 2),
                (self.pool_size, self.pool_size),
            )
            out = self.readout(pooled.flatten(1))          # [B, output_dim]

            if prev_out is not None:
                delta = (out - prev_out).abs().amax(dim=-1)
                newly_stopped = active & (delta < self.config.stop_threshold)
                stop_steps = torch.where(
                    newly_stopped,
                    torch.full_like(stop_steps, step),
                    stop_steps,
                )
                active = active & ~newly_stopped

            prev_out = out
            outputs.append(out)
            if not bool(active.any()):
                break

        return ProMaxNCOutput(
            outputs=torch.stack(outputs, dim=1),
            stop_steps=stop_steps,
            final_field=current,
        )

    def forward(
        self,
        inputs: Tensor,
        mode: Literal["nonsequential", "sequential"] = "nonsequential",
        *,
        sequence_mask: Optional[Tensor] = None,
        max_output_steps: Optional[int] = None,
    ) -> ProMaxNCOutput:
        if mode == "nonsequential":
            field = self.encode_nonsequential(inputs)
        else:
            field = self.encode_sequential(inputs, sequence_mask=sequence_mask)
        return self.decode_sequence(field, max_output_steps=max_output_steps)
