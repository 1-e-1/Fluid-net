from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "fluid_net.py"
sys.path.insert(0, str(ROOT))

FluidResidualSequenceConfig = None
FluidResidualSequenceNetwork = None


LINKS = torch.tensor([0.45, 0.35, 0.25], dtype=torch.float32)
REACH = float(LINKS.sum())
ACTION_LIMIT = 0.12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_task_seed(name: str) -> int:
    value = 10001
    for i, ch in enumerate(name):
        value += (i + 1) * ord(ch)
    return value


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def wrap_pi(q: Tensor) -> Tensor:
    return (q + math.pi) % (2.0 * math.pi) - math.pi


def fk(q: Tensor) -> Tensor:
    links = LINKS.to(q.device, q.dtype)
    a1 = q[:, 0]
    a2 = q[:, 0] + q[:, 1]
    a3 = q[:, 0] + q[:, 1] + q[:, 2]
    x = links[0] * torch.cos(a1) + links[1] * torch.cos(a2) + links[2] * torch.cos(a3)
    y = links[0] * torch.sin(a1) + links[1] * torch.sin(a2) + links[2] * torch.sin(a3)
    return torch.stack([x, y], dim=-1)


def jacobian(q: Tensor) -> Tensor:
    links = LINKS.to(q.device, q.dtype)
    a1 = q[:, 0]
    a2 = q[:, 0] + q[:, 1]
    a3 = q[:, 0] + q[:, 1] + q[:, 2]

    j11 = -links[0] * torch.sin(a1) - links[1] * torch.sin(a2) - links[2] * torch.sin(a3)
    j12 = -links[1] * torch.sin(a2) - links[2] * torch.sin(a3)
    j13 = -links[2] * torch.sin(a3)
    j21 = links[0] * torch.cos(a1) + links[1] * torch.cos(a2) + links[2] * torch.cos(a3)
    j22 = links[1] * torch.cos(a2) + links[2] * torch.cos(a3)
    j23 = links[2] * torch.cos(a3)
    return torch.stack(
        [
            torch.stack([j11, j12, j13], dim=-1),
            torch.stack([j21, j22, j23], dim=-1),
        ],
        dim=1,
    )


def ik_step(q: Tensor, target_xy: Tensor, gain: float = 0.95) -> Tensor:
    err = target_xy - fk(q)
    j = jacobian(q)
    jj_t = j @ j.transpose(1, 2)
    eye = torch.eye(2, device=q.device, dtype=q.dtype).expand(q.shape[0], 2, 2)
    inv = torch.linalg.inv(jj_t + 0.025 * eye)
    dq = gain * (j.transpose(1, 2) @ (inv @ err[..., None])).squeeze(-1)
    return dq.clamp(-ACTION_LIMIT, ACTION_LIMIT)


def sample_reaching_batch(batch: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    q0 = (torch.rand(batch, 3, device=device) * 2.0 - 1.0) * 1.5
    ee = fk(q0)
    radius = torch.empty(batch, device=device).uniform_(0.08, 0.35)
    angle = torch.empty(batch, device=device).uniform_(-math.pi, math.pi)
    delta = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
    target = ee + delta
    norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-6)
    target = torch.where(norm > 0.92 * REACH, target / norm * (0.92 * REACH), target)
    obstacle = torch.zeros(batch, 3, device=device)
    mode = torch.zeros(batch, 1, device=device)
    return q0, target, obstacle, mode, torch.ones(batch, 1, device=device)


def sample_obstacle_batch(
    batch: int,
    device: torch.device,
    include_mode: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    half = max(1, batch // 2)
    q_half = (torch.rand(half, 3, device=device) * 2.0 - 1.0) * 1.4
    start = fk(q_half)
    radius = torch.empty(half, device=device).uniform_(0.20, 0.42)
    angle = torch.empty(half, device=device).uniform_(-math.pi, math.pi)
    delta = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
    target_half = start + delta
    norm = torch.linalg.vector_norm(target_half, dim=-1, keepdim=True).clamp_min(1e-6)
    target_half = torch.where(norm > 0.90 * REACH, target_half / norm * (0.90 * REACH), target_half)
    midpoint = 0.5 * (start + target_half)
    obs_xy = midpoint + 0.02 * torch.randn_like(midpoint)
    obs_r = torch.empty(half, 1, device=device).uniform_(0.08, 0.13)
    obstacle_half = torch.cat([obs_xy, obs_r], dim=-1)

    q0 = torch.cat([q_half, q_half], dim=0)[:batch]
    target = torch.cat([target_half, target_half], dim=0)[:batch]
    obstacle = torch.cat([obstacle_half, obstacle_half], dim=0)[:batch]
    mode = torch.cat(
        [
            torch.ones(half, 1, device=device),
            -torch.ones(half, 1, device=device),
        ],
        dim=0,
    )[:batch]
    mode_obs = mode if include_mode else torch.zeros_like(mode)
    return q0, target, obstacle, mode, mode_obs


def sample_bimodal_batch(
    batch: int,
    device: torch.device,
    include_mode: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    half = max(1, batch // 2)
    q_half = (torch.rand(half, 3, device=device) * 2.0 - 1.0) * 1.2
    ee = fk(q_half)
    radius = torch.empty(half, device=device).uniform_(0.10, 0.32)
    angle = torch.empty(half, device=device).uniform_(-math.pi, math.pi)
    delta = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
    target_half = ee + delta
    norm = torch.linalg.vector_norm(target_half, dim=-1, keepdim=True).clamp_min(1e-6)
    target_half = torch.where(norm > 0.90 * REACH, target_half / norm * (0.90 * REACH), target_half)

    q0 = torch.cat([q_half, q_half], dim=0)[:batch]
    target = torch.cat([target_half, target_half], dim=0)[:batch]
    obstacle = torch.zeros(batch, 3, device=device)
    mode = torch.cat(
        [
            torch.ones(half, 1, device=device),
            -torch.ones(half, 1, device=device),
        ],
        dim=0,
    )[:batch]
    mode_obs = mode if include_mode else torch.zeros_like(mode)
    return q0, target, obstacle, mode, mode_obs


def make_observation_tokens(
    q: Tensor,
    target: Tensor,
    obstacle: Tensor,
    mode_obs: Tensor,
) -> Tensor:
    batch = q.shape[0]
    device = q.device
    dtype = q.dtype
    ee = fk(q)
    delta = target - ee
    dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    zeros = torch.zeros(batch, 10, device=device, dtype=dtype)

    token0 = zeros.clone()
    token0[:, 0:3] = q / math.pi
    token0[:, 3:5] = ee / REACH
    token0[:, 5] = 1.0

    token1 = zeros.clone()
    token1[:, 0:2] = target / REACH
    token1[:, 2:4] = delta / REACH
    token1[:, 4:5] = dist / REACH
    token1[:, 5] = 1.0

    token2 = zeros.clone()
    token2[:, 0:2] = obstacle[:, 0:2] / REACH
    token2[:, 2:3] = obstacle[:, 2:3] / REACH
    token2[:, 3:4] = mode_obs
    token2[:, 5] = 1.0

    token3 = zeros.clone()
    token3[:, 0] = 1.0
    token3[:, 1] = -1.0
    return torch.stack([token0, token1, token2, token3], dim=1)


def expert_actions(
    q0: Tensor,
    target: Tensor,
    obstacle: Tensor,
    mode: Tensor,
    horizon: int,
    task: str,
) -> Tensor:
    q = q0.clone()
    actions = []
    start_ee = fk(q0)
    obs_xy = obstacle[:, 0:2]
    for t in range(horizon):
        if task in ("reach", "bimodal"):
            waypoint = target
        else:
            alpha = torch.full((q.shape[0], 1), (t + 1) / horizon, device=q.device, dtype=q.dtype)
            base = (1.0 - alpha) * start_ee + alpha * target
            direction = target - start_ee
            normal = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)
            normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-6)
            bump = torch.sin(math.pi * alpha) * (obstacle[:, 2:3] + 0.24)
            waypoint = base + mode * bump * normal
            # Pull the waypoint away if the synthetic arc still gets too close.
            away = waypoint - obs_xy
            clearance = torch.linalg.vector_norm(away, dim=-1, keepdim=True).clamp_min(1e-6)
            min_clear = obstacle[:, 2:3] + 0.06
            waypoint = torch.where(clearance < min_clear, obs_xy + away / clearance * min_clear, waypoint)
        dq = ik_step(q, waypoint)
        if task == "bimodal":
            j = jacobian(q)
            jj_t = j @ j.transpose(1, 2)
            eye2 = torch.eye(2, device=q.device, dtype=q.dtype).expand(q.shape[0], 2, 2)
            pinv = j.transpose(1, 2) @ torch.linalg.inv(jj_t + 0.025 * eye2)
            eye3 = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], 3, 3)
            null = eye3 - pinv @ j
            desired = torch.zeros_like(q)
            desired[:, 1] = mode[:, 0] * 1.15
            desired[:, 2] = -mode[:, 0] * 0.75
            posture = (null @ (desired - q)[..., None]).squeeze(-1)
            dq = (dq + 0.055 * posture).clamp(-ACTION_LIMIT, ACTION_LIMIT)
        q = wrap_pi(q + dq)
        actions.append(dq / ACTION_LIMIT)
    return torch.stack(actions, dim=1)


def rollout_open_loop(q0: Tensor, action_norm: Tensor) -> Tensor:
    return fk(rollout_joint(q0, action_norm))


def rollout_joint(q0: Tensor, action_norm: Tensor) -> Tensor:
    q = q0.clone()
    for t in range(action_norm.shape[1]):
        q = wrap_pi(q + action_norm[:, t].clamp(-1.0, 1.0) * ACTION_LIMIT)
    return q


def path_clearance(q0: Tensor, action_norm: Tensor, obstacle: Tensor) -> Tensor:
    q = q0.clone()
    min_clear = torch.full((q.shape[0],), 1e9, device=q.device, dtype=q.dtype)
    for t in range(action_norm.shape[1]):
        q = wrap_pi(q + action_norm[:, t].clamp(-1.0, 1.0) * ACTION_LIMIT)
        ee = fk(q)
        clear = torch.linalg.vector_norm(ee - obstacle[:, 0:2], dim=-1) - obstacle[:, 2]
        min_clear = torch.minimum(min_clear, clear)
    return min_clear


def build_model(
    horizon: int,
    step_conditioned: bool,
    static_adr_steps: int,
    q_channels: int,
    q_depth: int,
    q_height: int,
    q_width: int,
    adr_hidden_channels: int,
    extractor_hidden_channels: int,
    output_hidden_dim: int,
    adr_step_size: float,
    advection_scheme: str,
    max_velocity: float,
    max_diffusion: float,
    reaction_scale: float,
    step_modulation_range: float,
) -> FluidResidualSequenceNetwork:
    extra_kwargs = {}
    config_fields = {f.name for f in fields(FluidResidualSequenceConfig)}
    if "enable_container_velocity" in config_fields:
        extra_kwargs["enable_container_velocity"] = getattr(build_model, "enable_container_velocity")
        extra_kwargs["container_velocity_hidden"] = getattr(build_model, "container_velocity_hidden")
        extra_kwargs["container_velocity_pos_dim"] = getattr(build_model, "container_velocity_pos_dim")
    if "max_viscosity" in config_fields:
        extra_kwargs["max_viscosity"] = getattr(build_model, "max_viscosity")
        extra_kwargs["enable_step_conditioned_viscosity"] = getattr(
            build_model, "enable_step_conditioned_viscosity"
        )
    if "extractor_dropout_rate" in config_fields:
        extra_kwargs["extractor_dropout_rate"] = getattr(build_model, "extractor_dropout_rate")
        extra_kwargs["reconstructor_dropout_rate"] = getattr(build_model, "reconstructor_dropout_rate")
    cfg = FluidResidualSequenceConfig(
        n=4,
        h=10,
        output_dim=3,
        max_output_steps=horizon,
        static_adr_steps=static_adr_steps,
        q_channels=q_channels,
        q_depth=q_depth,
        q_height=q_height,
        q_width=q_width,
        adr_step_size=adr_step_size,
        advection_scheme=advection_scheme,
        max_velocity=max_velocity,
        max_diffusion=max_diffusion,
        reaction_scale=reaction_scale,
        step_modulation_range=step_modulation_range,
        adr_hidden_channels=adr_hidden_channels,
        extractor_hidden_channels=extractor_hidden_channels,
        reconstructor_hidden_channels=extractor_hidden_channels,
        output_hidden_dim=output_hidden_dim,
        enable_step_conditioned_adr=step_conditioned,
        **extra_kwargs,
    )
    return FluidResidualSequenceNetwork(cfg)


def make_batch(
    batch: int,
    device: torch.device,
    horizon: int,
    task: str,
    include_mode: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if task == "reach":
        q0, target, obstacle, mode, mode_obs = sample_reaching_batch(batch, device)
    elif task == "bimodal":
        q0, target, obstacle, mode, mode_obs = sample_bimodal_batch(batch, device, include_mode)
    else:
        q0, target, obstacle, mode, mode_obs = sample_obstacle_batch(batch, device, include_mode)
    obs = make_observation_tokens(q0, target, obstacle, mode_obs)
    action = expert_actions(q0, target, obstacle, mode, horizon, task)
    return obs, action, q0, target, obstacle, mode


def evaluate(
    model: FluidResidualSequenceNetwork,
    device: torch.device,
    task: str,
    include_mode: bool,
    horizon: int,
    batches: int,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    mse_values = []
    final_errors = []
    expert_errors = []
    clearances = []
    mode_matches = []
    with torch.no_grad():
        for _ in range(batches):
            obs, expert, q0, target, obstacle, mode = make_batch(batch_size, device, horizon, task, include_mode)
            pred = model(obs, mode="sequential").outputs[:, :horizon, :]
            mse_values.append(F.mse_loss(pred, expert).detach())
            final_ee = rollout_open_loop(q0, pred)
            expert_ee = rollout_open_loop(q0, expert)
            final_errors.append(torch.linalg.vector_norm(final_ee - target, dim=-1))
            expert_errors.append(torch.linalg.vector_norm(expert_ee - target, dim=-1))
            if task == "obstacle":
                clearances.append(path_clearance(q0, pred, obstacle))
            if task == "bimodal":
                pred_sign = torch.sign(pred[:, :, 1].sum(dim=1))
                expert_sign = torch.sign(expert[:, :, 1].sum(dim=1))
                requested_sign = torch.sign(mode[:, 0])
                # If an IK target creates near-zero joint-2 motion, fall back to the requested mode sign.
                expert_sign = torch.where(expert_sign == 0, requested_sign, expert_sign)
                pred_sign = torch.where(pred_sign == 0, -requested_sign, pred_sign)
                mode_match = (pred_sign == expert_sign).float()
                mode_matches.append(mode_match)
    final_err = torch.cat(final_errors)
    expert_err = torch.cat(expert_errors)
    result = {
        "action_mse": float(torch.stack(mse_values).mean().cpu()),
        "open_loop_final_error": float(final_err.mean().cpu()),
        "open_loop_success_5cm": float((final_err < 0.05).float().mean().cpu()),
        "expert_final_error": float(expert_err.mean().cpu()),
        "expert_success_5cm": float((expert_err < 0.05).float().mean().cpu()),
    }
    if clearances:
        clearance = torch.cat(clearances)
        result["min_clearance_mean"] = float(clearance.mean().cpu())
        result["collision_rate"] = float((clearance < 0.0).float().mean().cpu())
    if mode_matches:
        result["mode_action_sign_acc"] = float(torch.cat(mode_matches).mean().cpu())
    return result


def receding_horizon_eval(
    model: FluidResidualSequenceNetwork,
    device: torch.device,
    task: str,
    include_mode: bool,
    horizon: int,
    cases: int,
    rollout_steps: int,
) -> dict[str, float]:
    model.eval()
    if task == "reach":
        q, target, obstacle, mode, mode_obs = sample_reaching_batch(cases, device)
    elif task == "bimodal":
        q, target, obstacle, mode, mode_obs = sample_bimodal_batch(cases, device, include_mode)
    else:
        q, target, obstacle, mode, mode_obs = sample_obstacle_batch(cases, device, include_mode)
    min_clear = torch.full((cases,), 1e9, device=device)
    with torch.no_grad():
        for _ in range(rollout_steps):
            obs = make_observation_tokens(q, target, obstacle, mode_obs)
            pred = model(obs, mode="sequential").outputs[:, 0, :].clamp(-1.0, 1.0)
            q = wrap_pi(q + pred * ACTION_LIMIT)
            if task == "obstacle":
                ee = fk(q)
                clear = torch.linalg.vector_norm(ee - obstacle[:, 0:2], dim=-1) - obstacle[:, 2]
                min_clear = torch.minimum(min_clear, clear)
    err = torch.linalg.vector_norm(fk(q) - target, dim=-1)
    result = {
        "rh_final_error": float(err.mean().cpu()),
        "rh_success_5cm": float((err < 0.05).float().mean().cpu()),
    }
    if task == "obstacle":
        result["rh_min_clearance_mean"] = float(min_clear.mean().cpu())
        result["rh_collision_rate"] = float((min_clear < 0.0).float().mean().cpu())
    return result


def train_task(
    *,
    task_name: str,
    task: str,
    include_mode: bool,
    device: torch.device,
    horizon: int,
    steps: int,
    batch_size: int,
    step_conditioned: bool,
    static_adr_steps: int,
    q_channels: int,
    q_depth: int,
    q_height: int,
    q_width: int,
    adr_hidden_channels: int,
    extractor_hidden_channels: int,
    output_hidden_dim: int,
    adr_step_size: float,
    advection_scheme: str,
    max_velocity: float,
    max_diffusion: float,
    reaction_scale: float,
    step_modulation_range: float,
    lr: float,
) -> dict[str, object]:
    set_seed(stable_task_seed(task_name))
    model = build_model(
        horizon,
        step_conditioned,
        static_adr_steps,
        q_channels,
        q_depth,
        q_height,
        q_width,
        adr_hidden_channels,
        extractor_hidden_channels,
        output_hidden_dim,
        adr_step_size,
        advection_scheme,
        max_velocity,
        max_diffusion,
        reaction_scale,
        step_modulation_range,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        model.train()
        obs, action, _, _, _, _ = make_batch(batch_size, device, horizon, task, include_mode)
        opt.zero_grad(set_to_none=True)
        pred = model(obs, mode="sequential").outputs[:, :horizon, :]
        loss = F.mse_loss(pred, action)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    sync(device)
    train_seconds = time.perf_counter() - t0
    open_loop = evaluate(model, device, task, include_mode, horizon, batches=8, batch_size=batch_size)
    receding = receding_horizon_eval(
        model,
        device,
        task,
        include_mode,
        horizon,
        cases=512,
        rollout_steps=horizon,
    )
    return {
        "task": task_name,
        "status": "ok",
        "domain": "robot_arm_sequence_generation",
        "paper_analogy": "observation-conditioned action horizon prediction with receding-horizon rollout",
        "task_type": task,
        "include_mode": include_mode,
        "horizon": horizon,
        "action_dim": 3,
        "observation_shape": "[B,4,10]",
        "steps": steps,
        "batch_size": batch_size,
        "step_conditioned_adr": step_conditioned,
        "static_adr_steps": static_adr_steps,
        "q_channels": q_channels,
        "q_depth": q_depth,
        "q_height": q_height,
        "q_width": q_width,
        "adr_hidden_channels": adr_hidden_channels,
        "extractor_hidden_channels": extractor_hidden_channels,
        "output_hidden_dim": output_hidden_dim,
        "adr_step_size": adr_step_size,
        "advection_scheme": advection_scheme,
        "max_velocity": max_velocity,
        "max_diffusion": max_diffusion,
        "reaction_scale": reaction_scale,
        "step_modulation_range": step_modulation_range,
        "lr": lr,
        "params": sum(p.numel() for p in model.parameters()),
        "train_last_mse": last_loss,
        "open_loop": open_loop,
        "receding_horizon": receding,
        "train_seconds": train_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--module", default="fluid_net")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--step-conditioned", action="store_true")
    parser.add_argument("--static-adr-steps", type=int, default=4)
    parser.add_argument("--q-channels", type=int, default=32)
    parser.add_argument("--q-depth", type=int, default=2)
    parser.add_argument("--q-height", type=int, default=2)
    parser.add_argument("--q-width", type=int, default=5)
    parser.add_argument("--adr-hidden-channels", type=int, default=32)
    parser.add_argument("--extractor-hidden-channels", type=int, default=48)
    parser.add_argument("--output-hidden-dim", type=int, default=192)
    parser.add_argument("--adr-step-size", type=float, default=0.05)
    parser.add_argument("--advection-scheme", choices=["upwind", "central"], default="upwind")
    parser.add_argument("--max-velocity", type=float, default=1.0)
    parser.add_argument("--max-diffusion", type=float, default=0.1)
    parser.add_argument("--reaction-scale", type=float, default=1.0)
    parser.add_argument("--step-modulation-range", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--enable-container-velocity", action="store_true")
    parser.add_argument("--container-velocity-hidden", type=int, default=64)
    parser.add_argument("--container-velocity-pos-dim", type=int, default=16)
    parser.add_argument("--max-viscosity", type=float, default=0.1)
    parser.add_argument("--enable-step-conditioned-viscosity", action="store_true")
    parser.add_argument("--extractor-dropout-rate", type=float, default=0.0)
    parser.add_argument("--reconstructor-dropout-rate", type=float, default=0.0)
    parser.add_argument("--tasks", default="reach,bimodal_mode,bimodal_hidden")
    args = parser.parse_args()

    global FluidResidualSequenceConfig, FluidResidualSequenceNetwork
    module = importlib.import_module(args.module)
    FluidResidualSequenceConfig = module.FluidResidualSequenceConfig
    FluidResidualSequenceNetwork = module.FluidResidualSequenceNetwork
    build_model.enable_container_velocity = args.enable_container_velocity
    build_model.container_velocity_hidden = args.container_velocity_hidden
    build_model.container_velocity_pos_dim = args.container_velocity_pos_dim
    build_model.max_viscosity = args.max_viscosity
    build_model.enable_step_conditioned_viscosity = args.enable_step_conditioned_viscosity
    build_model.extractor_dropout_rate = args.extractor_dropout_rate
    build_model.reconstructor_dropout_rate = args.reconstructor_dropout_rate

    device = torch.device(args.device)
    module_path = Path(module.__file__).resolve()
    start_hash = sha256(module_path)
    print(
        "ENV",
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "module": args.module,
                "module_path": str(module_path),
                "enable_container_velocity": args.enable_container_velocity,
                "max_viscosity": args.max_viscosity,
                "enable_step_conditioned_viscosity": args.enable_step_conditioned_viscosity,
                "extractor_dropout_rate": args.extractor_dropout_rate,
                "reconstructor_dropout_rate": args.reconstructor_dropout_rate,
                "module_sha256_start": start_hash,
            },
            ensure_ascii=False,
        ),
    )

    selected = {x.strip() for x in args.tasks.split(",") if x.strip()}
    specs = [
        ("arm_reach_ik_sequence", "reach", True, 1000),
        ("arm_bimodal_mode_conditioned", "bimodal", True, 1200),
        ("arm_bimodal_hidden_multimodal", "bimodal", False, 1200),
        ("arm_obstacle_mode_conditioned", "obstacle", True, 1400),
        ("arm_obstacle_hidden_multimodal", "obstacle", False, 1400),
    ]
    results = []
    for task_name, task, include_mode, base_steps in specs:
        should_run = (
            "all" in selected
            or task_name in selected
            or (task == "reach" and "reach" in selected)
            or (task == "bimodal" and include_mode and "bimodal_mode" in selected)
            or (task == "bimodal" and (not include_mode) and "bimodal_hidden" in selected)
            or (task == "obstacle" and include_mode and "obstacle_mode" in selected)
            or (task == "obstacle" and (not include_mode) and "obstacle_hidden" in selected)
        )
        if not should_run:
            continue
        result = train_task(
            task_name=task_name,
            task=task,
            include_mode=include_mode,
            device=device,
            horizon=args.horizon,
            steps=max(1, int(round(base_steps * args.steps_scale))),
            batch_size=args.batch_size,
            step_conditioned=args.step_conditioned,
            static_adr_steps=args.static_adr_steps,
            q_channels=args.q_channels,
            q_depth=args.q_depth,
            q_height=args.q_height,
            q_width=args.q_width,
            adr_hidden_channels=args.adr_hidden_channels,
            extractor_hidden_channels=args.extractor_hidden_channels,
            output_hidden_dim=args.output_hidden_dim,
            adr_step_size=args.adr_step_size,
            advection_scheme=args.advection_scheme,
            max_velocity=args.max_velocity,
            max_diffusion=args.max_diffusion,
            reaction_scale=args.reaction_scale,
            step_modulation_range=args.step_modulation_range,
            lr=args.lr,
        )
        results.append(result)
        print("TASK_RESULT", json.dumps(result, ensure_ascii=False))

    end_hash = sha256(module_path)
    print(
        "SUMMARY",
        json.dumps(
            {
                "num_tasks": len(results),
                "model_file_unchanged": start_hash == end_hash,
                "module": args.module,
                "module_path": str(module_path),
                "module_sha256_start": start_hash,
                "module_sha256_end": end_hash,
                "results": [
                    {
                        "task": r["task"],
                        "action_mse": r["open_loop"]["action_mse"],
                        "open_loop_success_5cm": r["open_loop"]["open_loop_success_5cm"],
                        "rh_success_5cm": r["receding_horizon"]["rh_success_5cm"],
                        "collision_rate": r["open_loop"].get("collision_rate"),
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
