from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import torch

import robot_arm_sequence_benchmark as rb


def configure_module() -> Path:
    module = importlib.import_module("fluid_net_pro_max")
    rb.FluidResidualSequenceConfig = module.FluidResidualSequenceConfig
    rb.FluidResidualSequenceNetwork = module.FluidResidualSequenceNetwork
    return Path(module.__file__).resolve()


def set_build_attrs(cfg: dict[str, object]) -> None:
    rb.build_model.enable_container_velocity = bool(cfg.get("enable_container_velocity", False))
    rb.build_model.container_velocity_hidden = int(cfg.get("container_velocity_hidden", 64))
    rb.build_model.container_velocity_pos_dim = int(cfg.get("container_velocity_pos_dim", 16))
    rb.build_model.max_viscosity = float(cfg.get("max_viscosity", 0.1))
    rb.build_model.enable_step_conditioned_viscosity = bool(
        cfg.get("enable_step_conditioned_viscosity", False)
    )
    rb.build_model.extractor_dropout_rate = float(cfg.get("extractor_dropout_rate", 0.0))
    rb.build_model.reconstructor_dropout_rate = float(cfg.get("reconstructor_dropout_rate", 0.0))


def run_one(
    *,
    name: str,
    cfg: dict[str, object],
    task: str,
    include_mode: bool,
    steps: int,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> dict[str, object]:
    set_build_attrs(cfg)
    result = rb.train_task(
        task_name=f"{name}_{task}_{'mode' if include_mode else 'hidden'}",
        task=task,
        include_mode=include_mode,
        device=device,
        horizon=horizon,
        steps=steps,
        batch_size=batch_size,
        step_conditioned=bool(cfg.get("enable_step_conditioned_adr", False)),
        static_adr_steps=int(cfg.get("static_adr_steps", 4)),
        q_channels=int(cfg.get("q_channels", 32)),
        q_depth=int(cfg.get("q_depth", 2)),
        q_height=int(cfg.get("q_height", 2)),
        q_width=int(cfg.get("q_width", 5)),
        adr_hidden_channels=int(cfg.get("adr_hidden_channels", 32)),
        extractor_hidden_channels=int(cfg.get("extractor_hidden_channels", 48)),
        output_hidden_dim=int(cfg.get("output_hidden_dim", 192)),
        adr_step_size=float(cfg.get("adr_step_size", 0.05)),
        advection_scheme=str(cfg.get("advection_scheme", "upwind")),
        max_velocity=float(cfg.get("max_velocity", 1.0)),
        max_diffusion=float(cfg.get("max_diffusion", 0.1)),
        reaction_scale=float(cfg.get("reaction_scale", 1.0)),
        step_modulation_range=float(cfg.get("step_modulation_range", 0.5)),
        lr=float(cfg.get("lr", 1.5e-3)),
    )
    result["sweep_name"] = name
    result["sweep_config"] = cfg
    return result


def candidate_configs() -> list[tuple[str, dict[str, object]]]:
    base = {
        "q_channels": 32,
        "q_depth": 2,
        "q_height": 2,
        "q_width": 5,
        "adr_hidden_channels": 32,
        "extractor_hidden_channels": 48,
        "output_hidden_dim": 192,
        "static_adr_steps": 4,
        "adr_step_size": 0.05,
        "advection_scheme": "upwind",
        "max_velocity": 1.0,
        "max_diffusion": 0.1,
        "reaction_scale": 1.0,
        "max_viscosity": 0.1,
        "lr": 1.5e-3,
    }

    def c(**kwargs: object) -> dict[str, object]:
        out = dict(base)
        out.update(kwargs)
        return out

    return [
        ("baseline", c()),
        ("low_viscosity", c(max_viscosity=0.02)),
        ("high_viscosity", c(max_viscosity=0.20)),
        ("step_conditioned", c(enable_step_conditioned_adr=True, step_modulation_range=0.5)),
        ("container", c(enable_container_velocity=True, container_velocity_hidden=64, container_velocity_pos_dim=16)),
        (
            "container_stepcond",
            c(
                enable_container_velocity=True,
                enable_step_conditioned_adr=True,
                container_velocity_hidden=64,
                container_velocity_pos_dim=16,
            ),
        ),
        (
            "q64_capacity",
            c(q_channels=64, adr_hidden_channels=48, extractor_hidden_channels=64, output_hidden_dim=256),
        ),
        (
            "full_q_capacity",
            c(q_channels=32, q_depth=4, q_height=4, q_width=10, adr_hidden_channels=48, extractor_hidden_channels=64, output_hidden_dim=256),
        ),
        (
            "mild_dropout",
            c(extractor_dropout_rate=0.05, reconstructor_dropout_rate=0.02),
        ),
        (
            "central_advection",
            c(advection_scheme="central"),
        ),
        (
            "low_transport",
            c(max_velocity=0.5, max_diffusion=0.03, reaction_scale=0.7, max_viscosity=0.02),
        ),
        (
            "strong_transport",
            c(max_velocity=1.25, max_diffusion=0.2, reaction_scale=1.3, max_viscosity=0.2),
        ),
        (
            "small_dt",
            c(adr_step_size=0.03),
        ),
        (
            "large_dt",
            c(adr_step_size=0.08),
        ),
        (
            "lower_lr",
            c(lr=8e-4),
        ),
        (
            "higher_lr",
            c(lr=2.5e-3),
        ),
        (
            "pro_max_combo",
            c(
                q_channels=64,
                adr_hidden_channels=48,
                extractor_hidden_channels=64,
                output_hidden_dim=256,
                enable_container_velocity=True,
                enable_step_conditioned_adr=True,
                max_viscosity=0.02,
                max_velocity=0.8,
                max_diffusion=0.06,
                reaction_scale=1.0,
                lr=1.5e-3,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--stage1-steps", type=int, default=240)
    parser.add_argument("--stage2-steps", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--skip-stage2", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    module_path = configure_module()
    start_hash = rb.sha256(module_path)
    print(
        "ENV",
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "module": "fluid_net_pro_max",
                "module_path": str(module_path),
                "module_sha256_start": start_hash,
                "note": "enable_step_conditioned_viscosity is excluded because smoke test reveals a shape-broadcast bug.",
            },
            ensure_ascii=False,
        ),
    )

    all_results: list[dict[str, object]] = []
    stage1 = []
    t0 = time.perf_counter()
    for name, cfg in candidate_configs():
        result = run_one(
            name=f"stage1_{name}",
            cfg=cfg,
            task="reach",
            include_mode=True,
            steps=args.stage1_steps,
            batch_size=args.batch_size,
            horizon=args.horizon,
            device=device,
        )
        stage1.append(result)
        all_results.append(result)
        print("SWEEP_RESULT", json.dumps(result, ensure_ascii=False))

    ranked = sorted(
        stage1,
        key=lambda r: (
            -float(r["receding_horizon"]["rh_success_5cm"]),
            float(r["open_loop"]["action_mse"]),
            float(r["receding_horizon"]["rh_final_error"]),
        ),
    )
    top = ranked[: args.top_k]
    print(
        "STAGE1_TOP",
        json.dumps(
            [
                {
                    "sweep_name": r["sweep_name"],
                    "rh_success_5cm": r["receding_horizon"]["rh_success_5cm"],
                    "action_mse": r["open_loop"]["action_mse"],
                    "rh_final_error": r["receding_horizon"]["rh_final_error"],
                    "config": r["sweep_config"],
                }
                for r in top
            ],
            ensure_ascii=False,
        ),
    )

    if not args.skip_stage2:
        for r in top:
            base_name = str(r["sweep_name"]).replace("stage1_", "stage2_")
            cfg = dict(r["sweep_config"])
            for task, include_mode in [("reach", True), ("bimodal", True), ("bimodal", False)]:
                result = run_one(
                    name=base_name,
                    cfg=cfg,
                    task=task,
                    include_mode=include_mode,
                    steps=args.stage2_steps if task == "reach" else max(600, int(args.stage2_steps * 0.75)),
                    batch_size=args.batch_size,
                    horizon=args.horizon,
                    device=device,
                )
                all_results.append(result)
                print("FINAL_RESULT", json.dumps(result, ensure_ascii=False))

    end_hash = rb.sha256(module_path)
    print(
        "SUMMARY",
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - t0,
                "num_results": len(all_results),
                "model_file_unchanged": start_hash == end_hash,
                "module_sha256_start": start_hash,
                "module_sha256_end": end_hash,
                "best_stage1": [
                    {
                        "name": r["sweep_name"],
                        "rh_success_5cm": r["receding_horizon"]["rh_success_5cm"],
                        "action_mse": r["open_loop"]["action_mse"],
                    }
                    for r in top
                ],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
