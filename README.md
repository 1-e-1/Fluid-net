# Fluid-net

Research implementations of Fluid Residual Sequence Networks based on
Advection-Diffusion-Reaction (ADR) field dynamics.

## Contents

- `fluid_net.py`, `fluid_net_pro.py`, `fluid_net_pro_max.py`: the base, Pro and Pro Max sequence models.
- `improvements/`: FluidFormer and FluidFormer Lite variants.
- `experiments_v2/`: deep residual FluidNet experiments.
- `experiments_v3/models/`: enhanced, multiscale and ProMax-NC models.
- `experiments_v3/vla/`: experimental vision-language-action fusion and benchmarks.
- `benchmarks/`: image, text, time-series and robot-sequence comparisons.
- `experiments_v3/paper/`: historical research drafts; claims require their original evaluation protocols.

## Setup

Use a dedicated Python environment. Install a compatible PyTorch/torchvision pair
for your CPU or CUDA platform, then install the listed dependencies:

```sh
python -m pip install -r requirements.txt
```

Run commands from the repository root. Benchmark scripts are research experiments;
many download data or start training. Inspect their settings and budgets before use.
Several older scripts use `download=False`: acquire the appropriate dataset in
`data/` first. No original datasets or pretrained weights are distributed here.

The public export resolves the former machine-specific project paths relative to
each source file. Model equations and benchmark budgets are unchanged.
The VLA branch is experimental; its presence is not evidence of robot deployment.
JSON result files are historical records, not newly rerun measurements.

## License

Original project code is licensed under MIT. External datasets, pretrained models
and dependencies retain their respective licenses. See `PUBLICATION_NOTES.md`.
