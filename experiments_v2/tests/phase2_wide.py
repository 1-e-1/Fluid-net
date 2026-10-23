"""Phase 2: Wide+Deep ADR - 宽通道 + 更多内部迭代"""
import sys, json, time, random, math
from itertools import cycle
sys.path.insert(0, ".")
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import fluid_net as fn

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

def img2field(x): return x.permute(0, 2, 3, 1).contiguous()
def make_subset(ds, n, seed=42):
    rng = np.random.default_rng(seed)
    return Subset(ds, rng.choice(len(ds), size=min(n, len(ds)), replace=False).tolist())

class FourierGlobalMix(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n, n // 2 + 1, h, 2) * 0.02)
    def forward(self, x):
        xf = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        xf = torch.view_as_real(xf) * (1.0 + self.weight)
        return torch.fft.irfft2(torch.view_as_complex(xf), dim=(1, 2), norm="ortho", s=(x.shape[1], x.shape[2])).to(x.dtype)

class WideDeepFluidFormer(nn.Module):
    """宽通道 + 深ADR迭代 + Fourier"""
    def __init__(self, n=28, h=16, num_classes=10, static_adr_steps=8):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.adr = fn.FluidResidualSequenceNetwork(
            fn.FluidResidualSequenceConfig(
                n=n, h=h, output_dim=num_classes,
                max_output_steps=1, static_adr_steps=static_adr_steps,
                q_channels=48, q_depth=4, q_height=4, q_width=2,
                extractor_hidden_channels=64,
                reconstructor_hidden_channels=64,
                output_hidden_dim=384,
                adr_hidden_channels=64,
            ))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        h = h + self.fourier_norm(self.fourier(h))
        return self.adr(h, mode=mode, max_output_steps=max_output_steps)

def train_eval(model, tr_ldr, vl_ldr, device, steps, lr, wd, name):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt,
        lambda s: s/warmup if s < warmup else 0.5*(1+math.cos(math.pi*(s-warmup)/max(1,steps-warmup))))
    tr_iter = cycle(tr_ldr); t0 = time.perf_counter()
    for _ in range(steps):
        model.train()
        x, y = next(tr_iter); x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(img2field(x), mode="nonsequential", max_output_steps=1)
        logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
        F.cross_entropy(logits, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    sec = time.perf_counter() - t0
    model.eval(); vs, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            out = model(img2field(x), mode="nonsequential", max_output_steps=1)
            logits = out.outputs[:, 0] if hasattr(out, "outputs") else out
            vs += float(F.cross_entropy(logits, y, reduction="sum").cpu())
            vc += int((logits.argmax(dim=-1) == y).sum().cpu()); vt += y.numel()
    return {"model": name, "params": sum(p.numel() for p in model.parameters()),
            "val_acc": vc/vt, "val_loss": vs/vt, "seconds": sec}

if __name__ == "__main__":
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    for ds_name, ds_cls in [("MNIST", datasets.MNIST), ("FashionMNIST", datasets.FashionMNIST)]:
        tr = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=True, download=False, transform=t), 3000)
        vl = make_subset(ds_cls(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=False, download=False, transform=t), 1000)
        tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
        vl_ldr = DataLoader(vl, 64)
        steps = 500 if ds_name == "MNIST" else 400

        print("\n" + "=" * 55, ds_name)
        
        # Baseline
        cfg = fn.FluidResidualSequenceConfig(n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
                                              q_channels=32, q_depth=7, q_height=7, q_width=1,
                                              extractor_hidden_channels=40, reconstructor_hidden_channels=40,
                                              output_hidden_dim=192)
        r0 = train_eval(fn.FluidResidualSequenceNetwork(cfg), tr_ldr, vl_ldr, device, steps, 1e-3, 1e-4, "Baseline(h=1)")
        print("  Baseline: acc={:.4f} P={:,}".format(r0["val_acc"], r0["params"]))

        # Wide+Deep variants
        for h, adr_s in [(8, 6), (12, 8), (16, 10)]:
            m = WideDeepFluidFormer(28, h, 10, adr_s)
            r = train_eval(m, tr_ldr, vl_ldr, device, steps, 1.5e-3, 2e-4,
                          "WD(h={},adr={})".format(h, adr_s))
            print("  WD(h={},adr={}): acc={:.4f} P={:,} {:.0f}s".format(h, adr_s, r["val_acc"], r["params"], r["seconds"]))
