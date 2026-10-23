import sys, torch, time, random
from itertools import cycle
sys.path.insert(0, ".")
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import fluid_net as fn

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
device = torch.device("cuda:0")

def img2field(x): return x.permute(0, 2, 3, 1).contiguous()

class TwoLayerFixed(nn.Module):
    def __init__(self, n=28, h=4, nc=10):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.layer1 = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(
            n=n, h=h, output_dim=nc, max_output_steps=1, static_adr_steps=4,
            q_channels=16, q_depth=4, q_height=4, q_width=1,
            extractor_hidden_channels=32, reconstructor_hidden_channels=32,
            output_hidden_dim=128, adr_hidden_channels=32))
        self.norm = nn.LayerNorm([n, n, h])
        self.layer2 = fn.FluidResidualSequenceNetwork(fn.FluidResidualSequenceConfig(
            n=n, h=h, output_dim=nc, max_output_steps=1, static_adr_steps=4,
            q_channels=16, q_depth=4, q_height=4, q_width=1,
            extractor_hidden_channels=32, reconstructor_hidden_channels=32,
            output_hidden_dim=128, adr_hidden_channels=32))

    def forward(self, x, mode="nonsequential", max_output_steps=1):
        h = self.proj(x)
        out1 = self.layer1(h, mode=mode, max_output_steps=max_output_steps)
        f1 = out1.final_field.squeeze(1)
        f1_normed = self.norm(f1)
        out2 = self.layer2(f1_normed, mode=mode, max_output_steps=max_output_steps)
        return {
            "logits1": out1.outputs[:, 0],
            "logits2": out2.outputs[:, 0],
            "fn1": f1.flatten(1).norm(dim=1).mean().item(),
            "fn2": out2.final_field.squeeze(1).flatten(1).norm(dim=1).mean().item(),
        }

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
tr = Subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=True, download=False, transform=t),
            np.random.default_rng(42).choice(60000, 2000, replace=False).tolist())
vl = Subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=False, download=False, transform=t),
            np.random.default_rng(42).choice(10000, 500, replace=False).tolist())
tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
vl_ldr = DataLoader(vl, 64)

model = TwoLayerFixed(28, 4, 10).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
tr_iter = cycle(tr_ldr)

print("Step   L1_acc  L2_acc  |F1|     |F2|")
print("-" * 42)

for step in range(300):
    model.train()
    x, y = next(tr_iter); x, y = x.to(device), y.to(device)
    opt.zero_grad(set_to_none=True)
    out = model(img2field(x))
    loss = F.cross_entropy(out["logits1"], y) + F.cross_entropy(out["logits2"], y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 30 == 0:
        a1 = float((out["logits1"].argmax(dim=-1) == y).float().mean().cpu())
        a2 = float((out["logits2"].argmax(dim=-1) == y).float().mean().cpu())
        print("{:5d} {:7.4f} {:7.4f} {:8.2f} {:8.2f}".format(step, a1, a2, out["fn1"], out["fn2"]))

model.eval(); c1 = c2 = ttl = 0
with torch.no_grad():
    for x, y in vl_ldr:
        x, y = x.to(device), y.to(device)
        out = model(img2field(x))
        c1 += int((out["logits1"].argmax(dim=-1) == y).sum().cpu())
        c2 += int((out["logits2"].argmax(dim=-1) == y).sum().cpu())
        ttl += y.numel()
print("\nFIXED Val: L1={:.4f} L2={:.4f}".format(c1/ttl, c2/ttl))
print("Field norms: L1={:.2f} L2={:.2f}".format(out["fn1"], out["fn2"]))

# Compare with single layer
cfg = fn.FluidResidualSequenceConfig(n=28, h=1, output_dim=10, max_output_steps=1, static_adr_steps=4,
                                      q_channels=32, q_depth=7, q_height=7, q_width=1,
                                      extractor_hidden_channels=40, reconstructor_hidden_channels=40,
                                      output_hidden_dim=192)
m1 = fn.FluidResidualSequenceNetwork(cfg).to(device)
opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
for _ in range(300):
    m1.train()
    x, y = next(tr_iter); x, y = x.to(device), y.to(device)
    opt1.zero_grad(set_to_none=True)
    out = m1(img2field(x), mode="nonsequential", max_output_steps=1)
    F.cross_entropy(out.outputs[:, 0], y).backward()
    opt1.step()
m1.eval(); c = 0
with torch.no_grad():
    for x, y in vl_ldr:
        x, y = x.to(device), y.to(device)
        out = m1(img2field(x), mode="nonsequential", max_output_steps=1)
        c += int((out.outputs[:, 0].argmax(dim=-1) == y).sum().cpu())
print("Single layer baseline: {:.4f}".format(c/ttl))
