import sys, json, time
sys.path.insert(0, ".")
sys.path.insert(0, "experiments_v2/models")
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from deep_fluid import *

set_seed(42)
device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
tr = make_subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=True, download=False, transform=t), 2000)
vl = make_subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=False, download=False, transform=t), 800)
tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
vl_ldr = DataLoader(vl, 64)

results = []

r0 = train_eval(build_baseline(), tr_ldr, vl_ldr, device, 300, lr=1e-3, name="Baseline")
r0["dataset"] = "MNIST"; results.append(r0)
print(f"Baseline: acc={r0['val_acc']:.4f}")

m1 = DeepFluidFormer(28, 8, 2, 10)
p1 = sum(p.numel() for p in m1.parameters())
print(f"Deep(h=8,nb=2): params={p1:,}")
r1 = train_eval(m1, tr_ldr, vl_ldr, device, 300, lr=1e-3, name="Deep(h=8,nb=2)")
r1["dataset"] = "MNIST"; results.append(r1)
print(f"Deep(h=8,nb=2): acc={r1['val_acc']:.4f} sec={r1['seconds']:.1f}")

print(json.dumps(results, indent=2, ensure_ascii=False))
