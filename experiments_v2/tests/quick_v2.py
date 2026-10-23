import sys, time, math, random, json
from itertools import cycle
sys.path.insert(0, ".")
sys.path.insert(0, "experiments_v2/tests")
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from phase1_v2 import *

set_seed(42)
device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")

t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
tr = make_subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=True, download=False, transform=t), 2000)
vl = make_subset(datasets.MNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=False, download=False, transform=t), 800)
tr_ldr = DataLoader(tr, 64, shuffle=True, drop_last=True)
vl_ldr = DataLoader(vl, 64)

results = []

r0 = train_eval(build_baseline(), tr_ldr, vl_ldr, device, 300, 1e-3, 1e-4, "Baseline")
r0["dataset"] = "MNIST"; results.append(r0)
print("Baseline: acc={:.4f} P={:,} {:.0f}s".format(r0["val_acc"], r0["params"], r0["seconds"]))

r1 = train_eval(FluidFormerV2(28, 8, 3, 10, 4, 0.1), tr_ldr, vl_ldr, device, 300, 1.5e-3, 2e-4, "V2-S(300st)")
r1["dataset"] = "MNIST"; results.append(r1)
print("V2-S(300): acc={:.4f} P={:,} {:.0f}s".format(r1["val_acc"], r1["params"], r1["seconds"]))

r2 = train_eval(FluidFormerV2(28, 8, 3, 10, 4, 0.1), tr_ldr, vl_ldr, device, 500, 1.5e-3, 2e-4, "V2-S(500st)")
r2["dataset"] = "MNIST"; results.append(r2)
print("V2-S(500): acc={:.4f} P={:,} {:.0f}s".format(r2["val_acc"], r2["params"], r2["seconds"]))

# FMNIST
tr2 = make_subset(datasets.FashionMNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=True, download=False, transform=t), 2000)
vl2 = make_subset(datasets.FashionMNIST(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'data'), train=False, download=False, transform=t), 800)
tr_ldr2 = DataLoader(tr2, 64, shuffle=True, drop_last=True)
vl_ldr2 = DataLoader(vl2, 64)

r3 = train_eval(build_baseline(), tr_ldr2, vl_ldr2, device, 400, 1e-3, 1e-4, "Base-FMNIST")
r3["dataset"] = "FMNIST"; results.append(r3)
print("FMNIST-Base: acc={:.4f}".format(r3["val_acc"]))

r4 = train_eval(FluidFormerV2(28, 8, 3, 10, 4, 0.1), tr_ldr2, vl_ldr2, device, 400, 1.5e-3, 2e-4, "V2S-FMNIST")
r4["dataset"] = "FMNIST"; results.append(r4)
print("FMNIST-V2S: acc={:.4f}".format(r4["val_acc"]))

print(json.dumps(results, indent=2, ensure_ascii=False))
