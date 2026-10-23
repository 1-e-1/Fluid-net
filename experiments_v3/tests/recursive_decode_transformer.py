# -*- coding: utf-8 -*-
"""
experiments_v3/tests/recursive_decode_transformer.py
=====================================================
Recursive-Peeling Transformer (RPT) decoder:
keep the standard Transformer ENCODER, replace the AUTOREGRESSIVE decoder
with a matching-pursuit style residual-peeling recursion.

RPT decoder, step k:
  R_k (residual slots [B,S,d]) --cross-attn--> C (encoder context)
  q_k = attention-pool(R_k)              # extract dominant component
  p_k = softmax(q_k E^T)                 # read out token k
  mark_k = MLP(soft_embed_k, q_k)        # imprint of the emitted token
  R_{k+1} = LayerNorm(R_k - mark_k)      # peel it off the residual

No autoregressive conditioning: the decoder never feeds predicted tokens
back as input; ordering must emerge from the residual dynamics.

Baselines: standard seq2seq Transformer decoder (autoregressive).

Tasks: Copy / Reverse, fixed length L=8, vocab=16.
"""
from __future__ import annotations

import sys, time, json
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
from improved_fluid import set_seed, CosineWarmupScheduler

torch.cuda.set_per_process_memory_fraction(0.2)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0")

VOCAB = 16
SEQ_LEN = 8
D = 64
HEADS = 4
ENC_LAYERS = 2
SLOTS = 4


# ==================== data ====================
def build_data(mode, n_train=8000, n_val=2000, seed=42):
    rng = np.random.default_rng(seed)
    def gen(n):
        x = rng.integers(0, VOCAB, size=(n, SEQ_LEN)).astype(np.int64)
        y = x[:, ::-1].copy() if mode == "reverse" else x.copy()
        return x, y
    xt, yt = gen(n_train); xv, yv = gen(n_val)
    tr = TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt))
    va = TensorDataset(torch.from_numpy(xv), torch.from_numpy(yv))
    return DataLoader(tr, 128, shuffle=True, drop_last=True), DataLoader(va, 256)


# ==================== shared encoder ====================
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, HEADS, dim_feedforward=4 * D,
                                           dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, ENC_LAYERS)

    def forward(self, x):
        return self.enc(self.embed(x) + self.pos)


# ==================== RPT decoder ====================
class RPTDecoder(nn.Module):
    """Recursive peeling decoder. use_step_embed=False -> pure residual-driven
    ordering; True -> adds a step-index hint (ablation)."""

    def __init__(self, use_step_embed=False):
        super().__init__()
        self.slots = SLOTS
        self.init_slots = nn.Parameter(torch.randn(1, SLOTS, D) * 0.02)
        self.query_pool = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.cross = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.cross_norm = nn.LayerNorm(D)
        self.readout_norm = nn.LayerNorm(D)
        self.embed_dec = nn.Embedding(VOCAB, D)
        self.head = nn.Linear(D, VOCAB)
        self.mark = nn.Sequential(
            nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, D))
        self.use_step_embed = use_step_embed
        if use_step_embed:
            self.step_embed = nn.Embedding(64, D)

    def forward(self, ctx, K=SEQ_LEN):
        """ctx [B,L,D] -> (logits [B,K,V], residual_norms [K])"""
        B = ctx.shape[0]
        R = self.init_slots.expand(B, -1, -1)             # [B,S,D]
        logits_list, norms = [], []
        for k in range(K):
            Rq = R
            if self.use_step_embed:
                se = self.step_embed(torch.tensor([k], device=ctx.device))  # [1,D]
                Rq = R + se.unsqueeze(1)
            attn, _ = self.cross(Rq, ctx, ctx)
            R1 = self.cross_norm(Rq + attn)               # [B,S,D]
            sim = (R1 @ self.query_pool.transpose(-1, -2)).squeeze(-1)  # [B,S]
            w = torch.softmax(sim, dim=-1)
            qk = (R1 * w.unsqueeze(-1)).sum(1)            # [B,D]
            logits = self.head(qk)                        # [B,V]
            logits_list.append(logits)

            p = torch.softmax(logits, dim=-1)
            soft_emb = p @ self.embed_dec.weight          # [B,D]
            mark = self.mark(torch.cat([soft_emb, qk], dim=-1)).unsqueeze(1)  # [B,1,D]
            raw_peel = R1 - mark
            r1_norm = float(R1.detach().norm(dim=(-2, -1)).mean().cpu())
            peel_norm = float(raw_peel.detach().norm(dim=(-2, -1)).mean().cpu())
            norms.append(peel_norm / max(r1_norm, 1e-8))
            R = self.readout_norm(raw_peel)

        return torch.stack(logits_list, dim=1), norms


# ==================== autoregressive baseline ====================
class ARDecoder(nn.Module):
    """Standard seq2seq Transformer decoder (teacher forcing / greedy)."""
    GO = VOCAB  # start token id

    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.embed = nn.Embedding(VOCAB + 1, D)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN + 1, D) * 0.02)
        layer = nn.TransformerDecoderLayer(D, HEADS, dim_feedforward=4 * D,
                                           dropout=0.1, batch_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(layer, ENC_LAYERS)
        self.head = nn.Linear(D, VOCAB)

    def forward_train(self, x, y):
        ctx = self.enc(x)
        B = x.shape[0]
        go = torch.full((B, 1), self.GO, dtype=torch.long, device=x.device)
        tgt = torch.cat([go, y[:, :-1]], dim=1)           # [B,K]
        tgt_emb = self.embed(tgt) + self.pos[:, :SEQ_LEN]
        mask = nn.Transformer.generate_square_subsequent_mask(SEQ_LEN, device=x.device)
        out = self.dec(tgt_emb, ctx, tgt_mask=mask)
        return self.head(out)                             # [B,K,V]

    def generate(self, x):
        ctx = self.enc(x)
        B = x.shape[0]
        toks = torch.full((B, 1), self.GO, dtype=torch.long, device=x.device)
        for _ in range(SEQ_LEN):
            tgt_emb = self.embed(toks) + self.pos[:, :toks.shape[1]]
            mask = nn.Transformer.generate_square_subsequent_mask(toks.shape[1], device=x.device)
            out = self.dec(tgt_emb, ctx, tgt_mask=mask)
            nxt = self.head(out[:, -1]).argmax(-1, keepdim=True)  # greedy
            toks = torch.cat([toks, nxt], dim=1)
        return toks[:, 1:]                                # [B,K]


# ==================== training ====================
def train_rpt(model, enc, dtr, dva, steps, lr, name):
    model.to(device); enc.to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(enc.parameters()), lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    it = iter(dtr)
    t0 = time.perf_counter()
    for s in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dtr); x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        ctx = enc(x)
        logits, _ = model(ctx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(enc.parameters()), 1.0)
        opt.step(); sched.step()
    model.eval(); enc.eval()
    ok, cnt, norms_acc = 0, 0, None
    with torch.no_grad():
        for x, y in dva:
            x, y = x.to(device), y.to(device)
            ctx = enc(x)
            logits, norms = model(ctx)
            pred = logits.argmax(-1)
            ok += int((pred == y).all(dim=-1).sum().cpu())
            cnt += x.shape[0]
            if norms_acc is None:
                norms_acc = [0.0] * len(norms)
            for i, v in enumerate(norms):
                norms_acc[i] += v
    n_b = len(dva)
    return {"acc": ok / cnt, "seconds": time.perf_counter() - t0,
            "params": sum(p.numel() for p in list(model.parameters()) + list(enc.parameters())),
            "residual_norms": [v / n_b for v in norms_acc] if norms_acc else None}


def train_ar(model, dtr, dva, steps, lr, name):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    it = iter(dtr)
    t0 = time.perf_counter()
    for s in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dtr); x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model.forward_train(x, y)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    ok, cnt = 0, 0
    with torch.no_grad():
        for x, y in dva:
            x, y = x.to(device), y.to(device)
            pred = model.generate(x)
            ok += int((pred == y).all(dim=-1).sum().cpu())
            cnt += x.shape[0]
    return {"acc": ok / cnt, "seconds": time.perf_counter() - t0,
            "params": sum(p.numel() for p in model.parameters())}


# ==================== main ====================
def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    report = {}

    for mode in (["copy", "reverse"] if only is None else [only]):
        print(f"\n===== task: {mode} (L={SEQ_LEN}) =====", flush=True)
        dtr, dva = build_data(mode)
        for name, use_se in [("RPT-pure", False), ("RPT-hint", True)]:
            torch.cuda.reset_peak_memory_stats()
            set_seed(42)
            enc = SharedEncoder()
            dec = RPTDecoder(use_step_embed=use_se)
            r = train_rpt(dec, enc, dtr, dva, 3000, 1.2e-3, name)
            r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            report.setdefault(mode, {})[name] = r
            print(f"  {name}: acc={r['acc']*100:.2f}% P={r['params']:,} "
                  f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
            if r["residual_norms"]:
                print(f"    peel-ratio ||R-mark||/||R|| per step: " +
                      " ".join(f"{v:.3f}" for v in r["residual_norms"]), flush=True)
            del enc, dec; torch.cuda.empty_cache()

        set_seed(42)
        enc = SharedEncoder()
        ar = ARDecoder(enc)
        r = train_ar(ar, dtr, dva, 3000, 1.2e-3, "AR-baseline")
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        report.setdefault(mode, {})["AR-baseline"] = r
        print(f"  AR-baseline: acc={r['acc']*100:.2f}% P={r['params']:,} "
              f"peak={r['peak_gb']:.2f}GB {r['seconds']:.0f}s", flush=True)
        del ar, enc; torch.cuda.empty_cache()

    out = ROOT / "experiments_v3" / "results" / "rpt_poc.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()