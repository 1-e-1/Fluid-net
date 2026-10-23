# -*- coding: utf-8 -*-
"""
rpt_decode_v2.py — hardening + mechanism probing for the Recursive-Peeling
Transformer (RPT) decoder. Standard Transformer encoder, RPT decoder only.

Tasks:
  A. reverse / copy with VARIABLE length L~U[8,12]  -> eval length generalization
     at L=8/12/16/24/32 (sinusoidal positions, no length cap).
  B. big-int column addition (8 digits + 8 digits -> 9 digits LSD-first),
     sequential carry propagation test.
  C. mechanism probes on the reverse model:
       1) peel orthogonality: mean |cos(q_k, R_{k+1})| (true peeling -> ~0)
       2) step-counter probe: linear probe q_k -> k  (residual trajectory
          encoding the output position without any step embedding)
Models: RPT-pure (no step hint), RPT-hint, AR-baseline (greedy).
"""
from __future__ import annotations
import sys, time, json, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
from improved_fluid import set_seed, CosineWarmupScheduler

torch.cuda.set_per_process_memory_fraction(0.2)
device = torch.device("cuda:0")

D = 64
HEADS = 4
ENC_LAYERS = 2
VOCAB_SEQ = 16          # copy / reverse tokens
DIGITS = 8              # addition digits
VOCAB_ADD = 10


def sinusoid_pos(seq_len, d, dev):
    pos = torch.arange(seq_len, device=dev).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=dev).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(1, seq_len, d, device=dev)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


# ---------------- data ----------------
def gen_seq_batch(mode, bs, L, rng, vocab):
    x = rng.integers(0, vocab, size=(bs, L)).astype(np.int64)
    y = x[:, ::-1].copy() if mode == "reverse" else x.copy()
    return torch.from_numpy(x), torch.from_numpy(y)


def gen_add_batch(bs, rng, digits=DIGITS):
    a = rng.integers(0, 10, size=(bs, digits)).astype(np.int64)
    b = rng.integers(0, 10, size=(bs, digits)).astype(np.int64)
    x = np.concatenate([a, b], axis=1)                    # [B, 16] LSD-first
    total = np.zeros(bs, dtype=np.int64)
    p = 1
    for i in range(digits):
        total += (a[:, i] + b[:, i]) * p
        p *= 10
    y = np.zeros((bs, digits + 1), dtype=np.int64)
    t = total.copy()
    for i in range(digits + 1):
        y[:, i] = t % 10
        t //= 10
    return torch.from_numpy(x), torch.from_numpy(y)


# ---------------- encoder ----------------
class SharedEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, dim_feedforward=4 * D,
                                           dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, ENC_LAYERS)

    def forward(self, x):
        L = x.shape[1]
        return self.enc(self.embed(x) + sinusoid_pos(L, D, x.device))


# ---------------- RPT decoder ----------------
class RPTDecoder(nn.Module):
    def __init__(self, vocab, use_step_embed=False, max_steps=64):
        super().__init__()
        self.vocab = vocab
        self.slots = 4
        self.init_slots = nn.Parameter(torch.randn(1, self.slots, D) * 0.02)
        self.query_pool = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.cross = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.cross_norm = nn.LayerNorm(D)
        self.readout_norm = nn.LayerNorm(D)
        self.embed_dec = nn.Embedding(vocab, D)
        self.head = nn.Linear(D, vocab)
        self.mark = nn.Sequential(nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, D))
        self.use_step_embed = use_step_embed
        if use_step_embed:
            self.step_embed = nn.Embedding(max_steps, D)

    def forward(self, ctx, K, return_traj=False):
        B = ctx.shape[0]
        R = self.init_slots.expand(B, -1, -1)
        logits_list, peel_list, qk_list, rnext_list = [], [], [], []
        for k in range(K):
            Rq = R
            if self.use_step_embed:
                Rq = R + self.step_embed(torch.tensor([k], device=ctx.device)).unsqueeze(1)
            attn, _ = self.cross(Rq, ctx, ctx)
            R1 = self.cross_norm(Rq + attn)
            sim = (R1 @ self.query_pool.transpose(-1, -2)).squeeze(-1)
            w = torch.softmax(sim, dim=-1)
            qk = (R1 * w.unsqueeze(-1)).sum(1)
            logits = self.head(qk)
            logits_list.append(logits)

            p = torch.softmax(logits, dim=-1)
            soft_emb = p @ self.embed_dec.weight
            mark = self.mark(torch.cat([soft_emb, qk], dim=-1)).unsqueeze(1)
            raw_peel = R1 - mark
            r1_norm = R1.detach().norm(dim=(-2, -1))
            peel_norm = raw_peel.detach().norm(dim=(-2, -1))
            peel_list.append((peel_norm / r1_norm.clamp_min(1e-8)).mean().item())
            qk_list.append(qk)
            rnext_list.append(raw_peel)
            R = self.readout_norm(raw_peel)

        logits = torch.stack(logits_list, dim=1)          # [B,K,V]
        if return_traj:
            return logits, peel_list, torch.stack(qk_list, dim=1), torch.stack(rnext_list, dim=1)
        return logits, peel_list


# ---------------- AR baseline ----------------
class ARDecoder(nn.Module):
    def __init__(self, enc, vocab, max_len=64):
        super().__init__()
        self.enc = enc
        self.vocab = vocab
        self.embed = nn.Embedding(vocab + 1, D)           # +1 for <GO>
        layer = nn.TransformerDecoderLayer(D, HEADS, dim_feedforward=4 * D,
                                           dropout=0.1, batch_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(layer, ENC_LAYERS)
        self.head = nn.Linear(D, vocab)

    def forward_train(self, x, y):
        ctx = self.enc(x)
        B, K = x.shape[0], y.shape[1]
        go = torch.full((B, 1), self.vocab, dtype=torch.long, device=x.device)
        tgt = torch.cat([go, y[:, :-1]], dim=1)
        tgt_emb = self.embed(tgt) + sinusoid_pos(K, D, x.device)
        mask = nn.Transformer.generate_square_subsequent_mask(K, device=x.device)
        return self.head(self.dec(tgt_emb, ctx, tgt_mask=mask))

    def generate(self, x, K):
        ctx = self.enc(x)
        B = x.shape[0]
        toks = torch.full((B, 1), self.vocab, dtype=torch.long, device=x.device)
        for _ in range(K):
            tgt_emb = self.embed(toks) + sinusoid_pos(toks.shape[1], D, x.device)
            mask = nn.Transformer.generate_square_subsequent_mask(toks.shape[1], device=x.device)
            out = self.dec(tgt_emb, ctx, tgt_mask=mask)
            nxt = self.head(out[:, -1]).argmax(-1, keepdim=True)
            toks = torch.cat([toks, nxt], dim=1)
        return toks[:, 1:]


# ---------------- training ----------------
def train(model, enc, task, steps, lr, bs, vocab, rng, seed):
    model.to(device); enc.to(device)
    params = list(dict.fromkeys(list(model.parameters()) + list(enc.parameters())))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    t0 = time.perf_counter()
    for s in range(steps):
        if task == "add":
            x, y = gen_add_batch(bs, rng)
        else:
            L = int(rng.integers(8, 13))                  # U[8,12]
            x, y = gen_seq_batch(task, bs, L, rng, vocab)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        ctx = enc(x)
        if isinstance(model, RPTDecoder):
            logits, _ = model(ctx, y.shape[1])
        else:
            logits = model.forward_train(x, y)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
    model.eval(); enc.eval()
    return time.perf_counter() - t0


@torch.no_grad()
def eval_seq(model, enc, mode, lengths, bs, vocab, rng):
    out = {}
    for L in lengths:
        x, y = gen_seq_batch(mode, bs, L, rng, vocab)
        x, y = x.to(device), y.to(device)
        ctx = enc(x)
        if isinstance(model, RPTDecoder):
            logits, _ = model(ctx, L)
            pred = logits.argmax(-1)
        else:
            pred = model.generate(x, L)
        out[L] = float((pred == y).all(-1).sum().item()) / bs
    return out


@torch.no_grad()
def eval_add(model, enc, bs, rng, digits=DIGITS):
    x, y = gen_add_batch(bs, rng, digits)
    x, y = x.to(device), y.to(device)
    ctx = enc(x)
    K = digits + 1
    if isinstance(model, RPTDecoder):
        logits, _ = model(ctx, K)
        pred = logits.argmax(-1)
    else:
        pred = model.generate(x, K)
    return float((pred == y).all(-1).sum().item()) / bs


# ---------------- probes (reverse model) ----------------
def probe_step_counter(dec, enc, n=4096, steps=400, lr=3e-3):
    """Linear probe: can the output-position k be read from the residual-state
    trajectory q_k? (i.e., does the recursion encode a step counter?)"""
    rng = np.random.default_rng(0)
    qs, ks, lens = [], [], []
    with torch.no_grad():
        for L in [8, 10, 12]:
            x, y = gen_seq_batch("reverse", n, L, rng, VOCAB_SEQ)
            ctx = enc(x.to(device))
            _, _, qk_traj, _ = dec(ctx, L, return_traj=True)   # [B,L,D]
            qs.append(qk_traj.reshape(-1, D))
            ks.append(torch.arange(L, device=device).repeat(n))
            lens += [L] * n
    Q = torch.cat(qs); Kt = torch.cat(ks)
    maxL = max(lens)
    probe = nn.Linear(D, maxL).to(device)
    optp = torch.optim.AdamW(probe.parameters(), lr=lr)
    idx = torch.randperm(Q.shape[0], device=device)
    tr = idx[: int(0.75 * Q.shape[0])]; te = idx[int(0.75 * Q.shape[0]):]
    for _ in range(steps):
        optp.zero_grad()
        loss = F.cross_entropy(probe(Q[tr]), Kt[tr])
        loss.backward(); optp.step()
    acc = float((probe(Q[te]).argmax(-1) == Kt[te]).float().mean().item())
    return acc


@torch.no_grad()
def probe_orthogonality(dec, enc, n=4096):
    """True matching-pursuit peeling => R_{k+1} _|_ q_k => |cos| ~ 0."""
    rng = np.random.default_rng(0)
    cos_acc = []
    for L in [8, 12]:
        x, y = gen_seq_batch("reverse", n, L, rng, VOCAB_SEQ)
        ctx = enc(x.to(device))
        _, _, qk_traj, rnext = dec(ctx, L, return_traj=True)
        # rnext[k] is residual BEFORE LayerNorm at step k+1
        # rnext[k]: [B,S,D] raw residual slots after peeling at step k.
    # residual energy left along the extracted direction q_k:
    proj = (rnext * qk_traj.unsqueeze(2)).sum(-1)            # [B,L,S] dot(q_k, slot)
    denom = qk_traj.norm(dim=-1).unsqueeze(2) * rnext.norm(dim=-1).clamp_min(1e-8)
    cos = (proj / denom).abs().mean(-1)                      # [B,L] mean over slots
    cos_acc.append(cos[:, :-1].mean().item())                # skip last
    return float(np.mean(cos_acc))


# ---------------- main ----------------
def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    seeds = [42] if only else [42, 7, 2026]
    report = {}

    # A) variable-length reverse / copy with length generalization
    modes = []
    if only in (None, 'reverse', 'copy'):
        modes = [only] if only else ['reverse', 'copy']
    for mode in modes:
        for seed in seeds:
            set_seed(seed)
            rng = np.random.default_rng(seed)
            tag = f"{mode}:L8-12:s{seed}"
            print(f"\n===== {tag} =====", flush=True)
            for name, use_se in [("RPT-pure", False), ("RPT-hint", True)]:
                torch.cuda.reset_peak_memory_stats()
                enc = SharedEncoder(VOCAB_SEQ)
                dec = RPTDecoder(VOCAB_SEQ, use_step_embed=use_se)
                dt = train(dec, enc, mode, 3000, 1.2e-3, 128, VOCAB_SEQ, rng, seed)
                res = eval_seq(dec, enc, mode, [8, 12, 16, 24, 32], 512, VOCAB_SEQ, rng)
                report.setdefault(tag, {})[name] = {"acc_by_len": res, "seconds": dt,
                    "params": sum(p.numel() for p in list(dec.parameters()) + list(enc.parameters())),
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
                print(f"  {name}: " + " ".join(f"L{l}={v*100:.0f}%" for l, v in res.items()), flush=True)
                del enc, dec; torch.cuda.empty_cache()
            set_seed(seed)
            enc = SharedEncoder(VOCAB_SEQ)
            ar = ARDecoder(enc, VOCAB_SEQ)
            dt = train(ar, enc, mode, 3000, 1.2e-3, 128, VOCAB_SEQ, rng, seed)
            res = eval_seq(ar, enc, mode, [8, 12, 16, 24, 32], 512, VOCAB_SEQ, rng)
            report.setdefault(tag, {})["AR-baseline"] = {"acc_by_len": res, "seconds": dt,
                "params": sum(p.numel() for p in ar.parameters()),
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(f"  AR-baseline: " + " ".join(f"L{l}={v*100:.0f}%" for l, v in res.items()), flush=True)
            del ar, enc; torch.cuda.empty_cache()

    # B) big-int addition (single seed 42)
    if only is None or only == "add":
        set_seed(42); rng = np.random.default_rng(42)
        tag = "add:8+8"
        print(f"\n===== {tag} =====", flush=True)
        for name, use_se in [("RPT-pure", False), ("RPT-hint", True)]:
            torch.cuda.reset_peak_memory_stats()
            enc = SharedEncoder(VOCAB_ADD)
            dec = RPTDecoder(VOCAB_ADD, use_step_embed=use_se)
            dt = train(dec, enc, "add", 4000, 1.2e-3, 128, VOCAB_ADD, rng, seed)
            acc = eval_add(dec, enc, 1024, rng)
            report.setdefault(tag, {})[name] = {"acc": acc, "seconds": dt,
                "params": sum(p.numel() for p in list(dec.parameters()) + list(enc.parameters())),
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(f"  {name}: acc={acc*100:.2f}%", flush=True)
            del enc, dec; torch.cuda.empty_cache()
        set_seed(42)
        enc = SharedEncoder(VOCAB_ADD)
        ar = ARDecoder(enc, VOCAB_ADD)
        dt = train(ar, enc, "add", 4000, 1.2e-3, 128, VOCAB_ADD, rng, seed)
        acc = eval_add(ar, enc, 1024, rng)
        report.setdefault(tag, {})["AR-baseline"] = {"acc": acc, "seconds": dt,
            "params": sum(p.numel() for p in ar.parameters()),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(f"  AR-baseline: acc={acc*100:.2f}%", flush=True)
        del ar, enc; torch.cuda.empty_cache()

    # incremental save
    _out = ROOT / "experiments_v3" / "results" / "rpt_v2.json"
    with open(_out, "w") as _f:
        json.dump(report, _f, indent=2)

    # C) mechanism probes on reverse seed-42 model
    if only not in (None, "probes"):
        print(f"Saved: {_out} (probes skipped)", flush=True)
        return
    set_seed(42); rng = np.random.default_rng(42)
    enc = SharedEncoder(VOCAB_SEQ)
    dec = RPTDecoder(VOCAB_SEQ, use_step_embed=False)
    train(dec, enc, "reverse", 3000, 1.2e-3, 128, VOCAB_SEQ, rng, 42)
    pc = probe_step_counter(dec, enc)
    po = probe_orthogonality(dec, enc)
    report["probes:reverse"] = {"step_counter_probe_acc": pc, "peel_orthogonality_|cos|": po}
    print(f"\n===== probes (RPT-pure, reverse) =====", flush=True)
    print(f"  step-counter probe acc (q_k -> k): {pc*100:.1f}%   peel |cos(q_k,R_k+1)|: {po:.3f}", flush=True)
    del dec, enc; torch.cuda.empty_cache()

    out = ROOT / "experiments_v3" / "results" / "rpt_v2.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
