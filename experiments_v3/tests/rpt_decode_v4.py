# -*- coding: utf-8 -*-
"""
rpt_decode_v4.py — NLP generation: character-level language modelling.
RPT decoder vs AR decoder (standard Transformer encoder, prefix->continuation).
Corpus: Tiny Shakespeare (char-level). Prefix P=64, continuation K=32.
Metrics: val loss / BPC / per-token acc, plus a greedy sample.
"""
from __future__ import annotations
import sys, time, json, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "experiments_v3" / "tests"))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
from improved_fluid import set_seed, CosineWarmupScheduler
from rpt_decode_v2 import SharedEncoder, RPTDecoder, ARDecoder, sinusoid_pos

torch.cuda.set_per_process_memory_fraction(0.2)
device = torch.device("cuda:0")

P = 64       # prefix length
K = 32       # continuation length
STEPS = 3000
BS = 64
LR = 1.2e-3


def load_corpus():
    txt = (ROOT / "experiments_v3" / "data" / "tiny_shakespeare.txt").read_text(encoding="utf-8")
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in txt], dtype=np.int64)
    n = len(ids)
    split = int(0.9 * n)
    return ids[:split], ids[split:], len(chars)


def gen_batch(ids, bs, rng, need=P + K):
    L = len(ids)
    starts = rng.integers(0, L - need, size=bs)
    xb = np.stack([ids[s:s + P] for s in starts])
    yb = np.stack([ids[s + P:s + need] for s in starts])
    return torch.from_numpy(xb), torch.from_numpy(yb)


def train(model, enc, tr_ids, steps, bs, vocab, rng):
    model.to(device); enc.to(device)
    params = list(dict.fromkeys(list(model.parameters()) + list(enc.parameters())))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    t0 = time.perf_counter()
    for s in range(steps):
        x, y = gen_batch(tr_ids, bs, rng)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        ctx = enc(x)
        if isinstance(model, RPTDecoder):
            logits, _ = model(ctx, K)
        else:
            logits = model.forward_train(x, y)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
    model.eval(); enc.eval()
    return time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, enc, va_ids, vocab, rng, n_batch=40):
    total_loss, total_tok, total_ok = 0.0, 0, 0
    for _ in range(n_batch):
        x, y = gen_batch(va_ids, 128, rng)
        x, y = x.to(device), y.to(device)
        ctx = enc(x)
        if isinstance(model, RPTDecoder):
            logits, _ = model(ctx, K)
        else:
            logits = model.forward_train(x, y)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        total_loss += float(loss.item()) * y.numel()
        total_tok += y.numel()
        total_ok += int((logits.argmax(-1) == y).sum().item())
    avg_loss = total_loss / total_tok
    return {"loss": avg_loss, "bpc": avg_loss / math.log(2), "tok_acc": total_ok / total_tok}


@torch.no_grad()
def sample(model, enc, prompt_ids, vocab, itos, n_new=60):
    x = torch.from_numpy(prompt_ids).unsqueeze(0).to(device)
    ctx = enc(x)
    if isinstance(model, RPTDecoder):
        logits, _ = model(ctx, n_new)
        toks = logits.argmax(-1)[0]
    else:
        toks = model.generate(x, n_new)[0]
    return "".join(itos[int(t)] for t in toks)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tr_ids, va_ids, V = load_corpus()
    chars = sorted(set((ROOT / "experiments_v3" / "data" / "tiny_shakespeare.txt").read_text(encoding="utf-8")))
    itos = {i: c for i, c in enumerate(chars)}
    print(f"corpus: vocab={V} train={len(tr_ids):,} val={len(va_ids):,} P={P} K={K}", flush=True)
    prompt = np.array([chars.index(c) for c in "FIRST CITIZEN:\nBefore we proceed any further, "], dtype=np.int64)
    _seen = ROOT / "experiments_v3" / "results" / "rpt_v4_char_lm.json"
    if _seen.exists():
        report = json.loads(_seen.read_text(encoding="utf-8"))
        print("loaded previous results:", list(report.keys()), flush=True)
    else:
        report = {}
    for name, use_se in [("RPT-pure", False), ("RPT-hint", True)]:
        if name in report:
            print(f"  {name}: skipped (already in results)", flush=True)
            continue
        set_seed(42); rng = np.random.default_rng(42)
        torch.cuda.reset_peak_memory_stats()
        enc = SharedEncoder(V)
        dec = RPTDecoder(V, use_step_embed=use_se)
        dt = train(dec, enc, tr_ids, STEPS, BS, V, rng)
        ev = evaluate(dec, enc, va_ids, V, rng)
        s = sample(dec, enc, prompt, V, itos)
        report[name] = {"eval": ev, "seconds": dt,
                        "params": sum(p.numel() for p in list(dec.parameters()) + list(enc.parameters())),
                        "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "sample": s}
        print(f"  {name}: loss={ev['loss']:.3f} bpc={ev['bpc']:.3f} tok_acc={ev['tok_acc']*100:.1f}% "
              f"{dt:.0f}s", flush=True)
        print(f"    sample: {s[:90]!r}", flush=True)
        with open(_seen, "w") as _f:
            json.dump(report, _f, indent=2)
        del enc, dec; torch.cuda.empty_cache()
    if "AR-baseline" in report:
        print("  AR-baseline: skipped (already in results)", flush=True)
    else:
        set_seed(42); rng = np.random.default_rng(42)
        enc = SharedEncoder(V)
        ar = ARDecoder(enc, V)
        dt = train(ar, enc, tr_ids, STEPS, BS, V, rng)
        ev = evaluate(ar, enc, va_ids, V, rng)
        s = sample(ar, enc, prompt, V, itos)
        report["AR-baseline"] = {"eval": ev, "seconds": dt,
                                 "params": sum(p.numel() for p in ar.parameters()),
                                 "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "sample": s}
        print(f"  AR-baseline: loss={ev['loss']:.3f} bpc={ev['bpc']:.3f} tok_acc={ev['tok_acc']*100:.1f}% "
              f"{dt:.0f}s", flush=True)
        print(f"    sample: {s[:90]!r}", flush=True)
        with open(_seen, "w") as _f:
            json.dump(report, _f, indent=2)
        del ar, enc; torch.cuda.empty_cache()

    out = ROOT / "experiments_v3" / "results" / "rpt_v4_char_lm.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
