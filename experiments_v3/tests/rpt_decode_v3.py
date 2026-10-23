# -*- coding: utf-8 -*-
"""
rpt_decode_v3.py — fairness + wide-length extrapolation for RPT decoder.
1) addition fairness: AR trained 4k/8k/16k steps (is 0% just undertraining?),
   per-digit accuracy to expose the error-cascade pattern; RPT-pure 4k ref.
2) wide-length reverse/copy: train L~U[8,32], eval L=8/32/48/64
   (exact-match + per-token accuracy).
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "experiments_v3" / "tests"))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
from improved_fluid import set_seed, CosineWarmupScheduler
from rpt_decode_v2 import (SharedEncoder, RPTDecoder, ARDecoder,
                           gen_add_batch, gen_seq_batch, VOCAB_SEQ, VOCAB_ADD,
                           DIGITS)

torch.cuda.set_per_process_memory_fraction(0.2)
device = torch.device("cuda:0")


def train2(model, enc, task, steps, lr, bs, vocab, rng, lo=8, hi=12):
    """task: 'add' | 'reverse' | 'copy'; seq lengths uniform in [lo, hi]."""
    model.to(device); enc.to(device)
    params = list(dict.fromkeys(list(model.parameters()) + list(enc.parameters())))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    t0 = time.perf_counter()
    for s in range(steps):
        if task == "add":
            x, y = gen_add_batch(bs, rng)
        else:
            L = int(rng.integers(lo, hi + 1))
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
def eval_seq2(model, enc, mode, lengths, bs, vocab, rng):
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
        exact = float((pred == y).all(-1).sum().item()) / bs
        per_token = float((pred == y).float().mean().item())
        out[L] = {"exact": exact, "token": per_token}
    return out


@torch.no_grad()
def eval_add_detail(model, enc, bs, rng, digits=DIGITS):
    x, y = gen_add_batch(bs, rng, digits)
    x, y = x.to(device), y.to(device)
    ctx = enc(x)
    K = digits + 1
    if isinstance(model, RPTDecoder):
        logits, _ = model(ctx, K)
        pred = logits.argmax(-1)
    else:
        pred = model.generate(x, K)
    exact = float((pred == y).all(-1).sum().item()) / bs
    per_digit = [(pred[:, i] == y[:, i]).float().mean().item() for i in range(K)]
    return {"exact": exact, "per_digit": per_digit}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    report = {}

    # ---- 1) addition fairness ----
    if only in (None, "add"):
        set_seed(42); rng = np.random.default_rng(42)
        tag = "add:8+8"
        print(f"\n===== {tag} fairness =====", flush=True)
        # AR at 4k/8k/16k
        for steps in [4000, 8000, 16000]:
            set_seed(42)
            enc = SharedEncoder(VOCAB_ADD)
            ar = ARDecoder(enc, VOCAB_ADD)
            dt = train2(ar, enc, "add", steps, 1.2e-3, 128, VOCAB_ADD, rng)
            res = eval_add_detail(ar, enc, 1024, rng)
            report.setdefault(tag, {})[f"AR-{steps//1000}k"] = {"exact": res["exact"],
                "per_digit": res["per_digit"], "seconds": dt,
                "params": sum(p.numel() for p in ar.parameters())}
            print(f"  AR-{steps//1000}k: exact={res['exact']*100:.2f}% "
                  f"per_digit=" + " ".join(f"{v*100:.1f}" for v in res["per_digit"]), flush=True)
            del ar, enc; torch.cuda.empty_cache()
        # RPT-pure reference at 4k
        set_seed(42)
        enc = SharedEncoder(VOCAB_ADD)
        dec = RPTDecoder(VOCAB_ADD, use_step_embed=False)
        dt = train2(dec, enc, "add", 4000, 1.2e-3, 128, VOCAB_ADD, rng)
        res = eval_add_detail(dec, enc, 1024, rng)
        report.setdefault(tag, {})["RPT-pure-4k"] = {"exact": res["exact"],
            "per_digit": res["per_digit"], "seconds": dt,
            "params": sum(p.numel() for p in list(dec.parameters()) + list(enc.parameters()))}
        print(f"  RPT-pure-4k: exact={res['exact']*100:.2f}% "
              f"per_digit=" + " ".join(f"{v*100:.1f}" for v in res["per_digit"]), flush=True)
        del dec, enc; torch.cuda.empty_cache()

    # ---- 2) wide-length extrapolation ----
    if only in (None, "wide"):
        for mode in ["reverse", "copy"]:
            set_seed(42); rng = np.random.default_rng(42)
            tag = f"{mode}:L8-32"
            print(f"\n===== {tag} -> eval 48/64 =====", flush=True)
            for name, use_se in [("RPT-pure", False), ("RPT-hint", True)]:
                torch.cuda.reset_peak_memory_stats()
                enc = SharedEncoder(VOCAB_SEQ)
                dec = RPTDecoder(VOCAB_SEQ, use_step_embed=use_se)
                dt = train2(dec, enc, mode, 4000, 1.2e-3, 128, VOCAB_SEQ, rng, 8, 32)
                res = eval_seq2(dec, enc, mode, [8, 32, 48, 64], 256, VOCAB_SEQ, rng)
                report.setdefault(tag, {})[name] = {"by_len": res, "seconds": dt,
                    "params": sum(p.numel() for p in list(dec.parameters()) + list(enc.parameters())),
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
                print(f"  {name}: " + " ".join(
                    f"L{l}={v['exact']*100:.0f}%(t{v['token']*100:.0f})" for l, v in res.items()), flush=True)
                del enc, dec; torch.cuda.empty_cache()
            set_seed(42)
            enc = SharedEncoder(VOCAB_SEQ)
            ar = ARDecoder(enc, VOCAB_SEQ)
            dt = train2(ar, enc, mode, 4000, 1.2e-3, 128, VOCAB_SEQ, rng, 8, 32)
            res = eval_seq2(ar, enc, mode, [8, 32, 48, 64], 256, VOCAB_SEQ, rng)
            report.setdefault(tag, {})["AR-baseline"] = {"by_len": res, "seconds": dt,
                "params": sum(p.numel() for p in ar.parameters()),
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            print(f"  AR-baseline: " + " ".join(
                f"L{l}={v['exact']*100:.0f}%(t{v['token']*100:.0f})" for l, v in res.items()), flush=True)
            del ar, enc; torch.cuda.empty_cache()

    out = ROOT / "experiments_v3" / "results" / "rpt_v3.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
