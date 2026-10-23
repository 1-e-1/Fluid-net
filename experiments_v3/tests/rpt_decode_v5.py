# -*- coding: utf-8 -*-
"""
rpt_decode_v5.py — variable-length output for RPT via STOP HEAD + ENERGY THRESHOLD.
Standard Transformer encoder + RPT decoder + energy gauge e_k + stop head.

Energy gauge: scalar e_k in [0,1] = "fraction of output energy remaining".
  e_0 = 1;  e_{k+1} = e_k * (1 - sigmoid(MLP(q_k)))     (multiplicative decay)
  supervised target: e_k = (L-k)/L  -> the model must infer L from context.
Stop head: P(stop) = sigmoid(StopHead([q_k, e_{k+1}])) ; target 1 on last step.
Inference: stop when stop_prob > 0.5 OR e < e_thresh (energy threshold);
           cap at K_max.

Ablations: RPT-var-full (energy supervised, stop head sees e),
           RPT-var-nogauge (stop head only, no energy supervision),
           AR baseline with EOS token.
Tasks: variable-length copy / reverse, train L~U[8,12], eval 8/12/16/24/32.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "experiments_v3" / "tests"))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))
from improved_fluid import set_seed, CosineWarmupScheduler
from rpt_decode_v2 import SharedEncoder, sinusoid_pos, gen_seq_batch

torch.cuda.set_per_process_memory_fraction(0.2)
device = torch.device("cuda:0")

D = 64
HEADS = 4
ENC_LAYERS = 2
VOCAB = 16
SLOTS = 4
L_TRAIN = (8, 12)
K_MAX = 48
E_THRESH = 0.02
LAM_S = 0.5
LAM_E = 1.0


class RPTVarDecoder(nn.Module):
    """RPT + energy gauge + stop head. use_gauge=True: supervise e and feed it
    to the stop head; False: ablation (stop head on q only, e unsupervised)."""

    def __init__(self, vocab, use_gauge=True):
        super().__init__()
        self.vocab = vocab
        self.use_gauge = use_gauge
        self.init_slots = nn.Parameter(torch.randn(1, SLOTS, D) * 0.02)
        self.query_pool = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.cross = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.cross_norm = nn.LayerNorm(D)
        self.readout_norm = nn.LayerNorm(D)
        self.embed_dec = nn.Embedding(vocab, D)
        self.head = nn.Linear(D, vocab)
        self.mark = nn.Sequential(nn.Linear(2 * D, D), nn.SiLU(), nn.Linear(D, D))
        self.energy_mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 1))
        stop_in = D + 1 if use_gauge else D
        self.stop_head = nn.Sequential(nn.Linear(stop_in, D), nn.SiLU(), nn.Linear(D, 1))

    def _step(self, R, ctx):
        attn, _ = self.cross(R, ctx, ctx)
        R1 = self.cross_norm(R + attn)
        sim = (R1 @ self.query_pool.transpose(-1, -2)).squeeze(-1)
        w = torch.softmax(sim, dim=-1)
        qk = (R1 * w.unsqueeze(-1)).sum(1)
        logits = self.head(qk)
        p = torch.softmax(logits, dim=-1)
        soft_emb = p @ self.embed_dec.weight
        mark = self.mark(torch.cat([soft_emb, qk], dim=-1)).unsqueeze(1)
        R = self.readout_norm(R1 - mark)
        return R, qk, logits

    def forward_train(self, ctx, L):
        B = ctx.shape[0]
        R = self.init_slots.expand(B, -1, -1)
        e = torch.ones(B, 1, device=ctx.device)
        logits_l, stop_l, e_l = [], [], []
        for k in range(L):
            R, qk, logits = self._step(R, ctx)
            delta = torch.sigmoid(self.energy_mlp(qk))
            e_next = e * (1 - delta)
            stop_in = torch.cat([qk, e_next], dim=-1) if self.use_gauge else qk
            stop_l.append(self.stop_head(stop_in))
            logits_l.append(logits)
            e_l.append(e_next)
            e = e_next
        return (torch.stack(logits_l, 1), torch.stack(stop_l, 1),
                torch.stack(e_l, 1).squeeze(-1))            # [B,L]

    @torch.no_grad()
    def generate(self, ctx, K_max=K_MAX, e_thresh=E_THRESH):
        B = ctx.shape[0]
        R = self.init_slots.expand(B, -1, -1)
        e = torch.ones(B, 1, device=ctx.device)
        toks, stops, sps, ehs = [], [], [], []
        done = torch.zeros(B, dtype=torch.bool, device=ctx.device)
        for _ in range(K_max):
            R, qk, logits = self._step(R, ctx)
            delta = torch.sigmoid(self.energy_mlp(qk))
            e_next = e * (1 - delta)
            stop_in = torch.cat([qk, e_next], dim=-1) if self.use_gauge else qk
            stop_prob = torch.sigmoid(self.stop_head(stop_in))
            stop = stop_prob.squeeze(-1) > 0.5
            energy_fired = torch.zeros(B, dtype=torch.bool, device=ctx.device)
            if self.use_gauge and e_thresh is not None:
                energy_fired = e_next.squeeze(-1) < e_thresh
                stop = stop | energy_fired
            toks.append(logits.argmax(-1))
            stops.append(stop)
            sps.append(stop_prob.squeeze(-1))
            ehs.append(e_next.squeeze(-1))
            e = e_next
            done = done | stop
            if bool(done.all().cpu()):
                break
        return (torch.stack(toks, 1), torch.stack(stops, 1),
                torch.stack(sps, 1), torch.stack(ehs, 1))   # [B,K] each


class ARVarDecoder(nn.Module):
    """Autoregressive baseline with EOS token."""

    def __init__(self, enc, vocab):
        super().__init__()
        self.enc = enc
        self.vocab = vocab
        self.eos = vocab
        self.embed = nn.Embedding(vocab + 1, D)
        layer = nn.TransformerDecoderLayer(D, HEADS, dim_feedforward=4 * D,
                                           dropout=0.1, batch_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(layer, ENC_LAYERS)
        self.head = nn.Linear(D, vocab + 1)

    def forward_train(self, x, y):
        ctx = self.enc(x)
        B, L = x.shape[0], y.shape[1]
        go = torch.full((B, 1), self.vocab, dtype=torch.long, device=x.device)
        tgt = torch.cat([go, y], dim=1)                      # [B, L+1]
        tgt_emb = self.embed(tgt) + sinusoid_pos(L + 1, D, x.device)
        mask = nn.Transformer.generate_square_subsequent_mask(L + 1, device=x.device)
        logits = self.head(self.dec(tgt_emb, ctx, tgt_mask=mask))  # [B, L+1, V+1]
        yt = torch.cat([y, torch.full((B, 1), self.eos, dtype=torch.long, device=x.device)], dim=1)
        return logits, yt

    @torch.no_grad()
    def generate(self, x, K_max=K_MAX):
        ctx = self.enc(x)
        B = x.shape[0]
        toks = torch.full((B, 1), self.vocab, dtype=torch.long, device=x.device)
        out = []
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        for _ in range(K_max):
            tgt_emb = self.embed(toks) + sinusoid_pos(toks.shape[1], D, x.device)
            mask = nn.Transformer.generate_square_subsequent_mask(toks.shape[1], device=x.device)
            logits = self.head(self.dec(tgt_emb, ctx, tgt_mask=mask))
            nxt = logits[:, -1].argmax(-1)
            out.append(nxt)
            done = done | (nxt == self.eos)
            if bool(done.all().cpu()):
                break
            toks = torch.cat([toks, nxt.unsqueeze(1)], dim=1)
        return torch.stack(out, 1)                            # [B,K] (EOS included)


def train(model, enc, mode, steps, bs, rng, seed):
    model.to(device); enc.to(device)
    params = list(dict.fromkeys(list(model.parameters()) + list(enc.parameters())))
    opt = torch.optim.AdamW(params, lr=1.2e-3, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, max(1, int(steps * 0.1)), steps)
    t0 = time.perf_counter()
    for s in range(steps):
        L = int(rng.integers(L_TRAIN[0], L_TRAIN[1] + 1))
        x, y = gen_seq_batch(mode, bs, L, rng, VOCAB)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        ctx = enc(x)
        if isinstance(model, RPTVarDecoder):
            logits, stop_logits, e = model.forward_train(ctx, L)
            loss_tok = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            stop_t = torch.zeros(bs, L, device=device)
            stop_t[:, -1] = 1.0
            loss_s = F.binary_cross_entropy_with_logits(stop_logits.squeeze(-1), stop_t)
            e_t = (L - torch.arange(1, L + 1, device=device).float()) / L   # (L-k-1)/L
            loss_e = F.mse_loss(e, e_t.unsqueeze(0).expand(bs, -1)) if model.use_gauge else torch.tensor(0.0)
            loss = loss_tok + LAM_S * loss_s + LAM_E * loss_e
        else:
            logits, yt = model.forward_train(x, y)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB + 1), yt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
    model.eval(); enc.eval()
    return time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, enc, mode, lengths, bs, rng):
    out = {}
    for L in lengths:
        x, y = gen_seq_batch(mode, bs, L, rng, VOCAB)
        x, y = x.to(device), y.to(device)
        ctx = enc(x)
        if isinstance(model, RPTVarDecoder):
            toks, stops, sps, ehs = model.generate(ctx)
            K = toks.shape[1]
            dec_len = stops.float().argmax(-1) + 1            # first stop position
            # pad pred to L
            if K >= L:
                pred = toks[:, :L]
            else:
                pred = F.pad(toks, (0, L - K), value=-1)
            exact = float(((pred == y).all(-1) & (dec_len == L)).sum().item()) / bs
            tok_acc = float((pred == y).float().mean().item())
            stop_acc = float((dec_len == L).float().mean().item())
            out[L] = {"exact": exact, "token": tok_acc, "stop": stop_acc}
            # diagnostics at first-stop step
            idx = stops.float().argmax(-1)
            e_at = ehs.gather(1, idx.unsqueeze(1)).squeeze(1)
            sp_at = sps.gather(1, idx.unsqueeze(1)).squeeze(1)
            en_fired = float((e_at < E_THRESH).float().mean().item()) if model.use_gauge else None
            hd_fired = float((sp_at > 0.5).float().mean().item())
            len_err = float((dec_len - L).float().abs().mean().item())
            out[L].update({"en_fired": en_fired, "hd_fired": hd_fired, "len_err": len_err})
        else:
            toks = model.generate(x)
            K = toks.shape[1]
            eq = toks == model.eos
            dec_len = torch.where(eq.any(-1), eq.float().argmax(-1) + 1, torch.full((bs,), K, device=device))
            # strip EOS token from prediction
            preds = []
            for b in range(bs):
                tl = int(dec_len[b].item()) - 1
                preds.append(toks[b, :tl] if tl > 0 else torch.full((1,), -1, device=device, dtype=torch.long))
            pad = max(int(dec_len.max().item()) - 1, 1)
            pred = torch.stack([F.pad(p, (0, max(0, L - p.shape[0])), value=-1)[:L] for p in preds])
            exact = float(((pred == y).all(-1) & (dec_len == L + 1)).sum().item()) / bs
            tok_acc = float((pred == y).float().mean().item())
            stop_acc = float((dec_len == L + 1).float().mean().item())
            out[L] = {"exact": exact, "token": tok_acc, "stop": stop_acc}
    return out


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    _out = ROOT / "experiments_v3" / "results" / "rpt_v5_varlen.json"
    if _out.exists():
        report = json.loads(_out.read_text(encoding="utf-8"))
        print("loaded existing:", list(report.keys()), flush=True)
    else:
        report = {}
    for mode in (["copy", "reverse"] if only is None else [only]):
        tag = f"var-{mode}:L{L_TRAIN[0]}-{L_TRAIN[1]}"
        print(f"\n===== {tag} =====", flush=True)
        if tag in report:
            print("  (skip, already in results)", flush=True)
            continue
        for name, use_g in [("RPT-var-full", True), ("RPT-var-nogauge", False)]:
            set_seed(42); rng = np.random.default_rng(42)
            torch.cuda.reset_peak_memory_stats()
            enc = SharedEncoder(VOCAB)
            dec = RPTVarDecoder(VOCAB, use_gauge=use_g)
            dt = train(dec, enc, mode, 3000, 128, rng, 42)
            res = evaluate(dec, enc, mode, [8, 12, 16, 24, 32], 512, rng)
            report.setdefault(tag, {})[name] = {"by_len": res, "seconds": dt,
                "params": sum(p.numel() for p in list(dec.parameters()) + list(enc.parameters())),
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            def _en(v_):
                e_ = v_.get('en_fired')
                return -1.0 if e_ is None else e_
            print(f"  {name}: " + " ".join(
                f"L{l}=x{v['exact']*100:.0f}/t{v['token']*100:.0f}/s{v['stop']*100:.0f}"
                + (f"[en{_en(v)*100:.0f}/hd{v.get('hd_fired', -1)*100:.0f}/le{v.get('len_err', -1):.1f}]"
                   if 'en_fired' in v else "")
                for l, v in res.items()), flush=True)
            del enc, dec; torch.cuda.empty_cache()
            with open(_out, "w") as _f:
                json.dump(report, _f, indent=2)
        set_seed(42); rng = np.random.default_rng(42)
        enc = SharedEncoder(VOCAB)
        ar = ARVarDecoder(enc, VOCAB)
        dt = train(ar, enc, mode, 3000, 128, rng, 42)
        res = evaluate(ar, enc, mode, [8, 12, 16, 24, 32], 512, rng)
        report.setdefault(tag, {})["AR-EOS"] = {"by_len": res, "seconds": dt,
            "params": sum(p.numel() for p in ar.parameters()),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(f"  AR-EOS: " + " ".join(
            f"L{l}=x{v['exact']*100:.0f}/t{v['token']*100:.0f}/s{v['stop']*100:.0f}"
            for l, v in res.items()), flush=True)
        del ar, enc; torch.cuda.empty_cache()
        with open(_out, "w") as _f:
            json.dump(report, _f, indent=2)

    out = _out
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
