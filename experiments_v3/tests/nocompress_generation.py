# -*- coding: utf-8 -*-
"""
experiments_v3/tests/nocompress_generation.py
==============================================
NoCompressFluidNet 生成能力测试

任务: 字符级序列生成
  语料: 确定性模式 (如 "hello world" "1234567890" "lorem ipsum" ...)
  输入: 前T个字符 → 预测下一个字符 (自回归生成)

模型对比:
  1. NC-Gen: 字符嵌入 → 场空间排列 → 同维度ADR演化 → 逐位置读出下一字符
  2. LSTM:  标准RNN生成器
  3. MLP:   无时序结构生成器

评估:
  - 教师强制 next-char 准确率
  - 自回归生成: 给定种子生成30字符, 测模式保持率
"""
from __future__ import annotations

import sys, time, json, math, random
from pathlib import Path
from itertools import cycle
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(str(__import__('pathlib').Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_v3" / "models"))

import fluid_net as fn
from improved_fluid import set_seed, FourierGlobalMix, CosineWarmupScheduler

device = torch.device("cuda:0")
torch.set_float32_matmul_precision("high")


def make_block(n, h, adr_hidden=32, step_size=0.30):
    cfg = fn.FluidResidualSequenceConfig(
        n=n, h=h, output_dim=h, max_output_steps=1, static_adr_steps=1,
        adr_hidden_channels=adr_hidden, adr_step_size=step_size,
        q_channels=8, q_depth=1, q_height=1, q_width=1,
        extractor_hidden_channels=8, reconstructor_hidden_channels=8, output_hidden_dim=64)
    return fn.ADRBlock3D(cfg)


# ==================== 数据: 字符级模式生成 ====================
PATTERNS = [
    "hello world, this is a pattern. ",
    "1234567890 abcdefghij. ",
    "the quick brown fox jumps. ",
    "lorem ipsum dolor sit amet. ",
    "0101010101010101. ",
    "ab cd ef gh ij kl mn op. ",
    "flow of information through fields. ",
    "0123456789 9876543210. ",
]

def build_char_dataset(seq_len=24, n_samples=4000, seed=42):
    rng = np.random.default_rng(seed)
    vocab = sorted(set("".join(PATTERNS)))
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    vocab_size = len(vocab)

    # 生成长文本
    full_text = "".join(PATTERNS) * 200

    xs, ys = [], []
    for _ in range(n_samples):
        start = rng.integers(0, len(full_text) - seq_len - 1)
        window = full_text[start:start + seq_len + 1]
        xs.append([char2idx[c] for c in window[:-1]])
        ys.append(char2idx[window[-1]])

    X = np.array(xs, dtype=np.int64)
    y = np.array(ys, dtype=np.int64)

    split = int(n_samples * 0.8)
    tr_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    vl_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    return tr_ds, vl_ds, vocab_size, char2idx, idx2char, vocab


# ==================== NC 生成器 ====================
class NCCharGen(nn.Module):
    """
    序列 → 场空间排列 → 同维度ADR演化 → 逐位置读出

    核心思想: 把时间步映射到场空间位置 (4x4=16 ≥ seq_len 用滑动窗口)
    ADR演化让邻近时间步互相传播信息 (物理时序建模)
    """
    def __init__(self, vocab_size, n=6, h=8, num_layers=4, seq_len=24,
                 pool_size=6, feat_dim=96, embed_dim=8, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n = n
        self.h = h
        self.seq_len = seq_len
        assert n * n >= seq_len, "场空间必须容纳序列长度"

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n, n, h) * 0.02)

        # 投影: 嵌入 → 场通道
        self.proj = nn.Linear(embed_dim, h)

        # 场初始化(填充未用位置)
        self.bias = nn.Parameter(torch.zeros(1, n, n, h))

        # Fourier + 同维度ADR堆叠
        self.fourier = FourierGlobalMix(n, h)
        self.fourier_norm = nn.LayerNorm([n, n, h])
        self.adr_layers = nn.ModuleList([make_block(n, h, 32, 0.30) for _ in range(num_layers)])
        self.adr_norms = nn.ModuleList([nn.LayerNorm([n, n, h]) for _ in range(num_layers)])

        # 逐位置读出 → 下一字符分布
        self.head = nn.Sequential(
            nn.LayerNorm([n, n, h]),
            nn.Linear(h, feat_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feat_dim, vocab_size),
        )

    def forward(self, tokens, mode="nonsequential", max_output_steps=1):
        """tokens: [B, seq_len] → logits: [B, seq_len, vocab] (每个位置预测下一字符)"""
        B, T = tokens.shape
        device = tokens.device

        # 嵌入 → 场 [B, n, n, h]
        emb = self.embed(tokens)                 # [B, T, embed_dim]
        emb = self.proj(emb)                     # [B, T, h]

        field = self.bias.expand(B, -1, -1, -1).clone()  # [B, n, n, h]
        # 将序列位置写入场 (逐行填充)
        for t in range(T):
            row, col = t // self.n, t % self.n
            field[:, row, col, :] = emb[:, t, :] + self.pos_embed[:, row, col, :]

        # Fourier + ADR演化
        f = field + self.fourier_norm(self.fourier(field))
        f5 = f.unsqueeze(1)
        for li, (adr, norm) in enumerate(zip(self.adr_layers, self.adr_norms)):
            step_idx = torch.full((f5.shape[0], 1), li, dtype=torch.long, device=f5.device)
            f5 = norm((f5 + adr(f5, step_idx)).squeeze(1)).unsqueeze(1)
        f_out = f5.squeeze(1)  # [B, n, n, h]

        # 逐位置读出: 位置t的读出预测t+1的字符
        logits = self.head(f_out)  # [B, n, n, vocab]
        logits = logits.reshape(B, self.n * self.n, self.vocab_size)
        return logits[:, :T]         # [B, T, vocab]


class LSTMCharGen(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden=64, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)
        self.seq_len = 24
    def forward(self, tokens, **kwargs):
        emb = self.embed(tokens)
        out, _ = self.lstm(emb)
        return self.head(out)


class MLPCharGen(nn.Module):
    def __init__(self, vocab_size, seq_len=24, embed_dim=16, hidden=128):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, vocab_size))
    def forward(self, tokens, **kwargs):
        return self.net(self.embed(tokens))


# ==================== 训练 ====================
def train_gen(model, tr_ldr, vl_ldr, device, steps=600, lr=1.2e-3, name=""):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, int(steps * 0.1))
    sched = CosineWarmupScheduler(opt, warmup, steps)
    tr_iter = cycle(tr_ldr)
    t0 = time.perf_counter()

    for _ in range(steps):
        model.train()
        x, y = next(tr_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, mode="nonsequential", max_output_steps=1)  # [B,T,vocab]
        if logits.ndim == 3:
            logits = logits[:, -1]          # ?????? ? [B, vocab]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    sec = time.perf_counter() - t0

    # 教师强制准确率
    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for x, y in vl_ldr:
            x, y = x.to(device), y.to(device)
            logits = model(x, mode="nonsequential", max_output_steps=1)
            if logits.ndim == 3:
                logits = logits[:, -1]
            pred = logits.argmax(-1)
            vc += int((pred == y).sum().cpu()); vt += y.numel()

    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": params, "val_acc": vc / vt, "seconds": sec}


# ==================== 自回归生成 ====================
def autoreg_generate(model, seed_tokens, n_gen=40, vocab_size=None, char2idx=None, idx2char=None, greedy=False):
    """从种子token自回归生成n_gen个字符, 返回字符串; greedy=True 用argmax"""
    model.eval()
    device = next(model.parameters()).device
    tokens = seed_tokens.copy()
    generated = []

    with torch.no_grad():
        for _ in range(n_gen):
            x = torch.tensor([tokens[-model.seq_len:]], dtype=torch.long, device=device)
            logits = model(x, mode="nonsequential", max_output_steps=1)  # [1,T,vocab]
            if logits.ndim == 3:
                probs = F.softmax(logits[0, -1], dim=-1)
            else:
                probs = F.softmax(logits[0], dim=-1)
            if greedy:
                next_idx = int(probs.argmax().item())
            else:
                next_idx = int(torch.multinomial(probs, 1).item())
            tokens.append(next_idx)
            generated.append(idx2char[next_idx])

    return "".join(generated)


# ==================== Main ====================
if __name__ == "__main__":
    set_seed(42)
    print("=" * 65)
    print("  NO-COMPRESS: 生成任务测试 (字符级序列生成)")
    print("=" * 65)

    tr_ds, vl_ds, vocab_size, char2idx, idx2char, vocab = build_char_dataset(seq_len=24)
    tr_ldr = DataLoader(tr_ds, 64, shuffle=True, drop_last=True)
    vl_ldr = DataLoader(vl_ds, 64)
    print(f"  Vocab: {vocab_size} chars | Train: {len(tr_ds)} | Val: {len(vl_ds)}")
    print(f"  Patterns: {PATTERNS[0][:30]}...")

    results = []

    # LSTM
    set_seed(42)
    lstm_model = LSTMCharGen(vocab_size)
    r = train_gen(lstm_model, tr_ldr, vl_ldr, device, steps=600, name="LSTM")
    results.append(r)
    print(f"  LSTM:     next-char acc={r['val_acc']:.4f} P={r['params']:,}")

    # MLP
    set_seed(42)
    mlp_model = MLPCharGen(vocab_size, seq_len=24)
    r = train_gen(mlp_model, tr_ldr, vl_ldr, device, steps=600, name="MLP")
    results.append(r)
    print(f"  MLP:      next-char acc={r['val_acc']:.4f} P={r['params']:,}")

    # NC-Gen L4
    set_seed(42)
    m = NCCharGen(vocab_size, n=6, h=8, num_layers=4, seq_len=24)
    r = train_gen(m, tr_ldr, vl_ldr, device, steps=600, name="NC-Gen(L4)")
    results.append(r)
    print(f"  NC-GenL4: next-char acc={r['val_acc']:.4f} P={r['params']:,}")

    # NC-Gen L8 (更深)
    set_seed(42)
    m8 = NCCharGen(vocab_size, n=6, h=8, num_layers=8, seq_len=24)
    r = train_gen(m8, tr_ldr, vl_ldr, device, steps=600, name="NC-Gen(L8)")
    results.append(r)
    print(f"  NC-GenL8: next-char acc={r['val_acc']:.4f} P={r['params']:,}")

        # ==================== Autoregressive Generation ====================
    print("\n" + "=" * 65)
    print("  Autoregressive comparison (seed='hello wor' -> generate 40 chars, sampling)")
    print("=" * 65)
    seed_str = "hello wor"
    seed_tokens = [char2idx[c] for c in seed_str]
    # pad prefix to seq_len using pattern head
    full_prefix = PATTERNS[0]
    pad = [char2idx[c] for c in full_prefix[:24 - len(seed_tokens)]]
    seed_padded = pad + seed_tokens
    assert len(seed_padded) == 24

    models = {
        "LSTM":       lstm_model,
        "MLP":        mlp_model,
        "NC-Gen(L4)": m,
        "NC-Gen(L8)": m8,
    }
    print(f"  Seed: '{seed_str}'  (prefix pad {len(pad)} chars)")
    print(f"  {'Model':<14s} {'sampled':<42s} {'greedy-match':>12s}")
    # ideal continuation: PATTERNS[0] repeated from offset len(seed_str)
    ideal = (PATTERNS[0] * 10)[len(seed_str):len(seed_str) + 40]
    for name, model in models.items():
        out_s = autoreg_generate(model, seed_padded, n_gen=40, idx2char=idx2char)
        out_g = autoreg_generate(model, seed_padded, n_gen=40, idx2char=idx2char, greedy=True)
        adh = sum(1 for a, b in zip(out_g, ideal) if a == b) / len(ideal)
        print(f"  {name:<14s} {out_s:<42s} {adh:>11.2%}")
    print(f"  ideal: {seed_str}|{ideal}")
    print()

# ==================== Summary ====================
    print("=" * 65)
    print(f"{'Model':<14s} {'Acc':>8s} {'Params':>10s} {'Time':>6s}")
    print("-" * 44)
    for r in sorted(results, key=lambda x: -x["val_acc"]):
        print(f"{r['model']:<14s} {r['val_acc']:>7.4f} {r['params']:>10,} {r['seconds']:>5.0f}s")

    with open(ROOT / "experiments_v3" / "results" / "nocompress_generation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved!")
