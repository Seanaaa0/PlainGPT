import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

# ---- 逐位置前饋 FFN ----


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # (B,T,d)
        return self.net(x)

# ---- 正弦餘弦 Positional Encoding ----


def build_sincos_posenc(max_len, d_model, device=None):
    pos = torch.arange(max_len, device=device,
                       dtype=torch.float32).unsqueeze(1)         # (T,1)
    i = torch.arange(0, d_model, 2, device=device,
                     dtype=torch.float32).unsqueeze(0)     # (1,d/2)
    angle = pos / (10000 ** (i / d_model))
    pe = torch.zeros(max_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe  # (T,d)

# ---- Pre-LN Transformer Block (Decoder) ----


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(
            d_model, n_head, attn_dropout=dropout, proj_dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x):  # (B,T,d)
        y, _ = self.mha(self.ln1(x), is_causal=True)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x

# ---- 最小 Decoder-only 語言模型 ----


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, inner_mult=4, dropout=0.1):
        super().__init__()
        inner = int((2/3) * inner_mult * d_model)  # ~2/3*4d
        self.w1 = nn.Linear(d_model, inner, bias=False)
        self.w3 = nn.Linear(d_model, inner, bias=False)
        self.w2 = nn.Linear(inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(self.w1(x) * F.silu(self.w3(x))))


class ParallelBlock(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head=None, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head, n_kv_head,
                                      attn_dropout=dropout, proj_dropout=dropout)
        self.ffn = SwiGLUFFN(d_model, inner_mult=4, dropout=dropout)

    def forward(self, x):
        a, _ = self.mha(self.n1(x), is_causal=True)  # 自回歸：必須 causal
        f = self.ffn(self.n2(x))
        return x + a + f

# 保留你的 sin/cos 位置編碼工具 build_sincos_posenc(...) 不變


class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layer=4,
                 max_seq_len=256, dropout=0.1, n_kv_head=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # 你目前用 sin/cos buffer，pos_emb 就留著不使用也沒關係
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ParallelBlock(d_model, n_head,
                          n_kv_head=n_kv_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # sin/cos position（固定）
        pe = build_sincos_posenc(max_seq_len, d_model)
        self.register_buffer("pos_enc", pe, persistent=False)

        # 權重綁定
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, "sequence length > max_seq_len"
        x = self.tok_emb(idx) * math.sqrt(self.d_model)
        x = x + self.pos_enc[:T].unsqueeze(0)  # (1,T,d)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(
                B*T, -1), targets.reshape(B*T))
        return logits, loss
