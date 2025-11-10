import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 核心：Scaled Dot-Product Attention ----


# def scaled_dot_product_attention(Q, K, V, mask=None, is_causal=False, dropout_p=0.0, training=False):
#     """
#     Q: (B, H, Tq, d)
#     K: (B, H, Tk, d)
#     V: (B, H, Tk, dv)
#     mask:
#         - bool: True=允許看，False=遮蔽，可為 (B, 1, Tq, Tk) 或 (B, Tq, Tk) 或 (Tq, Tk)
#         - float: additive mask，同形狀；遮蔽處請給 -inf（或很大負數）
#         - None: 不使用額外 mask
#     is_causal: 若 True，會套用下三角因果遮罩（禁止看未來）
#     """
#     d = Q.size(-1)
#     # (B,H,Tq,Tk)
#     scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

#     # 數值穩定：softmax 前先減去每列最大值
#     scores = scores - scores.amax(dim=-1, keepdim=True)

#     # 1) causal 下三角遮罩
#     if is_causal:
#         Tq, Tk = scores.size(-2), scores.size(-1)
#         causal = torch.ones(Tq, Tk, dtype=torch.bool,
#                             device=scores.device).tril()
#         # broadcast 為 (1,1,Tq,Tk)
#         scores = scores.masked_fill(~causal, float('-inf'))

#     # 2) 額外 mask
#     if mask is not None:
#         if mask.dtype == torch.bool:
#             # True=keep, False=mask
#             # 將 mask 擴成 (B,1,Tq,Tk) 以匹配 scores
#             while mask.dim() < scores.dim():
#                 mask = mask.unsqueeze(0)
#             # 若是 (B,Tq,Tk) 變成 (B,1,Tq,Tk)
#             if mask.size(1) != 1 and mask.size(1) != scores.size(1):
#                 mask = mask.unsqueeze(1)
#             scores = scores.masked_fill(~mask, float('-inf'))
#         else:
#             # additive mask（同形狀），直接相加
#             scores = scores + mask

#     # 注意力分佈
#     A = F.softmax(scores, dim=-1)
#     if dropout_p and training:
#         A = F.dropout(A, p=dropout_p, training=True)

#     # 輸出
#     out = torch.matmul(A, V)  # (B,H,Tq,dv)
#     return out, A


# ---- Multi-Head Attention ----

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head=None,
                 bias=True, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必須能被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        assert n_head % self.n_kv_head == 0, "n_head 需為 n_kv_head 的整數倍"
        self.d_head = d_model // n_head

        # Q 產生 n_head*d_head；K/V 只產生 n_kv_head*d_head（GQA）
        self.q_proj = nn.Linear(d_model, n_head * self.d_head, bias=bias)
        self.k_proj = nn.Linear(
            d_model, self.n_kv_head * self.d_head, bias=bias)
        self.v_proj = nn.Linear(
            d_model, self.n_kv_head * self.d_head, bias=bias)
        self.out_proj = nn.Linear(n_head * self.d_head, d_model, bias=bias)

        self.attn_dropout = attn_dropout
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _split(self, x, heads):
        B, T, _ = x.shape
        # (B, H, T, Dh)
        return x.view(B, T, heads, self.d_head).transpose(1, 2)

    def _merge(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, x_q, x_kv=None, mask=None, is_causal=False):
        if x_kv is None:
            x_kv = x_q

        # (B, H, Tq, Dh)
        q = self._split(self.q_proj(x_q), self.n_head)
        # (B, Hkv, Tk, Dh)
        k = self._split(self.k_proj(x_kv), self.n_kv_head)
        # (B, Hkv, Tk, Dh)
        v = self._split(self.v_proj(x_kv), self.n_kv_head)

        # GQA：把 K/V 複用到每組 Q 頭
        if self.n_kv_head != self.n_head:
            g = self.n_head // self.n_kv_head
            k = k.repeat_interleave(g, dim=1)  # (B, H, Tk, Dh)
            v = v.repeat_interleave(g, dim=1)

        # PyTorch SDPA（自動選 flash/mem-efficient），帶因果遮罩與 dropout
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if mask is None else mask,  # 如需額外 mask 可往這裡丟
            dropout_p=(self.attn_dropout if self.training else 0.0),
            is_causal=is_causal,
        )
        out = self._merge(out)                           # (B, Tq, d_model)
        out = self.proj_dropout(self.out_proj(out))
        return out, None
