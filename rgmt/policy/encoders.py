import torch
from torch import nn, Tensor
from rgmt.utils.pos_encoding import sinusoidal_pe


class _MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


class HistoryEncoder(nn.Module):
    def __init__(self, in_dim: int, n_embd: int = 128, n_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.embed = _MLP(in_dim, n_embd, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = _MLP(n_embd, n_embd * mlp_ratio, n_embd)
        self.ln_out = nn.LayerNorm(n_embd)

    def token_features(self, o_hist: Tensor) -> Tensor:
        B, T, _ = o_hist.shape
        x = self.embed(o_hist) + sinusoidal_pe(T, x_dim := self.ln1.normalized_shape[0],
                                               device=o_hist.device, dtype=o_hist.dtype)
        mask = torch.triu(torch.ones(T, T, device=o_hist.device, dtype=torch.bool), diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return self.ln_out(x)

    def forward(self, o_hist: Tensor) -> Tensor:
        feats = self.token_features(o_hist)        # (B,T,n_embd)
        return feats.max(dim=1).values             # element-wise max-pool over time


class CommandEncoder(nn.Module):
    def __init__(self, cmd_dim: int, n_embd: int = 128, n_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.q_proj = _MLP(n_embd, n_embd, n_embd)        # MLP_dyn(h) -> query
        self.tok = _MLP(cmd_dim, n_embd, n_embd)          # MLP_cmd(g) -> tokens
        self.ln_q = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = _MLP(n_embd, n_embd * mlp_ratio, n_embd)
        self.ln_out = nn.LayerNorm(n_embd)
        self.n_embd = n_embd

    def forward(self, h: Tensor, cmd_window: Tensor) -> Tensor:
        B, S, _ = cmd_window.shape
        q = self.q_proj(h).unsqueeze(1)                                   # (B,1,n_embd)
        z = self.tok(cmd_window) + sinusoidal_pe(S, self.n_embd,
                                                 device=h.device, dtype=h.dtype)
        a, _ = self.attn(self.ln_q(q), z, z, need_weights=False)          # cross-attn
        s = q + a
        s = s + self.mlp(self.ln2(s))
        return self.ln_out(s).squeeze(1)                                  # (B,n_embd)
