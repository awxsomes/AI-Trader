"""CNN+Transformer exactly as in Appendix C of Guijarro-Ordonez, Pelger &
Zanotti (2022), with the market-friction extension of Section III.J.

Appendix C.1  two causal conv layers (D=8 filters of size 2) each followed by
              instance normalization over the window and ReLU, plus a
              residual connection: x~ = x(2) + x(0)
Appendix C.2  one transformer encoder layer (H=4 heads, hidden units HDN=2D,
              dropout 0.25, add & normalize after attention and feedforward)
Allocation    a linear layer on the last time step; with trading frictions
              the previous allocation of that residual is an extra input
              (Appendix D), so the policy can learn to avoid turnover.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

L_WINDOW = 30


class LastStepEncoderLayer(nn.Module):
    """Post-norm transformer encoder layer (Vaswani et al. 2017; same
    parameters as nn.TransformerEncoderLayer) that returns only the output
    at the last time step. With one layer, the allocation only uses that
    output, so this is exact and ~30x cheaper than computing all steps."""

    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.h, self.p = heads, dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.f1, self.f2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x):                        # x: (B, L, d) -> (B, d)
        B, L, d = x.shape
        h, dh = self.h, d // self.h
        last = x[:, -1:, :]
        q = self.q(last).view(B, 1, h, dh).transpose(1, 2)        # (B,h,1,dh)
        k, v = self.kv(x).view(B, L, 2, h, dh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0)
        a = self.o(a.transpose(1, 2).reshape(B, 1, d))
        y = self.n1(last + F.dropout(a, self.p, self.training))
        ff = self.f2(F.dropout(F.relu(self.f1(y)), self.p, self.training))
        return self.n2(y + F.dropout(ff, self.p, self.training))[:, 0, :]


class CNNTransformerPaper(nn.Module):
    def __init__(self, L=L_WINDOW, D=8, d_size=2, heads=4, dropout=0.25,
                 use_prev=False):
        super().__init__()
        self.pad = d_size - 1
        self.conv1 = nn.Conv1d(1, D, d_size)
        self.conv2 = nn.Conv1d(D, D, d_size)
        self.pos = nn.Parameter(0.01 * torch.randn(1, L, D))   # positional encoding
        self.transformer = LastStepEncoderLayer(D, heads, 2 * D, dropout)
        self.use_prev = use_prev
        self.out = nn.Linear(D + (1 if use_prev else 0), 1)

    def forward(self, x, prev=None):            # x: (B, L) cumulative residuals
        x0 = x.unsqueeze(1)                      # (B, 1, L)
        y = self.conv1(F.pad(x0, (self.pad, 0)))
        x1 = F.relu(F.instance_norm(y))
        y = self.conv2(F.pad(x1, (self.pad, 0)))
        x2 = F.relu(F.instance_norm(y))
        xt = (x2 + x0).transpose(1, 2)           # (B, L, D), residual connection
        s = self.transformer(xt + self.pos)      # last-step output (B, D)
        if self.use_prev:
            s = torch.cat([s, prev.unsqueeze(1)], dim=1)
        return self.out(s).squeeze(-1)
