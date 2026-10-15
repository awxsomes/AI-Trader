"""Signal-extraction and allocation models from Guijarro-Ordonez, Pelger &
Zanotti (2022), Section II.D.

All models take as input the window of the last L=30 CUMULATIVE residuals
x = Int(eps) (the residual "price" path) and output an allocation weight
for each residual portfolio. Portfolio weights are normalized each day so
that sum(|w|) = 1 (the paper's leverage constraint).

Models:
  1. OUThreshold    - parametric benchmark (Avellaneda-Lee / Yeo-Papanicolaou
                      style): estimate an Ornstein-Uhlenbeck process on the
                      window, trade on the standardized deviation from the
                      long-run mean with an R^2 reliability cutoff.
  2. FourierFFN     - pre-specified frequency filter: the signal is the real
                      FFT representation of the window, mapped to a weight by
                      a feedforward network (trained on the trading objective).
  3. CNNTransformer - the paper's main model: 2-layer CNN with D=8 local
                      filters of size 2 (instance norm + residual connection)
                      feeding a 1-layer transformer encoder with H=4 attention
                      heads; the representation at the last time step is
                      mapped to the allocation weight by a feedforward network.
"""
import numpy as np
import torch
import torch.nn as nn

L_WINDOW = 30


# ---------------------------------------------------------------------------
# 1. Parametric benchmark: OU + threshold rule
# ---------------------------------------------------------------------------

def ou_threshold_weights(X: np.ndarray, c_thresh: float = 1.25,
                         r2_crit: float = 0.25) -> np.ndarray:
    """Vectorized OU+Threshold allocation.

    X : (n_samples, L) array of cumulative-residual windows.
    Returns raw weights in {-1, 0, +1} per sample (normalization happens at
    the portfolio level).

    The OU process dX = kappa (mu - X) dt + sigma dB is estimated from the
    AR(1) regression X_{l+1} = a + b X_l + xi (as in Avellaneda-Lee). The
    signal is s = (X_L - mu) / sigma_eq with sigma_eq the stationary std.
    Trade only if the fit is reliable (R^2 > r2_crit) and mean-reverting
    (0 < b < 1).
    """
    Xp, Xn = X[:, :-1], X[:, 1:]
    mx, my = Xp.mean(1), Xn.mean(1)
    sxx = ((Xp - mx[:, None]) ** 2).sum(1)
    sxy = ((Xp - mx[:, None]) * (Xn - my[:, None])).sum(1)
    b = np.divide(sxy, sxx, out=np.zeros_like(sxy), where=sxx > 1e-18)
    a = my - b * mx
    resid = Xn - (a[:, None] + b[:, None] * Xp)
    syy = ((Xn - my[:, None]) ** 2).sum(1)
    r2 = 1.0 - np.divide((resid ** 2).sum(1), syy,
                         out=np.ones_like(syy), where=syy > 1e-18)
    var_xi = (resid ** 2).sum(1) / max(X.shape[1] - 3, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_eq = np.sqrt(var_xi / np.clip(1.0 - b ** 2, 1e-12, None))
        mu = a / np.clip(1.0 - b, 1e-12, None)
    s = (X[:, -1] - mu) / np.clip(sigma_eq, 1e-12, None)
    ok = (b > 0) & (b < 1) & (r2 > r2_crit)
    w = np.zeros(len(X))
    w[ok & (s > c_thresh)] = -1.0
    w[ok & (s < -c_thresh)] = 1.0
    return w


# ---------------------------------------------------------------------------
# Shared FFN allocation head
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Fourier + FFN
# ---------------------------------------------------------------------------

class FourierFFN(nn.Module):
    """Signal = real FFT coefficients of the (per-window standardized)
    cumulative-residual path; allocation = FFN on those coefficients."""

    def __init__(self, L: int = L_WINDOW, d_hidden: int = 16):
        super().__init__()
        self.L = L
        self.head = FFN(2 * (L // 2 + 1), d_hidden)

    def forward(self, x):                      # x: (B, L)
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-8)
        z = (x - mu) / sd
        f = torch.fft.rfft(z, dim=1)
        feats = torch.cat([f.real, f.imag], dim=1)
        return self.head(feats)


# ---------------------------------------------------------------------------
# 3. CNN + Transformer (the paper's main model)
# ---------------------------------------------------------------------------

class CNNTransformer(nn.Module):
    def __init__(self, L: int = L_WINDOW, d: int = 8, d_size: int = 2,
                 n_heads: int = 4, d_hidden: int = 16):
        super().__init__()
        self.L, self.d = L, d
        # causal convolutions: pad on the left so block l only sees <= l
        self.conv1 = nn.Conv1d(1, d, kernel_size=d_size)
        self.conv2 = nn.Conv1d(d, d, kernel_size=d_size)
        self.norm_in = nn.InstanceNorm1d(1, affine=True)
        self.proj_res = nn.Conv1d(1, d, kernel_size=1)   # residual connection
        self.pad = d_size - 1
        self.pos = nn.Parameter(0.01 * torch.randn(1, L, d))
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=1,
                                                 enable_nested_tensor=False)
        self.head = FFN(d, d_hidden)

    def forward(self, x):                      # x: (B, L)
        z = x.unsqueeze(1)                     # (B, 1, L)
        z = self.norm_in(z)                    # instance normalization
        h = torch.relu(self.conv1(nn.functional.pad(z, (self.pad, 0))))
        h = torch.relu(self.conv2(nn.functional.pad(h, (self.pad, 0)))
                       + self.proj_res(z))     # componentwise residual addition
        h = h.transpose(1, 2)                  # (B, L, d)
        h = self.transformer(h + self.pos)
        sig = h[:, -1, :]                      # representation at last step
        return self.head(sig)
