"""Conditional latent factors (IPCA; Kelly, Pruitt & Su 2019) for the full
version of Guijarro-Ordonez, Pelger & Zanotti (2022).

Loadings are linear in lagged firm characteristics, beta_{i,m} = z_{i,m}' Gamma,
and Gamma is estimated by alternating least squares on monthly returns.

Characteristics. The paper uses 46 characteristics from CRSP/Compustat. Only
daily OHLCV is available here, so we build the 13 of them that can be
computed from prices and volume (same definitions in spirit; the size and
turnover measures use dollar volume because shares outstanding are not
available):
  ST_Rev, r2_1, r12_2, r12_7, r36_13, LT_Rev   past-return characteristics
  Beta, IdioVol, Variance                       risk
  Rel2High                                      closeness to 52-week high
  LME_proxy (log dollar volume)                 size / liquidity proxy
  SUV (abnormal volume), Spread (Corwin-Schultz high-low estimator)
Each month the characteristics are cross-sectionally rank-transformed to
[-0.5, 0.5] (missing -> 0, the median), plus a constant, as in the paper.

Protocol (paper Section III.B): Gamma is re-estimated every year on the last
240 months; daily out-of-sample residuals use the characteristics of the
previous month end and the Gamma estimated with data before that year:
  f_t = (beta' beta)^-1 beta' R_t,  eps_t = R_t - beta f_t.

Writes output/full/ipca5.npz in the same format as pca5.npz.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common_data as cd                                         # noqa: E402

OUT = os.path.join(ROOT, "output", "full")
K = 5
WINDOW_MONTHS = 240
MIN_STOCKS = 50
CHAR_NAMES = ["ST_Rev", "r2_1", "r12_2", "r12_7", "r36_13", "LT_Rev", "Beta",
              "IdioVol", "Variance", "Rel2High", "LME_proxy", "SUV", "Spread"]


# ---------------------------------------------------------------------------
# Characteristics
# ---------------------------------------------------------------------------

def load_panels():
    tick = cd.list_tickers()
    fields = {f: cd.load_field_panel(f, tickers=tick)
              for f in ["Open", "High", "Low", "Close", "Volume"]}
    idx = fields["Close"].index
    return {f: p.reindex(idx) for f, p in fields.items()}


def daily_characteristics(P):
    C, H, L, V = P["Close"], P["High"], P["Low"], P["Volume"]
    R = C.pct_change(fill_method=None).clip(-0.5, 0.5)
    mkt = R.mean(axis=1)
    ch = {}
    ch["ST_Rev"] = C / C.shift(21) - 1
    ch["r2_1"] = C.shift(21) / C.shift(42) - 1
    ch["r12_2"] = C.shift(42) / C.shift(252) - 1
    ch["r12_7"] = C.shift(147) / C.shift(252) - 1
    ch["r36_13"] = C.shift(273) / C.shift(756) - 1
    ch["LT_Rev"] = C.shift(273) / C.shift(1260) - 1
    w = 252
    mR = R.rolling(w, min_periods=200).mean()
    mM = mkt.rolling(w, min_periods=200).mean()
    cov = R.mul(mkt, axis=0).rolling(w, min_periods=200).mean() - mR.mul(mM, axis=0)
    varM = (mkt ** 2).rolling(w, min_periods=200).mean() - mM ** 2
    varR = (R ** 2).rolling(w, min_periods=200).mean() - mR ** 2
    beta = cov.div(varM, axis=0)
    ch["Beta"] = beta
    ch["IdioVol"] = np.sqrt((varR - beta ** 2 * varM.values[:, None]).clip(lower=0))
    ch["Variance"] = R.rolling(63, min_periods=50).var()
    ch["Rel2High"] = C / C.rolling(252, min_periods=200).max()
    dv = (C * V).replace(0, np.nan)
    ch["LME_proxy"] = np.log(dv.rolling(252, min_periods=200).mean())
    ch["SUV"] = np.log(V.replace(0, np.nan).rolling(21, min_periods=15).mean()
                       / V.replace(0, np.nan).rolling(252, min_periods=200).mean())
    # Corwin & Schultz (2012) two-day high-low spread estimator, 21-day average
    hl = np.log(H / L.replace(0, np.nan)) ** 2
    b = hl + hl.shift(1)
    g = np.log(np.maximum(H, H.shift(1)) / np.minimum(L, L.shift(1))) ** 2
    k = 3 - 2 * np.sqrt(2)
    a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    S = (2 * (np.exp(a) - 1) / (1 + np.exp(a))).clip(lower=0)
    ch["Spread"] = S.rolling(21, min_periods=15).mean()
    return R, ch


def rank_transform(df_row):
    """Cross-sectional rank to [-0.5, 0.5]; NaN stays NaN."""
    rk = df_row.rank(axis=1)                       # 1..n per row
    n = df_row.notna().sum(axis=1).clip(lower=2)
    return (rk - 1).div(n - 1, axis=0) - 0.5


def monthly_data(P):
    R, ch = daily_characteristics(P)
    C = P["Close"]
    me = C.groupby(C.index.to_period("M")).apply(lambda x: x.index[-1]).values
    me = pd.DatetimeIndex(me)
    Cm = C.loc[me]
    ret_next = (Cm.shift(-1) / Cm - 1).clip(-0.9, 2.0)      # return over month m+1
    Z = np.stack([rank_transform(ch[n].loc[me]).to_numpy() for n in CHAR_NAMES],
                 axis=2)                                      # (M, N, L)
    has_price = Cm.notna().to_numpy()
    valid_z = has_price & (np.isnan(Z).sum(axis=2) <= len(CHAR_NAMES) // 2)
    Z = np.nan_to_num(Z)
    Z = np.concatenate([Z, np.ones(Z.shape[:2] + (1,))], axis=2)   # constant
    return R, me, Z, valid_z, ret_next.to_numpy()


# ---------------------------------------------------------------------------
# IPCA estimation (alternating least squares)
# ---------------------------------------------------------------------------

def ipca_als(Zs, rs, k=K, iters=100, tol=1e-7):
    """Zs: list of (n_t, L) characteristics, rs: list of (n_t,) next returns."""
    Lc = Zs[0].shape[1]
    X = np.stack([Z.T @ r / len(r) for Z, r in zip(Zs, rs)], axis=1)   # (L, T)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    G = U[:, :k]
    ZZ = [Z.T @ Z for Z in Zs]
    Zr = [Z.T @ r for Z, r in zip(Zs, rs)]
    for it in range(iters):
        F = [np.linalg.solve(G.T @ A @ G + 1e-10 * np.eye(k), G.T @ b)
             for A, b in zip(ZZ, Zr)]
        A = sum(np.kron(a, np.outer(f, f)) for a, f in zip(ZZ, F))
        bb = sum(np.kron(b, f) for b, f in zip(Zr, F))
        Gn = np.linalg.solve(A + 1e-10 * np.eye(Lc * k), bb).reshape(Lc, k)
        Gn, _ = np.linalg.qr(Gn)                              # identification
        if np.abs(Gn @ Gn.T - G @ G.T).max() < tol:
            G = Gn
            break
        G = Gn
    return G


def estimate_gammas(me, Z, valid_z, ret_next, years):
    gammas = {}
    for y in years:
        # months m with return over m+1 fully realized before Jan 1 of year y
        last = np.searchsorted(me, np.datetime64(f"{y - 1}-12-01")) - 1
        first = max(0, last - WINDOW_MONTHS + 1)
        Zs, rs = [], []
        for m in range(first, last + 1):
            ok = valid_z[m] & np.isfinite(ret_next[m])
            if ok.sum() >= MIN_STOCKS:
                Zs.append(Z[m, ok])
                rs.append(ret_next[m, ok])
        gammas[y] = ipca_als(Zs, rs)
        print(f"  Gamma for {y}: {len(Zs)} months", flush=True)
    return gammas


# ---------------------------------------------------------------------------
# Daily out-of-sample residuals
# ---------------------------------------------------------------------------

def daily_residuals(R, me, Z, valid_z, gammas, start="1998-01-01"):
    dates = R.index[R.index >= pd.Timestamp(start)]
    Rv = R.loc[dates].to_numpy(np.float64)
    T, N = Rv.shape
    eps = np.full((T, N), np.nan, dtype=np.float32)
    Wd = np.zeros((T, N, K), dtype=np.float32)
    Bd = np.zeros((T, N, K), dtype=np.float32)
    m_idx = np.searchsorted(me, dates.values, side="left") - 1   # last month end < t
    for t in range(T):
        m = m_idx[t]
        if m < 0:
            continue
        G = gammas.get(dates[t].year)
        if G is None:
            continue
        ok = valid_z[m] & np.isfinite(Rv[t])
        if ok.sum() < MIN_STOCKS:
            continue
        beta = Z[m, ok] @ G                                   # (n, K)
        Wm = beta @ np.linalg.inv(beta.T @ beta + 1e-10 * np.eye(K))
        r = Rv[t, ok]
        eps[t, ok] = r - (r @ Wm) @ beta.T
        Wd[t, ok] = Wm
        Bd[t, ok] = beta
    keep = np.isfinite(eps).any(axis=1)
    return dates[keep], eps[keep], Rv[keep].astype(np.float32), Wd[keep], Bd[keep]


def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading OHLCV ...", flush=True)
    P = load_panels()
    print("characteristics ...", flush=True)
    R, me, Z, valid_z, ret_next = monthly_data(P)
    years = list(range(1998, pd.Timestamp(R.index[-1]).year + 1))
    print("IPCA estimation ...", flush=True)
    gammas = estimate_gammas(me, Z, valid_z, ret_next, years)
    print("daily residuals ...", flush=True)
    dates, eps, Rv, Wd, Bd = daily_residuals(R, me, Z, valid_z, gammas)
    np.savez(os.path.join(OUT, "ipca5.npz"),
             dates=dates.values.astype("datetime64[D]"),
             tickers=np.array(R.columns), eps=eps, R=Rv, W=Wd, B=Bd,
             gammas=np.stack([gammas[y] for y in years]), gamma_years=years)
    print("saved", eps.shape, flush=True)


if __name__ == "__main__":
    main()
