"""Hard-coded chart patterns (Fidelity / Kirkpatrick "Identifying Chart
Patterns" deck) as exact rules on DAILY OHLC bars.

Every pattern produces a Setup: a breakout level (trigger) that, if crossed
on a later bar, opens a trade with a protective stop and a price target.
Nothing here looks ahead:
  * a swing high/low at bar i is only used from bar i + SWING_K on, once
    SWING_K lower highs / higher lows have formed after it;
  * a setup "armed" at bar t uses data up to and including bar t's close,
    and can only be entered on bars t+1 .. expiry.

The same functions drive the backtest (patterns/backtest_patterns.py) and
the live first-hour trader (patterns/pattern_trade.py), so live trades are
the patterns the backtest measured.

Patterns (deck slide 17):
  horizontal congestion : double_top/bottom, triple_top/bottom, rectangle
  triangles             : triangle_sym, triangle_asc, triangle_desc,
                          wedge_rising, wedge_falling
  other multi-bar       : head_shoulders_top/bottom, cup_handle
  candlesticks          : doji, harami, hammer, hanging_man,
                          inverted_hammer, shooting_star, engulfing_bull/bear,
                          piercing_line, dark_cloud
  short-term            : flag_bull/bear (flags and pennants), gap_up/down
                          (explosion gap pivot), pipe_bottom/top, inside_bar,
                          nr4
Two-sided patterns (doji, harami, inside_bar, nr4, rectangle, triangles,
wedges) arm a long and a short setup; whichever breaks first cancels the
other.

Stops and targets follow the deck where it gives a rule (measured moves:
pattern height projected from the breakout; cup: right lip to bottom; flag:
pole height from the flag low; rising wedge: lowest trough). Otherwise the
stop sits half a pattern height behind the breakout (multi-bar) or at the
far side of the pattern bars (candlesticks and short-term), and the target
is 2x the risk.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters (all sizes in ATRs so one setting works for any price level).
# Fix these BEFORE looking at out-of-sample results.
# ---------------------------------------------------------------------------
P = dict(
    atr_n=14,              # ATR window
    swing_k=3,             # swing high = highest high within 3 bars each side
    min_swing_atr=0.75,    # ignore zigzag legs smaller than this
    trend_n=5,             # bars that define the prior trend for candles
    peak_tol_atr=1.0,      # double/triple tops: peaks this close together
    min_height_atr=2.0,    # multi-bar patterns at least this tall
    max_span=120,          # multi-bar patterns at most this many bars wide
    head_min_atr=0.5,      # H&S head above both shoulders by at least this
    shoulder_tol_atr=1.5,  # H&S shoulders this close together
    flat_slope_atr=0.04,   # trend line counts as flat below this (ATR/bar)
    line_fit_atr=0.6,      # every pivot within this of its trend line
    stop_frac=0.5,         # multi-bar stop: half the height behind trigger
    target_r=2.0,          # candle / short-term target in multiples of risk
    multi_expiry=15,       # bars a multi-bar setup stays armed
    flag_pole_atr=4.0,     # flag pole at least this tall ...
    flag_pole_bars=10,     # ... built within this many bars
    gap_min_atr=0.5,       # explosion gap at least this big
    pipe_range_mult=1.5,   # pipe bars' ranges vs the prior 10 bars
    confirm_frac=0.0,      # breakout filter: trigger * (1 + this) (slide 10)
    max_gap_frac=0.5,      # skip if the open already covered this share of
                           # the trigger-to-target distance (no chasing)
)

ALL_PATTERNS = [
    "double_top", "double_bottom", "triple_top", "triple_bottom",
    "rectangle", "triangle_sym", "triangle_asc", "triangle_desc",
    "wedge_rising", "wedge_falling", "head_shoulders_top",
    "head_shoulders_bottom", "cup_handle",
    "doji", "harami", "hammer", "hanging_man", "inverted_hammer",
    "shooting_star", "engulfing_bull", "engulfing_bear", "piercing_line",
    "dark_cloud",
    "flag_bull", "flag_bear", "gap_up", "gap_down", "pipe_bottom",
    "pipe_top", "inside_bar", "nr4",
]


@dataclass
class Setup:
    pattern: str
    direction: int                 # +1 long (buy breakout), -1 short
    armed: int                     # bar index whose close completes it
    expiry: int                    # last bar index an entry may happen on
    trigger: float                 # breakout level at bar `armed`
    slope: float = 0.0             # trend-line triggers move per bar
    stop: Optional[float] = None   # absolute stop, else from height
    target: Optional[float] = None # absolute target, else from height / R
    target_r: Optional[float] = None
    height: float = 0.0
    invalid: Optional[float] = None  # a close beyond this cancels it
    group: int = -1                # two-sided siblings share a group

    def trigger_at(self, j: int) -> float:
        return self.trigger + self.slope * (j - self.armed)

    def levels(self, j: int):
        """(trigger, stop, target) for an entry on bar j."""
        trig = self.trigger_at(j)
        d = self.direction
        stop = (self.stop if self.stop is not None
                else trig - d * P["stop_frac"] * self.height)
        if self.target is not None:
            target = self.target
        elif self.target_r is not None:
            target = trig + d * self.target_r * abs(trig - stop)
        else:
            target = trig + d * self.height
        return trig, stop, target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def atr(H, L, C, n):
    prev = np.r_[np.nan, C[:-1]]
    tr = np.nanmax(np.vstack([H - L, np.abs(H - prev), np.abs(L - prev)]), axis=0)
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def raw_pivots(H, L, k):
    """Swing highs/lows as (bar, price, +1 high / -1 low), in bar order.
    A swing at bar i is known from bar i + k on."""
    n = len(H)
    out = []
    if n < 2 * k + 1:
        return out
    from numpy.lib.stride_tricks import sliding_window_view as sw
    wh = sw(H, 2 * k + 1)
    wl = sw(L, 2 * k + 1)
    for m in range(len(wh)):
        i = m + k
        if H[i] >= wh[m].max() and H[i] > wh[m][:k].max():
            out.append((i, H[i], 1))
        if L[i] <= wl[m].min() and L[i] < wl[m][:k].min():
            out.append((i, L[i], -1))
    return out


class _Groups:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return self.n


# ---------------------------------------------------------------------------
# Multi-bar patterns, evaluated each time the zigzag of confirmed swings
# changes (at bar t = the confirming bar)
# ---------------------------------------------------------------------------

def _tops(seq, t, C, a, g, sgn):
    """Tops (sgn=+1, short) or bottoms (sgn=-1, long) from the zigzag.
    Works on signed prices so one routine covers both. seq ends with a
    peak in signed space. Returns (major, double): a head-and-shoulders or
    triple top/bottom, and a double top/bottom (each a Setup or None)."""
    d = -sgn
    c = sgn * C[t]
    pts = [(i, sgn * p) for i, p, _ in seq]
    exp = t + P["multi_expiry"]
    name = {1: ("head_shoulders_top", "triple_top", "double_top"),
            -1: ("head_shoulders_bottom", "triple_bottom", "double_bottom")}[sgn]

    def mk(pattern, q, h, invalid, target=None, slope=0.0, stop=None):
        # q: signed trigger, h: height
        return Setup(pattern, d, t, exp, sgn * q, slope=sgn * slope,
                     stop=None if stop is None else sgn * stop,
                     target=None if target is None else sgn * target,
                     height=h, invalid=sgn * invalid, group=g.next())

    if len(pts) >= 5:
        (i1, p1), (j1, q1), (i2, p2), (j2, q2), (i3, p3) = pts[-5:]
        if i3 - i1 <= P["max_span"]:
            # head and shoulders: neckline through the two troughs
            nb = (q2 - q1) / (j2 - j1)
            neck = lambda j: q1 + nb * (j - j1)
            h = p2 - neck(i2)
            if (p2 - max(p1, p3) >= P["head_min_atr"] * a
                    and abs(p1 - p3) <= P["shoulder_tol_atr"] * a
                    and h >= P["min_height_atr"] * a and c > neck(t)
                    and abs(nb) <= P["flat_slope_atr"] * 3 * a):
                q = neck(t)
                return mk(name[0], q, h, invalid=p2, target=None, slope=nb), None
            # triple top: three peaks level, break of the lowest trough
            top, bot = max(p1, p2, p3), min(q1, q2)
            if (top - min(p1, p2, p3) <= P["peak_tol_atr"] * a
                    and top - bot >= P["min_height_atr"] * a and c > bot):
                h = top - bot
                return mk(name[1], bot, h, invalid=top, target=bot - h), None
    dbl = None
    if len(pts) >= 3:
        (i1, p1), (j1, q1), (i2, p2) = pts[-3:]
        if i2 - i1 <= P["max_span"] and i2 - i1 >= 5:
            top = max(p1, p2)
            h = top - q1
            if (abs(p1 - p2) <= P["peak_tol_atr"] * a
                    and h >= P["min_height_atr"] * a and c > q1):
                dbl = mk(name[2], q1, h, invalid=top, target=q1 - h)
    return None, dbl


def _rectangle(seq, t, C, a, g):
    if len(seq) < 4:
        return []
    last = seq[-4:]
    if last[-1][0] - last[0][0] > P["max_span"]:
        return []
    hi = [p for _, p, s in last if s == 1]
    lo = [p for _, p, s in last if s == -1]
    top, bot = max(hi), min(lo)
    h = np.mean(hi) - np.mean(lo)
    if (max(hi) - min(hi) <= P["peak_tol_atr"] * a
            and max(lo) - min(lo) <= P["peak_tol_atr"] * a
            and h >= P["min_height_atr"] * a and bot < C[t] < top):
        grp = g.next()
        exp = t + P["multi_expiry"]
        return [Setup("rectangle", 1, t, exp, top, height=h, group=grp),
                Setup("rectangle", -1, t, exp, bot, height=h, group=grp)]
    return []


def _triangles(seq, t, C, a, g):
    """Triangles and wedges from the last 4-6 swings (>= 2 highs, 2 lows)."""
    for n in (6, 5, 4):
        if len(seq) < n:
            continue
        pts = seq[-n:]
        if pts[-1][0] - pts[0][0] > P["max_span"]:
            continue
        hx = np.array([i for i, _, s in pts if s == 1], float)
        hy = np.array([p for _, p, s in pts if s == 1])
        lx = np.array([i for i, _, s in pts if s == -1], float)
        ly = np.array([p for _, p, s in pts if s == -1])
        if len(hx) < 2 or len(lx) < 2:
            continue
        bu, au = np.polyfit(hx, hy, 1)
        bl, al = np.polyfit(lx, ly, 1)
        if (np.abs(au + bu * hx - hy).max() > P["line_fit_atr"] * a
                or np.abs(al + bl * lx - ly).max() > P["line_fit_atr"] * a):
            continue
        up_t, lo_t = au + bu * t, al + bl * t
        width0 = (au + bu * pts[0][0]) - (al + bl * pts[0][0])
        if not (lo_t < C[t] < up_t) or up_t - lo_t <= 0 or width0 <= up_t - lo_t:
            continue                                     # must converge
        f = P["flat_slope_atr"] * a
        su, sl = bu, bl
        flat_u, flat_l = abs(su) <= f, abs(sl) <= f
        kind = None
        if su < -f and sl > f:
            kind = "triangle_sym"
        elif flat_u and sl > f:
            kind = "triangle_asc"
        elif su < -f and flat_l:
            kind = "triangle_desc"
        elif su > f and sl > f and n >= 5:
            kind = "wedge_rising"                        # 5 touches (slide 26)
        elif su < -f and sl < -f and n >= 5:
            kind = "wedge_falling"
        if kind is None:
            continue
        h = hy.max() - ly.min()
        if h < P["min_height_atr"] * a:
            continue
        # stay armed until the lines meet (apex), at most multi_expiry bars
        apex = (al - au) / (bu - bl) if bu != bl else t + P["multi_expiry"]
        exp = int(min(t + P["multi_expiry"], max(t + 1, np.floor(apex) - 1)))
        grp = g.next()
        long_tgt = short_tgt = None
        if kind == "wedge_rising":
            short_tgt = ly.min()                         # deck: lowest trough
        if kind == "wedge_falling":
            long_tgt = hy.max()
        out = [Setup(kind, 1, t, exp, up_t, slope=bu, height=h,
                     target=long_tgt, group=grp),
               Setup(kind, -1, t, exp, lo_t, slope=bl, height=h,
                     target=short_tgt, group=grp)]
        # a deck target already behind the trigger is not a trade
        return [s for s in out if s.target is None
                or s.direction * (s.target - s.trigger) > 0]
    return []


def _cup_handle(seq, t, C, L, a, g):
    """Rounded cup between two similar lips, then a shallow handle.
    seq ends with the handle low (a swing low)."""
    if len(seq) < 3 or seq[-1][2] != -1 or seq[-2][2] != 1:
        return None
    hi_idx, lip2, _ = seq[-2]
    h_idx, h_low, _ = seq[-1]
    for i, p, s in reversed(seq[:-2]):
        if s != 1:
            continue
        span = hi_idx - i
        if span > 150:
            break
        if span < 20 or abs(p - lip2) > 0.05 * lip2:
            continue
        mids = [q for j, q, ss in seq if i < j < hi_idx and ss == 1]
        if mids and max(mids) > min(p, lip2):
            continue                                     # lips must be the highs
        bottom = L[i:hi_idx + 1].min()
        depth = min(p, lip2) - bottom
        if not (0.10 <= depth / min(p, lip2) <= 0.50):
            continue
        near = (L[i:hi_idx + 1] <= bottom + depth / 3).mean()
        if near < 0.25:
            continue                                     # V bottom, not a cup
        if (lip2 - h_low > depth / 2 or not (3 <= h_idx - hi_idx <= 25)
                or C[t] >= max(p, lip2)):
            continue
        trig = max(p, lip2)
        return Setup("cup_handle", 1, t, t + P["multi_expiry"], trig,
                     stop=h_low, target=trig + (lip2 - bottom),
                     height=lip2 - bottom, invalid=h_low, group=g.next())
    return None


def multibar_setups(H, L, C, A, g):
    k = P["swing_k"]
    seq = []
    out = []
    for i, p, s in raw_pivots(H, L, k):
        t = i + k
        a = A[t]
        if not np.isfinite(a) or a <= 0:
            continue
        if seq and seq[-1][2] == s:
            if s * (p - seq[-1][1]) > 0:
                seq[-1] = (i, p, s)                      # more extreme: replace
            else:
                continue
        elif seq and abs(p - seq[-1][1]) < P["min_swing_atr"] * a:
            continue                                     # leg too small
        else:
            seq.append((i, p, s))
        seq = seq[-12:]
        # one pattern per swing, most specific first:
        # H&S / triple > rectangle > double > triangle / wedge > cup
        major, double = _tops(seq, t, C, a, g, sgn=seq[-1][2])
        if major is not None:
            out.append(major)
            continue
        rect = _rectangle(seq, t, C, a, g)
        if rect:
            out += rect
            continue
        if double is not None:
            out.append(double)
            continue
        tri = _triangles(seq, t, C, a, g)
        if tri:
            out += tri
            continue
        cup = _cup_handle(seq, t, C, L, a, g)
        if cup is not None:
            out.append(cup)
    return out


# ---------------------------------------------------------------------------
# Candlesticks and short-term patterns (vectorized; armed at bar t, entry
# allowed on bar t+1 only unless stated)
# ---------------------------------------------------------------------------

def bar_setups(O, H, L, C, A, g):
    n = len(C)
    out = []
    if n < 30:
        return out
    tn = P["trend_n"]
    body = np.abs(C - O)
    rng = H - L
    upper = H - np.maximum(O, C)
    lower = np.minimum(O, C) - L
    white, black = C > O, C < O

    def sh(x, s):                              # x shifted forward by s bars
        y = np.full(len(x), np.nan)
        y[s:] = x[:-s]
        return y

    def shb(x, s):                             # same for True/False arrays
        y = np.zeros(len(x), dtype=bool)
        y[s:] = x[:-s]
        return y

    C1, O1, H1, L1 = sh(C, 1), sh(O, 1), sh(H, 1), sh(L, 1)
    body1, rng1 = sh(body, 1), sh(rng, 1)
    down1 = sh(C, 1) < sh(C, 1 + tn)           # trend into a 1-bar pattern
    up1 = sh(C, 1) > sh(C, 1 + tn)
    down2 = sh(C, 2) < sh(C, 2 + tn)           # trend into a 2-bar pattern
    up2 = sh(C, 2) > sh(C, 2 + tn)
    ok = np.isfinite(A) & (rng > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        big = rng >= 0.5 * A
        R = P["target_r"]

        def one_sided(mask, name, d, trig, stop, expiry_bars=1):
            for t in np.flatnonzero(mask & ok):
                out.append(Setup(name, d, t, t + expiry_bars, trig[t],
                                 stop=stop[t], target_r=R, group=g.next()))

        def two_sided(mask, name, hi, lo, expiry_bars=1):
            for t in np.flatnonzero(mask & ok):
                grp = g.next()
                out.append(Setup(name, 1, t, t + expiry_bars, hi[t],
                                 stop=lo[t], target_r=R, group=grp))
                out.append(Setup(name, -1, t, t + expiry_bars, lo[t],
                                 stop=hi[t], target_r=R, group=grp))

        hi2 = np.fmax(H, H1)
        lo2 = np.fmin(L, L1)

        # doji: open == close (tiny body), shadows about equal (slide 33)
        doji = (body <= 0.1 * rng) & (np.abs(upper - lower) <= 0.4 * rng) & big
        two_sided(doji, "doji", H, L)

        # hammer family: small body at one end, long shadow (slides 35-36)
        hammer_shape = (lower >= 2 * body) & (upper <= 0.1 * rng) & (body > 0.1 * rng) & big
        inv_shape = (upper >= 2 * body) & (lower <= 0.1 * rng) & (body > 0.1 * rng) & big
        one_sided(hammer_shape & down1, "hammer", 1, H, L)
        one_sided(hammer_shape & up1, "hanging_man", -1, L, H)
        one_sided(inv_shape & down1, "inverted_hammer", 1, H, L)
        one_sided(inv_shape & up1, "shooting_star", -1, L, H)

        # engulfing: second body covers the first (slide 37)
        eng_bull = shb(black, 1) & white & (O <= C1) & (C >= O1) & (body > body1)
        eng_bear = shb(white, 1) & black & (O >= C1) & (C <= O1) & (body > body1)
        one_sided(eng_bull & down2, "engulfing_bull", 1, hi2, lo2)
        one_sided(eng_bear & up2, "engulfing_bear", -1, lo2, hi2)

        # harami: large body, then a small opposite body inside it (slide 34)
        bigbody1 = (body1 >= 0.6 * rng1) & (body1 >= 0.7 * A)
        inside_body = (np.maximum(O, C) <= np.fmax(O1, C1)) & (np.minimum(O, C) >= np.fmin(O1, C1))
        opposite = (white & shb(black, 1)) | (black & shb(white, 1))
        harami = bigbody1 & inside_body & opposite & (body <= 0.5 * body1)
        two_sided(harami, "harami", hi2, lo2)

        # piercing line / dark cloud cover (slide 38)
        mid1 = (O1 + C1) / 2
        pierce = (shb(black, 1) & (body1 >= 0.6 * rng1) & white
                  & (O < C1) & (C > mid1) & (C < O1) & down2)
        cloud = (shb(white, 1) & (body1 >= 0.6 * rng1) & black
                 & (O > C1) & (C < mid1) & (C > O1) & up2)
        one_sided(pierce, "piercing_line", 1, hi2, lo2)
        one_sided(cloud, "dark_cloud", -1, lo2, hi2)

        # inside bar and NR4: low-volatility breakouts (slides 43-44)
        inside = (H < H1) & (L > L1)
        two_sided(inside, "inside_bar", H, L)
        r = pd.Series(rng)
        nr4 = rng < r.shift(1).rolling(3).min().to_numpy()
        two_sided(nr4, "nr4", H, L)

        # pipe bottom / top: two wide bars at the end of a move (slide 42)
        avg10 = r.shift(2).rolling(10).mean().to_numpy()
        wide = (rng >= P["pipe_range_mult"] * avg10) & (rng1 >= P["pipe_range_mult"] * avg10)
        low20 = pd.Series(L).shift(2).rolling(20).min().to_numpy()
        high20 = pd.Series(H).shift(2).rolling(20).max().to_numpy()
        pos1 = (C1 - L1) / rng1
        pos2 = (C - L) / np.where(rng > 0, rng, np.nan)
        pipe_b = wide & (np.abs(L - L1) <= 0.25 * A) & (lo2 <= low20) & (pos1 <= 0.25) & (pos2 >= 0.5)
        pipe_t = wide & (np.abs(H - H1) <= 0.25 * A) & (hi2 >= high20) & (pos1 >= 0.75) & (pos2 <= 0.5)
        for t in np.flatnonzero(pipe_b & ok):
            h = hi2[t] - lo2[t]
            out.append(Setup("pipe_bottom", 1, t, t + 3, hi2[t], stop=lo2[t],
                             target=hi2[t] + h, height=h, group=g.next()))
        for t in np.flatnonzero(pipe_t & ok):
            h = hi2[t] - lo2[t]
            out.append(Setup("pipe_top", -1, t, t + 3, lo2[t], stop=hi2[t],
                             target=lo2[t] - h, height=h, group=g.next()))

    out += _flags(H, L, C, A, g)
    out += _gaps(H, L, C, A, g)
    return out


def _flags(H, L, C, A, g):
    """Flag / pennant: a steep pole, then a short, shallow consolidation.
    Long above the consolidation high, target = flag low + pole (slide 39)."""
    out = []
    n = len(C)
    pb = P["flag_pole_bars"]
    for t in range(pb + 4, n):
        a = A[t]
        if not np.isfinite(a):
            continue
        for c in range(4, 13):                       # consolidation length
            s = t - c                                # last bar of the pole
            if s - pb < 0:
                break
            cons_h, cons_l = H[s + 1:t + 1].max(), L[s + 1:t + 1].min()
            # bull flag
            pole_lo = L[s - pb:s + 1].min()
            pole = H[s] - pole_lo
            if (pole >= P["flag_pole_atr"] * a and H[s] >= H[s - pb:s + 1].max()
                    and cons_h <= H[s] and cons_l >= H[s] - 0.5 * pole
                    and C[t] < cons_h):
                out.append(Setup("flag_bull", 1, t, t + 5, cons_h, stop=cons_l,
                                 target=cons_l + pole, height=pole,
                                 invalid=cons_l, group=g.next()))
                break
            # bear flag
            pole_hi = H[s - pb:s + 1].max()
            pole = pole_hi - L[s]
            if (pole >= P["flag_pole_atr"] * a and L[s] <= L[s - pb:s + 1].min()
                    and cons_l >= L[s] and cons_h <= L[s] + 0.5 * pole
                    and C[t] > cons_l):
                out.append(Setup("flag_bear", -1, t, t + 5, cons_l, stop=cons_h,
                                 target=cons_h - pole, height=pole,
                                 invalid=cons_h, group=g.next()))
                break
    return out


def _gaps(H, L, C, A, g):
    """Explosion gap pivot (slide 41): after a gap that is not covered, wait
    for the throwback to stop (pivot low), then buy above the gap bar's high;
    stop below the pivot low."""
    out = []
    n = len(C)
    for gi in range(1, n):
        a = A[gi]
        if not np.isfinite(a):
            continue
        up = L[gi] - H[gi - 1] >= P["gap_min_atr"] * a
        dn = L[gi - 1] - H[gi] >= P["gap_min_atr"] * a
        if not (up or dn):
            continue
        for t in range(gi + 2, min(gi + 11, n)):
            if up:
                lows = L[gi + 1:t + 1]
                if lows.min() <= H[gi - 1]:
                    break                            # throwback covered the gap
                m = gi + 1 + int(lows.argmin())
                if m < t and C[t] < H[gi]:           # pivot low held, below gap high
                    trig, stop = H[gi], L[m]
                    out.append(Setup("gap_up", 1, t, gi + 10, trig, stop=stop,
                                     target_r=P["target_r"], invalid=stop,
                                     group=g.next()))
                    break
            else:
                highs = H[gi + 1:t + 1]
                if highs.max() >= L[gi - 1]:
                    break
                m = gi + 1 + int(highs.argmax())
                if m < t and C[t] > L[gi]:
                    trig, stop = L[gi], H[m]
                    out.append(Setup("gap_down", -1, t, gi + 10, trig, stop=stop,
                                     target_r=P["target_r"], invalid=stop,
                                     group=g.next()))
                    break
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_setups(df: pd.DataFrame, patterns=None):
    """All setups in a daily OHLC frame (columns Open/High/Low/Close),
    each armed at the close of some bar. Overlapping repeats of the same
    pattern and direction are dropped (the earlier one stays)."""
    O, H, L, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    A = atr(H, L, C, P["atr_n"])
    g = _Groups()
    setups = multibar_setups(H, L, C, A, g) + bar_setups(O, H, L, C, A, g)
    if patterns is not None:
        keep = set(patterns)
        setups = [s for s in setups if s.pattern in keep]
    setups.sort(key=lambda s: (s.armed, s.group, -s.direction))
    out, busy = [], {}
    for s in setups:
        key = (s.pattern, s.direction)
        if busy.get(key, -1) >= s.armed:
            continue
        busy[key] = s.expiry
        out.append(s)
    return out


def entry_on_bar(s: Setup, j, O, H, L, C):
    """What happens to an armed setup on bar j (j > s.armed).
    Returns ("enter", fill, stop, target), ("cancel",) or None (still armed,
    check the next bar). Same rules as the live trader:
      * entry on a breakout through trigger * (1 + confirm_frac);
      * if the bar opens beyond the trigger, fill at the open, unless the
        gap already used up max_gap_frac of the move to the target;
      * an unfilled setup is cancelled by a close beyond `invalid` or after
        `expiry`."""
    if j > s.expiry:
        return ("cancel",)
    d = s.direction
    trig, stop, target = s.levels(j)
    lvl = trig * (1 + d * P["confirm_frac"])
    if d * (O[j] - lvl) >= 0:
        if d * (O[j] - trig) > P["max_gap_frac"] * d * (target - trig):
            return ("cancel",)
        fill = O[j]
    elif d * ((H[j] if d > 0 else L[j]) - lvl) >= 0:
        fill = lvl
    else:
        if s.invalid is not None and d * (C[j] - s.invalid) < 0:
            return ("cancel",)
        return None
    if not (d * (target - fill) > 0 and d * (fill - stop) > 0):
        return ("cancel",)
    return ("enter", fill, stop, target)


def live_setups(df: pd.DataFrame, patterns=None):
    """Setups that are still armed for the NEXT session, given daily bars
    through the last completed session (do not include today's partial bar).
    Walks every setup through the bars after it was armed, so one that
    already triggered, broke or expired is dropped."""
    O, H, L, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    last = len(df) - 1
    nxt = last + 1
    setups = [s for s in find_setups(df, patterns)
              if s.armed <= last and s.expiry >= nxt]
    alive, dead_groups = [], set()
    for s in setups:
        state = None
        for j in range(s.armed + 1, last + 1):
            state = entry_on_bar(s, j, O, H, L, C)
            if state is not None:
                break
        if state is None:
            alive.append(s)
        elif state[0] == "enter":
            dead_groups.add(s.group)                 # sibling already traded
    return [s for s in alive if s.group not in dead_groups]
