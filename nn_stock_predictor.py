"""
NN Stock Predictor for S&P 500 Minute-Level Data
Inspired by KAT Bobcat (Krafer Agent Trader) short-term prediction architecture.

FIXES APPLIED:
- Scaler fit on TRAIN ONLY (no data leakage)
- Predicts RETURNS, not price levels (survives scrutiny)
- Directional accuracy vs naive persistence baseline
- BatchNorm safe (drop_last=True)
- PyTorch 2.7+ compatible (no verbose in scheduler)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
import json
import os
import pickle
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MARKET REGIME ENUMS
# =============================================================================
class MarketRegime(Enum):
    CHOP = "chop"
    TRANSITION = "transition"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"

class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class VolumeRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    TICKER = "SPY"
    # Daily bars back to SPY's 1993 inception (~8,200 rows) instead of the
    # ~2,700 bars Yahoo allows for 1m data. Minute returns are dominated by
    # microstructure noise; daily returns carry far more signal per sample.
    INTERVAL = "1m"
    PERIOD = "7d"
    LOOKBACK_WINDOW = 30
    PREDICTION_HORIZON = 1

    # "direction"  -> forward return sign/magnitude (hard; no edge found)
    # "volatility" -> next-bar realized volatility. Volatility clusters, so
    #                 this genuinely is predictable — but the persistence
    #                 baseline is strong, so beating it is the real test.
    TARGET = "direction"  # or "volatility"

    # For TARGET="volatility": predict the AVERAGE log vol over the next N bars.
    # Tested 1 vs 5 on a GARCH series: averaging RAISES correlation (0.77 -> 0.82)
    # but raises the matched trailing-average baseline faster, so R2 vs
    # persistence went +0.47 -> -0.21. Keep 1 unless you re-test on real data.
    # Note >1 also makes test samples overlap, so the effective sample size is
    # smaller than n_eval suggests.
    VOL_HORIZON = 1
    TRAIN_SPLIT = 0.5

    INPUT_FEATURES = 14
    # Was 128/3 = ~1.0M params against ~1,300 training sequences. Even with
    # 8k daily bars that is heavily over-parameterized; 64/2 is ~140k.
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BIDIRECTIONAL = True

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    PATIENCE = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ADX_CHOP_THRESHOLD = 20.0
    ADX_TREND_THRESHOLD = 25.0
    VOLATILITY_PCT_LOW = 30
    VOLATILITY_PCT_HIGH = 70
    VOLUME_PCT_LOW = 20
    VOLUME_PCT_HIGH = 80

    # "auto"  -> try real bars, fall back to synthetic if the fetch fails
    # "real"   -> real bars only; raise instead of silently falling back
    # "synth"  -> synthetic random walk only (pipeline testing)
    SOURCE = "auto"
    SEED = 42                    # makes the synthetic fallback reproducible

    MODEL_SAVE_PATH = "models/nn_stock_predictor.pt"
    SCALER_SAVE_PATH = "models/scaler.pkl"


# =============================================================================
# TECHNICAL INDICATORS & REGIME DETECTION
# =============================================================================
def calculate_true_range(df):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr

def calculate_atr(df, period=14):
    tr = calculate_true_range(df)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    """
    Average Directional Index (ADX) using simple rolling mean.
    Note: True Wilder's smoothing uses exponential decay; this uses SMA for simplicity.
    """
    df = df.copy()
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di


def parkinson_volatility(df):
    """
    Parkinson (1980) range-based volatility estimator, one value per bar:

        sigma = sqrt( ln(H/L)^2 / (4 ln 2) )

    Uses only that bar's own high and low, so it is strictly causal. It is
    roughly 5x more efficient than a close-to-close estimator because the
    intraday range carries more information than the endpoints alone.
    """
    hl = np.log(df["High"] / df["Low"]).replace([np.inf, -np.inf], np.nan)
    return np.sqrt(hl ** 2 / (4 * np.log(2)))


def add_technical_indicators(df):
    df = df.copy()
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_ratio'] = df['sma_5'] / df['sma_10']
    df['volatility'] = df['Close'].rolling(window=10).std()
    df['bb_position'] = (df['Close'] - df['sma_10']) / (2 * df['volatility'] + 1e-8)
    df['volume_sma'] = df['Volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['atr'] = calculate_atr(df, period=14)
    df['parkinson_vol'] = parkinson_volatility(df)

    # HAR (Corsi 2009) volatility terms: log vol averaged over the last day,
    # week, and month. A linear model on just these three famously beats a
    # random-walk volatility forecast, so handing them to the network directly
    # saves it from having to rediscover the structure from raw OHLCV.
    log_vol = np.log(df['parkinson_vol'].replace(0, np.nan).ffill().bfill() + 1e-12)
    df['har_vol_1d'] = log_vol
    df['har_vol_5d'] = log_vol.rolling(5).mean()
    df['har_vol_22d'] = log_vol.rolling(22).mean()
    df['har_vol_ratio'] = df['har_vol_5d'] - df['har_vol_22d']  # short vs long
    df['atr_percentile'] = df['atr'].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, period=14)
    df['volume_percentile'] = df['Volume'].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    df = df.dropna()
    return df


def classify_market_regime(row):
    adx = row.get('adx', 0)
    plus_di = row.get('plus_di', 0)
    minus_di = row.get('minus_di', 0)
    if adx < Config.ADX_CHOP_THRESHOLD:
        return MarketRegime.CHOP
    elif adx < Config.ADX_TREND_THRESHOLD:
        return MarketRegime.TRANSITION
    elif plus_di > minus_di:
        return MarketRegime.TRENDING_UP
    else:
        return MarketRegime.TRENDING_DOWN

def classify_volatility_regime(row):
    atr_pct = row.get('atr_percentile', 50)
    if atr_pct < Config.VOLATILITY_PCT_LOW:
        return VolatilityRegime.LOW
    elif atr_pct > Config.VOLATILITY_PCT_HIGH:
        return VolatilityRegime.HIGH
    return VolatilityRegime.NORMAL

def classify_volume_regime(row):
    vol_pct = row.get('volume_percentile', 50)
    if vol_pct < Config.VOLUME_PCT_LOW:
        return VolumeRegime.LOW
    elif vol_pct > Config.VOLUME_PCT_HIGH:
        return VolumeRegime.HIGH
    return VolumeRegime.NORMAL

def is_trading_allowed(row):
    regime = classify_market_regime(row)
    vol_regime = classify_volatility_regime(row)
    vol_regime2 = classify_volume_regime(row)
    chop_ok = regime in [MarketRegime.CHOP, MarketRegime.TRANSITION]
    vol_ok = vol_regime != VolatilityRegime.HIGH
    volume_ok = vol_regime2 != VolumeRegime.HIGH
    return chop_ok and vol_ok and volume_ok


def get_regime_features(row):
    regime = classify_market_regime(row)
    vol_regime = classify_volatility_regime(row)
    return {
        'is_chop': 1.0 if regime == MarketRegime.CHOP else 0.0,
        'is_trending': 1.0 if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] else 0.0,
        'is_low_vol': 1.0 if vol_regime == VolatilityRegime.LOW else 0.0,
        'is_high_vol': 1.0 if vol_regime == VolatilityRegime.HIGH else 0.0,
        'adx_normalized': row.get('adx', 25) / 50.0,
        'atr_pct_normalized': row.get('atr_percentile', 50) / 100.0,
    }


# =============================================================================
# DATASET
# =============================================================================
class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def build_feature_matrix(df):
    """Build feature matrix from DataFrame. Returns (data, feature_names)."""
    base_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'returns', 'log_returns', 'sma_ratio', 
        'bb_position', 'volume_ratio', 'rsi',
        'adx', 'atr', 'volume_percentile',
        'har_vol_1d', 'har_vol_5d', 'har_vol_22d', 'har_vol_ratio'
    ]
    available_cols = [c for c in base_features if c in df.columns]
    data = df[available_cols].values

    regime_features = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        rf = get_regime_features(row)
        regime_features.append([
            rf['is_chop'], rf['is_trending'], 
            rf['is_low_vol'], rf['is_high_vol'],
            rf['adx_normalized'], rf['atr_pct_normalized']
        ])
    regime_features = np.array(regime_features)

    full_data = np.hstack([data, regime_features])
    feature_names = available_cols + [
        'is_chop', 'is_trending', 'is_low_vol', 
        'is_high_vol', 'adx_norm', 'atr_pct_norm'
    ]
    return full_data, feature_names


def prepare_data(df, lookback_window=30, prediction_horizon=1, train_split=0.5,
                 target="direction", vol_horizon=5):
    """
    Prepare sequences with NO DATA LEAKAGE.
    Scaler is fit on training data ONLY, then applied to val/test.
    Target is the FUTURE RETURN, not the price level.

    Window/target alignment: sample j anchored at bar i uses rows
    (i - lookback + 1 .. i) as input and predicts returns[i + horizon].
    With horizon=1 that is the return from bar i to bar i+1 — the very next
    bar after the last one the model can see, with no unobserved gap between
    input and target. (An earlier version ended the window at i-1, which left
    one invisible bar in between and quietly made this a 2-bar forecast.)
    """
    full_data, feature_names = build_feature_matrix(df)

    if target == "volatility":
        # Predict LOG volatility: vol is right-skewed and strictly positive, so
        # logs make the target closer to Gaussian and the MSE loss sensible.
        log_vol = np.log(df['parkinson_vol'].replace(0, np.nan)
                         .ffill().bfill().values + 1e-12)
        h = max(int(vol_horizon), 1)
        # Target: mean log vol over bars (i+1 .. i+h) — strictly future.
        # Baseline: mean log vol over bars (i-h+1 .. i) — strictly past, the
        # matched HAR-style random walk. Comparing a 5-day forward average to a
        # 1-day trailing value would flatter the model, so the windows match.
        X, y, baseline = [], [], []
        anchors = range(max(lookback_window - 1, h - 1), len(full_data) - h)
        for i in anchors:
            X.append(full_data[i - lookback_window + 1:i + 1])
            y.append(log_vol[i + 1:i + 1 + h].mean())
            baseline.append(log_vol[i - h + 1:i + 1].mean())
    else:
        series = df['returns'].values
        X, y, baseline = [], [], []
        anchors = range(lookback_window - 1, len(full_data) - prediction_horizon)
        for i in anchors:
            X.append(full_data[i - lookback_window + 1:i + 1])
            y.append(series[i + prediction_horizon])
            # Persistence baseline = the last value the model actually saw.
            baseline.append(series[i])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    baseline = np.array(baseline)

    # Trading allowed mask — one entry per sample, same order as X
    trade_mask = np.array([is_trading_allowed(df.iloc[i]) for i in anchors])

    # Split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    # Split the mask and the baseline the SAME way, so evaluate_model gets the
    # entries that line up with the test predictions instead of slicing the
    # front of a full-length array (which was the training region).
    mask_train, mask_test = trade_mask[:split_idx], trade_mask[split_idx:]
    base_train, base_test = baseline[:split_idx], baseline[split_idx:]

    val_split = int(len(X_train) * 0.8)
    X_train, X_val = X_train[:val_split], X_train[val_split:]
    y_train, y_val = y_train[:val_split], y_train[val_split:]

    # === FIT SCALER ON TRAIN ONLY ===
    # StandardScaler, not MinMax: MinMax fit on train maps price-level features
    # into [-1, 1] on train, but a trending test segment lands far outside that
    # range (Close reached 3.4 in testing). StandardScaler degrades gracefully.
    scaler = StandardScaler()

    # Reshape for scaler: (samples * timesteps, features)
    n_train_samples, seq_len, n_features = X_train.shape
    X_train_2d = X_train.reshape(-1, n_features)
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_train = X_train_2d.reshape(n_train_samples, seq_len, n_features)

    # Transform val/test with the SAME scaler
    n_val_samples = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, n_features)
    X_val_2d = scaler.transform(X_val_2d)
    X_val = X_val_2d.reshape(n_val_samples, seq_len, n_features)

    n_test_samples = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, n_features)
    X_test_2d = scaler.transform(X_test_2d)
    X_test = X_test_2d.reshape(n_test_samples, seq_len, n_features)

    # Scale targets separately (returns are already small, but let's be consistent)
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train)
    y_val = target_scaler.transform(y_val)
    y_test = target_scaler.transform(y_test)

    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            scaler, target_scaler, feature_names, mask_test, base_test)


# =============================================================================
# MODEL
# =============================================================================
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, 
                 dropout=0.2, bidirectional=True):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_out_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        self.fc1 = nn.Linear(lstm_out_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.activation = nn.LeakyReLU(0.1)

    def attention_weights(self, lstm_output):
        scores = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        return weights

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention_weights(lstm_out)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


# =============================================================================
# TRAINING
# =============================================================================
def train_model(model, train_loader, val_loader, config):
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"Training on {config.DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config.EPOCHS):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    return model, history


# =============================================================================
# EVALUATION
# =============================================================================
def _evaluate_volatility(pred, actual, baseline):
    """
    Volatility forecasting metrics. Everything is in LOG volatility units.

    Two R-squared values are reported and they are not interchangeable:

      R2 vs mean        — against the unconditional average. Easy to beat,
                          because volatility clusters. A high number here is
                          NOT evidence of a good model.
      R2 vs persistence — against "tomorrow's vol equals today's". This is the
                          honest test. A random-walk forecast is already strong,
                          so positive skill here is a real result.

    Also reports QLIKE, the standard loss for volatility forecasts, which
    penalizes under-prediction more than over-prediction (getting a calm
    forecast wrong before a crash is worse than the reverse).
    """
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)

    mean_pred = np.full_like(actual, actual.mean())
    mse_mean = mean_squared_error(actual, mean_pred)
    mse_persist = mean_squared_error(actual, baseline)

    r2_mean = 1 - mse / mse_mean
    r2_persist = 1 - mse / mse_persist

    corr = float(np.corrcoef(pred, actual)[0, 1])

    # QLIKE on variance scale (undo the log)
    va, vp = np.exp(actual) ** 2, np.exp(pred) ** 2
    qlike = float(np.mean(va / vp - np.log(va / vp) - 1))
    vb = np.exp(baseline) ** 2
    qlike_persist = float(np.mean(va / vb - np.log(va / vb) - 1))

    n = len(actual)
    se_r2 = float(np.sqrt((1 - r2_persist ** 2) / max(n - 2, 1)))

    # Guard against the degenerate win: a model that just predicts the average
    # every day beats persistence whenever volatility does NOT cluster, while
    # having no skill at all. Two checks catch it -- how much the forecast
    # actually varies, and whether it beats the unconditional mean too.
    spread_ratio = float(np.std(pred) / (np.std(actual) + 1e-12))
    near_constant = spread_ratio < 0.15
    real_skill = (r2_persist > 0 and r2_mean > 0 and qlike < qlike_persist
                  and not near_constant)

    print("\n" + "="*55)
    print("EVALUATION METRICS (Volatility Forecasting)")
    print("="*55)
    print(f"Test samples:               {n}")
    print(f"RMSE (log vol):             {rmse:.6f}")
    print(f"MAE  (log vol):             {mae:.6f}")
    print(f"Correlation with actual:    {corr:.4f}")
    print(f"R2 vs unconditional mean:   {r2_mean:.4f}   <- easy baseline")
    print(f"R2 vs persistence:          {r2_persist:.4f}   <- the real test", end="")
    print("   (NO skill)" if r2_persist <= 0 else "")
    print(f"QLIKE (model):              {qlike:.4f}")
    print(f"QLIKE (persistence):        {qlike_persist:.4f}", end="")
    print("   <- model is worse" if qlike > qlike_persist else "   <- model is better")
    print(f"Forecast spread / actual:   {spread_ratio:.3f}", end="")
    print("   <- NEAR-CONSTANT forecast, not a real model" if near_constant else "")
    print(f"VERDICT: {'real skill' if real_skill else 'NO real skill'} "
          f"(needs R2>0 vs BOTH baselines, better QLIKE, and a varying forecast)")
    print("="*55)

    return {
        'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae),
        'n_eval': n, 'corr': corr,
        'r2_vs_mean': float(r2_mean),
        'r2_vs_persistence': float(r2_persist),
        'r2_persistence_se': se_r2,
        'qlike': qlike, 'qlike_persistence': qlike_persist,
        'spread_ratio': spread_ratio,
        'near_constant': bool(near_constant),
        'beats_persistence': bool(real_skill),
        'predictions': pred, 'actuals': actual,
    }


def evaluate_model(model, test_loader, target_scaler, config, trade_mask=None,
                   baseline_returns=None, target="direction"):
    """
    Evaluate on returns (not price levels).
    Report directional accuracy vs naive persistence baseline.
    """
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.DEVICE)
            outputs = model(batch_x).cpu().numpy()
            predictions.extend(outputs.flatten().tolist())
            actuals.extend(batch_y.numpy().flatten().tolist())

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    # Inverse transform to get actual returns
    pred_returns = target_scaler.inverse_transform(predictions).flatten()
    actual_returns_arr = target_scaler.inverse_transform(actuals).flatten()

    # Metrics on returns
    mse = mean_squared_error(actual_returns_arr, pred_returns)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_returns_arr, pred_returns)

    if target == "volatility":
        return _evaluate_volatility(pred_returns, actual_returns_arr,
                                    baseline_returns)

    # Directional accuracy, ignoring flat bars where there is no call to make
    moved = actual_returns_arr != 0
    n_eval = int(moved.sum())
    pred_direction = np.sign(pred_returns)
    actual_direction = np.sign(actual_returns_arr)
    correct = pred_direction[moved] == actual_direction[moved]
    directional_acc = np.mean(correct) * 100

    # Standard error on that percentage. Without it, a couple of points above
    # 50% reads as an edge when it is usually just noise at this sample size.
    se = np.sqrt(directional_acc * (100 - directional_acc) / max(n_eval, 1))

    # Naive persistence baseline: same direction as the last return the model
    # actually saw. `baseline_returns` comes from prepare_data already aligned
    # to the test samples — no slicing guesswork.
    if baseline_returns is not None and len(baseline_returns) == len(actual_returns_arr):
        baseline_direction = np.sign(baseline_returns)
        baseline_acc = np.mean(baseline_direction[moved] == actual_direction[moved]) * 100
    else:
        baseline_acc = 50.0  # Random guess baseline

    # Majority-class baseline: always predict the more common direction.
    # On DAILY bars this is the baseline that matters — equities drift upward,
    # so "always up" scores ~54% with zero skill. A model that collapses to a
    # constant sign will look impressive against persistence alone. Reporting
    # only the persistence baseline would hide exactly that failure.
    up_rate = float(np.mean(actual_direction[moved] > 0) * 100)
    majority_acc = max(up_rate, 100 - up_rate)

    # Does the model actually vary its calls, or has it collapsed?
    pred_up_rate = float(np.mean(pred_direction[moved] > 0) * 100)
    collapsed = pred_up_rate > 97.0 or pred_up_rate < 3.0

    edge = directional_acc - baseline_acc
    edge_vs_majority = directional_acc - majority_acc

    # Regime-aware directional accuracy. trade_mask is now the TEST-aligned
    # mask from prepare_data, so no slicing is needed.
    chop_da = None
    if trade_mask is not None and len(trade_mask) == len(actual_returns_arr):
        chop_mask = trade_mask & moved
        if chop_mask.sum() > 0:
            chop_da = np.mean(pred_direction[chop_mask] == actual_direction[chop_mask]) * 100

    print("\n" + "="*55)
    print("EVALUATION METRICS (Return Prediction)")
    print("="*55)
    print(f"Test samples (non-flat bars): {n_eval}")
    print(f"MSE:  {mse:.8f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"Directional Accuracy:      {directional_acc:.2f}%  (+/- {se:.2f} SE)")
    print(f"Naive Persistence Baseline: {baseline_acc:.2f}%")
    print(f"Majority-Class Baseline:    {majority_acc:.2f}%  "
          f"(always {'up' if up_rate >= 50 else 'down'})")
    print(f"Edge over persistence:      {edge:+.2f}%", end="")
    print(f"   <- within noise (< 2 SE)" if abs(edge) < 2 * se else "")
    print(f"Edge over majority class:   {edge_vs_majority:+.2f}%", end="")
    print(f"   <- within noise (< 2 SE)" if abs(edge_vs_majority) < 2 * se else "")
    print(f"Model predicted 'up' on {pred_up_rate:.1f}% of bars", end="")
    print("   <- COLLAPSED to a constant sign, not a real model" if collapsed else "")
    if chop_da is not None:
        print(f"Directional Acc (Chop):    {chop_da:.2f}%  ({int(chop_mask.sum())} bars)")
    print("="*55)

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae,
        'n_eval': n_eval,
        'directional_acc': directional_acc,
        'directional_acc_se': se,
        'baseline_acc': baseline_acc,
        'majority_acc': majority_acc,
        'edge_over_baseline': edge,
        'edge_over_majority': edge_vs_majority,
        'pred_up_rate': pred_up_rate,
        'collapsed': collapsed,
        'chop_da': chop_da,
        'predictions': pred_returns,
        'actuals': actual_returns_arr
    }



# =============================================================================
# DATA LOADING
# =============================================================================
def synthetic_market_data(n=10000, seed=None):
    """
    A pure random walk. No predictable structure by construction — a model
    scoring ~50% directional accuracy on this is behaving correctly.

    Seeded so replays are reproducible. Without this the same command produces
    different bars each run, which makes any comparison meaningless.
    """
    if seed is not None:
        np.random.seed(seed)
    prices = 450.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.001, n)))
    return pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.0005, n)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.001, n))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, n))),
        'Close': prices,
        'Volume': np.random.randint(100000, 500000, n)
    }, index=pd.date_range('2024-01-01', periods=n, freq='min'))


def load_market_data(config):
    """
    Returns (df, source) where source is "real" or "synthetic", so callers can
    report which one was actually used instead of guessing from the logs.
    """
    seed = getattr(config, "SEED", None)
    if getattr(config, "SOURCE", "auto") == "synth":
        print(f"\nDATA SOURCE: synthetic (forced, seed={seed})")
        return synthetic_market_data(seed=seed), "synthetic"

    print(f"\nFetching {config.TICKER} minute data...")
    try:
        df = yf.download(config.TICKER, period=config.PERIOD,
                         interval=config.INTERVAL, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() if c.lower() in
                      ['open', 'high', 'low', 'close', 'volume'] else c
                      for c in df.columns]
        if df.empty:
            raise ValueError("Empty data")
        print(f"DATA SOURCE: real — {len(df)} bars of {config.TICKER} "
              f"{config.INTERVAL} over {config.PERIOD}")
        return df, "real"
    except Exception as e:
        if getattr(config, "SOURCE", "auto") == "real":
            raise RuntimeError(
                f"Real data required but the fetch failed: {e}. "
                f"Yahoo only serves ~7 days of 1m bars; check PERIOD/INTERVAL "
                f"and your connection.") from e
        print(f"DATA SOURCE: synthetic — real fetch failed ({e}), seed={seed}")
        return synthetic_market_data(seed=seed), "synthetic"



def save_inference_bundle(path, model_path, scaler, target_scaler,
                          feature_names, config, data_source="unknown"):
    """
    Everything needed to run the predictor later on ONE new bar.

    The .pt weights alone are useless without these: you could not reproduce
    the feature scaling or invert the target scaling, so predictions would be
    silently wrong. Saved together so they can't drift apart.
    """
    with open(path, 'wb') as f:
        pickle.dump({
            'model_path': model_path,
            'feature_scaler': scaler,
            'target_scaler': target_scaler,
            'feature_names': feature_names,
            'lookback_window': config.LOOKBACK_WINDOW,
            'prediction_horizon': config.PREDICTION_HORIZON,
            'hidden_size': config.HIDDEN_SIZE,
            'num_layers': config.NUM_LAYERS,
            'dropout': config.DROPOUT,
            'bidirectional': config.BIDIRECTIONAL,
            'ticker': config.TICKER,
            'interval': config.INTERVAL,
            'data_source': data_source,
            'saved_at': datetime.now().isoformat(),
        }, f)
    print(f"Inference bundle saved to: {path}")


def load_inference_bundle(path, device=None):
    """Rebuild the trained predictor plus its scalers from disk."""
    with open(path, 'rb') as f:
        b = pickle.load(f)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockPredictor(len(b['feature_names']), b['hidden_size'],
                           b['num_layers'], b['dropout'], b['bidirectional'])
    model.load_state_dict(torch.load(b['model_path'], map_location=device))
    model.to(device).eval()
    return model, b


# =============================================================================
# BRIDGE TO THE DQN
# =============================================================================
def train_and_forecast(config=None, seed=42):
    """
    Train the predictor, then return everything the DQN needs for the TEST
    segment: actual prices, predicted next prices, and regime flags.

    The DQN therefore trades on bars the predictor never trained on, which is
    the whole point — an agent tuned against in-sample forecasts learns to rely
    on accuracy it will never see again.

    Returns (prices, predicted_prices, regime_data).
    """
    from torch.utils.data import DataLoader

    config = config or Config()
    if getattr(config, "TARGET", "direction") != "direction":
        raise RuntimeError(
            "pipeline mode needs TARGET='direction': the DQN environment "
            "consumes a predicted PRICE, not a volatility forecast. Use "
            "`run.py predictor` for the volatility work.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs("models", exist_ok=True)

    df, data_source = load_market_data(config)
    df = add_technical_indicators(df)

    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, target_scaler, feature_names, mask_test, base_test) = prepare_data(
        df, config.LOOKBACK_WINDOW, config.PREDICTION_HORIZON, config.TRAIN_SPLIT,
        target=config.TARGET, vol_horizon=config.VOL_HORIZON)

    train_loader = DataLoader(StockDataset(X_train, y_train),
                              batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=config.BATCH_SIZE)

    model = StockPredictor(len(feature_names), config.HIDDEN_SIZE,
                           config.NUM_LAYERS, config.DROPOUT, config.BIDIRECTIONAL)
    model, _ = train_model(model, train_loader, val_loader, config)

    # Predicted forward returns on the held-out test samples
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_test), 256):
            batch = torch.FloatTensor(X_test[start:start + 256]).to(config.DEVICE)
            preds.append(model(batch).cpu().numpy())
    pred_returns = target_scaler.inverse_transform(np.vstack(preds)).flatten()

    # Line the samples back up with the DataFrame rows they came from.
    h = config.PREDICTION_HORIZON
    anchors = np.arange(config.LOOKBACK_WINDOW - 1, len(df) - h)
    test_anchors = anchors[int(len(anchors) * config.TRAIN_SPLIT):]

    prices = df['Close'].values[test_anchors]
    predicted_prices = prices * (1 + pred_returns)

    regime_data = {}
    for key in ('is_chop', 'is_trending', 'is_low_vol', 'is_high_vol'):
        regime_data[key] = np.array(
            [get_regime_features(df.iloc[i])[key] for i in test_anchors])

    bundle_path = config.MODEL_SAVE_PATH.replace('.pt', '_bundle.pkl')
    save_inference_bundle(bundle_path, config.MODEL_SAVE_PATH, scaler,
                          target_scaler, feature_names, config, data_source)

    print(f"\nForecasts ready: {len(prices)} predictor-unseen bars for the DQN")
    return prices, predicted_prices, regime_data, data_source


# =============================================================================
# MAIN
# =============================================================================
def main():
    config = Config()
    os.makedirs("models", exist_ok=True)

    print("="*60)
    print("NN STOCK PREDICTOR — S&P 500 Minute-Level")
    print("Target: Future Returns | Scaler: Fit on Train Only")
    print("="*60)

    df, data_source = load_market_data(config)

    print(f"Data shape: {df.shape}")

    print("Adding technical indicators & regime features...")
    df = add_technical_indicators(df)
    print(f"After indicators: {df.shape}")

    regimes = df.apply(classify_market_regime, axis=1)
    vol_regimes = df.apply(classify_volatility_regime, axis=1)
    print(f"\nMarket Regime Distribution:")
    print(regimes.value_counts())
    print(f"\nVolatility Regime Distribution:")
    print(vol_regimes.value_counts())

    trade_allowed = df.apply(is_trading_allowed, axis=1)
    print(f"\nTrading Allowed: {trade_allowed.sum()} / {len(trade_allowed)} "
          f"({trade_allowed.mean()*100:.1f}% of bars)")

    print("\nPreparing sequences (scaler fit on TRAIN ONLY)...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, target_scaler, feature_names, mask_test, base_test) = prepare_data(
        df, config.LOOKBACK_WINDOW, config.PREDICTION_HORIZON, config.TRAIN_SPLIT
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {feature_names}")

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    # drop_last=True prevents BatchNorm crash on batch size 1
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    model = StockPredictor(
        input_size=len(feature_names),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL
    )

    print("\nStarting training...")
    model, history = train_model(model, train_loader, val_loader, config)

    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, target_scaler, config,
                             trade_mask=mask_test, baseline_returns=base_test,
                             target=config.TARGET)

    metadata = {
        'ticker': config.TICKER,
        'interval': config.INTERVAL,
        'lookback_window': config.LOOKBACK_WINDOW,
        'features': feature_names,
        'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                    for k, v in results.items() if k not in ['predictions', 'actuals']},
        'regime_distribution': {str(k): int(v) for k, v in regimes.value_counts().items()},
        'timestamp': datetime.now().isoformat()
    }

    with open('models/nn_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nModel saved to:", config.MODEL_SAVE_PATH)
    print("Metadata saved to: models/nn_metadata.json")

    return model, scaler, target_scaler, results, df


if __name__ == "__main__":
    model, scaler, target_scaler, results, df = main()