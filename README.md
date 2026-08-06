# S&P 500 Minute-Level AI Trading Stack

A simulation-ready deep learning pipeline for short-term stock prediction and reinforcement learning-based trading decisions. Built for educational purposes and resume demonstration — **no real money involved**.

Inspired by the **KAT Bobcat** (Krafer Agent Trader) architecture, which excels at short-term (~30 bar) predictions in low-volatility chop markets.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────┐
│  S&P 500 Data   │────▶│  NN Stock Predictor         │
│  (Minute Bars)  │     │  ├─ Predicts next close     │
└─────────────────┘     │  ├─ ADX regime detection    │
                        │  ├─ ATR/volume filters      │
                        │  └─ is_trading_allowed?     │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  DQN Trader                 │
                        │  ├─ Receives prediction     │
                        │  ├─ Receives regime flags   │
                        │  ├─ Decides action + size   │
                        │  └─ Blocked if bad regime   │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  API Caller (simulation)    │
                        └─────────────────────────────┘
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| [`nn_stock_predictor.py`](nn_stock_predictor.py) | ~620 | Deep bidirectional LSTM with attention for minute-level price prediction |
| [`dqn_trader.py`](dqn_trader.py) | ~720 | Dueling DQN agent that decides trade direction **and** position size |
| [`api_caller.py`](api_caller.py) | ~550 | Pluggable API wrapper for Alpaca, IBKR, or custom endpoints |

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy pandas scikit-learn yfinance
```

### 2. Train the predictor

```bash
python nn_stock_predictor.py
```

- Fetches SPY minute data (or generates synthetic data if unavailable)
- Adds technical indicators + regime features
- Trains a bidirectional LSTM with attention
- Saves model to `models/nn_stock_predictor.pt`

### 3. Train the DQN trader

```bash
python dqn_trader.py
```

- Simulates a trading environment with regime-aware filtering
- Trains a Dueling DQN with position sizing
- Saves model to `models/dqn_trader.pt`

### 4. (Optional) Connect to a broker

```python
from api_caller import APICaller

api = APICaller(provider="alpaca")
api.connect(api_key="YOUR_KEY", api_secret="YOUR_SECRET")
api.place_order("SPY", qty=10, side="buy")
```

---

## Key Design Decisions

### KAT Bobcat Principles

| Principle | Implementation |
|-----------|---------------|
| **~30 bar relevance** | `LOOKBACK_WINDOW = 30` in the NN; `MAX_HOLDING_TIME = 30` in the DQN |
| **Chop markets only** | ADX-based regime filter blocks trading when ADX > 20 |
| **Low volatility** | ATR percentile filter blocks trades when volatility > 70th percentile |
| **Avoid news spikes** | Volume percentile filter blocks trades when volume > 80th percentile |

### NN Predictor

- **Architecture**: 3-layer bidirectional LSTM + attention mechanism + residual-style FC layers
- **Optimizer**: Adam with ReduceLROnPlateau scheduling and gradient clipping
- **Features**: OHLCV + RSI, Bollinger position, SMA ratios, volume ratios, returns, volatility, ADX, ATR, regime flags
- **Split**: 50/50 train/test (with 80/20 train/val inside the training half)

### DQN Trader

- **State space**: 13 features including predicted return, position info, unrealized PnL, and regime flags
- **Action space**: 9 actions — HOLD, BUY (25/50/100%), SELL_ALL, SHORT (25/50/100%), COVER_ALL
- **Architecture**: Dueling DQN with Double DQN updates for stable Q-value estimation
- **Risk management**: 2% stop loss, 4% take profit, max 30-bar holding time
- **Regime integration**: Trades blocked (with penalty) in trending or high-vol conditions

---

## Resume Bullet Points

- Built a deep bidirectional LSTM with attention for minute-level S&P 500 price prediction, trained on a 50/50 split with Adam optimizer
- Implemented ADX-based market regime detection to identify chop vs trending conditions, filtering predictions to only trade in favorable low-volatility regimes
- Designed a Dueling DQN with Double DQN updates that outputs both trade direction and position size (25%/50%/100%), integrating real-time regime flags into the state space
- Added risk management layer with dynamic stop-loss/take-profit scaled to position size and a 30-bar maximum holding window inspired by short-term prediction architectures

---

## Project Structure

```
.
├── nn_stock_predictor.py    # Neural network predictor
├── dqn_trader.py            # Reinforcement learning agent
├── api_caller.py            # Broker API wrapper
├── models/                  # Saved models & metadata
│   ├── nn_stock_predictor.pt
│   ├── nn_metadata.json
│   ├── dqn_trader.pt
│   └── dqn_metadata.json
└── README.md
```

---

## Future Extensions

- [ ] **Ensemble predictions**: Train multiple NNs with different architectures and ensemble outputs
- [ ] **Multi-asset**: Expand from SPY to a basket of S&P 500 constituents
- [ ] **Walk-forward validation**: Replace static 50/50 split with rolling window backtesting
- [ ] **Live paper trading**: Connect `api_caller.py` to Alpaca paper account
- [ ] **Macro features**: Integrate VIX, yield curve, or sector rotation signals (selectively — too many features degrade performance)
- [ ] **Model tiers**: Follow Krafer's tiered approach (Bobcat → Panther → Lion → Tiger for longer horizons)

---

## Disclaimer

This project is for **educational and simulation purposes only**. No real capital is at risk. Past performance of any model does not guarantee future results. The KAT Bobcat reference is used for architectural inspiration only.
