"""
DQN Trader — Reinforcement Learning Agent for Stock Trading

FIXES APPLIED:
- __init__ uses `is not None` (works with numpy arrays)
- Short-selling accounting fixed (COVER_ALL, stop-loss, liquidation)
- Synthetic predictions = next_price * noise (learnable signal)
- Transaction costs + slippage added
- Out-of-sample evaluation (holdout test segment)
- Sharpe computed from equity curve, not per-trade
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import json
import os
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class DQNConfig:
    STATE_SIZE = 13
    ACTION_SIZE = 9

    POSITION_SIZES = {
        0: 0.0, 1: 0.25, 2: 0.50, 3: 1.00,
        4: 1.00, 5: 0.25, 6: 0.50, 7: 1.00, 8: 1.00
    }

    ACTION_NAMES = {
        0: "HOLD", 1: "BUY_25%", 2: "BUY_50%", 3: "BUY_100%",
        4: "SELL_ALL", 5: "SHORT_25%", 6: "SHORT_50%", 7: "SHORT_100%", 8: "COVER_ALL"
    }

    HIDDEN_SIZES = [256, 128, 64]
    DROPOUT = 0.2
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    TAU = 1e-3
    MEMORY_SIZE = 100000

    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995

    EPISODES = 500
    MAX_STEPS_PER_EPISODE = 2000
    # Sharpe annualization factor. MUST match the bar interval or the number is
    # meaningless: 252 for daily bars, 252*390 for 1-minute bars.
    BARS_PER_YEAR = 252

    # Risk management
    MAX_POSITION_SIZE = 1.0
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.04
    MAX_HOLDING_TIME = 30

    # Transaction costs (per trade, as fraction of trade value)
    COMMISSION_PCT = 0.0005      # 5 bps per side
    SLIPPAGE_PCT = 0.0002        # 2 bps slippage
    TOTAL_COST_PER_TRADE = COMMISSION_PCT + SLIPPAGE_PCT

    # Regime filter
    ONLY_TRADE_IN_CHOP = True
    ONLY_TRADE_LOW_VOL = True

    # Train/test split for evaluation
    TRAIN_TEST_SPLIT = 0.8       # 80% train, 20% held out for final eval

    # Account
    INITIAL_CASH = 10000.0       # override with --cash
    SEED = 42                    # override with --seed (changes synthetic data
                                 # AND network init, so runs differ)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "models/dqn_trader.pt"


# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(DQNConfig.DEVICE)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(DQNConfig.DEVICE)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(DQNConfig.DEVICE)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(DQNConfig.DEVICE)
        dones = torch.FloatTensor(np.array([e.done for e in batch])).to(DQNConfig.DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN NETWORK
# =============================================================================
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128, 64], dropout=0.2):
        super(DQNetwork, self).__init__()
        layers = []
        prev_size = state_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_size = h
        self.feature = nn.Sequential(*layers)

        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(prev_size // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(prev_size // 2, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================
class TradingEnvironment:
    """
    Fixed environment with:
    - Correct short-selling accounting
    - Transaction costs + slippage
    - Proper numpy array handling in __init__
    """

    def __init__(self, config, price_data=None, predictions=None, 
                 regime_data=None, volatility_data=None, start_idx=0, end_idx=None):
        self.config = config

        # FIX: Use `is not None` instead of `or` to handle numpy arrays
        if price_data is not None:
            self.price_data = price_data
        else:
            self.price_data = self._generate_synthetic_data()

        if predictions is not None:
            self.predictions = predictions
        else:
            self.predictions = self._generate_synthetic_predictions()

        if regime_data is not None:
            self.regime_data = regime_data
        else:
            self.regime_data = self._generate_synthetic_regime()

        if volatility_data is not None:
            self.volatility_data = volatility_data
        else:
            self.volatility_data = self._generate_synthetic_volatility()

        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.price_data) - 1
        self.reset()

    def _generate_synthetic_data(self, n=5000):
        np.random.seed(self.config.SEED)
        returns = np.random.normal(0.0001, 0.001, n)
        prices = 450.0 * np.exp(np.cumsum(returns))
        return prices

    def _generate_synthetic_predictions(self):
        """FIX: Predict next price, not current price."""
        np.random.seed(self.config.SEED + 1)
        noise = np.random.normal(0, 0.0005, len(self.price_data))
        # Shift forward by 1 so predicted_next is actually the next price + noise
        shifted = np.roll(self.price_data, -1)
        shifted[-1] = self.price_data[-1]  # Last element has no next
        return shifted * (1 + noise)

    def _generate_synthetic_regime(self):
        np.random.seed(self.config.SEED + 2)
        return {
            'is_chop': np.random.choice([0, 1], size=len(self.price_data), p=[0.4, 0.6]),
            'is_trending': np.random.choice([0, 1], size=len(self.price_data), p=[0.7, 0.3]),
            'is_low_vol': np.random.choice([0, 1], size=len(self.price_data), p=[0.5, 0.5]),
            'is_high_vol': np.random.choice([0, 1], size=len(self.price_data), p=[0.8, 0.2]),
        }

    def _generate_synthetic_volatility(self):
        np.random.seed(self.config.SEED + 3)
        return np.random.uniform(0.0005, 0.002, len(self.price_data))

    def reset(self):
        self.current_step = self.start_idx + 30
        self.position = 0
        self.position_size = 0.0
        self.purchase_price = 0.0
        self.entry_prices = []
        self.entry_sizes = []
        self.holding_time = 0
        self.cash = self.config.INITIAL_CASH
        self.portfolio_value = self.cash
        self.trades = []
        self.blocked_trades = 0
        self.equity_curve = [self.cash]
        return self._get_state()

    def _get_regime(self):
        idx = min(self.current_step, len(self.price_data) - 1)
        return {
            'is_chop': self.regime_data.get('is_chop', [1]*len(self.price_data))[idx],
            'is_trending': self.regime_data.get('is_trending', [0]*len(self.price_data))[idx],
            'is_low_vol': self.regime_data.get('is_low_vol', [1]*len(self.price_data))[idx],
            'is_high_vol': self.regime_data.get('is_high_vol', [0]*len(self.price_data))[idx],
        }

    def _can_trade(self):
        regime = self._get_regime()
        if self.config.ONLY_TRADE_IN_CHOP and not regime['is_chop']:
            return False
        if self.config.ONLY_TRADE_LOW_VOL and regime['is_high_vol']:
            return False
        return True

    def _avg_entry_price(self):
        if not self.entry_prices:
            return 0.0
        total_value = sum(p * s for p, s in zip(self.entry_prices, self.entry_sizes))
        total_size = sum(self.entry_sizes)
        return total_value / total_size if total_size > 0 else 0.0

    def _get_state(self):
        # Clamp: the test env is constructed with end_idx == len(price_data),
        # so current_step can land exactly on the array length after the final
        # step. Indexing that raised IndexError before the episode could end.
        step = min(self.current_step, len(self.price_data) - 1)
        current_price = self.price_data[step]
        predicted_next = self.predictions[step]
        regime = self._get_regime()

        if step >= 10:
            recent = self.price_data[step-10:step]
            recent_returns = np.diff(recent) / recent[:-1]
            volatility = np.std(recent_returns)
        else:
            volatility = 0.001

        avg_entry = self._avg_entry_price()

        if self.position != 0 and avg_entry > 0:
            if self.position == 1:
                unrealized_pnl = (current_price - avg_entry) / avg_entry
            else:
                unrealized_pnl = (avg_entry - current_price) / avg_entry
        else:
            unrealized_pnl = 0.0

        predicted_direction = 1 if predicted_next > current_price else -1

        state = np.array([
            predicted_next / current_price - 1,
            current_price / 500.0,
            avg_entry / 500.0 if avg_entry > 0 else 0.0,
            float(self.position),
            self.position_size,
            self.holding_time / self.config.MAX_HOLDING_TIME,
            unrealized_pnl,
            float(predicted_direction),
            volatility * 100,
            float(regime['is_chop']),
            float(regime['is_trending']),
            float(regime['is_low_vol']),
            float(regime['is_high_vol'])
        ], dtype=np.float32)
        return state

    def _fee(self, notional):
        """
        Cost of a fill, as a fraction of NOTIONAL traded.

        This must never be applied to P&L. The previous version did
        `net_profit = gross_profit * (1 - cost)`, which shrinks a loss instead
        of deepening it — costs were paying you when a trade went against you.
        """
        return abs(notional) * self.config.TOTAL_COST_PER_TRADE

    def _equity(self, price):
        """
        Mark-to-market equity. Longs add share value, shorts subtract it
        (the sale proceeds are already sitting in cash), so this is correct on
        both sides and never reports the old phantom -100% drawdown.
        """
        if self.position == 0:
            return self.cash
        total_shares = sum(self.entry_sizes)
        return self.cash + self.position * total_shares * price

    def _close_position(self, price, reason):
        """
        Flatten the position at `price`, book the trade, return net P&L pct.
        One code path for SELL_ALL, COVER_ALL, stop-loss, take-profit and
        end-of-episode liquidation, so those can't drift apart again.
        """
        if self.position == 0:
            return 0.0

        avg_entry = self._avg_entry_price()
        total_shares = sum(self.entry_sizes)
        notional = total_shares * price
        fee = self._fee(notional)

        if self.position == 1:
            self.cash += notional - fee          # sell the shares
            gross_pnl = (price - avg_entry) / avg_entry
        else:
            self.cash -= notional + fee          # buy them back
            gross_pnl = (avg_entry - price) / avg_entry

        entry_notional = total_shares * avg_entry
        # Entry fee was already charged in cash at entry; charge the exit fee
        # here against the position's cost basis.
        net_pnl = gross_pnl - fee / entry_notional

        self.trades.append({
            'type': 'LONG' if self.position == 1 else 'SHORT',
            'entry': avg_entry, 'exit': price,
            'gross_pnl': gross_pnl, 'net_pnl': net_pnl,
            'holding_time': self.holding_time,
            'size': self.position_size, 'exit_reason': reason
        })

        self.position = 0
        self.position_size = 0.0
        self.entry_prices = []
        self.entry_sizes = []
        self.holding_time = 0
        return net_pnl

    def _open_or_add(self, direction, size_mult, price):
        """
        Add `size_mult` of equity in notional, capped by MAX_POSITION_SIZE.
        Sized off EQUITY, not cash: shorting credits cash, so sizing off cash
        would let each successive short add grow instead of shrink.
        """
        equity = self._equity(price)
        if equity <= 0:
            return False

        current_notional = sum(self.entry_sizes) * price
        notional = min(size_mult * equity,
                       self.config.MAX_POSITION_SIZE * equity - current_notional)
        if direction > 0:
            notional = min(notional, self.cash)   # no margin on longs
        if notional <= 1e-6:
            return False

        shares = notional / price
        fee = self._fee(notional)

        if direction > 0:
            self.cash -= notional + fee           # buy
        else:
            self.cash += notional - fee           # short sale credits proceeds

        if self.position == 0:
            self.position = direction
            self.holding_time = 0
        self.entry_prices.append(price)
        self.entry_sizes.append(shares)
        self.position_size = min(self.position_size + size_mult, 1.0)
        return True

    def step(self, action):
        current_price = self.price_data[self.current_step]
        reward = 0.0
        done = False
        info = {'action': action, 'price': current_price, 'blocked': False}

        # Regime filter
        if action != 0 and not self._can_trade():
            self.blocked_trades += 1
            info['blocked'] = True
            info['block_reason'] = 'regime_filter'
            reward -= 0.1
            action = 0

        size_mult = self.config.POSITION_SIZES.get(action, 0.0)

        # --- entries and exits, all through the shared helpers ---------------
        if action in [1, 2, 3] and self.position in [0, 1]:        # BUY / add
            if self._open_or_add(1, size_mult, current_price):
                info['trade'] = f'BUY_{int(size_mult*100)}%'

        elif action == 4 and self.position == 1:                    # SELL_ALL
            net_pnl = self._close_position(current_price, 'SELL_ALL')
            reward = net_pnl * 100
            info['trade'] = 'SELL_ALL'
            info['net_pnl'] = net_pnl

        elif action in [5, 6, 7] and self.position in [0, -1]:      # SHORT / add
            if self._open_or_add(-1, size_mult, current_price):
                info['trade'] = f'SHORT_{int(size_mult*100)}%'

        elif action == 8 and self.position == -1:                   # COVER_ALL
            net_pnl = self._close_position(current_price, 'COVER_ALL')
            reward = net_pnl * 100
            info['trade'] = 'COVER_ALL'
            info['net_pnl'] = net_pnl

        elif action == 0:
            if self.position != 0:
                self.holding_time += 1
                if self.holding_time > self.config.MAX_HOLDING_TIME:
                    reward -= 0.5

        # --- risk management: identical treatment for longs and shorts -------
        if self.position != 0 and self.entry_prices:
            avg_entry = self._avg_entry_price()
            if self.position == 1:
                pnl_pct = (current_price - avg_entry) / avg_entry
            else:
                pnl_pct = (avg_entry - current_price) / avg_entry

            exit_reason = None
            if pnl_pct <= -self.config.STOP_LOSS_PCT:
                exit_reason = 'STOP_LOSS'
            elif pnl_pct >= self.config.TAKE_PROFIT_PCT:
                exit_reason = 'TAKE_PROFIT'

            if exit_reason:
                net_pnl = self._close_position(current_price, exit_reason)
                reward = net_pnl * 100 - (1.0 if exit_reason == 'STOP_LOSS' else 0.0)
                info['forced_exit'] = exit_reason
                info['net_pnl'] = net_pnl

        # --- mark to market ---------------------------------------------------
        self.portfolio_value = self._equity(current_price)
        self.equity_curve.append(self.portfolio_value)

        self.current_step += 1
        if self.current_step >= self.end_idx:
            done = True
            if self.position != 0:
                final_price = self.price_data[self.current_step - 1]
                self._close_position(final_price, 'END_OF_DATA')
                self.portfolio_value = self.cash
                self.equity_curve[-1] = self.cash

        next_state = self._get_state()
        return next_state, reward, done, info

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_trade_summary(self):
        if not self.trades:
            return {"total_trades": 0, "blocked_trades": self.blocked_trades}

        net_pnls = [t['net_pnl'] for t in self.trades]
        winning = [p for p in net_pnls if p > 0]
        losing = [p for p in net_pnls if p <= 0]

        # Compute Sharpe from equity curve (not per-trade)
        equity = np.array(self.equity_curve)
        if len(equity) > 1:
            equity_returns = np.diff(equity) / equity[:-1]
            if np.std(equity_returns) > 1e-8:
                sharpe = np.mean(equity_returns) / np.std(equity_returns)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "total_trades": len(self.trades),
            "blocked_trades": self.blocked_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(net_pnls) * 100 if net_pnls else 0,
            "avg_net_pnl": np.mean(net_pnls) * 100,
            "best_trade": max(net_pnls) * 100,
            "worst_trade": min(net_pnls) * 100,
            "total_return": (self.portfolio_value / self.config.INITIAL_CASH - 1) * 100,
            # Reference point: what you'd have made holding the whole segment.
            # Without this a positive return says nothing about the strategy.
            "buy_and_hold_return": float(
                (self.price_data[min(self.current_step, len(self.price_data) - 1)] /
                 self.price_data[self.start_idx + 30] - 1) * 100),
            "sharpe_equity": sharpe,
            "sharpe_annualized": sharpe * np.sqrt(self.config.BARS_PER_YEAR),
            "max_drawdown": float(
                (equity / np.maximum.accumulate(equity) - 1).min() * 100),
            "avg_position_size": np.mean([t.get('size', 1.0) for t in self.trades]) * 100,
            "total_costs_pct": self.config.TOTAL_COST_PER_TRADE * 2 * 100  # round-trip
        }


# =============================================================================
# DQN AGENT
# =============================================================================
class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        self.q_network = DQNetwork(
            config.STATE_SIZE, config.ACTION_SIZE, 
            config.HIDDEN_SIZES, config.DROPOUT
        ).to(self.device)

        self.target_network = DQNetwork(
            config.STATE_SIZE, config.ACTION_SIZE,
            config.HIDDEN_SIZES, config.DROPOUT
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )

        self.memory = ReplayBuffer(config.MEMORY_SIZE)
        self.epsilon = config.EPSILON_START

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.ACTION_SIZE)

        # Dropout must be OFF for greedy selection, otherwise the "greedy"
        # policy is randomized and out-of-sample results are noise.
        was_training = self.q_network.training
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        if was_training:
            self.q_network.train()
        return action

    def learn(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.BATCH_SIZE)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.GAMMA * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update()
        return loss.item()

    def _soft_update(self):
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.TAU * local_param.data + (1.0 - self.config.TAU) * target_param.data
            )

    def decay_epsilon(self):
        self.epsilon = max(self.config.EPSILON_MIN, self.epsilon * self.config.EPSILON_DECAY)

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# =============================================================================
# TRAINING LOOP WITH OUT-OF-SAMPLE EVALUATION
# =============================================================================
def train_dqn(config=None, price_data=None, predictions=None, 
              regime_data=None, volatility_data=None):
    config = config or DQNConfig()
    os.makedirs("models", exist_ok=True)

    print("="*60)
    print("DQN TRADER — With Position Sizing & Regime Filters")
    print("="*60)

    # Split data: 80% train, 20% held out for final evaluation
    n = len(price_data) if price_data is not None else 5000
    split_idx = int(n * config.TRAIN_TEST_SPLIT)

    train_env = TradingEnvironment(config, price_data, predictions, 
                                   regime_data, volatility_data, 
                                   start_idx=0, end_idx=split_idx)

    # Test env uses the held-out portion
    test_env = TradingEnvironment(config, price_data, predictions,
                                  regime_data, volatility_data,
                                  start_idx=split_idx, end_idx=n)

    agent = DQNAgent(config)

    episode_rewards = []
    episode_portfolios = []

    print(f"\nTraining on {config.DEVICE}")
    print(f"Action space: {config.ACTION_SIZE} actions (with position sizing)")
    print(f"State size: {config.STATE_SIZE} features")
    print(f"Regime Filter: chop_only={config.ONLY_TRADE_IN_CHOP}, low_vol={config.ONLY_TRADE_LOW_VOL}")
    print(f"Costs: {config.TOTAL_COST_PER_TRADE*100:.2f}% per side")
    print(f"Train segment: 0-{split_idx}, Test segment: {split_idx}-{n}")
    print(f"Starting cash: ${config.INITIAL_CASH:,.2f} | Seed: {config.SEED}")
    print("-"*60)

    for episode in range(config.EPISODES):
        state = train_env.reset()
        episode_reward = 0

        for step in range(config.MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = train_env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        episode_portfolios.append(train_env.get_portfolio_value())

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_portfolio = np.mean(episode_portfolios[-50:])
            print(f"Episode {episode+1}/{config.EPISODES} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Portfolio: ${avg_portfolio:,.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    agent.save(config.MODEL_SAVE_PATH)

    # === OUT-OF-SAMPLE EVALUATION ===
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE EVALUATION (held-out test segment)")
    print("="*60)
    state = test_env.reset()
    done = False
    while not done:
        action = agent.select_action(state, training=False)
        state, _, done, info = test_env.step(action)

    summary = test_env.get_trade_summary()
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    metadata = {
        'episodes': config.EPISODES,
        'seed': config.SEED,
        'initial_cash': config.INITIAL_CASH,
        'final_epsilon': agent.epsilon,
        'action_space': config.ACTION_NAMES,
        'training_stats': {
            'avg_reward': float(np.mean(episode_rewards)),
            'avg_portfolio': float(np.mean(episode_portfolios))
        },
        'evaluation': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in summary.items()},
        'timestamp': datetime.now().isoformat()
    }

    meta_path = config.MODEL_SAVE_PATH.replace('.pt', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {config.MODEL_SAVE_PATH}")
    print(f"Metadata saved to: {meta_path}")

    return agent, test_env, episode_rewards


# =============================================================================
# INFERENCE API
# =============================================================================
def predict_action(agent, predicted_next_price, current_price, 
                   purchase_price=0.0, position=0, position_size=0.0,
                   holding_time=0, volatility=0.001,
                   is_chop=1, is_trending=0, is_low_vol=1, is_high_vol=0):
    predicted_direction = 1 if predicted_next_price > current_price else -1

    if position != 0 and purchase_price > 0:
        if position == 1:
            unrealized_pnl = (current_price - purchase_price) / purchase_price
        else:
            unrealized_pnl = (purchase_price - current_price) / purchase_price
    else:
        unrealized_pnl = 0.0

    state = np.array([
        predicted_next_price / current_price - 1,
        current_price / 500.0,
        purchase_price / 500.0 if purchase_price > 0 else 0.0,
        float(position),
        position_size,
        holding_time / 30.0,
        unrealized_pnl,
        float(predicted_direction),
        volatility * 100,
        float(is_chop),
        float(is_trending),
        float(is_low_vol),
        float(is_high_vol)
    ], dtype=np.float32)

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.q_network(state_tensor)
        action = q_values.argmax(dim=1).item()
        confidence = torch.softmax(q_values, dim=1).max().item()

    action_name = DQNConfig.ACTION_NAMES.get(action, "UNKNOWN")
    size_fraction = DQNConfig.POSITION_SIZES.get(action, 0.0)

    return action, action_name, confidence, size_fraction


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the DQN trader (simulation only)")
    parser.add_argument("--cash", type=float, default=DQNConfig.INITIAL_CASH,
                        help="starting simulated cash, e.g. 500")
    parser.add_argument("--seed", type=int, default=DQNConfig.SEED,
                        help="changes synthetic data and network init")
    parser.add_argument("--episodes", type=int, default=DQNConfig.EPISODES)
    parser.add_argument("--tag", type=str, default=None,
                        help="suffix for saved files, so parallel runs don't "
                             "overwrite each other (defaults to the seed)")
    parser.add_argument("--from-predictor", action="store_true",
                        help="train nn_stock_predictor first and feed its "
                             "out-of-sample forecasts to the DQN")
    args = parser.parse_args()

    cfg = DQNConfig()
    cfg.INITIAL_CASH = args.cash
    cfg.SEED = args.seed
    cfg.EPISODES = args.episodes
    tag = args.tag if args.tag is not None else f"seed{args.seed}"
    cfg.MODEL_SAVE_PATH = f"models/dqn_trader_{tag}.pt"
    cfg.DEVICE = DQNConfig.DEVICE
    _seed_everything(args.seed)

    price_data = predictions = regime_data = None
    if args.from_predictor:
        # Connect the two modules without needing a separate pipeline file.
        import nn_stock_predictor as N
        price_data, predictions, regime_data, _ = N.train_and_forecast(seed=args.seed)

    agent, env, rewards = train_dqn(cfg, price_data, predictions, regime_data)