import os, sys, warnings, random, collections
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ---------- TF/Keras & GPU Setup ----------
# The following line is commented out to allow TensorFlow to find and use the GPU.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42); random.seed(42)

import tensorflow as tf
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import trange # For progress bars

tf.random.set_seed(42)

# --- NEW: GPU VERIFICATION ---
# This block checks for an available GPU and prints the status.
print("[info] Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[info] SUCCESS: {len(gpus)} GPU(s) detected. Training will be accelerated.")
    except RuntimeError as e:
        print(f"[error] Error during GPU setup: {e}")
else:
    print("[info] WARNING: No GPU detected. Script will run on CPU.")
# --- END GPU VERIFICATION ---


# ========== Config ==========
DB_URL = os.getenv(
    "DB_URL",
    "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4"
)

DEFAULT_HORIZON = 21
LOOKBACK_WINDOW = 60 # Use last 60 days of price data as the state
OUTDIR_DEFAULT = "ml_outputs_raw_price"
TRANSACTION_COST = 0.001

# ========== Utilities ==========
def load_and_prepare_data(engine, symbol: str, horizon: int, window_size: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Loads price and indicator data, then creates the windowed states for the DQN agent.
    The full dataframe with indicators is returned for TA baseline comparisons.
    """
    with engine.begin() as conn:
        q = text("""
            SELECT
                i.trade_date, b.close, i.rsi14, i.macd_hist, i.stoch_k, i.sma20, i.sma50
            FROM indicators_daily i
            JOIN daily_bars b ON i.symbol = b.symbol AND i.trade_date = b.trade_date
            WHERE i.symbol=:s ORDER BY i.trade_date
        """)
        df = pd.read_sql(q, conn, params={"s": symbol}, parse_dates=['trade_date'])
    
    if df.empty:
        return pd.DataFrame(), np.array([]), np.array([])

    df = df.set_index('trade_date').sort_index()
    
    # Calculate forward returns (the target 'y')
    df['y'] = df['close'].shift(-horizon) / df['close'] - 1.0

    # Engineer feature needed for one of the TA strategies
    df["sma20_rel"] = (df["sma20"] - df["sma50"]) / df["sma50"]
    
    df = df.dropna()
    df["y"] = df["y"].clip(-0.2, 0.2)
    
    close_prices = df['close'].values
    y = df['y'].values
    
    # Create the windowed states from raw price data
    states = []
    for i in range(window_size, len(close_prices)):
        window = close_prices[i-window_size:i]
        # Normalize each window to be a series of percentage changes from its start
        normalized_window = window / window[0] - 1.0
        states.append(normalized_window)
    
    states = np.array(states)

    # Align the dataframe and y-values with the created states
    # The first state corresponds to the data at index `window_size - 1`
    start_idx = window_size - 1
    end_idx = start_idx + len(states)
    
    df_aligned = df.iloc[start_idx:end_idx]
    y_aligned = y[start_idx:end_idx]
    
    # Sanity check to ensure perfect alignment
    assert len(states) == len(df_aligned) == len(y_aligned), \
        f"Data alignment error: states={len(states)}, df={len(df_aligned)}, y={len(y_aligned)}"
    
    return df_aligned, states, y_aligned

def chrono_split_data(df, states, y, train_frac=0.7):
    n = len(states)
    split_idx = int(n * train_frac)
    
    X_tr, X_te = states[:split_idx], states[split_idx:]
    y_tr, y_te = y[:split_idx], y[split_idx:]
    df_tr, df_te = df.iloc[:split_idx], df.iloc[split_idx:]
    
    return df_tr, X_tr, y_tr, df_te, X_te, y_te

# ========== DQN Agent ==========
class DQNAgent:
    def __init__(self, state_size: int, action_size: int = 2,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.memory = collections.deque(maxlen=2000)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        return model

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def choose(self, state, greedy=False):
        if (not greedy) and (np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_size)
        q = self.model.predict(state, verbose=0)
        return int(np.argmax(q[0]))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.vstack([s for (s, _, _, _, _) in minibatch])
        next_states = np.vstack([ns for (_, _, _, ns, _) in minibatch])
        
        q_curr = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        for i, (_, a, r, _, done) in enumerate(minibatch):
            target = r if done else r + self.gamma * np.max(q_next[i])
            q_curr[i, a] = target

        self.model.fit(states, q_curr, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ========== Main Runner ==========
def run(symbol="AAPL",
        horizon=DEFAULT_HORIZON,
        window_size=LOOKBACK_WINDOW,
        episodes=20,
        outdir=OUTDIR_DEFAULT,
        train_frac=0.7,
        batch_size=32):
        
    print(f">>> DQN Agent (Raw Price)  Py={sys.executable}", flush=True)
    print(f"[config] symbol={symbol} horizon={horizon} window={window_size} episodes={episodes}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    engine = create_engine(DB_URL, future=True)

    df, states, y = load_and_prepare_data(engine, symbol, horizon, window_size)
    if df.empty:
        raise RuntimeError("No data loaded.")

    df_tr, X_tr, y_tr, df_te, X_te, y_te = chrono_split_data(df, states, y, train_frac)

    print(f"[data] {symbol}  train_samples={len(X_tr)}  test_samples={len(X_te)}  "
          f"full_range={df.index.min().date()}→{df.index.max().date()}", flush=True)

    state_size = X_tr.shape[1]
    agent = DQNAgent(state_size=state_size)

    # -------- Training --------
    print("[info] Starting training...", flush=True)
    n = len(X_tr)
    for ep in range(1, episodes + 1):
        state = X_tr[0].reshape(1, -1)
        prev_action = 0
        total_reward = 0.0
        for t in trange(n - 1, desc=f"Episode {ep}/{episodes}"):
            a = agent.choose(state)
            s_next = X_tr[t + 1].reshape(1, -1)
            
            reward = (y_tr[t] if a == 1 else 0.0) - (TRANSACTION_COST if a != prev_action else 0.0)
            done = (t == n - 2)

            agent.remember(state, a, reward, s_next, done)
            state = s_next
            prev_action = a
            total_reward += reward

            agent.replay(batch_size)

        print(f"[train] ep={ep}/{episodes}  eps={agent.epsilon:.3f}  total_reward={total_reward:.4f}", flush=True)

    # -------- Evaluation (greedy) --------
    print("\n[info] Evaluating on test data...", flush=True)
    actions = np.zeros(len(X_te), dtype=int)
    for t in range(len(X_te)):
        s = X_te[t].reshape(1, -1)
        actions[t] = agent.choose(s, greedy=True)
    
    # --- TA baselines on the SAME test window ---
    def ta_actions_from_df(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        rsi_long   = (df["rsi14"] >= 55) | (df["rsi14"] < 30)
        macd_long  = (df["macd_hist"] > 0) & (df["macd_hist"].diff() > 0)
        sma_rel_ln = (df["sma20_rel"] > 0)
        stoch_long = (df["stoch_k"] >= 60)
        all_signals = rsi_long & macd_long & sma_rel_ln & stoch_long
        ta = {
            "TA_RSI": rsi_long.to_numpy().astype(int),
            "TA_MACD": macd_long.to_numpy().astype(int),
            "TA_SMA_Crossover": sma_rel_ln.to_numpy().astype(int),
            "TA_Stochastic": stoch_long.to_numpy().astype(int),
            "TA_All_Signals": all_signals.to_numpy().astype(int)
        }
        return ta

    # --- Simulate portfolio value for all strategies ---
    close_prices = df_te['close'].values
    sim_df = pd.DataFrame(index=df_te.index)
    sim_df['close_price'] = close_prices
    
    # Add the TA strategies to the comparison
    ta_strats = ta_actions_from_df(df_te)
    strategies = {'RL_DQN_RawPrice': actions, **ta_strats, 'Buy_and_Hold': np.ones_like(actions)}
    
    initial_cash = close_prices[0]

    for name, acts in strategies.items():
        portfolio_values = []
        cash = initial_cash
        shares = 0
        for i in range(len(close_prices)):
            price_today = close_prices[i]
            action_today = acts[i]

            if action_today == 1 and shares == 0: # Buy signal
                shares = cash / price_today
                cash = 0
            elif action_today == 0 and shares > 0: # Sell signal
                cash = shares * price_today
                shares = 0
            
            current_value = cash + shares * price_today
            portfolio_values.append(current_value)
        sim_df[f'portfolio_value_{name}'] = portfolio_values
    
    sim_path = os.path.join(outdir, f"full_simulation_{symbol}_{horizon}d.csv")
    sim_df.to_csv(sim_path)
    print(f"[save] Full simulation log saved to: {sim_path}", flush=True)

    # --- Final Performance Metrics ---
    metrics = []
    for name in strategies.keys():
        final_value = sim_df[f'portfolio_value_{name}'].iloc[-1]
        returns = sim_df[f'portfolio_value_{name}'].pct_change().dropna()
        metrics.append({
            'strategy': name,
            'final_portfolio_value': final_value,
            'total_return_%': (final_value / initial_cash - 1) * 100,
            'sharpe_ratio_annualized': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(outdir, f"comparison_metrics_{symbol}_{horizon}d.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] Performance metrics saved to: {metrics_path}", flush=True)

    # --- Human-Readable Trading Log ---
    trade_actions = []
    if len(actions):
        trade_actions.append("BUY" if actions[0] == 1 else "HOLD")
        for i in range(1, len(actions)):
            if actions[i-1] == 0 and actions[i] == 1:
                trade_actions.append("BUY")
            elif actions[i-1] == 1 and actions[i] == 0:
                trade_actions.append("SELL")
            else:
                trade_actions.append("HOLD")
                
    log_df = pd.DataFrame({
        "close_price": close_prices,
        "position": ["LONG" if a == 1 else "FLAT" for a in actions],
        "action": trade_actions
    }, index=df_te.index)
    
    log_path = os.path.join(outdir, f"trading_log_{symbol}_{horizon}d.csv")
    log_df.to_csv(log_path)
    print(f"[save] Trading log saved to: {log_path}", flush=True)
    
    print("\n>>> done", flush=True)

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Full-size DQN Trading Agent from Raw Prices")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--window", type=int, default=LOOKBACK_WINDOW)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    try:
        run(symbol=args.symbol, horizon=args.horizon, window_size=args.window,
            episodes=args.episodes, outdir=args.outdir)
    except Exception as e:
        import traceback
        print("[error]", e, flush=True)
        traceback.print_exc()
        sys.exit(1)

