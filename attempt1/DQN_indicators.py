import os, sys, warnings, random, collections
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ---------- TF/Keras & GPU Setup ----------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU for stability unless GPU env is confirmed
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42); random.seed(42)

import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(42)

# --- GPU VERIFICATION ---
print("[info] Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[info] SUCCESS: {len(gpus)} GPU(s) detected. Training will be accelerated.")
    except RuntimeError as e:
        print(f"[error] Error during GPU setup: {e}")
else:
    print("[info] WARNING: No GPU detected. Script will run on CPU.")

# ========== Config ==========
DB_URL = os.getenv(
    "DB_URL", "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4"
)
OUTDIR_DEFAULT = "ml_outputs_dqn_paper"

# ========== State Space: From your calculate_indicators.py script ==========
STATE_FEATURES = [
    "rsi14", "sma20", "sma50", "obv", "mom", "stoch_k", "macd", "macd_hist", "cci", "adx", "trix", "roc",
    "sar", "tema", "trima", "wma", "dema", "mfi", "cmo", "stochrsi_k",
    "ultosc", "bop", "atr"
]
# We only need the indicators used by the agent for the state
AGENT_STATE_FEATURES = [f for f in STATE_FEATURES if f != 'sma50']


# ========== DQN Agent ==========
class DQNAgent:
    def __init__(self, state_size: int, action_size: int = 2, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001):
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
        model = tf.keras.models.Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        return model

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def choose_action(self, state, greedy=False):
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

# ========== Utilities ==========
def load_data(engine, symbol: str) -> pd.DataFrame:
    """ Loads all indicators mentioned in the paper, plus extra ones for TA baselines. """
    features_str = ", ".join([f"i.{f}" for f in STATE_FEATURES])
    with engine.begin() as conn:
        q = text(f"""
            SELECT i.trade_date, b.close, {features_str}
            FROM indicators_daily i
            JOIN daily_bars b ON i.symbol = b.symbol AND i.trade_date = b.trade_date
            WHERE i.symbol=:s ORDER BY i.trade_date
        """)
        df = pd.read_sql(q, conn, params={"s": symbol}, parse_dates=['trade_date'])
    df = df.set_index('trade_date').sort_index()
    # Engineer feature needed for one of the TA strategies
    df["sma20_rel"] = (df["sma20"] - df["sma50"]) / df["sma50"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def chrono_split_data(df, train_frac=0.7):
    n = len(df)
    split_idx = int(n * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

# ========== Main Runner ==========
def run(symbol="AAPL", episodes=50, outdir=OUTDIR_DEFAULT, batch_size=32):
    print(f">>> DQN Agent (Paper Methodology)  Py={sys.executable}", flush=True)
    print(f"[config] symbol={symbol} episodes={episodes}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    engine = create_engine(DB_URL, future=True)

    df = load_data(engine, symbol)
    if df.empty:
        raise RuntimeError(f"No data loaded for symbol {symbol}. Check database.")
        
    train_df, test_df = chrono_split_data(df)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(train_df[AGENT_STATE_FEATURES])
    X_te = scaler.transform(test_df[AGENT_STATE_FEATURES])
    
    log_returns_tr = np.log(train_df['close'] / train_df['close'].shift(1)).fillna(0).values

    print(f"[data] {symbol} train_samples={len(X_tr)} test_samples={len(X_te)}", flush=True)

    state_size = X_tr.shape[1]
    agent = DQNAgent(state_size=state_size)

    # -------- Training --------
    print("[info] Starting DQN training based on paper methodology...", flush=True)
    n = len(X_tr)
    for ep in range(1, episodes + 1):
        state = X_tr[0].reshape(1, -1)
        
        for t in trange(n - 1, desc=f"Episode {ep}/{episodes}"):
            action = agent.choose_action(state)
            next_state = X_tr[t+1].reshape(1, -1)
            
            reward = log_returns_tr[t] if action == 1 else 0.0
            done = t == n - 2

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            agent.replay(batch_size)
        
        print(f"[train] ep={ep}/{episodes}  eps={agent.epsilon:.3f}", flush=True)

    # -------- Evaluation --------
    print("\n[info] Evaluating on test data...", flush=True)
    actions = np.zeros(len(X_te), dtype=int)
    for t in range(len(X_te)):
        s = X_te[t].reshape(1, -1)
        actions[t] = agent.choose_action(s, greedy=True)
    
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

    # --- Simulate portfolio & calculate PnL and Sharpe Ratio ---
    close_prices = test_df['close'].values
    sim_df = pd.DataFrame(index=test_df.index)
    sim_df['close_price'] = close_prices
    
    ta_strats = ta_actions_from_df(test_df)
    strategies = {'DQN_Paper': actions, **ta_strats, 'Buy_and_Hold': np.ones_like(actions)}
    initial_cash = close_prices[0]

    for name, acts in strategies.items():
        portfolio_values = []
        cash = initial_cash
        shares = 0
        for i in range(len(close_prices)):
            price_today = close_prices[i]
            action_today = acts[i]
            if action_today == 1 and shares == 0:
                shares = cash / price_today
                cash = 0
            elif action_today == 0 and shares > 0:
                cash = shares * price_today
                shares = 0
            current_value = cash + shares * price_today
            portfolio_values.append(current_value)
        sim_df[f'portfolio_value_{name}'] = portfolio_values
    
    sim_df['pnl_DQN_Paper'] = sim_df['portfolio_value_DQN_Paper'] - initial_cash
    sim_path = os.path.join(outdir, f"full_simulation_{symbol}.csv")
    sim_df.to_csv(sim_path)
    print(f"[save] Full simulation log (with PnL) saved to: {sim_path}", flush=True)

    # --- Final Performance Metrics ---
    metrics = []
    for name in strategies.keys():
        final_value = sim_df[f'portfolio_value_{name}'].iloc[-1]
        returns = sim_df[f'portfolio_value_{name}'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        metrics.append({
            'strategy': name,
            'final_portfolio_value': final_value,
            'total_return_%': (final_value / initial_cash - 1) * 100,
            'sharpe_ratio_annualized': sharpe_ratio
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(outdir, f"comparison_metrics_{symbol}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] Performance metrics (with Sharpe Ratio) saved to: {metrics_path}", flush=True)
    
    # --- Human-Readable Trading Log for DQN Agent ---
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
    }, index=test_df.index)
    
    log_path = os.path.join(outdir, f"trading_log_{symbol}.csv")
    log_df.to_csv(log_path)
    print(f"[save] Trading log saved to: {log_path}", flush=True)
    
    print("\n>>> done", flush=True)

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DQN Trading Agent based on Paper Methodology")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    run(symbol=args.symbol, episodes=args.episodes, outdir=args.outdir)
