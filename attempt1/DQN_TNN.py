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
from keras import Input
from keras.models import Sequential
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
OUTDIR_DEFAULT = "ml_outputs_dqn_TNN"

# ========== State Space: From your calculate_indicators.py script ==========
STATE_FEATURES = [
    "rsi14", "sma20", "sma50", "obv", "mom", "stoch_k", "macd", "macd_hist", "cci", "adx", "trix", "roc",
    "sar", "tema", "trima", "wma", "dema", "mfi", "cmo", "stochrsi_k",
    "ultosc", "bop", "atr"
]
AGENT_STATE_FEATURES = [f for f in STATE_FEATURES if f != 'sma50']


# ========== Advanced DQN Agent with Target Network ==========
class DQNAgent:
    def __init__(self, state_size: int, action_size: int = 2):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters from the Paper (Table 3)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 1e-2
        self.memory = collections.deque(maxlen=100000) # Larger buffer size
        self.batch_size = 128 # Larger batch size
        self.target_update_interval = 1000 # As per the paper
        self.train_step_counter = 0

        # Online model learns every step, target model provides stable targets
        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # Initialize target model weights

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))
        return model
        
    def update_target_model(self):
        """Copy weights from the online model to the target model."""
        self.target_model.set_weights(self.online_model.get_weights())

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def choose_action(self, state, greedy=False):
        if (not greedy) and (np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_size)
        q = self.online_model.predict(state, verbose=0)
        return int(np.argmax(q[0]))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.vstack([s for (s, _, _, _, _) in minibatch])
        next_states = np.vstack([ns for (_, _, _, ns, _) in minibatch])
        
        # Use the ONLINE model to predict current Q-values
        q_curr = self.online_model.predict(states, verbose=0)
        # Use the TARGET model to predict future Q-values for stability
        q_next = self.target_model.predict(next_states, verbose=0)

        for i, (s, a, r, s_next, done) in enumerate(minibatch):
            if done:
                target = r
            else:
                # Bellman equation using the stable target network
                target = r + self.gamma * np.max(q_next[i])
            q_curr[i, a] = target

        # Train the ONLINE model on the updated targets
        self.online_model.fit(states, q_curr, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Periodically update the target network
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_interval == 0:
            self.update_target_model()
            print(f"    [info] Target network updated at step {self.train_step_counter}")


# ========== Utilities (Same as before) ==========
def load_data(engine, symbol: str) -> pd.DataFrame:
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
    df["sma20_rel"] = (df["sma20"] - df["sma50"]) / df["sma50"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def chrono_split_data(df, train_frac=0.7):
    n = len(df)
    split_idx = int(n * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

# ========== Main Runner ==========
def run(symbol="AAPL", episodes=50, outdir=OUTDIR_DEFAULT):
    print(f">>> Advanced DQN Agent (Paper Methodology)  Py={sys.executable}", flush=True)
    print(f"[config] symbol={symbol} episodes={episodes}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    engine = create_engine(DB_URL, future=True)

    df = load_data(engine, symbol)
    train_df, test_df = chrono_split_data(df)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(train_df[AGENT_STATE_FEATURES])
    X_te = scaler.transform(test_df[AGENT_STATE_FEATURES])
    
    log_returns_tr = np.log(train_df['close'] / train_df['close'].shift(1)).fillna(0).values

    print(f"[data] {symbol} train_samples={len(X_tr)} test_samples={len(X_te)}", flush=True)

    state_size = X_tr.shape[1]
    agent = DQNAgent(state_size=state_size)

    # -------- Training --------
    print("[info] Starting Advanced DQN training...", flush=True)
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
            agent.replay()
        
        print(f"[train] ep={ep}/{episodes}  eps={agent.epsilon:.3f}", flush=True)

    # -------- Evaluation (Same as before) --------
    print("\n[info] Evaluating on test data...", flush=True)
    actions = np.zeros(len(X_te), dtype=int)
    for t in range(len(X_te)):
        s = X_te[t].reshape(1, -1)
        actions[t] = agent.choose_action(s, greedy=True)
    
    def ta_actions_from_df(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        rsi_long = (df["rsi14"] >= 55) | (df["rsi14"] < 30)
        macd_long = (df["macd_hist"] > 0) & (df["macd_hist"].diff() > 0)
        sma_rel_ln = (df["sma20_rel"] > 0)
        stoch_long = (df["stoch_k"] >= 60)
        all_signals = rsi_long & macd_long & sma_rel_ln & stoch_long
        return { "TA_RSI": rsi_long.to_numpy().astype(int), "TA_MACD": macd_long.to_numpy().astype(int),
                 "TA_SMA_Crossover": sma_rel_ln.to_numpy().astype(int), "TA_Stochastic": stoch_long.to_numpy().astype(int),
                 "TA_All_Signals": all_signals.to_numpy().astype(int) }

    close_prices = test_df['close'].values
    sim_df = pd.DataFrame(index=test_df.index)
    sim_df['close_price'] = close_prices
    
    ta_strats = ta_actions_from_df(test_df)
    strategies = {'DQN_Advanced': actions, **ta_strats, 'Buy_and_Hold': np.ones_like(actions)}
    initial_cash = close_prices[0]

    for name, acts in strategies.items():
        portfolio_values = []
        cash, shares = initial_cash, 0
        for i in range(len(close_prices)):
            price_today, action_today = close_prices[i], acts[i]
            if action_today == 1 and shares == 0: shares, cash = cash / price_today, 0
            elif action_today == 0 and shares > 0: cash, shares = shares * price_today, 0
            portfolio_values.append(cash + shares * price_today)
        sim_df[f'portfolio_value_{name}'] = portfolio_values
    
    sim_df[f'pnl_DQN_Advanced'] = sim_df['portfolio_value_DQN_Advanced'] - initial_cash
    sim_path = os.path.join(outdir, f"full_simulation_{symbol}.csv")
    sim_df.to_csv(sim_path)
    print(f"[save] Full simulation log (with PnL) saved to: {sim_path}", flush=True)

    metrics = []
    for name in strategies.keys():
        final_value = sim_df[f'portfolio_value_{name}'].iloc[-1]
        returns = sim_df[f'portfolio_value_{name}'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        metrics.append({ 'strategy': name, 'final_portfolio_value': final_value,
                         'total_return_%': (final_value / initial_cash - 1) * 100,
                         'sharpe_ratio_annualized': sharpe_ratio })

    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(outdir, f"comparison_metrics_{symbol}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] Performance metrics (with Sharpe Ratio) saved to: {metrics_path}", flush=True)
    
    trade_actions = []
    if len(actions):
        trade_actions.append("BUY" if actions[0] == 1 else "HOLD")
        for i in range(1, len(actions)):
            if actions[i-1] == 0 and actions[i] == 1: trade_actions.append("BUY")
            elif actions[i-1] == 1 and actions[i] == 0: trade_actions.append("SELL")
            else: trade_actions.append("HOLD")
    
    log_df = pd.DataFrame({ "close_price": close_prices, "position": ["LONG" if a == 1 else "FLAT" for a in actions],
                            "action": trade_actions }, index=test_df.index)
    log_path = os.path.join(outdir, f"trading_log_{symbol}.csv")
    log_df.to_csv(log_path)
    print(f"[save] Trading log saved to: {log_path}", flush=True)
    
    print("\n>>> done", flush=True)

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Advanced DQN Trading Agent based on Paper Methodology")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    run(symbol=args.symbol, episodes=args.episodes, outdir=args.outdir)
