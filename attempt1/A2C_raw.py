import os, sys, warnings, random
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
DEFAULT_HORIZON = 21
LOOKBACK_WINDOW = 60
OUTDIR_DEFAULT = "ml_outputs_a2c_raw_vs_ta"
TRANSACTION_COST = 0.0

# ========== A2C Agent ==========
class A2CAgent:
    def __init__(self, state_size: int, action_size: int = 2, gamma=0.99, lr=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_optimizer = Adam(learning_rate=self.lr)
        self.critic_optimizer = Adam(learning_rate=self.lr)

    def _build_actor(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(64, activation='relu')(state_input)
        dense2 = Dense(64, activation='relu')(dense1)
        action_probs = Dense(self.action_size, activation='softmax')(dense2)
        return Model(inputs=state_input, outputs=action_probs)

    def _build_critic(self):
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(64, activation='relu')(state_input)
        dense2 = Dense(64, activation='relu')(dense1)
        state_value = Dense(1, activation=None)(dense2)
        return Model(inputs=state_input, outputs=state_value)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action_probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def learn(self, state, action, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]
        
        with tf.GradientTape() as tape_critic, tf.GradientTape() as tape_actor:
            value = self.critic(state, training=True)
            next_value = self.critic(next_state, training=True)
            
            target = reward + self.gamma * next_value * (1 - int(done))
            advantage = target - value
            
            # Critic loss
            critic_loss = tf.square(advantage)
            
            # Actor loss
            action_probs = self.actor(state, training=True)
            action_one_hot = tf.one_hot([action], self.action_size)
            selected_action_prob = tf.reduce_sum(action_one_hot * action_probs)
            log_prob = tf.math.log(selected_action_prob + 1e-10)
            actor_loss = -log_prob * tf.stop_gradient(advantage)

        critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))


# ========== Utilities ==========
def load_and_prepare_data(engine, symbol: str, horizon: int, window_size: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with engine.begin() as conn:
        q = text("""
            SELECT i.trade_date, b.close, i.rsi14, i.macd_hist, i.stoch_k, i.sma20, i.sma50
            FROM indicators_daily i
            JOIN daily_bars b ON i.symbol = b.symbol AND i.trade_date = b.trade_date
            WHERE i.symbol=:s ORDER BY i.trade_date
        """)
        df = pd.read_sql(q, conn, params={"s": symbol}, parse_dates=['trade_date'])
    df = df.set_index('trade_date').sort_index()
    df['y'] = df['close'].shift(-horizon) / df['close'] - 1.0
    df["sma20_rel"] = (df["sma20"] - df["sma50"]) / df["sma50"]
    df = df.dropna()
    df["y"] = df["y"].clip(-0.2, 0.2)
    close_prices = df['close'].values
    y = df['y'].values
    states = []
    for i in range(window_size, len(close_prices)):
        window = close_prices[i-window_size:i]
        normalized_window = window / window[0] - 1.0
        states.append(normalized_window)
    states = np.array(states)
    start_idx = window_size - 1
    end_idx = start_idx + len(states)
    df_aligned = df.iloc[start_idx:end_idx]
    y_aligned = y[start_idx:end_idx]
    assert len(states) == len(df_aligned) == len(y_aligned)
    return df_aligned, states, y_aligned

def chrono_split_data(df, states, y, train_frac=0.7):
    n = len(states)
    split_idx = int(n * train_frac)
    X_tr, X_te = states[:split_idx], states[split_idx:]
    y_tr, y_te = y[:split_idx], y[split_idx:]
    df_tr, df_te = df.iloc[:split_idx], df.iloc[split_idx:]
    return df_tr, X_tr, y_tr, df_te, X_te, y_te

# ========== Main Runner ==========
def run(symbol="AAPL", horizon=DEFAULT_HORIZON, window_size=LOOKBACK_WINDOW, episodes=50, outdir=OUTDIR_DEFAULT):
    print(f">>> A2C Agent (Raw Price) vs TA  Py={sys.executable}", flush=True)
    print(f"[config] symbol={symbol} horizon={horizon} window={window_size} episodes={episodes}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    engine = create_engine(DB_URL, future=True)

    df, states, y = load_and_prepare_data(engine, symbol, horizon, window_size)
    df_tr, X_tr, y_tr, df_te, X_te, y_te = chrono_split_data(df, states, y)

    print(f"[data] {symbol} train_samples={len(X_tr)} test_samples={len(X_te)}", flush=True)

    state_size = X_tr.shape[1]
    agent = A2CAgent(state_size=state_size)

    # -------- Training --------
    print("[info] Starting A2C training...", flush=True)
    for ep in range(1, episodes + 1):
        total_reward = 0.0
        prev_action = 0
        for t in trange(len(X_tr) - 1, desc=f"Episode {ep}/{episodes}"):
            state, next_state = X_tr[t], X_tr[t+1]
            action = agent.choose_action(state)
            reward = (y_tr[t] if action == 1 else 0.0) - (TRANSACTION_COST if action != prev_action else 0.0)
            done = t == len(X_tr) - 2
            agent.learn(state, action, reward, next_state, done)
            total_reward += reward
            prev_action = action
        
        print(f"[train] ep={ep}/{episodes} total_reward={total_reward:.4f}", flush=True)

    # -------- Evaluation --------
    print("\n[info] Evaluating A2C agent on test data...", flush=True)
    actions_a2c = np.zeros(len(X_te), dtype=int)
    for t in range(len(X_te)):
        actions_a2c[t] = agent.choose_action(X_te[t])
    
    # --- TA baselines on the SAME test window ---
    def ta_actions_from_df(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        rsi_long   = (df["rsi14"] >= 55) | (df["rsi14"] < 30)
        macd_long  = (df["macd_hist"] > 0) & (df["macd_hist"].diff() > 0)
        sma_rel_ln = (df["sma20_rel"] > 0)
        stoch_long = (df["stoch_k"] >= 60)
        all_signals = rsi_long & macd_long & sma_rel_ln & stoch_long
        return {
            "TA_RSI": rsi_long.to_numpy().astype(int), "TA_MACD": macd_long.to_numpy().astype(int),
            "TA_SMA_Crossover": sma_rel_ln.to_numpy().astype(int), "TA_Stochastic": stoch_long.to_numpy().astype(int),
            "TA_All_Signals": all_signals.to_numpy().astype(int)
        }

    # --- Simulate portfolio & calculate PnL and Sharpe Ratio ---
    close_prices = df_te['close'].values
    sim_df = pd.DataFrame(index=df_te.index, data={'close_price': close_prices})
    
    ta_strats = ta_actions_from_df(df_te)
    strategies = {'A2C_RawPrice': actions_a2c, **ta_strats, 'Buy_and_Hold': np.ones_like(actions_a2c)}
    initial_cash = close_prices[0]

    for name, acts in strategies.items():
        portfolio_values = []
        cash, shares = initial_cash, 0
        for i, price_today in enumerate(close_prices):
            action_today = acts[i]
            if action_today == 1 and shares == 0:
                shares, cash = cash / price_today, 0
            elif action_today == 0 and shares > 0:
                cash, shares = shares * price_today, 0
            portfolio_values.append(cash + shares * price_today)
        sim_df[f'portfolio_value_{name}'] = portfolio_values
    
    sim_df['pnl_A2C_RawPrice'] = sim_df['portfolio_value_A2C_RawPrice'] - initial_cash
    sim_path = os.path.join(outdir, f"full_simulation_{symbol}_{horizon}d.csv")
    sim_df.to_csv(sim_path)
    print(f"[save] Full simulation log (with PnL) saved to: {sim_path}", flush=True)

    # --- Final Performance Metrics ---
    metrics = []
    for name in strategies.keys():
        final_value = sim_df[f'portfolio_value_{name}'].iloc[-1]
        returns = sim_df[f'portfolio_value_{name}'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        metrics.append({
            'strategy': name, 'final_portfolio_value': final_value,
            'total_return_%': (final_value / initial_cash - 1) * 100,
            'sharpe_ratio_annualized': sharpe_ratio
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(outdir, f"comparison_metrics_{symbol}_{horizon}d.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] Performance metrics (with Sharpe Ratio) saved to: {metrics_path}", flush=True)

    # --- Human-Readable Trading Log for A2C Agent ---
    trade_actions = []
    if len(actions_a2c):
        trade_actions.append("BUY" if actions_a2c[0] == 1 else "HOLD")
        for i in range(1, len(actions_a2c)):
            if actions_a2c[i-1] == 0 and actions_a2c[i] == 1: trade_actions.append("BUY")
            elif actions_a2c[i-1] == 1 and actions_a2c[i] == 0: trade_actions.append("SELL")
            else: trade_actions.append("HOLD")
    
    log_df = pd.DataFrame({
        "close_price": close_prices,
        "position": ["LONG" if a == 1 else "FLAT" for a in actions_a2c],
        "action": trade_actions
    }, index=df_te.index)
    
    log_path = os.path.join(outdir, f"trading_log_{symbol}_{horizon}d.csv")
    log_df.to_csv(log_path)
    print(f"[save] Trading log saved to: {log_path}", flush=True)
    
    print("\n>>> done", flush=True)

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="A2C Trading Agent (Raw Price) vs TA Baselines")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--window", type=int, default=LOOKBACK_WINDOW)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    run(symbol=args.symbol, horizon=args.horizon, window_size=args.window, episodes=args.episodes, outdir=args.outdir)
