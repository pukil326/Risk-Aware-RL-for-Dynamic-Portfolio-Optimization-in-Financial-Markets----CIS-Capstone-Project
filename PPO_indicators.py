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
OUTDIR_DEFAULT = "ml_outputs_ppo_paper"

# ========== State Space: From your calculate_indicators.py script ==========
STATE_FEATURES = [
    "rsi14", "sma20", "obv", "mom", "stoch_k", "macd", "macd_hist", "cci", "adx", "trix", "roc",
    "sar", "tema", "trima", "wma", "dema", "mfi", "cmo", "stochrsi_k",
    "ultosc", "bop", "atr"
]
# Extra features needed for TA baselines
TA_FEATURES = ["sma50"]


# ========== PPO Agent ==========
class PPOAgent:
    def __init__(self, state_size: int, action_size: int = 2, gamma=0.99, lr=0.0001, gae_lambda=0.95, clip_ratio=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        
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

    def learn(self, states, actions, rewards, next_states, dones):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        with tf.GradientTape() as tape:
            old_probs = self.actor(states, training=True)
            actions_one_hot = tf.one_hot(actions, self.action_size)
            old_probs_selected = tf.reduce_sum(actions_one_hot * old_probs, axis=1)

            current_probs = self.actor(states, training=True)
            current_probs_selected = tf.reduce_sum(actions_one_hot * current_probs, axis=1)
            
            ratio = current_probs_selected / (old_probs_selected + 1e-10)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            values_pred = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(advantages + values - values_pred))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

# ========== Utilities ==========
def load_data(engine, symbol: str) -> pd.DataFrame:
    """ Loads all indicators mentioned in the paper, plus extra ones for TA baselines. """
    all_features = sorted(list(set(STATE_FEATURES + TA_FEATURES)))
    features_str = ", ".join([f"i.{f}" for f in all_features])
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
def run(symbol="AAPL", episodes=50, outdir=OUTDIR_DEFAULT):
    print(f">>> PPO Agent (Paper Methodology)  Py={sys.executable}", flush=True)
    print(f"[config] symbol={symbol} episodes={episodes}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    engine = create_engine(DB_URL, future=True)

    df = load_data(engine, symbol)
    if df.empty:
        raise RuntimeError(f"No data loaded for symbol {symbol}. Check database.")
        
    train_df, test_df = chrono_split_data(df)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(train_df[STATE_FEATURES])
    X_te = scaler.transform(test_df[STATE_FEATURES])
    
    log_returns_tr = np.log(train_df['close'] / train_df['close'].shift(1)).fillna(0).values

    print(f"[data] {symbol} train_samples={len(X_tr)} test_samples={len(X_te)}", flush=True)

    state_size = X_tr.shape[1]
    agent = PPOAgent(state_size=state_size)

    # -------- Training --------
    print("[info] Starting PPO training based on paper methodology...", flush=True)
    for ep in range(1, episodes + 1):
        states_mem, actions_mem, rewards_mem, next_states_mem, dones_mem = [], [], [], [], []
        
        for t in trange(len(X_tr) - 1, desc=f"Episode {ep}/{episodes}"):
            state = X_tr[t]
            action = agent.choose_action(state)
            next_state = X_tr[t+1]
            
            reward = log_returns_tr[t] if action == 1 else 0.0
            done = t == len(X_tr) - 2

            states_mem.append(state)
            actions_mem.append(action)
            rewards_mem.append(reward)
            next_states_mem.append(next_state)
            dones_mem.append(done)
        
        agent.learn(states_mem, actions_mem, rewards_mem, next_states_mem, dones_mem)
        print(f"[train] ep={ep}/{episodes} finished learning from trajectory.", flush=True)

    # -------- Evaluation --------
    print("\n[info] Evaluating on test data...", flush=True)
    actions = np.zeros(len(X_te), dtype=int)
    for t in range(len(X_te)):
        s = X_te[t]
        actions[t] = agent.choose_action(s)
    
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
    strategies = {'PPO_Paper': actions, **ta_strats, 'Buy_and_Hold': np.ones_like(actions)}
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
    
    sim_df['pnl_PPO_Paper'] = sim_df['portfolio_value_PPO_Paper'] - initial_cash
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
    
    # --- Human-Readable Trading Log for PPO Agent ---
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
    ap = argparse.ArgumentParser(description="PPO Trading Agent based on Paper Methodology")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    run(symbol=args.symbol, episodes=args.episodes, outdir=args.outdir)

