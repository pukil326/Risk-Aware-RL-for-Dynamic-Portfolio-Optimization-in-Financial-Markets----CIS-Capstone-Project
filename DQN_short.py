import os, sys, warnings, random, collections
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from keras import mixed_precision
from keras.models import Model
from keras.layers import Dense, LayerNormalization,Add,Lambda,Input
from keras.optimizers import Adam
from keras.losses import Huber
from tqdm import trange
from sklearn.preprocessing import MinMaxScaler


# ---------------- GPU: safe defaults ----------------
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[ok] Enabled memory growth for {gpu}")
    except Exception as e:
        print(f"[warn] Could not set memory growth for {gpu}: {e}")

np.random.seed(42); random.seed(42); tf.random.set_seed(42)
# mixed_precision.set_global_policy('mixed_float16')

# ---------------- Config ----------------
OUTDIR_DEFAULT = "ml_outputs_DDQN_SPY2"

# Full feature list
STATE_FEATURES = [
    "rsi14", "sma20", "sma50", "obv", "mom", "stoch_k", "macd", "macd_hist", "cci", "adx", "trix", "roc",
    "sar", "tema", "trima", "wma", "dema", "mfi", "cmo", "stochrsi_k", "ultosc", "bop", "atr"
]
# Agent state excludes sma50
AGENT_STATE_FEATURES_DEFAULT = [f for f in STATE_FEATURES if f != "sma50"]

POSITION_FEATURE_NAME = "__position__"
TARGET_CANDIDATES = ["future_return", "ret_1d", "label", "y"]  # optional; reward uses log-return anyway

# DDQN Agent (dueling)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int): # Defaulting action size
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.lr = 3e-5
        self.memory = collections.deque(maxlen=1_000_000)
        self.batch_size = 1024
        self.tau = 0.01

        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model(hard=True)

    def _build_model(self) -> Model:
        inp = Input(shape=(self.state_size,), name="state")

        # Block 1
        x = Dense(256, activation="relu", kernel_initializer="he_normal")(inp)
        x = LayerNormalization()(x)

        # Block 2 (residual)
        h = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
        h = LayerNormalization()(h)
        x = Add()([x, h])

        # Bottleneck
        x = Dense(128, activation="relu")(x)
        x = LayerNormalization()(x)

        # Dueling heads
        adv = Dense(128, activation="relu")(x)
        adv = Dense(self.action_size, activation="linear", dtype="float32")(adv)

        val = Dense(128, activation="relu")(x)
        val = Dense(1, activation="linear", dtype="float32")(val)

        # A(s,a) - mean_a A(s,a) as a *layer*
        adv_center = Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True),
            name="adv_center",
        )(adv)

        # Q(s,a) = V(s) + A_center(s,a)
        q = Add(name="q_values")([val, adv_center])

        model = Model(inputs=inp, outputs=q, name="dueling_ddqn")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr,
            clipnorm=10.0 
        )

        model.compile(
            loss=tf.keras.losses.Huber(delta=1.0),
            optimizer=optimizer
            , jit_compile=True  
        )
        return model

    @tf.function(jit_compile=True)
    def train_step(self, states, actions, targets):
        """One GPU-accelerated training step"""
        with tf.GradientTape() as tape:
            q = self.online_model(states, training=True)           # shape [B, A]
            idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_sa = tf.gather_nd(q, idx)                            # shape [B]
            loss = tf.keras.losses.huber(targets, q_sa)

        grads = tape.gradient(loss, self.online_model.trainable_variables)
        self.online_model.optimizer.apply_gradients(
            zip(grads, self.online_model.trainable_variables)
        )
        return loss
    
    def update_target_model(self, hard: bool = False):
        if hard:
            self.target_model.set_weights(self.online_model.get_weights())
        else:
            ow = self.online_model.get_weights()
            tw = self.target_model.get_weights()
            self.target_model.set_weights([self.tau * o + (1.0 - self.tau) * t
                                           for o, t in zip(ow, tw)])

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s.astype(np.float32), int(a), float(r), s_next.astype(np.float32), bool(done)))

    @tf.function(jit_compile=True)
    def _predict_action(self, state_tensor: tf.Tensor) -> tf.Tensor:
        """GPU-accelerated action prediction"""
        return self.online_model(state_tensor, training=False)

    def choose_action(self, state_1d: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (np.random.rand() <= self.epsilon):
            # Epsilon-greedy exploration
            return np.random.randint(self.action_size)
        
        # Exploit: run fast, compiled prediction
        state_tensor = tf.convert_to_tensor(state_1d.reshape(1, -1), dtype=tf.float32)
        q_values_tensor = self._predict_action(state_tensor)
        return int(tf.argmax(q_values_tensor[0]).numpy())


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        UPDATES_PER_STEP = 8
        for _ in range(UPDATES_PER_STEP):
            minibatch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

            # tensors on GPU
            states      = tf.convert_to_tensor(states,      dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            actions     = tf.convert_to_tensor(actions,     dtype=tf.int32)
            rewards     = tf.convert_to_tensor(rewards,     dtype=tf.float32)
            dones       = tf.convert_to_tensor(dones,       dtype=tf.float32)

            # --- DDQN target ---
            # 1) pick greedy actions with ONLINE net
            q_next_online = self.online_model(next_states)                          
            a_star = tf.argmax(q_next_online, axis=1, output_type=tf.int32)     

            # 2) evaluate those actions with TARGET net
            q_next_all = self.target_model(next_states)                     
            idx = tf.stack([tf.range(tf.shape(a_star)[0], dtype=tf.int32), a_star], axis=1) 
            q_next_best = tf.gather_nd(q_next_all, idx)
            
            q_next_best_f32 = tf.cast(q_next_best, dtype=tf.float32)
            gamma_f32 = tf.cast(self.gamma, dtype=tf.float32)
            q_target = rewards + (1.0 - dones) * gamma_f32 * q_next_best_f32

            # one accelerated update
            _ = self.train_step(states, actions, q_target)

        # epsilon decay & soft target update
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_model(hard=False)



# CSV data utilities
def _read_csvs(csv_dir: str):
    bars = pd.read_csv(os.path.join(csv_dir, "daily_bars.csv"),
                       parse_dates=["trade_date"], infer_datetime_format=True)
    ind = pd.read_csv(os.path.join(csv_dir, "indicators_daily.csv"),
                      parse_dates=["trade_date"], infer_datetime_format=True)
    fut = pd.read_csv(os.path.join(csv_dir, "future_returns_daily.csv"),
                      parse_dates=["trade_date"], infer_datetime_format=True)
    syms = pd.read_csv(os.path.join(csv_dir, "symbols.csv"))
    # normalize colnames
    for df in (bars, ind, fut):
        df.columns = [c.strip() for c in df.columns]
    return bars, ind, fut, syms


def load_from_csv(csv_dir: str, symbol: str):
    """Merge all CSVs on (symbol, trade_date); return df sorted by date."""
    bars, ind, fut, syms = _read_csvs(csv_dir)

    # basic schema checks
    for name, df in [("daily_bars.csv", bars), ("indicators_daily.csv", ind), ("future_returns_daily.csv", fut)]:
        if not {"symbol", "trade_date"}.issubset(df.columns):
            raise ValueError(f"[error] {name} must have columns: symbol, trade_date, ...")

    merged = (bars.merge(ind, on=["symbol", "trade_date"], how="inner")
                    .merge(fut, on=["symbol", "trade_date"], how="left")) 

    df = (merged[merged["symbol"].astype(str) == str(symbol)]
                .sort_values("trade_date")
                .reset_index(drop=True))

    if df.empty:
        raise ValueError(f"[error] No data after merge for symbol={symbol}. "
                         f"Confirm symbol values & dates match across CSVs.")

    # Extra engineered feature from your original code
    if {"sma20", "sma50"}.issubset(df.columns):
        df["sma20_rel"] = (df["sma20"] - df["sma50"]) / df["sma50"]

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    return df


def chrono_split_data(df: pd.DataFrame, train_frac=0.7):
    n = len(df)
    split_idx = int(n * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# --- CHANGE 2: Updated Realistic CASH-BASED Simulation Function (to allow shorting) ---
def simulate_strategy(
    prices: np.ndarray, 
    actions: np.ndarray, 
    positions_map: Dict[int, float], # e.g., {0: -1.0, 1: -0.5, 2: 0.0, ...}
    initial_cash: float, 
    txn_cost: float
) -> np.ndarray:
    """
    Simulates a strategy based on target position percentages with a cash constraint.
    'actions' are indices (e.g., 0, 1, 2) mapping to 'positions_map'.
    Handles rebalancing based on available cash and shares.
    Allows 'shares' to become negative (short selling).
    """
    cash, shares = initial_cash, 0.0
    portfolio_values = [initial_cash]
    current_pos_percent = 0.0 # Start flat

    for i in range(len(prices) - 1):
        current_price = prices[i]
        
        # --- 1. Calculate Current State ---
        # Mark-to-Market (MTM) value of the portfolio *before* today's trade
        # For short positions, 'shares' is negative, so this correctly subtracts
        # the value of the shorted stock from cash to get net liquid value.
        current_portfolio_value = cash + shares * current_price
        
        # --- 2. Get Target Position ---
        action_idx = int(actions[i])
        # Default to holding current position if action is undefined
        target_pos_percent = positions_map.get(action_idx, current_pos_percent) 
        
        # --- 3. Calculate Ideal Trade ---
        # How many shares we *want* to hold ideally
        target_shares_value = current_portfolio_value * target_pos_percent
        
        # Handle division by zero if price is somehow 0
        ideal_target_shares = 0.0
        if current_price > 1e-6:
             ideal_target_shares = target_shares_value / current_price
        
        # How many shares we need to trade (buy/sell) to reach the ideal
        # +ve = buy, -ve = sell
        shares_to_trade = ideal_target_shares - shares 

        # --- 4. Execute Trade with Cash/Share Constraints ---
        if shares_to_trade > 0: # --- WANT TO BUY (or cover short) ---
            # Calculate cash needed for the ideal trade
            cash_needed = shares_to_trade * current_price * (1.0 + txn_cost)
            
            if cash >= cash_needed:
                # We have enough cash, execute the ideal trade
                actual_shares_to_buy = shares_to_trade
                cash -= cash_needed
            else:
                # Not enough cash. Buy as many shares as possible.
                if cash > 0 and current_price > 1e-6:
                    affordable_shares = cash / (current_price * (1.0 + txn_cost))
                    actual_shares_to_buy = affordable_shares
                    cash = 0.0 # Used all cash
                else:
                    actual_shares_to_buy = 0 # No cash left or price is 0
            
            shares += actual_shares_to_buy

        elif shares_to_trade < 0: # --- WANT TO SELL (or open short) ---
            # This is the logic that was fixed.
            # We assume no margin limit, so we can always sell or short.
            # `shares_to_trade` is already negative here.
            actual_shares_to_trade = shares_to_trade
            
            cash_gained = abs(actual_shares_to_trade) * current_price * (1.0 - txn_cost)
            shares += actual_shares_to_trade # e.g., 0 + (-20) = -20
            cash += cash_gained
        
        # --- 5. Update position state and store next-day MTM value ---
        # Update the *actual* position percent
        # We must handle division by zero if value is 0
        new_portfolio_value = cash + shares * current_price
        if new_portfolio_value > 1e-6:
            current_pos_percent = (shares * current_price) / new_portfolio_value
        else:
            current_pos_percent = 0.0

        # Value is calculated at the *next* day's closing price
        portfolio_values.append(cash + shares * prices[i + 1])

    # Ensure length matches input prices
    return np.array(portfolio_values[:len(prices)])


# Runner
def run(symbol="SPY", episodes=30, outdir=OUTDIR_DEFAULT, csv_dir="data"):
    print(f">>> Advanced DQN Agent (CSV)  Py={sys.executable}")
    print(f"[config] symbol={symbol} episodes={episodes} csv_dir={csv_dir}")

    os.makedirs(outdir, exist_ok=True)

    # ------- Load & split -------
    df_all = load_from_csv(csv_dir, symbol)
    train_df, test_df = chrono_split_data(df_all)

    # Select features actually present
    present = [f for f in AGENT_STATE_FEATURES_DEFAULT if f in train_df.columns]
    if not present:
        raise ValueError("[error] No indicator features found. Check indicators_daily.csv columns.")
    if len(present) < len(AGENT_STATE_FEATURES_DEFAULT):
        missing = sorted(set(AGENT_STATE_FEATURES_DEFAULT) - set(present))
        print(f"[warn] Missing features (will skip): {missing}")

    # ------- Scale -------
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(train_df[present])
    X_te = scaler.transform(test_df[present])

    # Reward driver: log return from close
    close_tr = train_df["close"].values.squeeze()
    log_ret_tr = np.zeros_like(close_tr, dtype="float32")
    if len(close_tr) > 1:
        log_ret_tr[1:] = np.log(close_tr[1:] / close_tr[:-1])

    print(f"[data] {symbol} train_samples={len(X_tr)} test_samples={len(X_te)} features={len(present)}")

    # ------- Agent -------
    # --- CHANGE 1: Define 5-point position mapping (including short) ---
    # Action 0 -> -100% invested (Full-Short)
    # Action 1 -> -50% invested (Half-Short)
    # Action 2 -> 0% invested (Flat)
    # Action 3 -> 50% invested (Half-Long)
    # Action 4 -> 100% invested (Full-Long)
    POSITIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ACTION_SIZE = len(POSITIONS)
    DQN_POSITIONS_MAP = {i: p for i, p in enumerate(POSITIONS)}

    state_size = X_tr.shape[1] + 1  # + position
    agent = DQNAgent(state_size=state_size, action_size=ACTION_SIZE)
    txn_cost = 0.0005 # Transaction cost

    # ------- Train -------
    print("[info] Training...")
    n = len(X_tr)
    for ep in range(1, episodes + 1):
        pos = 0.0  # Start flat (0.0 = 0%)
        state = np.concatenate([X_tr[0], [pos]]).astype(np.float32)

        for t in trange(n - 1, desc=f"Episode {ep}/{episodes}"):
            action = agent.choose_action(state)  # 0, 1, 2, 3, or 4
            next_pos = POSITIONS[action] # Target position (e.g., -1.0, 0.5, etc.)

            # Reward: PnL from current position + cost proportional to trade size
            # This logic holds for shorting:
            # If pos = -1.0 (short) and return is negative (price drops),
            # reward becomes positive: (-1.0 * -0.01) = +0.01
            trade_size = abs(next_pos - pos)
            reward = (pos * log_ret_tr[t + 1]) - (txn_cost * trade_size)

            next_state = np.concatenate([X_tr[t + 1], [next_pos]]).astype(np.float32)
            done = (t == n - 2)

            # Action 'action' (0-4) led to this
            agent.remember(state, action, reward, next_state, done)
            state, pos = next_state, next_pos

            if t > agent.batch_size and t % 4 == 0:
                agent.replay()

        print(f"[train] ep={ep}/{episodes}  eps={agent.epsilon:.3f}")

    # ------- Evaluate -------
    print("[info] Evaluating...")
    close_te = test_df["close"].values.squeeze()
    
    # Generate DQN actions
    actions_dqn = np.zeros(len(X_te), dtype=int)
    pos = 0.0 # Start flat
    for t in range(len(X_te)): # Iterate all the way to generate all actions
        s = np.concatenate([X_te[t], [pos]]).astype(np.float32)
        a = agent.choose_action(s, greedy=True) # 0, 1, 2, 3, or 4
        actions_dqn[t] = a
        pos = POSITIONS[a] # Update position for next state input

    # Baseline technical strategies (same logic; cost included)
    def ta_actions_from_df(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        rsi_long = (df.get("rsi14", pd.Series(False, index=df.index)) >= 55) | (df.get("rsi14", 0) < 30)
        macd_long = (df.get("macd_hist", pd.Series(0, index=df.index)) > 0) & (df.get("macd_hist", 0).diff() > 0)
        sma_rel_ln = (df.get("sma20_rel", pd.Series(0.0, index=df.index)) > 0)
        stoch_long = (df.get("stoch_k", pd.Series(0, index=df.index)) >= 60)
        all_signals = rsi_long & macd_long & sma_rel_ln & stoch_long
        return {
            "TA_RSI": rsi_long.to_numpy().astype(int),
            "TA_MACD": macd_long.to_numpy().astype(int),
            "TA_SMA_Crossover": sma_rel_ln.to_numpy().astype(int),
            "TA_Stochastic": stoch_long.to_numpy().astype(int),
            "TA_All_Signals": all_signals.to_numpy().astype(int),
        }

    ta_strats = ta_actions_from_df(test_df)
    
    # Use the new `simulate_strategy` function for ALL strategies
    sim_df = pd.DataFrame(index=test_df.index)
    sim_df["close_price"] = close_te
    initial_cash = close_te[0]

    # Define position maps for baselines
    # TA strategies are binary: 0 -> 0% (Flat), 1 -> 100% (Long)
    BINARY_POSITIONS_MAP = {0: 0.0, 1: 1.0}
    # Buy & Hold: Action 1 -> 100% (Long)
    BUY_HOLD_MAP = {1: 1.0} 

    # Add DQN to the list of strategies to simulate
    strategies = {"DQN_Advanced": actions_dqn, **ta_strats, "Buy_and_Hold": np.ones_like(actions_dqn)}

    for name, acts in strategies.items():
        if name == "DQN_Advanced":
            map_to_use = DQN_POSITIONS_MAP
        elif name == "Buy_and_Hold":
            map_to_use = BUY_HOLD_MAP
        else: # All other TA strategies
            map_to_use = BINARY_POSITIONS_MAP
            
        vals = simulate_strategy(close_te, acts, map_to_use, initial_cash, txn_cost)
        sim_df[f"portfolio_value_{name}"] = vals[:len(sim_df)]

    # This PnL calculation is a cumulative Mark-to-Market PnL, which is correct.
    # It now reflects the cash-constrained simulation.
    sim_df["pnl_DQN_Advanced"] = sim_df["portfolio_value_DQN_Advanced"] - initial_cash
    
    # Add PNL for other strategies to sim_df
    for name in strategies.keys():
        if name != "DQN_Advanced":
             sim_df[f"pnl_{name}"] = sim_df[f"portfolio_value_{name}"] - initial_cash

    sim_path = os.path.join(outdir, f"full_simulation_{symbol}.csv")
    sim_df.to_csv(sim_path)
    print(f"[save] Full simulation: {sim_path}")

    metrics = []
    for name in strategies.keys():
        final_value = sim_df[f"portfolio_value_{name}"].iloc[-1]
        returns = sim_df[f"portfolio_value_{name}"].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        metrics.append({
            "strategy": name,
            "final_portfolio_value": float(final_value),
            "total_return_%": float((final_value / initial_cash - 1) * 100.0),
            "sharpe_ratio_annualized": float(sharpe_ratio),
        })
    pd.DataFrame(metrics).to_csv(os.path.join(outdir, f"comparison_metrics_{symbol}.csv"), index=False)
    print(f"[save] Metrics: {os.path.join(outdir, f'comparison_metrics_{symbol}.csv')}")

    # --- CHANGE 3: Update trading log for 5-state actions ---
    aa = strategies["DQN_Advanced"] # Actions are 0, 1, 2, 3, 4
    position_labels = {
        0: "FULL-SHORT",
        1: "HALF-SHORT",
        2: "FLAT",
        3: "HALF-LONG",
        4: "FULL-LONG"
    }
    
    trade_actions = ["HOLD"] * len(aa) # Default
    if len(aa):
        trade_actions[0] = position_labels[aa[0]] # Initial action
        for i in range(1, len(aa)):
            if aa[i] > aa[i-1]:
                trade_actions[i] = "BUY"
            elif aa[i] < aa[i-1]:
                trade_actions[i] = "SELL"
            else:
                trade_actions[i] = "HOLD"
                
    log_df = pd.DataFrame({
        "close_price": close_te,
        "position": [position_labels.get(a, "UNKNOWN") for a in aa],
        "action": trade_actions[:len(close_te)],
    }, index=test_df.index)
    log_df.to_csv(os.path.join(outdir, f"trading_log_{symbol}.csv"))
    print(f"[save] Trading log: {os.path.join(outdir, f'trading_log_{symbol}.csv')}")

    print(">>> done")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Advanced DQN Trading Agent (CSV inputs)")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT) 
    ap.add_argument("--csv_dir", default="data")
    args = ap.parse_args() 
    run(symbol=args.symbol, episodes=args.episodes, outdir=args.outdir, csv_dir=args.csv_dir)