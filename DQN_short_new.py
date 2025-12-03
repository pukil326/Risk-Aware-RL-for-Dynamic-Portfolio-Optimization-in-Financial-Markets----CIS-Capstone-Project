import os, sys, warnings, random, collections
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from keras import mixed_precision
from keras.models import Model
from keras.layers import Dense, LayerNormalization, Add, Lambda, Input, LSTM
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
# mixed_precision.set_global_policy('mixed_float16') # Disabled per original file

# ---------------- Config ----------------
OUTDIR_DEFAULT = "ml_outputs_DDQN_SPY2"

WINDOW_SIZE = 20 # Agent will look at the last 20 days of data

# Full feature list from your indicators pipeline (will auto-subset to those present)
STATE_FEATURES = [
    "rsi14", "sma20", "sma50", "obv", "mom", "stoch_k", "macd", "macd_hist", "cci", "adx", "trix", "roc",
    "sar", "tema", "trima", "wma", "dema", "mfi", "cmo", "stochrsi_k", "ultosc", "bop", "atr"
]
# Agent state excludes sma50
AGENT_STATE_FEATURES_DEFAULT = [f for f in STATE_FEATURES if f != "sma50"]

POSITION_FEATURE_NAME = "__position__"
TARGET_CANDIDATES = ["future_return", "ret_1d", "label", "y"]  # optional; reward uses log-return anyway

# DDQN Agent (dueling) — same structure as your original with minor polish


class DQNAgent:
    # --- state_size is now a tuple (window_size, num_features) ---
    def __init__(self, state_size: Tuple[int, int], action_size: int): # Defaulting action size
        self.state_size = state_size # e.g., (20, 25)
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 1e-4
        self.memory = collections.deque(maxlen=1_000_000)
        self.batch_size = 512
        self.tau = 0.01

        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model(hard=True)

    def _build_model(self) -> Model:
        # --- New architecture with LSTM ---
        # State input now has shape (window_size, num_features)
        inp = Input(shape=self.state_size, name="state_timeseries")

        # 1. Process the time-series data with an LSTM
        # This layer will learn to find patterns in the 20-day window
        x = LayerNormalization()(inp)
        x = LSTM(64, activation="tanh")(x) # Outputs a single vector representing the window
        
        # 2. Dueling heads (now processing the LSTM's output)
        adv = Dense(128, activation="relu")(x)
        adv = Dense(self.action_size, activation="linear", dtype="float32")(adv)

        val = Dense(128, activation="relu")(x)
        val = Dense(1, activation="linear", dtype="float32")(val)
        # --- End of new architecture ---

        # A(s,a) - mean_a A(s,a) as a *layer*
        adv_center = Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True),
            name="adv_center",
        )(adv)

        # Q(s,a) = V(s) + A_center(s,a)
        q = Add(name="q_values")([val, adv_center])

        model = Model(inputs=inp, outputs=q, name="dueling_ddqn_lstm")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr,
            clipnorm=10.0 
        )

        model.compile(
            loss=tf.keras.losses.Huber(delta=1.0),
            optimizer=optimizer
            , jit_compile=False
        )
        return model

    @tf.function
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
        # --- States are now (window, features) arrays ---
        self.memory.append((s.astype(np.float32), int(a), float(r), s_next.astype(np.float32), bool(done)))

    @tf.function
    def _predict_action(self, state_tensor: tf.Tensor) -> tf.Tensor:
        """GPU-accelerated action prediction"""
        return self.online_model(state_tensor, training=False)

    def choose_action(self, state_window: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (np.random.rand() <= self.epsilon):
            # Epsilon-greedy exploration
            return np.random.randint(self.action_size)
        
        # --- Reshape state for (batch, window, features) ---
        state_tensor = tf.convert_to_tensor(
            state_window.reshape(1, self.state_size[0], self.state_size[1]), 
            dtype=tf.float32
        )
        q_values_tensor = self._predict_action(state_tensor)
        return int(tf.argmax(q_values_tensor[0]).numpy())


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        UPDATES_PER_STEP = 8
        for _ in range(UPDATES_PER_STEP):
            minibatch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
            
            # --- States are now 3D (batch, window, features) ---
            # Keras/TF handles this shape automatically
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
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
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


# --- Helper function to create windowed data ---
def create_windowed_dataset(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Takes 2D data (samples, features) and creates 3D windowed data
    (samples - window + 1, window_size, features).
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    # Create an empty array to hold the windowed data
    # Shape: (num_windows, window_size, num_features)
    num_windows = n_samples - window_size + 1
    windowed_data = np.empty((num_windows, window_size, n_features), dtype=data.dtype)
    
    # Fill the array
    for i in range(num_windows):
        windowed_data[i] = data[i : i + window_size, :]
        
    return windowed_data


# --- Updated Realistic CASH-BASED Simulation Function (to allow shorting) ---
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
def run(symbol="SPY", episodes=200, outdir=OUTDIR_DEFAULT, csv_dir="data"):
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
    # Note: We scale the 2D data *before* windowing it
    X_tr_scaled = scaler.fit_transform(train_df[present])
    X_te_scaled = scaler.transform(test_df[present])

    # Reward driver: log return from close (as in your original)
    close_tr = train_df["close"].values.squeeze() # <--- FIX 1: Add .squeeze()
    log_ret_tr = np.zeros_like(close_tr, dtype="float32")
    if len(close_tr) > 1:
        log_ret_tr[1:] = np.log(close_tr[1:] / close_tr[:-1])

    # --- Create 3D Windowed Datasets ---
    # X_tr_windows shape: (n_samples - W + 1, window_size, n_features)
    X_tr_windows = create_windowed_dataset(X_tr_scaled, WINDOW_SIZE)
    
    # Align log returns. The first window (index 0) predicts for the
    # day *after* the window ends (day W), which corresponds to log_ret_tr[W].
    # So we slice log_ret_tr to start from WINDOW_SIZE
    log_ret_tr_aligned = log_ret_tr[WINDOW_SIZE:]
    
    # Align prices for simulation (needed for TA baselines)
    close_te = test_df["close"].values.squeeze() # <--- FIX 2: Add .squeeze()
    
    # Align test data for TA baselines
    # The TA baselines don't use windows, but we must align them to the
    # same starting point as the DQN agent for a fair comparison.
    test_df_aligned = test_df.iloc[WINDOW_SIZE - 1:].reset_index(drop=True)
    close_te_aligned = close_te[WINDOW_SIZE - 1:]
    
    # Create windowed test data
    X_te_windows = create_windowed_dataset(X_te_scaled, WINDOW_SIZE)
    
    # Data check
    n_tr, n_te = len(X_tr_windows), len(X_te_windows)
    print(f"[data] {symbol} train_windows={n_tr} test_windows={n_te} features={len(present)}")
    if n_tr < len(log_ret_tr_aligned):
        log_ret_tr_aligned = log_ret_tr_aligned[:n_tr]
    elif n_tr > len(log_ret_tr_aligned):
         X_tr_windows = X_tr_windows[:len(log_ret_tr_aligned)]
         n_tr = len(X_tr_windows)
         print(f"[data] Aligned train_windows to {n_tr} to match rewards")


    # ------- Agent -------
    # --- 5-point position mapping (including short) ---
    POSITIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ACTION_SIZE = len(POSITIONS)
    DQN_POSITIONS_MAP = {i: p for i, p in enumerate(POSITIONS)}

    # --- State size is now (window_size, features + 1 for position) ---
    num_features = X_tr_windows.shape[2]
    window_size = X_tr_windows.shape[1]
    
    # Our model input shape: (window_size, num_features + 1)
    # The +1 is for the position, which we'll stack
    STATE_SHAPE = (window_size, num_features + 1)

    agent = DQNAgent(state_size=STATE_SHAPE, action_size=ACTION_SIZE)
    txn_cost = 0.0005 # Transaction cost
    reward_scaler = 100.0 # --- Reward scaler ---

    # ------- Train -------
    print("[info] Training...")
    n = n_tr # Use number of *windows*
    
    # Create a 3D array for training states, including position
    # Shape: (n_samples, window_size, n_features + 1)
    pos_history_tr = np.zeros((n, window_size, 1), dtype=np.float32)
    state_inputs_tr = np.concatenate([X_tr_windows, pos_history_tr], axis=2)

    for ep in range(1, episodes + 1):
        pos = 0.0  # Start flat (0.0 = 0%)
        
        # Update the position in the *last time step* of the *first window*
        state_inputs_tr[0, -1, -1] = pos 
        
        for t in trange(n - 1, desc=f"Episode {ep}/{episodes}"):
            # Get the current windowed state
            current_state_window = state_inputs_tr[t]
            
            action = agent.choose_action(current_state_window)  # 0, 1, 2, 3, or 4
            next_pos = POSITIONS[action] # Target position (e.g., -1.0, 0.5, etc.)

            trade_size = abs(next_pos - pos)
            # --- Use aligned log returns and scale the reward ---
            reward = ((pos * log_ret_tr_aligned[t + 1]) - (txn_cost * trade_size)) * reward_scaler

            # Create the next state window
            # We take the *next* window (from t+1) and update *its* last
            # time step with the new position 'next_pos'
            next_state_window = state_inputs_tr[t+1].copy()
            next_state_window[-1, -1] = next_pos # Set position in last step
            
            done = (t == n - 2)

            agent.remember(current_state_window, action, reward, next_state_window, done)
            pos = next_pos
            
            # Update the *next* state's position in our main array for the *next* loop iteration
            if t < n - 2:
                state_inputs_tr[t+1, -1, -1] = pos

            if t > agent.batch_size and t % 4 == 0:
                agent.replay()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"[train] ep={ep}/{episodes}  eps={agent.epsilon:.3f}")

    # ------- Evaluate -------
    print("[info] Evaluating...")
    
    # Create 3D test states
    pos_history_te = np.zeros((n_te, window_size, 1), dtype=np.float32)
    state_inputs_te = np.concatenate([X_te_windows, pos_history_te], axis=2)

    actions_dqn = np.zeros(n_te, dtype=int)
    pos = 0.0 # Start flat
    
    for t in range(n_te): # Iterate all the way to generate all actions
        # Set current pos in the last step of the current window
        state_inputs_te[t, -1, -1] = pos
        current_state_window = state_inputs_te[t]
        
        a = agent.choose_action(current_state_window, greedy=True) # 0, 1, 2, 3, or 4
        actions_dqn[t] = a
        pos = POSITIONS[a] # Update position for next state input

    # --- Baseline TAs must use the ALIGNED dataframe ---
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

    ta_strats = ta_actions_from_df(test_df_aligned)
    
    # --- Use aligned prices and index ---
    sim_df = pd.DataFrame(index=test_df_aligned.index)
    sim_df["close_price"] = close_te_aligned
    initial_cash = close_te_aligned[0]

    # Define position maps for baselines
    # TA strategies are binary: 0 -> 0% (Flat), 1 -> 100% (Long)
    BINARY_POSITIONS_MAP = {0: 0.0, 1: 1.0}
    # Buy & Hold: Action 1 -> 100% (Long)
    BUY_HOLD_MAP = {1: 1.0} 

    # --- Align Buy & Hold actions to the new length ---
    buy_and_hold_actions = np.ones_like(actions_dqn)
    strategies = {"DQN_Advanced": actions_dqn, **ta_strats, "Buy_and_Hold": buy_and_hold_actions}

    for name, acts in strategies.items():
        if name == "DQN_Advanced":
            map_to_use = DQN_POSITIONS_MAP
        elif name == "Buy_and_Hold":
            map_to_use = BUY_HOLD_MAP
        else: # All other TA strategies
            map_to_use = BINARY_POSITIONS_MAP
            
        # --- Simulate using aligned prices ---
        vals = simulate_strategy(close_te_aligned, acts, map_to_use, initial_cash, txn_cost)
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

    # --- Update trading log for 5-state actions ---
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
        "close_price": close_te_aligned, # --- Use aligned prices
        "position": [position_labels.get(a, "UNKNOWN") for a in aa],
        "action": trade_actions[:len(close_te_aligned)], # --- Use aligned prices
    }, index=test_df_aligned.index) # --- Use aligned index
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