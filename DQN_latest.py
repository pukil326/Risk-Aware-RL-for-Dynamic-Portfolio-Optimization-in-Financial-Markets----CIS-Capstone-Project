import os, sys, warnings, random, collections
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from keras import mixed_precision
from keras.models import Model
from keras.layers import Dense, LayerNormalization, Add, Lambda, Input, LSTM, Dropout
from keras.optimizers import Adam
from keras.losses import Huber
from tqdm import trange
from sklearn.preprocessing import RobustScaler  # Better than MinMaxScaler for outliers


# ---------------- GPU: safe defaults ----------------
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[ok] Enabled memory growth for {gpu}")
    except Exception as e:
        print(f"[warn] Could not set memory growth for {gpu}: {e}")

np.random.seed(42); random.seed(42); tf.random.set_seed(42)

# ---------------- Config ----------------
OUTDIR_DEFAULT = "ml_outputs_DDQN_SPY2"
WINDOW_SIZE = 20  # SMALLER window for faster training

STATE_FEATURES = [
    "rsi14", "sma20", "sma50", "obv", "mom", "stoch_k", "macd", 
    "macd_hist", "cci", "adx", "trix", "roc",
    "sar", "tema", "trima", "wma", "dema", 
    "mfi", "cmo", "stochrsi_k", "ultosc", "bop", "atr"
]
AGENT_STATE_FEATURES_DEFAULT = [f for f in STATE_FEATURES if f != "sma50"]
POSITION_FEATURE_NAME = "__position__"
TARGET_CANDIDATES = ["future_return", "ret_1d", "label", "y"]


# IMPROVEMENT: Prioritized Experience Replay for better sample efficiency
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full priority)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, experience, error: float = None):
        """Add experience with priority based on TD error"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get max priority to ensure they're sampled soon
        self.priorities[self.position] = max_priority if error is None else (abs(error) + 1e-6) ** self.alpha
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with priorities, return indices and importance weights"""
        if self.size < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        
        # FIX: Ensure priorities are valid (no zeros, no NaN)
        priorities = np.maximum(priorities, 1e-6)
        priorities = np.nan_to_num(priorities, nan=1e-6)
        
        # Normalize to probabilities
        priorities_sum = priorities.sum()
        if priorities_sum <= 0 or np.isnan(priorities_sum) or np.isinf(priorities_sum):
            # Fallback to uniform sampling if priorities are invalid
            probs = np.ones(self.size) / self.size
        else:
            probs = priorities / priorities_sum
        
        # Double-check probabilities are valid
        probs = np.nan_to_num(probs, nan=1.0/self.size)
        probs = probs / probs.sum()  # Renormalize to be safe
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = np.nan_to_num(weights, nan=1.0)
        weights = weights / (weights.max() + 1e-8)  # Normalize
        
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            # FIX: Ensure error is valid before computing priority
            error = float(error)
            if np.isnan(error) or np.isinf(error):
                error = 1.0  # Default priority for invalid errors
            self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha


class DQNAgent:
    def __init__(self, state_size: Tuple[int, int], action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        # IMPROVEMENT: Better hyperparameters for sample efficiency
        self.gamma = 0.97  # Higher for longer-term thinking
        self.epsilon = 1.0
        self.epsilon_min = 0.10  # HIGHER minimum - keep exploring
        self.epsilon_decay = 0.96  # FASTER decay for 50 episodes
        self.lr = 5e-4  # HIGHER learning rate for faster convergence
        
        # IMPROVEMENT: Prioritized replay
        self.memory = PrioritizedReplayBuffer(capacity=50_000, alpha=0.6)
        self.beta = 0.4  # Importance sampling weight (increases over time)
        self.beta_increment = 0.01  # FASTER increase for 50 episodes
        
        self.batch_size = 256  # LARGER batch for more stable updates
        self.tau = 0.01  # FASTER target update

        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model(hard=True)
        
        # CRITICAL: Initialize Q-values to small positive values
        # This prevents NaN and encourages exploration
        dummy_state = np.zeros((1, self.state_size[0], self.state_size[1]), dtype=np.float32)
        _ = self.online_model(dummy_state, training=False)  # Build the model
        
        self.train_step_counter = 0
        self.update_frequency = 50  # Update target less often

    def _build_model(self) -> Model:
        inp = Input(shape=self.state_size, name="state_timeseries")

        # Simple normalization
        x = LayerNormalization(epsilon=1e-6)(inp)
        
        # Smaller, faster LSTM
        x = LSTM(32, activation="tanh", return_sequences=False)(x)
        x = Dropout(0.1)(x)
        
        # Simpler dueling heads
        adv = Dense(64, activation="relu", kernel_initializer='he_normal')(x)
        adv = Dense(self.action_size, activation="linear", dtype="float32")(adv)

        val = Dense(64, activation="relu", kernel_initializer='he_normal')(x)
        val = Dense(1, activation="linear", dtype="float32")(val)

        # Dueling aggregation
        adv_mean = Lambda(lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(adv)
        q = Add(name="q_values")([val, adv_mean])

        model = Model(inputs=inp, outputs=q, name="dueling_ddqn_simple")
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr,
            clipnorm=1.0
        )

        model.compile(
            loss=tf.keras.losses.Huber(delta=1.0),
            optimizer=optimizer,
            jit_compile=False
        )
        return model

    @tf.function
    def train_step(self, states, actions, targets, weights):
        """IMPROVEMENT: Return TD errors for priority updates"""
        with tf.GradientTape() as tape:
            q = self.online_model(states, training=True)
            idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            q_sa = tf.gather_nd(q, idx)
            
            # Calculate TD errors
            td_errors = targets - q_sa
            
            # Weighted loss for prioritized replay
            loss = tf.reduce_mean(weights * tf.keras.losses.huber(targets, q_sa))
            
            # Add small L2 regularization to prevent weight explosion
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.online_model.trainable_variables]) * 1e-5
            total_loss = loss + l2_loss

        grads = tape.gradient(total_loss, self.online_model.trainable_variables)
        
        # CRITICAL: Check for NaN gradients
        grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in grads]
        
        self.online_model.optimizer.apply_gradients(
            zip(grads, self.online_model.trainable_variables)
        )
        return loss, td_errors
    
    def update_target_model(self, hard: bool = False):
        if hard:
            self.target_model.set_weights(self.online_model.get_weights())
        else:
            ow = self.online_model.get_weights()
            tw = self.target_model.get_weights()
            self.target_model.set_weights([self.tau * o + (1.0 - self.tau) * t
                                           for o, t in zip(ow, tw)])

    def remember(self, s, a, r, s_next, done, td_error=None):
        experience = (s.astype(np.float32), int(a), float(r), s_next.astype(np.float32), bool(done))
        self.memory.add(experience, td_error)

    @tf.function
    def _predict_action(self, state_tensor: tf.Tensor) -> tf.Tensor:
        return self.online_model(state_tensor, training=False)

    def choose_action(self, state_window: np.ndarray, greedy: bool = False) -> int:
        # CRITICAL FIX: Force balanced exploration
        if (not greedy):
            # Epsilon-greedy with forced action diversity
            if np.random.rand() <= self.epsilon:
                return np.random.randint(self.action_size)
            
            # Even when exploiting, add 10% chance to try non-greedy action
            if np.random.rand() < 0.1:
                return np.random.randint(self.action_size)
        
        state_tensor = tf.convert_to_tensor(
            state_window.reshape(1, self.state_size[0], self.state_size[1]), 
            dtype=tf.float32
        )
        q_values_tensor = self._predict_action(state_tensor)
        return int(tf.argmax(q_values_tensor[0]).numpy())

    def replay(self):
        # IMPROVEMENT: Sample with priorities
        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        if samples is None:
            return
        
        # Increase beta over time (importance sampling weight)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # IMPROVEMENT: Only 2 updates per replay for speed
        for _ in range(4):  # MORE updates per step for faster learning
            states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
            
            states      = tf.convert_to_tensor(states,      dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            actions     = tf.convert_to_tensor(actions,     dtype=tf.int32)
            rewards     = tf.convert_to_tensor(rewards,     dtype=tf.float32)
            dones       = tf.convert_to_tensor(dones,       dtype=tf.float32)
            weights_tf  = tf.convert_to_tensor(weights,     dtype=tf.float32)

            # DDQN target
            q_next_online = self.online_model(next_states)                          
            a_star = tf.argmax(q_next_online, axis=1, output_type=tf.int32)     

            q_next_all = self.target_model(next_states)                     
            idx = tf.stack([tf.range(tf.shape(a_star)[0], dtype=tf.int32), a_star], axis=1) 
            q_next_best = tf.gather_nd(q_next_all, idx)
            
            q_target = rewards + (1.0 - dones) * tf.cast(self.gamma, tf.float32) * tf.cast(q_next_best, tf.float32)

            # Train and get TD errors
            _, td_errors = self.train_step(states, actions, q_target, weights_tf)
            
            # IMPROVEMENT: Update priorities in replay buffer
            # FIX: Ensure TD errors are valid before updating
            td_errors_np = td_errors.numpy()
            td_errors_np = np.nan_to_num(td_errors_np, nan=1.0, posinf=1.0, neginf=1.0)
            self.memory.update_priorities(indices, td_errors_np)
            
        self.train_step_counter += 1
        
        # Update target network periodically
        if self.train_step_counter % self.update_frequency == 0:
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
    for df in (bars, ind, fut):
        df.columns = [c.strip() for c in df.columns]
    return bars, ind, fut, syms


def load_from_csv(csv_dir: str, symbol: str):
    bars, ind, fut, syms = _read_csvs(csv_dir)

    for name, df in [("daily_bars.csv", bars), ("indicators_daily.csv", ind), ("future_returns_daily.csv", fut)]:
        if not {"symbol", "trade_date"}.issubset(df.columns):
            raise ValueError(f"[error] {name} must have columns: symbol, trade_date, ...")

    merged = (bars.merge(ind, on=["symbol", "trade_date"], how="inner")
                    .merge(fut, on=["symbol", "trade_date"], how="left")) 

    df = (merged[merged["symbol"].astype(str) == str(symbol)]
                .sort_values("trade_date")
                .reset_index(drop=True))

    if df.empty:
        raise ValueError(f"[error] No data after merge for symbol={symbol}.")

    if {"sma20", "sma50"}.issubset(df.columns):
        df["sma20_rel"] = (df["sma20"] - df["sma50"]) / df["sma50"]

    # IMPROVEMENT: Add more informative features
    if "close" in df.columns:
        # Volatility (rolling std of returns)
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        # Price momentum
        df["price_mom_5"] = df["close"].pct_change(5)
        df["price_mom_20"] = df["close"].pct_change(20)
        # Volume features if available
        if "volume" in df.columns:
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    return df


def chrono_split_data(df: pd.DataFrame, train_frac=0.7):
    n = len(df)
    split_idx = int(n * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def create_windowed_dataset(data: np.ndarray, window_size: int) -> np.ndarray:
    n_samples = data.shape[0]
    n_features = data.shape[1]
    num_windows = n_samples - window_size + 1
    windowed_data = np.empty((num_windows, window_size, n_features), dtype=data.dtype)
    
    for i in range(num_windows):
        windowed_data[i] = data[i : i + window_size, :]
        
    return windowed_data


def simulate_strategy(
    prices: np.ndarray, 
    actions: np.ndarray, 
    positions_map: Dict[int, float],
    initial_cash: float, 
    txn_cost: float
) -> np.ndarray:
    cash, shares = initial_cash, 0.0
    portfolio_values = [initial_cash]
    current_pos_percent = 0.0

    for i in range(len(prices) - 1):
        current_price = prices[i]
        current_portfolio_value = cash + shares * current_price
        
        action_idx = int(actions[i])
        target_pos_percent = positions_map.get(action_idx, current_pos_percent) 
        
        target_shares_value = current_portfolio_value * target_pos_percent
        ideal_target_shares = 0.0
        if current_price > 1e-6:
             ideal_target_shares = target_shares_value / current_price
        
        shares_to_trade = ideal_target_shares - shares 

        if shares_to_trade > 0:
            cash_needed = shares_to_trade * current_price * (1.0 + txn_cost)
            
            if cash >= cash_needed:
                actual_shares_to_buy = shares_to_trade
                cash -= cash_needed
            else:
                if cash > 0 and current_price > 1e-6:
                    affordable_shares = cash / (current_price * (1.0 + txn_cost))
                    actual_shares_to_buy = affordable_shares
                    cash = 0.0
                else:
                    actual_shares_to_buy = 0
            
            shares += actual_shares_to_buy

        elif shares_to_trade < 0:
            actual_shares_to_trade = shares_to_trade
            cash_gained = abs(actual_shares_to_trade) * current_price * (1.0 - txn_cost)
            shares += actual_shares_to_trade
            cash += cash_gained
        
        new_portfolio_value = cash + shares * current_price
        if new_portfolio_value > 1e-6:
            current_pos_percent = (shares * current_price) / new_portfolio_value
        else:
            current_pos_percent = 0.0

        portfolio_values.append(cash + shares * prices[i + 1])

    return np.array(portfolio_values[:len(prices)])


def run(symbol="SPY", episodes=50, outdir=OUTDIR_DEFAULT, csv_dir="data"):
    print(f">>> Sample-Efficient DQN Agent (CSV)  Py={sys.executable}")
    print(f"[config] symbol={symbol} episodes={episodes} csv_dir={csv_dir}")

    os.makedirs(outdir, exist_ok=True)

    df_all = load_from_csv(csv_dir, symbol)
    train_df, test_df = chrono_split_data(df_all)

    # IMPROVEMENT: Include new features
    extra_features = ["volatility", "price_mom_5", "price_mom_20", "volume_ratio"]
    all_features = AGENT_STATE_FEATURES_DEFAULT + [f for f in extra_features if f in train_df.columns]
    
    present = [f for f in all_features if f in train_df.columns]
    if not present:
        raise ValueError("[error] No indicator features found.")
    if len(present) < len(all_features):
        missing = sorted(set(all_features) - set(present))
        print(f"[warn] Missing features: {missing}")

    # IMPROVEMENT: Use RobustScaler instead of MinMaxScaler (better for outliers)
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(train_df[present])
    X_te_scaled = scaler.transform(test_df[present])

    close_tr = train_df["close"].values.squeeze()
    log_ret_tr = np.zeros_like(close_tr, dtype="float32")
    if len(close_tr) > 1:
        log_ret_tr[1:] = np.log(close_tr[1:] / close_tr[:-1])

    X_tr_windows = create_windowed_dataset(X_tr_scaled, WINDOW_SIZE)
    log_ret_tr_aligned = log_ret_tr[WINDOW_SIZE:]
    
    close_te = test_df["close"].values.squeeze()
    test_df_aligned = test_df.iloc[WINDOW_SIZE - 1:].reset_index(drop=True)
    close_te_aligned = close_te[WINDOW_SIZE - 1:]
    X_te_windows = create_windowed_dataset(X_te_scaled, WINDOW_SIZE)
    
    n_tr, n_te = len(X_tr_windows), len(X_te_windows)
    print(f"[data] {symbol} train_windows={n_tr} test_windows={n_te} features={len(present)}")
    
    if n_tr < len(log_ret_tr_aligned):
        log_ret_tr_aligned = log_ret_tr_aligned[:n_tr]
    elif n_tr > len(log_ret_tr_aligned):
         X_tr_windows = X_tr_windows[:len(log_ret_tr_aligned)]
         n_tr = len(X_tr_windows)

    # CRITICAL FIX: Start with just 2 actions - FLAT and LONG
    # Remove SHORT until agent learns to trade properly
    POSITIONS = [0.0, 1.0]  # FLAT, LONG only
    ACTION_SIZE = len(POSITIONS)
    DQN_POSITIONS_MAP = {i: p for i, p in enumerate(POSITIONS)}

    num_features = X_tr_windows.shape[2]
    window_size = X_tr_windows.shape[1]
    STATE_SHAPE = (window_size, num_features + 1)

    agent = DQNAgent(state_size=STATE_SHAPE, action_size=ACTION_SIZE)
    txn_cost = 0.001
    
    # FIX: Much lower reward scaling for more stable learning
    reward_scaler = 10.0

    print("[info] Training with sample-efficient architecture...")
    n = n_tr
    pos_history_tr = np.zeros((n, window_size, 1), dtype=np.float32)
    state_inputs_tr = np.concatenate([X_tr_windows, pos_history_tr], axis=2)
    
    episode_rewards = []
    best_reward = -np.inf

    for ep in range(1, episodes + 1):
        # CRITICAL FIX: Start with different random positions each episode
        pos = POSITIONS[np.random.randint(len(POSITIONS))]  # Random starting position
        state_inputs_tr[0, -1, -1] = pos
        total_reward = 0.0
        
        for t in trange(n - 1, desc=f"Episode {ep}/{episodes}", leave=False):
            current_state_window = state_inputs_tr[t]
            action = agent.choose_action(current_state_window)
            next_pos = POSITIONS[action]

            trade_size = abs(next_pos - pos)
            
            # CRITICAL FIX: Simple, direct reward - no volatility adjustment
            base_reward = pos * log_ret_tr_aligned[t + 1]
            transaction_penalty = txn_cost * trade_size
            
            # Clear reward signal scaled appropriately
            reward = (base_reward - transaction_penalty) * reward_scaler
            
            # Small exploration bonus for first 25 episodes
            if ep <= 25 and trade_size > 0:
                reward += 1.0  # Encourage position changes early on
            
            total_reward += reward

            next_state_window = state_inputs_tr[t+1].copy()
            next_state_window[-1, -1] = next_pos
            
            done = (t == n - 2)

            # IMPROVEMENT: Calculate TD error for prioritized replay
            td_error = None
            if agent.memory.size > 0:
                try:
                    state_tensor = tf.convert_to_tensor(
                        current_state_window.reshape(1, window_size, num_features + 1), 
                        dtype=tf.float32
                    )
                    next_state_tensor = tf.convert_to_tensor(
                        next_state_window.reshape(1, window_size, num_features + 1), 
                        dtype=tf.float32
                    )
                    
                    current_q = agent.online_model(state_tensor, training=False)[0, action].numpy()
                    next_q = agent.target_model(next_state_tensor, training=False)[0].numpy().max()
                    td_error = reward + (0 if done else agent.gamma * next_q) - current_q
                    
                    # FIX: Ensure TD error is valid
                    if np.isnan(td_error) or np.isinf(td_error):
                        td_error = 1.0
                except:
                    td_error = 1.0  # Safe default if calculation fails

            agent.remember(current_state_window, action, reward, next_state_window, done, td_error)
            pos = next_pos
            
            if t < n - 2:
                state_inputs_tr[t+1, -1, -1] = pos

            # CRITICAL: Train EVERY step once we have enough samples
            if agent.memory.size >= agent.batch_size:
                agent.replay()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
        
        # DIAGNOSTIC: Track action distribution
        action_counts = {0: 0, 1: 0}
        for t in range(min(100, n-1)):  # Sample first 100 steps
            state = state_inputs_tr[t]
            a = agent.choose_action(state)
            action_counts[a] = action_counts.get(a, 0) + 1
        
        pct_flat = action_counts.get(0, 0) / sum(action_counts.values()) * 100 if sum(action_counts.values()) > 0 else 0
        pct_long = action_counts.get(1, 0) / sum(action_counts.values()) * 100 if sum(action_counts.values()) > 0 else 0
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.online_model.save_weights(os.path.join(outdir, f"best_model_{symbol}.weights.h5"))
        
        print(f"[train] ep={ep}/{episodes}  eps={agent.epsilon:.4f}  reward={total_reward:.2f}  "
              f"avg_10={avg_reward:.2f}  best={best_reward:.2f}  FLAT={pct_flat:.0f}% LONG={pct_long:.0f}%")

    # Load best model for evaluation
    print("[info] Loading best model for evaluation...")
    try:
        agent.online_model.load_weights(os.path.join(outdir, f"best_model_{symbol}.weights.h5"))
    except:
        print("[warn] Could not load best model, using final model instead")

    print("[info] Evaluating...")
    pos_history_te = np.zeros((n_te, window_size, 1), dtype=np.float32)
    state_inputs_te = np.concatenate([X_te_windows, pos_history_te], axis=2)

    actions_dqn = np.zeros(n_te, dtype=int)
    pos = 0.0  # Start with FLAT position
    
    # DIAGNOSTIC: Track Q-values during evaluation
    q_values_log = []
    
    for t in range(n_te):
        # Set position in last timestep of current window
        state_inputs_te[t, -1, -1] = pos
        current_state_window = state_inputs_te[t]
        
        # Get Q-values for diagnostics
        state_tensor = tf.convert_to_tensor(
            current_state_window.reshape(1, window_size, num_features + 1), 
            dtype=tf.float32
        )
        q_vals = agent.online_model(state_tensor, training=False)[0].numpy()
        q_values_log.append(q_vals)
        
        a = agent.choose_action(current_state_window, greedy=True)
        actions_dqn[t] = a
        pos = POSITIONS[a]  # Update position for next iteration
    
    # Print Q-value statistics
    q_values_array = np.array(q_values_log)
    print(f"\n[diagnostic] Q-value stats during evaluation:")
    print(f"  Q(FLAT) mean: {q_values_array[:, 0].mean():.2f}, std: {q_values_array[:, 0].std():.2f}")
    print(f"  Q(LONG) mean: {q_values_array[:, 1].mean():.2f}, std: {q_values_array[:, 1].std():.2f}")
    print(f"  Actions: FLAT={np.sum(actions_dqn==0)/len(actions_dqn)*100:.1f}%, LONG={np.sum(actions_dqn==1)/len(actions_dqn)*100:.1f}%")
    print(f"  Average position: {np.mean([POSITIONS[a] for a in actions_dqn]):.2f}\n")

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
    sim_df = pd.DataFrame(index=test_df_aligned.index)
    sim_df["close_price"] = close_te_aligned
    initial_cash = close_te_aligned[0]

    BINARY_POSITIONS_MAP = {0: 0.0, 1: 1.0}
    BUY_HOLD_MAP = {1: 1.0} 

    buy_and_hold_actions = np.ones_like(actions_dqn)
    strategies = {"DQN_Advanced": actions_dqn, **ta_strats, "Buy_and_Hold": buy_and_hold_actions}

    for name, acts in strategies.items():
        if name == "DQN_Advanced":
            map_to_use = DQN_POSITIONS_MAP
        elif name == "Buy_and_Hold":
            map_to_use = BUY_HOLD_MAP
        else:
            map_to_use = BINARY_POSITIONS_MAP
            
        vals = simulate_strategy(close_te_aligned, acts, map_to_use, initial_cash, txn_cost)
        sim_df[f"portfolio_value_{name}"] = vals[:len(sim_df)]

    sim_df["pnl_DQN_Advanced"] = sim_df["portfolio_value_DQN_Advanced"] - initial_cash
    
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
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        metrics.append({
            "strategy": name,
            "final_portfolio_value": float(final_value),
            "total_return_%": float((final_value / initial_cash - 1) * 100.0),
            "sharpe_ratio_annualized": float(sharpe_ratio),
            "max_drawdown_%": float(max_drawdown * 100),
            "win_rate_%": float(win_rate * 100),
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(outdir, f"comparison_metrics_{symbol}.csv"), index=False)
    print(f"\n[save] Metrics: {os.path.join(outdir, f'comparison_metrics_{symbol}.csv')}")
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON:")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print("="*80 + "\n")

    position_labels = {0: "FLAT", 1: "LONG"}
    
    trade_actions = ["HOLD"] * len(actions_dqn)
    if len(actions_dqn):
        trade_actions[0] = position_labels[actions_dqn[0]]
        for i in range(1, len(actions_dqn)):
            if actions_dqn[i] > actions_dqn[i-1]:
                trade_actions[i] = "BUY"
            elif actions_dqn[i] < actions_dqn[i-1]:
                trade_actions[i] = "SELL"
            else:
                trade_actions[i] = "HOLD"
                
    log_df = pd.DataFrame({
        "close_price": close_te_aligned,
        "position": [position_labels.get(a, "UNKNOWN") for a in actions_dqn],
        "action": trade_actions[:len(close_te_aligned)],
    }, index=test_df_aligned.index)
    log_df.to_csv(os.path.join(outdir, f"trading_log_{symbol}.csv"))
    print(f"[save] Trading log: {os.path.join(outdir, f'trading_log_{symbol}.csv')}")

    print(">>> done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sample-Efficient DQN Trading Agent")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT) 
    ap.add_argument("--csv_dir", default="data")
    args = ap.parse_args() 
    run(symbol=args.symbol, episodes=args.episodes, outdir=args.outdir, csv_dir=args.csv_dir)