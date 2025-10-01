import os, sys, warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= Config =========
DB_URL = os.getenv(
    "DB_URL",
    "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4"
)

# We’ll use a *small* subset of indicators as the RL state to keep the Q-table tractable.
STATE_FEATURES: List[str] = [
    "rsi14",           # 0-100
    "macd_hist",       # centered
    "stoch_k",         # 0-100
    "sma20", "sma50"   # will make sma20_rel = (sma20 - sma50)/sma50
]
# bins per state feature (same length as engineered feature list used in discretization below)
N_BINS = [5, 5, 5, 5]  # 4 features → 5^4 = 625 states

DEFAULT_HORIZON = 21
OUTDIR_DEFAULT = "ml_outputs"
RNG = np.random.default_rng(42)

# ========= Utilities =========
def table_exists(engine, name: str) -> bool:
    with engine.begin() as c:
        return c.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = :t
        """), {"t": name}).scalar() > 0

def load_panel(engine, symbol: str, horizon: int) -> pd.DataFrame:
    """Load features + target y from DB; index by trade_date."""
    use_fwd = table_exists(engine, "future_returns_daily")
    with engine.begin() as conn:
        feats = pd.read_sql(text(f"""
            SELECT trade_date, {", ".join(STATE_FEATURES)}
            FROM indicators_daily
            WHERE symbol=:s ORDER BY trade_date
        """), conn, params={"s": symbol})
        if feats.empty:
            return feats

        feats["trade_date"] = pd.to_datetime(feats["trade_date"])
        feats = feats.set_index("trade_date").sort_index()

        if use_fwd:
            tgt = pd.read_sql(text("""
                SELECT trade_date, fwd_return
                FROM future_returns_daily
                WHERE symbol=:s AND horizon_days=:h
                ORDER BY trade_date
            """), conn, params={"s": symbol, "h": horizon}).rename(columns={"fwd_return":"y"})
        else:
            col = f"fwd_ret_{horizon}"
            tgt = pd.read_sql(text(f"""
                SELECT trade_date, {col} AS y
                FROM indicators_daily WHERE symbol=:s ORDER BY trade_date
            """), conn, params={"s": symbol})

        tgt["trade_date"] = pd.to_datetime(tgt["trade_date"])
        tgt = tgt.set_index("trade_date").sort_index()

    df = feats.join(tgt, how="inner")
    # engineer sma20_rel and drop originals we don't need in the state vector
    df["sma20_rel"] = (df["sma20"] - df["sma50"]) / df["sma50"]
    df = df.drop(columns=["sma20", "sma50"])
    # rename state columns to the final set we use
    df = df.rename(columns={"macd_hist": "macd_h", "stoch_k": "stoch"})
    # keep only the final state features + y
    df = df[["rsi14", "macd_h", "stoch", "sma20_rel", "y"]]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # optional: clip extreme returns to stabilize reward
    df["y"] = df["y"].clip(-0.2, 0.2)
    return df

def chrono_split(df: pd.DataFrame, train_frac=0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df); split = int(n * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def compute_bins(train_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Use quantile-based bin edges per feature (robust)."""
    edges = {}
    # features in order we’ll use for states
    features = ["rsi14", "macd_h", "stoch", "sma20_rel"]
    for feat, k in zip(features, N_BINS):
        # k bins → k-1 internal edges
        qs = np.linspace(0.0, 1.0, num=k+1)[1:-1]
        edges[feat] = train_df[feat].quantile(qs).to_numpy()
    return edges

def to_state_idx(row: pd.Series, edges: Dict[str, np.ndarray]) -> int:
    """Map row of 4 engineered features → single integer state index."""
    feats = ["rsi14", "macd_h", "stoch", "sma20_rel"]
    bins = []
    bases = []
    for feat, k in zip(feats, N_BINS):
        # np.digitize returns bin index in [0..k] with our edges length k-1
        b = int(np.digitize(row[feat], edges[feat], right=False))
        # clip to [0..k-1]
        b = min(max(b, 0), k-1)
        bins.append(b)
        bases.append(k)
    # convert mixed-radix vector to scalar index
    idx = 0
    for b, base in zip(bins, bases):
        idx = idx * base + b
    return idx

# ========= Q-learning Agent =========
class QLearner:
    def __init__(self, n_states: int, n_actions: int = 2, alpha=0.1, gamma=0.99, epsilon=0.10, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions  # 0=flat, 1=long
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose(self, s: int, greedy: bool = False) -> int:
        if (not greedy) and RNG.random() < self.epsilon:
            return RNG.integers(self.n_actions)
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int):
        best_next = float(np.max(self.Q[s_next]))
        td_target = r + self.gamma * best_next
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ========= Metrics & Equity =========
def equity_from_actions(dates: pd.Index, y: np.ndarray, actions: np.ndarray, thresh=0.0) -> pd.DataFrame:
    # actions: 0=flat, 1=long → position return = action * y
    pos_ret = actions * y
    return pd.DataFrame(
        {"strategy": np.cumprod(1.0 + pos_ret), "buy_and_hold": np.cumprod(1.0 + y)},
        index=dates
    )

def ann_sharpe(daily: pd.Series, rf: float = 0.0) -> float:
    sd = daily.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((daily.mean() - rf) / sd * np.sqrt(252))

def max_drawdown(cum_curve: pd.Series) -> float:
    roll = cum_curve.cummax()
    dd = (cum_curve - roll) / roll
    return float(dd.min())

# ========= Main Runner =========
def run(symbol="AAPL", horizon=DEFAULT_HORIZON, episodes=10, outdir=OUTDIR_DEFAULT):
    print(">>> start Q-learning", sys.executable, flush=True)
    os.makedirs(outdir, exist_ok=True)
    engine = create_engine(DB_URL, future=True)

    df = load_panel(engine, symbol, horizon)
    if df.empty or len(df) < 300:
        raise RuntimeError(f"Not enough rows for {symbol}. Got {len(df)}")

    train_df, test_df = chrono_split(df, train_frac=0.7)
    print(f"[data] {symbol} {horizon}d  train={len(train_df)}  test={len(test_df)}  "
          f"range={df.index.min().date()}→{df.index.max().date()}")

    # Quantile bins from train set; build state indices
    edges = compute_bins(train_df)
    n_states = np.prod(N_BINS)

    train_states = train_df.apply(lambda r: to_state_idx(r, edges), axis=1).to_numpy()
    test_states  = test_df.apply(lambda r: to_state_idx(r, edges), axis=1).to_numpy()

    y_tr = train_df["y"].to_numpy()
    y_te = test_df["y"].to_numpy()

    # Q-learner
    agent = QLearner(n_states=n_states, n_actions=2, alpha=0.1, gamma=0.99, epsilon=0.10, epsilon_min=0.01, epsilon_decay=0.995)

    # ======== Training over episodes on the train window ========
    for ep in range(1, episodes + 1):
        total_reward = 0.0
        for t in range(len(train_states) - 1):
            s = int(train_states[t])
            a = agent.choose(s, greedy=False)          # explore/exploit
            # reward: next-day return if long; 0 if flat
            r = float(y_tr[t + 1] if a == 1 else 0.0)
            s_next = int(train_states[t + 1])
            agent.update(s, a, r, s_next)
            total_reward += r
        agent.decay()
        print(f"[train] ep={ep:02d}  eps={agent.epsilon:.3f}  total_reward={total_reward:.6f}", flush=True)

    # ======== Greedy evaluation on test window ========
    actions = np.zeros_like(test_states, dtype=int)
    for t in range(len(test_states)):
        actions[t] = agent.choose(int(test_states[t]), greedy=True)

    # Save predictions-like CSV (ground truth + actions and per-day strategy return)
    pos_ret = actions * y_te
    preds_df = pd.DataFrame(
        {
            "y_true": y_te,
            "action": actions,
            "position_return": pos_ret,
            "state": test_states,
        },
        index=test_df.index,
    )
    pred_path = os.path.join(outdir, f"predictions_{symbol}_{horizon}d_qlearn.csv")
    preds_df.to_csv(pred_path); print("[save]", pred_path, flush=True)

    # Equity curves
    eq = equity_from_actions(preds_df.index, preds_df["y_true"].values, preds_df["action"].values)
    eq_path = os.path.join(outdir, f"equity_{symbol}_{horizon}d_qlearn.csv")
    eq.to_csv(eq_path); print("[save]", eq_path, flush=True)

    # Metrics (Sharpe & MaxDD for both curves)
    bh_daily = eq["buy_and_hold"].pct_change().dropna()
    st_daily = eq["strategy"].pct_change().dropna()
    eq_metrics = {
        "symbol": symbol,
        "horizon_days": int(horizon),
        "model": "qlearn",
        "episodes": episodes,
        "test_start": str(eq.index.min().date()),
        "test_end": str(eq.index.max().date()),
        "final_buy_and_hold": float(eq["buy_and_hold"].iloc[-1]),
        "final_strategy": float(eq["strategy"].iloc[-1]),
        "buy_and_hold_Sharpe": ann_sharpe(bh_daily),
        "buy_and_hold_MaxDrawdown": max_drawdown(eq["buy_and_hold"]),
        "strategy_Sharpe": ann_sharpe(st_daily),
        "strategy_MaxDrawdown": max_drawdown(eq["strategy"]),
        "direction_accuracy": float((preds_df["position_return"] > 0).mean()),  # when we’re long, % positive outcomes
    }
    metrics_path = os.path.join(outdir, f"equity_metrics_{symbol}_{horizon}d_qlearn.csv")
    pd.DataFrame([eq_metrics]).to_csv(metrics_path, index=False)
    print("[save]", metrics_path, flush=True)
    print(">>> done", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Q-learning trader (CSV-only)")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    try:
        run(symbol=args.symbol, horizon=args.horizon, episodes=args.episodes, outdir=args.outdir)
    except Exception as e:
        import traceback
        print("[error]", e, flush=True)
        traceback.print_exc()
        sys.exit(1)
