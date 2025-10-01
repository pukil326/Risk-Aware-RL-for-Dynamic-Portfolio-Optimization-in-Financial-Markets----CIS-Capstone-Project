import os
import argparse
import numpy as np
import pandas as pd

# Headless-safe backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Metrics helpers (fallback if metrics CSV missing) ----------
def annualized_sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float:
    std = daily_returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((daily_returns.mean() - rf) / std * np.sqrt(252))

def max_drawdown(cum_curve: pd.Series) -> float:
    roll_max = cum_curve.cummax()
    dd = (cum_curve - roll_max) / roll_max
    return float(dd.min())


def load_metrics_or_compute(eq: pd.DataFrame,
                            symbol: str, horizon: int, model: str,
                            outdir: str) -> dict:
    """
    Try to load equity_metrics_<symbol>_<horizon>d_<model>.csv.
    If not found, compute Sharpe & MaxDD from equity curve.
    """
    metrics_path = os.path.join(outdir, f"equity_metrics_{symbol}_{horizon}d_{model}.csv")
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0].to_dict()
        return {
            "bh_sharpe": float(m.get("buy_and_hold_Sharpe", np.nan)),
            "bh_mdd": float(m.get("buy_and_hold_MaxDrawdown", np.nan)),
            "st_sharpe": float(m.get("strategy_Sharpe", np.nan)),
            "st_mdd": float(m.get("strategy_MaxDrawdown", np.nan)),
            "final_bh": float(m.get("final_buy_and_hold", np.nan)),
            "final_st": float(m.get("final_strategy", np.nan)),
        }

    # Compute from eq if metrics file missing
    bh_daily = eq["buy_and_hold"].pct_change().dropna()
    st_daily = eq["strategy"].pct_change().dropna()
    return {
        "bh_sharpe": annualized_sharpe(bh_daily),
        "bh_mdd": max_drawdown(eq["buy_and_hold"]),
        "st_sharpe": annualized_sharpe(st_daily),
        "st_mdd": max_drawdown(eq["strategy"]),
        "final_bh": float(eq["buy_and_hold"].iloc[-1]),
        "final_st": float(eq["strategy"].iloc[-1]),
    }


def _contiguous_spans(idx: pd.DatetimeIndex, mask: pd.Series):
    """
    Yield (start, end) timestamps for contiguous True regions in mask.
    """
    mask = mask.astype(bool).values
    dates = idx.values
    spans = []
    if len(mask) == 0:
        return spans
    in_span = False
    start = None
    for i, flag in enumerate(mask):
        if flag and not in_span:
            in_span = True
            start = dates[i]
        elif not flag and in_span:
            in_span = False
            spans.append((start, dates[i-1]))
    if in_span:
        spans.append((start, dates[-1]))
    return spans


def main(symbol: str, horizon: int, outdir: str):
    model = "qlearn"  # fixed for your RL paper

    # ---------- Load CSVs ----------
    pred_path = os.path.join(outdir, f"predictions_{symbol}_{horizon}d_{model}.csv")
    eq_path   = os.path.join(outdir, f"equity_{symbol}_{horizon}d_{model}.csv")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    if not os.path.exists(eq_path):
        raise FileNotFoundError(f"Missing equity file: {eq_path}")

    preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    eq    = pd.read_csv(eq_path,   index_col=0, parse_dates=True)

    # Ensure required cols exist
    for col in ["y_true", "action"]:
        if col not in preds.columns:
            raise ValueError(f"Column '{col}' not found in {pred_path}")

    # ---------- Load or compute metrics ----------
    M = load_metrics_or_compute(eq, symbol, horizon, model, outdir)

    # ---------- 1) Pred vs Actual with LONG shading ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(preds.index, preds["y_true"], label="Actual (fwd)")

    # Shade regions where agent is long (action==1)
    long_mask = (preds["action"] == 1)
    for (s, e) in _contiguous_spans(preds.index, long_mask):
        ax.axvspan(pd.to_datetime(s), pd.to_datetime(e), alpha=0.15, label=None)

    ax.set_title(f"{symbol} {horizon}d — RL Q-learning (Test)\n"
                 f"Long periods shaded (action=1)")
    ax.set_xlabel("trade_date"); ax.set_ylabel("Return")
    ax.legend()
    fig.autofmt_xdate()
    f1 = os.path.join(outdir, f"pred_vs_actual_{symbol}_{horizon}d_{model}.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print("[save]", f1)

    # ---------- 2) Equity curve with Sharpe/MDD summary ----------
    subtitle = (f"Strategy: Sharpe {M['st_sharpe']:.2f}, MDD {M['st_mdd']:.1%}, Final {M['final_st']:.2f}x | "
                f"Buy&Hold: Sharpe {M['bh_sharpe']:.2f}, MDD {M['bh_mdd']:.1%}, Final {M['final_bh']:.2f}x")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq.index, eq["strategy"], label="Strategy (long when action=1)")
    ax.plot(eq.index, eq["buy_and_hold"], label="Buy & Hold")
    ax.set_title(f"{symbol} {horizon}d — Equity Curve (Test) [Q-learning]\n{subtitle}")
    ax.set_xlabel("trade_date"); ax.set_ylabel("Cumulative multiple")
    ax.legend()
    fig.autofmt_xdate()
    f2 = os.path.join(outdir, f"equity_curve_{symbol}_{horizon}d_{model}.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print("[save]", f2)

    # ---------- 3) Drawdown plot (both curves) ----------
    dd_bh = (eq["buy_and_hold"] / eq["buy_and_hold"].cummax()) - 1.0
    dd_st = (eq["strategy"]    / eq["strategy"].cummax())      - 1.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dd_bh.index, dd_bh, label="Buy & Hold DD")
    ax.plot(dd_st.index, dd_st, label="Strategy DD")
    ax.set_title(f"{symbol} {horizon}d — Drawdown (Test) [Q-learning]")
    ax.set_xlabel("trade_date"); ax.set_ylabel("Drawdown")
    ax.legend()
    ax.set_ylim(min(dd_bh.min(), dd_st.min()) * 1.05, 0.02)
    fig.autofmt_xdate()
    f3 = os.path.join(outdir, f"drawdown_{symbol}_{horizon}d_{model}.png")
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    print("[save]", f3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render RL Q-learning plots from CSV outputs.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--outdir", default="ml_outputs")
    args = parser.parse_args()
    main(args.symbol, args.horizon, args.outdir)
