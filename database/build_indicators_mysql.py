# build_indicators_mysql.py
import os
import math
import warnings
from datetime import date
from typing import Optional, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=FutureWarning)

# ---- DB URL (override via env var DB_URL) -----------------------------------
DB_URL = os.getenv(
    "DB_URL",
    "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4",
)
engine = create_engine(DB_URL, future=True)

# ---- DDL: indicators table (with forward returns) ---------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS indicators_daily (
  symbol        VARCHAR(16) NOT NULL,
  trade_date    DATE NOT NULL,
  sma20         DECIMAL(18,6) NULL,
  sma50         DECIMAL(18,6) NULL,
  sma200        DECIMAL(18,6) NULL,
  ema12         DECIMAL(18,6) NULL,
  ema26         DECIMAL(18,6) NULL,
  macd          DECIMAL(18,6) NULL,
  macd_signal   DECIMAL(18,6) NULL,
  macd_hist     DECIMAL(18,6) NULL,
  rsi14         DECIMAL(18,6) NULL,
  atr14         DECIMAL(18,6) NULL,
  bb_mid        DECIMAL(18,6) NULL,
  bb_upper      DECIMAL(18,6) NULL,
  bb_lower      DECIMAL(18,6) NULL,
  adx14         DECIMAL(18,6) NULL,
  di_plus       DECIMAL(18,6) NULL,
  di_minus      DECIMAL(18,6) NULL,
  stoch_k       DECIMAL(18,6) NULL,
  stoch_d       DECIMAL(18,6) NULL,
  obv           DECIMAL(20,0) NULL,
  PRIMARY KEY (symbol, trade_date),
  CONSTRAINT fk_ind_symbol FOREIGN KEY (symbol) REFERENCES symbols(symbol)
    ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;
"""

INDEX_SQLS = [
    "CREATE INDEX idx_indicators_date ON indicators_daily (trade_date)",
    "CREATE INDEX idx_indicators_symbol_date ON indicators_daily (symbol, trade_date)",
]

UPSERT_SQL = """
INSERT INTO indicators_daily (
  symbol, trade_date,
  sma20, sma50, sma200, ema12, ema26,
  macd, macd_signal, macd_hist, rsi14, atr14,
  bb_mid, bb_upper, bb_lower,
  adx14, di_plus, di_minus,
  stoch_k, stoch_d, obv,
  fwd_ret_21, fwd_ret_63, fwd_ret_126
) VALUES (
  :symbol, :trade_date,
  :sma20, :sma50, :sma200, :ema12, :ema26,
  :macd, :macd_signal, :macd_hist, :rsi14, :atr14,
  :bb_mid, :bb_upper, :bb_lower,
  :adx14, :di_plus, :di_minus,
  :stoch_k, :stoch_d, :obv
)
ON DUPLICATE KEY UPDATE
  sma20=VALUES(sma20), sma50=VALUES(sma50), sma200=VALUES(sma200),
  ema12=VALUES(ema12), ema26=VALUES(ema26),
  macd=VALUES(macd), macd_signal=VALUES(macd_signal), macd_hist=VALUES(macd_hist),
  rsi14=VALUES(rsi14), atr14=VALUES(atr14),
  bb_mid=VALUES(bb_mid), bb_upper=VALUES(bb_upper), bb_lower=VALUES(bb_lower),
  adx14=VALUES(adx14), di_plus=VALUES(di_plus), di_minus=VALUES(di_minus),
  stoch_k=VALUES(stoch_k), stoch_d=VALUES(stoch_d), obv=VALUES(obv);
"""

# -----------------------------------------------------------------------------
# Helper: Wilder smoothing (EMA with alpha = 1/length)
def wilder_smooth(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

# Indicators implemented in numpy/pandas only
def compute_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = compute_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_rsi(close: pd.Series, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_smooth(gain, length)
    avg_loss = wilder_smooth(loss, length)
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = wilder_smooth(tr, length)
    return atr

def compute_bbands(close: pd.Series, length=20, nstd=2.0):
    mid = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + nstd * std
    lower = mid - nstd * std
    return mid, upper, lower

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, length=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = compute_atr(high, low, close, length=1)  # true range (not smoothed)
    tr_s = wilder_smooth(tr, length)
    plus_dm_s = wilder_smooth(plus_dm, length)
    minus_dm_s = wilder_smooth(minus_dm, length)

    di_plus = 100 * (plus_dm_s / tr_s.replace(0, np.nan))
    di_minus = 100 * (minus_dm_s / tr_s.replace(0, np.nan))
    dx = ( (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) ) * 100
    adx = wilder_smooth(dx, length)
    return adx, di_plus, di_minus

def compute_stoch(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3, smooth_k=3):
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    raw_k = 100 * (close - lowest) / (highest - lowest)
    stoch_k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    stoch_d = stoch_k.rolling(d, min_periods=d).mean()
    return stoch_k, stoch_d

def compute_obv(close: pd.Series, volume: pd.Series):
    sign = np.sign(close.diff().fillna(0))
    return (volume * sign).cumsum()


# -----------------------------------------------------------------------------
def ensure_schema():
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL)
        for stmt in INDEX_SQLS:
            try:
                conn.exec_driver_sql(stmt)
            except Exception as e:
                if "1061" in str(e) or "duplicate key name" in str(e).lower():
                    pass
                else:
                    raise

def list_symbols() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM daily_bars")).fetchall()
    return [r[0] for r in rows]

def last_indicator_date(symbol: str) -> Optional[pd.Timestamp]:
    with engine.begin() as conn:
        d = conn.execute(
            text("SELECT MAX(trade_date) FROM indicators_daily WHERE symbol=:s"),
            {"s": symbol},
        ).scalar()
    return None if d is None else pd.to_datetime(d)

def read_prices(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    end = end or date.today().isoformat()
    sql = """
      SELECT trade_date, `open`, high, low, `close`, volume
      FROM daily_bars
      WHERE symbol=:s AND trade_date BETWEEN :start AND :end
      ORDER BY trade_date
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"s": symbol, "start": start, "end": end})
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    out["sma20"]  = compute_sma(close, 20)
    out["sma50"]  = compute_sma(close, 50)
    out["sma200"] = compute_sma(close, 200)
    out["ema12"]  = compute_ema(close, 12)
    out["ema26"]  = compute_ema(close, 26)

    macd, macds, macdh = compute_macd(close, 12, 26, 9)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd, macds, macdh

    out["rsi14"] = compute_rsi(close, 14)
    out["atr14"] = compute_atr(high, low, close, 14)

    bb_mid, bb_upper, bb_lower = compute_bbands(close, 20, 2.0)
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bb_mid, bb_upper, bb_lower

    adx, di_p, di_m = compute_adx(high, low, close, 14)
    out["adx14"], out["di_plus"], out["di_minus"] = adx, di_p, di_m

    st_k, st_d = compute_stoch(high, low, close, 14, 3, 3)
    out["stoch_k"], out["stoch_d"] = st_k, st_d

    out["obv"] = compute_obv(close, vol)

    out.insert(0, "trade_date", df["trade_date"].dt.date)
    return out

def upsert(symbol: str, inds: pd.DataFrame, batch: int = 1000):
    if inds.empty:
        return
    recs = []
    for r in inds.itertuples(index=False):
        recs.append(dict(
            symbol=symbol,
            trade_date=r.trade_date,
            sma20=_nn(r.sma20),  sma50=_nn(r.sma50),  sma200=_nn(r.sma200),
            ema12=_nn(r.ema12),  ema26=_nn(r.ema26),
            macd=_nn(r.macd),    macd_signal=_nn(r.macd_signal), macd_hist=_nn(r.macd_hist),
            rsi14=_nn(r.rsi14),  atr14=_nn(r.atr14),
            bb_mid=_nn(r.bb_mid), bb_upper=_nn(r.bb_upper), bb_lower=_nn(r.bb_lower),
            adx14=_nn(r.adx14),  di_plus=_nn(r.di_plus), di_minus=_nn(r.di_minus),
            stoch_k=_nn(r.stoch_k), stoch_d=_nn(r.stoch_d),
            obv=None if pd.isna(r.obv) else int(r.obv),
        ))
        if len(recs) >= batch:
            _flush(recs); recs.clear()
    if recs:
        _flush(recs)

def _flush(records):
    with engine.begin() as conn:
        conn.execute(text(UPSERT_SQL), records)

def _nn(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

# -----------------------------------------------------------------------------
def build_indicators(start_if_new="2015-09-23", end: Optional[str] = None, symbols: Optional[List[str]] = None):
    ensure_schema()
    syms = symbols or list_symbols()
    for i, sym in enumerate(syms, 1):
        try:
            last = last_indicator_date(sym)
            start = start_if_new if last is None else (pd.to_datetime(last) + pd.Timedelta(days=1)).date().isoformat()
            if end is not None and pd.to_datetime(end) < pd.to_datetime(start):
                continue
            px = read_prices(sym, start, end)
            print(f"[{i}/{len(syms)}] {sym}: rows={len(px)} from {start} -> {(end or date.today().isoformat())}")
            if px.empty:
                continue
            inds = compute_all_indicators(px)
            upsert(sym, inds)
        except Exception as e:
            print(f"Error on {sym}: {e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute indicators + forward returns (pure pandas/numpy) to MySQL.")
    ap.add_argument("--start-if-new", default="2015-09-23", help="Start date for symbols without indicators")
    ap.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional subset of symbols")
    args = ap.parse_args()
    build_indicators(start_if_new=args.start_if_new, end=args.end, symbols=args.symbols)
