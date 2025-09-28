import os
import math
import warnings
from datetime import date
from typing import Optional, List

import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=FutureWarning)

# Update DB_URL as needed
DB_URL = os.getenv("DB_URL", "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4")
engine = create_engine(DB_URL, future=True)

# ----------------------------- DDL ---------------------------------
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
    "CREATE INDEX idx_indicators_symbol_date ON indicators_daily (symbol, trade_date)"
]

UPSERT_SQL = """
INSERT INTO indicators_daily (
  symbol, trade_date, sma20, sma50, sma200, ema12, ema26,
  macd, macd_signal, macd_hist, rsi14, atr14,
  bb_mid, bb_upper, bb_lower,
  adx14, di_plus, di_minus,
  stoch_k, stoch_d, obv
) VALUES (
  :symbol, :trade_date, :sma20, :sma50, :sma200, :ema12, :ema26,
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

def ensure_schema():
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL)
        for stmt in INDEX_SQLS:
            try:
                conn.exec_driver_sql(stmt)
            except Exception as e:
                msg = str(e).lower()
                if "duplicate key name" in msg or "1061" in msg:
                    pass
                else:
                    raise

# ------------------------- Helpers ---------------------------------
def get_symbols() -> List[str]:
    """Get symbols that have data in daily_bars."""
    with engine.begin() as conn:
        sql = "SELECT DISTINCT symbol FROM daily_bars"
        return [r[0] for r in conn.execute(text(sql)).fetchall()]

def get_start_date_for_symbol(symbol: str, fallback_start: str) -> str:
    """Incremental: continue from last computed date + 1 or fallback_start if no records."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT MAX(trade_date) FROM indicators_daily WHERE symbol=:s"),
            {"s": symbol}
        ).scalar()
    if row is None:
        return fallback_start
    return (pd.to_datetime(row) + pd.Timedelta(days=1)).date().isoformat()

def read_price_slice(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    end_date = end_date or date.today().isoformat()
    sql = """
      SELECT trade_date, `open`, high, low, `close`, volume
      FROM daily_bars
      WHERE symbol=:s AND trade_date BETWEEN :start AND :end
      ORDER BY trade_date
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"s": symbol, "start": start_date, "end": end_date})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df

# ------------------------- Indicator Calculation -------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df_i = df.copy()
    # SMA and EMA
    df_i["sma20"]  = ta.sma(df_i["close"], length=20)
    df_i["sma50"]  = ta.sma(df_i["close"], length=50)
    df_i["sma200"] = ta.sma(df_i["close"], length=200)

    df_i["ema12"]  = ta.ema(df_i["close"], length=12)
    df_i["ema26"]  = ta.ema(df_i["close"], length=26)

    # MACD
    macd = ta.macd(df_i["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df_i["macd"]        = macd.iloc[:, 0]
        df_i["macd_signal"] = macd.iloc[:, 1]
        df_i["macd_hist"]   = macd.iloc[:, 2]

    # RSI
    df_i["rsi14"] = ta.rsi(df_i["close"], length=14)

    # ATR
    df_i["atr14"] = ta.atr(df_i["high"], df_i["low"], df_i["close"], length=14)

    # Bollinger Bands
    def _bollinger(close, length=20, nstd=2.0):
        try:
            bb = ta.bbands(close, length=length, std=nstd)
            if isinstance(bb, pd.DataFrame) and not bb.empty:
                bb_lower = bb.filter(regex=r"^BBL_").iloc[:, 0]
                bb_mid   = bb.filter(regex=r"^BBM_").iloc[:, 0]
                bb_upper = bb.filter(regex=r"^BBU_").iloc[:, 0]
                return bb_mid, bb_upper, bb_lower
            raise ValueError("Empty bbands result")
        except Exception:
            mid = close.rolling(length).mean()
            std = close.rolling(length).std(ddof=0)
            upper = mid + nstd * std
            lower = mid - nstd * std
            return mid, upper, lower

    df_i["bb_mid"], df_i["bb_upper"], df_i["bb_lower"] = _bollinger(df_i["close"], length=20, nstd=2.0)

    # ADX
    adx = ta.adx(df_i["high"], df_i["low"], df_i["close"], length=14)
    if adx is not None:
        df_i["adx14"]   = adx.iloc[:, 0]
        df_i["di_plus"] = adx.iloc[:, 1]
        df_i["di_minus"]= adx.iloc[:, 2]

    # Stochastic Oscillator
    stoch = ta.stoch(df_i["high"], df_i["low"], df_i["close"], k=14, d=3, smooth_k=3)
    if stoch is not None:
        df_i["stoch_k"] = stoch.iloc[:, 0]
        df_i["stoch_d"] = stoch.iloc[:, 1]

    # OBV
    df_i["obv"] = ta.obv(df_i["close"], df_i["volume"])

    cols = [
        "trade_date","sma20","sma50","sma200","ema12","ema26","macd","macd_signal","macd_hist",
        "rsi14","atr14","bb_mid","bb_upper","bb_lower","adx14","di_plus","di_minus",
        "stoch_k","stoch_d","obv"
    ]
    return df_i[cols]

def upsert_indicators(symbol: str, df_i: pd.DataFrame):
    if df_i.empty:
        return
    records = []
    for r in df_i.itertuples(index=False):
        rec = dict(
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
        )
        records.append(rec)
    with engine.begin() as conn:
        conn.execute(text(UPSERT_SQL), records)

def _nn(x):
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

# ----------------------------- Main --------------------------------
def build_indicators(start_if_new: str = "2015-09-23", end: Optional[str] = None, symbols: Optional[List[str]] = None):
    ensure_schema()
    if symbols is None:
        symbols = get_symbols()

    for i, sym in enumerate(symbols, 1):
        try:
            start = get_start_date_for_symbol(sym, start_if_new)
            if end is not None and pd.to_datetime(end) < pd.to_datetime(start):
                continue
            print(f"[{i}/{len(symbols)}] {sym}  (from {start})")
            px = read_price_slice(sym, start, end)
            if px.empty:
                continue
            df_i = compute_indicators(px)
            upsert_indicators(sym, df_i)
        except Exception as e:
            print(f"Error on {sym}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build/update technical indicators into MySQL.")
    ap.add_argument("--end", default=None, help="Optional end date (default=today)")
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional list of symbols to process")
    args = ap.parse_args()
    build_indicators(start_if_new="2015-09-23", end=args.end, symbols=args.symbols)
