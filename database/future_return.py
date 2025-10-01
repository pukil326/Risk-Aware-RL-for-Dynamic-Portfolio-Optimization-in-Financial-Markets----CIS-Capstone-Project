import os
from datetime import date
from typing import Optional, List

import pandas as pd
from sqlalchemy import create_engine, text

# ---- DB URL (override in environment) ----
DB_URL = os.getenv(
    "DB_URL",
    "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4"
)
engine = create_engine(DB_URL, future=True)

# ---- DDL: narrow (tidy) table of forward returns ----
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS future_returns_daily (
  symbol        VARCHAR(16) NOT NULL,
  trade_date    DATE NOT NULL,
  horizon_days  INT NOT NULL,
  fwd_return    DECIMAL(18,6) NULL,
  PRIMARY KEY (symbol, trade_date, horizon_days),
  CONSTRAINT fk_fwd_symbol FOREIGN KEY (symbol) REFERENCES symbols(symbol)
    ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;
"""

INDEX_SQLS = [
    "CREATE INDEX idx_fwd_symbol_date ON future_returns_daily (symbol, trade_date)",
    "CREATE INDEX idx_fwd_date ON future_returns_daily (trade_date)",
    "CREATE INDEX idx_fwd_horizon ON future_returns_daily (horizon_days)"
]

UPSERT_SQL = """
INSERT INTO future_returns_daily
  (symbol, trade_date, horizon_days, fwd_return)
VALUES
  (:symbol, :trade_date, :h, :fwd_return)
ON DUPLICATE KEY UPDATE
  fwd_return = VALUES(fwd_return);
"""

# ----------------- helpers -----------------
def ensure_schema():
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL)
        for stmt in INDEX_SQLS:
            try:
                conn.exec_driver_sql(stmt)
            except Exception as e:
                # MySQL 1061 duplicate index; ignore
                if "1061" not in str(e):
                    raise

def list_symbols_with_data() -> List[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM daily_bars")).fetchall()
    return [r[0] for r in rows]

def last_computed_date(symbol: str, horizon: int) -> Optional[pd.Timestamp]:
    with engine.begin() as conn:
        d = conn.execute(
            text("""SELECT MAX(trade_date)
                    FROM future_returns_daily
                    WHERE symbol=:s AND horizon_days=:h"""),
            {"s": symbol, "h": horizon}
        ).scalar()
    return None if d is None else pd.to_datetime(d)

def read_prices(symbol: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    end = end_date or date.today().isoformat()
    sql = """
      SELECT trade_date, `close`, adj_close
      FROM daily_bars
      WHERE symbol=:s AND trade_date BETWEEN :start AND :end
      ORDER BY trade_date
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"s": symbol, "start": start_date, "end": end})
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # prefer adj_close; fallback to close
    price = df["adj_close"].where(df["adj_close"].notna(), df["close"])
    df["price"] = pd.to_numeric(price, errors="coerce")
    return df[["trade_date", "price"]].dropna()

def compute_forward_returns(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("trade_date").reset_index(drop=True)
    for h in horizons:
        out[f"fwd_ret_{h}"] = out["price"].shift(-h) / out["price"] - 1
    return out

def incremental_start(symbol: str, horizon: int, fallback_start: str) -> str:
    last = last_computed_date(symbol, horizon)
    if last is None:
        return fallback_start
    # continue from next day after last stored trade_date
    return (last + pd.Timedelta(days=1)).date().isoformat()

def upsert_returns(symbol: str, df_ret: pd.DataFrame, horizons: List[int], batch: int = 2000):
    if df_ret.empty:
        return
    records = []
    for row in df_ret.itertuples(index=False):
        td = row.trade_date.date()
        for h in horizons:
            val = getattr(row, f"fwd_ret_{h}")
            rec = dict(symbol=symbol, trade_date=td, h=int(h),
                       fwd_return=None if pd.isna(val) else float(val))
            records.append(rec)
            if len(records) >= batch:
                _flush(records)
                records.clear()
    if records:
        _flush(records)

def _flush(records):
    with engine.begin() as conn:
        conn.execute(text(UPSERT_SQL), records)

# ----------------- main -----------------
def build_future_returns(
    start_if_new: str = "2015-09-23",
    horizons: Optional[List[int]] = None,
    symbols: Optional[List[str]] = None,
    end: Optional[str] = None
):
    ensure_schema()
    horizons = horizons or [21, 63, 126]
    symbols = symbols or list_symbols_with_data()

    for i, sym in enumerate(symbols, 1):
        try:
            # compute the earliest needed start across horizons (per-horizon incremental)
            per_h_start = {}
            earliest = None
            for h in horizons:
                s = incremental_start(sym, h, start_if_new)
                per_h_start[h] = s
                if (earliest is None) or (pd.to_datetime(s) < pd.to_datetime(earliest)):
                    earliest = s

            px = read_prices(sym, earliest, end)
            print(f"[{i}/{len(symbols)}] {sym}: rows={len(px)} from {earliest} -> {(end or date.today().isoformat())}")
            if px.empty:
                continue

            df_full = compute_forward_returns(px, horizons)

            # For each horizon, keep only rows on/after that horizon's incremental start
            for h in horizons:
                df_h = df_full[df_full["trade_date"] >= pd.to_datetime(per_h_start[h])][
                    ["trade_date", f"fwd_ret_{h}"]
                ].copy()
                if not df_h.empty:
                    upsert_returns(sym, df_h, [h])
        except Exception as e:
            print(f"Error on {sym}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build/update future (forward) returns into MySQL.")
    ap.add_argument("--start-if-new", default="2015-09-23", help="Start date if symbol+horizon has no rows yet")
    ap.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD)")
    ap.add_argument("--horizons", nargs="*", type=int, default=[21, 63, 126], help="Trading-day horizons")
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional subset of symbols")
    args = ap.parse_args()
    build_future_returns(
        start_if_new=args.start_if_new,
        horizons=args.horizons,
        symbols=args.symbols,
        end=args.end
    )
