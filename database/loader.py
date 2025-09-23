import os
import sys
import time
import datetime as dt
import requests
from typing import List
from io import StringIO

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL", "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4")
engine = create_engine(DB_URL, future=True)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS symbols (
  symbol      VARCHAR(16) PRIMARY KEY,
  name        VARCHAR(255),
  sector      VARCHAR(128),
  is_index    TINYINT(1) NOT NULL DEFAULT 0,
  active      TINYINT(1) NOT NULL DEFAULT 1,
  created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS daily_bars (
  symbol      VARCHAR(16) NOT NULL,
  trade_date  DATE NOT NULL,
  `open`      DECIMAL(18,6) NOT NULL,
  high        DECIMAL(18,6) NOT NULL,
  low         DECIMAL(18,6) NOT NULL,
  `close`     DECIMAL(18,6) NOT NULL,
  volume      BIGINT UNSIGNED NOT NULL,
  adj_close   DECIMAL(18,6) NULL,
  PRIMARY KEY (symbol, trade_date),
  CONSTRAINT fk_daily_symbol FOREIGN KEY (symbol) REFERENCES symbols(symbol)
    ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;
"""

INDEX_SQLS = [
    "CREATE INDEX idx_daily_bars_date ON daily_bars (trade_date)",
    "CREATE INDEX idx_daily_bars_symbol_date ON daily_bars (symbol, trade_date)",
]

UPSERT_SYMBOL_MYSQL = """
INSERT INTO symbols (symbol, name, sector, is_index, active)
VALUES (:symbol, :name, :sector, :is_index, :active)
ON DUPLICATE KEY UPDATE
  name = COALESCE(VALUES(name), name),
  sector = COALESCE(VALUES(sector), sector),
  is_index = VALUES(is_index),
  active = VALUES(active);
"""

UPSERT_BAR_MYSQL = """
INSERT INTO daily_bars
  (symbol, trade_date, `open`, high, low, `close`, volume, adj_close)
VALUES
  (:symbol, :trade_date, :open, :high, :low, :close, :volume, :adj_close)
ON DUPLICATE KEY UPDATE
  `open` = VALUES(`open`),
  high = VALUES(high),
  low = VALUES(low),
  `close` = VALUES(`close`),
  volume = VALUES(volume),
  adj_close = VALUES(adj_close);
"""

def ensure_schema():
    from sqlalchemy import text
    with engine.begin() as conn:
        # create tables
        for stmt in SCHEMA_SQL.strip().split(";\n"):
            s = stmt.strip()
            if s:
                conn.exec_driver_sql(s + ";")
        # create indexes; ignore if they already exist
        for stmt in INDEX_SQLS:
            try:
                conn.exec_driver_sql(stmt + ";")
            except Exception as e:
                # MySQL error 1061 = "Duplicate key name"
                msg = str(e).lower()
                if "1061" in msg or "duplicate key name" in msg:
                    pass
                else:
                    raise

def fetch_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/91.0.4472.124 Safari/537.36"
    }

    # Fetch HTML using requests with a browser-like User-Agent
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch S&P 500 list from Wikipedia: HTTP Error {resp.status_code}: {resp.reason}")

    # Parse tables using pandas
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]  # The first table is the S&P 500 list
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B for Yahoo Finance
    return df[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "symbol", "Security": "name", "GICS Sector": "sector"}
    )

def upsert_symbols(df: pd.DataFrame):
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(
                text(UPSERT_SYMBOL_MYSQL),
                dict(
                    symbol=r["symbol"],
                    name=None if pd.isna(r["name"]) else str(r["name"]),
                    sector=None if pd.isna(r["sector"]) else str(r["sector"]),
                    is_index=0,
                    active=1,
                ),
            )
        # Add index and SPY for convenience
        for idxsym in ["^GSPC", "SPY"]:
            conn.execute(
                text(UPSERT_SYMBOL_MYSQL),
                dict(symbol=idxsym, name=None, sector=None, is_index=1 if idxsym.startswith("^") else 0, active=1),
            )

def download_and_store(symbol: str, start: str = "1990-01-01", end: str | None = None, pause_sec: float = 0.25):
    import numpy as np
    if end is None:
        end = dt.date.today().isoformat()

    print(f"Downloading {symbol} ...")
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)

    if df is None or df.empty:
        print(f"  No data for {symbol}")
        return

    # --- Normalize columns (handles occasional MultiIndex from yfinance) ---
    if isinstance(df.columns, pd.MultiIndex):
        # keep first level names: ('Open', symbol) -> 'Open'
        df.columns = [str(c[0]) for c in df.columns]

    # --- Standardize schema ---
    df = df.reset_index().rename(columns={
        "Date": "trade_date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # Coerce numerics safely
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Upsert using itertuples (scalars, no Series ambiguity) ---
    with engine.begin() as conn:
        for r in df.itertuples(index=False):
            conn.execute(
                text(UPSERT_BAR_MYSQL),
                dict(
                    symbol=symbol,
                    trade_date=r.trade_date,
                    open=None if pd.isna(r.open) else float(r.open),
                    high=None if pd.isna(r.high) else float(r.high),
                    low=None if pd.isna(r.low) else float(r.low),
                    close=None if pd.isna(r.close) else float(r.close),
                    volume=0 if pd.isna(r.volume) else int(r.volume),
                    adj_close=None if pd.isna(r.adj_close) else float(r.adj_close),
                ),
            )
    time.sleep(pause_sec)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load S&P 500 (current constituents) OHLCV into MySQL.")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--throttle", type=float, default=0.25)
    parser.add_argument("--include-index-spy", action="store_true", help="Also load ^GSPC and SPY")
    args = parser.parse_args()

    ensure_schema()

    # Get current S&P 500 list
    try:
        members = fetch_constituents()
    except Exception as e:
        print("Failed to fetch S&P 500 list from Wikipedia:", e, file=sys.stderr)
        sys.exit(1)

    upsert_symbols(members)

    # Compose ticker list
    tickers: List[str] = members["symbol"].tolist()
    if args.include_index_spy:
        tickers += ["^GSPC", "SPY"]

    # Download all
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}")
        try:
            download_and_store(t, start=args.start, end=args.end, pause_sec=args.throttle)
        except Exception as e:
            print(f"Error on {t}: {e}", file=sys.stderr)
            continue

if __name__ == "__main__":
    main()
