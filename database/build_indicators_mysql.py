import os
import sys
import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine, text, inspect
from sqlalchemy import Table, Column, MetaData, String, Date, Float, PrimaryKeyConstraint
from typing import List

# ========== Config ==========
DB_URL = os.getenv(
    "DB_URL", "mysql+pymysql://root:578797@localhost:3306/marketdata?charset=utf8mb4"
)

# ========== Indicator List from Paper (with additions for TA baselines) ==========
# We now include sma50 which is needed for a TA baseline comparison
STATE_FEATURES = [
    "rsi14", "sma20", "sma50", "obv", "mom", "stoch_k", "macd", "macd_hist", "cci", "adx", "trix", "roc",
    "sar", "tema", "trima", "wma", "dema", "mfi", "cmo", "stochrsi_k",
    "ultosc", "bop", "atr"
]

# ========== Main Logic ==========
def get_all_symbols(engine) -> List[str]:
    """ Fetches all active symbols from the 'symbols' table. """
    print("[info] Fetching all active symbols from the 'symbols' table...")
    with engine.begin() as conn:
        q = text("SELECT symbol FROM symbols WHERE active=1 ORDER BY symbol")
        symbols_df = pd.read_sql(q, conn)
    if symbols_df.empty:
        print("[warn] No symbols found in the 'symbols' table.")
        return []
    symbols = symbols_df['symbol'].tolist()
    print(f"[info] Found {len(symbols)} active symbols to process.")
    return symbols

def create_indicators_table_if_not_exists(engine):
    """ Defines the schema for the indicators_daily table and creates it if needed. """
    metadata = MetaData()
    
    indicators_daily = Table('indicators_daily', metadata,
        Column('symbol', String(16), nullable=False),
        Column('trade_date', Date, nullable=False),
        Column('rsi14', Float), 
        Column('sma20', Float), 
        Column('sma50', Float), # Added SMA50
        Column('obv', Float), Column('mom', Float), Column('stoch_k', Float),
        Column('macd', Float), Column('macd_hist', Float), 
        Column('cci', Float), Column('adx', Float), Column('trix', Float),
        Column('roc', Float), Column('sar', Float), Column('tema', Float), Column('trima', Float),
        Column('wma', Float), Column('dema', Float), Column('mfi', Float), Column('cmo', Float),
        Column('stochrsi_k', Float), Column('ultosc', Float), Column('bop', Float), Column('atr', Float),
        PrimaryKeyConstraint('symbol', 'trade_date', name='pk_indicators_daily')
    )
    
    inspector = inspect(engine)
    if not inspector.has_table('indicators_daily'):
        print("[info] 'indicators_daily' table not found. Creating it...")
        metadata.create_all(engine)
        print("[info] Table created successfully.")
    else:
        print("[info] 'indicators_daily' table already exists.")

def calculate_and_store_indicators(engine, symbol: str):
    """
    Calculates all indicators for a given symbol and stores them in the database.
    """
    print(f"\n>>> Processing symbol: {symbol}")
    
    print(f"[load] Reading daily bars for {symbol} from database...")
    with engine.begin() as conn:
        q = text("SELECT trade_date, open, high, low, close, volume FROM daily_bars WHERE symbol=:s ORDER BY trade_date")
        df = pd.read_sql(q, conn, params={"s": symbol}, index_col='trade_date', parse_dates=['trade_date'])
    
    if df.empty or len(df) < 60: # Increased minimum length for sma50
        print(f"[warn] Not enough data found for symbol {symbol} (found {len(df)} rows). Skipping.")
        return

    print(f"[load] Found {len(df)} rows for {symbol}.")

    print("[calc] Calculating technical indicators...")
    
    custom_strategy = ta.Strategy(
        name="Paper_Indicators",
        description="The indicators used in the research paper plus additions for TA",
        ta=[
            {"kind": "rsi", "length": 14},
            {"kind": "sma", "length": 20},
            {"kind": "sma", "length": 50}, # Added SMA50
            {"kind": "obv"},
            {"kind": "mom"},
            {"kind": "stoch", "k": 14, "d": 3},
            {"kind": "macd"},
            {"kind": "cci"},
            {"kind": "adx"},
            {"kind": "trix"},
            {"kind": "roc"},
            {"kind": "psar"},
            {"kind": "tema"},
            {"kind": "trima"},
            {"kind": "wma"},
            {"kind": "dema"},
            {"kind": "mfi"},
            {"kind": "cmo"},
            {"kind": "stochrsi"},
            {"kind": "uo"},
            {"kind": "bop"},
            {"kind": "atr"},
        ]
    )
    df.ta.strategy(custom_strategy)

    # Rename the default pandas_ta columns to our desired schema names
    df.rename(columns={
        "RSI_14": "rsi14",
        "SMA_20": "sma20", 
        "SMA_50": "sma50", # Added SMA50
        "OBV": "obv", 
        "MOM_10": "mom", 
        "STOCHk_14_3_3": "stoch_k",
        "MACD_12_26_9": "macd",
        "MACDh_12_26_9": "macd_hist",
        "CCI_14_0.015": "cci", 
        "ADX_14": "adx", 
        "TRIX_30_9": "trix",
        "ROC_10": "roc", 
        "PSARl_0.02_0.2": "sar",
        "TEMA_10": "tema", 
        "TRIMA_10": "trima", 
        "WMA_10": "wma",
        "DEMA_10": "dema", 
        "MFI_14": "mfi", 
        "CMO_14": "cmo", 
        "STOCHRSIk_14_14_3_3": "stochrsi_k",
        "UO_7_14_28": "ultosc", 
        "BOP": "bop", 
        "ATRr_14": "atr"
    }, inplace=True)

    final_df = df[STATE_FEATURES].copy()
    final_df.dropna(inplace=True)
    final_df.reset_index(inplace=True)
    final_df.insert(0, 'symbol', symbol)

    if final_df.empty:
        print(f"[warn] No valid indicator data for {symbol}. Skipping.")
        return

    print(f"[save] Saving {len(final_df)} rows for {symbol} to database...")
    
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM indicators_daily WHERE symbol=:s"), {"s": symbol})

    final_df.to_sql('indicators_daily', engine, if_exists='append', index=False)
    print(f"[save] Successfully saved indicators for {symbol}.")

if __name__ == "__main__":
    try:
        engine = create_engine(DB_URL, future=True)
        create_indicators_table_if_not_exists(engine)
        symbols_to_process = get_all_symbols(engine)
        
        if not symbols_to_process:
            print("[info] No symbols to process. Exiting.")
            sys.exit(0)
        
        for sym in symbols_to_process:
            calculate_and_store_indicators(engine, sym)
            
        print("\n>>> All symbols processed successfully.")
        
    except ImportError:
        print("[error] The 'pandas_ta' library is not installed. Please run: pip install pandas_ta")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"[error] An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

