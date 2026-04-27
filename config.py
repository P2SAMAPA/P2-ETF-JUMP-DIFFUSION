"""
Configuration for P2-ETF-JUMP-DIFFUSION engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-jump-diffusion-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Columns ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Jump-Diffusion Parameters ---
DAILY_LOOKBACK = 504                 # Days for daily training
GLOBAL_TRAIN_START = "2008-01-01"    # Start for global training
JUMP_THRESHOLD_STD = 2.5
MIN_OBSERVATIONS = 252
GLOBAL_MIN_OBSERVATIONS = 1008       # 4 years for global

# Jump intensity caps
LAMBDA_CAP_DAILY = 10.0              # daily mode cap
LAMBDA_CAP_GLOBAL = 25.0             # global mode cap (higher, for crisis years)

# Macro conditioning (VIX scaling)
USE_MACRO_CONDITIONING = True        # scale λ by VIX/VIX_avg
VIX_AVG_LOOKBACK = 252               # days for VIX average baseline

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
