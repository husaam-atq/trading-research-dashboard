# config.py

UNIVERSE = [
    "MSFT", "AAPL", "NVDA", "AMD", "QCOM",
    "META", "GOOGL", "AMZN", "ORCL", "CSCO",
    "INTC", "ADBE", "CRM", "TXN", "MU"
]

SECTOR_PRESETS = {
    "Semiconductors": ["NVDA", "AMD", "QCOM", "INTC", "TXN", "MU", "AVGO", "ADI", "ON", "MCHP"],
    "Big Tech Platforms": ["MSFT", "AAPL", "META", "GOOGL", "AMZN", "NFLX", "ORCL", "CSCO", "ADBE", "CRM"],
    "Enterprise Software": ["MSFT", "ORCL", "ADBE", "CRM", "NOW", "INTU", "SAP", "PANW", "SNOW", "WDAY"],
    "Internet Commerce": ["AMZN", "EBAY", "SHOP", "MELI", "ETSY", "PDD", "BABA", "JD"],
    "Banks": ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "BK", "TFC"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "MDLZ", "GIS", "KHC"],
}

BENCHMARK = "SPY"

START_DATE = "2018-01-01"
END_DATE = "2026-01-01"

# Data / split
TRAIN_RATIO = 0.50
VALIDATION_RATIO = 0.20
MIN_OBSERVATIONS = 252

# Pair screening
MAX_COINTEGRATION_PVALUE = 0.05
MAX_ADF_PVALUE = 0.05
MIN_HALF_LIFE = 2
MAX_HALF_LIFE = 60
TOP_N_PAIRS = 5

# Parameter search on training set only
LOOKBACK_GRID = [15, 20, 30]
ENTRY_Z_GRID = [1.5, 2.0, 2.5]
EXIT_Z_GRID = [0.25, 0.5, 0.75]
STOP_Z_GRID = [3.0, 3.5, 4.0]

# Walk-forward
RECALIBRATION_WINDOW = 126
RECALIBRATION_STEP = 21

# Risk and portfolio
INITIAL_CAPITAL = 100000.0
TARGET_PORTFOLIO_VOL = 0.12
MAX_GROSS_LEVERAGE = 2.0
MAX_PAIR_WEIGHT = 0.35
MIN_PAIR_VOL = 1e-6
QUALITY_WEIGHT_STRENGTH = 0.75

# Trading costs
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 2.0

# Entry / regime filters
REQUIRE_ENTRY_CROSS = True
COOLDOWN_DAYS = 5
TREND_LOOKBACK = 15
MAX_TREND_ZSCORE_SLOPE = 0.18
VOL_FILTER_LOOKBACK = 20
MAX_RECENT_VOL_MULTIPLIER = 2.0

# Metrics
RISK_FREE_RATE = 0.0

# Output
SAVE_RESULTS = True
RESULTS_DIR = "results"
PLOT_FIGURES = True