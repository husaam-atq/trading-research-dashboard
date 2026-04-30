from __future__ import annotations

import pandas as pd
import yfinance as yf

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "NVDA",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "KO",
    "PEP",
    "WMT",
    "COST",
    "HD",
    "UNH",
    "MRK",
    "V",
    "MA",
    "ORCL",
    "CSCO",
    "INTC",
]

STOCK_PEER_GROUPS = {
    "mega_cap_tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU"],
    "banks": ["JPM", "BAC", "C", "GS", "MS", "WFC"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB"],
    "retail_consumer": ["COST", "WMT", "HD", "LOW", "TGT"],
    "healthcare": ["UNH", "MRK", "PFE", "ABBV", "JNJ"],
}

ETF_PEER_GROUPS = {
    "broad_equity": ["SPY", "IVV", "VOO", "VTI", "SCHB"],
    "nasdaq_growth": ["QQQ", "XLK", "VGT", "IYW"],
    "international_equity": ["EFA", "VEA", "IEFA", "EEM", "VWO"],
    "treasuries": ["SHY", "IEI", "IEF", "TLH", "TLT", "GOVT"],
    "gold": ["GLD", "IAU", "SGOL"],
    "real_estate": ["VNQ", "SCHH", "IYR"],
    "energy": ["XLE", "VDE", "XOP", "OIH"],
    "financials": ["XLF", "VFH", "KBE", "KRE"],
}

PEER_GROUP_UNIVERSES = STOCK_PEER_GROUPS


def research_universes() -> dict[str, dict[str, list[str]]]:
    return {"stock_peer_groups": STOCK_PEER_GROUPS, "etf_peer_groups": ETF_PEER_GROUPS}


def flatten_universes(universes: dict[str, dict[str, list[str]]]) -> list[str]:
    tickers: list[str] = []
    for groups in universes.values():
        for group in groups.values():
            tickers.extend(group)
    tickers.append("SPY")
    return sorted(dict.fromkeys(tickers))


DEFAULT_START = "2015-01-01"
DEFAULT_END = "2024-12-31"


def download_adjusted_close(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Download adjusted close prices and return a clean wide DataFrame."""
    unique_tickers = list(dict.fromkeys([ticker.strip().upper() for ticker in tickers if ticker.strip()]))
    if len(unique_tickers) < 2:
        raise ValueError("At least two tickers are required.")
    end_exclusive = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=unique_tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No price data was downloaded. Check tickers, dates, or network access.")

    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[field].copy()
    else:
        field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[field]].copy()
        prices.columns = unique_tickers[:1]

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(axis=1, how="all").dropna(how="all")
    prices = prices.ffill().dropna(axis=1, thresh=max(30, int(len(prices) * 0.8)))
    prices = prices.dropna()

    if prices.shape[1] < 2:
        raise ValueError("Fewer than two tickers have usable adjusted close histories.")

    return prices


def get_pair_prices(prices: pd.DataFrame, ticker_y: str, ticker_x: str) -> pd.DataFrame:
    """Return aligned adjusted close prices for one pair."""
    pair = prices[[ticker_y, ticker_x]].dropna().copy()
    pair.columns = ["y", "x"]
    return pair
