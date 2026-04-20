# data_loader.py

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Returns a clean wide DataFrame with dates as index and tickers as columns.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if data.empty:
        raise ValueError("No data downloaded. Check tickers/date range/internet connection.")

    if "Adj Close" in data.columns.get_level_values(0):
        px = data["Adj Close"].copy()
    elif "Close" in data.columns.get_level_values(0):
        px = data["Close"].copy()
    else:
        raise ValueError("Downloaded data does not contain 'Adj Close' or 'Close'.")

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.sort_index().dropna(how="all")
    px = px.dropna(axis=1, how="all")

    return px


def get_pair_frame(prices: pd.DataFrame, ticker_y: str, ticker_x: str) -> pd.DataFrame:
    """
    Return aligned two-column frame for a pair.
    """
    df = prices[[ticker_y, ticker_x]].dropna().copy()
    df.columns = ["y", "x"]
    return df