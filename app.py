from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest import run_pair_backtest
from src.data import DEFAULT_END, DEFAULT_START, DEFAULT_UNIVERSE, download_adjusted_close
from src.metrics import performance_metrics
from src.pairs import analyse_pair

st.set_page_config(
    page_title="Trading Research Dashboard",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return download_adjusted_close(list(tickers), start, end)


def pct(value: float) -> str:
    return f"{value:.2%}"


def num(value: float) -> str:
    return f"{value:.2f}"


st.title("Trading Research Dashboard: Pairs Trading & Walk-Forward Backtesting")
st.caption(
    "Interactive statistical-arbitrage research using adjusted close data, spread z-scores, "
    "lagged signals, transaction costs, and transparent performance analytics."
)

with st.sidebar:
    st.header("Research Controls")
    universe_text = st.text_area("Universe", ", ".join(DEFAULT_UNIVERSE), height=110)
    tickers = tuple(dict.fromkeys([t.strip().upper() for t in universe_text.split(",") if t.strip()]))
    start_date = st.date_input("Start date", pd.to_datetime(DEFAULT_START))
    end_date = st.date_input("End date", pd.to_datetime(DEFAULT_END))
    ticker_y = st.selectbox("First leg", tickers, index=0)
    ticker_x_options = [ticker for ticker in tickers if ticker != ticker_y]
    ticker_x = st.selectbox("Second leg", ticker_x_options, index=min(1, len(ticker_x_options) - 1))
    entry_threshold = st.slider("Entry z-score", 1.0, 3.5, 2.0, 0.1)
    exit_threshold = st.slider("Exit z-score", 0.0, 1.5, 0.5, 0.1)
    stop_threshold = st.slider("Stop z-score", 2.5, 5.0, 3.0, 0.1)
    transaction_cost_bps = st.slider("Transaction cost, bps per one-way trade", 0.0, 25.0, 5.0, 0.5)
    zscore_window = st.slider("Rolling z-score window", 20, 180, 60, 5)

if len(tickers) < 2:
    st.error("Enter at least two tickers.")
    st.stop()

if ticker_y == ticker_x:
    st.error("Choose two different tickers.")
    st.stop()

if exit_threshold >= entry_threshold:
    st.error("Exit threshold must be below the entry threshold.")
    st.stop()

if stop_threshold <= entry_threshold:
    st.error("Stop threshold must be above the entry threshold.")
    st.stop()

with st.spinner("Downloading adjusted close data and running the pair backtest..."):
    prices = load_prices(tickers, str(start_date), str(end_date))
    result = run_pair_backtest(
        prices,
        ticker_y,
        ticker_x,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
        zscore_window=zscore_window,
        transaction_cost_bps=transaction_cost_bps,
    )
    pair_stats = analyse_pair(prices[ticker_y], prices[ticker_x])

daily = result.daily
metrics = result.metrics

top = st.columns(6)
top[0].metric("CAGR", pct(metrics["cagr"]))
top[1].metric("Sharpe", num(metrics["sharpe_ratio"]))
top[2].metric("Max drawdown", pct(metrics["max_drawdown"]))
top[3].metric("Trades", f"{int(metrics['number_of_trades'])}")
top[4].metric("Win rate", pct(metrics["win_rate"]))
top[5].metric("VaR / ES 95%", f"{pct(metrics['var_95'])} / {pct(metrics['expected_shortfall_95'])}")

st.subheader(f"{ticker_y}/{ticker_x} Research View")
left, right = st.columns([1, 1])
with left:
    st.write(
        pd.DataFrame(
            {
                "Statistic": [
                    "Correlation",
                    "Engle-Granger p-value",
                    "Hedge ratio",
                    "Spread mean",
                    "Spread volatility",
                    "Cost impact",
                ],
                "Value": [
                    num(pair_stats["correlation"]),
                    num(pair_stats["coint_pvalue"]),
                    num(metrics["hedge_ratio"]),
                    num(pair_stats["spread_mean"]),
                    num(pair_stats["spread_std"]),
                    pct(metrics["transaction_cost_impact"]),
                ],
            }
        )
    )
with right:
    st.write(
        pd.DataFrame(
            {
                "Metric": [
                    "Total return",
                    "Annualised volatility",
                    "Sortino",
                    "Calmar",
                    "Monthly win rate",
                    "Best month",
                    "Worst month",
                ],
                "Value": [
                    pct(metrics["total_return"]),
                    pct(metrics["annualised_volatility"]),
                    num(metrics["sortino_ratio"]),
                    num(metrics["calmar_ratio"]),
                    pct(metrics["monthly_win_rate"]),
                    pct(metrics["best_month"]),
                    pct(metrics["worst_month"]),
                ],
            }
        )
    )

price_chart = prices[[ticker_y, ticker_x]].dropna()
price_chart = price_chart / price_chart.iloc[0]
st.subheader("Price Series")
st.line_chart(price_chart)

chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Spread")
    st.line_chart(daily["spread"])
with chart_right:
    st.subheader("Z-Score")
    z_frame = pd.DataFrame(
        {
            "zscore": daily["zscore"],
            "entry": entry_threshold,
            "-entry": -entry_threshold,
            "exit": exit_threshold,
            "-exit": -exit_threshold,
        },
        index=daily.index,
    )
    st.line_chart(z_frame)

curve_left, curve_right = st.columns(2)
with curve_left:
    st.subheader("Equity Curve")
    st.line_chart(daily["equity"])
with curve_right:
    st.subheader("Drawdown")
    st.area_chart(daily["drawdown"])

st.subheader("Metrics Table")
metrics_table = pd.DataFrame([metrics]).T.rename(columns={0: "value"})
st.dataframe(metrics_table, use_container_width=True)

st.subheader("Assumptions and Limitations")
st.markdown(
    """
- Signals are generated from end-of-day z-scores and applied with a one-day lag before returns are realised.
- Adjusted close data comes from yfinance and can contain revisions, missing values, or vendor inconsistencies.
- The universe is static, so survivorship bias is not removed.
- Borrow fees, short availability, taxes, financing, and intraday execution are not modelled.
- Transaction costs are simplified as a fixed bps charge per one-way leg.
- Cointegration and spread relationships can break down, especially outside the sample used to estimate them.
- Dashboard results are exploratory and should not be read as live-trading evidence.
"""
)
