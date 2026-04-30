from __future__ import annotations

from itertools import combinations

import pandas as pd
import streamlit as st

from src.backtest import run_pair_backtest
from src.data import DEFAULT_END, DEFAULT_START, download_adjusted_close, research_universes
from src.pairs import analyse_pair, half_life_z_window
from src.walk_forward import walk_forward_pair

CORRELATION_THRESHOLD = 0.85
COINTEGRATION_PVALUE_THRESHOLD = 0.05
ADF_PVALUE_THRESHOLD = 0.05
MIN_HALF_LIFE = 3.0
MAX_HALF_LIFE = 45.0
MIN_THRESHOLD_CROSSINGS = 8
ENTRY_GRID = [1.5, 2.0, 2.5]
EXIT_GRID = [0.0, 0.5, 1.0]
STOP_GRID = [3.0, 3.5, 4.0]

st.set_page_config(
    page_title="Trading Research Dashboard",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return download_adjusted_close(list(tickers), start, end)


@st.cache_data(show_spinner=False)
def run_walk_forward_cached(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    transaction_cost_bps: float,
    max_holding_period: int,
    zscore_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    return walk_forward_pair(
        prices=prices,
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        entry_grid=ENTRY_GRID,
        exit_grid=EXIT_GRID,
        stop_grid=STOP_GRID,
        transaction_cost_bps=transaction_cost_bps,
        max_holding_period=max_holding_period,
        zscore_window=zscore_window,
        train_window=252,
        test_window=63,
        min_trades=3,
    )


def pct(value: float) -> str:
    return f"{value:.2%}"


def num(value: float) -> str:
    return f"{value:.2f}"


st.title("Trading Research Dashboard: Pairs Trading & Walk-Forward Backtesting")
st.caption(
    "Peer-group pairs research with cointegration, half-life diagnostics, static versus rolling hedge ratios, "
    "transaction costs, time-stops, volatility filters, and nested walk-forward threshold selection."
)

with st.sidebar:
    st.header("Research Controls")
    universes = research_universes()
    universe_mode = st.radio("Universe mode", list(universes.keys()), horizontal=True)
    peer_group = st.selectbox("Peer group preset", list(universes[universe_mode].keys()))
    tickers = tuple(universes[universe_mode][peer_group])
    pairs = [f"{a}/{b}" for a, b in combinations(tickers, 2)]
    pair_choice = st.selectbox("Ticker pair", pairs)
    ticker_y, ticker_x = pair_choice.split("/")
    start_date = st.date_input("Start date", pd.to_datetime(DEFAULT_START))
    end_date = st.date_input("End date", pd.to_datetime(DEFAULT_END))
    hedge_mode = st.radio("Hedge ratio", ["static", "rolling", "kalman"], horizontal=True)
    entry_threshold = st.slider("Entry z-score", 1.0, 3.5, 2.0, 0.1)
    exit_threshold = st.slider("Exit z-score", 0.0, 1.5, 0.5, 0.1)
    stop_threshold = st.slider("Stop z-score", 2.5, 5.0, 3.0, 0.1)
    max_holding_period = st.slider("Max holding period", 5, 60, 20, 1)
    transaction_cost_bps = st.slider("Transaction cost, bps per one-way trade", 0.0, 25.0, 5.0, 0.5)
    zscore_window_mode = st.radio("Z-score window mode", ["half-life based", "fixed"], horizontal=True)
    fixed_zscore_window = st.slider("Fixed z-score window", 20, 180, 60, 5)
    use_volatility_filter = st.checkbox("Volatility filter", value=True)

if exit_threshold >= entry_threshold:
    st.error("Exit threshold must be below the entry threshold.")
    st.stop()

if stop_threshold <= entry_threshold:
    st.error("Stop threshold must be above the entry threshold.")
    st.stop()

with st.spinner("Downloading adjusted close data and running the selected pair..."):
    prices = load_prices(tuple(sorted(set(tickers) | {"SPY"})), str(start_date), str(end_date))
    pair_stats = analyse_pair(prices[ticker_y], prices[ticker_x], 0, entry_threshold)
    zscore_window = half_life_z_window(pair_stats["half_life"]) if zscore_window_mode == "half-life based" else fixed_zscore_window
    result = run_pair_backtest(
        prices,
        ticker_y,
        ticker_x,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_threshold=stop_threshold,
        zscore_window=zscore_window,
        transaction_cost_bps=transaction_cost_bps,
        max_holding_period=max_holding_period,
        hedge_mode=hedge_mode,
        hedge_training_window=252,
        fit_window=min(756, len(prices)),
        use_volatility_filter=use_volatility_filter,
    )

daily = result.daily
metrics = result.metrics

top = st.columns(6)
top[0].metric("CAGR", pct(metrics["cagr"]))
top[1].metric("Sharpe", num(metrics["sharpe_ratio"]))
top[2].metric("Max drawdown", pct(metrics["max_drawdown"]))
top[3].metric("Trades", f"{int(metrics['number_of_trades'])}")
top[4].metric("Win rate", pct(metrics["win_rate"]))
top[5].metric("VaR / ES 95%", f"{pct(metrics['var_95'])} / {pct(metrics['expected_shortfall_95'])}")

st.subheader(f"{pair_choice} Diagnostics")
diagnostics = pd.DataFrame(
    {
        "Statistic": [
            "Peer group",
            "Universe mode",
            "Correlation",
            "Engle-Granger p-value",
            "ADF p-value",
            "Half-life",
            "Z-score window",
            "Threshold crossings",
            "Latest hedge ratio",
            "Filter pass",
        ],
        "Value": [
            peer_group,
            universe_mode,
            num(pair_stats["correlation"]),
            num(pair_stats["coint_pvalue"]),
            num(pair_stats["adf_pvalue"]),
            num(pair_stats["half_life"]) if pd.notna(pair_stats["half_life"]) else "n/a",
            str(zscore_window),
            f"{int(pair_stats['threshold_crossings'])}",
            num(metrics["hedge_ratio"]),
            str(
                abs(pair_stats["correlation"]) >= CORRELATION_THRESHOLD
                and pair_stats["coint_pvalue"] <= COINTEGRATION_PVALUE_THRESHOLD
                and pair_stats["adf_pvalue"] <= ADF_PVALUE_THRESHOLD
                and pd.notna(pair_stats["half_life"])
                and MIN_HALF_LIFE <= pair_stats["half_life"] <= MAX_HALF_LIFE
                and pair_stats["threshold_crossings"] >= MIN_THRESHOLD_CROSSINGS
            ),
        ],
    }
)
st.dataframe(diagnostics, use_container_width=True, hide_index=True)

price_chart = prices[[ticker_y, ticker_x]].dropna()
price_chart = price_chart / price_chart.iloc[0]
st.subheader("Price Series")
st.line_chart(price_chart)

chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Hedge Ratio")
    st.line_chart(daily["hedge_ratio"])
with chart_right:
    st.subheader("Spread")
    st.line_chart(daily["spread"])

chart_left, chart_right = st.columns(2)
with chart_left:
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
with chart_right:
    st.subheader("Entry Filter")
    st.line_chart(daily["can_enter"].astype(float))

curve_left, curve_right = st.columns(2)
with curve_left:
    st.subheader("Equity Curve")
    st.line_chart(daily["equity"])
with curve_right:
    st.subheader("Drawdown")
    st.area_chart(daily["drawdown"])

st.subheader("Trade Analytics")
trade_summary = pd.DataFrame(
    [
        {
            "average_trade_return": metrics["average_trade_return"],
            "median_trade_return": metrics["median_trade_return"],
            "average_holding_period": metrics["average_holding_period"],
            "profit_factor": metrics["profit_factor"],
            "stop_loss_exit_pct": metrics["stop_loss_exit_pct"],
            "time_stop_exit_pct": metrics["time_stop_exit_pct"],
            "mean_reversion_exit_pct": metrics["mean_reversion_exit_pct"],
        }
    ]
)
st.dataframe(trade_summary, use_container_width=True)

st.subheader("Trade Log Preview")
if result.trades.empty:
    st.info("No completed trades for the selected settings.")
else:
    st.dataframe(result.trades.tail(20), use_container_width=True)

with st.spinner("Running nested walk-forward validation for the selected pair..."):
    wf_daily, wf_thresholds, wf_metrics = run_walk_forward_cached(
        prices[[ticker_y, ticker_x]].dropna(),
        ticker_y,
        ticker_x,
        transaction_cost_bps,
        max_holding_period,
        zscore_window,
    )

st.subheader("Walk-Forward Metrics")
wf_table = pd.DataFrame(
    [
        {
            "total_return": wf_metrics["total_return"],
            "cagr": wf_metrics["cagr"],
            "sharpe_ratio": wf_metrics["sharpe_ratio"],
            "max_drawdown": wf_metrics["max_drawdown"],
            "number_of_trades": wf_metrics["number_of_trades"],
            "segments": wf_metrics["segments"],
        }
    ]
)
st.dataframe(wf_table, use_container_width=True)
st.line_chart(wf_daily["strategy_return"].fillna(0.0).add(1.0).cumprod())

st.subheader("Selected Walk-Forward Thresholds")
st.dataframe(wf_thresholds.tail(15), use_container_width=True)

st.subheader("Assumptions and Limitations")
st.markdown(
    """
- Static hedge ratios are estimated from the initial training window; rolling hedge ratios use only prior observations.
- Walk-forward thresholds are selected inside each training window and applied only to the following test window.
- Signals are generated from end-of-day z-scores and applied with a one-day lag before returns are realised.
- Adjusted close data comes from yfinance and can contain revisions, missing values, or vendor inconsistencies.
- The universe is static, so survivorship bias is not removed.
- Borrow fees, short availability, taxes, financing, and intraday execution are not modelled.
- Transaction costs are simplified as a fixed bps charge per one-way leg.
- Cointegration and spread relationships can break down outside the estimation sample.
"""
)
