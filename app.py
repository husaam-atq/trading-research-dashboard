from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import run_pair_backtest
from src.data import download_adjusted_close
from src.pairs import analyse_pair, half_life_z_window
from src.universes import expanded_research_universes


ROOT = Path(__file__).resolve().parent
BASELINE_DATA = ROOT / "reports" / "baseline" / "data"
DEVELOPMENT_DATA = ROOT / "reports" / "development" / "final"
CONFIRMATION_DATA = ROOT / "reports" / "confirmation"
FROZEN_CONFIG = ROOT / "config" / "final_methodology_v1.json"


st.set_page_config(page_title="Trading Research Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def read_table(directory: str, name: str, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = Path(directory) / f"{name}.csv"
    return pd.read_csv(path, parse_dates=list(dates))


@st.cache_data(show_spinner=False)
def load_live_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return download_adjusted_close(list(tickers), start, end)


def pct(value: float) -> str:
    return f"{value:.2%}"


def number(value: float) -> str:
    return f"{value:.2f}"


def metric_row(frame: pd.DataFrame, labels: list[tuple[str, str, str]]) -> None:
    row = frame.iloc[0]
    columns = st.columns(len(labels))
    for column, (label, field, style) in zip(columns, labels):
        value = float(row[field])
        rendered = pct(value) if style == "pct" else f"{int(value)}" if style == "int" else number(value)
        column.metric(label, rendered)


def equity_and_drawdown(daily: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    returns = daily.set_index("date")["strategy_return"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return equity, drawdown


required = [
    DEVELOPMENT_DATA / "performance_metrics.csv",
    CONFIRMATION_DATA / "performance_metrics.csv",
    FROZEN_CONFIG,
]
missing = [path for path in required if not path.exists()]
if missing:
    st.error("Generated research artifacts are missing. Run `python main.py --phase development` and the documented freeze/confirmation workflow.")
    st.stop()

development = read_table(str(DEVELOPMENT_DATA), "performance_metrics")
confirmation = read_table(str(CONFIRMATION_DATA), "performance_metrics")
baseline = read_table(str(BASELINE_DATA), "sharpe_optimised_results")
frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
frozen_hash = (FROZEN_CONFIG.with_suffix(".json.sha256").read_text(encoding="ascii").split()[0])

with st.sidebar:
    st.subheader("Research Status")
    st.success("Methodology frozen")
    st.caption(f"Config: {frozen_hash[:12]}...")
    st.caption("Development: 2015-2024")
    st.caption("Confirmation: 2025-01-01 to 2026-07-31")
    st.divider()
    st.caption("5 bps per leg, one way")
    st.caption("504 / 126 / 63 train-validation-test")
    st.caption("63-day contiguous test step")

st.title("Trading Research Dashboard")
st.caption("Pairs Trading & Leakage-Controlled Walk-Forward Research")
st.info(
    "The broader final method is negative in 2015-2024 development data but positive in the single frozen 2025-2026 confirmation. "
    "The confirmation sample is small and is not presented as proof of deployable alpha."
)

tabs = st.tabs(
    ["Overview", "Pair Discovery", "Walk-Forward", "Portfolio", "Robustness", "Confirmation", "Pair Lab"]
)

with tabs[0]:
    st.subheader("Research Evidence")
    baseline_row = baseline[baseline["method"] == "portfolio_vol_target_8"].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "period": "Archived baseline",
                "total_return": baseline_row["total_return"],
                "sharpe": baseline_row["sharpe_ratio"],
                "max_drawdown": baseline_row["max_drawdown"],
                "trades": baseline_row["number_of_trades"],
                "unique_pairs": 5,
                "peer_groups": 3,
            },
            {
                "period": "Final development",
                "total_return": development.iloc[0]["total_return"],
                "sharpe": development.iloc[0]["sharpe_ratio"],
                "max_drawdown": development.iloc[0]["max_drawdown"],
                "trades": development.iloc[0]["completed_trades"],
                "unique_pairs": development.iloc[0]["unique_pairs"],
                "peer_groups": development.iloc[0]["peer_groups"],
            },
            {
                "period": "Frozen confirmation",
                "total_return": confirmation.iloc[0]["total_return"],
                "sharpe": confirmation.iloc[0]["sharpe_ratio"],
                "max_drawdown": confirmation.iloc[0]["max_drawdown"],
                "trades": confirmation.iloc[0]["completed_trades"],
                "unique_pairs": confirmation.iloc[0]["unique_pairs"],
                "peer_groups": confirmation.iloc[0]["peer_groups"],
            },
        ]
    )
    display = comparison.copy()
    for field in ["total_return", "max_drawdown"]:
        display[field] = display[field].map(pct)
    display["sharpe"] = display["sharpe"].map(number)
    for field in ["trades", "unique_pairs", "peer_groups"]:
        display[field] = display[field].astype(int)
    st.dataframe(display, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Final Development")
        metric_row(
            development,
            [("Return", "total_return", "pct"), ("Sharpe", "sharpe_ratio", "num"), ("Max DD", "max_drawdown", "pct")],
        )
        dev_daily = read_table(str(DEVELOPMENT_DATA), "portfolio_daily_returns", ("date",))
        dev_equity, _ = equity_and_drawdown(dev_daily)
        st.line_chart(dev_equity, height=260)
    with right:
        st.markdown("#### Frozen Confirmation")
        metric_row(
            confirmation,
            [("Return", "total_return", "pct"), ("Sharpe", "sharpe_ratio", "num"), ("Max DD", "max_drawdown", "pct")],
        )
        conf_daily = read_table(str(CONFIRMATION_DATA), "portfolio_daily_returns", ("date",))
        conf_equity, _ = equity_and_drawdown(conf_daily)
        st.line_chart(conf_equity, height=260)

    st.markdown("#### Frozen Method")
    method = frozen["methodology"]
    st.dataframe(
        pd.DataFrame(
            [
                {"component": "Discovery", "choice": "Training-only peer screening with repeated 252-day stability checks"},
                {"component": "Hedge", "choice": "Fixed Kalman recursive hedge"},
                {"component": "Signal", "choice": "2.0 entry / 0.0 exit / 3.5 stop / 20-day time stop"},
                {"component": "Selection", "choice": "Training-ranked, ticker/group/correlation constrained, maximum four pairs"},
                {"component": "Weights", "choice": method["portfolio"]["weighting_method"].replace("_", " ")},
                {"component": "Vol target", "choice": "None"},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Training-Only Candidate Discovery")
    diagnostics = read_table(str(DEVELOPMENT_DATA), "candidate_diagnostics", ("train_start", "train_end"))
    segment = st.selectbox("Development segment", sorted(diagnostics["segment"].unique()), key="discovery_segment")
    segment_diagnostics = diagnostics[diagnostics["segment"] == segment].copy()
    only_passed = st.toggle("Show training-pass candidates only", value=True)
    if only_passed:
        segment_diagnostics = segment_diagnostics[segment_diagnostics["selected_candidate"]]
    columns = [
        "pair", "universe_mode", "peer_group", "correlation", "coint_pvalue", "adf_pvalue", "half_life",
        "coint_pass_rate", "adf_pass_rate", "hedge_ratio_cv", "training_trade_count", "training_score", "selected_candidate",
    ]
    st.dataframe(segment_diagnostics[columns].head(50), width="stretch", hide_index=True)
    st.caption(
        "Every row is computed from that segment's training window. Validation and test dates are not available to discovery."
    )

with tabs[2]:
    st.subheader("Contiguous Walk-Forward Evaluation")
    boundaries = read_table(
        str(DEVELOPMENT_DATA),
        "walk_forward_boundaries",
        ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end"),
    )
    segment_metrics = read_table(str(DEVELOPMENT_DATA), "segment_metrics")
    segment_chart = segment_metrics.set_index("segment")[["total_return", "selected_pairs", "completed_trades"]]
    st.line_chart(segment_chart[["total_return"]], height=260)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Window Boundaries")
        st.dataframe(boundaries, width="stretch", hide_index=True, height=430)
    with col2:
        st.markdown("#### OOS Segment Metrics")
        st.dataframe(
            segment_metrics[["segment", "selected_pairs", "completed_trades", "total_return", "sharpe_ratio", "max_drawdown"]],
            width="stretch",
            hide_index=True,
            height=430,
        )
    st.caption("Each test starts flat. The next test begins immediately after the previous 63-day test ends.")

with tabs[3]:
    st.subheader("Breadth, Exposure And Contribution")
    selected = read_table(str(DEVELOPMENT_DATA), "selected_pairs_by_segment")
    contributions = read_table(str(DEVELOPMENT_DATA), "pair_contributions")
    concentration = read_table(str(DEVELOPMENT_DATA), "concentration")
    portfolio_segment = st.selectbox("Portfolio segment", sorted(selected["segment"].unique()), key="portfolio_segment")
    segment_selection = selected[selected["segment"] == portfolio_segment]
    st.dataframe(
        segment_selection[["pair", "universe_mode", "peer_group", "hedge_mode", "weight", "training_score", "validation_volatility"]],
        width="stretch",
        hide_index=True,
    )
    group_exposure = segment_selection.groupby("peer_group")["weight"].sum().sort_values(ascending=False)
    ticker_rows: list[dict[str, float | str]] = []
    for row in segment_selection.itertuples(index=False):
        ticker_rows.extend([{"ticker": row.ticker_y, "weight": row.weight / 2}, {"ticker": row.ticker_x, "weight": row.weight / 2}])
    ticker_exposure = pd.DataFrame(ticker_rows).groupby("ticker")["weight"].sum().sort_values(ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Peer-Group Exposure")
        st.bar_chart(group_exposure, height=250)
    with right:
        st.markdown("#### Ticker Exposure")
        st.bar_chart(ticker_exposure, height=250)
    st.markdown("#### Pair P&L Contribution")
    st.bar_chart(contributions.set_index("pair")["weighted_pnl"].head(20), height=330)
    st.dataframe(concentration, width="stretch", hide_index=True, height=260)

with tabs[4]:
    st.subheader("Robustness And Uncertainty")
    costs = read_table(str(DEVELOPMENT_DATA), "cost_sensitivity")
    borrow = read_table(str(DEVELOPMENT_DATA), "borrow_cost_sensitivity")
    methods = read_table(str(DEVELOPMENT_DATA), "portfolio_method_comparison")
    targets = read_table(str(DEVELOPMENT_DATA), "volatility_target_comparison")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Transaction Costs")
        st.line_chart(costs.set_index("transaction_cost_bps_per_leg")[["total_return", "sharpe_ratio"]], height=280)
        st.dataframe(costs[["transaction_cost_bps_per_leg", "total_return", "sharpe_ratio", "max_drawdown"]], hide_index=True, width="stretch")
    with right:
        st.markdown("#### Assumed Borrow Costs")
        st.line_chart(borrow.set_index("assumed_annual_borrow_rate")[["total_return", "sharpe_ratio"]], height=280)
        st.dataframe(borrow[["assumed_annual_borrow_rate", "estimated_borrow_cost", "total_return", "sharpe_ratio"]], hide_index=True, width="stretch")
    st.markdown("#### Development Sensitivities")
    col1, col2 = st.columns(2)
    col1.dataframe(methods[["method", "total_return", "sharpe_ratio", "max_drawdown"]], hide_index=True, width="stretch")
    col2.dataframe(targets[["target", "total_return", "sharpe_ratio", "max_drawdown"]], hide_index=True, width="stretch")
    metric_row(
        development,
        [
            ("Bootstrap Sharpe low", "bootstrap_sharpe_lower", "num"),
            ("Bootstrap Sharpe high", "bootstrap_sharpe_upper", "num"),
            ("Newey-West p-value", "newey_west_pvalue", "num"),
            ("SPY beta", "beta_to_spy", "num"),
        ],
    )

with tabs[5]:
    st.subheader("Fresh 2025-2026 Confirmation")
    st.warning("This section is evaluated once under the frozen configuration. It is not used to retune the model.")
    metric_row(
        confirmation,
        [
            ("Return", "total_return", "pct"),
            ("Sharpe", "sharpe_ratio", "num"),
            ("Max DD", "max_drawdown", "pct"),
            ("Trades", "completed_trades", "int"),
            ("Unique pairs", "unique_pairs", "int"),
            ("Groups", "peer_groups", "int"),
        ],
    )
    conf_daily = read_table(str(CONFIRMATION_DATA), "portfolio_daily_returns", ("date",))
    conf_equity, conf_drawdown = equity_and_drawdown(conf_daily)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Equity")
        st.line_chart(conf_equity, height=300)
    with right:
        st.markdown("#### Drawdown")
        st.area_chart(conf_drawdown, height=300)
    conf_segments = read_table(str(CONFIRMATION_DATA), "segment_metrics")
    conf_pairs = read_table(str(CONFIRMATION_DATA), "selected_pairs_by_segment")
    conf_contributions = read_table(str(CONFIRMATION_DATA), "pair_contributions")
    st.dataframe(conf_segments[["segment", "selected_pairs", "completed_trades", "total_return", "sharpe_ratio", "max_drawdown"]], hide_index=True, width="stretch")
    col1, col2 = st.columns(2)
    col1.dataframe(conf_pairs[["segment", "pair", "peer_group", "weight"]], hide_index=True, width="stretch", height=330)
    col2.dataframe(conf_contributions[["pair", "weighted_pnl", "absolute_pnl_share", "turnover"]].head(20), hide_index=True, width="stretch", height=330)

with tabs[6]:
    st.subheader("Interactive Pair Lab")
    st.caption("This exploratory view is separate from the frozen portfolio evidence above.")
    universes = expanded_research_universes()
    controls = st.columns([1, 1, 1, 1])
    universe_mode = controls[0].selectbox("Universe", list(universes), key="lab_universe")
    peer_group = controls[1].selectbox("Peer group", list(universes[universe_mode]), key="lab_group")
    members = universes[universe_mode][peer_group]
    pair_options = [f"{a}/{b}" for a, b in combinations(members, 2)]
    pair_choice = controls[2].selectbox("Pair", pair_options, key="lab_pair")
    hedge_mode = controls[3].selectbox("Hedge", ["static", "rolling", "kalman"], index=2, key="lab_hedge")
    ticker_y, ticker_x = pair_choice.split("/")

    settings = st.columns(6)
    entry = settings[0].number_input("Entry", 1.0, 3.0, 2.0, 0.25)
    exit_ = settings[1].number_input("Exit", 0.0, 1.0, 0.0, 0.25)
    stop = settings[2].number_input("Stop", 2.5, 5.0, 3.5, 0.25)
    max_hold = settings[3].number_input("Max hold", 5, 60, 20, 5)
    cost = settings[4].number_input("Cost bps/leg", 0.0, 20.0, 5.0, 1.0)
    fixed_window = settings[5].number_input("Z window", 20, 90, 60, 5)
    dates = st.columns(2)
    start = dates[0].date_input("Start", pd.Timestamp("2018-01-01"), key="lab_start")
    end = dates[1].date_input("End", pd.Timestamp("2024-12-31"), key="lab_end")

    if st.button("Run pair lab", type="primary", width="stretch"):
        if exit_ >= entry or stop <= entry:
            st.error("Exit must be below entry and stop must be above entry.")
        else:
            with st.spinner("Loading adjusted prices and running the pair..."):
                prices = load_live_prices((ticker_y, ticker_x), str(start), str(end))
                pair_stats = analyse_pair(prices[ticker_y], prices[ticker_x], 0, entry)
                z_window = half_life_z_window(pair_stats["half_life"]) if np.isfinite(pair_stats["half_life"]) else int(fixed_window)
                result = run_pair_backtest(
                    prices,
                    ticker_y,
                    ticker_x,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                    stop_threshold=stop,
                    zscore_window=z_window,
                    transaction_cost_bps=cost,
                    max_holding_period=int(max_hold),
                    hedge_mode=hedge_mode,
                    fit_window=min(504, len(prices)),
                    use_volatility_filter=False,
                    trade_start=prices.index[min(504, len(prices) - 1)],
                )
            metric_row(
                pd.DataFrame([result.metrics]),
                [("Return", "total_return", "pct"), ("Sharpe", "sharpe_ratio", "num"), ("Max DD", "max_drawdown", "pct"), ("Trades", "number_of_trades", "int")],
            )
            diagnostics = pd.DataFrame(
                [
                    {"metric": "Correlation", "value": pair_stats["correlation"]},
                    {"metric": "Cointegration p-value", "value": pair_stats["coint_pvalue"]},
                    {"metric": "ADF p-value", "value": pair_stats["adf_pvalue"]},
                    {"metric": "Half-life", "value": pair_stats["half_life"]},
                    {"metric": "Z-score window", "value": z_window},
                ]
            )
            st.dataframe(diagnostics, hide_index=True, width="stretch")
            chart_left, chart_right = st.columns(2)
            chart_left.line_chart(result.daily[["spread"]], height=260)
            chart_right.line_chart(result.daily[["zscore"]], height=260)
            curve = (1.0 + result.daily["strategy_return"]).cumprod()
            st.line_chart(curve, height=300)
            st.dataframe(result.trades.tail(30), hide_index=True, width="stretch")

st.divider()
st.caption(
    "Research limitations: static present-day universes, Yahoo Finance data, simplified fixed costs, assumed borrow sensitivity, "
    "no short-availability history, no intraday execution, and limited confirmation sample size."
)
