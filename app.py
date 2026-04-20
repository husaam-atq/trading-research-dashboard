# app.py

from __future__ import annotations

import ast
import pandas as pd
import streamlit as st

from config import (
    UNIVERSE,
    SECTOR_PRESETS,
    BENCHMARK,
    START_DATE,
    END_DATE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    MIN_OBSERVATIONS,
    MAX_COINTEGRATION_PVALUE,
    MAX_ADF_PVALUE,
    MIN_HALF_LIFE,
    MAX_HALF_LIFE,
    TOP_N_PAIRS,
    LOOKBACK_GRID,
    ENTRY_Z_GRID,
    EXIT_Z_GRID,
    STOP_Z_GRID,
    RECALIBRATION_WINDOW,
    RECALIBRATION_STEP,
    INITIAL_CAPITAL,
    COMMISSION_BPS,
    SLIPPAGE_BPS,
    TARGET_PORTFOLIO_VOL,
    MAX_GROSS_LEVERAGE,
    MAX_PAIR_WEIGHT,
    MIN_PAIR_VOL,
    QUALITY_WEIGHT_STRENGTH,
    RISK_FREE_RATE,
    REQUIRE_ENTRY_CROSS,
    COOLDOWN_DAYS,
    TREND_LOOKBACK,
    MAX_TREND_ZSCORE_SLOPE,
    VOL_FILTER_LOOKBACK,
    MAX_RECENT_VOL_MULTIPLIER,
)
from pipeline import run_research_pipeline


st.set_page_config(
    page_title="Quant Pairs Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: #0d1b2a;
            --panel-2: #102338;
            --text: #e8eef7;
            --muted: #9eb0c7;
            --line: rgba(255,255,255,0.08);
            --accent: #66b3ff;
            --accent-2: #8ef0c5;
            --danger: #ff7b7b;
            --warning: #ffd166;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(102,179,255,0.08), transparent 25%),
                radial-gradient(circle at top right, rgba(142,240,197,0.05), transparent 18%),
                linear-gradient(180deg, #06101c 0%, #081321 100%);
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1625 0%, #0c1727 100%);
            border-right: 1px solid var(--line);
        }

        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin-bottom: 1.2rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(102,179,255,0.14), rgba(142,240,197,0.08));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1.35rem 1.4rem 1.15rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }

        .section-card {
            background: rgba(12, 25, 40, 0.80);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.14);
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(14,32,50,0.95), rgba(10,24,38,0.95));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 124px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.78rem;
        }

        .section-title {
            font-size: 1.22rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.85rem;
        }

        .small-tag {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 999px;
            background: rgba(255,255,255,0.03);
            color: var(--muted);
            font-size: 0.77rem;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
        }

        .guide-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .guide-item {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
        }

        .guide-item b {
            color: #ffffff;
        }

        .guide-item span {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .status-good {
            color: var(--accent-2);
            font-weight: 700;
        }

        .status-bad {
            color: var(--danger);
            font-weight: 700;
        }

        .status-warn {
            color: var(--warning);
            font-weight: 700;
        }

        .block-note {
            border-left: 4px solid rgba(102,179,255,0.8);
            background: rgba(102,179,255,0.06);
            padding: 0.85rem 0.95rem;
            border-radius: 10px;
            color: var(--text);
            margin-top: 0.4rem;
            margin-bottom: 0.7rem;
        }

        .interpret-box {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-top: 0.4rem;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.3rem;
        }

        button[data-baseweb="tab"] {
            background: rgba(255,255,255,0.03) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.05) !important;
            padding: 0.45rem 0.8rem !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(102,179,255,0.12) !important;
            border: 1px solid rgba(102,179,255,0.35) !important;
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.06);
        }

        .sidebar-header {
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .sidebar-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 1rem;
        }

        hr {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.08);
            margin: 1rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_int_list(text: str) -> list[int]:
    values = ast.literal_eval(text)
    if not isinstance(values, list):
        raise ValueError("Expected a Python list such as [15, 20, 30]")
    return [int(v) for v in values]


def parse_float_list(text: str) -> list[float]:
    values = ast.literal_eval(text)
    if not isinstance(values, list):
        raise ValueError("Expected a Python list such as [1.5, 2.0, 2.5]")
    return [float(v) for v in values]


@st.cache_data(show_spinner=False)
def cached_run_pipeline(
    universe: list[str],
    benchmark_ticker: str,
    start_date: str,
    end_date: str,
    train_ratio: float,
    validation_ratio: float,
    min_observations: int,
    max_coint_pvalue: float,
    max_adf_pvalue: float,
    min_half_life: float,
    max_half_life: float,
    top_n_pairs: int,
    lookback_grid: tuple[int, ...],
    entry_z_grid: tuple[float, ...],
    exit_z_grid: tuple[float, ...],
    stop_z_grid: tuple[float, ...],
    recalibration_window: int,
    recalibration_step: int,
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    target_portfolio_vol: float,
    max_gross_leverage: float,
    max_pair_weight: float,
    min_pair_vol: float,
    risk_free_rate: float,
    cooldown_days: int,
    trend_lookback: int,
    max_trend_zscore_slope: float,
    vol_filter_lookback: int,
    max_recent_vol_multiplier: float,
    require_entry_cross: bool,
    quality_weight_strength: float,
) -> dict:
    return run_research_pipeline(
        universe=list(universe),
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        min_observations=min_observations,
        max_coint_pvalue=max_coint_pvalue,
        max_adf_pvalue=max_adf_pvalue,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        top_n_pairs=top_n_pairs,
        lookback_grid=list(lookback_grid),
        entry_z_grid=list(entry_z_grid),
        exit_z_grid=list(exit_z_grid),
        stop_z_grid=list(stop_z_grid),
        recalibration_window=recalibration_window,
        recalibration_step=recalibration_step,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        target_portfolio_vol=target_portfolio_vol,
        max_gross_leverage=max_gross_leverage,
        max_pair_weight=max_pair_weight,
        min_pair_vol=min_pair_vol,
        risk_free_rate=risk_free_rate,
        cooldown_days=cooldown_days,
        trend_lookback=trend_lookback,
        max_trend_zscore_slope=max_trend_zscore_slope,
        vol_filter_lookback=vol_filter_lookback,
        max_recent_vol_multiplier=max_recent_vol_multiplier,
        require_entry_cross=require_entry_cross,
        quality_weight_strength=quality_weight_strength,
    )


def format_pct(x: float) -> str:
    return f"{x:.2%}"


def format_num(x: float) -> str:
    return f"{x:.4f}"


def performance_summary(metrics: dict[str, float]) -> tuple[str, str]:
    sharpe = metrics.get("Sharpe Ratio", 0.0)
    total_return = metrics.get("Total Return", 0.0)
    max_dd = metrics.get("Max Drawdown", 0.0)

    if sharpe > 1.0 and total_return > 0 and max_dd > -0.15:
        return "Constructive", "status-good"
    if sharpe > 0 and total_return > 0:
        return "Mixed", "status-warn"
    return "Weak", "status-bad"


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="main-title">Quant Pairs Lab</div>
            <div class="subtitle">
                Multi-pair statistical arbitrage research dashboard with pair screening, train/validation/test separation,
                parameter tuning, walk-forward recalibration, volatility targeting, sector presets, dynamic replacement,
                and portfolio-level analysis.
            </div>
            <span class="small-tag">Pairs Trading</span>
            <span class="small-tag">Cointegration</span>
            <span class="small-tag">Walk-Forward Testing</span>
            <span class="small-tag">Volatility Targeting</span>
            <span class="small-tag">Research Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_and_guide() -> None:
    col1, col2 = st.columns([1.2, 1.0], gap="large")

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">What this app does</div>
                <div class="section-subtitle">
                    Use the sidebar to define a stock universe and research settings, then run a full
                    multi-pair mean-reversion workflow.
                </div>
                <div class="guide-grid">
                    <div class="guide-item"><b>1. Pair Screening</b><br><span>The app searches your universe for pairs that pass cointegration, ADF, and half-life filters.</span></div>
                    <div class="guide-item"><b>2. Train/Validation/Test Split</b><br><span>Pairs are screened on training data, compared on validation, and finally evaluated on an unseen test segment.</span></div>
                    <div class="guide-item"><b>3. Parameter Tuning</b><br><span>Lookback and Z-score rules are tuned per pair using only training data.</span></div>
                    <div class="guide-item"><b>4. Walk-Forward Portfolio</b><br><span>Selected pairs are recalibrated through time, filtered dynamically, and combined into a weighted portfolio.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">How to read the outputs</div>
                <div class="section-subtitle">
                    Focus on the portfolio metrics first, then inspect the ranking, selection log, and pair drilldown.
                </div>
                <div class="guide-grid">
                    <div class="guide-item"><b>Total / Annualised Return</b><br><span>How much the strategy made overall and per year on a compounded basis.</span></div>
                    <div class="guide-item"><b>Sharpe Ratio</b><br><span>Risk-adjusted return. Higher is better. Negative values usually indicate weak strategy quality.</span></div>
                    <div class="guide-item"><b>Max Drawdown</b><br><span>The worst peak-to-trough portfolio decline during the test period.</span></div>
                    <div class="guide-item"><b>Benchmark / Excess Return</b><br><span>Shows whether the strategy added value versus simply holding the benchmark.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_parameter_guide() -> None:
    with st.expander("Parameter guide and what changing each input means", expanded=False):
        st.markdown(
            """
            <div class="guide-grid">
                <div class="guide-item"><b>Universe preset</b><br><span>Quickly switch to a more coherent sector or industry group. Pairs trading often works better within economically similar stocks.</span></div>
                <div class="guide-item"><b>Universe tickers</b><br><span>The actual stock list used for screening. You can still edit this manually even after applying a preset.</span></div>
                <div class="guide-item"><b>Train ratio</b><br><span>The fraction of history used for initial pair screening and parameter tuning.</span></div>
                <div class="guide-item"><b>Validation ratio</b><br><span>The middle slice of history used to compare candidate pairs and choose final settings before the untouched test period.</span></div>
                <div class="guide-item"><b>Max cointegration p-value</b><br><span>Lower values are stricter and demand stronger evidence of cointegration.</span></div>
                <div class="guide-item"><b>Max ADF p-value</b><br><span>Lower values require stronger evidence that the spread is stationary.</span></div>
                <div class="guide-item"><b>Half-life bounds</b><br><span>Filters how quickly the spread is estimated to revert. Very high half-lives can imply slow or unstable mean reversion.</span></div>
                <div class="guide-item"><b>Top N pairs</b><br><span>How many candidates can be used at each rebalance. More pairs diversify, but can dilute quality.</span></div>
                <div class="guide-item"><b>Lookback grid</b><br><span>Candidate rolling windows for Z-score normalisation. Smaller windows react faster but may be noisier.</span></div>
                <div class="guide-item"><b>Entry / Exit / Stop Z grids</b><br><span>Control when trades open, close, and stop out.</span></div>
                <div class="guide-item"><b>Require fresh entry cross</b><br><span>If enabled, a trade only opens when Z-score freshly crosses into the signal region instead of merely sitting there.</span></div>
                <div class="guide-item"><b>Cooldown days</b><br><span>After a stop-out, the pair waits this many bars before it is allowed to enter again.</span></div>
                <div class="guide-item"><b>Trend lookback / max trend slope</b><br><span>Blocks entries when the spread appears to be trending too strongly instead of reverting.</span></div>
                <div class="guide-item"><b>Vol filter lookback / max recent vol multiplier</b><br><span>Blocks entries when recent spread volatility is too high relative to baseline calibration volatility.</span></div>
                <div class="guide-item"><b>Quality weight strength</b><br><span>Controls how strongly portfolio weights tilt toward better validation and stability characteristics.</span></div>
                <div class="guide-item"><b>Target portfolio vol / leverage</b><br><span>Scales portfolio aggressiveness. Higher values increase both upside and downside.</span></div>
                <div class="guide-item"><b>Commission / slippage</b><br><span>Trading frictions deducted from returns.</span></div>
                <div class="guide-item"><b>Max pair weight</b><br><span>Caps concentration in any one pair.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Research Inputs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-note">Tune the research setup here. Explanations for each control are shown in the main page.</div>',
            unsafe_allow_html=True,
        )

        preset_options = ["Custom"] + list(SECTOR_PRESETS.keys())
        preset = st.selectbox("Universe preset", options=preset_options, key="universe_preset")

        if preset != "Custom":
            preset_text = ", ".join(SECTOR_PRESETS[preset])
            if st.session_state.get("_last_preset_applied") != preset:
                st.session_state["universe_text"] = preset_text
                st.session_state["_last_preset_applied"] = preset
        elif "universe_text" not in st.session_state:
            st.session_state["universe_text"] = ", ".join(UNIVERSE)

        st.text_area(
            "Universe tickers (comma-separated)",
            key="universe_text",
            height=120,
        )
        st.session_state["benchmark_ticker"] = st.text_input("Benchmark ticker", value=BENCHMARK)

        c1, c2 = st.columns(2)
        with c1:
            st.session_state["start_date"] = st.text_input("Start date", value=START_DATE)
        with c2:
            st.session_state["end_date"] = st.text_input("End date", value=END_DATE)

        st.markdown("---")
        st.subheader("Training / screening")
        st.session_state["train_ratio"] = st.slider("Train ratio", 0.40, 0.80, float(TRAIN_RATIO), 0.05)
        st.session_state["validation_ratio"] = st.slider("Validation ratio", 0.10, 0.30, float(VALIDATION_RATIO), 0.05)
        st.session_state["min_observations"] = st.number_input("Minimum observations", min_value=100, value=int(MIN_OBSERVATIONS))
        st.session_state["max_coint_pvalue"] = st.number_input("Max cointegration p-value", min_value=0.001, max_value=0.20, value=float(MAX_COINTEGRATION_PVALUE), step=0.001, format="%.3f")
        st.session_state["max_adf_pvalue"] = st.number_input("Max ADF p-value", min_value=0.001, max_value=0.20, value=float(MAX_ADF_PVALUE), step=0.001, format="%.3f")
        st.session_state["min_half_life"] = st.number_input("Min half-life", min_value=1.0, value=float(MIN_HALF_LIFE), step=1.0)
        st.session_state["max_half_life"] = st.number_input("Max half-life", min_value=2.0, value=float(MAX_HALF_LIFE), step=1.0)
        st.session_state["top_n_pairs"] = st.slider("Top N pairs", 1, 10, int(TOP_N_PAIRS), 1)

        st.markdown("---")
        st.subheader("Parameter grids")
        st.session_state["lookback_grid_text"] = st.text_input("Lookback grid", value=str(LOOKBACK_GRID))
        st.session_state["entry_z_grid_text"] = st.text_input("Entry Z grid", value=str(ENTRY_Z_GRID))
        st.session_state["exit_z_grid_text"] = st.text_input("Exit Z grid", value=str(EXIT_Z_GRID))
        st.session_state["stop_z_grid_text"] = st.text_input("Stop Z grid", value=str(STOP_Z_GRID))
        st.session_state["require_entry_cross"] = st.checkbox("Require fresh entry cross", value=bool(REQUIRE_ENTRY_CROSS))

        st.markdown("---")
        st.subheader("Walk-forward")
        st.session_state["recalibration_window"] = st.number_input("Recalibration window", min_value=50, value=int(RECALIBRATION_WINDOW))
        st.session_state["recalibration_step"] = st.number_input("Recalibration step", min_value=5, value=int(RECALIBRATION_STEP))
        st.session_state["cooldown_days"] = st.number_input("Cooldown days", min_value=0, value=int(COOLDOWN_DAYS), step=1)
        st.session_state["trend_lookback"] = st.number_input("Trend lookback", min_value=5, value=int(TREND_LOOKBACK), step=1)
        st.session_state["max_trend_zscore_slope"] = st.number_input("Max trend slope", min_value=0.01, value=float(MAX_TREND_ZSCORE_SLOPE), step=0.01, format="%.2f")
        st.session_state["vol_filter_lookback"] = st.number_input("Vol filter lookback", min_value=5, value=int(VOL_FILTER_LOOKBACK), step=1)
        st.session_state["max_recent_vol_multiplier"] = st.number_input("Max recent vol multiplier", min_value=0.50, value=float(MAX_RECENT_VOL_MULTIPLIER), step=0.10, format="%.2f")

        st.markdown("---")
        st.subheader("Portfolio / costs")
        st.session_state["initial_capital"] = st.number_input("Initial capital", min_value=1000.0, value=float(INITIAL_CAPITAL), step=1000.0)
        st.session_state["commission_bps"] = st.number_input("Commission (bps)", min_value=0.0, value=float(COMMISSION_BPS), step=0.5)
        st.session_state["slippage_bps"] = st.number_input("Slippage (bps)", min_value=0.0, value=float(SLIPPAGE_BPS), step=0.5)
        st.session_state["target_portfolio_vol"] = st.number_input("Target portfolio vol", min_value=0.01, max_value=1.00, value=float(TARGET_PORTFOLIO_VOL), step=0.01, format="%.2f")
        st.session_state["max_gross_leverage"] = st.number_input("Max gross leverage", min_value=0.1, max_value=10.0, value=float(MAX_GROSS_LEVERAGE), step=0.1)
        st.session_state["max_pair_weight"] = st.number_input("Max pair weight", min_value=0.01, max_value=1.0, value=float(MAX_PAIR_WEIGHT), step=0.01, format="%.2f")
        st.session_state["min_pair_vol"] = st.number_input("Min pair vol", min_value=0.000001, value=float(MIN_PAIR_VOL), step=0.000001, format="%.6f")
        st.session_state["quality_weight_strength"] = st.number_input("Quality weight strength", min_value=0.0, max_value=2.0, value=float(QUALITY_WEIGHT_STRENGTH), step=0.05, format="%.2f")
        st.session_state["risk_free_rate"] = st.number_input("Risk-free rate", value=float(RISK_FREE_RATE), step=0.01, format="%.2f")

        st.markdown("---")
        return st.button("Run Backtest", type="primary", use_container_width=True)


def run_model() -> None:
    universe = [x.strip().upper() for x in st.session_state["universe_text"].split(",") if x.strip()]
    lookback_grid = parse_int_list(st.session_state["lookback_grid_text"])
    entry_z_grid = parse_float_list(st.session_state["entry_z_grid_text"])
    exit_z_grid = parse_float_list(st.session_state["exit_z_grid_text"])
    stop_z_grid = parse_float_list(st.session_state["stop_z_grid_text"])

    result = cached_run_pipeline(
        universe=universe,
        benchmark_ticker=st.session_state["benchmark_ticker"].strip().upper(),
        start_date=st.session_state["start_date"],
        end_date=st.session_state["end_date"],
        train_ratio=float(st.session_state["train_ratio"]),
        validation_ratio=float(st.session_state["validation_ratio"]),
        min_observations=int(st.session_state["min_observations"]),
        max_coint_pvalue=float(st.session_state["max_coint_pvalue"]),
        max_adf_pvalue=float(st.session_state["max_adf_pvalue"]),
        min_half_life=float(st.session_state["min_half_life"]),
        max_half_life=float(st.session_state["max_half_life"]),
        top_n_pairs=int(st.session_state["top_n_pairs"]),
        lookback_grid=tuple(lookback_grid),
        entry_z_grid=tuple(entry_z_grid),
        exit_z_grid=tuple(exit_z_grid),
        stop_z_grid=tuple(stop_z_grid),
        recalibration_window=int(st.session_state["recalibration_window"]),
        recalibration_step=int(st.session_state["recalibration_step"]),
        initial_capital=float(st.session_state["initial_capital"]),
        commission_bps=float(st.session_state["commission_bps"]),
        slippage_bps=float(st.session_state["slippage_bps"]),
        target_portfolio_vol=float(st.session_state["target_portfolio_vol"]),
        max_gross_leverage=float(st.session_state["max_gross_leverage"]),
        max_pair_weight=float(st.session_state["max_pair_weight"]),
        min_pair_vol=float(st.session_state["min_pair_vol"]),
        risk_free_rate=float(st.session_state["risk_free_rate"]),
        cooldown_days=int(st.session_state["cooldown_days"]),
        trend_lookback=int(st.session_state["trend_lookback"]),
        max_trend_zscore_slope=float(st.session_state["max_trend_zscore_slope"]),
        vol_filter_lookback=int(st.session_state["vol_filter_lookback"]),
        max_recent_vol_multiplier=float(st.session_state["max_recent_vol_multiplier"]),
        require_entry_cross=bool(st.session_state["require_entry_cross"]),
        quality_weight_strength=float(st.session_state["quality_weight_strength"]),
    )
    st.session_state["last_result"] = result


def render_metrics(metrics: dict[str, float]) -> None:
    status_text, status_class = performance_summary(metrics)
    time_in_cash = metrics.get("Percentage Time in Cash", 0.0)

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Portfolio snapshot</div>
            <div class="section-subtitle">
                Overall strategy quality assessment:
                <span class="{status_class}">{status_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if time_in_cash >= 0.80:
        st.markdown(
            """
            <div class="block-note">
                The strategy spent most of the test period in cash. This usually means the universe/preset did not
                produce enough acceptable opportunities under the current validation and regime filters.
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    cards = [
        ("Total Return", format_pct(metrics.get("Total Return", 0.0)), "Overall compounded test-period return"),
        ("Annualised Return", format_pct(metrics.get("Annualised Return", 0.0)), "Average annualised growth rate"),
        ("Annualised Volatility", format_pct(metrics.get("Annualised Volatility", 0.0)), "Annualised portfolio risk"),
        ("Sharpe Ratio", format_num(metrics.get("Sharpe Ratio", 0.0)), "Risk-adjusted return"),
        ("Max Drawdown", format_pct(metrics.get("Max Drawdown", 0.0)), "Worst peak-to-trough decline"),
        ("Calmar Ratio", format_num(metrics.get("Calmar Ratio", 0.0)), "Return relative to drawdown"),
        ("Excess Return", format_pct(metrics.get("Excess Return vs Benchmark", 0.0)), "Annualised excess return vs benchmark"),
        ("Information Ratio", format_num(metrics.get("Information Ratio", 0.0)), "Benchmark-relative risk-adjusted return"),
        ("Portfolio Leverage", format_num(metrics.get("Portfolio Leverage Applied", 0.0)), "Vol-target scaling applied"),
        ("Number of Pairs", format_num(metrics.get("Number of Pairs", 0.0)), "Pairs used in the portfolio"),
        ("Time in Cash", format_pct(metrics.get("Percentage Time in Cash", 0.0)), "Share of test period with no active pair exposure"),
        ("Pair Hit Rate", format_pct(metrics.get("Pair Trade Hit Rate", 0.0)), "Winning trade share across pair logs"),
    ]

    rows = [cards[i:i + 4] for i in range(0, len(cards), 4)]
    for row in rows:
        cols = st.columns(4, gap="medium")
        for col, (label, value, note) in zip(cols, row):
            with col:
                st.markdown(metric_card(label, value, note), unsafe_allow_html=True)

    sharpe = metrics.get("Sharpe Ratio", 0.0)
    total_return = metrics.get("Total Return", 0.0)
    benchmark_return = metrics.get("Benchmark Return", 0.0)
    excess_return = metrics.get("Excess Return vs Benchmark", 0.0)
    info_ratio = metrics.get("Information Ratio", 0.0)

    interpretation_lines = []

    if sharpe < 0:
        interpretation_lines.append("The strategy produced negative risk-adjusted performance in this configuration.")
    elif sharpe < 0.5:
        interpretation_lines.append("The strategy is only weakly attractive on a risk-adjusted basis.")
    else:
        interpretation_lines.append("The strategy is showing at least some positive risk-adjusted quality.")

    if total_return < benchmark_return:
        interpretation_lines.append("It underperformed the benchmark over the tested period.")
    else:
        interpretation_lines.append("It outperformed the benchmark over the tested period.")

    if excess_return > 0 and info_ratio > 0:
        interpretation_lines.append("The benchmark-relative profile is constructive.")
    else:
        interpretation_lines.append("There is still room to improve benchmark-relative performance.")

    interpretation_lines.append("Poor or mixed results do not invalidate the project — they usually indicate that the current universe, filters, or trading rules still need refinement.")

    st.markdown(
        f"""
        <div class="interpret-box">
            <b>Interpretation</b><br>
            {'<br>'.join('• ' + line for line in interpretation_lines)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_charts(portfolio_results: pd.DataFrame) -> None:
    chart_tabs = st.tabs(["Equity Curve", "Daily Returns", "Portfolio Table"])

    with chart_tabs[0]:
        st.markdown(
            """
            <div class="block-note">
                Compare the strategy equity curve against the benchmark. Large divergence below the benchmark usually means
                the strategy needs stronger filters, a better universe, or improved entry logic.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.line_chart(portfolio_results[["equity_curve", "benchmark_equity_curve"]])

    with chart_tabs[1]:
        st.markdown(
            """
            <div class="block-note">
                Daily returns help reveal whether gains are smooth, noisy, or dominated by a few large swings.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.line_chart(portfolio_results[["portfolio_return", "benchmark_return"]])

    with chart_tabs[2]:
        st.dataframe(portfolio_results, use_container_width=True)


def render_tables(
    selected_pairs: pd.DataFrame,
    pair_params_df: pd.DataFrame,
    ranking: pd.DataFrame,
    selection_log: pd.DataFrame | None = None,
) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Research outputs</div>
            <div class="section-subtitle">Inspect the chosen pairs, tuned parameters, ranking, and dynamic selection history.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_names = ["Selected Pairs", "Pair Parameters", "Ranking"]
    if selection_log is not None and not selection_log.empty:
        tab_names.append("Selection Log")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        preferred_cols = [
            "ticker_y",
            "ticker_x",
            "train_coint_pvalue",
            "train_adf_pvalue",
            "train_half_life",
            "validation_score",
            "validation_sharpe",
            "validation_ann_return",
            "validation_total_return",
            "validation_max_drawdown",
        ]
        available_cols = [c for c in preferred_cols if c in selected_pairs.columns]
        st.dataframe(selected_pairs[available_cols], use_container_width=True)

    with tabs[1]:
        preferred_cols = [
            "pair_name",
            "ticker_y",
            "ticker_x",
            "lookback",
            "entry_z",
            "exit_z",
            "stop_z",
            "validation_score",
            "validation_sharpe",
            "validation_ann_return",
            "validation_total_return",
        ]
        available_cols = [c for c in preferred_cols if c in pair_params_df.columns]
        st.dataframe(pair_params_df[available_cols], use_container_width=True)

    with tabs[2]:
        preferred_cols = [
            "ticker_y",
            "ticker_x",
            "coint_pvalue",
            "adf_pvalue",
            "half_life",
            "valid",
        ]
        available_cols = [c for c in preferred_cols if c in ranking.columns]
        st.dataframe(ranking[available_cols], use_container_width=True)

    if selection_log is not None and not selection_log.empty:
        with tabs[3]:
            preferred_cols = [
                "rebalance_date",
                "pair_name",
                "weight",
                "validation_sharpe",
                "validation_total_return",
                "current_coint_pvalue",
                "current_adf_pvalue",
                "current_half_life",
                "hist_vol",
            ]
            available_cols = [c for c in preferred_cols if c in selection_log.columns]
            st.dataframe(selection_log[available_cols], use_container_width=True)


def render_pair_drilldown(pair_results: dict[str, pd.DataFrame], pair_trade_logs: dict[str, pd.DataFrame]) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Per-pair drilldown</div>
            <div class="section-subtitle">
                Inspect the spread path, Z-score behaviour, position changes, and completed trades for an individual pair.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pair_results:
        st.warning(
            "No active pair-level results are available for this run. "
            "This usually means the strategy stayed in cash because the preset/universe "
            "did not produce acceptable trading candidates under the current filters."
        )
        return

    pair_names = list(pair_results.keys())
    if len(pair_names) == 0:
        st.warning(
            "No pair drilldown is available for this run."
        )
        return

    selected_pair_name = st.selectbox("Choose a pair", options=pair_names)

    if selected_pair_name not in pair_results:
        st.warning("Selected pair is not available.")
        return

    df = pair_results[selected_pair_name]
    trade_log = pair_trade_logs.get(selected_pair_name, pd.DataFrame())

    tabs = st.tabs(["Spread", "Z-score", "Position", "Trade Log"])

    with tabs[0]:
        if "spread" in df.columns:
            st.line_chart(df[["spread"]])
        else:
            st.info("Spread series not available.")

    with tabs[1]:
        if "zscore" in df.columns:
            st.line_chart(df[["zscore"]])
        else:
            st.info("Z-score series not available.")

    with tabs[2]:
        if "position" in df.columns:
            st.line_chart(df[["position"]])
        else:
            st.info("Position series not available.")

    with tabs[3]:
        if trade_log.empty:
            st.info("No completed trades recorded for this pair.")
        else:
            preferred_cols = [
                "entry_date",
                "exit_date",
                "entry_position",
                "entry_zscore",
                "exit_zscore",
                "holding_days",
                "trade_return",
            ]
            available_cols = [c for c in preferred_cols if c in trade_log.columns]
            st.dataframe(trade_log[available_cols], use_container_width=True)

def render_download_preview(portfolio_results: pd.DataFrame) -> None:
    with st.expander("Download-ready tables preview", expanded=False):
        st.dataframe(portfolio_results.tail(50), use_container_width=True)


inject_css()
render_header()
render_overview_and_guide()
render_parameter_guide()

run_button = render_sidebar()

if run_button:
    try:
        with st.spinner("Running research pipeline..."):
            run_model()
    except Exception as e:
        st.error(f"Run failed: {e}")

result = st.session_state.get("last_result")

if result is None:
    st.info("Set your research inputs in the sidebar, then click Run Backtest.")
else:
    ranking = result["ranking"]
    selected_pairs = result["selected_pairs"]
    pair_params_df = result["pair_params_df"]
    portfolio_output = result["portfolio_output"]
    split_summary = result["split_summary"]
    selection_log = getattr(portfolio_output, "selection_log", pd.DataFrame())

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Data split</div>
            <div class="section-subtitle">
                Train: {split_summary['train_ratio']:.0%} &nbsp;|&nbsp;
                Validation: {split_summary['validation_ratio']:.0%} &nbsp;|&nbsp;
                Test: {split_summary['test_ratio']:.0%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metrics(portfolio_output.metrics)
    render_charts(portfolio_output.portfolio_results)
    render_tables(selected_pairs, pair_params_df, ranking, selection_log)
    render_pair_drilldown(
        portfolio_output.pair_results,
        portfolio_output.pair_trade_logs,
    )
    render_download_preview(portfolio_output.portfolio_results)