# strategy.py

from __future__ import annotations

from itertools import product
import pandas as pd

from core import rolling_zscore, sharpe_ratio, annualised_return


def generate_positions(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    can_trade: pd.Series | None = None,
    cooldown_days: int = 0,
    require_crossing: bool = False,
) -> pd.Series:
    """
    Position convention:
    +1 => long spread
    -1 => short spread
     0 => flat

    Options:
    - can_trade controls whether new entries are allowed
    - cooldown_days waits after a stop-out
    - require_crossing only allows fresh entries when z-score crosses into signal zone
    """
    if can_trade is None:
        can_trade = pd.Series(True, index=zscore.index)

    can_trade = can_trade.reindex(zscore.index).fillna(False)

    current_pos = 0
    cooldown_remaining = 0
    prev_z = None
    out = []

    for dt, z in zscore.items():
        allow_new_entry = bool(can_trade.loc[dt])

        if cooldown_remaining > 0 and current_pos == 0:
            allow_new_entry = False
            cooldown_remaining -= 1

        if pd.isna(z):
            out.append(current_pos)
            prev_z = z
            continue

        crossed_short = False
        crossed_long = False

        if prev_z is not None and not pd.isna(prev_z):
            crossed_short = prev_z < entry_z and z >= entry_z
            crossed_long = prev_z > -entry_z and z <= -entry_z

        if current_pos == 0:
            if allow_new_entry:
                if require_crossing:
                    if crossed_short:
                        current_pos = -1
                    elif crossed_long:
                        current_pos = 1
                else:
                    if z >= entry_z:
                        current_pos = -1
                    elif z <= -entry_z:
                        current_pos = 1

        elif current_pos == 1:
            if abs(z) <= exit_z:
                current_pos = 0
            elif abs(z) >= stop_z:
                current_pos = 0
                cooldown_remaining = cooldown_days

        elif current_pos == -1:
            if abs(z) <= exit_z:
                current_pos = 0
            elif abs(z) >= stop_z:
                current_pos = 0
                cooldown_remaining = cooldown_days

        out.append(current_pos)
        prev_z = z

    return pd.Series(out, index=zscore.index, name="position")


def compute_trade_filter(
    spread: pd.Series,
    trend_lookback: int,
    max_trend_zscore_slope: float,
    vol_filter_lookback: int,
    max_recent_vol_multiplier: float,
    baseline_spread_vol: float,
) -> pd.DataFrame:
    """
    Build entry filters for:
    1. spread trend not too strong
    2. recent spread volatility not too extreme
    """
    rolling_mean = spread.rolling(trend_lookback).mean()
    rolling_std = spread.rolling(trend_lookback).std(ddof=0).replace(0.0, pd.NA)

    trend_score = (rolling_mean.diff() / rolling_std).abs()
    trend_ok = trend_score <= max_trend_zscore_slope

    recent_vol = spread.diff().rolling(vol_filter_lookback).std(ddof=0)
    vol_limit = max(baseline_spread_vol, 1e-12) * max_recent_vol_multiplier
    regime_ok = recent_vol <= vol_limit

    can_trade = trend_ok.fillna(False) & regime_ok.fillna(False)

    return pd.DataFrame({
        "trend_score": trend_score,
        "trend_ok": trend_ok.fillna(False),
        "recent_spread_vol": recent_vol,
        "regime_ok": regime_ok.fillna(False),
        "can_trade": can_trade,
    })


def simple_pair_returns(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    position: pd.Series,
    commission_bps: float,
    slippage_bps: float
) -> pd.Series:
    """
    Simple training/validation-period backtest.
    """
    df = pd.DataFrame({"y": y, "x": x, "position": position}).dropna()
    df["ret_y"] = df["y"].pct_change()
    df["ret_x"] = df["x"].pct_change()
    df["spread_return"] = df["ret_y"] - hedge_ratio * df["ret_x"]
    df["position_lag"] = df["position"].shift(1).fillna(0.0)
    df["position_change"] = df["position"].diff().abs().fillna(df["position"].abs())

    total_cost_bps = commission_bps + slippage_bps
    df["trading_cost"] = (df["position_change"] * 2.0 * total_cost_bps) / 10000.0

    df["net_return"] = df["position_lag"] * df["spread_return"] - df["trading_cost"]
    return df["net_return"].fillna(0.0)


def tune_parameters_for_pair(
    spread: pd.Series,
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    lookback_grid: list[int],
    entry_z_grid: list[float],
    exit_z_grid: list[float],
    stop_z_grid: list[float],
    commission_bps: float,
    slippage_bps: float,
    require_crossing: bool = False,
) -> dict:
    """
    Tune on training data only.
    Objective: highest Sharpe, then annualised return.
    """
    records = []

    for lookback, entry_z, exit_z, stop_z in product(
        lookback_grid, entry_z_grid, exit_z_grid, stop_z_grid
    ):
        if exit_z >= entry_z or stop_z <= entry_z:
            continue
        if len(spread.dropna()) < lookback + 20:
            continue

        z = rolling_zscore(spread, lookback)
        pos = generate_positions(
            zscore=z,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            require_crossing=require_crossing,
        )
        ret = simple_pair_returns(
            y=y,
            x=x,
            hedge_ratio=hedge_ratio,
            position=pos,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps
        )

        sr = sharpe_ratio(ret)
        ar = annualised_return(ret)

        records.append({
            "lookback": lookback,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "stop_z": stop_z,
            "sharpe": sr,
            "ann_return": ar
        })

    if not records:
        raise ValueError("No valid parameter combinations were tested.")

    ranked = pd.DataFrame(records).sort_values(
        by=["sharpe", "ann_return"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return ranked.iloc[0].to_dict()