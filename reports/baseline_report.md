# Baseline Research Report

## Provenance

- Source commit: `581bf9a0156f8f1fb839fa89eeb4d5b606200eb7`
- Reproduction date: 2026-08-25
- Development data requested: 2015-01-01 through 2024-12-31
- Data source: Yahoo Finance adjusted close data downloaded by `yfinance`
- Command: `py main.py`
- Exit status: successful
- Archived artifacts: `reports/baseline/data/` and `reports/baseline/charts/`
- Integrity manifest: `reports/baseline/SHA256SUMS.txt`

## Existing Methodology

The baseline restricts pairs to stock and ETF peer groups, screens the first 504 observations, freezes a candidate pool from that initial period, and then performs rolling 504-day training, 126-day validation, and 63-day test experiments. Validation chooses one hedge mode and pair-specific thresholds. Test returns lag target positions and hedge ratios by one trading day. The reported headline applies an 8% annualised volatility target with a 1.5x leverage cap to the nested OOS portfolio.

The transaction-cost setting is 5 bps. In the baseline implementation, a unit position change is charged twice this amount to represent the two pair legs, so a simple entry or exit costs approximately 10 bps before any subsequent position change.

## Reproduced Headline

| Metric | Unscaled inverse-volatility portfolio | Reported 8% portfolio-vol target |
| --- | ---: | ---: |
| Total return | 13.77% | 20.72% |
| CAGR | 3.50% | 5.15% |
| Annualised volatility | 6.73% | 7.53% |
| Sharpe | 0.52 | 0.68 |
| Sortino | 0.47 | 0.66 |
| Maximum drawdown | -9.47% | -9.06% |
| Profit factor | 1.18 | 1.24 |
| Completed OOS trades | 29 | 29 |
| Average holding period | 12.9 days | 12.9 days |
| Beta to SPY | 0.007 | 0.010 |
| Correlation to SPY | 0.023 | 0.028 |

## Breadth And Coverage

- Walk-forward segments: 15
- OOS observations: 945
- First OOS date: 2017-07-05
- Last OOS date: 2024-10-07
- Selected pairs per segment: 1 in every segment
- Segments marked `breadth_limited`: 15 of 15
- Unique selected pairs: 5
- Peer groups represented: 3
- Pair selection frequency: META/AMZN 8, VEA/VWO 3, COST/LOW 2, COST/HD 1, VEA/EEM 1
- Stable candidates in the initial screen: 1

The test step is 126 trading days while each test window is 63 days. Consequently, roughly one quarter is evaluated and the following quarter is skipped. The reported result is not a contiguous development-period OOS record.

## Known Limitations

1. `MAX_VALIDATION_CANDIDATES = 1` prevents the selector from evaluating enough candidates to construct a three-pair portfolio.
2. Candidate identities are frozen from the first 504 observations rather than rediscovered within each rolling training window.
3. The one-pair fallback makes all unscaled weighting methods identical.
4. A strategy selected at validation end is replayed through validation before test extraction, permitting a validation-entered position to carry into test.
5. Post-hoc cost and filter sensitivities rebuild later segments from the global historical prefix rather than each segment's own training boundary.
6. Static all-pairs comparison returns include the fitting period and are not fully OOS.
7. Portfolio weight clipping is followed by renormalisation, which can violate the stated maximum weight.
8. The result has only 29 trades and is concentrated in META/AMZN by selection frequency.
9. The present-day universe creates survivorship bias, and adjusted-close data does not establish historical liquidity or borrow availability.
10. Borrow fees, short availability, market impact, and execution slippage beyond the fixed cost assumption are absent.

This archived result is the comparison baseline. Later results must not overwrite it or present it as a broader or more statistically certain finding than it is.
