# Development Results

## Scope

- Data: 2015-01-01 through 2024-12-31
- Nested OOS observations: 1,886
- Train / validation / test: 504 / 126 / 63 trading days
- Test step: 63 trading days, producing contiguous non-overlapping OOS windows
- Transaction cost: 5 bps per leg per one-way position change
- Test state: flat at every test boundary

## Original Baseline Versus Final Development Method

| Metric | Archived baseline headline | Final development method |
| --- | ---: | ---: |
| Total return | 20.72% | -9.41% |
| Sharpe | 0.68 | -0.30 |
| Maximum drawdown | -9.06% | -11.12% |
| Completed OOS trades | 29 | 199 |
| Unique pairs | 5 | 62 |
| Peer groups | 3 | 17 |
| Walk-forward segments | 15 | 30 |
| Average selected pairs | 1.00 | 3.77 |
| Maximum absolute pair P&L share | Not reported | 5.98% |

The baseline headline is higher, but it evaluates alternating quarters, freezes pair identities from the initial training window, allows validation state to cross into test, and selects only one pair per segment. The final development method is broader and cleaner but does not demonstrate profitable 2015-2024 OOS performance.

## Final Development Evidence

| Metric | Result |
| --- | ---: |
| CAGR | -1.31% |
| Annualised volatility | 4.40% |
| Sharpe | -0.30 |
| Sortino | -0.36 |
| Maximum drawdown | -11.12% |
| Monthly win rate | 50.0% |
| Positive segment rate | 43.3% |
| Beta to SPY | 0.020 |
| Correlation to SPY | 0.087 |
| Bootstrap Sharpe 95% interval | [-0.93, 0.34] |
| Newey-West mean-return p-value | 0.458 |

The confidence interval spans both materially negative and modestly positive Sharpe values. The sample does not support a strong claim of positive alpha.

## Cost Sensitivity

Selections and leverage are held fixed. The cost parameter is bps per leg per one-way position change.

| Cost | Total return | Sharpe | Max drawdown |
| ---: | ---: | ---: | ---: |
| 0 bps | -0.61% | -0.02 | -6.49% |
| 1 bp | -2.44% | -0.07 | -6.94% |
| 2 bps | -4.23% | -0.13 | -7.39% |
| 5 bps | -9.41% | -0.30 | -11.12% |
| 10 bps | -17.44% | -0.57 | -18.48% |

## Illustrative Borrow Sensitivity

No historical borrow series is available. This sensitivity assumes each active pair has 50% weighted short gross exposure.

| Assumed annual borrow rate | Total return | Sharpe | Max drawdown |
| ---: | ---: | ---: | ---: |
| 0% | -9.41% | -0.30 | -11.12% |
| 1% | -11.03% | -0.35 | -12.60% |
| 3% | -14.19% | -0.46 | -15.50% |

## Decision

The final development method is frozen despite negative returns because it provides materially stronger evidence: contiguous coverage, 199 trades, many distinct pairs and groups, low P&L concentration, low market beta, and simpler fixed parameters. The more complex alternatives performed worse and showed no validation-to-test predictiveness.
