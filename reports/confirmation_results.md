# 2025-2026 Frozen Confirmation Results

## Integrity Record

- Frozen methodology SHA-256: `06a2b8567e4247684fe07a8ca37cbdf7887ed8a709cf177fcaef8caacc44fd1f`
- Confirmation period: 2025-01-01 through 2026-07-31
- Confirmation run count: one
- Test segments: 6 contiguous 63-day windows
- Model changes after observation: none

## Result

| Metric | Result |
| --- | ---: |
| Total return | 3.83% |
| CAGR | 2.54% |
| Annualised volatility | 4.29% |
| Sharpe | 0.59 |
| Sortino | 0.84 |
| Maximum drawdown | -3.85% |
| Completed OOS trades | 43 |
| Average selected pairs | 3.83 |
| Unique pairs | 19 |
| Peer groups | 12 |
| Positive segments | 4 of 6 |
| Beta to SPY | -0.010 |
| Correlation to SPY | -0.043 |
| Maximum absolute pair P&L share | 28.34% |
| Bootstrap Sharpe 95% interval | [-0.88, 2.00] |
| Newey-West mean-return p-value | 0.421 |

## Interpretation

The frozen method produced a positive, low-beta confirmation result after 5 bps per leg costs. It remained positive at 10 bps, but the evidence is not statistically decisive: there are only 378 daily observations, 43 trades, and six segments, while the bootstrap interval is very wide. One pair, MS/NTRS, contributed 28.34% of absolute P&L, above the preferred 25% concentration gate.

This is encouraging evidence that the broad, fixed methodology can behave better in a fresh period. It is not evidence of deployable profitability and does not overturn the negative 2015-2024 development result.

## Cost Sensitivity

| Cost | Total return | Sharpe | Max drawdown |
| ---: | ---: | ---: | ---: |
| 0 bps | 5.95% | 0.92 | -3.66% |
| 1 bp | 5.52% | 0.85 | -3.70% |
| 2 bps | 5.10% | 0.79 | -3.74% |
| 5 bps | 3.83% | 0.59 | -3.85% |
| 10 bps | 1.76% | 0.27 | -4.05% |

## Illustrative Borrow Sensitivity

Using the same assumed 50% short gross exposure convention:

| Assumed annual borrow rate | Total return | Sharpe | Max drawdown |
| ---: | ---: | ---: | ---: |
| 0% | 3.83% | 0.59 | -3.85% |
| 1% | 3.49% | 0.54 | -3.89% |
| 3% | 2.79% | 0.43 | -3.98% |
