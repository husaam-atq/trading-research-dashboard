# Final Frozen Methodology

- Frozen configuration: `config\final_methodology_v1.json`
- SHA-256: `06a2b8567e4247684fe07a8ca37cbdf7887ed8a709cf177fcaef8caacc44fd1f`
- Development period: 2015-2024
- Confirmation period: 2025-01-01 through 2026-07-31

## Decision

The final method uses rolling training-only pair discovery, repeated 252-day stability diagnostics, a fixed Kalman hedge, fixed 2.0 entry / 0.0 exit / 3.5 stop thresholds, a 20-day time stop, training-ranked diversified pair selection, constrained inverse-volatility weights, no volatility target, and 5 bps per leg per one-way position change.

It was selected because it is simpler and less dependent on noisy 126-day validation winners than the staged optimiser. Constrained inverse-volatility weighting improved development drawdown and risk-adjusted performance without changing pair identities, strategy parameters, or trade count. The more complex score-risk weighting was rejected despite a slightly less negative development result.

## Development Evidence At Freeze

# Final Development Methodology

| Metric | Result |
| --- | ---: |
| Total return | -9.41% |
| CAGR | -1.31% |
| Annualised volatility | 4.40% |
| Sharpe | -0.30 |
| Sortino | -0.36 |
| Maximum drawdown | -11.12% |
| Completed OOS trades | 199 |
| Average selected pairs | 3.77 |
| Unique pairs | 62 |
| Peer groups | 17 |
| Positive segment rate | 43.3% |
| Maximum absolute pair P&L share | 6.0% |
| Sharpe bootstrap 95% interval | [-0.93, 0.34] |
| Newey-West mean-return p-value | 0.458 |


The development result is negative. The methodology is frozen for research credibility and breadth, not because it demonstrates profitable alpha.
