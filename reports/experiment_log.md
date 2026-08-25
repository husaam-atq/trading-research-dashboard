# Experiment Log

All experiments below use only 2015-2024 data. Later designs were informed by development evidence, so their nested test records are development pseudo-OOS rather than pristine confirmation evidence. The 2025+ run remained untouched until the final configuration was frozen.

| Experiment | Return | Sharpe | Max DD | Trades | Unique pairs | Groups | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Archived baseline headline | 20.72% | 0.68 | -9.06% | 29 | 5 | 3 | Preserved, but rejected as final evidence because coverage and breadth are limited |
| Rolling discovery, original universes | -32.49% | -0.85 | -35.25% | 217 | 50 | 10 | Rejected: breadth rose but test performance and drawdown deteriorated |
| Expanded discovery plus staged optimiser | -35.58% | -0.83 | -38.18% | 337 | 58 | 13 | Rejected: validation Sharpe did not predict next-window returns and parameter search overfit |
| Fixed Kalman, validation-ranked pairs | -16.89% | -0.43 | -20.95% | 146 | 46 | 15 | Rejected: simpler and better, but pair-level validation ranking remained noisy |
| Fixed Kalman, training-ranked equal weight | -11.52% | -0.33 | -13.07% | 199 | 62 | 17 | Retained as core selection design |
| Final constrained inverse volatility | -9.41% | -0.30 | -11.12% | 199 | 62 | 17 | Frozen: simple weighting improved risk without changing pair identities or trades |

## Rejected Features

- Pair-specific threshold and hedge-mode optimisation: selected validation Sharpe averaged roughly 2.8 but had approximately zero correlation with subsequent test returns.
- Portfolio volatility targeting: 5%, 8%, and 10% targets all worsened the training-ranked development result.
- Validation-score and score-risk weighting: score-risk was numerically best among weighting variants, but constrained inverse volatility was chosen for lower model dependence and easier interpretation.
- Correlation/trend/combined filters from the baseline: retained in the baseline archive as negative evidence; they did not justify inclusion in the final fixed model.
- Expected-edge filter from the baseline: did not materially change entries and was not retained.

## Confirmation

The single frozen 2025-2026 confirmation produced 3.83% total return, 0.59 Sharpe, -3.85% maximum drawdown, 43 trades, 19 pairs, and 12 peer groups. No design was changed after observing it.
