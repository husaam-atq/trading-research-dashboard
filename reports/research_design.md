# Research Design And Audit

## Research Question

Can a rolling, economically constrained pair-discovery process produce a broader and more stable net OOS pairs portfolio than the archived baseline without using test-period information for selection?

## Current Architecture

- `src/data.py`: universe definitions and Yahoo Finance adjusted-close loading.
- `src/pairs.py`: OLS spread construction, Engle-Granger, ADF, OU, half-life, and pair screening.
- `src/signals.py`: z-score positions, exits, time stops, and cooldown state.
- `src/backtest.py`: hedge estimators, signal filters, lagged pair returns, costs, and trade logs.
- `src/walk_forward.py`: older single-pair walk-forward implementation.
- `src/robust.py`: stability diagnostics, validation grid, nested pair selection, weighting, and sensitivities.
- `main.py`: orchestration, output generation, benchmark comparison, and charts.
- `app.py`: interactive single-pair Streamlit research view.

## Audit Findings

### Timing controls that are sound

- Target positions are shifted one day before realised pair returns.
- Hedge ratios are shifted one day before return attribution.
- Rolling OLS uses observations strictly before its estimation date.
- Kalman parameters use information through the signal date and are applied to the next realised return.
- Portfolio volatility scaling is based on trailing returns and shifted one day.
- Training, validation, and test slices are chronologically ordered in the primary nested loop.

### Integrity and implementation weaknesses

1. Test-boundary state is not clean. Chosen parameters are replayed over validation, and an open position can cross into test even though the parameter choice did not exist at its validation entry date.
2. The initial candidate pool freezes pair identities using only the first training window. Later training windows cannot discover new relationships.
3. Only one candidate reaches validation despite a nominal top-three portfolio objective.
4. The 126-day step leaves alternate 63-day quarters unevaluated.
5. Sensitivity replays use global prefixes, which misaligns fitting data for later rolling segments.
6. Weight caps can be broken by normalising after clipping.
7. Pair-level concentration, ticker reuse, peer-group concentration, and pair-return correlation are uncontrolled.
8. Screening trusts one full training sample. It does not measure whether correlation, stationarity, half-life, or hedge ratio persist across training subwindows.
9. Validation ranking relies heavily on noisy Sharpe and profit factor estimates from a 126-day sample.
10. Cost accounting is conservative but ambiguously documented, and short borrow cost is absent.
11. Data cleaning drops rows globally after forward filling, coupling usable history to the least complete instrument.
12. No automated tests enforce timing, accounting, concentration, or deterministic output properties.

## Proposed Development Experiments

All experiments use 2015-2024 only. Each segment uses a 504-day train window, 126-day validation window, 63-day untouched test window, and a 63-day step for contiguous OOS coverage.

1. **Rolling discovery, existing universe**: remove the frozen candidate pool and evaluate multiple training-ranked candidates.
2. **Expanded economic peers**: add larger, configurable stock industry groups and retain economically related ETF groups.
3. **Subwindow stability**: require or reward repeated correlation, residual stationarity, finite half-life, and hedge-ratio stability across training subwindows.
4. **Parsimonious validation search**: stage hedge-mode and threshold selection to avoid an indiscriminate Cartesian search.
5. **Breadth controls**: select several positive-validation candidates subject to pair, ticker, peer-group, and pair-return-correlation constraints.
6. **Weighting**: compare equal, constrained inverse-volatility, and validation-score risk budgets using validation information only.
7. **Implementation sensitivity**: preserve fixed two-leg costs and add clearly labelled assumed borrow-cost sensitivity.
8. **Negative evidence**: retain filter and expected-edge variants that fail to improve balanced development evidence.

## Adoption Criteria

The final method will not be the configuration with the highest development Sharpe automatically. Selection will balance:

- net nested-OOS Sharpe and drawdown;
- completed trades and active-pair breadth;
- unique pairs and peer groups;
- positive-segment frequency;
- pair and ticker P&L concentration;
- turnover and cost resilience;
- bootstrap uncertainty;
- simplicity and interview-level explainability.

A broader result with modestly lower Sharpe may be preferred to a concentrated high-Sharpe result. No 2025+ observation may influence this decision.

## Confirmation Protocol

After development experiments are complete, the chosen methodology and evaluation gates will be written to a versioned JSON file and hashed. Only then will data from 2025-01-01 through the last completed month be downloaded and evaluated. The confirmation run will use the frozen rolling protocol and will not trigger any subsequent model, universe, threshold, filter, or weighting change.
