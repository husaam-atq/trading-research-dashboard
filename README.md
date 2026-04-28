# Trading Research Dashboard: Pairs Trading & Walk-Forward Backtesting

An interactive quant research dashboard for statistical arbitrage and pairs-trading analysis. The project combines yfinance adjusted close data, correlation and Engle-Granger pair screening, OLS hedge-ratio estimation, z-score signal generation, transaction cost modelling, walk-forward validation, and Streamlit visualisation.

The emphasis is research discipline rather than attractive backtest optics. The current generated results are mixed to poor, especially under walk-forward validation, and the interpretation below reflects that honestly.

## Why It Matters

Pairs trading is a common statistical arbitrage framework: find two historically related assets, model their spread, and trade mean reversion when the spread becomes unusually wide. The hard part is not producing a backtest; it is avoiding look-ahead bias, accounting for costs, testing out of sample, and accepting that relationships can break down.

This repository is designed to demonstrate that workflow in a way an interviewer or reviewer can run and inspect.

## Methodology

1. Download adjusted close prices with yfinance.
2. Screen all pair combinations using log-price correlation.
3. Run an Engle-Granger cointegration test with statsmodels.
4. Estimate the hedge ratio with OLS on log prices.
5. Calculate the residual spread and rolling z-score.
6. Enter long spread when z-score is below the negative entry threshold.
7. Enter short spread when z-score is above the positive entry threshold.
8. Exit when z-score reverts near zero or breaches the stop threshold.
9. Apply a one-day lag between target signal and realised return.
10. Charge transaction costs on executed position changes.
11. Validate with rolling 252-trading-day training windows and 63-trading-day test windows.

## Look-Ahead Bias Control

The batch workflow screens and estimates selected pairs on the first 756 observations, then applies the selected hedge ratios to the full backtest path. Daily strategy returns use `position.shift(1)`, so a signal observed at the end of one session is only realised in the following session.

Walk-forward validation is stricter: each segment estimates hedge ratio, spread mean, and spread volatility on the 252-day training window only, then tests the next 63 trading days. Segment parameters are not estimated from the test period.

## Data And Universe

Default universe:

`AAPL, MSFT, GOOGL, META, AMZN, NVDA, JPM, BAC, XOM, CVX, KO, PEP, WMT, COST, HD, UNH, MRK, V, MA, ORCL, CSCO, INTC`

Default period: `2015-01-01` to `2024-12-31`.

## Project Structure

```text
trading-research-dashboard/
|-- README.md
|-- requirements.txt
|-- app.py
|-- main.py
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- pairs.py
|   |-- signals.py
|   |-- backtest.py
|   |-- metrics.py
|   |-- walk_forward.py
|   `-- plots.py
|-- outputs/
|   `-- .gitkeep
`-- .gitignore
```

## Run The Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard lets you choose a ticker pair, date range, z-score thresholds, transaction cost, and rolling z-score window. It displays price series, spread, z-score bands, equity curve, drawdown, and a metrics table.

## Run Batch Analysis

```bash
pip install -r requirements.txt
python main.py
```

The batch script generates:

- `outputs/pair_screening_results.csv`
- `outputs/backtest_results.csv`
- `outputs/walk_forward_results.csv`
- `outputs/daily_returns.csv`
- `outputs/equity_curves.png`
- `outputs/spread_zscore.png`
- `outputs/drawdowns.png`
- `outputs/walk_forward_performance.png`
- `outputs/pair_comparison.png`

CSV files are ignored by git because they are reproducible generated data. PNG charts are kept so the README can display the generated examples.

## Generated Results

The following tables come from `python main.py` using the default universe and period above. If yfinance revises data or returns slightly different histories, rerunning the script can change these values.

### Pair Screening

Top screened pairs from the first 756 observations:

| Pair | Correlation | Coint p-value | Hedge ratio |
| --- | ---: | ---: | ---: |
| COST/HD | 0.91 | 0.0062 | 0.54 |
| AMZN/META | 0.97 | 0.0076 | 1.30 |
| META/PEP | 0.96 | 0.0079 | 2.55 |
| AMZN/GOOGL | 0.97 | 0.0134 | 1.68 |
| GOOGL/MSFT | 0.95 | 0.0185 | 0.86 |

### Static Selected-Pair Backtests

| Pair | Total return | CAGR | Sharpe | Max drawdown | Trades | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COST/HD | -14.02% | -1.50% | -0.20 | -28.14% | 53 | 52.83% |
| AMZN/META | -26.15% | -2.99% | -0.27 | -38.80% | 51 | 50.98% |
| META/PEP | -1.96% | -0.20% | -0.02 | -34.72% | 58 | 60.34% |
| AMZN/GOOGL | 4.69% | 0.46% | 0.05 | -26.20% | 57 | 54.39% |
| GOOGL/MSFT | 10.61% | 1.02% | 0.14 | -15.46% | 69 | 56.52% |

### Walk-Forward Validation

| Pair | Total return | CAGR | Sharpe | Max drawdown | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| COST/HD | -11.96% | -1.41% | -0.21 | -30.08% | 42 |
| AMZN/META | -11.48% | -1.35% | -0.16 | -22.28% | 35 |
| META/PEP | -49.51% | -7.33% | -0.58 | -55.94% | 62 |
| AMZN/GOOGL | -4.37% | -0.50% | -0.06 | -21.07% | 43 |
| GOOGL/MSFT | -23.04% | -2.87% | -0.45 | -30.27% | 57 |
| Equal-weight selected pairs | -19.80% | -2.43% | -0.55 | -24.35% | 239 |

## Example Charts

![Equity curves](outputs/equity_curves.png)

![Spread z-score](outputs/spread_zscore.png)

![Drawdowns](outputs/drawdowns.png)

![Walk-forward performance](outputs/walk_forward_performance.png)

![Pair comparison](outputs/pair_comparison.png)

## Interpretation

The selected pairs show strong initial correlations and low Engle-Granger p-values in the screening window, but that does not translate into robust profitability. Most static pair backtests lose money after costs, and the equal-weight walk-forward portfolio is negative. This is a useful research outcome: the pair relationships identified early in the sample were not sufficiently stable to produce attractive out-of-sample returns with these simple thresholds and cost assumptions.

The dashboard remains useful for exploring why: z-score excursions can persist, drawdowns can be large, and positive win rates do not necessarily overcome transaction costs and adverse spread moves.

## Limitations

- yfinance data quality can vary and may include missing values, revisions, or vendor-specific adjustments.
- The default universe is static and therefore exposed to survivorship bias.
- Borrow fees, short availability constraints, taxes, and financing costs are not modelled.
- Transaction costs are simplified as a fixed bps cost per one-way trade leg.
- Cointegration relationships can break down or become economically untradeable.
- Backtests may overfit pair selection, threshold choices, and the initial screening window.
- The project does not include a live execution system.

## Future Improvements

- Add sector-neutral and industry-specific universe presets.
- Estimate hedge ratios with rolling or Kalman-filter methods.
- Add borrow fee and financing assumptions.
- Include bootstrap or permutation tests for pair robustness.
- Compare thresholds through nested walk-forward validation.
- Add benchmark and market beta diagnostics.
- Export richer trade logs and attribution reports.
