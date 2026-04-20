# Statistical Arbitrage Research Dashboard

An interactive research dashboard for exploring **statistical arbitrage** and **pairs trading** ideas using a more realistic research workflow.

This project is designed as a **research environment**, not a finished live-trading system. It includes pair screening, train/validation/test separation, parameter tuning, walk-forward recalibration, dynamic pair replacement, portfolio construction, and benchmark-relative evaluation through a Streamlit web app.

## Project status

**Active work in progress**

Current functionality includes:
- cointegration-based pair screening
- ADF-based spread stationarity filtering
- half-life filtering
- train / validation / test workflow
- per-pair parameter tuning
- walk-forward recalibration
- dynamic pair filtering and replacement
- portfolio construction with volatility targeting
- benchmark comparison
- interactive Streamlit dashboard
- sector / universe presets

This is an ongoing research project and is still being improved.

## Motivation

The goal of this project is to build a more credible and transparent workflow for pairs trading research rather than relying on naive in-sample backtests.

Instead of treating a backtest as “proof” of a strategy, the project is designed to:
- separate training, validation, and test periods
- expose weak strategies honestly
- compare candidate pairs more systematically
- explore how pair stability changes through time
- provide an interactive interface for testing assumptions

## Features

### Research pipeline
- Download historical market data with `yfinance`
- Screen candidate pairs from a user-defined universe
- Filter pairs using:
  - Engle-Granger cointegration
  - ADF spread stationarity
  - half-life bounds
- Tune trading parameters on the training set only
- Compare candidate pairs on a validation set
- Run the final strategy on an unseen test set

### Strategy logic
- Mean-reversion trading on spread z-scores
- Entry / exit / stop thresholds
- Optional requirement for a fresh signal crossing
- Cooldown period after stop-outs
- Trend filter to avoid trading strongly trending spreads
- Volatility / regime filter to block unstable periods
- Dynamic pair replacement from a reserve candidate pool

### Portfolio construction
- Multi-pair portfolio
- Inverse-volatility style weighting
- quality-weight tilt
- volatility targeting
- leverage cap
- benchmark-relative performance metrics

### Interactive app
- Streamlit dashboard
- editable universes
- sector presets
- parameter controls
- portfolio metrics
- equity curve and returns charts
- selected pairs table
- ranking table
- selection log
- per-pair drilldown

## Tech stack

- Python
- pandas
- numpy
- statsmodels
- yfinance
- scikit-learn
- matplotlib
- Streamlit

## Project structure

```text
pairs_trading/
├── app.py
├── main.py
├── config.py
├── core.py
├── data_loader.py
├── pair_selection.py
├── strategy.py
├── portfolio.py
├── pipeline.py
├── reporting.py
├── requirements.txt
├── .gitignore
└── README.md
```
## How it works
***1. Pair screening***

Candidate pairs are tested on the training segment using:

- cointegration p-value
- ADF p-value on the spread
- estimated half-life of mean reversion

Only pairs passing these filters move forward.

***2. Parameter tuning***

For each candidate pair, the model tests combinations of:

- lookback window
- entry z-score
- exit z-score
- stop z-score

These are tuned on the training period only.

***3. Validation ranking***

Candidate pairs are then scored on the validation period using metrics such as:

- validation score
- validation Sharpe
- validation annualised return
- validation total return
- validation drawdown
- activity / trade frequency

This helps reduce overfitting to the training sample.

***4. Dynamic test-period portfolio***

During the final test period:

- pairs are re-evaluated block by block
- unstable pairs can be dropped
- stronger reserve candidates can replace weaker ones
- the final portfolio is built dynamically

## Current limitations

This is still research code and has important limitations:

- not intended for live trading
- no intraday execution logic
- no order book / microstructure modelling
- transaction cost model is still simplified
- no borrow fee modelling
- no corporate actions / survivorship checks beyond data source defaults
- strategy performance is still mixed and under active improvement
- benchmark-relative performance remains a challenge

## Current interpretation

This project should be viewed as:

- a research / experimentation environment
- an active statistical arbitrage project
- a work in progress

It should not be interpreted as a claim of a production-ready profitable trading system.

## Planned improvements

Planned next steps include:

- preset comparison mode across sectors / universes
- improved validation scoring
- better regime detection
- stronger benchmark-relative analysis
- improved entry confirmation logic
- enhanced portfolio allocation rules
- richer trade attribution and diagnostics
- broader universe testing
- cleaner reporting / export features

## Running locally

***1. Create a virtual environment***

Windows:
```
py -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```
***2. Install requirements***
```
pip install -r requirements.txt
```
***3. Run the Streamlit app***
```
python -m streamlit run app.py
```
***4. Run the CLI version***
```
python main.py
```
## Example use cases

- exploring whether sector-specific universes produce more stable pairs
- comparing parameter settings under different validation rules
- testing how dynamic replacement affects performance
- evaluating benchmark-relative performance of a market-neutral style strategy
- demonstrating a structured research workflow

## Why this project matters

A lot of market strategy projects stop at a simple backtest. This one is intended to go further by explicitly showing:

- how fragile naive strategies can be
- how methodology affects results
- how validation and unseen testing can change conclusions
- how an idea evolves through structured research rather than cherry-picked performance
