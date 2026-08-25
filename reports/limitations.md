# Limitations

1. Yahoo Finance adjusted-close data can be revised, contain vendor errors, and does not provide executable bid/ask prices.
2. The present-day peer universe creates survivorship and constituent-selection bias. Historical membership is not reconstructed.
3. Liquidity is assumed from instrument choice, not verified from historical volume, spreads, or market depth.
4. Transaction costs are fixed at 5 bps per leg per one-way position change. Market impact and spread variation are simplified.
5. Short availability is not modelled. Borrow sensitivity uses an illustrative 50% short gross assumption rather than historical borrow data.
6. Cointegration and ADF tests have low power and are sensitive to sample choice. Near-collinear ETFs can produce unreliable cointegration warnings.
7. The development process tested several architectures on 2015-2024. Its final nested record is therefore development pseudo-OOS, not pristine holdout evidence.
8. The fresh confirmation has only 378 daily observations, 43 trades, and six segments. Statistical uncertainty remains high.
9. Confirmation P&L concentration exceeded the preferred 25% gate for one pair.
10. The strategy is evaluated at daily frequency and assumes next-day implementation without intraday path modelling.
11. Corporate actions are delegated to adjusted-close data. Taxes, financing, margin rules, and operational constraints are absent.
12. There is no live execution, order management, monitoring, or production risk system.

The repository should be interpreted as a research-validation framework and portfolio project, not as evidence that the strategy is ready for capital deployment.
