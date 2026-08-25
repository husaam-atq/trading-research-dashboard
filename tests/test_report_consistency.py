from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_readme_headlines_match_generated_outputs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    baseline = pd.read_csv(ROOT / "reports/baseline/data/sharpe_optimised_results.csv")
    baseline_row = baseline[baseline["method"] == "portfolio_vol_target_8"].iloc[0]
    development = pd.read_csv(ROOT / "reports/development/final/performance_metrics.csv").iloc[0]
    confirmation = pd.read_csv(ROOT / "reports/confirmation/performance_metrics.csv").iloc[0]

    expected = [
        f"{baseline_row['total_return']:.2%}",
        f"{baseline_row['sharpe_ratio']:.2f}",
        f"{development['total_return']:.2%}",
        f"{development['sharpe_ratio']:.2f}",
        f"{confirmation['total_return']:.2%}",
        f"{confirmation['sharpe_ratio']:.2f}",
    ]
    for value in expected:
        assert value in readme


def test_frozen_config_hash_matches_manifest() -> None:
    from hashlib import sha256

    config_path = ROOT / "config/final_methodology_v1.json"
    expected = (ROOT / "config/final_methodology_v1.json.sha256").read_text(encoding="ascii").split()[0]
    assert sha256(config_path.read_bytes()).hexdigest() == expected
