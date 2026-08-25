from __future__ import annotations


EXPANDED_STOCK_PEER_GROUPS: dict[str, list[str]] = {
    "mega_cap_platforms": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "enterprise_software": ["MSFT", "ORCL", "CRM", "ADBE", "INTU", "NOW", "ADSK"],
    "semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU", "TXN", "ADI", "MCHP", "NXPI"],
    "money_center_banks": ["JPM", "BAC", "C", "WFC"],
    "capital_markets": ["GS", "MS", "SCHW", "NTRS", "STT"],
    "payments_credit": ["V", "MA", "AXP", "COF", "SYF"],
    "integrated_energy": ["XOM", "CVX", "COP", "OXY"],
    "oilfield_services": ["SLB", "HAL", "BKR"],
    "big_box_retail": ["WMT", "COST", "TGT", "BJ"],
    "home_improvement": ["HD", "LOW", "TSCO"],
    "beverages": ["KO", "PEP", "KDP", "MNST"],
    "large_pharma": ["JNJ", "MRK", "PFE", "ABBV", "BMY", "LLY"],
    "managed_care": ["UNH", "CI", "CVS", "HUM", "CNC"],
    "industrial_machinery": ["CAT", "DE", "CMI", "PCAR"],
    "railroads": ["UNP", "CSX", "NSC"],
}


EXPANDED_ETF_PEER_GROUPS: dict[str, list[str]] = {
    "broad_us_equity": ["SPY", "IVV", "VOO", "VTI", "SCHB"],
    "us_growth_technology": ["QQQ", "XLK", "VGT", "IYW"],
    "developed_international": ["EFA", "VEA", "IEFA"],
    "emerging_markets": ["EEM", "VWO", "IEMG"],
    "short_treasuries": ["SHY", "VGSH", "SCHO"],
    "intermediate_treasuries": ["IEI", "IEF", "VGIT", "SCHR", "GOVT"],
    "long_treasuries": ["TLH", "TLT", "VGLT", "EDV"],
    "physical_gold": ["GLD", "IAU", "SGOL"],
    "us_real_estate": ["VNQ", "SCHH", "IYR"],
    "energy_equity": ["XLE", "VDE", "XOP", "OIH"],
    "financial_equity": ["XLF", "VFH", "KBE", "KRE"],
}


def expanded_research_universes() -> dict[str, dict[str, list[str]]]:
    return {
        "stock_peer_groups": EXPANDED_STOCK_PEER_GROUPS,
        "etf_peer_groups": EXPANDED_ETF_PEER_GROUPS,
    }


def flatten_expanded_universe() -> list[str]:
    tickers = {
        ticker
        for groups in expanded_research_universes().values()
        for members in groups.values()
        for ticker in members
    }
    tickers.add("SPY")
    return sorted(tickers)
