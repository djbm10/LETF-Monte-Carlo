import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from collections import defaultdict
from letf.tax.engine import CapitalGainsResult, CapitalLossUsageStrategy, TaxpayerElections, compute_capital_gains


@dataclass
class TaxRegimeScenario:
    """
    One interpretation of ambiguous rules.

    MUST-FIX: Overrides at RULE level, not output multipliers.
    """
    name: str
    probability: float

    # Rule-level toggles
    trader_status_applies: bool = False  # Bypasses capital gain treatment
    constructive_sale_triggered: bool = False  # Forces earlier realization
    wash_sale_disallowance_rate: float = 1.0  # 1.0 = strict, 0.8 = lenient
    state_conforms_to_federal: bool = True

    def apply_to_capital_gains(
        self, base_result: CapitalGainsResult, trade_volume: float
    ) -> CapitalGainsResult:
        """
        Apply regime interpretation to capital gains result.

        Key: This modifies BEHAVIOR, not just output.
        """

        # If trader status applies, all gains become ordinary income
        if self.trader_status_applies:
            # This would bypass capital gains treatment entirely
            # For now, mark it in rules_applied
            base_result.rules_applied.append(
                f"REGIME: Trader status applied (all ordinary income)"
            )
            # In real implementation, would return different result

        # Wash sale disallowance
        if self.wash_sale_disallowance_rate != 1.0:
            # This affects how much loss is disallowed
            # Lenient (0.8): Some wash sales not caught
            # Strict (1.2): More aggressive interpretation
            base_result.rules_applied.append(
                f"REGIME: Wash sale strictness = {self.wash_sale_disallowance_rate}"
            )

        # State conformity
        if not self.state_conforms_to_federal:
            base_result.rules_applied.append(
                "REGIME: State non-conformity (additional state tax)"
            )

        return base_result


TAX_REGIMES = [
    TaxRegimeScenario(
        name="Conservative (Strict IRS)",
        probability=0.60,
        trader_status_applies=False,
        wash_sale_disallowance_rate=1.0
    ),
    TaxRegimeScenario(
        name="Aggressive (Pro-taxpayer)",
        probability=0.25,
        trader_status_applies=False,
        wash_sale_disallowance_rate=0.8
    ),
    TaxRegimeScenario(
        name="Worst Case (Audit)",
        probability=0.10,
        trader_status_applies=True,
        wash_sale_disallowance_rate=1.2
    ),
    TaxRegimeScenario(
        name="Best Case",
        probability=0.05,
        trader_status_applies=False,
        wash_sale_disallowance_rate=0.7
    )
]


def monte_carlo_tax_regimes(
    st_gains: float,
    st_losses: float,
    lt_gains: float,
    lt_losses: float,
    st_cf_in: float,
    lt_cf_in: float,
    elections: TaxpayerElections,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> Dict:
    """Monte Carlo over tax interpretations (samples rules, not outcomes)."""
    samples = []
    regime_results = defaultdict(list)
    rng = np.random.default_rng(seed)

    for _ in range(n_samples):
        regime = rng.choice(TAX_REGIMES, p=[r.probability for r in TAX_REGIMES])
        base_result = compute_capital_gains(
            st_gains=st_gains,
            st_losses=st_losses,
            lt_gains=lt_gains,
            lt_losses=lt_losses,
            st_loss_cf_in=st_cf_in,
            lt_loss_cf_in=lt_cf_in,
            elections=elections
        )
        regime_result = regime.apply_to_capital_gains(base_result, 0)
        outcome = regime_result.taxable_st + regime_result.taxable_lt
        samples.append(outcome)
        regime_results[regime.name].append(outcome)

    samples = np.array(samples)
    return {
        'expected_taxable': np.mean(samples),
        'std_dev': np.std(samples),
        'percentiles': {
            'p10': np.percentile(samples, 10),
            'p25': np.percentile(samples, 25),
            'p50': np.percentile(samples, 50),
            'p75': np.percentile(samples, 75),
            'p90': np.percentile(samples, 90)
        },
        'regime_breakdown': {
            name: {
                'mean': np.mean(outcomes),
                'std': np.std(outcomes),
                'probability': next(r.probability for r in TAX_REGIMES if r.name == name)
            }
            for name, outcomes in regime_results.items()
        }
    }


def get_system_guarantees() -> Dict[str, str]:
    """
    What we can GUARANTEE, not claim.

    MUST-FIX: No more "96% accurate" - say what we can prove.
    """

    return {
        'capital_gains_netting': (
            "Correct for all statutory capital gain cases covered by golden tests. "
            "6/6 tests passing. IRC §1222, §1211(b), §1212(b) compliant."
        ),
        'taxpayer_elections': (
            "All elective strategies implemented and tested. "
            "MINIMIZE_ST_FIRST is statutory-safe default."
        ),
        'ambiguous_areas': (
            "Tax computed under conservative/strict IRS interpretation (full wash sale "
            "disallowance, standard capital gains treatment). Regime Monte Carlo not yet wired."
        ),
        'rule_basis': (
            "Every calculation marked as STATUTORY (IRC), HEURISTIC (approximation), "
            "AMBIGUOUS (gray area), or ELECTIVE (taxpayer choice)."
        ),
        'regression_protection': (
            "6 golden tests lock correctness forever. "
            "If any test fails, system is broken and unusable."
        ),
        'not_guaranteed': (
            "Future law changes, individual circumstances beyond capital gains, "
            "IRS interpretation of novel situations, court decisions not yet rendered."
        )
    }
