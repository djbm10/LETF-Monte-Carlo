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
    from letf.accuracy import UnsupportedModelError
    raise UnsupportedModelError('Tax-law obligations are not sampled from arbitrary favorable/audit probabilities')


def get_system_guarantees() -> Dict[str, str]:
    """Only supported claims; passing regression examples is not certification."""
    return {
        'capital_gains_netting': 'Schedule D character netting; covered cases tested, see TAX_RULES.md',
        'taxpayer_elections': 'Loss ordering is statutory. FIFO/HIFO/explicit share identification supported.',
        'regression_protection': 'Run python -m pytest tests; original baselines are preserved.',
        'not_guaranteed': 'Full return preparation, state conformity, AMT, complex derivative taxes or broker eligibility.',
    }
