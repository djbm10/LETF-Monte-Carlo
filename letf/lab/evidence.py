"""Result evidence is independent of source integrity and numerical precision."""
from enum import Enum


class ResultClass(str, Enum):
    OBSERVED_MARKET = 'OBSERVED_MARKET'
    RECONSTRUCTED_WITH_VALIDATION = 'RECONSTRUCTED_WITH_VALIDATION'
    HISTORICAL_COUNTERFACTUAL = 'HISTORICAL_COUNTERFACTUAL'
    CONDITIONAL_STRESS = 'CONDITIONAL_STRESS'
    SOFTWARE_FIXTURE = 'SOFTWARE_FIXTURE'
    BLOCKED = 'BLOCKED'


def evidence(classification, *, limitations=(), model_approved=False):
    classification = ResultClass(classification)
    if model_approved:
        raise ValueError('No investment forecasting model has approval evidence in this release')
    return {'classification': classification.value, 'model_approved': False,
            'investment_prediction_certified': False, 'limitations': list(limitations)}
