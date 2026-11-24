"""Common fail-closed errors and result labels for audited financial APIs."""
from enum import Enum


class UnsupportedModelError(ValueError):
    """The inputs/mechanics cannot support the requested financial claim."""


class Evidence(str, Enum):
    OBSERVED = 'OBSERVED'
    VALIDATED_RECONSTRUCTION = 'VALIDATED_RECONSTRUCTION'
    HYPOTHETICAL = 'HYPOTHETICAL_STRESS'
    UNSUPPORTED = 'BLOCKED'


def require_supported(condition, reason):
    if not condition:
        raise UnsupportedModelError(reason)
