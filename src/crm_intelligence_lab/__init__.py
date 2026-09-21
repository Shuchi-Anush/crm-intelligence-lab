"""CRM Intelligence Lab analytics foundation."""

from .analytics import (
    calculate_profitability,
    next_best_action,
    run_analytics,
    score_campaign_response,
    score_churn,
    score_cross_sell,
    segment_customers,
)
from .data_generation import generate_crm_data, write_synthetic_dataset

__all__ = [
    "calculate_profitability",
    "generate_crm_data",
    "next_best_action",
    "run_analytics",
    "score_campaign_response",
    "score_churn",
    "score_cross_sell",
    "segment_customers",
    "write_synthetic_dataset",
]


def main() -> None:
    """Generate the default demonstration dataset."""
    print(write_synthetic_dataset())
