"""Deterministic synthetic CRM data generation for seminar demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CRM_COLUMNS = [
    "customer_id",
    "age",
    "income",
    "tenure_months",
    "monthly_spend",
    "products_owned",
    "support_tickets",
    "email_engagement",
    "website_visits",
    "last_purchase_days",
    "previous_campaign_response",
    "total_revenue",
    "service_cost",
    "churned",
    "cross_sell_response",
    "campaign_response",
]


def generate_crm_data(n_customers: int = 750, random_state: int = 42) -> pd.DataFrame:
    """Generate a reproducible customer table with correlated behavior."""
    if n_customers < 20:
        raise ValueError("n_customers must be at least 20 for useful model training")

    rng = np.random.default_rng(random_state)
    customer_id = [f"C{number:05d}" for number in range(1, n_customers + 1)]
    age = np.clip(rng.normal(42, 12, n_customers).round(), 18, 80).astype(int)
    income = np.clip(rng.lognormal(np.log(58_000), 0.42, n_customers), 22_000, 220_000)

    value_signal = (income / 58_000) + rng.normal(0, 0.35, n_customers)
    tenure_months = np.clip(12 + value_signal * 20 + rng.normal(0, 10, n_customers), 1, 120)
    products_owned = np.clip(
        1 + (value_signal / 1.7) + rng.normal(0, 0.9, n_customers), 1, 6
    ).round().astype(int)
    monthly_spend = np.clip(
        35 + income / 2_300 + products_owned * 18 + rng.normal(0, 28, n_customers),
        20,
        600,
    )

    engagement_signal = (
        0.45 * (products_owned / 4)
        + 0.35 * (monthly_spend / 250)
        + rng.normal(0, 0.18, n_customers)
    )
    email_engagement = np.clip(engagement_signal * 100, 2, 98)
    website_visits = np.clip(
        3 + engagement_signal * 22 + rng.normal(0, 5, n_customers), 0, 80
    ).round().astype(int)

    risk_signal = (
        0.9
        - 0.9 * engagement_signal
        + 0.25 * (support_tickets := np.clip(rng.poisson(1.3, n_customers), 0, 10))
        + rng.normal(0, 0.35, n_customers)
    )
    last_purchase_days = np.clip(25 + risk_signal * 45 + rng.normal(0, 18, n_customers), 1, 180)
    previous_campaign_response = rng.binomial(
        1, np.clip(0.18 + engagement_signal * 0.48 - risk_signal * 0.08, 0.03, 0.9)
    )

    total_revenue = np.maximum(monthly_spend * tenure_months * rng.normal(1.0, 0.08, n_customers), 0)
    service_cost = np.maximum(
        75 + support_tickets * 42 + monthly_spend * 0.08 + rng.normal(0, 14, n_customers),
        20,
    )

    churn_probability = 1 / (1 + np.exp(-(risk_signal - 0.75)))
    churned = rng.binomial(1, np.clip(churn_probability, 0.03, 0.85))
    campaign_probability = 1 / (
        1 + np.exp(-(1.15 * engagement_signal + 0.35 * previous_campaign_response - 0.8))
    )
    campaign_response = rng.binomial(1, np.clip(campaign_probability, 0.04, 0.9))
    cross_sell_probability = 1 / (
        1
        + np.exp(
            -(0.9 * engagement_signal + 0.35 * website_visits / 25 - 0.22 * products_owned - 0.15)
        )
    )
    cross_sell_response = rng.binomial(1, np.clip(cross_sell_probability, 0.03, 0.85))

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income": income.round(2),
            "tenure_months": tenure_months.round(1),
            "monthly_spend": monthly_spend.round(2),
            "products_owned": products_owned,
            "support_tickets": support_tickets,
            "email_engagement": email_engagement.round(1),
            "website_visits": website_visits,
            "last_purchase_days": last_purchase_days.round().astype(int),
            "previous_campaign_response": previous_campaign_response,
            "total_revenue": total_revenue.round(2),
            "service_cost": service_cost.round(2),
            "churned": churned,
            "cross_sell_response": cross_sell_response,
            "campaign_response": campaign_response,
        },
        columns=CRM_COLUMNS,
    )


def write_synthetic_dataset(
    output_path: str | Path = "data/customers.csv",
    n_customers: int = 750,
    random_state: int = 42,
) -> Path:
    """Generate and write the demonstration dataset, returning its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_crm_data(n_customers, random_state).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print(write_synthetic_dataset())