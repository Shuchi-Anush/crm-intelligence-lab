"""Reusable CRM profitability, modeling, and recommendation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEGMENT_FEATURES = [
    "income",
    "tenure_months",
    "monthly_spend",
    "products_owned",
    "support_tickets",
    "email_engagement",
    "website_visits",
    "last_purchase_days",
]
CHURN_FEATURES = [
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
]
CAMPAIGN_FEATURES = CHURN_FEATURES


def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing CRM columns: {', '.join(missing)}")


def _model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )


def calculate_profitability(data: pd.DataFrame) -> pd.DataFrame:
    """Add customer profitability as revenue less service cost."""
    _require_columns(data, ["total_revenue", "service_cost"])
    result = data.copy()
    result["profitability"] = (
        pd.to_numeric(result["total_revenue"], errors="coerce").fillna(0)
        - pd.to_numeric(result["service_cost"], errors="coerce").fillna(0)
    ).round(2)
    return result


def segment_customers(data: pd.DataFrame, n_segments: int = 4) -> pd.DataFrame:
    """Add KMeans segment IDs and descriptions based on customer behavior."""
    _require_columns(data, SEGMENT_FEATURES)
    if data.empty:
        raise ValueError("Cannot segment an empty customer table")
    cluster_count = min(max(2, n_segments), len(data))
    features = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    transformed = features.fit_transform(data[SEGMENT_FEATURES])
    model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
    result = data.copy()
    result["segment_id"] = model.fit_predict(transformed)

    profiles = result.groupby("segment_id")[SEGMENT_FEATURES].mean()
    descriptions: dict[int, str] = {}
    spend_high = profiles["monthly_spend"].median()
    engagement_high = profiles["email_engagement"].median()
    recency_high = profiles["last_purchase_days"].median()
    products_low = profiles["products_owned"].median()
    for segment_id, profile in profiles.iterrows():
        if profile.monthly_spend >= spend_high and profile.email_engagement >= engagement_high:
            label = "High-value engaged"
        elif profile.monthly_spend >= spend_high and profile.support_tickets >= profiles.support_tickets.median():
            label = "High-value support intensive"
        elif profile.last_purchase_days >= recency_high and profile.email_engagement < engagement_high:
            label = "At-risk disengaged"
        elif profile.products_owned <= products_low:
            label = "Developing relationship"
        else:
            label = "Steady customers"
        descriptions[int(segment_id)] = label
    result["segment_description"] = result["segment_id"].map(descriptions)
    return result


def _score_classification(
    data: pd.DataFrame, features: list[str], target: str, output: str
) -> pd.DataFrame:
    _require_columns(data, features + [target])
    result = data.copy()
    y = pd.to_numeric(result[target], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        result[output] = float(y.mean())
        return result

    model = _model_pipeline()
    model.fit(result[features], y)
    probabilities = model.predict_proba(result[features])
    positive_index = list(model.named_steps["classifier"].classes_).index(1)
    result[output] = probabilities[:, positive_index].round(4)
    return result


def score_churn(data: pd.DataFrame) -> pd.DataFrame:
    """Add a model-based probability that each customer will churn."""
    return _score_classification(data, CHURN_FEATURES, "churned", "churn_probability")


def score_campaign_response(data: pd.DataFrame) -> pd.DataFrame:
    """Add a model-based probability that each customer responds to a campaign."""
    return _score_classification(data, CAMPAIGN_FEATURES, "campaign_response", "campaign_response_probability")


def _normalized_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    minimum, maximum = numeric.min(), numeric.max()
    if pd.isna(minimum) or maximum == minimum:
        return pd.Series(0.5, index=values.index)
    return ((numeric.fillna(numeric.median()) - minimum) / (maximum - minimum)).clip(0, 1)


def score_cross_sell(data: pd.DataFrame) -> pd.DataFrame:
    """Add an explainable cross-sell score from engagement and product gaps."""
    _require_columns(data, ["email_engagement", "website_visits", "products_owned", "previous_campaign_response"])
    result = data.copy()
    engagement = _normalized_series(result["email_engagement"])
    visits = _normalized_series(result["website_visits"])
    product_gap = 1 - _normalized_series(result["products_owned"])
    prior_response = pd.to_numeric(result["previous_campaign_response"], errors="coerce").fillna(0).clip(0, 1)
    score = (0.35 * engagement + 0.25 * visits + 0.25 * product_gap + 0.15 * prior_response).clip(0, 1)
    result["cross_sell_opportunity_score"] = score.round(4)
    result["cross_sell_explanation"] = np.select(
        [score >= 0.7, score >= 0.45],
        ["Strong engagement and product gap", "Moderate engagement or product gap"],
        default="Limited current cross-sell signals",
    )
    return result


def next_best_action(data: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent recommendation using model scores and CRM signals."""
    _require_columns(
        data,
        [
            "churn_probability",
            "campaign_response_probability",
            "cross_sell_opportunity_score",
            "last_purchase_days",
            "email_engagement",
        ],
    )
    result = data.copy()

    def recommendation(row: pd.Series) -> str:
        if row.churn_probability >= 0.65:
            return "Retention outreach: resolve risk before selling"
        if row.cross_sell_opportunity_score >= 0.65:
            return "Offer a relevant complementary product"
        if row.campaign_response_probability >= 0.6:
            return "Send a personalized campaign offer"
        if row.last_purchase_days >= 90 or row.email_engagement < 20:
            return "Re-engage with a timely reminder"
        return "Nurture relationship with useful content"

    result["next_best_action"] = result.apply(recommendation, axis=1)
    return result


def run_analytics(data: pd.DataFrame) -> pd.DataFrame:
    """Run the complete foundation pipeline in a predictable order."""
    result = calculate_profitability(data)
    result = segment_customers(result)
    result = score_churn(result)
    result = score_campaign_response(result)
    result = score_cross_sell(result)
    return next_best_action(result)