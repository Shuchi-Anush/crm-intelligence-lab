"""Streamlit presentation layer for the CRM Intelligence Lab seminar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crm_intelligence_lab import generate_crm_data, run_analytics
from crm_intelligence_lab.analytics import CAMPAIGN_FEATURES, CHURN_FEATURES


RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "customers.csv"
PAGES = [
    "Executive Dashboard",
    "Customer 360",
    "Profitability",
    "Segmentation",
    "Churn & Scoring",
    "Next Best Action",
    "Closed-Loop CRM",
    "CRM Architecture",
]


st.set_page_config(
    page_title="CRM Intelligence Lab",
    page_icon="CRM",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #18323b; --muted: #5e7478; --teal: #0c7774; --coral: #db745c; --paper: #f6f8f3; }
    .stApp { background: var(--paper); color: var(--ink); }
    .main .block-container { color: var(--ink); }
    .stMarkdown, .stText, .stCaption, .stAlert, .stSelectbox, .stDataFrame { color: var(--ink); }
    .stMarkdown p, .stMarkdown li, [data-testid="stCaptionContainer"] p { color: var(--ink); }
    [data-testid="stSidebar"] { background: #18323b; }
    [data-testid="stSidebar"] * { color: #eef4ed; }
    [data-testid="stMetric"] { background: white; border: 1px solid #d9e3dc; border-radius: 8px; padding: 14px; }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #526a70 !important; font-size: 1rem !important; }
    [data-testid="stMetricValue"] { color: #18323b !important; font-size: 1.75rem !important; }
    [data-testid="stMetricDelta"] { color: #526a70 !important; }
    [data-testid="stDataFrame"] { color: #18323b; }
    [data-testid="stSelectbox"] label, [data-testid="stSelectbox"] p { color: #18323b !important; font-size: 1rem !important; }
    .hero { background: #18323b; color: #eef4ed; padding: 26px 30px; border-radius: 10px; margin-bottom: 22px; }
    .hero h1 { color: #f4c78d; margin: 0 0 4px 0; font-size: 2.1rem; }
    .hero p { color: #d8e5dc; margin: 0; font-size: 1rem; }
    .decision { background: #fff7e8; border-left: 6px solid #db745c; padding: 18px 22px; border-radius: 6px; margin: 10px 0 20px; }
    .decision h3 { color: #a64d3d; margin: 0 0 6px; }
    .why { background: white; border: 1px solid #d9e3dc; padding: 16px 20px; border-radius: 8px; }
    .section-note { color: var(--muted); font-size: .95rem; margin: -8px 0 18px; }
    h1, h2, h3 { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading customer data and computing CRM scores...")
def load_scored_customers() -> pd.DataFrame:
    """Load the committed dataset and run the existing analytics foundation."""
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        data = generate_crm_data(random_state=RANDOM_STATE)
    if data.empty:
        return data
    return run_analytics(data)


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def safe_mean(data: pd.DataFrame, column: str) -> float:
    if data.empty or column not in data:
        return 0.0
    return float(pd.to_numeric(data[column], errors="coerce").fillna(0).mean())


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        column.metric(label, value, delta)


def safe_bar_chart(data: pd.DataFrame, color: str, x_title: str) -> None:
    """Render a light, explicit bar chart without taking down the page."""
    try:
        if data.empty:
            st.info("No chart data is available.")
            return
        x_column, y_column = data.columns
        spec = {
            "mark": {"type": "bar", "color": color},
            "encoding": {
                "x": {"field": x_column, "type": "nominal", "axis": {"title": x_title, "labelColor": "#18323b", "titleColor": "#18323b", "labelAngle": 0}},
                "y": {"field": y_column, "type": "quantitative", "axis": {"title": "Customers", "labelColor": "#18323b", "titleColor": "#18323b", "gridColor": "#d9e3dc"}},
            },
            "config": {"background": "#ffffff", "view": {"stroke": "#d9e3dc"}},
        }
        st.vega_lite_chart(data.to_dict("records"), spec=spec, theme=None, use_container_width=True)
    except Exception as error:
        st.warning(f"This chart could not be rendered: {error}")


def safe_scatter_chart(data: pd.DataFrame) -> None:
    """Render the segmentation map with an explicit light chart theme."""
    try:
        if data.empty:
            st.info("No chart data is available.")
            return
        spec = {
            "mark": {"type": "circle", "size": 70, "opacity": 0.72},
            "encoding": {
                "x": {"field": "monthly_spend", "type": "quantitative", "axis": {"title": "Monthly spend", "labelColor": "#18323b", "titleColor": "#18323b", "gridColor": "#d9e3dc"}},
                "y": {"field": "email_engagement", "type": "quantitative", "axis": {"title": "Email engagement (%)", "labelColor": "#18323b", "titleColor": "#18323b", "gridColor": "#d9e3dc"}},
                "color": {"field": "segment_description", "type": "nominal", "legend": {"labelColor": "#18323b", "titleColor": "#18323b"}},
            },
            "config": {"background": "#ffffff", "view": {"stroke": "#d9e3dc"}},
        }
        st.vega_lite_chart(data.to_dict("records"), spec=spec, theme=None, use_container_width=True)
    except Exception as error:
        st.warning(f"This chart could not be rendered: {error}")


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<h1>{title}</h1><p class='section-note'>{subtitle}</p>", unsafe_allow_html=True)


def selected_customer(data: pd.DataFrame, key: str) -> pd.Series | None:
    if data.empty:
        st.warning("No customer records are available for this view.")
        return None
    customer_ids = data["customer_id"].astype(str).tolist()
    customer_id = st.selectbox("Select customer", customer_ids, key=key)
    return data.loc[data["customer_id"].astype(str) == customer_id].iloc[0]


def explain_decision(customer: pd.Series) -> str:
    reasons: list[str] = []
    if customer["churn_probability"] >= 0.65:
        reasons.append(
            f"churn risk is {pct(customer['churn_probability'])}, with {int(customer['last_purchase_days'])} days since the last purchase"
        )
    if customer["cross_sell_opportunity_score"] >= 0.65:
        reasons.append(
            f"cross-sell opportunity is {pct(customer['cross_sell_opportunity_score'])}, while the customer owns {int(customer['products_owned'])} products"
        )
    if customer["campaign_response_probability"] >= 0.6:
        reasons.append(
            f"campaign response probability is {pct(customer['campaign_response_probability'])}"
        )
    if not reasons:
        reasons.append(
            f"engagement is {customer['email_engagement']:.0f}% with {int(customer['website_visits'])} website visits and no dominant high-priority score"
        )
    return "The recommendation prioritizes " + "; ".join(reasons) + "."


def recommended_channel(customer: pd.Series) -> str:
    if customer["churn_probability"] >= 0.65:
        return "Phone or service outreach"
    if customer["campaign_response_probability"] >= 0.6:
        return "Email campaign"
    if customer["cross_sell_opportunity_score"] >= 0.65:
        return "Email plus in-product message"
    return "Educational email"


def show_customer_fields(customer: pd.Series) -> None:
    fields = {
        "Customer ID": customer["customer_id"],
        "Age": f"{customer['age']:.0f}",
        "Income": money(customer["income"]),
        "Tenure": f"{customer['tenure_months']:.0f} months",
        "Monthly Spend": money(customer["monthly_spend"]),
        "Products Owned": f"{customer['products_owned']:.0f}",
        "Support Tickets": f"{customer['support_tickets']:.0f}",
        "Email Engagement": pct(customer["email_engagement"] / 100),
        "Website Visits": f"{customer['website_visits']:.0f}",
        "Last Purchase": f"{customer['last_purchase_days']:.0f} days ago",
        "Total Revenue": money(customer["total_revenue"]),
        "Service Cost": money(customer["service_cost"]),
        "Profitability": money(customer["profitability"]),
        "Customer Segment": customer["segment_description"],
        "Churn Probability": pct(customer["churn_probability"]),
        "Campaign Response": pct(customer["campaign_response_probability"]),
        "Cross-Sell Opportunity": pct(customer["cross_sell_opportunity_score"]),
    }
    rows = list(fields.items())
    left, right = st.columns(2)
    for column, start in ((left, 0), (right, 9)):
        field_rows = rows[start : start + 9]
        column.table(
            pd.DataFrame(field_rows, columns=["Field", "Value"]),
        )


def render_executive_dashboard(data: pd.DataFrame) -> None:
    page_header("Executive Dashboard", "From customer data to a portfolio-level CRM view")
    if data.empty:
        st.warning("The customer dataset is empty.")
        return
    high_value = int((data["profitability"] >= data["profitability"].quantile(0.75)).sum())
    high_risk = int((data["churn_probability"] >= 0.65).sum())
    opportunities = int((data["cross_sell_opportunity_score"] >= 0.65).sum())
    metric_row(
        [
            ("Total Customers", f"{len(data):,}", None),
            ("Total Revenue", money(data["total_revenue"].sum()), None),
            ("Total Service Cost", money(data["service_cost"].sum()), None),
            ("Total Profitability", money(data["profitability"].sum()), None),
        ]
    )
    st.write("")
    metric_row(
        [
            ("High-Value Customers", f"{high_value:,}", "Top quartile profitability"),
            ("High-Risk Customers", f"{high_risk:,}", "Churn probability >= 65%"),
            ("Cross-Sell Opportunities", f"{opportunities:,}", "Opportunity score >= 65%"),
        ]
    )
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Customer segment distribution")
        segments = data["segment_description"].value_counts().rename("Customers").rename_axis("Segment").reset_index()
        safe_bar_chart(segments, "#0c7774", "Segment")
    with right:
        st.subheader("Profitability distribution")
        profitability_bins = [-np.inf, 1_200, 2_600, 4_000, 7_000, np.inf]
        profitability_labels = [
            "≤ ₹1,200",
            "₹1,200–₹2,600",
            "₹2,600–₹4,000",
            "₹4,000–₹7,000",
            "> ₹7,000",
        ]
        profitability_ranges = pd.cut(
            data["profitability"],
            bins=profitability_bins,
            labels=profitability_labels,
            ordered=True,
        )
        chart = (
            profitability_ranges.value_counts(sort=False)
            .rename("Customers")
            .rename_axis("Profitability range")
            .reset_index()
        )
        safe_bar_chart(chart, "#db745c", "Profitability range")
    st.subheader("Churn-risk distribution")
    risk_bins = pd.cut(data["churn_probability"], bins=[0, .2, .4, .6, .8, 1], include_lowest=True)
    risk_counts = risk_bins.value_counts().sort_index()
    risk_counts.index = risk_counts.index.astype(str)
    risk_chart = risk_counts.rename("Customers").rename_axis("Churn risk range").reset_index()
    safe_bar_chart(risk_chart, "#d89b4a", "Churn risk range")


def render_customer_360(data: pd.DataFrame) -> None:
    page_header("Customer 360", "One record, interpreted as a relationship rather than a row")
    customer = selected_customer(data, "customer_360")
    if customer is None:
        return
    show_customer_fields(customer)
    st.divider()
    st.markdown(
        f"<div class='decision'><h3>NEXT BEST ACTION</h3><strong>{customer['next_best_action']}</strong><br>Recommended channel: {recommended_channel(customer)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='why'><strong>WHY THIS DECISION?</strong><br>{explain_decision(customer)}</div>", unsafe_allow_html=True)


def render_profitability(data: pd.DataFrame) -> None:
    page_header("Profitability", "Separate what a customer spends from what the relationship contributes")
    st.info("Profitability = Revenue - Cost to Serve. Revenue is customer value generated; profitability subtracts the service cost required to support that relationship.")
    metric_row(
        [
            ("Total Revenue", money(data["total_revenue"].sum()), None),
            ("Total Service Cost", money(data["service_cost"].sum()), None),
            ("Total Profitability", money(data["profitability"].sum()), None),
            ("Average Profitability", money(safe_mean(data, "profitability")), None),
        ]
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Most profitable customers")
        st.dataframe(data.nlargest(10, "profitability")[ ["customer_id", "total_revenue", "service_cost", "profitability", "segment_description"] ], hide_index=True, use_container_width=True)
    with right:
        st.subheader("Least profitable customers")
        st.dataframe(data.nsmallest(10, "profitability")[ ["customer_id", "total_revenue", "service_cost", "profitability", "segment_description"] ], hide_index=True, use_container_width=True)
    st.subheader("Profitability by segment")
    by_segment = data.groupby("segment_description").agg(Customers=("customer_id", "count"), Revenue=("total_revenue", "sum"), Profitability=("profitability", "sum")).sort_values("Profitability", ascending=False)
    st.dataframe(by_segment.style.format({"Revenue": "${:,.0f}", "Profitability": "${:,.0f}"}), use_container_width=True)


def render_segmentation(data: pd.DataFrame) -> None:
    page_header("Segmentation", "KMeans groups customers with similar value, behavior, and engagement")
    st.write("KMeans compares standardized customer characteristics and assigns each customer to the nearest behavioral cluster. The segment labels are business descriptions added after clustering, so the model remains a discovery tool while CRM teams get language they can act on.")
    summary = data.groupby(["segment_id", "segment_description"]).agg(Customers=("customer_id", "count"), AvgSpend=("monthly_spend", "mean"), AvgEngagement=("email_engagement", "mean"), AvgRecency=("last_purchase_days", "mean"), AvgProfitability=("profitability", "mean")).reset_index()
    st.subheader("Segment sizes and profiles")
    st.dataframe(summary.style.format({"AvgSpend": "${:,.0f}", "AvgEngagement": "{:.1f}%", "AvgRecency": "{:.0f} days", "AvgProfitability": "${:,.0f}"}), hide_index=True, use_container_width=True)
    st.subheader("Behavior map")
    st.caption("Each point is a customer. The axes show monthly spend and email engagement; color identifies the discovered segment.")
    safe_scatter_chart(data[["monthly_spend", "email_engagement", "segment_description"]])
    with st.expander("Business interpretation"):
        for _, row in summary.iterrows():
            st.write(f"**{row['segment_description']}**: {int(row['Customers'])} customers, average spend {money(row['AvgSpend'])}, engagement {row['AvgEngagement']:.0f}%, and {row['AvgRecency']:.0f} days since purchase.")


def evaluate_models(data: pd.DataFrame, features: list[str], target: str) -> dict[str, float | str]:
    if len(data) < 30 or data[target].nunique() < 2:
        return {"status": "Evaluation unavailable: insufficient rows or only one target class"}
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target].astype(int), test_size=.25, random_state=RANDOM_STATE, stratify=data[target])
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
    try:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        return {"Accuracy": accuracy_score(y_test, predictions), "Precision": precision_score(y_test, predictions, zero_division=0), "Recall": recall_score(y_test, predictions, zero_division=0), "F1": f1_score(y_test, predictions, zero_division=0), "ROC-AUC": roc_auc_score(y_test, probabilities)}
    except (ValueError, RuntimeError) as error:
        return {"status": f"Evaluation unavailable: {error}"}


def render_churn_scoring(data: pd.DataFrame) -> None:
    page_header("Churn & Scoring", "Predictions become scores, then become CRM decisions")
    metric_row([("Average Churn Probability", pct(safe_mean(data, "churn_probability")), None), ("Average Campaign Response", pct(safe_mean(data, "campaign_response_probability")), None), ("Average Cross-Sell Opportunity", pct(safe_mean(data, "cross_sell_opportunity_score")), None)])
    left, right = st.columns(2)
    with left:
        st.subheader("Predictive model metrics: churn")
        st.dataframe(pd.DataFrame([evaluate_models(data, CHURN_FEATURES, "churned")]).T.rename(columns={0: "Value"}).style.format({"Value": "{:.3f}"}), use_container_width=True)
    with right:
        st.subheader("Predictive model metrics: campaign")
        st.dataframe(pd.DataFrame([evaluate_models(data, CAMPAIGN_FEATURES, "campaign_response")]).T.rename(columns={0: "Value"}).style.format({"Value": "{:.3f}"}), use_container_width=True)
    st.subheader("Prediction → Score → CRM Decision")
    st.markdown(
        "**Prediction:** the classification model estimates a probability from customer behavior.  \n"
        "**Score:** the probability makes customers comparable and supports prioritization.  \n"
        "**CRM Decision:** the score is combined with profitability, engagement, and recency to select a next-best action."
    )
    st.dataframe(data.nlargest(12, "churn_probability")[["customer_id", "churn_probability", "campaign_response_probability", "cross_sell_opportunity_score", "next_best_action"]], hide_index=True, use_container_width=True)


def render_next_best_action(data: pd.DataFrame) -> None:
    page_header("Next Best Action", "A decision console for turning customer scores into action")
    customer = selected_customer(data, "nba_customer")
    if customer is None:
        return
    metric_row(
        [
            ("Customer Value", money(customer["profitability"]), None),
            ("Segment", customer["segment_description"], None),
            ("Churn Risk", pct(customer["churn_probability"]), None),
        ]
    )
    st.write("")
    metric_row(
        [
            ("Campaign Response", pct(customer["campaign_response_probability"]), None),
            ("Cross-Sell Opportunity", pct(customer["cross_sell_opportunity_score"]), None),
        ]
    )
    st.markdown(f"<div class='decision'><h3>NEXT BEST ACTION</h3><strong>{customer['next_best_action']}</strong><br><strong>RECOMMENDED CHANNEL:</strong> {recommended_channel(customer)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='why'><strong>WHY THIS DECISION?</strong><br>{explain_decision(customer)}</div>", unsafe_allow_html=True)


def simulate_response(customer: pd.Series) -> dict[str, float | int | str]:
    probability = float(np.clip(.20 + .55 * customer["campaign_response_probability"] + .25 * customer["cross_sell_opportunity_score"] - .30 * customer["churn_probability"], .05, .95))
    stable_draw = ((int(str(customer["customer_id"])[1:]) * 37 + RANDOM_STATE) % 1000) / 1000
    responded = int(stable_draw < probability)
    uplift = float(customer["monthly_spend"] * (.35 + .25 * customer["cross_sell_opportunity_score"])) if responded else 0.0
    return {"probability": probability, "draw": stable_draw, "responded": responded, "uplift": uplift, "cost": 18.0}


def render_closed_loop(data: pd.DataFrame) -> None:
    page_header("Closed-Loop CRM", "A deterministic demonstration of action, response, outcome, and measurement")
    st.markdown(
        "<div class='why'><strong>DEMONSTRATION MODE</strong><br>"
        "These outcomes are deterministic teaching examples, not real business results.</div>",
        unsafe_allow_html=True,
    )
    customer = selected_customer(data, "closed_loop_customer")
    if customer is None:
        return
    result = simulate_response(customer)
    steps = [
        ("CUSTOMER DATA", f"Selected customer: {customer['customer_id']}"),
        ("PREDICTION", f"Campaign response probability: {pct(result['probability'])}"),
        ("CRM ACTION", customer["next_best_action"]),
        ("CUSTOMER RESPONSE", "Responded" if result["responded"] else "No response"),
        ("BUSINESS OUTCOME", f"Estimated revenue uplift: {money(result['uplift'])}"),
        ("MEASUREMENT", "Positive" if result["uplift"] > result["cost"] else "Needs review"),
    ]
    for label, value in steps:
        st.markdown(
            f"<div class='decision'><h3>{label}</h3><strong>{value}</strong></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='text-align:center;color:#db745c;font-size:1.3rem'>↓</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;color:#0c7774;font-size:1.1rem;font-weight:700'>"
        "↺ &nbsp; NEW DATA</div>",
        unsafe_allow_html=True,
    )
    st.info("The measured outcome becomes new customer evidence for future CRM decisions.")


def render_architecture() -> None:
    page_header("CRM Architecture", "The seminar's operating model from data to action and back again")
    stages = [
        "Customer Data",
        "Data Preparation",
        "Data Mining",
        "Customer Scoring",
        "CRM Decision",
        "Customer Action",
        "Measurement",
    ]
    for index, stage in enumerate(stages):
        st.markdown(f"<div style='text-align:center;font-size:1.2rem;font-weight:700;color:#18323b;padding:5px'>{stage}</div>", unsafe_allow_html=True)
        if index < len(stages) - 1:
            st.markdown("<div style='text-align:center;color:#db745c;font-size:1.3rem'>↓</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#0c7774;font-size:1.1rem;font-weight:700;padding-top:4px'>↺ Closed Loop</div>", unsafe_allow_html=True)
    st.divider()
    columns = st.columns(3)
    explanations = [
        ("Descriptive CRM", "What happened? Customer records, revenue, service cost, engagement, and segment profiles describe the relationship.", "Customer understanding"),
        ("Predictive CRM", "What is likely to happen? Churn and response models estimate probabilities for each customer.", "Customer scoring"),
        ("Prescriptive CRM", "What should we do next? The decision layer selects an explainable action and channel.", "Customer action"),
    ]
    for column, (title, body, label) in zip(columns, explanations):
        with column:
            st.subheader(title)
            st.write(body)
            st.caption(label)
    st.info("The closed loop matters: measure the response to an action, capture the outcome as new customer evidence, and improve the next CRM decision.")


def main() -> None:
    st.sidebar.markdown("# CRM Intelligence Lab")
    st.sidebar.caption("From Customer Data to Customer Action")
    page = st.sidebar.radio("Seminar sections", PAGES)
    st.sidebar.divider()
    st.sidebar.caption("Deterministic seminar environment")
    st.markdown("<div class='hero'><h1>CRM Intelligence Lab</h1><p>Customer data → understanding → scoring → decision → action → measurement</p></div>", unsafe_allow_html=True)
    try:
        data = load_scored_customers()
    except (ValueError, KeyError, FileNotFoundError) as error:
        st.error(f"The CRM analytics pipeline could not be loaded: {error}")
        return
    renderers = {
        "Executive Dashboard": render_executive_dashboard,
        "Customer 360": render_customer_360,
        "Profitability": render_profitability,
        "Segmentation": render_segmentation,
        "Churn & Scoring": render_churn_scoring,
        "Next Best Action": render_next_best_action,
        "Closed-Loop CRM": render_closed_loop,
        "CRM Architecture": lambda _: render_architecture(),
    }
    renderers[page](data)


if __name__ == "__main__":
    main()