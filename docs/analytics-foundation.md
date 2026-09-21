# CRM Analytics Foundation

The foundation is deliberately model-at-runtime so the seminar can show the data-to-action flow without model artifacts or external services.

## Generate data

```powershell
uv run python -m crm_intelligence_lab.data_generation
```

This writes the deterministic 750-customer table to `data/customers.csv`.

## Run analytics

```python
import pandas as pd
from crm_intelligence_lab import run_analytics

customers = pd.read_csv("data/customers.csv")
scored_customers = run_analytics(customers)
```

The returned table includes profitability, KMeans segment information, churn and campaign probabilities, an explainable cross-sell score, and a next-best-action recommendation. Numeric model features are median-imputed; classification gracefully falls back to the observed class rate when a training target contains only one class.
