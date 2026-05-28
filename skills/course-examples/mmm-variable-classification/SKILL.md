---
name: mmm-variable-classification
description: >
  How to correctly classify variables for Marketing Mix Models (MMM) in this
  course project. Use this skill when selecting columns for an MMM model,
  deciding what goes into media vs controls vs excluded, working with
  config_file.csv or data_dict.csv, debugging unexpected variables in model
  results, or when the user mentions variable selection, column classification,
  or feature selection for any MMM notebook — whether OLS-based (week-2) or
  Bayesian/Meridian (week-3/4). Also trigger when you see catch-all column
  filters like "'Spend' in col" or "df.columns if col not in exclude_cols".
---

# MMM Variable Classification

This skill helps correctly classify columns from the workshop dataset into the right model roles. Misclassification is the most common and consequential bug in MMM notebooks — it silently produces wrong results without raising errors.

## The Workshop Dataset

The data lives in `data/MMM_Workshop_Data.xlsx` (sheet "Data"). The data dictionary at `data/data_dict.csv` describes every column. **Always consult the data dictionary** before classifying columns — column names alone can be misleading.

The dataset has 54 columns across these categories:

### 1. KPI / Dependent Variable
| Column | Description |
|--------|-------------|
| `Sales_Revenue_Total` | Total Sales Revenue in USD (typical KPI for revenue models) |
| `Sales_Volume_Total` | Total units sold (alternative KPI for volume models) |

### 2. Brand M's Own Media Spend
These are the advertising dollars Brand M chose to spend. They are the "treatment" in an MMM.

| Column | Description |
|--------|-------------|
| `TV_Spends` | TV advertising spend |
| `Radio_Spends` | Radio advertising spend |
| `Outdoor_Spends` | Outdoor advertising spend |
| `Paid_Search_Spends` | Paid search advertising spend |
| `Programmatic_Display_Spends` | Programmatic display spend |
| `Google_Display_Spend` | Google display spend |
| `Direct_Display_Spend` | Direct display spend |
| `Meta1_Spends` | META platform 1 spend |
| `Meta2_Spends` | META platform 2 spend |
| `Programmatic_Video_Spends` | Programmatic video spend |
| `Youtube_Spends` | YouTube advertising spend |
| `Influencer_Marketing_Spends` | Influencer marketing spend |
| `Meta_Spends_Agg` | **Aggregate** of Meta1 + Meta2 (use one OR the other, never both) |

### 3. Brand M's Media Volume/Impressions
These measure the *reach* of Brand M's advertising, not the cost. They pair with spend columns.

| Column | Pairs with |
|--------|-----------|
| `TV_GRP` | `TV_Spends` — Gross Rating Points |
| `Paid_Search_Impressions` | `Paid_Search_Spends` |
| `Programmatic_Display_Impressions` | `Programmatic_Display_Spends` |
| `Google_Display_Impressions` | `Google_Display_Spend` |
| `Direct_Display_Impressions` | `Direct_Display_Spend` |
| `Online_Video_Impressions` | `Programmatic_Video_Spends` |
| `YouTube_Views` | `Youtube_Spends` |
| `Meta_Agg_Impressions` | `Meta_Spends_Agg` (aggregate level) |
| `Paid_Search_Clicks` | `Paid_Search_Spends` (click metric) |
| `META_Clicks` | Meta spend (click metric) |
| `Online_Video_Views` | Video spend (view metric) |

### 4. Control Variables (External / Non-actionable)
These affect the KPI but Brand M cannot change them through marketing.

| Column | Description |
|--------|-------------|
| `Inflation_Rate` | Macroeconomic factor |
| `Average_Price_Total` | Brand M's average product price |
| `Brand_PH_Market_Share` | Competitor PH market share |
| `Brand_B_Market_Share` | Competitor B market share |
| `Brand_P_Market_Share` | Competitor P market share |
| `Brand_PH_Avg_Price` | Competitor PH average price (**100% NaN — always exclude**) |
| `Brand_B_Avg_Price` | Competitor B average price (has NaN gaps) |
| `Brand_P_Avg_Price` | Competitor P average price (has NaN gaps) |
| `Brand_B_ATL_Spends` | Competitor B above-the-line spend |
| `Brand_PH_ATL_Spends` | Competitor PH above-the-line spend |
| `Brand_P_ATL_Spends` | Competitor P above-the-line spend |

### 5. Excluded (Do Not Use as Predictors)
| Column | Why excluded |
|--------|-------------|
| `Month` | Date column, not a feature |
| `Market_Share_Brand_M_Total` | Outcome metric, not a driver |
| `Sales_Revenue_Category1/2/3` | Decompositions of the KPI |
| `Sales_Volume_*_Category1/2/3` | Decompositions of the KPI |
| `Average_Price_Category1/2/3` | Sub-category detail, not a driver |
| `Sales_Revenue_Channel1/2` | Decompositions of the KPI |
| `Sales_Volume_*_Channel1/2` | Decompositions of the KPI |
| `Average_Price_Channel1/2` | Sub-channel detail, not a driver |

## Anti-Patterns to Avoid

### 1. Catch-all column filters
```python
import pandas as pd

df = pd.DataFrame(
    columns=[
        "TV_Spends",
        "Brand_B_ATL_Spends",
        "Paid_Search_Impressions",
        "Sales_Volume_Total",
    ]
)
exclude_cols = ["Sales_Volume_Total"]

# BAD — catches competitor spends as media
media_cols = [col for col in df.columns if 'Spend' in col]

# BAD — dumps everything else as controls (including our own impressions)
control_cols = [col for col in df.columns if col not in exclude_cols]
```
Always use **explicit lists**. The dataset has enough columns with overlapping naming patterns that substring matching will misclassify something.

### 2. Competitor spend as media channels
`Brand_B_ATL_Spends`, `Brand_PH_ATL_Spends`, `Brand_P_ATL_Spends` have "Spends" in the name but are **competitor** advertising. They belong in controls, not media. The data dictionary makes this clear: "Above The Line Marketing Spends of **Competitor** Brand B."

### 3. Own impressions/clicks as controls
`Paid_Search_Impressions`, `META_Clicks`, `YouTube_Views` etc. are Brand M's own media metrics. Putting them as controls creates a model where Brand M's own advertising volume is treated as an external confounder — this absorbs the media effect and biases all channel ROIs downward.

### 4. Aggregates alongside components
`Meta_Spends_Agg` = `Meta1_Spends` + `Meta2_Spends` exactly. Including all three causes perfect multicollinearity (VIF > 1000). Use either the aggregate OR the components, never both.

### 5. Forgetting to check for NaN
Several competitor columns have NaN in the first 12 months:
- `Brand_PH_Avg_Price`: **100% NaN** — must be dropped entirely
- `Brand_B_Avg_Price`, `Brand_P_Avg_Price`: 12/36 NaN
- `Brand_B_Market_Share`, `Brand_PH_Market_Share`, `Brand_P_Market_Share`: 12/36 NaN

For partial NaN, forward-fill then back-fill (`df[cols].ffill().bfill()`) is appropriate since these are slow-moving variables.

## Framework-Specific Notes

### OLS models (week-2 notebooks)
- Use `config_file.csv` to drive variable selection via `load_config()`
- `paid_media_spends` in the config includes both spend AND volume columns (TV_GRP is listed there)
- Adstock/saturation transforms are applied to `paid_media_spends`
- `untransformed_vars` (Inflation_Rate, Average_Price_Total) skip transforms

### Meridian (week-3/4 notebooks)
- See the `meridian-model` skill for Meridian-specific InputData construction
- Spend and volume are separate arrays that must be paired 1:1
- Controls go in a dedicated `controls` array (not mixed with media)
- Meridian runs EDA checks before fitting — NaN in any array will block `sample_posterior`
