# Session 03 Ols Mmm

# Session 3: Baby MMM from Scratch -- OLS Foundation

**Marketing Science Bootcamp -- Week 2, Live Session**

---

In this session we build a full Marketing Mix Model using **Ordinary Least Squares (OLS)**. We will:

1. Apply adstock + saturation transformations via grid search.
2. Select the best transformation per channel (highest correlation with KPI).
3. Fit an OLS model and inspect diagnostics.
4. Decompose sales into channel contributions.
5. Iterate on model specs and compare results.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import sys

# Add project root for utils
sys.path.insert(0, "../..")

from utils.mmm_utils import (
    adstock_geometric,
    saturation_hill,
    geometric_hill_transform,
    best_transformation,
    build_ols_model,
    model_diagnostics,
    compute_contributions,
    create_contribution_plot,
    compute_decomp_rssd,
    PARAM_RANGES,
)
from utils.eda_utils import load_workshop_data, load_config

# Plotting defaults
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True

print("Setup complete.")
```

```python
# Load data and configuration
data = load_workshop_data("../../data/MMM_Workshop_Data.xlsx")
config = load_config("../../data/config_file.csv")

# Extract key variable lists
dv = config["dependent_var"]
time_col = config["time_column"]
paid_media = config["paid_media_spends"]
competition = config["competition_spend_vars"]
untransformed = config["untransformed_vars"]
tv_vars = config["tv_vars"]
traditional_vars = config["traditional_vars"]
atl_vars = config["atl_vars"]
digital_vars = config["digital_vars"]

y = data[dv].values

print(f"Dependent variable: {dv}")
print(f"Time column: {time_col}")
print(f"Paid media variables ({len(paid_media)}): {paid_media}")
print(f"TV vars: {tv_vars}")
print(f"Traditional vars: {traditional_vars}")
print(f"ATL (competition) vars: {atl_vars}")
print(f"Digital vars: {digital_vars}")
print(f"Untransformed vars: {untransformed}")
print(f"\nData shape: {data.shape}")
```

---

## Step 1: Apply Transformations via Grid Search

Instead of hand-picking parameters (like we did in Notebook 2), we now **search** over a grid of `(theta, alpha, gamma)` values for each channel. For every combination, we:

1. Apply geometric adstock with that theta.
2. Apply Hill saturation with that alpha and gamma.
3. Store the resulting transformed series.

Later we will pick the combination that yields the highest correlation with the dependent variable.

```python
# Review the default parameter ranges
for channel_type, ranges in PARAM_RANGES.items():
    n_combos = len(ranges["theta"]) * len(ranges["alpha"]) * len(ranges["gamma"])
    print(f"{channel_type:>12}: theta {ranges['theta'][[0,-1]]}, "
          f"alpha {ranges['alpha'][[0,-1]]}, "
          f"gamma {ranges['gamma'][[0,-1]]} "
          f"-> {n_combos} combinations")
```

```python
# Apply grid search transformations for each variable category

# TV variables
transformed_tv = {}
for var in tv_vars:
    transformed_tv[var] = geometric_hill_transform(
        data[var], var,
        theta_range=PARAM_RANGES["tv"]["theta"],
        alpha_range=PARAM_RANGES["tv"]["alpha"],
        gamma_range=PARAM_RANGES["tv"]["gamma"],
    )
    print(f"{var}: {transformed_tv[var].shape[1] - 1} transformed columns")

# Digital variables
transformed_digital = {}
for var in digital_vars:
    transformed_digital[var] = geometric_hill_transform(
        data[var], var,
        theta_range=PARAM_RANGES["digital"]["theta"],
        alpha_range=PARAM_RANGES["digital"]["alpha"],
        gamma_range=PARAM_RANGES["digital"]["gamma"],
    )
    print(f"{var}: {transformed_digital[var].shape[1] - 1} transformed columns")

# Traditional variables
transformed_traditional = {}
for var in traditional_vars:
    transformed_traditional[var] = geometric_hill_transform(
        data[var], var,
        theta_range=PARAM_RANGES["traditional"]["theta"],
        alpha_range=PARAM_RANGES["traditional"]["alpha"],
        gamma_range=PARAM_RANGES["traditional"]["gamma"],
    )
    print(f"{var}: {transformed_traditional[var].shape[1] - 1} transformed columns")

# ATL (competition) variables
transformed_atl = {}
for var in atl_vars:
    transformed_atl[var] = geometric_hill_transform(
        data[var], var,
        theta_range=PARAM_RANGES["atl"]["theta"],
        alpha_range=PARAM_RANGES["atl"]["alpha"],
        gamma_range=PARAM_RANGES["atl"]["gamma"],
    )
    print(f"{var}: {transformed_atl[var].shape[1] - 1} transformed columns")

print("\nGrid search transformations complete.")
```

---

## Step 2: Select Best Transformation per Variable

For each variable, we pick the `(theta, alpha, gamma)` combination that produces the **highest Pearson correlation** with the dependent variable (`Sales_Volume_Total`).

For competition variables, we use `ascending=True` to select the most **negative** correlation (competitor spend should hurt our sales).

```python
# Combine all transformed DataFrames
all_transformed = {}
all_transformed.update(transformed_tv)
all_transformed.update(transformed_digital)
all_transformed.update(transformed_traditional)
all_transformed.update(transformed_atl)

# Find best transformation for each variable
best_results = []

for var, tf_df in all_transformed.items():
    # Competition vars: look for negative correlation
    is_competition = var in competition
    best_col, corr, pval, cor_df = best_transformation(
        tf_df, y, var, ascending=is_competition
    )
    best_results.append({
        "Variable": var,
        "Best_Column": best_col,
        "Correlation": corr,
        "P_Value": pval,
        "Is_Competition": is_competition,
    })

best_df = pd.DataFrame(best_results)
best_df = best_df.sort_values("Correlation", key=abs, ascending=False)
print("Best transformation per variable:")
best_df
```

---

## Step 3: Build OLS Model (Model 1)

We select a subset of variables for our first model specification. A good starting point:

- **Media channels:** TV_Spends, TV_GRP, a few digital channels
- **Controls:** Inflation_Rate, Average_Price_Total
- **Competition:** Brand_B_ATL_Spends

```python
# Model 1 specification: select variables to include
model1_vars = [
    "TV_Spends",
    "TV_GRP",
    "Paid_Search_Spends",
    "Programmatic_Display_Spends",
    "Meta_Spends_Agg",
    "Radio_Spends",
    "Brand_B_ATL_Spends",
]

# Build X matrix using best transformations
X_model1 = pd.DataFrame()

for var in model1_vars:
    best_row = best_df[best_df["Variable"] == var]
    if best_row.empty:
        print(f"WARNING: {var} not found in best_df, using raw values")
        X_model1[var] = data[var].values
    else:
        best_col = best_row["Best_Column"].values[0]
        X_model1[best_col] = all_transformed[var][best_col].values
        print(f"{var} -> {best_col} (r = {best_row['Correlation'].values[0]:.4f})")

# Add untransformed control variables
for var in untransformed:
    if var in data.columns:
        X_model1[var] = data[var].values
        print(f"{var} -> raw (untransformed control)")

print(f"\nX matrix shape: {X_model1.shape}")
X_model1.head()
```

```python
# Fit OLS model
model1 = build_ols_model(y, X_model1)
print(model1.summary())
```

---

## Step 4: Model Diagnostics

We evaluate model quality on several dimensions:

| Metric | Good Value | Meaning |
|--------|-----------|--------|
| **R-squared** | > 0.80 | Model explains most variance |
| **MAPE** | < 10% | Average prediction error |
| **NRMSE** | < 15% | Normalized root mean square error |
| **VIF** | < 10 (ideally < 5) | Low multicollinearity |
| **Durbin-Watson** | Close to 2.0 | No autocorrelation in residuals |
| **AIC** | Lower is better | Model parsimony |
| **Coefficient signs** | Match business logic | Positive for own media, negative for competition |

```python
# Compute and display diagnostics
diag1 = model_diagnostics(model1, y)
diag1
```

```python
# Interpret key diagnostics
print("=" * 60)
print("MODEL 1 DIAGNOSTICS SUMMARY")
print("=" * 60)
print(f"R-squared:        {diag1['R_squared'].iloc[0]:.4f}")
print(f"Adj R-squared:    {diag1['Adj_R_squared'].iloc[0]:.4f}")
print(f"MAPE:             {diag1['MAPE'].iloc[0]:.4%}")
print(f"NRMSE:            {diag1['NRMSE'].iloc[0]:.4%}")
print(f"AIC:              {diag1['AIC'].iloc[0]:.1f}")
print(f"Durbin-Watson:    {diag1['Durbin_Watson'].iloc[0]:.4f}")
print()
print("VIF by variable:")
for _, row in diag1.iterrows():
    flag = " ** HIGH **" if row["VIF"] > 10 else ""
    print(f"  {row['Variable']:<45} VIF = {row['VIF']:>8.2f}{flag}")
print()
print("Coefficient signs:")
for _, row in diag1.iterrows():
    sign = "+" if row["Coefficient"] > 0 else "-"
    print(f"  {row['Variable']:<45} {sign} {row['Coefficient']:>12.4f}  (p = {row['p_value']:.4f})")
```

### Interpreting the Diagnostics

Check the following:

- **VIF > 10**: Indicates severe multicollinearity. Consider removing one of the collinear variables (e.g., TV_Spends vs TV_GRP may be highly correlated).
- **Durbin-Watson far from 2.0**: Suggests autocorrelation -- you may need to add a trend or seasonality variable.
- **Negative coefficients on own media**: This is a red flag. A channel should not have a negative effect on your own sales (unless it is a competition variable).
- **MAPE > 15%**: The model may not be capturing enough of the signal.

---

## Step 5: Contribution Analysis

We decompose the predicted sales into each variable's contribution using the **Beta x Mean** approach:

$$
\text{Contribution}_j = \beta_j \times \bar{X}_j
$$

```python
# Compute contributions
contrib1 = compute_contributions(model1)
contrib1.sort_values("Contribution_pct", ascending=False)
```

```python
# Contribution bar chart
fig = create_contribution_plot(contrib1, title="Model 1: Channel Contribution Chart")
plt.show()
```

```python
# DECOMP.RSSD: How well do spend shares align with effect shares?
# Lower is better -- means the model's attribution matches actual spending patterns
media_spend_vars_model1 = [v for v in model1_vars if v in paid_media]
rssd1 = compute_decomp_rssd(media_spend_vars_model1, contrib1, data)
print(f"Model 1 DECOMP.RSSD: {rssd1:.4f}")
print("(Lower = better alignment between spend share and effect share)")
```

---

## Step 6: Iterate -- Model 2

Good modeling is **iterative**. Based on Model 1 diagnostics, we may want to:

- Remove variables with wrong signs.
- Remove variables with very high VIF (multicollinearity).
- Add or swap channels.
- Add additional control variables.

Let's build a second model spec.

```python
# Model 2 specification: adjust based on Model 1 diagnostics
# Example: drop TV_GRP (collinear with TV_Spends), add Youtube_Spends
model2_vars = [
    "TV_Spends",
    "Paid_Search_Spends",
    "Programmatic_Display_Spends",
    "Meta_Spends_Agg",
    "Youtube_Spends",
    "Outdoor_Spends",
    "Brand_B_ATL_Spends",
]

# Build X matrix for Model 2
X_model2 = pd.DataFrame()

for var in model2_vars:
    best_row = best_df[best_df["Variable"] == var]
    if best_row.empty:
        print(f"WARNING: {var} not found in best_df, using raw values")
        X_model2[var] = data[var].values
    else:
        best_col = best_row["Best_Column"].values[0]
        X_model2[best_col] = all_transformed[var][best_col].values
        print(f"{var} -> {best_col} (r = {best_row['Correlation'].values[0]:.4f})")

# Add untransformed control variables
for var in untransformed:
    if var in data.columns:
        X_model2[var] = data[var].values
        print(f"{var} -> raw (untransformed control)")

print(f"\nX matrix shape: {X_model2.shape}")
```

```python
# Fit Model 2
model2 = build_ols_model(y, X_model2)
print(model2.summary())
```

```python
# Model 2 diagnostics
diag2 = model_diagnostics(model2, y)

# Side-by-side comparison
comparison = pd.DataFrame({
    "Metric": ["R_squared", "Adj_R_squared", "MAPE", "NRMSE", "AIC", "Durbin_Watson"],
    "Model_1": [
        diag1["R_squared"].iloc[0],
        diag1["Adj_R_squared"].iloc[0],
        diag1["MAPE"].iloc[0],
        diag1["NRMSE"].iloc[0],
        diag1["AIC"].iloc[0],
        diag1["Durbin_Watson"].iloc[0],
    ],
    "Model_2": [
        diag2["R_squared"].iloc[0],
        diag2["Adj_R_squared"].iloc[0],
        diag2["MAPE"].iloc[0],
        diag2["NRMSE"].iloc[0],
        diag2["AIC"].iloc[0],
        diag2["Durbin_Watson"].iloc[0],
    ],
})

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison.to_string(index=False))
```

```python
# Model 2 contributions
contrib2 = compute_contributions(model2)
fig = create_contribution_plot(contrib2, title="Model 2: Channel Contribution Chart")
plt.show()
```

---

## Exercise: Build Your Own Model Spec

Now it is your turn. Using the tools above, build **at least one more model** (Model 3).

Consider:
- Which variables had wrong signs? Remove them.
- Which variables had high VIF? Remove one from each correlated pair.
- Are there channels missing that should be included?
- Does adding impression-based variables (instead of spend) improve the model?

```python
# TODO: Define your Model 3 variable list
model3_vars = [
    # Add your chosen variables here
]

# TODO: Build X matrix, fit model, run diagnostics, compute contributions
# Follow the same pattern as Model 1 and Model 2 above.
```

---

## Deliverable

By the end of this session, you should have:

1. A **working OLS MMM** using the workshop data.
2. **Contribution charts** for at least 2 model specifications.
3. A **comparison table** of diagnostics across your models.
4. An understanding of which variables drive sales and how transformation parameters affect results.

Save your notebook -- we will build on these results in **Session 4** when we move to Bayesian estimation.

