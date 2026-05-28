# utils/mmm_utils.py

## Purpose

Core Marketing Mix Modeling utility functions used throughout Weeks 2-4 of the bootcamp. Covers the full MMM pipeline from raw media spend to contribution analysis.

## Dependencies

```
numpy, pandas, matplotlib, statsmodels, scipy, re, math
```

## Module Sections

### 1. Adstock Transformations (lines 19-134)

Media advertising has carry-over effects -- a TV ad viewed today still influences purchase behavior next week. Adstock models this decay.

#### `adstock_geometric(x, theta) -> dict`

Applies the Koyck geometric decay: `x_decayed[t] = x[t] + theta * x_decayed[t-1]`.

- **x**: Raw media series (array or pd.Series).
- **theta**: Decay rate in [0, 1]. Higher = slower decay (TV ~0.5-0.8, digital ~0.0-0.3).
- **Returns**: Dict with `x`, `x_decayed`, `thetaVecCum`, `inflation_total`.

#### `normalize(x) -> np.ndarray`

Min-max scales an array to [0, 1]. Used internally by the Weibull adstock.

#### `adstock_weibull(x, shape, scale, windlen, adstock_type) -> dict`

Applies Weibull-distributed adstock (CDF or PDF variant). More flexible than geometric -- can model delayed peaks.

- **shape**: Weibull shape parameter.
- **scale**: Quantile-based scale parameter (0-1 range).
- **adstock_type**: `"pdf"` (flexible peak) or `"cdf"` (monotone decay).

### 2. Saturation Transformations (lines 137-178)

Beyond a certain spend level, additional dollars produce less incremental response (diminishing returns).

#### `saturation_hill(x, alpha, gamma) -> np.ndarray`

Hill function: `x^alpha / (x^alpha + inflexion^alpha)` where the inflection point is interpolated between min(x) and max(x) using gamma.

- **alpha**: Steepness (higher = sharper S-curve).
- **gamma**: Inflection position in [0, 1] (0 = near min, 1 = near max).

#### `saturation_power(x, n) -> np.ndarray`

Simple power law: `x^n`. For 0 < n < 1, produces concave diminishing returns.

### 3. Grid-Search Pipeline (lines 181-280)

#### `geometric_hill_transform(x, varname, theta_range, alpha_range, gamma_range) -> pd.DataFrame`

Exhaustive grid search over (theta, alpha, gamma) combinations. Applies `adstock_geometric` then `saturation_hill` for every parameter triple. Returns a DataFrame with one column per combination, named `{varname}_{theta}_{alpha}_{gamma}`.

#### `PARAM_RANGES` (dict)

Default parameter grids per channel type:

| Type | Theta Range | Use Case |
|---|---|---|
| `tv` | 0.3 - 0.8 | TV, brand awareness |
| `digital` | 0.0 - 0.3 | Paid search, programmatic |
| `traditional` | 0.1 - 0.4 | Radio, outdoor, print |
| `organic` | 0.1 - 0.4 | Organic/social |
| `atl` | 0.1 - 0.8 | ATL/competition spend |

#### `best_transformation(transformed_df, y, varname, ascending) -> tuple`

Selects the transformed column with the highest absolute Pearson correlation to the dependent variable. For competition variables, set `ascending=True` to pick the most negative correlation.

### 4. OLS Model Building (lines 283-341)

#### `build_ols_model(y, X, add_constant=True) -> statsmodels.OLSResults`

Fits OLS via `statsmodels.api.OLS`. Adds an intercept column by default.

#### `model_diagnostics(model, y) -> pd.DataFrame`

Comprehensive diagnostics table with per-variable rows containing:

| Column | Description |
|---|---|
| `Coefficient`, `StdError`, `t_value`, `p_value` | Standard regression output |
| `VIF` | Variance Inflation Factor (>10 = severe multicollinearity) |
| `CI_lower`, `CI_upper` | 95% confidence interval bounds |
| `R_squared`, `Adj_R_squared` | Model fit |
| `MAPE`, `NRMSE` | Forecast accuracy |
| `AIC` | Akaike Information Criterion |
| `Durbin_Watson` | Autocorrelation test (~2.0 = no autocorrelation) |

### 5. Contribution Analysis (lines 344-423)

#### `compute_contributions(model) -> pd.DataFrame`

Decomposes fitted values into channel contributions using the Beta * mean(X) approach. Output includes absolute and signed contribution percentages plus a `Variable_clean` column with numeric suffixes stripped.

#### `compute_decomp_rssd(media_spend_vars, contrib_df, data) -> float`

Computes DECOMP.RSSD: the root sum of squared differences between each channel's spend share and its effect share. Lower is better -- indicates the model's attribution aligns with actual spending patterns.

### 6. Plotting (lines 425-474)

#### `create_contribution_plot(contrib_df, col, title, figsize) -> matplotlib.Figure`

Produces a horizontal bar chart showing each variable's contribution (typically `Contribution_with_sign`). Auto-sizes based on the number of variables.

## Usage Example

```python
from utils.mmm_utils import *
from utils.eda_utils import load_workshop_data

data = load_workshop_data("data/MMM_Workshop_Data.xlsx")
y = data["Sales_Volume_Total"]

# Transform TV spend
transformed = geometric_hill_transform(
    data["TV_Spends"], "TV_Spends",
    PARAM_RANGES["tv"]["theta"],
    PARAM_RANGES["tv"]["alpha"],
    PARAM_RANGES["tv"]["gamma"],
)

# Find best transformation
best_col, corr, pval, _ = best_transformation(transformed, y, "TV_Spends")

# Build and diagnose OLS model
X = transformed[[best_col]]
model = build_ols_model(y, X)
diag = model_diagnostics(model, y)
contrib = compute_contributions(model)
fig = create_contribution_plot(contrib)
```
