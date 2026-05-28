# utils/\_\_init\_\_.py

## Purpose

Package initializer for the `utils` module. Re-exports all public functions from `mmm_utils` and `eda_utils` so that notebooks can import them with a single statement:

```python
from utils import adstock_geometric, load_workshop_data
# or
from utils import *
```

## Exported API

### From `mmm_utils`

| Function | Description |
|---|---|
| `adstock_geometric` | Geometric (Koyck) carry-over adstock transformation |
| `adstock_weibull` | Weibull CDF/PDF adstock transformation |
| `saturation_hill` | Hill S-curve saturation |
| `saturation_power` | Power-law saturation (x^n) |
| `normalize` | Min-max normalization to [0, 1] |
| `geometric_hill_transform` | Grid-search: adstock + saturation over parameter ranges |
| `compute_contributions` | Beta * mean(X) contribution decomposition |
| `compute_decomp_rssd` | DECOMP.RSSD: spend-share vs effect-share alignment |
| `create_contribution_plot` | Horizontal bar chart of channel contributions |
| `build_ols_model` | Fit OLS regression via statsmodels |
| `model_diagnostics` | VIF, MAPE, NRMSE, AIC, Durbin-Watson diagnostics |

### From `eda_utils`

| Function | Description |
|---|---|
| `load_workshop_data` | Load the MMM workshop Excel/CSV dataset |
| `load_config` | Parse the config CSV into a variable-category dict |
| `ccf_plot` | Cross-correlation function plot (single pair) |
| `ccf_plot_all` | CCF grid for all variables vs KPI |
| `correlation_heatmap` | Lower-triangle Pearson correlation heatmap |
| `dual_axis_chart` | Dual y-axis time series chart (KPI vs spend) |
| `dual_axis_chart_all` | Grid of dual-axis charts |
| `summary_statistics` | Descriptive statistics for all numeric columns |

## Notes

- `geo_utils` is **not** re-exported here because it is used only in Week 4 notebooks and imported directly.
- The package has no install-time side effects; all imports are lightweight.
