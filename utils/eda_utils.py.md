# utils/eda_utils.py

## Purpose

Exploratory Data Analysis utility functions for the Marketing Mix Modeling bootcamp. Provides data loading, summary statistics, cross-correlation analysis, correlation heatmaps, and dual-axis time-series charts. Used primarily in Week 1 (EDA sessions) and referenced throughout later weeks.

Refactored from three standalone scripts (`ccf_plot.py`, `correlation_heatmap_code.py`, `Dualaxis_line_chart_updated_axis.py`) into a single, notebook-ready module.

## Dependencies

```
numpy, pandas, matplotlib, seaborn, statsmodels (tsa.stattools.ccf), pathlib
```

## Module Sections

### 1. Data Loading Helpers (lines 17-88)

#### `load_workshop_data(path, sheet_name="Data") -> pd.DataFrame`

Loads the MMM workshop dataset from Excel or CSV. Auto-detects file type by extension.

- **path**: Path to `.xlsx`, `.xls`, or `.csv` file.
- **sheet_name**: Excel sheet to read (ignored for CSV).
- **Returns**: DataFrame with all non-date columns coerced to numeric and NaN filled with 0.

#### `load_config(path) -> dict`

Parses the MMM config CSV file (`data/config_file.csv`) which categorises variables by channel type.

- **path**: Path to the config CSV.
- **Returns**: Dict with keys:

| Key | Description |
|---|---|
| `dependent_var` | KPI column name (e.g., `"Sales_Volume_Total"`) |
| `time_column` | Time column name (e.g., `"Month"`) |
| `paid_media_spends` | List of all paid media spend column names |
| `competition_spend_vars` | Competitor spend columns |
| `untransformed_vars` | Variables to include without adstock/saturation |
| `tv_vars` | TV-specific spend columns |
| `traditional_vars` | Traditional media columns (radio, outdoor, print) |
| `atl_vars` | ATL / competition columns |
| `digital_vars` | Derived: `paid_media_spends` minus TV, traditional, ATL |

### 2. Summary Statistics (lines 92-114)

#### `summary_statistics(df, time_col="Month") -> pd.DataFrame`

Produces `describe().T` for all numeric columns except the time column. Useful as a quick data quality check in notebooks.

- **df**: Input DataFrame.
- **time_col**: Column to exclude from numeric summary.
- **Returns**: Transposed descriptive statistics (count, mean, std, min, quartiles, max), rounded to 2 decimal places.

### 3. CCF (Cross-Correlation Function) Plots (lines 118-245)

#### `ccf_plot(y, x, y_name="KPI", x_name="Variable", max_lag=4, ax=None) -> Figure | None`

Plots the cross-correlation function between a KPI series and a media variable at lags from `-max_lag` to `+max_lag`. Includes 95% confidence interval bounds.

Supports two calling conventions:
- **Array mode**: `ccf_plot(y_array, x_array, y_name="Sales", x_name="TV")`
- **DataFrame convenience mode**: `ccf_plot(df, kpi_col, var_col)` — extracts columns automatically.

| Parameter | Description |
|---|---|
| `y` | Reference series (array) or DataFrame (convenience mode) |
| `x` | Comparison series (array) or KPI column name (convenience mode) |
| `y_name` | Label for y, or variable column name in convenience mode |
| `x_name` | Label for x (unused in convenience mode) |
| `max_lag` | Maximum lag in each direction (default 4) |
| `ax` | Pass an existing Axes to embed in a subplot grid |

Returns a `matplotlib.Figure` if it creates one, `None` if an existing `ax` was provided.

#### `ccf_plot_all(df, kpi_col="Sales_Volume_Total", time_col="Month", max_lag=4, cols_per_row=3, figsize_per_plot=(4,3)) -> Figure`

Generates a grid of CCF plots for every numeric column in the DataFrame against the KPI. Automatically determines grid dimensions and hides unused subplot cells.

- **kpi_col**: Defaults to `"Sales_Volume_Total"` so `ccf_plot_all(data)` works with no positional args.
- **time_col**: Column excluded from variable list.
- **cols_per_row**: Number of subplots per row (default 3).

### 4. Correlation Heatmap (lines 248-292)

#### `correlation_heatmap(df, cols=None, figsize=(14,10), annot=True, cmap="coolwarm_r") -> Figure`

Creates a lower-triangle Seaborn heatmap of pairwise Pearson correlations.

- **cols**: Subset of columns; defaults to all numeric columns.
- **annot**: Show correlation values in cells.
- **cmap**: Colormap (default `"coolwarm_r"` — red = negative, blue = positive).
- **Returns**: `matplotlib.Figure`.

### 5. Dual-Axis Line Charts (lines 296-413)

#### `dual_axis_chart(df, kpi_col, var_col=None, time_col="Month", ax=None, spend_col=None) -> Figure | None`

Creates a dual y-axis time-series chart: KPI on left axis (green), comparison variable on right axis (blue). Both axes are formatted with comma-separated integers.

- **var_col**: Column for the right axis.
- **spend_col**: Alias for `var_col` (notebook convenience — some cells use `spend_col=`).
- Raises `ValueError` if neither `var_col` nor `spend_col` is provided.

#### `dual_axis_chart_all(df, kpi_col, time_col="Month", cols_per_row=2, figsize_per_plot=(6,4), spend_cols=None) -> Figure`

Generates a grid of dual-axis charts for multiple variables.

- **spend_cols**: Explicit list of variables to plot. If `None`, plots all columns except `kpi_col` and `time_col`.
- Automatically hides unused subplot cells.

## Usage Example

```python
from utils.eda_utils import (
    load_workshop_data, load_config, summary_statistics,
    ccf_plot, ccf_plot_all, correlation_heatmap,
    dual_axis_chart, dual_axis_chart_all,
)

# Load data and config
data = load_workshop_data("data/MMM_Workshop_Data.xlsx")
config = load_config("data/config_file.csv")

# Quick summary
summary_statistics(data)

# CCF for a single variable
ccf_plot(data, config["dependent_var"], "TV_Spends")

# All CCFs at once
ccf_plot_all(data, kpi_col=config["dependent_var"])

# Correlation heatmap
correlation_heatmap(data, cols=config["paid_media_spends"])

# Dual-axis chart
dual_axis_chart(data, config["dependent_var"], spend_col="TV_Spends")

# All dual-axis charts for paid media
dual_axis_chart_all(data, config["dependent_var"], spend_cols=config["paid_media_spends"])
```
