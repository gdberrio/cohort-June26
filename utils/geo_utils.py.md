# utils/geo_utils.py

## Purpose

Geo-experiment utility functions for Week 4 of the bootcamp (experimentation and GeoLift). Provides Python-native helpers for loading, pivoting, and plotting geo-level time-series data, plus a simple difference-in-differences (DiD) estimator. Also includes an optional R bridge via `rpy2` for students who have the GeoLift R package installed.

## Dependencies

```
numpy, pandas, pathlib, matplotlib (lazy import in plot_geo_timeseries)
Optional: rpy2 (for R bridge functions)
```

## Module Sections

### 1. Pure-Python Geo Helpers (lines 12-136)

#### `load_geo_data(path) -> pd.DataFrame`

Loads a GeoLift-formatted CSV file and parses the `date` column to datetime.

- **path**: Path to CSV (e.g., `data/GeoLift_PreTest.csv`).
- **Returns**: DataFrame with columns `location`, `Y`, `date` (datetime64).

#### `pivot_geo_data(df, value_col="Y", location_col="location", date_col="date") -> pd.DataFrame`

Pivots long-format geo data into wide format — dates as rows, locations as columns. Useful for synthetic control and power analysis workflows.

- **Returns**: DataFrame with `date` index and one column per location.

#### `plot_geo_timeseries(df, locations=None, title="Geo Time Series", figsize=(14,6)) -> Figure`

Plots time series for selected geo locations from long-format data.

- **df**: Long-format DataFrame with `location`, `Y`, `date` columns.
- **locations**: List of location names to plot; `None` plots all.
- **Returns**: `matplotlib.Figure` with legend positioned outside plot area.

#### `compute_treatment_effect(pre_df, post_df, treatment_locations, control_locations=None, ...) -> dict`

Computes a simple difference-in-differences (DiD) treatment effect estimate.

| Parameter | Description |
|---|---|
| `pre_df` | Pre-test period geo data (long format) |
| `post_df` | Post-test period geo data (long format) |
| `treatment_locations` | `str` or `list[str]` — treatment location(s). A single string is auto-wrapped in a list. |
| `control_locations` | `list[str]` or `None` — if `None`, all non-treatment locations are used as controls |
| `value_col` | Outcome column name (default `"Y"`) |
| `location_col` | Location column name (default `"location"`) |
| `date_col` | Date column name (default `"date"`) |

**Returns** a dict with keys:

| Key | Description |
|---|---|
| `post_treatment_avg` | Mean outcome for treatment geos in post period |
| `post_control_avg` | Mean outcome for control geos in post period |
| `pre_treatment_avg` | Mean outcome for treatment geos in pre period |
| `pre_control_avg` | Mean outcome for control geos in pre period |
| `did_estimate` | DiD estimate: `(post_treat - pre_treat) - (post_ctrl - pre_ctrl)` |
| `lift_pct` | Percentage lift relative to counterfactual |

### 2. R Bridge — Optional (lines 139-173)

These functions require `rpy2` and the GeoLift R package to be installed. They are used in optional exercises and will fail gracefully if dependencies are missing.

#### `try_import_geolift() -> tuple(module | None, bool)`

Attempts to import the GeoLift R package via `rpy2.robjects.packages.importr`. Returns `(GeoLift_module, True)` on success or `(None, False)` on failure.

#### `r_to_pandas(r_df) -> pd.DataFrame`

Converts an R dataframe to a pandas DataFrame using the `rpy2.robjects.pandas2ri` converter.

#### `pandas_to_r(pd_df) -> R DataFrame`

Converts a pandas DataFrame to an R dataframe. Used to pass Python data to GeoLift R functions.

## Usage Example

```python
from utils.geo_utils import load_geo_data, pivot_geo_data, plot_geo_timeseries, compute_treatment_effect

# Load pre-test and test period data
pre = load_geo_data("data/GeoLift_PreTest.csv")
post = load_geo_data("data/GeoLift_Test.csv")

# Visualise
plot_geo_timeseries(pre, title="Pre-Test Period")

# Pivot for synthetic control analysis
wide = pivot_geo_data(pre)

# Compute DiD effect
result = compute_treatment_effect(pre, post, treatment_locations="chicago")
print(f"DiD estimate: {result['did_estimate']:.2f}")
print(f"Lift: {result['lift_pct']:.1f}%")
```
