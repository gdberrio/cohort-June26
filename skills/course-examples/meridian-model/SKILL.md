---
name: meridian-model
description: >
  How to prepare data, build InputData, and visualize results for Google Meridian
  MMM models — both single-geo (national) and multi-geo setups. Use this skill
  whenever the user is working with Meridian, building a Meridian InputData,
  preparing media/control variables for Meridian, debugging Meridian EDA errors
  (multicollinearity, NaN, dimension mismatches), fitting a Meridian model via
  sample_posterior, or plotting results (response curves, contributions, ROI).
  Also use when the user mentions "media mix model" alongside "Meridian" or
  "Google", asks about media_spend vs media arrays, control variables in Meridian,
  Meridian dimension names, or response_curves() / visualizer API issues.
---

# Google Meridian: Data Preparation, InputData & Visualization

This skill encodes hard-won knowledge about correctly preparing data for Google Meridian, constructing InputData for both single-geo and multi-geo models, and using the built-in visualization API. Many of the pitfalls here produce confusing errors (like `MissingDataError: exog contains inf or nans` during a VIF check) that don't obviously point back to the real cause.

## Variable Classification

The single most important step is correctly classifying columns. The workshop dataset (`data/MMM_Workshop_Data.xlsx`) contains columns that look similar but belong in very different places. Consult `data/data_dict.csv` for column descriptions when in doubt.

### Media spend (`media_spend`)

Only include **Brand M's own advertising expenditure** — the money the brand chose to spend on its own channels.

```python
media_cols = [
    'TV_Spends',
    'Radio_Spends',
    'Outdoor_Spends',
    'Paid_Search_Spends',
    'Programmatic_Display_Spends',
    'Google_Display_Spend',
    'Direct_Display_Spend',
    'Meta1_Spends',
    'Programmatic_Video_Spends',
    'Youtube_Spends',
    'Meta2_Spends',
    'Influencer_Marketing_Spends',
]
```

**Common mistake — catching competitor spend:** A naive `'Spend' in col` filter will also grab `Brand_B_ATL_Spends`, `Brand_PH_ATL_Spends`, and `Brand_P_ATL_Spends`. These are *competitor* advertising spends and belong in controls, not media. Always use explicit lists rather than substring matching.

**Drop aggregates that cause collinearity:** `Meta_Spends_Agg` = `Meta1_Spends` + `Meta2_Spends`. Including both the parts and the sum creates perfect collinearity (VIF > 1000) and Meridian's EDA guardrail will block fitting.

### Media volume (`media`)

Meridian requires a paired volume/impressions array with the **same channels in the same order** as `media_spend`. Where a direct volume metric exists, use it; otherwise use spend as a proxy.

```python
media_volume_cols = [
    'TV_GRP',                           # GRPs — the standard TV volume metric
    'Radio_Spends',                     # no volume metric → spend proxy
    'Outdoor_Spends',                   # no volume metric → spend proxy
    'Paid_Search_Impressions',          # impressions
    'Programmatic_Display_Impressions', # impressions
    'Google_Display_Impressions',       # impressions
    'Direct_Display_Impressions',       # impressions
    'Meta1_Spends',                     # no channel-level impressions → spend proxy
    'Online_Video_Impressions',         # impressions (covers programmatic video)
    'YouTube_Views',                    # views
    'Meta2_Spends',                     # no channel-level impressions → spend proxy
    'Influencer_Marketing_Spends',      # no volume metric → spend proxy
]
```

**Common mistake — putting impressions/clicks as controls:** Columns like `Paid_Search_Impressions`, `META_Clicks`, `YouTube_Views` are volume metrics for Brand M's own channels. They are NOT external factors — they should either pair with their spend channel in the `media` array, or be excluded entirely. Never put them in controls.

### Controls (`controls`)

Controls are **non-actionable external factors** — things that affect the KPI but that Brand M cannot change through marketing decisions. They adjust for confounders.

```python
control_cols = [
    'Inflation_Rate',
    'Average_Price_Total',
    'Brand_PH_Market_Share',
    'Brand_B_Market_Share',
    'Brand_P_Market_Share',
    'Brand_B_Avg_Price',
    'Brand_P_Avg_Price',
    'Brand_B_ATL_Spends',
    'Brand_PH_ATL_Spends',
    'Brand_P_ATL_Spends',
]
```

Note: `Brand_PH_Avg_Price` is excluded because it is 100% NaN in the dataset.

### What to exclude entirely

These columns should not appear in media, controls, or KPI:
- **Sales breakdowns** (Sales_Revenue_Category1, Sales_Volume_Channel2, etc.) — these are decompositions of the dependent variable, not predictors
- **Market_Share_Brand_M_Total** — this is an outcome, not a driver
- **Aggregate impression columns** (`Meta_Agg_Impressions`, `Online_Video_Views`) when individual-channel metrics are already used
- **Click columns** (`Paid_Search_Clicks`, `META_Clicks`) — redundant with impressions

## Handling NaN and Missing Data

Meridian's EDA engine runs a VIF (multicollinearity) check before allowing `sample_posterior`. If any column contains NaN or inf, the check crashes with `MissingDataError: exog contains inf or nans` — which looks like a multicollinearity error but is really a data quality issue.

**Before reshaping arrays:**
1. Drop any column that is 100% NaN (like `Brand_PH_Avg_Price`)
2. Forward-fill then back-fill partial NaN gaps in controls:
   ```python
   df[control_cols] = df[control_cols].ffill().bfill()
   ```
   This is reasonable for slow-moving variables like competitor market shares and prices where the first 12 months may be missing.
3. Verify no NaN/inf remain:
   ```python
   assert df[control_cols].isna().sum().sum() == 0, "Controls still have NaN"
   ```

## Building InputData

### Dimension names matter

Meridian validates dimension names strictly. Using the wrong name produces errors like `The dimension list of array 'controls' doesn't match any of the following dimension lists`.

| Array | Required dims | Required name= |
|-------|--------------|----------------|
| `kpi` | `['geo', 'time']` | `name='kpi'` |
| `population` | `['geo']` | `name='population'` |
| `media_spend` | `['geo', 'time', 'media_channel']` | `name='media_spend'` |
| `media` | `['geo', 'media_time', 'media_channel']` | `name='media'` |
| `controls` | `['geo', 'time', 'control_variable']` | `name='controls'` |

The `name=` parameter on every `xr.DataArray` is **required** — Meridian validates it with `Array 'None' should have name 'kpi'` if omitted.

Note the subtle differences:
- `media` uses `media_time` (not `time`) — this allows extra leading periods for adstock warm-up
- Controls use `control_variable` (not `control`) as the third dimension name

### Single-geo (national) InputData

```python
time_coords = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(df[date_col])]
geo_coords = ['national']  # single-geo national model

input_data = InputData(
    kpi=xr.DataArray(
        kpi_array, name='kpi',
        dims=['geo', 'time'],
        coords={'time': time_coords, 'geo': geo_coords}
    ),
    kpi_type='revenue',  # revenue from kpi is used directly; revenue_per_kpi ignored
    population=xr.DataArray(
        [1.0], name='population',
        dims=['geo'],
        coords={'geo': geo_coords}
    ),
    media_spend=xr.DataArray(
        media_spend_array, name='media_spend',
        dims=['geo', 'time', 'media_channel'],
        coords={'time': time_coords, 'geo': geo_coords, 'media_channel': media_cols}
    ),
    media=xr.DataArray(
        media_volume_array, name='media',
        dims=['geo', 'media_time', 'media_channel'],
        coords={'media_time': time_coords, 'geo': geo_coords, 'media_channel': media_cols}
    ),
    controls=xr.DataArray(
        control_array, name='controls',
        dims=['geo', 'time', 'control_variable'],
        coords={'time': time_coords, 'geo': geo_coords, 'control_variable': control_cols}
    ),
)
```

### Multi-geo InputData

With multi-geo data, the key differences are:
- **Geos are real** — not a dummy `['national']` singleton
- **Population matters** — Meridian scales KPI per capita across regions of different sizes
- **Reshape via pivot** — the long-format DataFrame must be pivoted into `(n_geos, n_times, n_channels)` arrays

```python
# Sort for consistent reshaping
df = df.sort_values([geo_col, date_col]).reset_index(drop=True)
geo_names = sorted(df[geo_col].unique())
time_values = sorted(df[date_col].unique())

# Pivot each variable into (n_geos, n_times) then stack channels on axis=-1
kpi_array = df.pivot(index=geo_col, columns=date_col, values=kpi_col) \
              .loc[geo_names, time_values].values  # (n_geos, n_times)

media_spend_array = np.stack([
    df.pivot(index=geo_col, columns=date_col, values=col).loc[geo_names, time_values].values
    for col in media_spend_cols
], axis=-1)  # (n_geos, n_times, n_media)

# Population: approximate regional populations (used for per-capita scaling)
population_array = np.array([population_map[g] for g in geo_names])

# InputData — same structure, just real geo coordinates and population
input_data = InputData(
    kpi=xr.DataArray(kpi_array, name='kpi', dims=['geo', 'time'],
                     coords={'time': time_values, 'geo': geo_names}),
    kpi_type='revenue',
    population=xr.DataArray(population_array, name='population', dims=['geo'],
                            coords={'geo': geo_names}),
    # ... media_spend, media, controls same as single-geo but with real geo_names
)
```

Multi-geo models enable hierarchical pooling (geo-level coefficients drawn from a shared national prior), stronger causal identification (geo × time variation), and geo-targeted budget optimization.

### Accessing control dimension after creation

When printing or accessing the control dimension from an InputData object, use `control_variable`:
```python
list(input_data.controls.control_variable.values)  # correct
list(input_data.controls.control.values)            # AttributeError!
```

## Visualization: Use the Built-in Visualizer API

Meridian ships a `visualizer` module (`meridian.analysis.visualizer`) with Altair-based charts. **Always use these instead of manually parsing raw xarray datasets** — the raw data structures are version-dependent and fragile.

### Key visualizer classes

```python
from meridian.analysis import visualizer

# Response curves, adstock decay, Hill curves
media_effects = visualizer.MediaEffects(mmm)
media_effects.plot_response_curves(confidence_level=0.9, include_ci=True)
media_effects.plot_adstock_decay(include_ci=True)
media_effects.plot_hill_curves(include_prior=True, include_ci=True)

# ROI, contributions, spend vs effect
media_summary = visualizer.MediaSummary(mmm)
media_summary.plot_roi_bar_chart(include_ci=True)
media_summary.plot_contribution_waterfall_chart()
media_summary.plot_channel_contribution_area_chart(time_granularity='monthly')  # or 'weekly'/'quarterly'
media_summary.plot_spend_vs_contribution()
media_summary.plot_roi_vs_mroi()
media_summary.summary_table(include_prior=True, include_posterior=True)

# Convergence diagnostics, prior vs posterior
diagnostics = visualizer.ModelDiagnostics(mmm)
diagnostics.plot_rhat_boxplot()
diagnostics.plot_prior_and_posterior_distribution(parameter='roi_m')

# Model fit (expected vs actual)
model_fit = visualizer.ModelFit(mmm)
model_fit.plot_model_fit(include_baseline=True, include_ci=True)
```

All methods return Altair `alt.Chart` objects that render in Jupyter with `.display()`.

### Do NOT manually parse `response_curves()` output

The raw `analyzer.response_curves()` returns an xarray Dataset with dimensions `(channel, spend_multiplier)` and variables `spend` and `incremental_outcome`. It does **not** have `mean`, `ci_lo`, `ci_hi` as separate variables — those are encoded in a `metric` dimension of `incremental_outcome`, but the exact structure varies by Meridian version. Manually accessing `rc['mean']` or `rc['ci_lo']` will raise `KeyError`.

If you need raw data, use the data methods on the visualizer classes instead:
```python
media_effects.response_curves_data(confidence_level=0.9)  # structured for plotting
media_effects.adstock_decay_dataframe()                    # pd.DataFrame
media_effects.hill_curves_dataframe()                      # pd.DataFrame
```

### HTML summary report

For a one-page interactive report covering model fit, contributions, ROI, and budget insights:
```python
from meridian.analysis.summarizer import Summarizer
summary = Summarizer(mmm)
summary.output_model_results_summary(
    filename='meridian_summary', filepath='.', start_date='2022-01-01', end_date='2023-12-31'
)
```

## Quick Checklist

Before calling `sample_posterior`, verify:

- [ ] Media columns are only Brand M's own spend (no competitor Brand_B/PH/P columns)
- [ ] No aggregate + component columns together (e.g., Meta_Spends_Agg with Meta1+Meta2)
- [ ] Volume array uses actual impressions/GRPs where available
- [ ] Volume and spend arrays have identical channel count and order
- [ ] Controls are external/non-actionable factors only
- [ ] No NaN or inf in any array (especially controls)
- [ ] All DataArrays have `name=` set
- [ ] Dimension names match exactly: `control_variable`, `media_channel`, `media_time`
- [ ] Time coordinates are `YYYY-MM-DD` strings
- [ ] Controls are actually passed to InputData (easy to forget!)
- [ ] For multi-geo: population array has real values (not all 1.0)

For visualization after fitting:

- [ ] Use `visualizer.MediaEffects` / `MediaSummary` — not manual matplotlib on raw xarray
- [ ] Never access `rc['mean']` or `rc['ci_lo']` on `analyzer.response_curves()` output
