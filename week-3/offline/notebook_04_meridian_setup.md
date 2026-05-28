# Notebook 04 Meridian Setup

# Notebook 4: Google Meridian - Setup & Data Formatting

## Overview

**Google Meridian** is Google's open-source Bayesian Marketing Mix Model (MMM) framework built on **JAX/NumPyro**. It provides a production-grade implementation of hierarchical Bayesian MMM with built-in support for:

- **Hill-Adstock** media transformations
- **Geo-level** hierarchical modeling
- **Budget optimization** via response curves
- **Prior calibration** from experiments

This notebook verifies your Meridian installation and formats the workshop data into the structure Meridian expects. We will use this formatted data in Session 5 for a full model walkthrough.

```python
# --- Installation Verification ---
try:
    import meridian
    from meridian import data as meridian_data
    from meridian.model import model as meridian_model
    from meridian.model import spec as meridian_spec
    print(f"Meridian version: {meridian.__version__}")
    print("Meridian imported successfully!")
except ImportError as e:
    print(f"Meridian not available: {e}")
    print("Install with: pip install google-meridian")
```

## Data Requirements

Meridian expects data in a very specific format. The core inputs are **matrices** with dimensions organized by **time** and **geo** (geography):

| Input | Shape | Description |
|-------|-------|-------------|
| `media_spend` | `(n_times, n_geos, n_media_channels)` | Spend per channel per geo per time period |
| `media` (volume) | `(n_times, n_geos, n_media_channels)` | Impressions/GRPs (optional, defaults to spend) |
| `kpi` | `(n_times, n_geos)` | Target variable (revenue, conversions, etc.) |
| `controls` | `(n_times, n_geos, n_controls)` | Control variables (seasonality, macro, etc.) |

**Single-geo case:** When you have national-level data (no geo breakdown), the geo dimension is simply 1. So a dataset with 104 weeks and 5 media channels would have `media_spend` shaped `(104, 1, 5)`.

Data is passed to Meridian via an **`InputData`** object, which wraps xarray datasets.

```python
# --- Format Workshop Data ---
import numpy as np
import pandas as pd

# Load the workshop dataset
df = pd.read_excel('../../data/MMM_Workshop_Data.xlsx')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()
```

```python
# --- Identify columns ---
# Adjust these based on your actual column names
date_col = 'Date'  # or 'Week', 'date', etc.
kpi_col = 'Revenue'  # or 'Sales', 'Conversions', etc.

# Media spend columns (update to match your data)
media_cols = [col for col in df.columns if 'Spend' in col or 'spend' in col]
print(f"Detected media columns: {media_cols}")

# Control columns (everything that is not date, KPI, or media)
exclude_cols = [date_col, kpi_col] + media_cols
control_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
print(f"Detected control columns: {control_cols}")

n_times = len(df)
n_geos = 1  # single-geo (national level)
n_media = len(media_cols)
n_controls = len(control_cols)

print(f"\nDimensions: {n_times} time periods, {n_geos} geo, {n_media} media channels, {n_controls} controls")
```

```python
# --- Create arrays in Meridian-expected shapes ---
# Meridian expects (n_times, n_geos, n_channels) for media
# and (n_times, n_geos) for KPI

# Media spend: (n_times, 1, n_media)
media_spend_array = df[media_cols].values.reshape(n_times, n_geos, n_media)
print(f"media_spend shape: {media_spend_array.shape}")

# KPI: (n_times, 1)
kpi_array = df[kpi_col].values.reshape(n_times, n_geos)
print(f"kpi shape: {kpi_array.shape}")

# Controls: (n_times, 1, n_controls)
if n_controls > 0:
    controls_array = df[control_cols].values.reshape(n_times, n_geos, n_controls)
    print(f"controls shape: {controls_array.shape}")
else:
    controls_array = None
    print("No control variables detected.")

# Optional: media volume (impressions) - use spend as proxy if not available
media_volume_array = media_spend_array.copy()
print(f"\nAll arrays formatted successfully!")
```

```python
# --- Create Meridian InputData Object ---
import xarray as xr

try:
    # Build coordinate labels
    time_coords = list(range(n_times))
    geo_coords = ['national']
    media_channel_coords = media_cols
    control_coords = control_cols if n_controls > 0 else []

    # Create xarray DataArrays
    media_spend_da = xr.DataArray(
        media_spend_array,
        dims=['time', 'geo', 'media_channel'],
        coords={'time': time_coords, 'geo': geo_coords, 'media_channel': media_channel_coords}
    )

    kpi_da = xr.DataArray(
        kpi_array,
        dims=['time', 'geo'],
        coords={'time': time_coords, 'geo': geo_coords}
    )

    # Build InputData (API may vary by Meridian version)
    input_data = meridian_data.InputData(
        media_spend=media_spend_da,
        media=xr.DataArray(
            media_volume_array,
            dims=['time', 'geo', 'media_channel'],
            coords={'time': time_coords, 'geo': geo_coords, 'media_channel': media_channel_coords}
        ),
        kpi=kpi_da,
    )

    print("InputData created successfully!")
    print(f"  Media channels: {list(input_data.media_spend.media_channel.values)}")
    print(f"  Time periods: {input_data.media_spend.sizes['time']}")
    print(f"  Geos: {input_data.media_spend.sizes['geo']}")

except Exception as e:
    print(f"Could not create InputData: {e}")
    print("This is expected if Meridian is not installed.")
    print("The arrays above are correctly formatted - we will use them in Session 5.")
```

```python
# --- Minimal Model Spec ---
try:
    model_spec = meridian_spec.ModelSpec(
        media_channel_names=media_cols,
    )

    print("ModelSpec created successfully!")
    print(f"\nModel Specification:")
    print(f"  Media channels: {model_spec.media_channel_names}")
    print(f"  Max lag: {model_spec.max_lag}")
    print(f"\nThis minimal spec uses Meridian's default priors for all parameters.")
    print("In Session 5, we will customize priors and run the full model.")

except Exception as e:
    print(f"Could not create ModelSpec: {e}")
    print("This is expected if Meridian is not installed.")
```

## Troubleshooting

### Common Installation Issues

1. **JAX not found**: Meridian requires JAX. Install with:
   ```bash
   pip install jax jaxlib
   ```
   On Mac with Apple Silicon, you may need:
   ```bash
   pip install jax-metal
   ```

2. **NumPyro version conflict**: Meridian pins specific NumPyro versions. Use a dedicated virtual environment:
   ```bash
   python -m venv meridian_env
   source meridian_env/bin/activate
   pip install google-meridian
   ```

3. **Memory issues**: Meridian's MCMC sampling is memory-intensive. Ensure at least 8GB RAM available.

### Google Colab Fallback

If local installation fails, use Google Colab:

1. Open [Google Colab](https://colab.research.google.com)
2. Run: `!pip install google-meridian`
3. Upload the workshop data file
4. Copy the code cells from this notebook

Colab provides a pre-configured environment with JAX support and GPU access (which accelerates MCMC sampling significantly).

### Verifying Your Setup

If the import cell above ran without errors, you are ready for Session 5. If not, try the Colab fallback before the session.

