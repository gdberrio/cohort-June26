# Session 05 Meridian

# Session 5: Google Meridian - Full Walkthrough

```python
# --- Setup ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Meridian imports
try:
    import meridian
    from meridian import data as meridian_data
    from meridian.model import model as meridian_model
    from meridian.model import spec as meridian_spec
    import xarray as xr
    print(f"Meridian version: {meridian.__version__}")
    MERIDIAN_AVAILABLE = True
except ImportError as e:
    print(f"Meridian not available: {e}")
    print("See Notebook 4 for installation instructions.")
    MERIDIAN_AVAILABLE = False

# Load workshop data
df = pd.read_excel('../../data/MMM_Workshop_Data.xlsx')
print(f"\nDataset: {df.shape[0]} rows x {df.shape[1]} columns")
df.head()
```

## Meridian Architecture

Google Meridian implements a **Bayesian hierarchical MMM** with the following key components:

### Model Structure

```
KPI(t,g) = intercept(g)
          + SUM_c [ beta_c * HillAdstock(media_c(t,g)) ]
          + SUM_k [ gamma_k * control_k(t,g) ]
          + epsilon(t,g)
```

Where:
- **t** = time period, **g** = geography
- **HillAdstock** combines:
  - **Adstock** (geometric or Weibull decay) to capture lagged effects
  - **Hill function** (saturation curve) to capture diminishing returns
- **beta_c** = media coefficient (hierarchical across geos)
- **gamma_k** = control variable coefficient

### Backend: JAX / NumPyro

- **JAX**: Google's high-performance numerical computing library (like NumPy but with auto-differentiation and GPU/TPU support)
- **NumPyro**: Probabilistic programming built on JAX; runs NUTS (No U-Turn Sampler) MCMC
- This combination makes Meridian significantly faster than Stan-based alternatives

### Key Features

| Feature | Description |
|---------|-------------|
| Geo-level modeling | Hierarchical priors share information across geographies |
| Prior calibration | Incorporate results from experiments as informative priors |
| Budget optimization | Built-in optimizer uses fitted response curves |
| Reach & frequency | Native support for R&F media data |

```python
# --- Data Preparation ---
# Identify columns (adjust to match your data)
date_col = 'Date'
kpi_col = 'Revenue'

# Media spend columns
media_cols = [col for col in df.columns if 'Spend' in col or 'spend' in col]
print(f"Media columns ({len(media_cols)}): {media_cols}")

# Control columns
exclude_cols = [date_col, kpi_col] + media_cols
control_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
print(f"Control columns ({len(control_cols)}): {control_cols}")

n_times = len(df)
n_geos = 1  # single-geo national data
n_media = len(media_cols)

# Reshape for Meridian: (time, geo, channels)
media_spend_array = df[media_cols].values.reshape(n_times, n_geos, n_media)
media_volume_array = media_spend_array.copy()  # use spend as volume proxy
kpi_array = df[kpi_col].values.reshape(n_times, n_geos)

print(f"\nFormatted shapes:")
print(f"  media_spend: {media_spend_array.shape}")
print(f"  kpi: {kpi_array.shape}")
```

```python
# --- Build InputData ---
if MERIDIAN_AVAILABLE:
    time_coords = list(range(n_times))
    geo_coords = ['national']

    input_data = meridian_data.InputData(
        media_spend=xr.DataArray(
            media_spend_array,
            dims=['time', 'geo', 'media_channel'],
            coords={'time': time_coords, 'geo': geo_coords, 'media_channel': media_cols}
        ),
        media=xr.DataArray(
            media_volume_array,
            dims=['time', 'geo', 'media_channel'],
            coords={'time': time_coords, 'geo': geo_coords, 'media_channel': media_cols}
        ),
        kpi=xr.DataArray(
            kpi_array,
            dims=['time', 'geo'],
            coords={'time': time_coords, 'geo': geo_coords}
        ),
    )
    print("InputData created successfully!")
    print(f"  Channels: {list(input_data.media_spend.media_channel.values)}")
    print(f"  Time periods: {input_data.kpi.sizes['time']}")
else:
    print("Skipping InputData creation (Meridian not installed).")
    print("Arrays are formatted and ready - use Colab if needed.")
```

## Model Specification

Meridian's `ModelSpec` defines the model structure and **prior distributions**. Key priors include:

- **Media coefficients (`beta`)**: Half-normal priors (constrained positive, media should help)
- **Hill half-saturation (`ec`)**: Controls where diminishing returns kick in
- **Hill slope**: Controls steepness of saturation
- **Adstock decay (`alpha`)**: How quickly media effects fade over time

### Why Priors Matter

In Bayesian MMM, priors encode our **domain knowledge**:
- We expect media to have positive ROI (hence half-normal for betas)
- We expect adstock decay within 1-8 weeks for most digital channels
- We expect diminishing returns (Hill saturation) at some spend level

Meridian's defaults are reasonable starting points, but you should calibrate priors with experimental data when available.

```python
# --- Model Specification with Prior Configurations ---
if MERIDIAN_AVAILABLE:
    model_spec = meridian_spec.ModelSpec(
        media_channel_names=media_cols,
        # max_lag controls the adstock memory window
        max_lag=8,  # 8 weeks of carry-over effects
    )

    print("ModelSpec created with:")
    print(f"  Media channels: {model_spec.media_channel_names}")
    print(f"  Max lag: {model_spec.max_lag}")
    print(f"\nUsing Meridian default priors:")
    print(f"  - Half-normal priors on media coefficients (positive effect)")
    print(f"  - Beta priors on adstock decay (0-1 range)")
    print(f"  - LogNormal priors on Hill parameters")
else:
    print("ModelSpec would be created here with Meridian.")
    print("Key parameters: max_lag=8, default priors on Hill/Adstock.")
```

### Note: Model fitting may take 5-15 minutes

MCMC sampling is computationally intensive. The cell below runs multiple chains of the NUTS sampler. On a CPU this typically takes **5-15 minutes** depending on data size and number of channels.

**Tips for faster fitting:**
- Use Google Colab with GPU runtime
- Reduce `num_warmup` and `num_samples` for initial exploration
- Reduce `num_chains` (but keep at least 2 for convergence diagnostics)

```python
# --- Model Fitting ---
if MERIDIAN_AVAILABLE:
    # Initialize model
    mmm = meridian_model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )

    # Fit with MCMC sampling
    print("Starting MCMC sampling...")
    print("This may take 5-15 minutes on CPU.\n")

    mmm.fit(
        num_warmup=500,   # warm-up iterations per chain
        num_samples=500,  # posterior samples per chain
        num_chains=2,     # parallel MCMC chains
    )

    print("\nModel fitting complete!")
else:
    print("Model fitting requires Meridian. Use Colab if local install failed.")
```

```python
# --- Convergence Diagnostics ---
if MERIDIAN_AVAILABLE:
    # Check convergence via R-hat and effective sample size
    summary = mmm.get_summary()

    print("=== Convergence Diagnostics ===")
    print(f"\nR-hat values (should be < 1.1 for convergence):")
    if hasattr(summary, 'r_hat'):
        print(summary.r_hat)
    else:
        print(summary)

    print(f"\nAll R-hat < 1.1: indicates chains have converged.")
    print(f"Effective sample size > 100: indicates sufficient posterior samples.")
else:
    print("Convergence diagnostics require a fitted model.")
    print("\nKey metrics to check:")
    print("  - R-hat < 1.1 for all parameters (chains agree)")
    print("  - Effective sample size > 100 (enough independent samples)")
    print("  - No divergent transitions (sampler explored well)")
```

```python
# --- Results Extraction: ROI and mROI ---
if MERIDIAN_AVAILABLE:
    # Channel ROI table
    roi_df = mmm.get_roi_summary()
    print("=== Channel ROI Summary ===")
    print(roi_df.to_string())

    # Marginal ROI (mROI)
    mroi_df = mmm.get_mroi_summary()
    print("\n=== Marginal ROI Summary ===")
    print(mroi_df.to_string())
    print("\nROI = total return / total spend (average efficiency)")
    print("mROI = incremental return from next dollar (current marginal efficiency)")
else:
    print("ROI extraction requires a fitted model.")
    print("\nExpected output:")
    print("  - ROI per channel: total incremental revenue / total spend")
    print("  - mROI per channel: marginal return of next dollar spent")
```

```python
# --- Response Curves Visualization ---
if MERIDIAN_AVAILABLE:
    fig, axes = plt.subplots(1, n_media, figsize=(5 * n_media, 4), squeeze=False)

    for i, channel in enumerate(media_cols):
        ax = axes[0, i]
        # Generate spend range
        max_spend = media_spend_array[:, 0, i].max()
        spend_range = np.linspace(0, max_spend * 1.5, 100)

        # Get response curve from model
        try:
            response = mmm.get_response_curves()
            ax.plot(response[channel]['spend'], response[channel]['response'], 'b-', linewidth=2)
            # Mark current average spend
            avg_spend = media_spend_array[:, 0, i].mean()
            ax.axvline(avg_spend, color='red', linestyle='--', alpha=0.7, label='Avg. spend')
            ax.legend()
        except Exception:
            ax.text(0.5, 0.5, 'Response curve\nnot available',
                    transform=ax.transAxes, ha='center', va='center')

        ax.set_title(channel)
        ax.set_xlabel('Spend')
        ax.set_ylabel('Incremental KPI')

    plt.suptitle('Response Curves by Channel', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("Response curve visualization requires a fitted model.")
    print("\nResponse curves show the Hill-Adstock transformed relationship")
    print("between spend and incremental KPI for each channel.")
    print("The curve flattens at high spend levels (diminishing returns).")
```

```python
# --- Contribution Over Time ---
if MERIDIAN_AVAILABLE:
    try:
        contributions = mmm.get_media_contributions()

        fig, ax = plt.subplots(figsize=(14, 6))

        # Stacked area chart of channel contributions
        bottom = np.zeros(n_times)
        colors = plt.cm.Set2(np.linspace(0, 1, n_media))

        for i, channel in enumerate(media_cols):
            contrib = contributions[:, 0, i] if contributions.ndim == 3 else contributions[:, i]
            ax.fill_between(range(n_times), bottom, bottom + contrib,
                           alpha=0.7, label=channel, color=colors[i])
            bottom += contrib

        ax.plot(range(n_times), kpi_array[:, 0], 'k-', linewidth=1.5, label='Actual KPI', alpha=0.8)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('KPI')
        ax.set_title('Media Contribution Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Contribution chart error: {e}")
        print("API may differ by Meridian version.")
else:
    print("Contribution chart requires a fitted model.")
```

```python
# --- Posterior Distributions for Key Parameters ---
if MERIDIAN_AVAILABLE:
    try:
        posterior = mmm.get_posterior_samples()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Media coefficients
        ax = axes[0, 0]
        if 'beta_media' in posterior:
            for i, ch in enumerate(media_cols):
                samples = posterior['beta_media'][:, i]
                ax.hist(samples, bins=30, alpha=0.5, label=ch, density=True)
        ax.set_title('Media Coefficients (beta)')
        ax.legend(fontsize=8)

        # Hill EC (half-saturation)
        ax = axes[0, 1]
        if 'ec' in posterior:
            for i, ch in enumerate(media_cols):
                samples = posterior['ec'][:, i]
                ax.hist(samples, bins=30, alpha=0.5, label=ch, density=True)
        ax.set_title('Hill Half-Saturation (ec)')
        ax.legend(fontsize=8)

        # Hill slope
        ax = axes[1, 0]
        if 'slope' in posterior:
            for i, ch in enumerate(media_cols):
                samples = posterior['slope'][:, i]
                ax.hist(samples, bins=30, alpha=0.5, label=ch, density=True)
        ax.set_title('Hill Slope')
        ax.legend(fontsize=8)

        # Adstock decay
        ax = axes[1, 1]
        if 'alpha' in posterior:
            for i, ch in enumerate(media_cols):
                samples = posterior['alpha'][:, i]
                ax.hist(samples, bins=30, alpha=0.5, label=ch, density=True)
        ax.set_title('Adstock Decay (alpha)')
        ax.legend(fontsize=8)

        plt.suptitle('Posterior Distributions', fontsize=14)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Posterior visualization error: {e}")
else:
    print("Posterior distributions require a fitted model.")
    print("\nKey parameters to examine:")
    print("  - beta_media: channel effectiveness (higher = more impact per unit)")
    print("  - ec: half-saturation point (higher = more headroom before saturation)")
    print("  - slope: saturation steepness (higher = sharper diminishing returns)")
    print("  - alpha: adstock decay rate (higher = effects persist longer)")
```

## Prior vs. Posterior Comparison

A hallmark of Bayesian modeling is comparing what we assumed **before** seeing the data (prior) with what we learned **after** fitting the model (posterior). This comparison reveals:

- **Which channels the data is informative about** - posteriors much narrower than priors
- **Where priors dominate** - posteriors similar to priors (data alone is insufficient)
- **Potential misspecification** - posteriors pushed to extreme values

```python
# --- Prior vs Posterior Comparison ---
if MERIDIAN_AVAILABLE:
    try:
        # Sample from priors
        prior_samples = mmm.get_prior_samples()
        posterior_samples = mmm.get_posterior_samples()

        fig, axes = plt.subplots(1, n_media, figsize=(5 * n_media, 4), squeeze=False)

        for i, channel in enumerate(media_cols):
            ax = axes[0, i]
            if 'beta_media' in prior_samples and 'beta_media' in posterior_samples:
                ax.hist(prior_samples['beta_media'][:, i], bins=30, alpha=0.4,
                        density=True, color='gray', label='Prior')
                ax.hist(posterior_samples['beta_media'][:, i], bins=30, alpha=0.6,
                        density=True, color='steelblue', label='Posterior')
            ax.set_title(channel)
            ax.legend()
            ax.set_xlabel('beta_media')

        plt.suptitle('Prior vs. Posterior: Media Coefficients', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

        print("\nInterpretation:")
        print("- Narrow posteriors = data is informative for that channel")
        print("- Posteriors similar to priors = limited data signal")
        print("- Posteriors shifted far from priors = data contradicts assumptions")
    except Exception as e:
        print(f"Prior/posterior comparison error: {e}")
else:
    print("Prior vs. posterior comparison requires a fitted model.")
    print("\nWhat to look for:")
    print("  - Did the posterior narrow significantly from the prior?")
    print("    -> Yes: the data is informative about this parameter.")
    print("  - Is the posterior still wide like the prior?")
    print("    -> The data alone cannot identify this parameter well.")
    print("  - Was the posterior pushed to the boundary of the prior?")
    print("    -> Consider widening the prior or checking data quality.")
```

## Comparison to Scratch Models

In Week 2, we built MMMs from scratch using:
1. **OLS regression** with manually-specified adstock and saturation
2. **Bayesian regression** (PyMC) with priors on transformation parameters

### How Meridian Compares

| Aspect | Scratch OLS | Scratch Bayesian | Meridian |
|--------|------------|-----------------|----------|
| **Estimation** | Point estimates | Full posterior | Full posterior |
| **Uncertainty** | Confidence intervals | Credible intervals | Credible intervals |
| **Transformations** | Manual adstock + saturation | Parameterized in model | Hill-Adstock built-in |
| **Geo support** | Manual | Manual | Built-in hierarchical |
| **Budget optimization** | Manual from coefficients | Manual from posterior | Built-in optimizer |
| **Prior calibration** | N/A | Manual | Structured framework |
| **Speed** | Fast | Moderate (PyMC/Stan) | Fast (JAX/NumPyro) |

### Discussion Points

- Do channel rankings agree across approaches? If not, why?
- Are the ROI magnitudes similar? What drives differences?
- How do the adstock and saturation parameters compare?
- When would you prefer the scratch approach vs. Meridian?

**Key insight**: Different modeling frameworks should give broadly consistent results if the data is clean and the model is well-specified. Large disagreements signal either data issues or model misspecification.

## Exercise: Fit Meridian on Workshop Data

Your turn! Complete the tasks below:

1. **Fit the default model** using the code above (or on Colab)
2. **Change the priors** - try a more informative or more diffuse prior on media coefficients
3. **Compare results** - do ROI rankings change with different priors?
4. **Increase max_lag** to 12 - how does this affect adstock estimates?

```python
# TODO: Exercise 1 - Fit the default model
# Use the code from above, or copy to Colab
# Record the ROI for each channel

# default_roi = ...
```

```python
# TODO: Exercise 2 - Try different prior specifications
# Example: create a ModelSpec with tighter priors

# model_spec_tight = meridian_spec.ModelSpec(
#     media_channel_names=media_cols,
#     max_lag=8,
#     # Add custom prior configurations here
# )

# mmm_tight = meridian_model.Meridian(input_data=input_data, model_spec=model_spec_tight)
# mmm_tight.fit(num_warmup=500, num_samples=500, num_chains=2)
```

```python
# TODO: Exercise 3 - Compare results
# Compare ROI tables from default vs. tight priors

# tight_roi = mmm_tight.get_roi_summary()
# print("Default ROI:")
# print(default_roi)
# print("\nTight Prior ROI:")
# print(tight_roi)
```

```python
# TODO: Exercise 4 - Increase max_lag to 12
# Does the adstock decay parameter change?

# model_spec_lag12 = meridian_spec.ModelSpec(
#     media_channel_names=media_cols,
#     max_lag=12,
# )
# mmm_lag12 = meridian_model.Meridian(input_data=input_data, model_spec=model_spec_lag12)
# mmm_lag12.fit(num_warmup=500, num_samples=500, num_chains=2)
```

## Deliverable

By the end of this session, you should have:

- A **fitted Meridian model** on the workshop data (or on Colab)
- An **ROI summary table** showing each channel's return on investment
- **Response curves** showing diminishing returns for each channel
- Understanding of how **prior choices** affect model results
- Comparison with your **scratch model results** from Week 2

Save your ROI summary - we will use it in Session 6 for budget optimization.

