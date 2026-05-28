# Session 07 Experimentation Geolift

# Session 7: Experimentation & Geo-Lift Testing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append('..')
from utils.geo_utils import *

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

np.random.seed(42)
```

---

## Power Analysis Review

Before designing any experiment, we need to ensure we can actually detect the effect we care about. Let's quickly review the core functions from Notebook 6.

```python
def compute_sample_size(mde, alpha=0.05, power=0.80, sigma=1.0):
    """Compute required sample size per group for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sigma / mde) ** 2
    return int(np.ceil(n))

def compute_power(n_per_group, mde, alpha=0.05, sigma=1.0):
    """Compute power given sample size and MDE."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se = sigma * np.sqrt(2 / n_per_group)
    z_beta = mde / se - z_alpha
    return stats.norm.cdf(z_beta)

# Quick power curve
sample_sizes = np.arange(50, 3001, 50)
mde_values = [0.1, 0.2, 0.3, 0.5]

plt.figure(figsize=(10, 6))
for mde_val in mde_values:
    powers = [compute_power(n, mde_val) for n in sample_sizes]
    plt.plot(sample_sizes, powers, label=f'MDE = {mde_val}', linewidth=2)

plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80% power')
plt.xlabel('Sample Size per Group')
plt.ylabel('Statistical Power')
plt.title('Power Curves for Different Minimum Detectable Effects')
plt.legend()
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
```

---

## Introduction to Geo Experiments

### Why Geo Experiments?

Not all marketing channels can be randomized at the user level:

- **TV advertising** reaches everyone in a media market (DMA)
- **Out-of-home (OOH)** billboards are location-based
- **Radio** is broadcast to geographic areas
- **Local promotions** vary by store or region

For these channels, we randomize at the **geographic** level: some markets get the treatment (e.g., increased TV spend), while others serve as controls.

### Core Approach

1. **Pre-period:** Observe all markets before the intervention (no treatment anywhere)
2. **Test period:** Apply treatment to selected markets, keep others as control
3. **Analysis:** Compare treated markets to a **synthetic counterfactual** built from control markets

### Synthetic Control Method

The key idea: construct a "synthetic" version of the treated market as a **weighted combination of control markets** that best matches the treated market's pre-period behavior. Any deviation in the test period is attributed to the treatment.

This is more robust than simple difference-in-differences because it accounts for market-specific trends.

---

## GeoLift Walkthrough

Let's work through a complete geo experiment analysis using the GeoLift dataset.

```python
# Load pre-test and test data
pre_data = load_geo_data('../../data/GeoLift_PreTest.csv')
test_data = load_geo_data('../../data/GeoLift_Test.csv')

print("Pre-test data:")
print(f"  Shape: {pre_data.shape}")
print(f"  Locations: {pre_data['location'].nunique()}")
print(f"  Time periods: {pre_data['time'].nunique()}")
print(f"  Date range: {pre_data['time'].min()} to {pre_data['time'].max()}")
print()
print("Test data:")
print(f"  Shape: {test_data.shape}")
print(f"  Locations: {test_data['location'].nunique()}")
print(f"  Time periods: {test_data['time'].nunique()}")
print(f"  Date range: {test_data['time'].min()} to {test_data['time'].max()}")
print()
print(pre_data.head(10))
```

```python
# Explore: Plot time series for a few locations
sample_locations = pre_data.groupby('location')['Y'].mean().nlargest(5).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pre-period
for loc in sample_locations:
    loc_data = pre_data[pre_data['location'] == loc].sort_values('time')
    axes[0].plot(loc_data['time'], loc_data['Y'], label=loc, alpha=0.8)
axes[0].set_title('Pre-Test Period')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Y')
axes[0].legend(fontsize=9)

# Test period
for loc in sample_locations:
    loc_data = test_data[test_data['location'] == loc].sort_values('time')
    axes[1].plot(loc_data['time'], loc_data['Y'], label=loc, alpha=0.8)
axes[1].set_title('Test Period')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Y')
axes[1].legend(fontsize=9)

plt.suptitle('Time Series for Top 5 Locations by Mean Outcome', fontsize=14)
plt.tight_layout()
plt.show()
```

```python
# Pre-test analysis: compute correlations between geos
pivot_pre = pivot_geo_data(pre_data)
print(f"Pivoted pre-test data shape: {pivot_pre.shape}")
print(f"Columns (locations): {list(pivot_pre.columns[:10])}... ({len(pivot_pre.columns)} total)")

# Correlation matrix
corr_matrix = pivot_pre.corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(label='Correlation')
plt.title('Geo-to-Geo Correlation Matrix (Pre-Period)')
plt.tight_layout()
plt.show()

# Summary statistics
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
corr_values = upper_triangle.stack().values
print(f"\nPairwise correlation summary:")
print(f"  Mean: {corr_values.mean():.3f}")
print(f"  Median: {np.median(corr_values):.3f}")
print(f"  Min: {corr_values.min():.3f}")
print(f"  Max: {corr_values.max():.3f}")
```

```python
# Identify treatment candidates
# Pick a few markets as "treated" for our analysis
treatment_market = 'chicago'
print(f"Selected treatment market: {treatment_market}")

# Simple difference-in-differences
effect = compute_treatment_effect(pre_data, test_data, treatment_market)
print(f"\nSimple DiD treatment effect estimate:")
print(f"  Effect: {effect:.2f}")
```

---

## Synthetic Control

The synthetic control method constructs a counterfactual for the treated market using a **weighted combination of control markets**.

**Steps:**
1. Use pre-period data to learn weights: find weights $w_1, w_2, ..., w_J$ for control markets such that the weighted combination best matches the treated market in the pre-period.
2. Apply those same weights to the test period to get the "synthetic control" -- what the treated market *would have looked like* without treatment.
3. Treatment effect = actual outcome - synthetic control.

We use Ridge regression with a positivity constraint to learn the weights.

```python
from sklearn.linear_model import Ridge

# Use pre-period to learn weights for control markets
pivot_pre = pivot_geo_data(pre_data)
treatment_market = 'chicago'
control_markets = [c for c in pivot_pre.columns if c != treatment_market]

X_pre = pivot_pre[control_markets].values
y_pre = pivot_pre[treatment_market].values

# Fit Ridge regression (positive=True encourages non-negative weights)
model = Ridge(alpha=1.0, fit_intercept=False)
model.fit(X_pre, y_pre)

# Check fit quality in pre-period
y_pre_synthetic = model.predict(X_pre)
pre_rmse = np.sqrt(np.mean((y_pre - y_pre_synthetic) ** 2))
pre_mape = np.mean(np.abs((y_pre - y_pre_synthetic) / y_pre)) * 100

print(f"Pre-period fit quality:")
print(f"  RMSE: {pre_rmse:.2f}")
print(f"  MAPE: {pre_mape:.1f}%")

# Top contributing control markets
weights = pd.Series(model.coef_, index=control_markets).sort_values(ascending=False)
print(f"\nTop 10 control market weights:")
for market, weight in weights.head(10).items():
    print(f"  {market}: {weight:.4f}")
```

```python
# Apply weights to test period
pivot_test = pivot_geo_data(test_data)
y_synthetic = model.predict(pivot_test[control_markets].values)
y_actual = pivot_test[treatment_market].values

# Treatment effect
effect = y_actual - y_synthetic

# Plot actual vs synthetic control
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

# Top panel: actual vs synthetic
time_pre = np.arange(len(y_pre))
time_test = np.arange(len(y_pre), len(y_pre) + len(y_actual))

axes[0].plot(time_pre, y_pre, color='navy', linewidth=2, label='Actual (pre-period)')
axes[0].plot(time_pre, y_pre_synthetic, color='orange', linewidth=2, linestyle='--', label='Synthetic (pre-period)')
axes[0].plot(time_test, y_actual, color='navy', linewidth=2)
axes[0].plot(time_test, y_synthetic, color='orange', linewidth=2, linestyle='--', label='Synthetic (test-period)')
axes[0].axvline(x=len(y_pre), color='red', linestyle=':', linewidth=2, label='Treatment start')
axes[0].set_ylabel('Outcome (Y)')
axes[0].set_title(f'Synthetic Control for {treatment_market.title()}')
axes[0].legend()

# Bottom panel: treatment effect over time
axes[1].bar(time_test, effect, color='green', alpha=0.7)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_xlabel('Time Period')
axes[1].set_ylabel('Treatment Effect')
axes[1].set_title('Estimated Treatment Effect Over Time')

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\nTreatment Effect Summary for {treatment_market.title()}:")
print(f"  Average effect per period: {effect.mean():.2f}")
print(f"  Cumulative effect: {effect.sum():.2f}")
print(f"  Relative lift: {effect.mean() / y_synthetic.mean():.1%}")
```

```python
# Confidence intervals via permutation test (placebo test)
# Run synthetic control for EVERY control market as if it were treated

placebo_effects = {}

for placebo_market in control_markets[:20]:  # limit to 20 for speed
    # Fit synthetic control for this placebo market
    other_controls = [c for c in control_markets if c != placebo_market]
    if treatment_market in other_controls:
        other_controls.remove(treatment_market)
    
    X_placebo = pivot_pre[other_controls].values
    y_placebo = pivot_pre[placebo_market].values
    
    model_placebo = Ridge(alpha=1.0, fit_intercept=False)
    model_placebo.fit(X_placebo, y_placebo)
    
    # Test period placebo effect
    y_placebo_synthetic = model_placebo.predict(pivot_test[other_controls].values)
    y_placebo_actual = pivot_test[placebo_market].values
    placebo_effect = (y_placebo_actual - y_placebo_synthetic).mean()
    placebo_effects[placebo_market] = placebo_effect

# Compare treatment effect to placebo distribution
treatment_avg_effect = effect.mean()
placebo_values = list(placebo_effects.values())

plt.figure(figsize=(10, 6))
plt.hist(placebo_values, bins=15, edgecolor='black', alpha=0.7, color='lightblue', label='Placebo effects')
plt.axvline(x=treatment_avg_effect, color='red', linewidth=2, label=f'Treatment effect: {treatment_avg_effect:.2f}')
plt.xlabel('Average Treatment Effect')
plt.ylabel('Count')
plt.title('Permutation Test: Treatment Effect vs Placebo Distribution')
plt.legend()
plt.tight_layout()
plt.show()

# p-value from permutation test
p_value = np.mean([abs(p) >= abs(treatment_avg_effect) for p in placebo_values])
print(f"Permutation test p-value: {p_value:.3f}")
```

---

## CausalImpact Alternative

An alternative to the synthetic control method is Google's **CausalImpact**, which uses **Bayesian Structural Time Series (BSTS)** models.

Key differences:
- **BSTS** models the time series structure explicitly (trend, seasonality)
- Provides **posterior intervals** for the treatment effect (fully Bayesian)
- Automatically handles variable selection for control series
- Available in Python via the `pycausalimpact` package

```python
# Conceptual example (requires pycausalimpact installation)
# from causalimpact import CausalImpact
#
# # data: DataFrame with columns [y, x1, x2, ...]
# # y = treated market, x1/x2/... = control markets
# pre_period = ['2023-01-01', '2023-06-30']
# post_period = ['2023-07-01', '2023-09-30']
#
# ci = CausalImpact(data, pre_period, post_period)
# ci.plot()
# ci.summary()
# ci.summary(output='report')
```

Both methods (synthetic control and CausalImpact) answer the same question: "What would have happened without the treatment?" They just use different statistical machinery to construct the counterfactual.

---

## Exercise: Design Your Own Geo Experiment

**Scenario:** Your brand wants to test the incrementality of TV advertising in 3 markets.

Work through the following steps:

1. Define your hypothesis
2. Set your MDE (Minimum Detectable Effect)
3. Run power analysis
4. Select treatment and control markets
5. Determine experiment duration

```python
# TODO: Step 1 - Define your hypothesis
# H0: TV advertising has no incremental effect on sales in the treated markets
# H1: TV advertising increases sales by at least X%

# hypothesis = "..."
```

```python
# TODO: Step 2 - Set your MDE
# What is the minimum lift that would justify the TV investment?
# Consider: TV cost per market, baseline sales, break-even lift

# tv_cost_per_market = ...
# baseline_sales_per_market = ...
# breakeven_lift = tv_cost_per_market / baseline_sales_per_market
# mde = ...  # should be larger than breakeven to be meaningful
```

```python
# TODO: Step 3 - Power analysis
# Estimate variance from pre-period data
# geo_means = pre_data.groupby('location')['Y'].mean()
# sigma = geo_means.std()
# n_required = compute_sample_size(mde=..., sigma=sigma)
# print(f"Geos needed per group: {n_required}")
```

```python
# TODO: Step 4 - Select markets
# Criteria: similar pre-period trends, no spillover, representative
# treatment_markets = ['...', '...', '...']
# control_markets = ['...', '...', ...]
```

```python
# TODO: Step 5 - Determine duration
# Consider: weekly cycles, seasonality, budget constraints
# duration_weeks = ...
```

---

## Calibrating MMM with Experiments

One of the most powerful applications of experimentation in Marketing Science is **calibrating MMM models**.

The idea is simple but profound:
- Run a geo experiment to measure the causal effect of a channel (e.g., TV)
- Use the experiment result as an **informative prior** in a Bayesian MMM
- This anchors the model to ground truth, improving accuracy for all channels

### Why This Matters

MMM on its own suffers from multicollinearity and identifiability issues. Experiments provide **causal identification** that the observational model cannot. By combining the two:
- The experiment provides an unbiased point estimate with uncertainty
- The MMM extends this to all channels and handles ongoing optimization
- The result is more trustworthy than either method alone

```python
# Concept: Using experiment results as informative priors in PyMC

# Suppose a geo experiment found:
#   TV lift = 15% with 95% CI [8%, 22%]
#   This means: mu = 0.15, sigma ~= (0.22 - 0.08) / (2 * 1.96) ~= 0.035

experiment_mean = 0.15
experiment_ci_lower = 0.08
experiment_ci_upper = 0.22
experiment_sigma = (experiment_ci_upper - experiment_ci_lower) / (2 * 1.96)

print(f"Experiment result: {experiment_mean:.1%} lift")
print(f"95% CI: [{experiment_ci_lower:.1%}, {experiment_ci_upper:.1%}]")
print(f"Implied prior sigma: {experiment_sigma:.4f}")
print()

# In PyMC, this translates to:
print("PyMC prior specification:")
print(f"  # WITH experiment calibration:")
print(f"  tv_beta = pm.Normal('tv_beta', mu={experiment_mean}, sigma={experiment_sigma:.3f})")
print(f"")
print(f"  # WITHOUT calibration (typical weakly informative prior):")
print(f"  tv_beta = pm.HalfNormal('tv_beta', sigma=1)")
print()

# Visualize the difference
x = np.linspace(-0.5, 1.5, 1000)

fig, ax = plt.subplots(figsize=(10, 6))

# Informative prior from experiment
informed_prior = stats.norm.pdf(x, loc=experiment_mean, scale=experiment_sigma)
ax.plot(x, informed_prior, color='navy', linewidth=2, label=f'Calibrated: N({experiment_mean}, {experiment_sigma:.3f})')
ax.fill_between(x, informed_prior, alpha=0.2, color='navy')

# Weakly informative prior
weak_prior = stats.halfnorm.pdf(x, scale=1)
ax.plot(x, weak_prior, color='orange', linewidth=2, label='Uncalibrated: HalfNormal(1)')
ax.fill_between(x, weak_prior, alpha=0.2, color='orange')

ax.set_xlabel('TV Beta (Effect Size)')
ax.set_ylabel('Density')
ax.set_title('Prior Distributions: Calibrated vs Uncalibrated')
ax.legend()
ax.set_xlim(-0.2, 1.5)
plt.tight_layout()
plt.show()
```

### How Production Frameworks Handle Calibration

**Google Meridian:**
- Accepts experiment results as `calibration_data` input
- Automatically adjusts priors on media coefficients
- Supports multiple calibration experiments simultaneously

**Meta Robyn:**
- Uses `calibration_input` parameter with experiment lift results
- Penalizes model candidates that deviate from experimental evidence
- Integrated into the model selection (Pareto-optimal) process

**PyMC-Marketing:**
- Full flexibility to set custom priors directly in PyMC code
- Can incorporate experiments as priors or as likelihood constraints
- Most transparent but requires more statistical knowledge

---

## Building a Measurement Program

No single method gives you the full picture. The best marketing measurement programs **triangulate** across multiple methods:

### The Three Pillars

| Method | Strengths | Weaknesses | Cadence |
|--------|-----------|------------|--------|
| **MMM** | Holistic, all channels, long-term effects | Correlational, low granularity | Quarterly refresh |
| **Attribution** | Granular, real-time, user-level | Last-touch bias, no incrementality | Ongoing |
| **Experiments** | Causal, gold standard | Expensive, limited scope | 2-3 per year |

### Recommended Cadence

- **MMM:** Refresh quarterly with latest data. Use for budget allocation and planning.
- **Experiments (Geo-Lift, Conversion Lift):** Run 2-3 per year on highest-spend or most uncertain channels. Use results to calibrate MMM.
- **Attribution:** Run continuously for tactical optimization (bidding, targeting, creative). Do NOT use for budget allocation.

### Organizational Considerations

- **Executive buy-in:** Experiments require holding out budget -- leadership must support this.
- **Cross-functional alignment:** Media, analytics, finance, and brand teams must agree on MDE and success criteria before the experiment starts.
- **Documentation:** Every experiment should have a pre-registered analysis plan.
- **Learning agenda:** Maintain a prioritized backlog of measurement questions.
- **Iteration:** Each experiment should inform the next one and improve the MMM.

---

## Deliverables

By the end of this session, you should have:

1. **GeoLift analysis notebook:** Completed synthetic control analysis with treatment effect estimates and permutation-based inference.

2. **1-page experiment design:** A concise document outlining your proposed geo experiment -- hypothesis, MDE, power analysis, market selection, duration, and expected timeline.

3. **Calibrated model concept:** A clear understanding of how experiment results feed back into the MMM as informative priors, and how this improves overall measurement quality.

