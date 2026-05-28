# Notebook 06 Power Analysis

# Notebook 7: Power Analysis & Sample Size Calculations

## Overview

Before running any experiment, we need to know: **How many observations?** **How long?** **What effect size can we detect?**

This notebook covers power analysis from scratch. We will:

1. Review the key statistical concepts behind hypothesis testing and power
2. Derive the analytical formula for sample size calculations
3. Build power curves to visualize trade-offs
4. Use Monte Carlo simulation as an alternative to analytical formulas
5. Apply power analysis to geo experiment design
6. Discuss duration planning for experiments

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

np.random.seed(42)
```

## Key Concepts

Before diving into calculations, let's review the foundational concepts:

- **Type I error (alpha):** Rejecting H0 when it is true (false positive). We typically set alpha = 0.05, meaning we accept a 5% chance of incorrectly declaring an effect exists.

- **Type II error (beta):** Failing to reject H0 when it is false (false negative). This means we miss a real effect.

- **Power = 1 - beta:** The probability of detecting a real effect when one truly exists. Convention is to target power = 0.80 (80%), meaning we want at least an 80% chance of detecting the effect.

- **Minimum Detectable Effect (MDE):** The smallest effect size we want to be able to detect. Smaller MDE requires larger sample sizes. The MDE should be driven by business relevance -- what is the smallest lift that would justify the cost of the intervention?

- **Sample size (n):** The number of observations needed per group to achieve the desired power at the specified alpha and MDE.

These five quantities are interconnected: given any four, you can solve for the fifth.

## Analytical Power Analysis

For a two-sample t-test (comparing treatment vs control means), the required sample size per group is:

$$n = 2 \left( \frac{(z_{\alpha/2} + z_{\beta}) \cdot \sigma}{\text{MDE}} \right)^2$$

where:
- $z_{\alpha/2}$ is the critical value for the significance level
- $z_{\beta}$ is the critical value for the desired power
- $\sigma$ is the standard deviation of the outcome
- MDE is the minimum detectable effect (absolute difference in means)

```python
def compute_sample_size(mde, alpha=0.05, power=0.80, sigma=1.0):
    """Compute required sample size per group for a two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sigma / mde) ** 2
    return int(np.ceil(n))

# Example: detect 5% lift in conversions
baseline_rate = 0.10
mde = 0.005  # absolute MDE
sigma = np.sqrt(baseline_rate * (1 - baseline_rate))
n = compute_sample_size(mde, sigma=sigma)
print(f"Baseline conversion rate: {baseline_rate:.1%}")
print(f"Absolute MDE: {mde} (relative lift: {mde/baseline_rate:.1%})")
print(f"Standard deviation: {sigma:.4f}")
print(f"Required sample size per group: {n:,}")
print(f"Total sample size (both groups): {2*n:,}")
```

## Power Curves

Power curves help us visualize the trade-offs between sample size, effect size, and power. They are essential tools for communicating with stakeholders about experiment design.

```python
# --- Plot 1: Power as a function of sample size for different MDE values ---

def compute_power(n_per_group, mde, alpha=0.05, sigma=1.0):
    """Compute power given sample size and MDE."""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    se = sigma * np.sqrt(2 / n_per_group)
    z_beta = mde / se - z_alpha
    return stats.norm.cdf(z_beta)

sample_sizes = np.arange(50, 5001, 50)
mde_values = [0.1, 0.2, 0.3, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Power vs sample size
for mde_val in mde_values:
    powers = [compute_power(n, mde_val) for n in sample_sizes]
    axes[0].plot(sample_sizes, powers, label=f'MDE = {mde_val}', linewidth=2)

axes[0].axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80% power')
axes[0].set_xlabel('Sample Size per Group')
axes[0].set_ylabel('Statistical Power')
axes[0].set_title('Power vs Sample Size for Different Effect Sizes')
axes[0].legend()
axes[0].set_ylim(0, 1.05)

# Right plot: Required sample size vs MDE
mde_range = np.linspace(0.05, 0.6, 100)
required_n = [compute_sample_size(m, power=0.80) for m in mde_range]

axes[1].plot(mde_range, required_n, color='navy', linewidth=2)
axes[1].set_xlabel('Minimum Detectable Effect (MDE)')
axes[1].set_ylabel('Required Sample Size per Group')
axes[1].set_title('Required Sample Size vs MDE (power=0.80, alpha=0.05)')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

print("Key insight: Halving the MDE roughly quadruples the required sample size!")
```

## Simulation-Based Power Analysis (Monte Carlo)

When the analytical formula does not apply (e.g., complex test statistics, non-normal data, clustered designs), we can estimate power through simulation. The idea is simple:

1. Simulate many experiments under the alternative hypothesis (effect exists)
2. Run the statistical test on each simulated dataset
3. Power = proportion of simulations where we reject H0

```python
def simulate_power(n_per_group, true_effect, sigma=1.0, alpha=0.05, n_simulations=5000):
    """Estimate power via Monte Carlo simulation."""
    rejections = 0
    for _ in range(n_simulations):
        control = np.random.normal(0, sigma, n_per_group)
        treatment = np.random.normal(true_effect, sigma, n_per_group)
        t_stat, p_value = stats.ttest_ind(control, treatment)
        if p_value < alpha:
            rejections += 1
    return rejections / n_simulations

# Compare analytical vs simulation
n_test = 500
effect_test = 0.2

power_analytical = compute_power(n_test, effect_test)
power_sim = simulate_power(n_per_group=n_test, true_effect=effect_test)

print(f"Analytical power: {power_analytical:.3f}")
print(f"Simulated power:  {power_sim:.3f}")
print(f"Difference:       {abs(power_analytical - power_sim):.3f}")
```

```python
# Validate across multiple sample sizes
test_sizes = [100, 200, 500, 1000, 2000]
effect = 0.2

print(f"{'n_per_group':>12} {'Analytical':>12} {'Simulated':>12} {'Difference':>12}")
print("-" * 52)

for n in test_sizes:
    p_analytical = compute_power(n, effect)
    p_simulated = simulate_power(n, effect, n_simulations=3000)
    print(f"{n:>12,} {p_analytical:>12.3f} {p_simulated:>12.3f} {abs(p_analytical - p_simulated):>12.3f}")
```

## Geo Experiment Context

Power analysis for geo experiments is fundamentally different from user-level A/B tests:

- **Fewer units:** Instead of millions of users, we have tens or hundreds of geos (cities, DMAs, states).
- **Longer time series:** Each geo provides many time-period observations, which helps but introduces autocorrelation.
- **Higher variance:** Between-geo variation is typically much larger than between-user variation.
- **Variance estimation from pre-period data:** We use historical data to estimate the noise level.

The key implication: geo experiments can only detect **large** effects (typically 10-30% lifts), unlike user-level tests that can detect 1-2% lifts.

```python
# Load GeoLift pre-test data and estimate between-geo variance
pre_data = pd.read_csv('../../data/GeoLift_PreTest.csv')
print(f"Pre-test data shape: {pre_data.shape}")
print(f"Columns: {list(pre_data.columns)}")
print(f"Number of geos: {pre_data['location'].nunique()}")
print(f"Time periods: {pre_data['time'].nunique()}")
print()

# Compute between-geo standard deviation
geo_means = pre_data.groupby('location')['Y'].mean()
geo_std = geo_means.std()
print(f"Mean outcome across geos: {geo_means.mean():.2f}")
print(f"Between-geo std dev: {geo_std:.2f}")
print(f"CV (coefficient of variation): {geo_std / geo_means.mean():.2%}")
print()

# How many geos would we need per group?
mde_geo = 0.1 * geo_means.mean()  # 10% lift
n_geos = compute_sample_size(mde=mde_geo, sigma=geo_std)
print(f"To detect a 10% lift (MDE = {mde_geo:.2f}):")
print(f"  Geos needed per group: {n_geos}")
print(f"  Total geos needed: {2 * n_geos}")
```

```python
# Visualize geo-level variation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribution of geo means
axes[0].hist(geo_means.values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(geo_means.mean(), color='red', linestyle='--', label=f'Grand mean: {geo_means.mean():.1f}')
axes[0].set_xlabel('Mean Outcome (Y)')
axes[0].set_ylabel('Number of Geos')
axes[0].set_title('Distribution of Geo-Level Means')
axes[0].legend()

# Time series for a few geos
sample_geos = geo_means.nlargest(3).index.tolist() + geo_means.nsmallest(3).index.tolist()
for geo in sample_geos:
    geo_ts = pre_data[pre_data['location'] == geo].sort_values('time')
    axes[1].plot(geo_ts['time'], geo_ts['Y'], label=geo, alpha=0.7)

axes[1].set_xlabel('Time Period')
axes[1].set_ylabel('Outcome (Y)')
axes[1].set_title('Time Series for Selected Geos')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

## Duration Planning

Experiment duration is a critical design choice that affects power in several ways:

1. **More data points per geo:** Longer duration means more time-period observations per geo, reducing the standard error of the estimated treatment effect.

2. **Weekly vs daily aggregation:** Aggregating to weekly data reduces noise from day-of-week effects but gives fewer data points. Daily data has more noise but more observations.

3. **Seasonality and novelty effects:** Too short may miss weekly cycles or capture only novelty effects. Too long is costly and delays decisions.

**Minimum duration recommendations:**
- User-level A/B tests: at least 1-2 full weeks (to capture weekly patterns)
- Geo experiments: at least 4-8 weeks (to accumulate enough signal)
- Budget optimization experiments: at least 2-4 weeks (to allow ad platforms to optimize)

```python
# How does experiment duration affect power in a geo experiment?

def geo_power_by_duration(n_geos_per_group, true_effect_pct, weekly_std, 
                          durations_weeks, alpha=0.05, n_sims=2000):
    """
    Simulate power for a geo experiment at different durations.
    Assumes weekly observations and between-geo + within-geo variance.
    """
    results = []
    baseline_mean = 100  # arbitrary baseline
    true_effect = baseline_mean * true_effect_pct
    
    for weeks in durations_weeks:
        rejections = 0
        for _ in range(n_sims):
            # Generate geo-level random effects
            control_effects = np.random.normal(0, weekly_std, (n_geos_per_group, weeks))
            treatment_effects = np.random.normal(true_effect, weekly_std, (n_geos_per_group, weeks))
            
            # Average over weeks for each geo
            control_means = control_effects.mean(axis=1)
            treatment_means = treatment_effects.mean(axis=1)
            
            _, p_value = stats.ttest_ind(control_means, treatment_means)
            if p_value < alpha:
                rejections += 1
        
        results.append({'weeks': weeks, 'power': rejections / n_sims})
    
    return pd.DataFrame(results)

# Simulate
durations = [1, 2, 3, 4, 6, 8, 10, 12]
geo_counts = [5, 10, 15]

plt.figure(figsize=(10, 6))

for n_geo in geo_counts:
    results = geo_power_by_duration(
        n_geos_per_group=n_geo,
        true_effect_pct=0.10,  # 10% lift
        weekly_std=20,
        durations_weeks=durations
    )
    plt.plot(results['weeks'], results['power'], marker='o', label=f'{n_geo} geos/group', linewidth=2)

plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='80% power')
plt.xlabel('Experiment Duration (weeks)')
plt.ylabel('Statistical Power')
plt.title('Geo Experiment Power vs Duration (10% true effect)')
plt.legend()
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
```

---

## Exercise: Compute Sample Size for Your Business Scenario

Choose a realistic business scenario and work through the power analysis.

**Scenario ideas:**
- A/B test for a new checkout flow (baseline conversion = 3%)
- Email campaign test (baseline open rate = 20%)
- Geo experiment for TV advertising (baseline weekly sales = $50K per market)
- Pricing experiment (baseline AOV = $45)

```python
# TODO: Define your scenario
# What is the baseline metric?
# What is the minimum lift you care about?
# What alpha and power do you want?

# baseline = ...
# mde_relative = ...  # e.g., 0.05 for 5% lift
# mde_absolute = baseline * mde_relative
# sigma = ...  # estimate or compute from data
# alpha = 0.05
# power = 0.80
```

```python
# TODO: Compute the required sample size
# n = compute_sample_size(mde=mde_absolute, sigma=sigma, alpha=alpha, power=power)
# print(f"Required sample size per group: {n:,}")

# TODO: How long would this take given your daily traffic or number of available geos?
# daily_traffic = ...
# days_needed = 2 * n / daily_traffic
# print(f"Days needed: {days_needed:.0f}")
```

```python
# TODO: Create a power curve for your scenario
# Show how sample size changes if you relax the MDE
# What is the trade-off between precision and feasibility?
```

```python
# TODO: Validate with simulation
# Use simulate_power() to confirm your analytical calculation
# Do they match? If not, why?
```

