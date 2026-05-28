# Notebook 03 Pymc Basics

# Notebook 3: Introduction to Bayesian Inference with PyMC

**Marketing Science Bootcamp -- Week 2 Offline**

---

This notebook introduces the **Bayesian approach** to statistical modeling using PyMC. We start from first principles and build up to Bayesian linear regression -- the foundation for Bayesian MMM.

```python
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Plotting defaults
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True
az.style.use("arviz-darkgrid")

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")
print("Setup complete.")
```

---

## Section 1: Bayesian Thinking Review

### The Core Idea

In Bayesian inference, we update our **beliefs** about parameters using data:

$$
\underbrace{P(\theta | \text{data})}_{\text{Posterior}} \propto \underbrace{P(\text{data} | \theta)}_{\text{Likelihood}} \times \underbrace{P(\theta)}_{\text{Prior}}
$$

- **Prior** $P(\theta)$: What we believe about the parameter *before* seeing data.
- **Likelihood** $P(\text{data} | \theta)$: How probable the observed data is given a particular parameter value.
- **Posterior** $P(\theta | \text{data})$: Our updated belief *after* seeing the data.

### Why Bayesian for MMM?

1. **Uncertainty quantification**: We get full distributions, not just point estimates.
2. **Prior knowledge**: We can encode domain expertise (e.g., "TV should have a positive effect").
3. **Regularization**: Priors act as regularizers, reducing overfitting.
4. **Small data**: Bayesian methods handle small samples better than frequentist approaches.

### Example: Estimating a Conversion Rate

Suppose we run an A/B test. Out of 100 visitors, 60 convert. What is the true conversion rate?

- **Prior**: Beta(1, 1) = Uniform(0, 1) -- we have no prior opinion.
- **Likelihood**: Binomial(n=100, p=?) with 60 successes observed.
- **Posterior**: Beta(1 + 60, 1 + 40) = Beta(61, 41) -- the posterior is analytically known for the Beta-Binomial conjugate pair.

---

## Section 2: First Bayesian Model -- Coin Flip

Let's build this Beta-Binomial model in PyMC and compare the sampled posterior to the analytical solution.

```python
# Bayesian coin flip model
with pm.Model() as coin_model:
    # Prior: uniform over [0, 1]
    p = pm.Beta("p", alpha=1, beta=1)

    # Likelihood: 60 successes out of 100 trials
    obs = pm.Binomial("obs", n=100, p=p, observed=60)

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=RANDOM_SEED)

print("Sampling complete.")
```

```python
# Trace plot: shows sampling chains and posterior distribution
az.plot_trace(trace, var_names=["p"])
plt.tight_layout()
plt.show()
```

```python
# Summary statistics
summary = az.summary(trace, var_names=["p"])
print(summary)

# Compare with analytical posterior: Beta(61, 41)
analytical_mean = 61 / (61 + 41)
analytical_std = np.sqrt(61 * 41 / ((61 + 41)**2 * (61 + 41 + 1)))
print(f"\nAnalytical posterior: Beta(61, 41)")
print(f"  Mean = {analytical_mean:.4f}")
print(f"  Std  = {analytical_std:.4f}")
print(f"\nPyMC posterior:")
print(f"  Mean = {summary['mean'].values[0]:.4f}")
print(f"  Std  = {summary['sd'].values[0]:.4f}")
print("\nThey should be very close!")
```

---

## Section 3: Bayesian Linear Regression

Now let's build something closer to an MMM. We generate synthetic data from a known linear relationship and recover the parameters using Bayesian inference.

**True model:** $y = 3x + 2 + \varepsilon$, where $\varepsilon \sim N(0, 1)$

```python
# Generate synthetic data
n = 100
true_intercept = 2.0
true_slope = 3.0
true_sigma = 1.0

x = np.random.uniform(0, 10, size=n)
y = true_intercept + true_slope * x + np.random.normal(0, true_sigma, size=n)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y, alpha=0.6, s=30)
ax.plot(np.sort(x), true_intercept + true_slope * np.sort(x), "r-", linewidth=2,
        label=f"True: y = {true_slope}x + {true_intercept}")
ax.set_title("Synthetic Data", fontsize=14, fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.tight_layout()
plt.show()
```

```python
# Bayesian linear regression model
with pm.Model() as linear_model:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    slope = pm.Normal("slope", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Expected value
    mu = intercept + slope * x

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Sample
    trace_linear = pm.sample(2000, tune=1000, cores=2, random_seed=RANDOM_SEED)

print("Sampling complete.")
```

```python
# Trace plots
az.plot_trace(trace_linear, var_names=["intercept", "slope", "sigma"])
plt.tight_layout()
plt.show()
```

```python
# Summary and r_hat check
summary_linear = az.summary(trace_linear, var_names=["intercept", "slope", "sigma"])
print(summary_linear)

print("\n--- Convergence Check ---")
for param in ["intercept", "slope", "sigma"]:
    rhat = summary_linear.loc[param, "r_hat"]
    status = "PASS" if rhat < 1.01 else "FAIL"
    print(f"  {param}: r_hat = {rhat:.4f} [{status}]")

print("\n--- True vs. Estimated ---")
print(f"  Intercept: true = {true_intercept:.2f}, estimated = {summary_linear.loc['intercept', 'mean']:.2f}")
print(f"  Slope:     true = {true_slope:.2f}, estimated = {summary_linear.loc['slope', 'mean']:.2f}")
print(f"  Sigma:     true = {true_sigma:.2f}, estimated = {summary_linear.loc['sigma', 'mean']:.2f}")
```

```python
# Compare with OLS
import statsmodels.api as sm

X_ols = sm.add_constant(x)
ols_model = sm.OLS(y, X_ols).fit()

print("OLS Results:")
print(f"  Intercept: {ols_model.params[0]:.4f}")
print(f"  Slope:     {ols_model.params[1]:.4f}")
print(f"  R-squared: {ols_model.rsquared:.4f}")

print("\nBayesian Posterior Means:")
print(f"  Intercept: {summary_linear.loc['intercept', 'mean']:.4f}")
print(f"  Slope:     {summary_linear.loc['slope', 'mean']:.4f}")

print("\nWith wide priors and enough data, Bayesian and OLS converge to similar point estimates.")
print("The Bayesian advantage: we also get full posterior distributions (uncertainty).")
```

---

## Section 4: Prior Sensitivity

What happens when we use **tight (informative) priors** vs. **wide (vague) priors**? And how does sample size interact with prior strength?

Let's re-run with a small dataset (n=10) and compare tight vs. wide priors.

```python
# Small dataset (n=10)
n_small = 10
x_small = np.random.uniform(0, 10, size=n_small)
y_small = true_intercept + true_slope * x_small + np.random.normal(0, true_sigma, size=n_small)

# Model A: Wide priors (vague)
with pm.Model() as wide_prior_model:
    intercept_w = pm.Normal("intercept", mu=0, sigma=100)
    slope_w = pm.Normal("slope", mu=0, sigma=100)
    sigma_w = pm.HalfNormal("sigma", sigma=50)
    mu_w = intercept_w + slope_w * x_small
    y_obs_w = pm.Normal("y_obs", mu=mu_w, sigma=sigma_w, observed=y_small)
    trace_wide = pm.sample(2000, tune=1000, cores=2, random_seed=RANDOM_SEED)

# Model B: Tight priors (informative, centered near truth)
with pm.Model() as tight_prior_model:
    intercept_t = pm.Normal("intercept", mu=2, sigma=1)
    slope_t = pm.Normal("slope", mu=3, sigma=0.5)
    sigma_t = pm.HalfNormal("sigma", sigma=2)
    mu_t = intercept_t + slope_t * x_small
    y_obs_t = pm.Normal("y_obs", mu=mu_t, sigma=sigma_t, observed=y_small)
    trace_tight = pm.sample(2000, tune=1000, cores=2, random_seed=RANDOM_SEED)

print("Both models sampled.")
```

```python
# Compare posteriors side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, param in enumerate(["intercept", "slope"]):
    az.plot_posterior(trace_wide, var_names=[param], ax=axes[i], color="steelblue",
                     hdi_prob=0.94, point_estimate="mean")
    axes[i].set_title(f"{param} -- Wide Prior", fontsize=13, fontweight="bold")

plt.suptitle("Wide Priors (sigma=100) with n=10", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, param in enumerate(["intercept", "slope"]):
    az.plot_posterior(trace_tight, var_names=[param], ax=axes[i], color="darkorange",
                     hdi_prob=0.94, point_estimate="mean")
    axes[i].set_title(f"{param} -- Tight Prior", fontsize=13, fontweight="bold")

plt.suptitle("Tight Priors (informed) with n=10", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()

print("Notice: with small data, tight priors pull the posterior closer to the prior.")
print("With large data, the likelihood dominates and priors matter less.")
```

```python
# Numeric comparison
summary_wide = az.summary(trace_wide, var_names=["intercept", "slope", "sigma"])
summary_tight = az.summary(trace_tight, var_names=["intercept", "slope", "sigma"])

comparison = pd.DataFrame({
    "True_Value": [true_intercept, true_slope, true_sigma],
    "Wide_Prior_Mean": summary_wide["mean"].values,
    "Wide_Prior_SD": summary_wide["sd"].values,
    "Tight_Prior_Mean": summary_tight["mean"].values,
    "Tight_Prior_SD": summary_tight["sd"].values,
}, index=["intercept", "slope", "sigma"])

print("Prior Sensitivity Comparison (n=10):")
print(comparison.round(4))
print("\nTight priors reduce uncertainty (lower SD) but may introduce bias if misspecified.")
```

---

## Section 5: Posterior Predictive Check

A **posterior predictive check (PPC)** asks: *can the model generate data that looks like our observed data?*

We sample new data from the posterior predictive distribution and compare it to the actual observations.

```python
# Posterior predictive check on the full linear model
with linear_model:
    ppc = pm.sample_posterior_predictive(trace_linear, random_seed=RANDOM_SEED)

print("Posterior predictive samples generated.")
```

```python
# PPC plot
az.plot_ppc(ppc, observed_rug=True, figsize=(12, 5))
plt.title("Posterior Predictive Check", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("The dark line is observed data. The light lines are simulated datasets.")
print("If they overlap well, the model is a good fit.")
```

---

## Exercise: Extend to Multiple Regression

Your task: build a Bayesian **multiple regression** model with two predictors.

**True model:** $y = 2 + 3 x_1 + (-1.5) x_2 + \varepsilon$

Steps:
1. Generate synthetic data with two predictors.
2. Build a PyMC model with priors on `intercept`, `beta1`, `beta2`, and `sigma`.
3. Sample and check convergence.
4. Compare posterior means with OLS estimates.

```python
# TODO: Generate synthetic data
n_multi = 100
true_b0 = 2.0
true_b1 = 3.0
true_b2 = -1.5
true_sigma_multi = 1.0

x1 = np.random.uniform(0, 10, size=n_multi)
x2 = np.random.uniform(0, 5, size=n_multi)
y_multi = true_b0 + true_b1 * x1 + true_b2 * x2 + np.random.normal(0, true_sigma_multi, size=n_multi)

print(f"Generated {n_multi} data points.")
print(f"True parameters: b0={true_b0}, b1={true_b1}, b2={true_b2}, sigma={true_sigma_multi}")
```

```python
# TODO: Build and sample from a PyMC multiple regression model
# Hint: follow the pattern from the simple linear regression above,
# but add a second slope parameter (beta2) and include x2 in the mu equation.

# with pm.Model() as multi_model:
#     intercept = pm.Normal("intercept", mu=0, sigma=10)
#     beta1 = pm.Normal("beta1", mu=0, sigma=10)
#     beta2 = pm.Normal("beta2", mu=0, sigma=10)
#     sigma = pm.HalfNormal("sigma", sigma=5)
#     mu = intercept + beta1 * x1 + beta2 * x2
#     y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_multi)
#     trace_multi = pm.sample(2000, tune=1000, cores=2, random_seed=42)

# az.summary(trace_multi, var_names=["intercept", "beta1", "beta2", "sigma"])
```

```python
# TODO: Compare with OLS
# X_ols_multi = sm.add_constant(np.column_stack([x1, x2]))
# ols_multi = sm.OLS(y_multi, X_ols_multi).fit()
# print(ols_multi.summary())
```

---

## Key Takeaways

1. **PyMC** lets us define probabilistic models and sample from their posteriors using MCMC.
2. **Trace plots** and **r_hat** are essential convergence diagnostics.
3. **Priors matter** -- especially with small data. With large data, the likelihood dominates.
4. **Posterior predictive checks** validate that the model can reproduce the observed data.
5. With wide priors, Bayesian posterior means converge to OLS point estimates.

In **Session 4**, we will apply these ideas to build a full Bayesian MMM with adstock and saturation baked directly into the model.

