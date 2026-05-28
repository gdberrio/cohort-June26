# Session 4: Baby MMM from Scratch -- Bayesian with PyMC

**Marketing Science Bootcamp -- Week 2, Live Session (2 hours)**

---

## Session Overview

In Session 3, we built a Marketing Mix Model using **OLS regression** -- a frequentist approach where
transformation parameters (theta, alpha, gamma) were found via grid search and the regression was
solved analytically. That workflow has two fundamental limitations:

1. **No uncertainty on parameters** -- OLS gives point estimates, not distributions.
2. **Disconnected estimation** -- transformation parameters are optimized separately from coefficients.

Today we rebuild that same model inside a **Bayesian framework** using PyMC. The Bayesian approach
learns *all* parameters jointly -- adstock decay, saturation shape, and channel coefficients -- in a
single probabilistic model. The output is not a single "best" answer but a full **posterior
distribution** for every parameter, giving us credible intervals on channel contributions and ROI.

### Session Agenda

| Segment | Duration | Format |
|---|---|---|
| Recap: PyMC basics Q&A | 10 min | Discussion |
| Live coding: Bayesian MMM with PyMC | 50 min | Coding walkthrough |
| Exercise: Extend the Bayesian model | 25 min | Hands-on coding |
| Comparing OLS vs Bayesian results | 15 min | Discussion + coding |
| BYOD check-in: data preparation status | 10 min | Group discussion |
| Wrap-up | 10 min | Presentation |

### Deliverable

By the end of this session you should have:

1. A **working Bayesian MMM notebook** with adstock and saturation learned jointly.
2. **Convergence diagnostics** that pass (trace plots, r_hat, posterior predictive checks).
3. A **written comparison** listing 3 key differences between your OLS and Bayesian results.

---

## 1. Setup & Data Loading

```python
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
import sys

# Add project root for utils
sys.path.insert(0, "../..")

from utils.mmm_utils import (
    adstock_geometric,
    saturation_hill,
    compute_contributions,
    create_contribution_plot,
    PARAM_RANGES,
)
from utils.eda_utils import load_workshop_data, load_config

# Plotting defaults
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True
az.style.use("arviz-darkgrid")

# Suppress excessive warnings during sampling
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytensor")

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"PyMC version:  {pm.__version__}")
print(f"ArviZ version: {az.__version__}")
print("Setup complete.")
```

```python
# Load data and configuration
data = load_workshop_data("../../data/MMM_Workshop_Data.xlsx")
config = load_config("../../data/config_file.csv")

# Extract key variable lists
dv = config["dependent_var"]
time_col = config["time_column"]
paid_media = config["paid_media_spends"]
competition = config["competition_spend_vars"]
untransformed = config["untransformed_vars"]
tv_vars = config["tv_vars"]
traditional_vars = config["traditional_vars"]
atl_vars = config["atl_vars"]
digital_vars = config["digital_vars"]

y = data[dv].values.astype(float)

print(f"Dependent variable: {dv}")
print(f"Time column: {time_col}")
print(f"Paid media variables ({len(paid_media)}): {paid_media}")
print(f"Competition vars: {competition}")
print(f"Untransformed controls: {untransformed}")
print(f"\nData shape: {data.shape}")
print(f"Y ({dv}) range: [{y.min():,.0f}, {y.max():,.0f}]")
print(f"Y mean: {y.mean():,.0f}")
```

---

## 2. Recap: Key PyMC Concepts

In **Notebook 3 (offline)** you learned the fundamentals of Bayesian inference with PyMC. Here is a
quick refresher before we apply them to MMM.

### Bayes' Theorem

$$
\underbrace{P(\theta \mid \text{data})}_{\text{Posterior}} \propto
\underbrace{P(\text{data} \mid \theta)}_{\text{Likelihood}} \times
\underbrace{P(\theta)}_{\text{Prior}}
$$

### Key Concepts Checklist

| Concept | What You Should Know |
|---|---|
| **Prior** | Encodes domain knowledge *before* seeing data |
| **Likelihood** | How probable the observed data is given the parameters |
| **Posterior** | Updated belief after combining prior and likelihood |
| **MCMC Sampling** | `pm.sample()` draws from the posterior when analytical solutions are unavailable |
| **Trace plots** | Visual check that chains are "hairy caterpillars" (well-mixed) |
| **r_hat** | Gelman-Rubin convergence statistic; should be < 1.01 |
| **Posterior predictive check** | Can the fitted model generate data that looks like the real data? |
| **Prior sensitivity** | With small data, priors matter; with large data, likelihood dominates |

### Questions from Notebook 3?

*This is a good time to ask about anything that was unclear in the offline notebook.*

---

## 3. Building the Bayesian MMM

We will build the model step by step:

1. **Data preparation** -- select channels, scale data for numerical stability.
2. **Define priors** -- choose distributions for every parameter.
3. **Custom adstock** -- implement geometric decay using `pytensor.scan`.
4. **Custom saturation** -- implement the Hill function using PyTensor tensor ops.
5. **Complete model** -- assemble everything into a `pm.Model`.
6. **Sampling** -- run MCMC and check for convergence.

### 3a. Data Preparation

Bayesian models via MCMC are computationally heavier than OLS, so we start with a focused
subset of channels. We also **scale** both the dependent variable and the media data to improve
sampling efficiency. Poor scaling is one of the most common reasons for MCMC convergence failures.

```python
# Select a focused set of media channels for the Bayesian model
# We pick 5 channels to keep the model tractable during live coding.
# The exercise at the end will ask you to add more.
channel_list = [
    "TV_Spends",
    "Paid_Search_Spends",
    "Programmatic_Display_Spends",
    "Meta_Spends_Agg",
    "Youtube_Spends",
]

# Prepare media data matrix
media_data = data[channel_list].values.astype(float)
n_obs, n_channels = media_data.shape

print(f"Channels ({n_channels}): {channel_list}")
print(f"Observations: {n_obs}")
print(f"\nRaw media ranges:")
for j, ch in enumerate(channel_list):
    print(f"  {ch:<40} min={media_data[:, j].min():>12,.0f}  max={media_data[:, j].max():>12,.0f}")
```

```python
# Scale y to [0, 1] range for numerical stability
y_min, y_max = y.min(), y.max()
y_scaled = (y - y_min) / (y_max - y_min)

# Scale each media channel to [0, 1]
media_scaled = np.zeros_like(media_data)
media_scales = []  # store (min, max) for unscaling later

for j in range(n_channels):
    col_min = media_data[:, j].min()
    col_max = media_data[:, j].max()
    if col_max > col_min:
        media_scaled[:, j] = (media_data[:, j] - col_min) / (col_max - col_min)
    else:
        media_scaled[:, j] = 0.0
    media_scales.append((col_min, col_max))

print("Scaling complete.")
print(f"\ny_scaled range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
print(f"y_scaled mean:  {y_scaled.mean():.4f}")
print(f"\nMedia scaled ranges (all should be [0, 1]):")
for j, ch in enumerate(channel_list):
    print(f"  {ch:<40} [{media_scaled[:, j].min():.4f}, {media_scaled[:, j].max():.4f}]")
```

### 3b. Define Priors

Choosing priors is where we encode domain knowledge. Here is our rationale for each parameter:

| Parameter | Prior | Shape | Reasoning |
|---|---|---|---|
| **theta** (adstock decay) | `Beta(2, 2)` | Per channel | Centered at 0.5, supports the full [0,1] range. TV might warrant `Beta(3, 2)` (higher decay). |
| **alpha** (Hill steepness) | `1 + Gamma(2, 1)` | Per channel | Keeps `alpha > 1`, which gives a smooth S-curve and avoids singular gradients at zero media spend. Mode around 2, with room for steeper curves. |
| **gamma** (Hill inflection) | `Beta(2, 2)` | Per channel | Centered at 0.5 and bounded [0,1]. Controls where diminishing returns kick in. |
| **beta_media** (channel effect) | `HalfNormal(0.5)` | Per channel | **Positive only** -- media should help sales. This is a key advantage over OLS. |
| **intercept** | `Normal(0.5, 0.5)` | Scalar | Centered near y_scaled mean (~0.5). Represents baseline sales without media. |
| **sigma** (noise) | `HalfNormal(0.2)` | Scalar | Positive. Scaled data has range [0,1], so noise should be small. |

**Why constrain alpha above 1?** The Hill transformation includes `x ** alpha`. This dataset has real
zero-spend periods after scaling. When `0 < alpha < 1`, the gradient of `x ** alpha` at `x = 0` is
singular, which can make NUTS collapse into divergent transitions. We still learn `alpha` from the
data, but we parameterize it as `1 + alpha_offset` so the sampler stays in a numerically stable region.

**Why HalfNormal for beta_media?** In OLS, media coefficients can go negative (which makes no business
sense -- more spending leading to fewer sales). The `HalfNormal` prior constrains coefficients to be
non-negative, encoding the prior belief that *own media should have a positive effect on sales*.

```python
# Visualize the prior distributions to build intuition
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

from scipy import stats as sp_stats

# theta ~ Beta(2, 2)
x_grid = np.linspace(0, 1, 200)
axes[0, 0].plot(x_grid, sp_stats.beta.pdf(x_grid, 2, 2), linewidth=2, color="steelblue")
axes[0, 0].set_title("theta ~ Beta(2, 2)\n(Adstock Decay)", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("theta")
axes[0, 0].fill_between(x_grid, sp_stats.beta.pdf(x_grid, 2, 2), alpha=0.2, color="steelblue")

# alpha = 1 + Gamma(2, 1)
x_grid2 = np.linspace(1, 10, 200)
alpha_density = sp_stats.gamma.pdf(x_grid2 - 1, 2, scale=1)
axes[0, 1].plot(x_grid2, alpha_density, linewidth=2, color="darkorange")
axes[0, 1].set_title("alpha = 1 + Gamma(2, 1)\n(Hill Steepness)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("alpha")
axes[0, 1].fill_between(x_grid2, alpha_density, alpha=0.2, color="darkorange")
axes[0, 1].axvline(1, color="black", linestyle="--", alpha=0.5, linewidth=1)

# gamma ~ Beta(2, 2)
axes[0, 2].plot(x_grid, sp_stats.beta.pdf(x_grid, 2, 2), linewidth=2, color="seagreen")
axes[0, 2].set_title("gamma ~ Beta(2, 2)\n(Hill Inflection)", fontsize=12, fontweight="bold")
axes[0, 2].set_xlabel("gamma")
axes[0, 2].fill_between(x_grid, sp_stats.beta.pdf(x_grid, 2, 2), alpha=0.2, color="seagreen")

# beta_media ~ HalfNormal(0.5)
x_grid3 = np.linspace(0, 2, 200)
axes[1, 0].plot(x_grid3, sp_stats.halfnorm.pdf(x_grid3, scale=0.5), linewidth=2, color="crimson")
axes[1, 0].set_title("beta_media ~ HalfNormal(0.5)\n(Channel Effect)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("beta")
axes[1, 0].fill_between(x_grid3, sp_stats.halfnorm.pdf(x_grid3, scale=0.5), alpha=0.2, color="crimson")

# intercept ~ Normal(0.5, 0.5)
x_grid4 = np.linspace(-1, 2, 200)
axes[1, 1].plot(x_grid4, sp_stats.norm.pdf(x_grid4, loc=0.5, scale=0.5), linewidth=2, color="purple")
axes[1, 1].set_title("intercept ~ Normal(0.5, 0.5)\n(Baseline Sales)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("intercept")
axes[1, 1].fill_between(x_grid4, sp_stats.norm.pdf(x_grid4, loc=0.5, scale=0.5), alpha=0.2, color="purple")

# sigma ~ HalfNormal(0.2)
x_grid5 = np.linspace(0, 1, 200)
axes[1, 2].plot(x_grid5, sp_stats.halfnorm.pdf(x_grid5, scale=0.2), linewidth=2, color="teal")
axes[1, 2].set_title("sigma ~ HalfNormal(0.2)\n(Noise)", fontsize=12, fontweight="bold")
axes[1, 2].set_xlabel("sigma")
axes[1, 2].fill_between(x_grid5, sp_stats.halfnorm.pdf(x_grid5, scale=0.2), alpha=0.2, color="teal")

fig.suptitle("Prior Distributions for Bayesian MMM Parameters", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("These priors encode our domain knowledge:")
print("  - theta: agnostic about decay rate, centered at 0.5")
print("  - alpha: expect an S-curve with alpha > 1 for stable gradients at zero spend")
print("  - gamma: agnostic about inflection point")
print("  - beta_media: MUST be positive (media helps sales)")
print("  - intercept: centered near scaled y mean")
print("  - sigma: expect small noise relative to scaled data")
```

### 3c. Custom Adstock with PyTensor Scan

In Session 3 (OLS), our `adstock_geometric()` function used a plain Python `for` loop over NumPy arrays.
That works for pre-computing transformations, but it **cannot** be used inside a PyMC model because
PyMC needs to compute gradients through all operations.

We need to rewrite the adstock using **`pytensor.scan`**, which is PyTensor's loop construct.
It works like `functools.reduce` -- at each time step, the function receives the current input and
the previous output.

**Geometric adstock formula (recap):**

$$
x_t^{\text{adstocked}} = x_t + \theta \cdot x_{t-1}^{\text{adstocked}}
$$

```python
def adstock_geometric_pytensor(x, theta):
    """Apply geometric adstock using pytensor.scan.

    Parameters
    ----------
    x : pytensor tensor, shape (n_obs,)
        Input media series (scaled).
    theta : pytensor scalar
        Decay rate in [0, 1]. Higher = slower decay.

    Returns
    -------
    pytensor tensor, shape (n_obs,) -- adstocked series.

    Notes
    -----
    scan iterates over the sequence `x` and, at each step, calls `step(x_t, x_prev, theta)`.
    `outputs_info` provides the initial value for x_prev (zero -- no prior advertising).
    """
    def step(x_t, x_prev, theta):
        return x_t + theta * x_prev

    from pytensor.scan.basic import scan as pt_scan
    result, _ = pt_scan(
        fn=step,
        sequences=[x],              # x_t comes from this sequence
        outputs_info=[pt.zeros(())], # x_prev starts at 0
        non_sequences=[theta],       # theta is constant across steps
    )
    return result


print("adstock_geometric_pytensor defined.")
print("\nThis function:")
print("  1. Takes a media series x and decay rate theta as pytensor tensors.")
print("  2. Uses scan to loop: x_decayed[t] = x[t] + theta * x_decayed[t-1].")
print("  3. Returns a fully differentiable pytensor tensor (PyMC can compute gradients).")
```

```python
# Verify: the pytensor adstock should match our NumPy adstock_geometric()
# We test with a known theta on the first channel
test_theta = 0.5
test_x = media_scaled[:, 0]  # TV_Spends scaled

# NumPy version (from mmm_utils)
numpy_result = adstock_geometric(test_x, test_theta)["x_decayed"]

# PyTensor version (compile and evaluate)
x_sym = pt.dvector("x")
theta_sym = pt.dscalar("theta")
adstocked_sym = adstock_geometric_pytensor(x_sym, theta_sym)

# Compile a function to evaluate the symbolic expression
import pytensor
adstock_fn = pytensor.function([x_sym, theta_sym], adstocked_sym)
pytensor_result = adstock_fn(test_x, test_theta)

# Compare
max_diff = np.max(np.abs(numpy_result - pytensor_result))
print(f"Max absolute difference between NumPy and PyTensor adstock: {max_diff:.2e}")
assert max_diff < 1e-10, "Mismatch between implementations!"
print("PASS: Both implementations agree.")
```

### 3d. Custom Saturation (Hill Function)

The Hill saturation function uses standard tensor operations (power, min, max), so it does **not**
need `scan`. We simply translate the NumPy formula into PyTensor:

$$
\text{saturation}(x) = \frac{x^\alpha}{x^\alpha + \text{inflexion}^\alpha}
$$

where $\text{inflexion} = \min(x) \cdot (1 - \gamma) + \max(x) \cdot \gamma$.

```python
def hill_saturation_pytensor(x, alpha, gamma):
    """Hill saturation transformation using pytensor ops.

    Parameters
    ----------
    x : pytensor tensor
        Input series (typically adstocked, shape (n_obs,)).
    alpha : pytensor scalar
        Shape parameter controlling curve steepness.
    gamma : pytensor scalar
        Parameter in [0, 1] controlling inflection point position.

    Returns
    -------
    pytensor tensor -- saturated series in [0, 1].
    """
    inflexion = pt.min(x) * (1 - gamma) + pt.max(x) * gamma
    return x ** alpha / (x ** alpha + inflexion ** alpha)


print("hill_saturation_pytensor defined.")
print("\nThis function:")
print("  1. Computes the inflection point from gamma and the data range.")
print("  2. Applies the Hill formula using PyTensor power ops.")
print("  3. Outputs values in [0, 1] -- fully saturated at 1.")
```

```python
# Verify: the pytensor Hill function should match our NumPy saturation_hill()
test_alpha = 2.0
test_gamma = 0.5

# NumPy version
numpy_hill = saturation_hill(numpy_result, test_alpha, test_gamma)

# PyTensor version
alpha_sym = pt.dscalar("alpha")
gamma_sym = pt.dscalar("gamma")
hill_sym = hill_saturation_pytensor(adstocked_sym, alpha_sym, gamma_sym)
hill_fn = pytensor.function([x_sym, theta_sym, alpha_sym, gamma_sym], hill_sym)
pytensor_hill = hill_fn(test_x, test_theta, test_alpha, test_gamma)

max_diff_hill = np.max(np.abs(numpy_hill - pytensor_hill))
print(f"Max absolute difference between NumPy and PyTensor Hill: {max_diff_hill:.2e}")
assert max_diff_hill < 1e-10, "Mismatch between Hill implementations!"
print("PASS: Both Hill implementations agree.")
```

### 3e. Complete Model Specification

Now we assemble the full Bayesian MMM. For **each channel**, the model:

1. Applies **geometric adstock** with a learnable `theta[j]`.
2. Applies **Hill saturation** with learnable `alpha[j]` and `gamma[j]`.
3. Multiplies the saturated signal by a **channel coefficient** `beta_media[j]`.

Then it sums all channel contributions, adds an `intercept`, and defines a **Normal likelihood**.

```
y_scaled ~ Normal(mu, sigma)

where:
    mu = intercept + sum_j( beta_media[j] * Hill(Adstock(media[j], theta[j]), alpha[j], gamma[j]) )
```

```python
# Convert media data to a shared pytensor variable
media_pt = pt.as_tensor_variable(media_scaled)

with pm.Model() as bayesian_mmm:

    # === PRIORS: Transformation parameters (per channel) ===
    theta = pm.Beta("theta", alpha=2, beta=2, shape=n_channels)              # adstock decay [0, 1]
    alpha_offset = pm.Gamma("alpha_offset", alpha=2, beta=1, shape=n_channels)  # positive offset
    alpha = pm.Deterministic("alpha", 1 + alpha_offset)                      # Hill steepness > 1
    gamma = pm.Beta("gamma", alpha=2, beta=2, shape=n_channels)              # Hill inflection [0, 1]

    # === PRIORS: Regression coefficients ===
    beta_media = pm.HalfNormal("beta_media", sigma=0.5, shape=n_channels)  # positive only
    intercept = pm.Normal("intercept", mu=0.5, sigma=0.5)                  # baseline sales
    sigma = pm.HalfNormal("sigma", sigma=0.2)                              # observation noise

    # === TRANSFORMATIONS: Apply adstock + saturation per channel ===
    contributions = []
    for j in range(n_channels):
        # Step 1: Geometric adstock
        x_adstocked = adstock_geometric_pytensor(media_pt[:, j], theta[j])

        # Step 2: Hill saturation
        x_saturated = hill_saturation_pytensor(x_adstocked, alpha[j], gamma[j])

        # Step 3: Scale by channel coefficient
        contributions.append(beta_media[j] * x_saturated)

    # === LINEAR COMBINATION ===
    mu = intercept + pt.sum(pt.stack(contributions, axis=0), axis=0)

    # === LIKELIHOOD ===
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

    print("Model specification complete.")
    print(f"\nModel parameters:")
    print(f"  theta:       {n_channels} params (adstock decay per channel)")
    print(f"  alpha:       {n_channels} deterministic params (1 + positive offset)")
    print(f"  gamma:       {n_channels} params (Hill inflection per channel)")
    print(f"  beta_media:  {n_channels} params (channel effects)")
    print(f"  intercept:   1 param")
    print(f"  sigma:       1 param")
    print(f"  TOTAL:       {4 * n_channels + 2} free parameters")
```

```python
# Visualize the model graph (requires graphviz)
try:
    graph = pm.model_to_graphviz(bayesian_mmm)
    display(graph)
except Exception as e:
    print(f"Could not render model graph (graphviz may not be installed): {e}")
    print("\nModel structure (text):")
    print(bayesian_mmm)
```

### 3f. Sampling

Now we run MCMC sampling. Key arguments:

| Argument | Value | Meaning |
|---|---|---|
| `draws` | 2000 | Number of posterior samples per chain |
| `tune` | 1000 | Warm-up samples (discarded) -- sampler adapts step size |
| `cores` | 4 | Number of parallel chains |
| `target_accept` | 0.99 | Higher = smaller step size = fewer divergences but slower |
| `random_seed` | 42 | Reproducibility |

**Expect this to take 5-15 minutes** depending on your machine. The `scan` operations for adstock
are sequential by nature, so this model is slower than a standard regression.

**Debugging convergence issues:**
- If you see many **divergent transitions**, first check the model geometry: exact zeros plus `alpha < 1` can create singular gradients in the Hill transform.
- We use `target_accept=0.99` here because the Hill transformation creates a curved posterior geometry; this is slower but more reliable for teaching.
- If chains get **stuck**, check your priors -- they may be too tight or conflicting with data.
- If r_hat > 1.01, try increasing `draws` and `tune`.
- If sampling is very slow, reduce the number of channels or observations.

```python
%%time

with bayesian_mmm:
    trace_mmm = pm.sample(
        draws=2000,
        tune=1000,
        cores=4,
        random_seed=RANDOM_SEED,
        target_accept=0.99,
        return_inferencedata=True,
    )

print("\nSampling complete.")
print(f"Trace shape: {trace_mmm.posterior.dims}")

# Check for divergences
n_divergences = trace_mmm.sample_stats["diverging"].sum().values
print(f"\nNumber of divergent transitions: {n_divergences}")
if n_divergences > 0:
    pct = n_divergences / (2000 * 4) * 100
    print(f"  ({pct:.1f}% of total samples -- consider increasing target_accept)")
else:
    print("  No divergences -- sampling looks healthy.")
```

### 3g. What Caused the Divergences?

If you previously ran this notebook and saw every posterior draw diverge, the problem was not simply
that the sampler needed more patience. The model geometry was broken in a specific place.

The Hill saturation function contains this term:

$$
x^{lpha}
$$

Our media inputs are min-max scaled to `[0, 1]`, and several channels have real zero-spend periods.
For example, a month with no YouTube spend becomes `0.0` after scaling. That is fine for the model
value: `0 ** alpha` is still zero for positive `alpha`.

The issue is the **gradient** that NUTS needs in order to sample efficiently. When the old prior allowed
`0 < alpha < 1`, the derivative of `x ** alpha` at `x = 0` becomes singular. In plain language: the
curve has a sharp, nearly vertical edge at zero. NUTS tries to follow the posterior geometry through
that edge, its step size collapses, and the result is a wall of divergent transitions.

That is why simply increasing `target_accept` was not enough. A higher `target_accept` asks NUTS to use
smaller steps, but it does not remove the singular-gradient region. The sampler was being cautious in
a geometry that was still fundamentally hard to traverse.

The fix is to parameterize Hill steepness as:

```python
alpha_offset = pm.Gamma("alpha_offset", alpha=2, beta=1, shape=n_channels)
alpha = pm.Deterministic("alpha", 1 + alpha_offset)
```

This keeps `alpha > 1`. The Hill curve can still learn different saturation shapes, but it avoids the
problematic `alpha < 1` region where zero media spend creates unstable gradients. We also use
`target_accept=0.99` because this posterior is still curved and mildly difficult; that setting makes
NUTS take smaller, more reliable steps after the model geometry has been fixed.

So the lesson is:

1. Divergences are not just warnings to silence. They tell us the sampler found geometry it could not
   explore reliably.
2. The data mattered: exact zeros in scaled media channels exposed the problem.
3. The prior mattered: allowing `alpha < 1` made the Hill transform numerically awkward at those zeros.
4. The fix works by changing the parameterization, not by changing the observed data or hiding the
   divergences.

---

## 4. Posterior Analysis

Now comes the payoff: we examine what the model learned. The key diagnostic checks are:

1. **Trace plots** -- visual check that chains are well-mixed.
2. **Parameter summary** -- posterior means, HDI, r_hat.
3. **Posterior predictive check** -- can the fitted model reproduce the observed data?

### 4a. Trace Plots & Convergence

Each parameter should show:
- **Left panel (KDE):** All chains should overlap (same distribution).
- **Right panel (trace):** Should look like a "hairy caterpillar" -- well-mixed, no trends or drift.

```python
# Trace plots for transformation parameters
az.plot_trace(
    trace_mmm,
    var_names=["theta", "alpha", "gamma"],
    compact=True,
    figsize=(14, 10),
)
plt.suptitle("Transformation Parameters: Trace Plots", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("Look for:")
print("  - KDE plots: all chains overlap (same distribution)")
print("  - Trace plots: no trends, no stuck chains")
```

```python
# Trace plots for regression parameters
az.plot_trace(
    trace_mmm,
    var_names=["beta_media", "intercept", "sigma"],
    compact=True,
    figsize=(14, 8),
)
plt.suptitle("Regression Parameters: Trace Plots", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
```

### 4b. Parameter Summary

The `az.summary()` table gives us:
- **mean**: Posterior mean (analogous to OLS point estimate).
- **sd**: Posterior standard deviation (uncertainty on the parameter).
- **hdi_3%** / **hdi_97%**: 94% Highest Density Interval (credible interval).
- **r_hat**: Gelman-Rubin convergence diagnostic. Must be < 1.01.
- **ess_bulk** / **ess_tail**: Effective sample size. Should be > 400.

```python
# Full summary table
summary_mmm = az.summary(
    trace_mmm,
    var_names=["theta", "alpha", "gamma", "beta_media", "intercept", "sigma"],
    round_to=4,
)
print("Posterior Summary:")
print("=" * 100)
print(summary_mmm.to_string())
```

```python
# Convergence check: r_hat and ESS
print("\n" + "=" * 60)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 60)

# r_hat check
rhat_vals = summary_mmm["r_hat"]
rhat_ok = (rhat_vals < 1.01).all()
print(f"\nr_hat < 1.01 for ALL parameters: {'PASS' if rhat_ok else 'FAIL'}")
if not rhat_ok:
    print("  Parameters that failed:")
    print(summary_mmm[rhat_vals >= 1.01][["r_hat"]])
else:
    print(f"  Max r_hat: {rhat_vals.max():.4f}")

# ESS check
ess_bulk_vals = summary_mmm["ess_bulk"]
ess_ok = (ess_bulk_vals > 400).all()
print(f"\nESS_bulk > 400 for ALL parameters: {'PASS' if ess_ok else 'WARNING'}")
if not ess_ok:
    print("  Parameters with low ESS:")
    print(summary_mmm[ess_bulk_vals <= 400][["ess_bulk"]])
else:
    print(f"  Min ESS_bulk: {ess_bulk_vals.min():.0f}")

# Divergences
n_div = trace_mmm.sample_stats["diverging"].sum().values
print(f"\nDivergent transitions: {n_div}")
```

```python
# Energy plot -- another diagnostic for sampling health
az.plot_energy(trace_mmm, figsize=(10, 5))
plt.title("Energy Plot (marginal vs. transition energy)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("If the two distributions overlap well, sampling is healthy.")
print("Large gaps indicate sampling difficulties.")
```

```python
# Extract and display posterior means in a clean table
theta_post = trace_mmm.posterior["theta"].mean(dim=["chain", "draw"]).values
alpha_post = trace_mmm.posterior["alpha"].mean(dim=["chain", "draw"]).values
gamma_post = trace_mmm.posterior["gamma"].mean(dim=["chain", "draw"]).values
beta_post = trace_mmm.posterior["beta_media"].mean(dim=["chain", "draw"]).values
intercept_post = float(trace_mmm.posterior["intercept"].mean(dim=["chain", "draw"]).values)
sigma_post = float(trace_mmm.posterior["sigma"].mean(dim=["chain", "draw"]).values)

print("Posterior Mean Estimates:")
print("=" * 75)
print(f"{'Channel':<35} {'theta':>7} {'alpha':>7} {'gamma':>7} {'beta':>9}")
print("-" * 75)
for j, ch in enumerate(channel_list):
    print(f"{ch:<35} {theta_post[j]:>7.3f} {alpha_post[j]:>7.3f} {gamma_post[j]:>7.3f} {beta_post[j]:>9.4f}")
print("-" * 75)
print(f"{'Intercept':<35} {'':>7} {'':>7} {'':>7} {intercept_post:>9.4f}")
print(f"{'Sigma (noise)':<35} {'':>7} {'':>7} {'':>7} {sigma_post:>9.4f}")
```

### 4c. Posterior Predictive Check

A **posterior predictive check (PPC)** answers the question: *can this fitted model generate data
that looks like our observed data?*

We draw samples from the posterior predictive distribution and compare them to the actual `y_scaled`.

```python
# Generate posterior predictive samples
with bayesian_mmm:
    ppc = pm.sample_posterior_predictive(trace_mmm, random_seed=RANDOM_SEED)

print("Posterior predictive samples generated.")
```

```python
# PPC distribution plot
az.plot_ppc(ppc, observed_rug=True, figsize=(12, 5))
plt.title("Posterior Predictive Check: Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("Dark line = observed data distribution.")
print("Light lines = simulated datasets from the posterior.")
print("Good overlap means the model captures the data-generating process well.")
```

```python
# PPC as a time-series overlay: posterior mean prediction vs. actuals
ppc_samples = ppc.posterior_predictive["y_obs"].values  # shape: (chain, draw, n_obs)
ppc_flat = ppc_samples.reshape(-1, n_obs)               # flatten chains x draws

ppc_mean = ppc_flat.mean(axis=0)
ppc_lower = np.percentile(ppc_flat, 2.5, axis=0)
ppc_upper = np.percentile(ppc_flat, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(16, 6))
time_idx = np.arange(n_obs)

ax.fill_between(time_idx, ppc_lower, ppc_upper, alpha=0.3, color="steelblue",
                label="95% Credible Interval")
ax.plot(time_idx, ppc_mean, color="steelblue", linewidth=2, label="Posterior Mean Prediction")
ax.plot(time_idx, y_scaled, color="black", linewidth=2, linestyle="--", label="Observed (scaled)")

ax.set_title("Posterior Predictive: Actual vs. Predicted (Time Series)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Time Period", fontsize=12)
ax.set_ylabel("Sales (scaled)", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Compute R-squared equivalent
ss_res = np.sum((y_scaled - ppc_mean) ** 2)
ss_tot = np.sum((y_scaled - y_scaled.mean()) ** 2)
r_squared_bayes = 1 - ss_res / ss_tot
mape_bayes = np.mean(np.abs((y_scaled - ppc_mean) / y_scaled))

print(f"\nBayesian model fit:")
print(f"  R-squared (posterior mean): {r_squared_bayes:.4f}")
print(f"  MAPE (posterior mean):      {mape_bayes:.4%}")
```

---

## 5. Contribution Analysis (Bayesian)

In OLS (Session 3), we computed a single contribution per channel using the Beta x Mean approach.
The Bayesian model gives us something much richer: a **distribution of contributions** for each
channel, reflecting parameter uncertainty.

We will:
1. Compute contributions using posterior mean parameters (point estimate).
2. Compute contribution distributions by iterating over posterior samples.
3. Derive credible intervals on ROI.

### 5a. Extract Posterior Contributions

```python
# Compute contributions using posterior means (point estimate)
contributions_np = np.zeros((n_obs, n_channels))
for j in range(n_channels):
    x_ad = adstock_geometric(media_scaled[:, j], theta_post[j])["x_decayed"]
    x_sat = saturation_hill(x_ad, alpha_post[j], gamma_post[j])
    contributions_np[:, j] = beta_post[j] * x_sat

# Total contribution per channel (summed over time)
total_contrib = contributions_np.sum(axis=0)
intercept_total = intercept_post * n_obs
total_all = total_contrib.sum() + intercept_total
contrib_pct = total_contrib / total_all * 100
intercept_pct = intercept_total / total_all * 100

# Build contribution summary table
contrib_summary = pd.DataFrame({
    "Channel": list(channel_list) + ["Base (intercept)"],
    "Total_Contribution_Scaled": list(total_contrib) + [intercept_total],
    "Contribution_Pct": list(contrib_pct) + [intercept_pct],
})
contrib_summary = contrib_summary.sort_values("Contribution_Pct", ascending=False).reset_index(drop=True)

print("Channel Contributions (Bayesian MMM -- Posterior Means):")
print("=" * 65)
print(contrib_summary.to_string(index=False, float_format="{:.2f}".format))
print(f"\nTotal: {total_all:.4f} (scaled units)")
```

```python
# Contribution bar chart
fig, ax = plt.subplots(figsize=(12, 6))

df_plot = contrib_summary.sort_values("Contribution_Pct", ascending=True)
colors = ["#002060" if ch != "Base (intercept)" else "#808080" for ch in df_plot["Channel"]]

ax.barh(df_plot["Channel"], df_plot["Contribution_Pct"], color=colors, height=0.7)

for idx, (val, ch) in enumerate(zip(df_plot["Contribution_Pct"], df_plot["Channel"])):
    ha = "left" if val > 0 else "right"
    offset = 0.3 if val > 0 else -0.3
    ax.text(val + offset, idx, f"{val:.1f}%", va="center", ha=ha,
            fontsize=11, fontweight="bold")

ax.set_title("Bayesian MMM: Channel Contribution (%)", fontsize=16, fontweight="bold")
ax.set_xlabel("Contribution (%)", fontsize=13)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
# Compute contribution DISTRIBUTIONS by sampling from the posterior
# This is the key advantage of Bayesian: we get uncertainty on contributions

n_posterior_samples = 500  # use a subset for speed

theta_samples = trace_mmm.posterior["theta"].values.reshape(-1, n_channels)
alpha_samples = trace_mmm.posterior["alpha"].values.reshape(-1, n_channels)
gamma_samples = trace_mmm.posterior["gamma"].values.reshape(-1, n_channels)
beta_samples = trace_mmm.posterior["beta_media"].values.reshape(-1, n_channels)
intercept_samples = trace_mmm.posterior["intercept"].values.flatten()

# Randomly select posterior samples
rng = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(len(theta_samples), size=n_posterior_samples, replace=False)

# Compute contributions for each posterior sample
contrib_dist = np.zeros((n_posterior_samples, n_channels))  # shape: (samples, channels)

for s_idx, s in enumerate(sample_idx):
    for j in range(n_channels):
        x_ad = adstock_geometric(media_scaled[:, j], theta_samples[s, j])["x_decayed"]
        x_sat = saturation_hill(x_ad, alpha_samples[s, j], gamma_samples[s, j])
        contrib_dist[s_idx, j] = (beta_samples[s, j] * x_sat).sum()

print(f"Computed contribution distributions using {n_posterior_samples} posterior samples.")
print(f"Shape: {contrib_dist.shape} (samples x channels)")
```

```python
# Contribution credible intervals
# Normalize each posterior sample's contributions to percentages
intercept_contribs = intercept_samples[sample_idx] * n_obs
total_per_sample = contrib_dist.sum(axis=1) + intercept_contribs
contrib_pct_dist = contrib_dist / total_per_sample[:, None] * 100

contrib_ci = pd.DataFrame({
    "Channel": channel_list,
    "Mean_Pct": contrib_pct_dist.mean(axis=0),
    "Median_Pct": np.median(contrib_pct_dist, axis=0),
    "CI_2.5%": np.percentile(contrib_pct_dist, 2.5, axis=0),
    "CI_97.5%": np.percentile(contrib_pct_dist, 97.5, axis=0),
}).sort_values("Mean_Pct", ascending=False).reset_index(drop=True)

print("Channel Contribution Credible Intervals (95%):")
print("=" * 75)
print(contrib_ci.to_string(index=False, float_format="{:.2f}".format))
print("\nWider intervals = more uncertainty about that channel's true contribution.")
```

```python
# Forest plot for channel effects (beta_media)
fig, ax = plt.subplots(figsize=(10, 5))

az.plot_forest(
    trace_mmm,
    var_names=["beta_media"],
    combined=True,
    hdi_prob=0.94,
    ax=ax,
)
ax.set_title("Channel Effects (beta_media) -- 94% HDI", fontsize=14, fontweight="bold")

# Add channel names as y-tick labels
ax.set_yticklabels(channel_list[::-1])

plt.tight_layout()
plt.show()

print("Channels with HDI far from zero have strong evidence of a positive effect.")
print("Channels with HDI close to zero might not be contributing much.")
```

```python
# Visualize contribution distributions as violin plots
fig, ax = plt.subplots(figsize=(12, 6))

parts = ax.violinplot(contrib_pct_dist, positions=range(n_channels), showmeans=True, showmedians=True)

# Color the violins
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.6)

ax.set_xticks(range(n_channels))
ax.set_xticklabels([ch.replace("_", "\n") for ch in channel_list], fontsize=10)
ax.set_ylabel("Contribution (%)", fontsize=12)
ax.set_title("Posterior Distribution of Channel Contributions (%)",
             fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("Wider violins = more uncertainty.")
print("This uncertainty is invisible in OLS -- it only gives you a single number.")
```

### 5b. Credible Intervals on ROI

ROI is typically defined as:

$$
\text{ROI}_j = \frac{\text{Revenue attributed to channel } j}{\text{Spend on channel } j}
$$

Since we have posterior distributions of contributions, we can compute **posterior distributions of ROI**
-- something impossible with OLS.

```python
# Compute ROI distributions
# contrib_dist is in scaled units; we need to unscale to get actual sales contribution
# Unscale: actual_contribution = scaled_contribution * (y_max - y_min)

contrib_dist_actual = contrib_dist * (y_max - y_min)  # back to original sales units

# Total spend per channel
total_spend = media_data.sum(axis=0)  # shape: (n_channels,)

# ROI = revenue attributed / spend
roi_dist = contrib_dist_actual / total_spend[None, :]  # shape: (n_samples, n_channels)

roi_summary = pd.DataFrame({
    "Channel": channel_list,
    "Total_Spend": total_spend,
    "ROI_Mean": roi_dist.mean(axis=0),
    "ROI_Median": np.median(roi_dist, axis=0),
    "ROI_CI_2.5%": np.percentile(roi_dist, 2.5, axis=0),
    "ROI_CI_97.5%": np.percentile(roi_dist, 97.5, axis=0),
}).sort_values("ROI_Mean", ascending=False).reset_index(drop=True)

print("Channel ROI with 95% Credible Intervals:")
print("=" * 85)
print(roi_summary.to_string(index=False, float_format="{:.4f}".format))
print("\nInterpretation: ROI of X means $X incremental revenue per $1 spent.")
print("Wider CI = more uncertainty about the true ROI.")
```

```python
# ROI forest-style plot with credible intervals
fig, ax = plt.subplots(figsize=(12, 5))

roi_sorted = roi_summary.sort_values("ROI_Mean", ascending=True).reset_index(drop=True)

y_positions = range(len(roi_sorted))

# Error bars showing 95% CI
ax.hlines(y_positions, roi_sorted["ROI_CI_2.5%"], roi_sorted["ROI_CI_97.5%"],
          color="steelblue", linewidth=3, alpha=0.5)
ax.scatter(roi_sorted["ROI_Mean"], y_positions, color="darkblue", s=80, zorder=5,
           label="Mean ROI")
ax.scatter(roi_sorted["ROI_Median"], y_positions, color="darkorange", s=60, zorder=5,
           marker="D", label="Median ROI")

ax.set_yticks(y_positions)
ax.set_yticklabels(roi_sorted["Channel"], fontsize=11)
ax.set_xlabel("ROI (incremental sales per $ spent)", fontsize=12)
ax.set_title("Channel ROI with 95% Credible Intervals", fontsize=14, fontweight="bold")
ax.axvline(0, color="red", linestyle="--", alpha=0.5)
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

print("The horizontal bars show the 95% credible interval for each channel's ROI.")
print("Longer bars = more uncertain. Use this when making budget allocation decisions.")
```

---

## 6. Exercise: Extend the Bayesian Model (25 minutes)

Now it is your turn to modify and extend the model. Choose one or more of the following:

### Exercise A: Add a Control Variable
Add `Average_Price_Total` or `Inflation_Rate` as an untransformed control variable.
Unlike media channels, control variables **can** have negative effects (higher price = lower sales),
so use a `Normal` prior (not `HalfNormal`).

### Exercise B: Add More Channels
Add `Radio_Spends`, `Outdoor_Spends`, or `Brand_B_ATL_Spends` (competition) to the model.
For competition variables, use `pm.HalfNormal` with a negative sign in the `mu` equation
(competition spending should *hurt* your sales).

### Exercise C: Prior Sensitivity
Change the TV adstock prior from `Beta(2, 2)` to `Beta(3, 1)` to bias toward higher decay
(TV effects last longer). How does this change the theta posterior and TV's contribution?

### Exercise D: Prior Predictive Check
Use `pm.sample_prior_predictive()` to verify that your priors produce reasonable y values
*before* fitting. If the prior predictive generates values far outside the data range,
your priors need adjustment.

```python
# Exercise A: Add a control variable
# TODO: Prepare the control variable data

# control_var = "Average_Price_Total"  # or "Inflation_Rate"
# control_data = data[control_var].values.astype(float)
# control_scaled = (control_data - control_data.min()) / (control_data.max() - control_data.min())
# control_pt = pt.as_tensor_variable(control_scaled)

# TODO: Build the extended model
# with pm.Model() as extended_mmm:
#     # --- Same transformation priors as before ---
#     theta = pm.Beta("theta", alpha=2, beta=2, shape=n_channels)
#     alpha_offset = pm.Gamma("alpha_offset", alpha=2, beta=1, shape=n_channels)
#     alpha = pm.Deterministic("alpha", 1 + alpha_offset)
#     gamma = pm.Beta("gamma", alpha=2, beta=2, shape=n_channels)
#
#     # --- Regression coefficients ---
#     beta_media = pm.HalfNormal("beta_media", sigma=0.5, shape=n_channels)
#     beta_control = pm.Normal("beta_control", mu=0, sigma=1)  # can be negative!
#     intercept = pm.Normal("intercept", mu=0.5, sigma=0.5)
#     sigma = pm.HalfNormal("sigma", sigma=0.2)
#
#     # --- Transformations per channel ---
#     contributions = []
#     for j in range(n_channels):
#         x_adstocked = adstock_geometric_pytensor(media_pt[:, j], theta[j])
#         x_saturated = hill_saturation_pytensor(x_adstocked, alpha[j], gamma[j])
#         contributions.append(beta_media[j] * x_saturated)
#
#     # --- Linear combination (now includes control) ---
#     mu = intercept + pt.sum(pt.stack(contributions, axis=0), axis=0) + beta_control * control_pt
#
#     # --- Likelihood ---
#     y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)
#
#     # --- Sample ---
#     trace_extended = pm.sample(draws=2000, tune=1000, cores=4, random_seed=42, target_accept=0.99)
#
# az.summary(trace_extended, var_names=["beta_control", "beta_media", "intercept"])

print("Exercise A: Uncomment and run the code above to add a control variable.")
```

```python
# Exercise B: Add more channels
# TODO: Extend channel_list and rebuild the model

# extended_channel_list = channel_list + ["Radio_Spends", "Outdoor_Spends"]
# extended_media_data = data[extended_channel_list].values.astype(float)
# n_obs_ext, n_channels_ext = extended_media_data.shape
#
# # Scale the extended media data
# extended_media_scaled = np.zeros_like(extended_media_data)
# for j in range(n_channels_ext):
#     col_min = extended_media_data[:, j].min()
#     col_max = extended_media_data[:, j].max()
#     if col_max > col_min:
#         extended_media_scaled[:, j] = (extended_media_data[:, j] - col_min) / (col_max - col_min)
#     else:
#         extended_media_scaled[:, j] = 0.0
#
# # TODO: Build and sample the extended model (same structure, just n_channels_ext)

print("Exercise B: Uncomment and run the code above to add more channels.")
```

```python
# Exercise C: Prior sensitivity -- change TV adstock prior
# TODO: Rebuild the model with per-channel priors instead of shared priors

# Hint: Instead of a single Beta(2,2) for all channels, define separate priors:
# theta_tv = pm.Beta("theta_TV", alpha=3, beta=1)       # biased toward high decay
# theta_other = pm.Beta("theta_other", alpha=2, beta=2, shape=n_channels - 1)
# theta = pt.concatenate([[theta_tv], theta_other])

# TODO: Compare the posterior theta for TV under the two different priors.
# If the posterior is very different, the data is not informative enough to override the prior.
# If the posterior is similar, the data dominates (which is what we want).

print("Exercise C: Uncomment and modify the code to test prior sensitivity.")
```

```python
# Exercise D: Prior predictive check
# TODO: Generate samples from the prior predictive and check they are reasonable

# with bayesian_mmm:
#     prior_pred = pm.sample_prior_predictive(samples=500, random_seed=42)
#
# # Plot the prior predictive distribution of y
# prior_y = prior_pred.prior_predictive["y_obs"].values.flatten()
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.hist(prior_y, bins=100, density=True, alpha=0.6, color="steelblue", label="Prior Predictive")
# ax.axvline(y_scaled.min(), color="red", linestyle="--", label=f"Observed min ({y_scaled.min():.2f})")
# ax.axvline(y_scaled.max(), color="red", linestyle="-", label=f"Observed max ({y_scaled.max():.2f})")
# ax.set_title("Prior Predictive Check: Does the prior generate plausible data?",
#              fontsize=14, fontweight="bold")
# ax.set_xlabel("y (scaled)")
# ax.legend()
# plt.tight_layout()
# plt.show()
#
# print(f"Prior predictive y range: [{prior_y.min():.2f}, {prior_y.max():.2f}]")
# print(f"Observed y range:         [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")
# print("If the prior predictive range is wildly different, reconsider your priors.")

print("Exercise D: Uncomment and run the code to perform a prior predictive check.")
```

---

## 7. OLS vs. Bayesian Comparison

Let us now compare the OLS results from Session 3 with the Bayesian results from this session.
This is where the conceptual differences become concrete.

```python
# OLS vs. Bayesian comparison table
# NOTE: Fill in the OLS values from your Session 3 results.
# The OLS model used different channel specs and grid-searched transformation parameters.

# Placeholder OLS values -- REPLACE with your actual Session 3 results
ols_contributions_pct = {
    "TV_Spends":                     None,  # e.g., 15.3
    "Paid_Search_Spends":            None,  # e.g., 8.2
    "Programmatic_Display_Spends":   None,  # e.g., 5.1
    "Meta_Spends_Agg":               None,  # e.g., 6.7
    "Youtube_Spends":                None,  # e.g., 3.4
}

# Bayesian values (from our model)
bayesian_contribs_pct = dict(zip(
    contrib_ci["Channel"].values,
    contrib_ci["Mean_Pct"].values,
))

bayesian_ci_lower = dict(zip(
    contrib_ci["Channel"].values,
    contrib_ci["CI_2.5%"].values,
))

bayesian_ci_upper = dict(zip(
    contrib_ci["Channel"].values,
    contrib_ci["CI_97.5%"].values,
))

comparison_df = pd.DataFrame({
    "Channel": channel_list,
    "OLS_Contrib_Pct": [ols_contributions_pct.get(ch) for ch in channel_list],
    "Bayes_Mean_Pct": [bayesian_contribs_pct.get(ch, 0) for ch in channel_list],
    "Bayes_CI_Lower": [bayesian_ci_lower.get(ch, 0) for ch in channel_list],
    "Bayes_CI_Upper": [bayesian_ci_upper.get(ch, 0) for ch in channel_list],
})

print("OLS vs. Bayesian Contribution Comparison:")
print("(Fill in OLS values from your Session 3 notebook)")
print("=" * 85)
print(comparison_df.to_string(index=False, float_format="{:.2f}".format))
print("\nThe Bayesian model provides credible intervals -- OLS gives only point estimates.")
```

```python
# Compare transformation parameters: OLS grid search vs. Bayesian posterior
# NOTE: Fill in the OLS grid-search winners from Session 3

# Placeholder OLS best transformation parameters -- REPLACE with Session 3 results
ols_params = {
    "TV_Spends":                     {"theta": None, "alpha": None, "gamma": None},
    "Paid_Search_Spends":            {"theta": None, "alpha": None, "gamma": None},
    "Programmatic_Display_Spends":   {"theta": None, "alpha": None, "gamma": None},
    "Meta_Spends_Agg":               {"theta": None, "alpha": None, "gamma": None},
    "Youtube_Spends":                {"theta": None, "alpha": None, "gamma": None},
}

param_comparison = []
for j, ch in enumerate(channel_list):
    param_comparison.append({
        "Channel": ch,
        "OLS_theta": ols_params[ch]["theta"],
        "Bayes_theta": theta_post[j],
        "OLS_alpha": ols_params[ch]["alpha"],
        "Bayes_alpha": alpha_post[j],
        "OLS_gamma": ols_params[ch]["gamma"],
        "Bayes_gamma": gamma_post[j],
    })

param_comp_df = pd.DataFrame(param_comparison)
print("Transformation Parameter Comparison (OLS Grid Search vs. Bayesian Posterior Mean):")
print("=" * 100)
print(param_comp_df.to_string(index=False))
print("\nDifferences arise because:")
print("  1. OLS optimizes each channel independently; Bayesian learns them jointly.")
print("  2. Bayesian priors pull parameters toward prior beliefs.")
print("  3. Bayesian accounts for parameter interactions across channels.")
```

```python
# Visual comparison of channel effects with uncertainty
fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 5), sharey=False)

beta_flat = trace_mmm.posterior["beta_media"].values.reshape(-1, n_channels)

for j, ch in enumerate(channel_list):
    ax = axes[j]
    ax.hist(beta_flat[:, j], bins=50, density=True, alpha=0.7, color="steelblue",
            edgecolor="white")
    ax.axvline(beta_flat[:, j].mean(), color="darkblue", linewidth=2,
               label=f"Mean: {beta_flat[:, j].mean():.3f}")
    ax.set_title(ch.replace("_", "\n"), fontsize=10, fontweight="bold")
    ax.set_xlabel("beta")
    ax.legend(fontsize=8)

fig.suptitle("Posterior Distributions of Channel Effects (beta_media)",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()

print("These distributions are the Bayesian advantage:")
print("  - OLS gives a single bar (point estimate).")
print("  - Bayesian gives the full histogram (distribution of plausible values).")
```

### Discussion: 3 Key Differences Between OLS and Bayesian Results

**After reviewing your comparison tables and plots, write down 3 key differences:**

| # | Category | Your Observation |
|---|---|---|
| 1 | **Uncertainty** | OLS: single point estimate per channel. Bayesian: full posterior distribution with credible intervals. |
| 2 | **Coefficient signs** | OLS: may produce negative media coefficients (business-illogical). Bayesian: HalfNormal prior enforces positive effects. |
| 3 | **Parameter estimation** | OLS: grid search (theta/alpha/gamma separate from regression). Bayesian: all parameters learned jointly, accounting for interdependencies. |

**Additional differences to consider:**
- **Regularization**: Bayesian priors act as regularizers, potentially reducing overfitting.
- **Speed**: OLS is fast (seconds); Bayesian is slow (minutes to hours).
- **Interpretability**: Bayesian gives probability statements ("there is a 95% probability that TV ROI is between X and Y"), while OLS confidence intervals have a less intuitive interpretation.
- **Small data**: Bayesian handles small samples better because priors provide information when data is scarce.

---

## 8. BYOD Check-in: Data Preparation Status

Starting in Week 3, you will begin applying these techniques to **your own data** (Bring Your Own Data).

### Data Readiness Checklist

- [ ] Your data is in a clean CSV or Excel file
- [ ] You have a clear **dependent variable** (KPI: sales, revenue, conversions, etc.)
- [ ] You have a **time column** (weekly or monthly granularity)
- [ ] **Media spend** or activity columns are identified and labeled
- [ ] Any **control variables** (price, macro indicators, seasonality) are included
- [ ] You have reviewed the data for **missing values, outliers, and scale issues**
- [ ] You have thought about which channels you expect to have:
  - Higher adstock decay (TV, video)
  - Lower adstock decay (search, display)
  - Stronger saturation effects

### Group Discussion (10 min)

1. **What KPI** are you modeling? (sales volume, revenue, leads, etc.)
2. **How many channels** do you have? Are they spend-based or impression-based?
3. **What time granularity** is your data? (weekly is preferred for MMM)
4. **Any data quality concerns** you have discovered during preparation?
5. **Which approach** are you leaning toward: OLS (fast iteration) or Bayesian (richer output)?

**Bring your formatted data to Session 6 (Week 3 workshop).** If you have questions about data
preparation, reach out to the instructor before then.

---

## 9. Key Takeaways & Next Steps

### What We Accomplished Today

1. **Rebuilt the OLS model as Bayesian** -- same business logic (adstock + saturation + regression),
   but now with full uncertainty quantification.
2. **Implemented PyTensor operations** -- `scan` for adstock, tensor ops for Hill saturation.
   These are differentiable, so PyMC can compute gradients for efficient MCMC.
3. **Diagnosed convergence** -- trace plots, r_hat, energy plots, divergence counts.
4. **Extracted credible intervals** on contributions and ROI -- the key deliverable for stakeholders.
5. **Compared OLS vs. Bayesian** and identified where each approach has strengths.

### The Bayesian Advantage for MMM

| Benefit | Why It Matters |
|---|---|
| **Credible intervals on ROI** | Tell stakeholders "ROI is between X and Y with 95% probability" instead of just "ROI is X" |
| **Prior knowledge** | Encode that media should help sales, that TV decays slowly, etc. |
| **Joint estimation** | Transformation and regression parameters are learned together, not in separate steps |
| **Robustness** | Priors act as regularization, reducing overfitting on small marketing datasets |

### What Comes Next

- **Session 5**: We will explore **PyMC-Marketing (Meridian)**, a purpose-built Bayesian MMM library
  that handles adstock, saturation, and model building out of the box.
- **Session 6 (Week 3)**: You will apply these techniques to your own data (BYOD workshop).
- **Offline**: Review your OLS vs. Bayesian comparison and refine your model specs.

### Deliverable Reminder

Before the next session, make sure you have:

1. This notebook with a **working Bayesian MMM** (sampling completed, convergence checks passed).
2. A **written comparison** of 3 key differences between OLS and Bayesian results (see Section 7).
3. Your **BYOD data** cleaned and formatted for Week 3.
