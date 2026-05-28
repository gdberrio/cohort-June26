# Reading 7: Understanding Meridian's Prior System

A reference guide to prior distributions in Google Meridian. Priors encode your
beliefs about model parameters *before* seeing the data. Choosing the right
priors — and calibrating them with experimental evidence — is one of the most
impactful decisions in Bayesian MMM.

**Sources:**
[Prior types](https://developers.google.com/meridian/docs/advanced-modeling/how-to-choose-treatment-prior-types) |
[ROI/mROI/contribution parameterizations](https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations) |
[Default priors](https://developers.google.com/meridian/docs/advanced-modeling/default-prior-distributions) |
[Unknown revenue/KPI](https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-default)

---

## 1. Prior Types: ROI, mROI, Contribution, Coefficient

Meridian supports four ways to parameterize treatment priors. The choice
determines *what metric* you're placing beliefs on.

### ROI Prior (`media_prior_type='roi'`)

**Definition:** Incremental Outcome / Spend

- The default and most intuitive option for paid media
- Aligns naturally with geo-experiment and lift study results
- Default: `LogNormal(0.2, 0.9)` — prior mean ROI of 1.83, 95% CI [0.25, 9.0]
- Requires `LogNormal` media effects distribution (positive ROI only)

**Best for:** Paid media channels where you have (or can estimate) a
dollar-in-dollar-out return.

### mROI Prior (`media_prior_type='mroi'`)

**Definition:** Return on the next incremental dollar of spend

- Useful for budget regularization: setting mROI = 1.0 with tight std says
  "current budgets are roughly optimal"
- Default: `LogNormal(0.0, 0.5)` — prior mean mROI of 1.13, 95% CI [0.33, 2.66]
- Tighter default than ROI because marginal returns are more constrained

**Best for:** When you want to regularize budget optimization recommendations
toward historical allocations, or when you have marginal efficiency estimates.

### Contribution Prior (`media_prior_type='contribution'`)

**Definition:** Incremental Outcome / Total Observed Outcome (a share, 0-100%)

- Default: `Beta(1.0, 99.0)` — prior mean of 1% per channel, 95% CI [0.03%, 3.7%]
- The only option available for organic media and non-media treatments
  (which have no spend denominator)
- Non-media treatments use `TruncatedNormal(0.0, 0.1, -1.0, 1.0)` —
  allows negative effects (e.g., price increases hurt volume)

**Best for:** Organic channels, non-media treatments, or when you think in
terms of "what share of revenue does this channel drive?"

### Coefficient Prior (`media_prior_type='coefficient'`)

**Definition:** The raw regression coefficient (beta)

- Default: `HalfNormal(5.0)` — uninformative
- Lacks intuitive business interpretation
- Rarely recommended unless you have strong statistical reasons

**Best for:** Advanced users who need direct control over the regression
coefficients and understand the model's internal scaling.

### Applicability by Channel Type

| Prior Type | Paid Media | Organic Media | Non-Media Treatments |
|-----------|:----------:|:-------------:|:--------------------:|
| ROI       | Yes        | No            | No                   |
| mROI      | Yes        | No            | No                   |
| Contribution | Yes     | Yes (default) | Yes (default)        |
| Coefficient | Yes      | Yes           | Yes                  |

### Configuring Prior Types in ModelSpec

```python
model_spec = spec.ModelSpec(
    prior=my_prior,
    media_prior_type='roi',                     # paid impression media
    rf_prior_type='roi',                        # paid reach & frequency media
    organic_media_prior_type='contribution',    # organic impression media
    organic_rf_prior_type='contribution',       # organic R&F media
    non_media_treatments_prior_type='contribution',  # non-media treatments
)
```

---

## 2. How ROI/mROI Priors Work Mathematically

Understanding the math helps you reason about what your prior *actually implies*
for the model's internal parameters.

### The Beta-ROI Relationship

Meridian's model estimates channel impact through coefficients (beta). The ROI
prior re-parameterizes the model so that ROI is a direct parameter, and beta
becomes a derived quantity:

```
Incremental_Outcome_m = SUM_g,t [ beta_gm * M_gm(t) ]
                      = ROI_m * Cost_m
```

where `M_gm(t)` is the Hill-Adstock transformed media term. Solving for beta:

- **LogNormal effects:** `beta_m = log(ROI_m * Cost_m) - log(SUM[exp(eta * Z) * M])`
- **Normal effects:** `beta_m = (ROI_m * Cost_m - eta * SUM[Z * M]) / SUM[M]`

### Key Implication

Because beta depends on ROI *and* the adstock/saturation parameters (alpha, ec,
slope), placing independent priors on (ROI, alpha, ec, slope) induces a
*dependent* prior on beta. This is intentional — it ensures the ROI
interpretation is consistent with the transformation parameters.

### mROI Parameterization

mROI measures the return from a 1% spend increase:

```
Marginal_Outcome_m = SUM [ beta_gm * (HillAdstock(1.01 * x) - HillAdstock(x)) ]
mROI_m = Marginal_Outcome_m / (0.01 * Cost_m)
```

The implementation substitutes `0.01 * mROI_m` for `ROI_m` in the same equations.

### Prior Parity Warning

Setting identical ROI priors across channels does **not** produce identical mROI
priors (and vice versa). This is because the Hill-Adstock transformation
creates a nonlinear mapping between ROI and mROI. If you need parity on a
specific metric, set priors directly on that metric.

### Inspecting Induced Priors

You can examine the priors that ROI induces on other metrics:

```python
analyzer = Analyzer(mmm)

# What mROI does the ROI prior imply?
induced_mroi = analyzer.marginal_roi(use_posterior=False)

# What ROI does the ROI prior imply? (sanity check)
induced_roi = analyzer.roi(use_posterior=False)
```

---

## 3. Default Prior Distributions (Complete Reference)

### Media ROI and Contribution

| Parameter | Default Distribution | Mean | 95% CI | Notes |
|-----------|---------------------|------|--------|-------|
| `roi_m` | `LogNormal(0.2, 0.9)` | 1.83 | [0.25, 9.0] | Paid impression media ROI |
| `roi_rf` | `LogNormal(0.2, 0.9)` | 1.83 | [0.25, 9.0] | Paid R&F media ROI |
| `mroi_m` | `LogNormal(0.0, 0.5)` | 1.13 | [0.33, 2.66] | Marginal ROI |
| `mroi_rf` | `LogNormal(0.0, 0.5)` | 1.13 | [0.33, 2.66] | Marginal ROI (R&F) |
| `contribution_m` | `Beta(1.0, 99.0)` | 1% | [0.03%, 3.7%] | Media contribution share |
| `contribution_rf` | `Beta(1.0, 99.0)` | 1% | [0.03%, 3.7%] | R&F contribution share |
| `contribution_om` | `Beta(1.0, 99.0)` | 1% | [0.03%, 3.7%] | Organic media contribution |
| `contribution_orf` | `Beta(1.0, 99.0)` | 1% | [0.03%, 3.7%] | Organic R&F contribution |
| `contribution_n` | `TruncatedNormal(0, 0.1, -1, 1)` | 0% | [-19.6%, 19.6%] | Non-media (allows negative) |

### Media Effect Coefficients (Hierarchical)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `beta_m` | `HalfNormal(5.0)` | Hierarchical mean of media coefficients |
| `beta_rf` | `HalfNormal(5.0)` | Hierarchical mean of R&F coefficients |
| `beta_om` | `HalfNormal(5.0)` | Hierarchical mean of organic media |
| `beta_orf` | `HalfNormal(5.0)` | Hierarchical mean of organic R&F |
| `eta_m` | `HalfNormal(1.0)` | Hierarchical std dev (encourages geo pooling) |
| `eta_rf` | `HalfNormal(1.0)` | Hierarchical std dev (R&F) |
| `eta_om` | `HalfNormal(1.0)` | Hierarchical std dev (organic) |
| `eta_orf` | `HalfNormal(1.0)` | Hierarchical std dev (organic R&F) |

### Adstock Decay

| Parameter | Default | Notes |
|-----------|---------|-------|
| `alpha_m` | `Uniform(0, 1)` | Fully uninformative — data determines decay |
| `alpha_rf` | `Uniform(0, 1)` | Same for R&F channels |
| `alpha_om` | `Uniform(0, 1)` | Same for organic |
| `alpha_orf` | `Uniform(0, 1)` | Same for organic R&F |

**Tip:** If you know digital channels decay faster than TV, use informative
priors like `Beta(2, 5)` for digital (fast decay) and `Beta(5, 2)` for TV
(slow decay).

### Hill Saturation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ec_m` | `TruncatedNormal(0.8, 0.8, 0.1, 10)` | Half-saturation for impression media. When ec=1, saturation occurs at the median non-zero media level |
| `ec_rf` | `LogNormal(0.7, 0.4) + 0.1` | Half-saturation for R&F (shifted). Set jointly with slope_rf for optimal frequency |
| `ec_om` | `TruncatedNormal(0.8, 0.8, 0.1, 10)` | Same as ec_m |
| `ec_orf` | `LogNormal(0.7, 0.4) + 0.1` | Same as ec_rf |
| `slope_m` | `Deterministic(1.0)` | Fixed at 1 — restricts to concave Hill curves |
| `slope_rf` | `LogNormal(0.7, 0.4)` | Moderately informative, paired with ec_rf |
| `slope_om` | `Deterministic(1.0)` | Same as slope_m |
| `slope_orf` | `LogNormal(0.7, 0.4)` | Same as slope_rf |

### Controls, Time, and Geo

| Parameter | Default | Notes |
|-----------|---------|-------|
| `gamma_c` | `Normal(0, 5)` | Control variable coefficients (allows positive/negative) |
| `gamma_n` | `Normal(0, 5)` | Non-media treatment coefficients |
| `xi_c` | `HalfNormal(5.0)` | Geo variation in control effects |
| `xi_n` | `HalfNormal(5.0)` | Geo variation in non-media effects |
| `knot_values` | `Normal(0, 5)` | Temporal trend spline coefficients |
| `tau_g_excl_baseline` | `Normal(0, 5)` | Geo-level intercept offsets |
| `sigma` | `HalfNormal(5.0)` | Residual noise std dev |

---

## 4. When `revenue_per_kpi` Is Unknown

When your KPI is not revenue (e.g., conversions, sign-ups) and you cannot
specify `revenue_per_kpi`, Meridian changes its default behavior:

### What Changes

Instead of applying per-channel ROI priors, Meridian uses a **total paid media
contribution prior**: the proportion of the KPI attributable to all paid media
combined has a prior mean of **40%** with standard deviation of **20%**.

This means:
- ROI becomes "incremental KPI units per spend unit" (not dollars)
- The model constrains total media impact rather than per-channel ROI
- Channel-level priors are not individually specified

### Recommendations

1. **Best option:** Provide `revenue_per_kpi` if at all possible — even a rough
   estimate enables proper monetary ROI priors
2. **Alternative:** Set custom per-channel contribution priors if you have
   channel-level incrementality data
3. **Last resort:** Accept the 40%/20% default and interpret results in KPI
   units rather than revenue

---

## 5. Practical Guidance

### Decision Tree for Choosing Prior Types

```
Do you have spend data for this channel?
├── Yes → Is your KPI revenue (or do you have revenue_per_kpi)?
│   ├── Yes → Do you have experiment results?
│   │   ├── Yes → Use ROI prior calibrated with experiment data
│   │   └── No  → Use default ROI prior (LogNormal(0.2, 0.9))
│   └── No  → Use contribution prior (Beta) or provide revenue_per_kpi
└── No  → Use contribution prior (Beta(1, 99) default)
```

### When to Customize Priors

| Scenario | Recommendation |
|----------|---------------|
| Geo-experiment results available | Set `roi_m` using `lognormal_dist_from_mean_std()` |
| Lift study with confidence interval | Set `roi_m` using `lognormal_dist_from_range()` |
| Industry benchmark for a channel | Use wider std (0.5-1.0) to reflect uncertainty |
| Budget is roughly optimal | Set `mroi_m` priors near 1.0 with tight std |
| Digital channels have fast decay | Set `alpha_m` to `Beta(2, 5)` for those channels |
| TV has slow, persistent effects | Set `alpha_m` to `Beta(5, 2)` for TV |
| Organic channel with no spend | Use `contribution_om` with `Beta` distribution |
| Price/promotion effects | Use `contribution_n` with `TruncatedNormal` allowing negatives |

### Common Pitfalls

1. **Priors too tight without evidence.** A tight prior on ROI = 3.0 when you
   don't have experiment data will force the model to conform, potentially
   distorting other channels' estimates.

2. **Ignoring induced priors.** Setting ROI priors also induces implicit priors
   on mROI and beta. Always check induced priors with
   `analyzer.marginal_roi(use_posterior=False)`.

3. **Applying experiment priors to the wrong time window.** Use
   `roi_calibration_period` to restrict experiment-based priors to the actual
   experiment dates.

4. **Assuming prior parity.** Equal ROI priors across channels do not imply
   equal mROI priors due to the nonlinear Hill-Adstock transformation.

5. **Forgetting the `name` argument.** When creating TFP distributions for
   `roi_m`, include `name=constants.ROI_M` (i.e., `name='roi_m'`).

---

## 6. Quick Reference: PriorDistribution Fields

```python
from meridian.model import prior_distribution
import tensorflow_probability as tfp
tfd = tfp.distributions

prior = prior_distribution.PriorDistribution(
    # --- ROI/mROI (paid media, when media_prior_type='roi' or 'mroi') ---
    roi_m=tfd.LogNormal(0.2, 0.9, name='roi_m'),       # impression media
    roi_rf=tfd.LogNormal(0.2, 0.9, name='roi_rf'),      # R&F media
    mroi_m=tfd.LogNormal(0.0, 0.5, name='mroi_m'),      # marginal ROI
    mroi_rf=tfd.LogNormal(0.0, 0.5, name='mroi_rf'),

    # --- Contribution (organic, non-media, or when prior_type='contribution') ---
    contribution_m=tfd.Beta(1.0, 99.0),
    contribution_rf=tfd.Beta(1.0, 99.0),
    contribution_om=tfd.Beta(1.0, 99.0),
    contribution_orf=tfd.Beta(1.0, 99.0),
    contribution_n=tfd.TruncatedNormal(0.0, 0.1, -1.0, 1.0),

    # --- Coefficients (when media_prior_type='coefficient') ---
    beta_m=tfd.HalfNormal(5.0),
    beta_rf=tfd.HalfNormal(5.0),
    beta_om=tfd.HalfNormal(5.0),
    beta_orf=tfd.HalfNormal(5.0),
    eta_m=tfd.HalfNormal(1.0),    # hierarchical std dev
    eta_rf=tfd.HalfNormal(1.0),
    eta_om=tfd.HalfNormal(1.0),
    eta_orf=tfd.HalfNormal(1.0),

    # --- Adstock decay ---
    alpha_m=tfd.Uniform(0.0, 1.0),
    alpha_rf=tfd.Uniform(0.0, 1.0),
    alpha_om=tfd.Uniform(0.0, 1.0),
    alpha_orf=tfd.Uniform(0.0, 1.0),

    # --- Hill saturation ---
    ec_m=tfd.TruncatedNormal(0.8, 0.8, 0.1, 10.0),
    slope_m=tfd.Deterministic(1.0),

    # --- Controls & structure ---
    gamma_c=tfd.Normal(0.0, 5.0),
    gamma_n=tfd.Normal(0.0, 5.0),
    xi_c=tfd.HalfNormal(5.0),
    xi_n=tfd.HalfNormal(5.0),
    knot_values=tfd.Normal(0.0, 5.0),
    tau_g_excl_baseline=tfd.Normal(0.0, 5.0),
    sigma=tfd.HalfNormal(5.0),
)
```

### Helper Functions

```python
# From experiment point estimate + standard error
dist = prior_distribution.lognormal_dist_from_mean_std(mean=2.0, std=0.5)

# From experiment confidence interval
dist = prior_distribution.lognormal_dist_from_range(low=1.0, high=4.0, mass_percent=0.95)

# Per-channel priors (ensures correct channel ordering)
builder = input_data.get_paid_media_channels_argument_builder()
roi_tuples = builder(TV=(0.2, 0.5), Search=(0.3, 0.7), Display=(0.4, 0.6))
mu_list, sigma_list = zip(*roi_tuples)
roi_prior = tfd.LogNormal(mu_list, sigma_list, name='roi_m')
```
