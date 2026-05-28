# Reading 6: Calibrating MMM with Experiments

**Estimated time: ~20 minutes**

---

## What Calibration Means

An MMM, no matter how well specified, produces estimates that are fundamentally observational. The model identifies statistical associations between marketing inputs and outcomes, but those associations can be confounded (as discussed in Reading 3). **Calibration** is the process of using experimental results -- which *do* measure causal effects -- to anchor, validate, and improve MMM estimates.

Think of it this way:
- An **uncalibrated MMM** is like a compass that might be pointing slightly off-North. It gives you a direction, but you are not sure how accurate it is.
- A **calibrated MMM** has been checked against known landmarks. You have used experiments to verify (or correct) the compass readings for at least some channels.

Calibration does not replace the model. It strengthens it by injecting causal evidence where the model otherwise relies on correlational inference.

---

## The Calibration Workflow

The end-to-end calibration process follows five steps:

```
Step 1          Step 2             Step 3            Step 4           Step 5
Run             Extract lift       Encode as         Re-fit           Compare
experiment      estimate           Bayesian prior    the MMM          results

  |                 |                  |                 |                |
  v                 v                  v                 v                v
Geo test or     "TV drives a       Set prior on       Run Bayesian     Do MMM
A/B test on     15% lift in        TV coefficient     model with       estimates
a channel       sales, 95% CI      centered at        informative      align with
                is [8%, 22%]"      the experiment     priors           experiment?
                                   result
```

### Step 1: Run the experiment

Design and execute a randomized experiment on the channel you want to calibrate. This could be a geo holdout, a user-level A/B test, or an incrementality test (details below).

### Step 2: Extract the lift estimate

From the experiment, obtain:
- **Point estimate:** The best estimate of the channel's causal effect (e.g., "TV drives a 15% lift in sales")
- **Confidence interval:** The uncertainty around that estimate (e.g., "95% CI: 8% to 22%")

Both the point estimate and the uncertainty are critical. A noisy experiment with a wide interval should influence the MMM less than a precise experiment with a tight interval.

### Step 3: Encode as a Bayesian prior

Translate the experiment result into an informative prior distribution for the corresponding coefficient in the Bayesian MMM. The prior expresses your belief about the parameter *before* seeing the MMM's data, informed by the experiment.

### Step 4: Re-fit the MMM

Run the Bayesian MMM with the informative prior. The posterior distribution will blend the experimental evidence (prior) with the observational evidence (data likelihood). If both agree, the posterior will be tight and centered near the experiment result. If they disagree, the posterior will reflect the tension.

### Step 5: Compare results

Examine how the calibrated model differs from the uncalibrated version. Check whether the calibrated coefficients are more consistent with the experiment, and whether the model fit (MAPE, R-squared) is maintained or improved.

---

## Types of Experiments for Calibration

### Geo Experiments (Holdout Markets)

The most common calibration method for MMM. You select geographic regions (cities, states, DMAs) and randomly assign them to treatment (media continues) or control (media is suppressed or reduced).

**Design:**
```
Treatment geos:  Chicago, Houston, Atlanta, Phoenix (media ON)
Control geos:    Denver, Portland, Nashville, Milwaukee (media OFF)

Duration:        4-8 weeks
Measurement:     Compare sales lift between treatment and control geos
```

**Strengths:** Works for any channel including offline (TV, radio, OOH). Directly measures incremental sales at the level MMM operates (aggregate).

**Limitations:** Requires sufficient geographic variation, loses revenue in holdout markets, potential contamination from spillover (people in control markets see digital ads).

### A/B Tests (User-Level)

Standard randomized controlled trials at the user level. Users are randomly assigned to see an ad (treatment) or not see it (control).

**Best for:** Digital channels where user-level randomization is possible (paid social, display, email).

**Limitations:** Cannot test offline channels, affected by tracking limitations, typically measure short-term effects only.

### Incrementality Tests (Ghost Ads, PSA Tests)

Specialized designs for digital channels:

- **Ghost ads (ghost bidding):** In the treatment group, ads are served normally. In the control group, the platform identifies users *who would have seen the ad* but does not show it to them. This controls for selection bias (only comparing people who were eligible to see the ad).

- **PSA tests (public service announcements):** The control group sees a non-commercial ad (e.g., a charity PSA) instead of the brand ad. This ensures both groups have the same exposure frequency, isolating the effect of the ad content.

**Strengths:** More precise than simple A/B holdout because they control for ad-exposure propensity.

**Limitations:** Require platform support (Meta, Google), only work for digital channels.

---

## Encoding Experiment Results as Bayesian Priors

This is the technical heart of calibration. The experiment result becomes an informative prior in the Bayesian model.

### Example: TV Calibration

Suppose a geo experiment found that TV advertising drives a **15% lift in sales**, with a 95% confidence interval of **[8%, 22%]**.

To convert this into a prior:

1. **Point estimate becomes the prior mean:** mu = 0.15
2. **Confidence interval determines the prior standard deviation:** The 95% CI spans approximately 4 standard deviations (from -2 sigma to +2 sigma). So: sigma = (0.22 - 0.08) / 4 = 0.035

In PyMC, this becomes:

```python
import pymc as pm

with pm.Model() as calibrated_model:
    # Informative prior from geo experiment
    tv_coef = pm.Normal("tv_coef", mu=0.15, sigma=0.035)

    # Weakly informative priors for other channels (no experiments yet)
    social_coef = pm.HalfNormal("social_coef", sigma=0.5)
    search_coef = pm.HalfNormal("search_coef", sigma=0.5)

    # ... rest of the model
```

### Example: Social Media Calibration

An A/B test on Meta shows a **$2.50 incremental CPA**, with 90% CI of **[$1.80, $3.20]**.

Converting to a coefficient scale depends on your model specification. If the model uses log-transformed spend and log-transformed sales (elasticity form), you would need to convert the CPA into an elasticity estimate first and then set the prior accordingly.

```python
# If experiment implies an elasticity of 0.12 with SE 0.04
social_coef = pm.Normal("social_coef", mu=0.12, sigma=0.04)
```

### Key Principle: Prior Strength Reflects Experiment Quality

| Experiment Quality | Prior Type | Example |
|---|---|---|
| Large-scale geo experiment, tight CI | Strong informative prior | `pm.Normal(mu=0.15, sigma=0.02)` |
| Small experiment, wide CI | Weakly informative prior | `pm.Normal(mu=0.15, sigma=0.10)` |
| No experiment available | Default weakly informative prior | `pm.HalfNormal(sigma=0.5)` |

A strong prior will pull the MMM estimate firmly toward the experiment result. A weak prior will let the data speak more, using the experiment as a gentle nudge rather than a hard constraint.

---

## Handling Conflicts: When MMM and Experiment Disagree

Sometimes, the uncalibrated MMM says TV elasticity is 0.30, but the experiment says 0.15. What do you do?

### Diagnose the conflict first

Before adjusting anything, ask:
1. **Is the experiment valid?** Was randomization successful? Was there contamination? Was the test period long enough to capture carry-over effects?
2. **Is the MMM confounded?** Is TV spend correlated with seasonality or promotions that the model does not adequately control?
3. **Are they measuring the same thing?** The experiment measures short-run lift during the test window. The MMM estimates the average effect across the full data period, including long-run effects.

### Approaches to resolution

| Approach | When to Use | How |
|---|---|---|
| **Trust the experiment** | Large, well-designed experiment; suspected confounding in MMM | Use a strong informative prior |
| **Compromise** | Both sources are credible but imperfect | Use a moderately informative prior |
| **Trust the model** | Small or flawed experiment; strong model diagnostics | Use a very weak prior or ignore the experiment |
| **Prior sensitivity analysis** | Unsure which to trust | Run the model with multiple prior strengths and compare |

### Prior Sensitivity Analysis

Run the model three times with different prior settings for the channel in question:

```python
# Scenario A: Strong prior (trust experiment)
tv_coef_a = pm.Normal("tv_coef", mu=0.15, sigma=0.02)

# Scenario B: Moderate prior (compromise)
tv_coef_b = pm.Normal("tv_coef", mu=0.15, sigma=0.05)

# Scenario C: Weak prior (trust model)
tv_coef_c = pm.Normal("tv_coef", mu=0.15, sigma=0.15)
```

If the posterior is similar across all three scenarios, the data is strong enough to overwhelm the prior and the conflict is less concerning. If the posterior shifts dramatically, the model is sensitive to the prior, and you need more data or a better experiment to resolve the disagreement.

---

## How Meta Robyn Handles Calibration

Meta's open-source MMM framework **Robyn** supports calibration through the `calibration_input` parameter. You provide experiment results directly:

```r
calibration_input <- data.frame(
  channel       = "tv_spend",
  lift_start_date = as.Date("2024-03-01"),
  lift_end_date   = as.Date("2024-04-15"),
  lowerbound    = 0.08,    # Lower bound of experiment CI
  upperbound    = 0.22,    # Upper bound of experiment CI
  metric        = "sales"
)
```

Robyn then uses this information during its model selection process. Models whose estimated channel effects fall within the experiment's confidence interval are preferred during the multi-objective optimization (Pareto front selection). Models that conflict with experimental evidence are penalized.

**Key nuance:** Robyn is a frequentist (ridge regression) framework, not Bayesian. It does not use priors in the statistical sense. Instead, it uses calibration as a **model selection filter** -- only accepting models from the Pareto front that are consistent with experimental results.

---

## How Google Meridian Handles Calibration

Google's **Meridian** framework is fully Bayesian and supports experimental calibration natively through the `prior_broadcast` parameter (for reach/frequency models) and custom prior specifications.

In Meridian, you can set informative priors on media coefficients based on experiment results:

```python
from meridian import prior as prior_lib

# Set an informative prior for TV based on geo experiment
custom_priors = prior_lib.PriorDistribution(
    beta_m=prior_lib.LogNormal(
        mu=[0.15, ...],    # Prior means for each media channel
        sigma=[0.035, ...] # Prior SDs derived from experiment
    )
)

model = meridian.Meridian(
    input_data=input_data,
    model_spec=model_spec,
    prior=custom_priors
)
```

Because Meridian is Bayesian, the experimental prior is blended with the data likelihood through standard Bayesian updating. The posterior reflects both sources of evidence, weighted by their relative precision.

**Key advantage:** Meridian produces posterior distributions, so you can directly see how much the calibration shifted the estimate and how much uncertainty remains.

---

## Best Practices for Calibration

1. **Stagger experiments across channels.** You cannot run geo holdouts for every channel simultaneously -- the holdout markets would have no marketing at all. Plan a testing calendar that covers 2-3 channels per quarter.

2. **Refresh experiments regularly.** A TV experiment from two years ago may not reflect today's media landscape. Re-test channels at least annually, or when major changes occur (new creative, new audience targeting, market shifts).

3. **Document all assumptions.** Record how you translated the experiment result into a prior: what the raw lift was, how you converted units, what sigma you chose, and why. Future analysts (including your future self) will need this context.

4. **Start with your largest channels.** Calibrate the channels with the biggest spend first -- errors in large channels have the largest impact on budget allocation decisions.

5. **Use calibration to build credibility.** Showing stakeholders that your MMM is anchored in experimental evidence dramatically increases trust in the results. Lead with this in executive presentations.

6. **Accept that not everything can be calibrated.** Some channels (organic social, PR, word-of-mouth) are extremely difficult to experiment on. For these, transparent uncertainty in your priors is better than false precision.

---

## Practice Questions

**Question 1:** A geo experiment for paid social shows a 20% lift in conversions (95% CI: 5% to 35%). Your uncalibrated MMM estimates social's elasticity at 0.08. Describe how you would encode the experiment result as a prior and discuss whether the wide confidence interval changes your approach.

**Question 2:** Your company ran a ghost ad test for display advertising and found a 3% incremental lift with a very tight confidence interval (95% CI: 2% to 4%). The MMM estimates display's contribution at 12% of total sales. What are three possible explanations for this discrepancy?

**Question 3:** You run a prior sensitivity analysis for TV calibration with three prior strengths (sigma = 0.02, 0.05, 0.15). The posterior TV elasticity is 0.14, 0.18, and 0.28 respectively. What does this tell you about your model, and what would you do next?

**Question 4:** A colleague argues that calibration with experiments makes MMM unnecessary -- "Why not just run experiments for everything?" Construct a response explaining why both are needed.

---

*This reading prepares you for Session 7 on experimentation and geo-lift testing*
