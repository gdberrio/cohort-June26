# Reading 4: Introduction to Bayesian Thinking

**Estimated time: ~45 minutes (text + video)**

---

## Two Schools of Statistics

There are two major approaches to statistical inference. This course uses both, but leans heavily toward the Bayesian approach for our advanced models.

### Frequentist Approach
- **Parameters are fixed** but unknown
- **Data is random** (imagines repeated sampling)
- **Inference:** p-values, confidence intervals, hypothesis tests
- **Example:** "If we collected data 100 times, 95 of those confidence intervals would contain the true parameter"
- **Used in:** OLS regression (Week 2, Session 3)

### Bayesian Approach
- **Parameters are random** (have probability distributions)
- **Data is fixed** (we observe what we observe)
- **Inference:** posterior distributions, credible intervals, probability statements
- **Example:** "There is a 95% probability that the true parameter lies in this interval"
- **Used in:** Bayesian MMM, PyMC, Meridian (Week 2-4)

## Bayes' Theorem

At the heart of Bayesian statistics is a simple formula:

```
P(θ | data) = P(data | θ) × P(θ) / P(data)
```

In words:

```
Posterior = Likelihood × Prior / Evidence
```

### The Components

- **Prior P(θ):** What we believe about the parameter *before* seeing the data
  - Marketing example: "TV ROI is probably between 0.5 and 3.0 based on industry benchmarks"

- **Likelihood P(data | θ):** How probable the observed data is, given a parameter value
  - Marketing example: "If TV ROI were 2.0, how likely would we see these sales numbers?"

- **Posterior P(θ | data):** Our updated belief *after* seeing the data
  - Marketing example: "Given our data and prior knowledge, TV ROI is most likely around 1.8 with 95% probability between 1.2 and 2.5"

- **Evidence P(data):** Normalizing constant (ensures posterior integrates to 1)
  - Usually handled computationally; we don't need to calculate it directly

## A Marketing Example

Suppose you're estimating the ROI of a new social media channel.

### Step 1: Set Your Prior
Based on industry benchmarks and similar channels, you believe ROI is around 1.5 with some uncertainty:
```
Prior: ROI ~ Normal(mean=1.5, std=0.5)
```
This says: "We think ROI is probably between 0.5 and 2.5 (±2 standard deviations)"

### Step 2: Observe Data
You run the channel for 6 months and collect spend + sales data. The data suggests ROI might be around 2.0.

### Step 3: Compute Posterior
The posterior combines prior and data:
- If data is strong (lots of observations, low noise): posterior is close to the data estimate (2.0)
- If data is weak (few observations, high noise): posterior is closer to the prior (1.5)
- Result might be: `Posterior: ROI ~ Normal(mean=1.8, std=0.3)`

The posterior is *always* a compromise between prior knowledge and observed data, weighted by their relative certainty.

## Why Bayesian for MMM?

### 1. Incorporating Prior Knowledge
- We know media effects should be positive (media spend shouldn't decrease sales)
- We have industry benchmarks for ROI ranges
- Previous studies, experiments, and business knowledge inform our priors
- OLS ignores all this — it only uses the data

### 2. Uncertainty Quantification
- Bayesian models give full probability distributions, not just point estimates
- "TV ROI is 2.1 ± 0.4 (95% credible interval: 1.3 to 2.9)" is more useful than just "TV ROI is 2.1"
- Decision-makers can assess risk: "There's a 90% chance reallocating budget will improve revenue"

### 3. No Ad-Hoc Grid Search
- OLS MMM requires grid search over adstock/saturation parameters (hundreds of combinations)
- Bayesian MMM estimates these parameters directly within the model
- Parameters have posterior distributions, not just "best" values

### 4. Experimental Calibration
- Experiment results can be encoded as informative priors
- "Our geo experiment found TV lift of 15% (CI: 8-22%)" → Prior for TV coefficient
- This bridges the gap between correlation (MMM) and causation (experiments)

## Introduction to MCMC (Conceptual)

Computing the posterior analytically is usually impossible for complex models. Instead, we use **Markov Chain Monte Carlo (MCMC)** — an algorithm that generates samples from the posterior distribution.

### How MCMC Works (Simplified)

1. Start at a random point in parameter space
2. Propose a new point nearby
3. If the new point has higher posterior probability, move there
4. If lower, move there with some probability (to avoid getting stuck)
5. Repeat thousands of times
6. The visited points form a sample from the posterior distribution

### Key MCMC Concepts

- **Chains:** Multiple independent MCMC runs (typically 4)
- **Warmup/Tune:** Initial samples discarded as the chain finds the right region
- **Draws:** Samples kept for inference
- **Convergence:** All chains should agree (R-hat ≈ 1.0)
- **Trace plot:** Visual check that chains are mixing well (look like "hairy caterpillars")

### In Practice (PyMC)
```python
import numpy as np
import pymc as pm

spend = np.array([10, 12, 8, 15, 11, 14])
base = 100
noise = 5
data = np.array([116, 120, 111, 132, 119, 128])

with pm.Model():
    # Prior
    roi = pm.Normal("roi", mu=1.5, sigma=0.5)

    # Likelihood
    sales = pm.Normal("sales", mu=base + roi * spend, sigma=noise, observed=data)

    # Sample from posterior using MCMC. Use larger draws/chains for real analysis.
    trace = pm.sample(draws=200, tune=200, chains=2, cores=1, progressbar=False)
```

We'll use PyMC extensively starting in Week 2.

## Frequentist vs. Bayesian: Quick Comparison

| Aspect | Frequentist (OLS) | Bayesian (PyMC) |
|---|---|---|
| Parameters | Fixed, unknown | Random variables with distributions |
| Priors | Not used | Explicitly specified |
| Result | Point estimates + CI | Full posterior distributions |
| Interpretation | "95% CI contains true value in repeated sampling" | "95% probability parameter is in this range" |
| Uncertainty | Standard errors | Credible intervals |
| Grid search | Required for adstock/saturation | Estimated within model |
| Computation | Fast (closed-form) | Slower (MCMC sampling) |
| Calibration | Ad-hoc constraints | Informative priors |

## Practice Questions

1. In your own words, explain what a "prior" represents in Bayesian analysis.
2. Why would we use a HalfNormal prior (positive only) for media coefficients in MMM?
3. What does it mean when R-hat is 1.5 for a parameter? Is this good or bad?
4. How would you encode the result of a geo experiment (TV lift = 12%, 95% CI: 5-19%) as a Bayesian prior?
5. If you have very little data, will the posterior look more like the prior or the likelihood? Why?

---

*You're now ready for Session 1! Make sure you've also completed Notebook 0 (Explore the Workshop Dataset) and set up your Python environment using the smoke test notebook.*
