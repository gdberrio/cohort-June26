# Session 08 Capstone Wrapup

# Session 8: Capstone Presentations & Course Wrap-up

---

## Advanced Topics Lightning Round

Before we move to capstone presentations, let's do a quick tour of advanced topics that extend what we've learned throughout the course.

### Topic 1: LSTM Attribution (5 min concept)

Long Short-Term Memory (LSTM) networks can be used for attribution by modeling the customer journey as a **sequence**:

- Input: ordered sequence of touchpoints (e.g., [display, search, email, search, purchase])
- Output: conversion probability
- Attribution: use attention weights or gradient-based methods to assign credit to each touchpoint

**Advantages:** Captures order effects ("search after display" differs from "display after search"), handles variable-length journeys, learns non-linear interactions.

**Challenges:** Requires large volumes of path-level data, black-box nature makes interpretation difficult, computationally expensive.

### Topic 2: Regression-based Attribution

A simpler ML approach: treat attribution as a classification/regression problem where touchpoint presence in a journey predicts conversion. Feature importance from the model becomes the attribution weight.

### Topic 3: Bayesian Structural Time Series (BSTS) / CausalImpact

BSTS models decompose a time series into:
- **Trend:** local linear trend or random walk
- **Seasonality:** weekly, monthly, annual patterns
- **Regression component:** control series (other markets, covariates)

CausalImpact uses BSTS to construct the counterfactual, providing:
- Point estimates of causal effect
- Posterior credible intervals
- Cumulative effect over time

### Topic 4: Future of Marketing Measurement

The measurement landscape is evolving rapidly:

- **Privacy Sandbox:** Google's Topics API, Attribution Reporting API replacing third-party cookies
- **Incrementality-first approach:** Moving from "who converted" to "what caused the conversion"
- **Always-on experimentation:** Continuous geo holdouts for ongoing calibration
- **AI-native MMM:** LLMs for prior elicitation, automated diagnostics, natural language reporting
- **Unified measurement:** Frameworks that combine MMM + attribution + experiments in a single model

---

## Lightning Demo: Regression-Based Attribution

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Simulate customer journey data
# Each row = one customer journey
# Columns = binary flags for whether they were exposed to each channel
n_journeys = 10000

channels = ['display', 'paid_search', 'organic_search', 'email', 'social', 'direct']

# Generate touchpoint exposure data
journey_data = pd.DataFrame({
    'display': np.random.binomial(1, 0.40, n_journeys),
    'paid_search': np.random.binomial(1, 0.35, n_journeys),
    'organic_search': np.random.binomial(1, 0.50, n_journeys),
    'email': np.random.binomial(1, 0.25, n_journeys),
    'social': np.random.binomial(1, 0.30, n_journeys),
    'direct': np.random.binomial(1, 0.20, n_journeys),
})

# True conversion model (with known coefficients)
true_betas = {'display': 0.3, 'paid_search': 0.8, 'organic_search': 0.5,
              'email': 0.6, 'social': 0.2, 'direct': 1.0}

logit = -2.0  # base rate
for ch in channels:
    logit = logit + true_betas[ch] * journey_data[ch]

prob_convert = 1 / (1 + np.exp(-logit))
journey_data['converted'] = np.random.binomial(1, prob_convert)

print(f"Overall conversion rate: {journey_data['converted'].mean():.1%}")
print(f"Total journeys: {n_journeys:,}")
print(f"Total conversions: {journey_data['converted'].sum():,}")
```

```python
# Fit logistic regression
X = journey_data[channels]
y = journey_data['converted']

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

# Feature importance = attribution weights
attribution_weights = pd.Series(model.coef_[0], index=channels).sort_values(ascending=True)

# Compare estimated vs true coefficients
comparison = pd.DataFrame({
    'True Beta': [true_betas[ch] for ch in channels],
    'Estimated Beta': [model.coef_[0][i] for i, ch in enumerate(channels)]
}, index=channels)

print("Attribution Weights (Logistic Regression Coefficients):")
print(comparison.sort_values('Estimated Beta', ascending=False).to_string())

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(channels))
width = 0.35

bars1 = ax.barh(x_pos - width/2, [true_betas[ch] for ch in channels], width, 
                label='True Coefficients', color='steelblue', alpha=0.8)
bars2 = ax.barh(x_pos + width/2, model.coef_[0], width,
                label='Estimated Coefficients', color='coral', alpha=0.8)

ax.set_yticks(x_pos)
ax.set_yticklabels(channels)
ax.set_xlabel('Coefficient Value')
ax.set_title('Regression-Based Attribution: True vs Estimated Weights')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Lightning Demo: BSTS Concept

Bayesian Structural Time Series (BSTS) decompose a time series into interpretable components. This is the backbone of Google's CausalImpact methodology.

**Model structure:**

$$y_t = \mu_t + \gamma_t + \beta^T x_t + \epsilon_t$$

where:
- $\mu_t$ = local level/trend (random walk or local linear trend)
- $\gamma_t$ = seasonal component
- $\beta^T x_t$ = regression on control series
- $\epsilon_t$ = observation noise

For causal inference:
1. Fit the model on pre-intervention data
2. Forecast what would have happened post-intervention (counterfactual)
3. Causal effect = observed - counterfactual

```python
# Visualize the concept of structural time series decomposition

np.random.seed(42)
T = 104  # 2 years of weekly data
t = np.arange(T)

# Trend component: slow upward drift
trend = 100 + 0.3 * t + np.cumsum(np.random.normal(0, 0.5, T))

# Seasonal component: weekly pattern (period = 52 weeks)
seasonality = 10 * np.sin(2 * np.pi * t / 52) + 5 * np.cos(4 * np.pi * t / 52)

# Noise
noise = np.random.normal(0, 3, T)

# Treatment effect (starts at week 78)
treatment_start = 78
treatment_effect = np.zeros(T)
treatment_effect[treatment_start:] = 15  # step increase

# Observed series
y_observed = trend + seasonality + noise + treatment_effect
y_counterfactual = trend + seasonality + noise  # what would have happened

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axes[0].plot(t, y_observed, color='navy', linewidth=1.5, label='Observed')
axes[0].plot(t, y_counterfactual, color='orange', linewidth=1.5, linestyle='--', label='Counterfactual')
axes[0].axvline(x=treatment_start, color='red', linestyle=':', linewidth=2)
axes[0].set_ylabel('Y')
axes[0].set_title('Observed vs Counterfactual')
axes[0].legend()

axes[1].plot(t, trend, color='green', linewidth=1.5)
axes[1].set_ylabel('Y')
axes[1].set_title('Trend Component')

axes[2].plot(t, seasonality, color='purple', linewidth=1.5)
axes[2].set_ylabel('Y')
axes[2].set_title('Seasonal Component')

axes[3].bar(t[treatment_start:], treatment_effect[treatment_start:], color='red', alpha=0.7)
axes[3].axhline(y=0, color='black', linewidth=0.5)
axes[3].set_xlabel('Week')
axes[3].set_ylabel('Y')
axes[3].set_title('Causal Effect (Observed - Counterfactual)')

for ax in axes:
    ax.axvline(x=treatment_start, color='red', linestyle=':', linewidth=1, alpha=0.5)

plt.suptitle('Bayesian Structural Time Series Decomposition', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
```

---

## Capstone Presentation Guidelines

Each student will present their capstone project for **5-7 minutes**, followed by 2-3 minutes of Q&A.

### Recommended Presentation Structure

1. **Business Context** (1 min)
   - What business problem are you solving?
   - Who are the stakeholders? What decisions will this inform?

2. **Data Overview** (1 min)
   - What data did you use? Time range, granularity, channels.
   - Any data quality issues or preprocessing decisions.

3. **Methodology** (1-2 min)
   - Which modeling approach did you use? (OLS, Bayesian, framework)
   - Key modeling choices: adstock, saturation, priors, seasonality.
   - Why did you make these choices?

4. **Results** (1-2 min)
   - Channel contribution estimates (ROAS, mROAS)
   - Key findings and surprises
   - Budget optimization recommendations

5. **Validation & Diagnostics** (1 min)
   - Model fit metrics (R-squared, MAPE)
   - Posterior predictive checks or residual analysis
   - Sensitivity analysis or cross-validation

6. **Recommendations & Next Steps** (1 min)
   - Actionable budget reallocation
   - Proposed measurement plan (experiments)
   - Limitations and future improvements

### Evaluation Criteria

You will be evaluated on five dimensions (see rubric below). Focus on **demonstrating understanding** rather than complexity -- a well-explained simple model beats a poorly understood complex one.

---

## Capstone Rubric

| Criteria | Excellent (4) | Good (3) | Adequate (2) | Needs Work (1) |
|----------|---------------|----------|--------------|----------------|
| **Technical Correctness** | Model is correctly specified; adstock/saturation properly implemented; coefficients are sensible; no major errors in code or methodology | Model is mostly correct with minor issues; core methodology is sound | Model has some specification issues; results may be partially unreliable | Fundamental errors in model setup, data processing, or interpretation |
| **Diagnostics & Validation** | Comprehensive diagnostics: residual analysis, posterior checks, cross-validation, sensitivity analysis; clearly explains what each diagnostic reveals | Good set of diagnostics with meaningful interpretation; covers major validation checks | Basic diagnostics included (e.g., R-squared) but limited interpretation or missing key checks | No or minimal diagnostics; no evidence model was validated |
| **Business Relevance** | Clear business framing; MDE and ROI are grounded in realistic assumptions; recommendations are specific and actionable with clear dollar impact | Good business context; recommendations are reasonable though may lack specificity | Some business framing but disconnected from model results; generic recommendations | No business context; purely technical exercise without practical application |
| **Communication Quality** | Clear, logical flow; visualizations are publication-quality; explains trade-offs and limitations honestly; appropriate level of detail | Well-organized with good visuals; mostly clear explanations; acknowledges some limitations | Understandable but could be better organized; some visuals are confusing or missing labels | Disorganized; hard to follow; poor or missing visualizations; no limitation discussion |
| **Measurement Plan** | Comprehensive plan with specific experiments, timelines, and integration with MMM; demonstrates understanding of triangulation | Good measurement plan with at least one concrete experiment proposal and calibration concept | Basic plan mentions experiments but lacks specifics on design or calibration | No measurement plan or only vague mention of future work |

---

## Measurement Plan Template

Use this template to structure your ongoing measurement program:

### Experiment 1
- **Channel:** [e.g., TV, Paid Social, Display]
- **Hypothesis:** [e.g., "TV drives incremental conversions beyond what MMM estimates"]
- **Design:** [e.g., Geo holdout in 5 DMAs for 8 weeks]
- **MDE:** [e.g., 10% lift in target markets]
- **Timeline:** [e.g., Q2 2025]

### Experiment 2
- **Channel:** [e.g., Paid Search Brand]
- **Hypothesis:** [e.g., "Brand search captures demand that would convert anyway"]
- **Design:** [e.g., Conversion lift study via platform, 4 weeks]
- **MDE:** [e.g., 5% lift in conversions]
- **Timeline:** [e.g., Q3 2025]

### Experiment 3
- **Channel:** [e.g., Email]
- **Hypothesis:** [e.g., "Increased email frequency improves repeat purchase rate"]
- **Design:** [e.g., User-level A/B test, 3 frequency groups, 6 weeks]
- **MDE:** [e.g., 2% lift in repeat purchase rate]
- **Timeline:** [e.g., Q3 2025]

### How Results Feed Back into MMM

After each experiment:
1. Document the causal estimate and confidence interval
2. Compare to the current MMM estimate for that channel
3. If they diverge significantly, update the MMM prior using the experiment result
4. Re-run the MMM with calibrated priors
5. Document the change in budget recommendations

---

## Course Summary

### Week 1: Foundations
- The marketing measurement landscape: MMM, Attribution, Experiments
- Exploratory data analysis for marketing data
- Understanding media transformations: adstock and saturation
- Statistical foundations for causal inference

### Week 2: MMM from Scratch
- OLS-based Media Mix Modeling
- Introduction to Bayesian inference with PyMC
- Bayesian MMM: priors, posteriors, and posterior predictive checks
- Model diagnostics and validation

### Week 3: Production Frameworks + Attribution
- Production MMM frameworks: PyMC-Marketing, Meridian, Robyn
- Budget optimization and scenario planning
- Multi-touch attribution models
- Comparing MMM and attribution results

### Week 4: Experimentation + Capstone
- Power analysis and sample size calculations
- Geo-lift testing and synthetic control methods
- Calibrating MMM with experimental results
- Building a comprehensive measurement program

### Key Takeaway: Triangulation

No single measurement method is perfect. The gold standard in marketing measurement is **triangulation** -- using multiple methods to converge on the truth:

- **MMM** gives you the big picture: which channels drive the most value?
- **Attribution** gives you the detailed picture: which campaigns, creatives, audiences?
- **Experiments** give you causal truth: does this channel actually work?

When all three agree, you can be confident. When they disagree, you've found something worth investigating.

---

## Resources for Continued Learning

### Key Papers

- **"Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects"** (Google, Jin et al., 2017) -- Foundational paper for modern Bayesian MMM with Hill saturation and geometric adstock.

- **"Challenges and Opportunities in Media Mix Modeling"** (Meta/Facebook, 2019) -- Covers practical challenges: multicollinearity, identifiability, and the case for open-source MMM.

- **"Causal Impact of Marketing Actions"** (Brodersen et al., 2015) -- The CausalImpact paper introducing BSTS for causal inference in time series.

- **"Geo-Level Bayesian Hierarchical Media Mix Modeling"** (Google, Zhang & Vaver, 2017) -- Extends MMM to geo-level data for better identification.

- **"Bias Correction for Paid Search in Media Mix Modeling"** (Google, 2017) -- Addresses the endogeneity problem in search advertising.

### Tools & Documentation

- **PyMC-Marketing:** [pymc-marketing.readthedocs.io](https://pymc-marketing.readthedocs.io) -- Python-native Bayesian MMM with full flexibility.

- **Google Meridian:** [github.com/google/meridian](https://github.com/google/meridian) -- Google's next-generation open-source MMM framework.

- **Meta Robyn:** [facebookexperimental.github.io/Robyn](https://facebookexperimental.github.io/Robyn) -- Meta's automated MMM in R with Python wrapper.

- **GeoLift:** [facebookincubator.github.io/GeoLift](https://facebookincubator.github.io/GeoLift) -- Meta's geo experiment design and analysis tool.

- **CausalImpact (R):** [google.github.io/CausalImpact](https://google.github.io/CausalImpact) -- Google's original R package for causal inference.

### Communities

- **PyMC Discourse:** [discourse.pymc.io](https://discourse.pymc.io) -- Active community for Bayesian modeling questions.

- **Marketing Science subreddit:** r/MarketingScience -- Industry discussions and paper reviews.

- **Measure Slack:** Community for marketing measurement practitioners.

- **Marketing Mix Modeling LinkedIn groups:** Several active groups for MMM practitioners.

### Books

- *Bayesian Analysis with Python* (Osvaldo Martin) -- Excellent introduction to Bayesian modeling with PyMC.
- *Statistical Rethinking* (Richard McElreath) -- Best conceptual introduction to Bayesian inference.
- *Causal Inference: The Mixtape* (Scott Cunningham) -- Accessible introduction to causal inference methods.

---

## Final Q&A

Thank you for participating in the Marketing Science Bootcamp!

You now have the tools and frameworks to:

- Build Media Mix Models from scratch (OLS and Bayesian)
- Use production frameworks (PyMC-Marketing, Meridian, Robyn)
- Implement multi-touch attribution models
- Design and analyze geo experiments
- Calibrate models with experimental evidence
- Build a comprehensive, triangulated measurement program

The most important thing is not which tool you use, but that you **think critically about causality**, **validate your results from multiple angles**, and **communicate findings in business terms**.

Good luck with your measurement journeys!

