# Reading 7: Capstone Project Guide

**Estimated time: ~15 minutes**

---

## Project Overview

The capstone project is your opportunity to demonstrate end-to-end Marketing Mix Modeling capability. You will take a real dataset -- either the course-provided dataset or your own BYOD (Bring Your Own Data) -- and produce a complete analysis from data exploration through model building, diagnostics, business recommendations, and a forward-looking measurement plan.

This is not an academic exercise. The capstone is designed to mirror what you would deliver in a professional setting: a rigorous analysis presented to both technical peers and non-technical stakeholders.

---

## Three Deliverables

### 1. Technical Notebook (Primary Deliverable)

A reproducible Jupyter notebook that walks through every stage of the analysis. This is the core of your project. It must run end-to-end without errors and contain both code and narrative explanations.

Use the provided `capstone_template.ipynb` as your starting point. Every section must be completed.

### 2. Executive Presentation (5-7 Slides)

A concise slide deck designed for a non-technical stakeholder audience (e.g., CMO, VP of Marketing). This presentation should:

- Lead with the business question and key findings
- Translate model results into business language (dollars, percentages, recommendations)
- Include visualizations that do not require statistical knowledge to interpret
- Fit within a 10-minute presentation window

### 3. Measurement Plan (1 Page)

A one-page plan outlining the experiments you would run over the next 6 months to validate and improve the MMM. Include specific hypotheses, experimental designs, minimum detectable effects, and estimated durations.

---

## Grading Rubric

| Criteria | Weight | Description |
|---|---|---|
| **Technical Correctness** | 30% | Model is properly specified, code runs without errors, transformations are appropriate, diagnostics are correctly computed |
| **Diagnostics & Validation** | 20% | VIF checked, Durbin-Watson reported, MAPE computed, coefficient signs validated, out-of-sample checks performed |
| **Business Relevance** | 20% | Insights are actionable, recommendations make business sense, budget optimization uses realistic constraints |
| **Communication Quality** | 15% | Notebook is readable, charts are clear and labeled, narrative flows logically, executive presentation is polished |
| **Measurement Plan** | 15% | Experiments are well-designed, calibration approach is sound, hypotheses are specific and testable |

---

## Section-by-Section Guidance

### Section 1: Business Context

Frame the analysis as if you are presenting to a CMO. Answer these questions up front:

- What is the business? What do they sell, and through what channels?
- What is the marketing question? (e.g., "How should we allocate our $10M quarterly budget?")
- What time period does the data cover?
- What decisions will this analysis inform?

Do not jump straight into code. Spend 2-3 paragraphs setting the stage. A strong business context demonstrates that you understand *why* the analysis matters, not just *how* to run it.

### Section 2: Data Overview and EDA

Describe the dataset and explore it visually. Highlight what is unique or challenging:

- How many observations? What is the time granularity (daily, weekly)?
- What media channels are present? What is the spend distribution?
- Are there any anomalies, gaps, or outliers?
- What does the dependent variable (sales/revenue) look like over time?
- Is there visible seasonality, trend, or structural breaks?

Include at least: a time series plot of the target variable, a correlation heatmap, and spend distribution charts.

### Section 3: Data Preparation

Document every transformation and explain *why* you made it:

- **Adstock transformation:** Which decay rate(s) did you use? How did you select them?
- **Saturation transformation:** Hill function, logistic, or diminishing returns curve? What parameters?
- **Control variables:** What did you include (trend, seasonality, holidays, price, macroeconomic indicators)? What did you intentionally exclude, and why?
- **Scaling:** Did you standardize or normalize? Why?

This section should make it possible for another analyst to reproduce your exact data pipeline.

### Section 4: Model Building

At minimum, build an OLS model. Ideally, also build a Bayesian model (PyMC) or use Google Meridian.

- Clearly state the model equation
- Explain your choice of priors (if Bayesian)
- Show the model fitting process and any convergence diagnostics (trace plots, R-hat)
- If you build multiple models, explain how and why they differ

```
Example model equation:

log(Sales_t) = beta_0
             + beta_1 * adstock(TV_t)
             + beta_2 * adstock(Social_t)
             + beta_3 * adstock(Search_t)
             + gamma_1 * trend_t
             + gamma_2 * seasonality_t
             + epsilon_t
```

### Section 5: Model Diagnostics

Every metric in the diagnostics checklist must be reported:

| Diagnostic | What to Check | Acceptable Range |
|---|---|---|
| R-squared (in-sample) | Overall model fit | > 0.80 for weekly data |
| MAPE (out-of-sample) | Prediction accuracy | < 15% |
| VIF (per variable) | Multicollinearity | < 5 (ideally < 3) |
| Durbin-Watson | Autocorrelation of residuals | 1.5 - 2.5 |
| Coefficient signs | Directional correctness | All media coefficients positive |
| Residual plots | Patterns in errors | No visible patterns |

If a diagnostic fails, discuss what it means and what you did (or would do) about it. Acknowledging a problem is far better than ignoring it.

### Section 6: Results and Contribution Analysis

This is where the model results become business insights. Include:

- **Contribution waterfall:** How much of total sales is explained by base, each media channel, and controls?
- **ROI table:** For each channel, show spend, attributed revenue, and ROI (revenue per dollar spent)
- **Response curves:** Plot the diminishing returns curve for each channel, showing where the current spend sits on the curve

```
Example ROI Table:

Channel       Spend ($)     Revenue ($)    ROI     Marginal ROI
---------     ----------    -----------    ----    ------------
TV            2,000,000     4,800,000      2.40    1.20
Paid Social   1,500,000     3,150,000      2.10    1.85
Paid Search   1,000,000     2,700,000      2.70    0.95
Display         500,000       600,000      1.20    0.60
```

### Section 7: Budget Optimization

Move from "what happened" to "what should we do." Provide:

- A recommended budget allocation with specific dollar amounts
- Expected lift compared to the current allocation
- Constraints used (minimum spend per channel, maximum reallocation percentage)
- Sensitivity analysis: how do recommendations change if the model is wrong by +/- 20%?

Do not present a single "optimal" allocation without acknowledging uncertainty. Show a range of scenarios.

### Section 8: Validation and Limitations

Be honest about what the model can and cannot tell you. Address:

- What confounders might the model be missing?
- Are there channels with suspicious coefficients (too high, too low)?
- How sensitive are the results to your adstock and saturation assumptions?
- What would change the recommendations? (e.g., "If the true TV decay rate is 0.5 instead of 0.8, TV ROI drops by 30%")

This section demonstrates analytical maturity. Every model has limitations. Stating them clearly builds credibility.

### Section 9: Measurement Plan

Design specific experiments to validate and improve the model over the next 6 months. For each experiment, specify:

- **Channel to test**
- **Hypothesis** (e.g., "Pausing TV in 4 holdout DMAs for 6 weeks will reduce sales by 12-18%")
- **Design** (geo holdout, A/B test, ghost ad test)
- **Minimum Detectable Effect (MDE):** The smallest effect size the experiment is powered to detect
- **Duration:** How long the test needs to run
- **Estimated cost:** Revenue at risk from holdout

### Section 10: Next Steps

What would you do with three more months of work? Consider:

- Additional data sources to incorporate (competitor spend, economic indicators, weather)
- Model improvements (hierarchical models, time-varying coefficients, longer time series)
- Deeper channel analysis (creative-level effects, audience segmentation)
- Organizational recommendations (data pipeline, measurement cadence, stakeholder reporting)

---

## Tips for the Executive Presentation

**Lead with the business question, not the methodology.** Your first slide should not say "We built a Bayesian regression model with adstock transformations." It should say "We analyzed 2 years of marketing data to determine how to improve marketing ROI by 15%."

**Use the "So What?" test.** For every slide, ask: "If I showed this to a CMO, would they know what to do with this information?" If the answer is no, revise the slide until the implication is clear.

**Include one slide on limitations.** This shows maturity and builds trust. Stakeholders are more likely to act on recommendations when they know you have thought critically about what could be wrong.

**End with clear, ranked recommendations.** Your final slide should list 3-5 specific actions in priority order, with expected impact for each. Make it easy for a decision-maker to say "yes."

**Structure suggestion:**

```
Slide 1: Title + Business Question
Slide 2: Key Findings (3 bullet points maximum)
Slide 3: Channel Performance (ROI chart or contribution waterfall)
Slide 4: Budget Recommendation (current vs optimized allocation)
Slide 5: Expected Impact (projected lift with uncertainty range)
Slide 6: Limitations + What We'd Validate Next
Slide 7: Recommended Next Steps (ranked)
```

---

## Common Mistakes to Avoid

**Not checking coefficient signs.** Media variables should have positive coefficients -- spending more on a channel should not decrease sales. If you see a negative media coefficient, it usually signals multicollinearity, omitted confounders, or an incorrect transformation. Do not proceed without investigating.

**Overfitting to R-squared without checking out-of-sample.** An R-squared of 0.99 is not impressive if the model fails on held-out data. Always report out-of-sample MAPE. Split your data (e.g., last 8-12 weeks as a test set) and evaluate prediction accuracy on the portion the model has not seen.

**Recommendations without uncertainty ranges.** Never say "Reallocate $500K from TV to Social." Instead say "Reallocating $500K from TV to Social is expected to increase revenue by 8-14%, based on model confidence intervals." Decisions without uncertainty are overconfident and will eventually erode trust.

**Missing control variables.** A model without trend and seasonality controls will attribute seasonal sales peaks to whatever marketing happened to be running at the time. At minimum, include: time trend, monthly or weekly seasonality indicators, and any known external factors (promotions, pricing changes, holidays).

**Not running any diagnostics.** A model without diagnostics is an opinion, not an analysis. The diagnostics table in Section 5 is not optional -- every cell must be filled with a value and an interpretation.

---

## Timeline Reminder

| Milestone | Deadline | What's Due |
|---|---|---|
| BYOD data identified | Week 2, Session 4 | Dataset uploaded, initial EDA started |
| First model built | Week 3, Session 6 | Working OLS model with basic diagnostics |
| Capstone complete | Session 8 | All three deliverables submitted |

Do not wait until the final week to start. The most successful projects iterate -- build a rough model early, get feedback, and refine. A polished model built in one night will almost always be weaker than one that evolved over multiple sessions.

---

*Good luck. Build something you would be proud to present in a job interview.*
