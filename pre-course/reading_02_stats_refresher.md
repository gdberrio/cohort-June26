# Reading 2: Statistics Refresher for Marketing Science

**Estimated time: ~90 minutes (text + video)**

---

## Why This Matters

Marketing Mix Modeling is built on regression analysis. Before we start building models, we need to make sure our statistical foundations are solid. This reading covers the key concepts you'll use every day in the course.

## 1. Linear Regression (OLS)

### The Model

Ordinary Least Squares (OLS) regression finds the line (or hyperplane) that minimizes the sum of squared residuals:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
```

Where:
- **y** = dependent variable (e.g., Sales)
- **βᵢ** = coefficients (the effect of each variable)
- **xᵢ** = independent variables (e.g., TV spend, price)
- **ε** = error term (what the model can't explain)

### Coefficient Interpretation

- **β₁ = 2.5** means: a one-unit increase in x₁ is associated with a 2.5-unit increase in y, holding all other variables constant
- In MMM: "For every additional unit of transformed TV spend, sales increase by 2.5 units"

### Key Assumptions (and what happens when they're violated)

| Assumption | What It Means | MMM Violation Risk | Diagnostic |
|---|---|---|---|
| Linearity | Relationship is linear | Media effects are nonlinear → use saturation transforms | Residual plots |
| No multicollinearity | Predictors aren't too correlated | Channels often co-vary (e.g., TV + radio campaigns together) | VIF |
| Homoscedasticity | Constant variance of errors | Variance may change with scale | Breusch-Pagan test |
| No autocorrelation | Errors are independent | Time series data is almost always autocorrelated | Durbin-Watson |
| Normality of errors | Errors are normally distributed | Usually acceptable with enough data | Q-Q plot |

### Key Diagnostics

- **R²:** Proportion of variance explained (0.7+ is decent for MMM)
- **Adjusted R²:** R² penalized for number of predictors
- **VIF (Variance Inflation Factor):** VIF > 5 suggests multicollinearity; VIF > 10 is serious
- **Durbin-Watson:** Ranges from 0 to 4; values near 2 indicate no autocorrelation; below 1.5 suggests positive autocorrelation
- **MAPE:** Mean Absolute Percentage Error (< 10% is good for MMM)
- **NRMSE:** Normalized Root Mean Squared Error
- **AIC/BIC:** Information criteria for model comparison (lower is better)

## 2. Transformations

### Log Transformations

Taking the natural log of variables is common in MMM:
- **log(y):** Useful when the DV is right-skewed or when you want to model percentage changes
- **log(x):** Useful when the relationship is concave (diminishing returns)
- **log-log model:** Both y and x are logged; β₁ is interpreted as an elasticity (% change in y per % change in x)

### Power Transformations

- **x^n** where 0 < n < 1 compresses large values → models diminishing returns
- More on this when we cover saturation curves in Week 2

## 3. Time Series Basics

Marketing data is time series data. Key concepts:

### Trend
- Long-term direction (upward, downward, flat)
- Must be controlled for in MMM (otherwise media variables get credit for organic growth)

### Seasonality
- Recurring patterns at fixed intervals (weekly, monthly, quarterly)
- Q4 holiday spike in retail, summer in travel, etc.
- Model with dummy variables or Fourier terms

### Autocorrelation
- Current values are correlated with past values
- Sales this week are partly determined by sales last week
- Causes standard errors to be underestimated → inflated significance
- Solution: include lagged DV, use Newey-West standard errors, or model explicitly

### Stationarity
- A stationary series has constant mean and variance over time
- Non-stationary data can produce spurious regression results
- Tests: Augmented Dickey-Fuller (ADF), KPSS

## 4. Correlation vs. Causation

This is perhaps the most important concept in marketing measurement.

### Correlation ≠ Causation

**Example:** Ice cream sales and drowning deaths are highly correlated. Does ice cream cause drowning? No — both are caused by summer heat (a confounder).

**Marketing example:** You observe that Meta spend and sales are positively correlated. Does Meta *cause* sales? Not necessarily:
- Both might increase during seasonal peaks (confounder)
- You might increase Meta spend *because* sales are up (reverse causation)
- Meta might only reach people who would have bought anyway (no incrementality)

### What MMM Can and Cannot Tell You

- **Can:** Estimate statistical associations between media inputs and outcomes, controlling for observed confounders
- **Cannot:** Definitively prove causation without experimental validation
- **Bayesian MMM helps by:** Incorporating prior knowledge (including experimental results) to produce more credible estimates

### The Path to Causation

1. **Observational data + regression** → Association (MMM)
2. **+ Controlling for confounders** → Stronger association
3. **+ Experimental validation** → Causal estimate
4. **+ Feeding experiments back into MMM** → Calibrated causal model

This is why Week 4 covers experimentation and calibration.

## Practice Questions

1. What does a VIF of 8.5 tell you about a variable in your model?
2. If the Durbin-Watson statistic is 0.9, what does this suggest?
3. Why might a log-transformed dependent variable be appropriate for a sales model?
4. Give a marketing example of confounding that could bias MMM results.
5. What is the difference between R² and Adjusted R²?

---

*Next reading: "Introduction to Causal Inference for Marketers"*
