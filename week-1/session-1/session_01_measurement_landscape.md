# Session 01 Measurement Landscape

# Session 1: The Marketing Measurement Landscape & Statistical Foundations

**Marketing Science Bootcamp | Week 1**

---

## Learning Objectives

By the end of this session, you will be able to:

1. **Distinguish** between the three pillars of marketing measurement: MMM, Attribution, and Experiments
2. **Construct** causal DAGs (Directed Acyclic Graphs) to reason about marketing effects
3. **Run and interpret** a basic OLS regression for marketing data
4. **Diagnose** common regression pitfalls: multicollinearity, autocorrelation, and omitted variable bias

## Agenda

| Time | Topic |
|------|-------|
| 10 min | The Measurement Trinity: MMM vs. Attribution vs. Experiments |
| 15 min | DAGs for Marketing: Visualizing Causal Assumptions |
| 20 min | Regression Refresher: Building a Naive Model |
| 20 min | Why the Naive Model Fails: Diagnostics Deep-Dive |
| 15 min | Exercise: Build Your First Regression |
| 10 min | Key Takeaways & Preview of Session 2 |

---

## 1. Setup

```python
# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical modeling
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

# Causal graphs
import networkx as nx

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

%matplotlib inline

print('All libraries loaded successfully.')
```

```python
# Load the dataset
data = pd.read_excel('../../data/MMM_Workshop_Data.xlsx', sheet_name='Data')

print(f'Dataset shape: {data.shape}')
print(f'Columns: {data.columns.tolist()}')
data.head()
```

---

## 2. The Measurement Trinity

Marketing measurement rests on **three complementary approaches**. No single method provides the complete picture -- the real power comes from **triangulation**.

### Marketing Mix Modeling (MMM)
- **What**: A top-down, aggregate regression model that estimates the impact of each marketing channel on a business outcome (e.g., sales) using historical time-series data.
- **Strengths**: Captures long-term and offline effects; covers all channels in one model; privacy-friendly (no user-level data needed).
- **Limitations**: Requires 2-3 years of data; limited granularity; can struggle to separate correlated channels.
- **When to use**: Strategic planning, budget allocation across channels, understanding macro trends.

### Multi-Touch Attribution (MTA)
- **What**: A bottom-up, user-level model that assigns fractional credit to each touchpoint in a customer's journey.
- **Strengths**: Granular, real-time, captures user-level paths.
- **Limitations**: Digital-only; biased by observable data (misses offline, view-through); degraded by privacy regulations (iOS 14.5+, cookie deprecation).
- **When to use**: Real-time digital optimization, intra-channel allocation (e.g., which Meta ad sets perform best).

### Experiments (A/B Tests & Geo-Lift)
- **What**: Randomized controlled trials -- either at the user level (A/B test) or geographic level (geo-lift) -- that measure the causal incremental impact of a treatment.
- **Strengths**: Gold standard for causal inference; unambiguous incrementality.
- **Limitations**: Expensive; limited scope (one channel/tactic at a time); requires holdout groups.
- **When to use**: Validating MMM coefficients, measuring incrementality for specific tactics, calibrating other models.

### Triangulation: Bringing It All Together

| Dimension | MMM | MTA | Experiments |
|-----------|-----|-----|-------------|
| Granularity | Aggregate | User-level | Varies |
| Time horizon | Long-term | Short-term | Point-in-time |
| Channels covered | All | Digital only | One at a time |
| Causal rigor | Moderate (observational) | Low (correlational) | High (randomized) |
| Privacy impact | Low | High | Low |

> **Key insight**: Use experiments to *calibrate* your MMM, use MTA for *real-time optimization*, and use MMM for *strategic budget allocation*. When all three agree on a channel's value, you can be highly confident in the finding.

---

## 3. DAGs for Marketing: Visualizing Causal Assumptions

A **Directed Acyclic Graph (DAG)** is a visual tool for encoding our causal assumptions about how variables relate to each other. Every model implies a DAG -- the question is whether we make it explicit.

DAGs help us:
- Identify what we need to **control for** (confounders)
- Identify what we should **not** control for (mediators, colliders)
- Reason about **omitted variable bias**

Let's build three increasingly realistic DAGs for a marketing model.

### DAG 1: The Naive Model

This is the simplest assumption: each media channel directly causes sales, and there are no other relationships.

```python
# DAG 1: Naive Model -- media channels directly drive sales
G1 = nx.DiGraph()
G1.add_edges_from([
    ('TV_Spend', 'Sales'),
    ('Meta_Spend', 'Sales'),
    ('Radio_Spend', 'Sales')
])

pos1 = nx.spring_layout(G1, seed=42)

fig, ax = plt.subplots(figsize=(8, 5))
nx.draw(G1, pos1, with_labels=True, node_color='lightblue', node_size=3000,
        font_size=10, font_weight='bold', arrows=True, arrowsize=20, ax=ax)
ax.set_title('DAG 1: Naive Model -- Direct Effects Only', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Problem**: This DAG assumes no confounders. In reality, many factors influence both spend and sales simultaneously.

### DAG 2: With a Confounder (Seasonality)

Seasonality is a classic **confounder**: it drives both sales (people buy more at certain times of year) and spend (marketers spend more during peak seasons). If we ignore seasonality, we get **omitted variable bias** -- we attribute seasonal sales lifts to media spend.

```python
# DAG 2: With Confounder -- Seasonality affects both Spend and Sales
G2 = nx.DiGraph()
G2.add_edges_from([
    ('TV_Spend', 'Sales'),
    ('Meta_Spend', 'Sales'),
    ('Radio_Spend', 'Sales'),
    ('Seasonality', 'Sales'),
    ('Seasonality', 'TV_Spend')
])

pos2 = nx.spring_layout(G2, seed=42)

fig, ax = plt.subplots(figsize=(8, 5))
# Draw confounder node in a different color
node_colors = ['#FF9999' if node == 'Seasonality' else 'lightblue' for node in G2.nodes()]
nx.draw(G2, pos2, with_labels=True, node_color=node_colors, node_size=3000,
        font_size=10, font_weight='bold', arrows=True, arrowsize=20, ax=ax)
ax.set_title('DAG 2: With Confounder (Seasonality)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Key insight**: Because `Seasonality` is a confounder (it causes both `TV_Spend` and `Sales`), we **must** control for it. Otherwise, our TV coefficient will be biased upward -- it will pick up seasonal sales that have nothing to do with TV advertising.

### DAG 3: With a Mediator (Brand Search)

Some media channels work **indirectly**. For instance, TV advertising might drive people to Google your brand, and then that brand search leads to a purchase. Brand search is a **mediator**.

```python
# DAG 3: With Mediator -- TV drives Brand Search, which drives Sales
G3 = nx.DiGraph()
G3.add_edges_from([
    ('TV_Spend', 'Brand_Search'),
    ('Brand_Search', 'Sales'),
    ('TV_Spend', 'Sales'),
    ('Meta_Spend', 'Sales'),
    ('Radio_Spend', 'Sales'),
    ('Seasonality', 'Sales'),
    ('Seasonality', 'TV_Spend')
])

pos3 = nx.spring_layout(G3, seed=42)

fig, ax = plt.subplots(figsize=(10, 6))
# Color mediator differently
node_colors = []
for node in G3.nodes():
    if node == 'Seasonality':
        node_colors.append('#FF9999')   # confounder in red
    elif node == 'Brand_Search':
        node_colors.append('#99FF99')   # mediator in green
    else:
        node_colors.append('lightblue') # standard nodes

nx.draw(G3, pos3, with_labels=True, node_color=node_colors, node_size=3000,
        font_size=10, font_weight='bold', arrows=True, arrowsize=20, ax=ax)
ax.set_title('DAG 3: With Mediator (Brand Search) & Confounder (Seasonality)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Key insight**: If we include both `TV_Spend` and `Brand_Search` in the same regression, the TV coefficient captures only the **direct** effect. The **indirect** effect (TV -> Brand Search -> Sales) gets absorbed by the Brand Search coefficient. This is fine if we want the direct effect, but it means the TV coefficient *underestimates* TV's total impact.

> **Rule of thumb for DAGs:**
> - **Confounders**: Always control for them (or you get bias).
> - **Mediators**: Only control for them if you want the *direct* effect. Omit them if you want the *total* effect.
> - **Colliders**: Never control for them (or you *create* bias).

---

## 4. Regression Refresher: The Naive Model

Now let's build an actual regression model using our data. We will start with the simplest possible model -- regressing Sales on a few raw media spend variables -- and then diagnose what goes wrong.

### OLS Assumptions (Brief Recap)

For Ordinary Least Squares (OLS) to produce **unbiased, consistent, and efficient** estimates, we need:

1. **Linearity**: The relationship between X and Y is linear (in parameters).
2. **No perfect multicollinearity**: The independent variables are not perfectly correlated.
3. **Exogeneity**: E(error | X) = 0 -- no omitted variables, no reverse causality.
4. **Homoscedasticity**: Constant variance of errors.
5. **No autocorrelation**: Errors are independent across observations (critical for time-series).
6. **Normality of errors**: Needed for valid hypothesis testing in small samples.

In practice with marketing data, **assumptions 2, 3, and 5** are almost always violated in naive models. Let's see this in action.

```python
# Quick look at key variables
key_vars = ['Sales_Volume_Total', 'TV_Spends', 'Meta1_Spends', 'Radio_Spends']
data[key_vars].describe()
```

```python
# Define the dependent variable and independent variables
y = data['Sales_Volume_Total']

# Naive model: raw media spends only, no controls
X_naive = data[['TV_Spends', 'Meta1_Spends', 'Radio_Spends']]
X_naive = sm.add_constant(X_naive)  # adds intercept column

# Fit OLS regression
naive_model = sm.OLS(y, X_naive).fit()

# Display the full summary
print(naive_model.summary())
```

### Interpreting the Results

Look at the output above and consider:

- **R-squared**: How much variance in Sales does this model explain?
- **Coefficient signs**: Do they all make business sense? Are any negative? A negative coefficient on a media spend variable would imply *more spending = less sales*, which is rarely the true causal effect.
- **Statistical significance**: Which variables have p-values < 0.05?
- **Durbin-Watson statistic**: Is it close to 2.0 (no autocorrelation), or does it deviate?

> **The "negative coefficient" problem**: If you see a negative coefficient on a media variable, that is usually a sign of **multicollinearity** or **omitted variable bias**, not a true negative effect. The model is confusing the correlated movements of the variables.

---

## 5. Why the Naive Model Fails: Diagnostics Deep-Dive

Let's systematically check the three most common problems in marketing regression models.

### 5.1 Multicollinearity -- Variance Inflation Factor (VIF)

**Multicollinearity** occurs when independent variables are highly correlated with each other. In marketing, this is extremely common: brands tend to increase *all* media spend during the same periods (e.g., holidays, product launches).

**Why it matters**: Multicollinearity inflates the standard errors of coefficients, making them unstable and unreliable. Coefficients can flip signs or become non-significant even when the underlying effect is real.

**VIF interpretation**:
- VIF = 1: No correlation with other variables
- VIF = 1-5: Moderate (usually acceptable)
- VIF > 5: High multicollinearity (concerning)
- VIF > 10: Severe multicollinearity (action needed)

```python
# Compute VIF for each variable in the naive model
vif_data = pd.DataFrame()
vif_data['Variable'] = X_naive.columns
vif_data['VIF'] = [variance_inflation_factor(X_naive.values, i) for i in range(X_naive.shape[1])]

print('Variance Inflation Factors (VIF):')
print('-' * 40)
print(vif_data.to_string(index=False))
print()

# Flag high-VIF variables
high_vif = vif_data[vif_data['VIF'] > 5]
if len(high_vif) > 0:
    print('WARNING: Variables with VIF > 5 (high multicollinearity):')
    print(high_vif.to_string(index=False))
else:
    print('No variables with VIF > 5 detected.')
```

```python
# Correlation heatmap of the independent variables
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = data[['TV_Spends', 'Meta1_Spends', 'Radio_Spends']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, fmt='.3f', ax=ax)
ax.set_title('Correlation Matrix of Media Spend Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 5.2 Autocorrelation -- Durbin-Watson Statistic

**Autocorrelation** (serial correlation) means that the model's residuals are correlated across time. In time-series marketing data, this is extremely common because:
- Sales have momentum (a good week tends to be followed by another good week)
- Seasonality patterns create systematic errors if not controlled for
- Ad effects carry over from one period to the next (adstock)

**Why it matters**: Autocorrelation does not bias the coefficients, but it makes the standard errors *too small*, leading to **overconfident** p-values and confidence intervals.

**Durbin-Watson interpretation**:
- DW close to 2.0: No autocorrelation
- DW close to 0: Positive autocorrelation (most common in marketing data)
- DW close to 4: Negative autocorrelation (rare)

```python
# Durbin-Watson test for autocorrelation
dw_stat = durbin_watson(naive_model.resid)
print(f'Durbin-Watson statistic: {dw_stat:.4f}')
print()

if dw_stat < 1.5:
    print('DIAGNOSIS: Strong positive autocorrelation detected (DW < 1.5).')
    print('The residuals are correlated over time -- standard errors are unreliable.')
elif dw_stat < 1.8:
    print('DIAGNOSIS: Mild positive autocorrelation (1.5 < DW < 1.8).')
    print('Some concern -- consider adding time-based controls or using Newey-West standard errors.')
else:
    print('DIAGNOSIS: No strong evidence of autocorrelation (DW close to 2.0).')
```

```python
# Visualize residuals over time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals over time
axes[0].plot(naive_model.resid.values, marker='o', markersize=3, linewidth=0.8)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].set_title('Residuals Over Time', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Observation (time period)')
axes[0].set_ylabel('Residual')

# Residuals vs fitted values
axes[1].scatter(naive_model.fittedvalues, naive_model.resid, alpha=0.6, edgecolors='k', linewidths=0.5)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[1].set_title('Residuals vs Fitted Values', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residual')

plt.tight_layout()
plt.show()
```

### 5.3 Omitted Variable Bias

**Omitted Variable Bias (OVB)** occurs when a variable that affects both the dependent variable (Sales) and one or more independent variables (media spend) is left out of the model.

In our naive model, we omitted:

- **Seasonality**: Sales and spend both spike during holidays
- **Price / promotions**: Lower prices drive sales independent of media
- **Competitive activity**: Competitor spend affects our sales
- **Macroeconomic factors**: Inflation, economic cycles

The direction and magnitude of OVB depends on:

$$\text{Bias} = \beta_{\text{omitted}} \times \frac{\text{Cov}(X_{\text{included}}, X_{\text{omitted}})}{\text{Var}(X_{\text{included}})}$$

If both the omitted variable's effect on Y and its correlation with X are positive, the included variable's coefficient is **biased upward** (overestimated). This is the classic seasonality problem in MMM.

```python
# Demonstrate OVB: Compare naive model with a model that includes a control variable
# Let's check which control variables are available in our dataset
print('Available columns in the dataset:')
print(data.columns.tolist())
print()

# Check for potential control variables
potential_controls = ['Average_Price_Total', 'Inflation_Rate']
available_controls = [col for col in potential_controls if col in data.columns]
print(f'Available control variables: {available_controls}')
```

```python
# Model WITH controls to demonstrate OVB
if available_controls:
    X_with_controls = data[['TV_Spends', 'Meta1_Spends', 'Radio_Spends'] + available_controls]
    X_with_controls = sm.add_constant(X_with_controls)
    
    model_with_controls = sm.OLS(y, X_with_controls).fit()
    
    # Compare coefficients
    comparison = pd.DataFrame({
        'Naive Model': naive_model.params,
        'With Controls': model_with_controls.params
    })
    
    print('Coefficient Comparison: Naive vs. Model with Controls')
    print('=' * 60)
    print(comparison.to_string())
    print()
    print('Notice how the media coefficients change when we add controls.')
    print('This difference is an estimate of the omitted variable bias.')
else:
    print('No control variables found. Check the column names above and add appropriate controls.')
```

### Summary of Diagnostic Findings

| Diagnostic | What We Found | Implication |
|-----------|---------------|-------------|
| **VIF (Multicollinearity)** | Check values above | High VIF -> unstable coefficients |
| **Durbin-Watson (Autocorrelation)** | Check value above | DW far from 2 -> unreliable standard errors |
| **OVB (Omitted Variables)** | Coefficients change when controls added | Naive model coefficients are biased |

**Bottom line**: The naive model is not trustworthy for decision-making. This is why MMM requires sophisticated techniques like **adstock transformations**, **saturation curves**, and **proper control variables** -- all of which we will build in upcoming sessions.

---

## 6. Exercise: Build Your First Regression

Now it's your turn. Using the dataset, build your own regression model and diagnose it.

### Instructions

1. **Choose your variables**: Pick a different combination of media spend variables (and optionally, control variables) as predictors of `Sales_Volume_Total`.
2. **Fit an OLS model**: Use `sm.OLS()` to fit the model.
3. **Check diagnostics**: Compute VIF and the Durbin-Watson statistic.
4. **Reflect**: Do the coefficients make business sense? What would you change?

**Available media variables** (check `data.columns` for the full list):
- `TV_Spends`, `Radio_Spends`, `Outdoor_Spends`
- `Meta1_Spends`, `Meta2_Spends`, `Youtube_Spends`
- `Google_Display_Spend`, `Direct_Display_Spend`
- `Paid_Search_Spends`, `Programmatic_Display_Spends`, `Programmatic_Video_Spends`
- `Influencer_Marketing_Spends`

**Available control variables**:
- `Average_Price_Total`, `Inflation_Rate`

```python
# EXERCISE STEP 1: Define your variables
# TODO: Add your variables here -- pick a different combination than the naive model

y_exercise = data['Sales_Volume_Total']

# Example: uncomment and modify the line below
# X_exercise = data[['TV_Spends', 'Outdoor_Spends', 'Meta2_Spends', 'Average_Price_Total']]

X_exercise = data[[
    # TODO: Add your variables here
]]

X_exercise = sm.add_constant(X_exercise)
```

```python
# EXERCISE STEP 2: Fit the OLS model
# TODO: Fit your model and print the summary

# exercise_model = sm.OLS(y_exercise, X_exercise).fit()
# print(exercise_model.summary())
```

```python
# EXERCISE STEP 3: Compute diagnostics
# TODO: Compute VIF for your model

# vif_exercise = pd.DataFrame()
# vif_exercise['Variable'] = X_exercise.columns
# vif_exercise['VIF'] = [variance_inflation_factor(X_exercise.values, i) for i in range(X_exercise.shape[1])]
# print(vif_exercise.to_string(index=False))
```

```python
# EXERCISE STEP 4: Check Durbin-Watson
# TODO: Compute and interpret the Durbin-Watson statistic for your model

# dw_exercise = durbin_watson(exercise_model.resid)
# print(f'Durbin-Watson: {dw_exercise:.4f}')
```

```python
# EXERCISE STEP 5: Reflect
# TODO: Write your observations as comments

# Questions to consider:
# 1. Do all coefficients have the expected sign (positive for media spend)?
# 2. Which variables are statistically significant?
# 3. Is the R-squared better or worse than the naive model?
# 4. Are there multicollinearity issues (VIF > 5)?
# 5. Is there autocorrelation (DW far from 2)?
# 6. What would you change in the next iteration?
```

---

## 7. Key Takeaways

1. **The Measurement Trinity is essential for robust marketing analytics.** MMM, Attribution, and Experiments each have strengths and blindspots. Triangulation -- using multiple methods and checking that they converge -- gives you the most confidence in your findings.

2. **Every regression model encodes causal assumptions -- make them explicit with DAGs.** Before fitting a model, draw the DAG. It forces you to think about what to control for (confounders), what not to control for (mediators), and what is missing (omitted variables). A model is only as good as the assumptions behind it.

3. **Naive regression on raw marketing data is almost always misleading.** Multicollinearity, autocorrelation, and omitted variable bias conspire to produce coefficients that look plausible but are wrong. This is why MMM requires feature engineering (adstock, saturation curves) and careful model specification -- which is exactly what we will build in the sessions ahead.

---

## 8. Preview of Session 2: Feature Engineering for MMM

In the next session, we will tackle the core feature engineering techniques that transform a naive regression into a proper Marketing Mix Model:

- **Adstock transformations**: How to model the carryover effect of advertising (geometric decay, Weibull)
- **Saturation curves (Hill function)**: Capturing diminishing returns to media spend
- **Hyperparameter search**: Finding the right adstock decay rate and saturation shape for each channel
- **Correlation-based feature selection**: Choosing the best transformation for each variable

These techniques address the fundamental problems we identified today and are the building blocks of every modern MMM.

**Preparation**: Review the concepts of geometric decay and the S-curve (Hill function). Think about which channels you expect to have longer carryover effects (hint: TV vs. paid search).

