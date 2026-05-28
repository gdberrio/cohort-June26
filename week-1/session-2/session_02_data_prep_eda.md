# Session 02 Data Prep Eda

# Session 2: Data Preparation & EDA Workshop

**Marketing Science Bootcamp -- Week 1, Live Session**

---
## Setup

```python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.stattools import ccf

# Add project root to path so we can import shared utilities
sys.path.insert(0, '../..')
from utils.eda_utils import *
from utils.eda_utils import load_config

%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
```

```python
# Load the workshop dataset
data = pd.read_excel('../../data/MMM_Workshop_Data.xlsx', sheet_name='Data')
print(f'Shape: {data.shape}')
data.head()
```

```python
# Load the configuration file
config = load_config('../../data/config_file.csv')
print('Config loaded successfully.')
```

---
## Recap: Session 1 Key Points

Before we dive in, let's recap the key takeaways from Session 1:

1. **MMM is a regression-based approach** that quantifies the impact of marketing activities on a KPI (e.g., sales).
2. **Data is the foundation** -- garbage in, garbage out. Understanding your data before modeling is non-negotiable.
3. **Key data requirements**: weekly/daily granularity, sufficient history (2+ years ideal), spend by channel, KPI, and control variables.
4. **The modeling pipeline**: Data Collection --> EDA --> Data Preparation --> Feature Engineering --> Model Building --> Validation --> Optimization.

Today we focus on **EDA** and **Data Preparation** -- steps 2 and 3 of the pipeline.

---
## 1. Full EDA Pipeline

We will walk through the complete EDA pipeline using our workshop data and the config-driven approach.

### 1.1 Understanding the Config

The config file defines the structure of our model: which variable is the dependent variable,
which are paid media spends, which are organic/control, etc. This config-driven approach
ensures reproducibility and makes it easy to swap datasets.

```python
# Inspect the config contents
print('Dependent variable:', config.get('dependent_var'))
print()
print('Paid media spends:', config.get('paid_media_spends'))
print()
print('Organic variables:', config.get('organic_vars'))
print()
print('Control variables:', config.get('control_vars'))
```

### 1.2 Summary Statistics

```python
stats = summary_statistics(data)
stats
```

**Discussion points:**
- Look at the min values -- any variables that are always zero or have many zeros?
- Compare mean vs median -- large gaps indicate skewness.
- Check the max values for potential outliers.

### 1.3 Cross-Correlation (CCF) Analysis

CCF plots help us understand the **lag structure** between media spend and sales.
The peak of the CCF tells us how many periods (weeks) it takes for a channel's
spend to show maximum correlation with sales.

```python
# Full CCF grid
ccf_plot_all(data)
```

**What we see:**
- TV typically peaks at lag 0-2 -- broad reach channels act relatively quickly.
- Digital channels often peak at lag 0-1 -- faster path to conversion.
- If a variable shows no clear peak, it may not have a direct effect on sales, or the relationship may be nonlinear.

These lag insights will directly inform our **adstock transformation** parameters in Week 2.

### 1.4 Correlation Heatmap

The correlation heatmap reveals pairwise relationships and potential multicollinearity.

```python
# Select numeric columns for the heatmap
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
correlation_heatmap(data[numeric_cols])
```

**Discussion: Multicollinearity concerns**

- Correlations above **0.7** between independent variables are a red flag.
- When two spend variables are highly correlated, the model cannot reliably separate their individual effects.
- Solutions: combine correlated variables, drop one, or use ridge regression.
- Note which pairs are concerning -- we will revisit this during model building.

### 1.5 Dual-Axis Time Series Charts

Visual inspection of spend vs. KPI over time reveals patterns that pure statistics can miss.

```python
# Dual-axis charts for key spend variables
spend_vars = [
    'TV_Spends',
    'Digital_Spends',
    'Sponsorship_Spends',
    'Content_Spends',
    'Online_Spends',
    'Print_Spends'
]

# Filter to columns that actually exist in the data
spend_vars = [c for c in spend_vars if c in data.columns]

dual_axis_chart_all(data, kpi_col='Sales_Volume_Total', spend_cols=spend_vars)
```

---
## 2. Data Preparation for MMM

Now that we understand our data, we need to prepare it for modeling. This section covers
the key data preparation steps for MMM.

### 2.1 Missing Value Handling

Missing values need to be addressed before modeling. The strategy depends on the variable type
and the nature of the missingness.

```python
# Check for missing values
missing = data.isnull().sum()
missing_pct = (data.isnull().sum() / len(data)) * 100

missing_report = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})

# Show only columns with missing values
missing_report[missing_report['Missing Count'] > 0]
```

```python
# Common fill strategies for MMM data:

# Strategy 1: Fill spend variables with 0 (no spend = $0)
# spend_columns = config.get('paid_media_spends', [])
# data[spend_columns] = data[spend_columns].fillna(0)

# Strategy 2: Forward fill for slowly-changing variables (e.g., price, distribution)
# data['Price'] = data['Price'].fillna(method='ffill')

# Strategy 3: Interpolate for KPI if only a few values missing
# data['Sales_Volume_Total'] = data['Sales_Volume_Total'].interpolate(method='linear')

print('Missing values after handling:')
print(data.isnull().sum().sum(), 'total missing values remaining')
```

### 2.2 Zero-Inflation Check

Many media channels have intermittent spend ("flighting"). It is important to understand
how often each channel has zero spend, because this affects model estimation.

```python
# Check zero-inflation for spend variables
numeric_cols = data.select_dtypes(include=[np.number]).columns
zero_counts = (data[numeric_cols] == 0).sum()
zero_pct = ((data[numeric_cols] == 0).sum() / len(data) * 100).round(1)

zero_report = pd.DataFrame({
    'Zero Count': zero_counts,
    'Zero %': zero_pct,
    'Total Rows': len(data)
})

# Show variables with significant zero-inflation (>20%)
zero_report[zero_report['Zero %'] > 20].sort_values('Zero %', ascending=False)
```

**Discussion:** Variables with very high zero-inflation (>50%) may be difficult to model
with a simple linear approach. Consider:
- Aggregating weekly to monthly to reduce zeros.
- Using a flag variable (on/off) instead of raw spend.
- Grouping low-spend channels together.

### 2.3 Creating Control Variables

Control variables capture non-media effects on sales: seasonality, trend, holidays, promotions, etc.
If we omit important controls, their effect gets incorrectly attributed to media (omitted variable bias).

```python
# Create month number from date column
data['month_num'] = pd.to_datetime(data['Month']).dt.month

# Create monthly seasonality dummies (drop_first to avoid multicollinearity)
month_dummies = pd.get_dummies(data['month_num'], prefix='month', drop_first=True)

# Create a linear trend variable
data['trend'] = range(1, len(data) + 1)

# Combine with main data
data = pd.concat([data, month_dummies], axis=1)

print('New columns added:')
print('- month_num: numeric month (1-12)')
print(f'- {len(month_dummies.columns)} month dummies: {list(month_dummies.columns)}')
print('- trend: linear trend (1, 2, 3, ...)')
print(f'\nData shape after adding controls: {data.shape}')
```

### 2.4 Log Transforms

In MMM, we often consider log-transforming the dependent variable (and sometimes independent variables).

**When to use log-transformed DV vs. raw:**

| Aspect | Raw DV | Log-transformed DV |
|--------|--------|---------------------|
| Interpretation | Coefficients = absolute change in sales per unit change in X | Coefficients = % change in sales per unit change in X |
| When to use | Sales are roughly normally distributed | Sales are right-skewed; you want multiplicative effects |
| Benefit | Simpler interpretation | Handles heteroscedasticity; coefficients are elasticities |
| Caution | June have heteroscedastic residuals | Cannot handle zero values (use log1p instead) |

```python
# Log-transform the dependent variable
data['log_sales'] = np.log1p(data['Sales_Volume_Total'])

# Compare distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data['Sales_Volume_Total'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Raw Sales Distribution')
axes[0].set_xlabel('Sales Volume')
axes[0].set_ylabel('Frequency')

axes[1].hist(data['log_sales'], bins=30, color='coral', edgecolor='white')
axes[1].set_title('Log-Transformed Sales Distribution')
axes[1].set_xlabel('log(1 + Sales Volume)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(f'Raw Sales -- Skewness: {data["Sales_Volume_Total"].skew():.3f}')
print(f'Log Sales -- Skewness: {data["log_sales"].skew():.3f}')
```

---
## 3. Config-Driven Approach

A config-driven approach makes your MMM pipeline **reproducible** and **scalable**.
Instead of hard-coding variable names and transformation parameters, we store them in
a configuration file.

### 3.1 Config File Structure

The `config_file.csv` typically contains:

| Column | Description |
|--------|-------------|
| variable_name | Name of the column in the dataset |
| variable_type | Category: dependent, paid_media, organic, control, date |
| channel_group | Grouping: tv_vars, digital_vars, print_vars, etc. |
| transform_type | Adstock type: geometric, weibull, etc. |
| decay_min / decay_max | Range for decay parameter search |
| saturation_type | Hill, log, etc. |

This structure drives the entire modeling pipeline -- from feature engineering to optimization.

```python
# Load and inspect the raw config file
config_df = pd.read_csv('../../data/config_file.csv')
config_df
```

```python
# Show how config categorizes variables
if 'variable_type' in config_df.columns:
    print('Variable categories:')
    print(config_df.groupby('variable_type')['variable_name'].apply(list).to_string())
elif 'channel_group' in config_df.columns:
    print('Channel groups:')
    print(config_df.groupby('channel_group')['variable_name'].apply(list).to_string())
else:
    print('Config columns:', config_df.columns.tolist())
    print()
    print(config_df.head(10))
```

**Why this matters for transformation parameter ranges:**

- **TV variables** typically have longer decay (0.5-0.9) -- effects persist over multiple weeks.
- **Digital variables** typically have shorter decay (0.1-0.5) -- effects fade faster.
- **Print/OOH** can vary widely depending on the market.

The config file lets us set different parameter search ranges for each channel group,
which speeds up optimization and produces more realistic results.

---
## 4. Workshop: Hands-On EDA

Now it is your turn. Use the cells below to explore the workshop data further,
or apply the EDA pipeline to your own data.

**Suggested exercises:**

1. Create a CCF plot for a variable you have not explored yet.
2. Try a correlation heatmap with a different subset of variables.
3. Investigate the trend and seasonality in the sales data.
4. Check if any variables need special treatment (log transform, aggregation, etc.).

```python
# TODO: Explore a new CCF relationship
# ccf_plot(data, 'Sales_Volume_Total', 'Your_Variable_Here')
```

```python
# TODO: Create a custom correlation heatmap
# custom_cols = ['Sales_Volume_Total', 'var1', 'var2', ...]
# correlation_heatmap(data[custom_cols])
```

```python
# TODO: Investigate seasonality -- plot sales by month
# data.groupby('month_num')['Sales_Volume_Total'].mean().plot(kind='bar')
# plt.title('Average Sales by Month')
# plt.ylabel('Sales Volume')
# plt.show()
```

```python
# TODO: Additional custom analysis
# Write your own exploration code here
```

---
## 5. Deliverable

Before next session, please complete the following:

1. **Complete your EDA notebook** (Notebook 1: Offline Exercise) if you have not already.
2. **Write 3-5 bullet points** summarizing data quality and key findings.
   Focus on:
   - Missing data and how you handled it.
   - Zero-inflation patterns across channels.
   - Key lag structures you identified from CCF.
   - Multicollinearity concerns.
   - Seasonality and trend observations.
3. **If using your own data (BYOD):** Ensure your data is loaded and the EDA pipeline runs cleanly.

---
## 6. Preview of Week 2

Next week we will move from understanding the data to **building the model**. Here is what is ahead:

- **Feature Engineering**: Adstock transformations (geometric decay, Weibull) and saturation curves (Hill function).
- **Model Building**: OLS regression with transformed variables.
- **Model Diagnostics**: R-squared, coefficient signs, VIF for multicollinearity, residual analysis.
- **Decomposition**: Breaking down predicted sales into base + incremental contribution by channel.

The EDA insights from this week will directly inform your transformation parameter choices.
For example, the CCF lag peaks will guide your adstock decay ranges.

**Come prepared with:**
- Your completed EDA summary.
- Questions about your data or anything unclear from Week 1.
- Your BYOD dataset loaded and ready (if applicable).

