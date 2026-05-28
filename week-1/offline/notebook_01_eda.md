# Notebook 01 Eda

# Notebook 1: Exploratory Data Analysis for MMM

**Marketing Science Bootcamp -- Week 1 Offline Exercise**

Complete this notebook between Session 1 and Session 2.

## Overview

Exploratory Data Analysis (EDA) is a critical first step before building any Marketing Mix Model. 
Before we touch a regression or any optimization routine, we need to deeply understand our data: 
its distributions, relationships, temporal patterns, and potential pitfalls.

In this notebook you will produce:

- **Cross-Correlation Function (CCF) plots** to identify lag structures between media spend and sales.
- **Correlation heatmaps** to detect multicollinearity and understand variable relationships.
- **Dual-axis charts** to visually compare spend and KPI time series side by side.

By the end of this exercise you should have a clear picture of which variables matter, 
what lag effects exist, and what data quality issues need to be addressed before modeling.

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

%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
```

---
## Load Data

```python
data = pd.read_excel('../../data/MMM_Workshop_Data.xlsx', sheet_name='Data')
print(f'Shape: {data.shape}')
data.head()
```

---
## 1. Summary Statistics

Before any visualization, start with the numbers. Summary statistics reveal the central tendency,
spread, and shape of each variable. Pay attention to:

- **High variance** -- variables with very large standard deviations relative to their mean.
- **Zero-inflation** -- variables where the minimum is zero and the median is close to zero (spend that is intermittent).
- **Skewness** -- heavily right-skewed distributions may benefit from log transforms.
- **Scale differences** -- variables on vastly different scales may need normalization for some analyses.

```python
stats = summary_statistics(data)
stats
```

**Your turn:** Look at the output above and note:

- Which variables have the highest variance?
- Are any spend variables zero-inflated (many weeks with zero spend)?
- Do any variables appear heavily skewed?
- Are there any obvious data quality issues (negative values, implausible ranges)?

---
## 2. Cross-Correlation (CCF) Plots

### What is CCF?

The **Cross-Correlation Function** measures the correlation between two time series at different
lags. In MMM, this is essential because advertising does not always affect sales immediately:

- **TV** often shows effects at lag 0-2 weeks (broad reach, quick awareness).
- **Digital channels** may show effects at lag 0-1 weeks (faster path to conversion).
- **Print or OOH** may have longer lag structures.

A positive spike at lag *k* means that spend at time *t* is correlated with sales at time *t + k*.

Understanding lag structures helps us decide which lag/adstock transformations to apply during modeling.

```python
# Single CCF plot: Sales vs TV_Spends
ccf_plot(data, 'Sales_Volume_Total', 'TV_Spends')
```

```python
# Full CCF grid for all relevant variables
ccf_plot_all(data)
```

**Reflection questions:**

- What lags do you observe for **TV**? Is the peak at lag 0, or does it take a week or two?
- What lags do you observe for **digital channels**? Are they faster-acting than TV?
- Are there any variables where the CCF is essentially flat (no clear relationship)?
- Do any variables show a negative cross-correlation at certain lags?

---
## 3. Correlation Heatmap

A correlation heatmap shows the pairwise **Pearson correlation** between variables. In MMM, this helps us:

- Identify which spend variables are most correlated with the KPI.
- Detect **multicollinearity** -- when two or more independent variables are highly correlated with
  each other (e.g., TV spend and total media spend). Multicollinearity inflates standard errors
  and makes coefficient estimates unstable.
- Spot unexpected relationships that may indicate confounding.

```python
# Select the KPI and main spend variables for the heatmap
key_columns = [
    'Sales_Volume_Total',
    'TV_Spends',
    'Digital_Spends',
    'Sponsorship_Spends',
    'Content_Spends',
    'Online_Spends',
    'Print_Spends'
]

# Filter to columns that actually exist in the data
key_columns = [c for c in key_columns if c in data.columns]

correlation_heatmap(data[key_columns])
```

**Reflection questions:**

- Which variables are most highly correlated with **Sales_Volume_Total**?
- Which pairs of independent variables are highly correlated with **each other**?
  (correlations above 0.7 are a warning sign for multicollinearity)
- What multicollinearity concerns do you see? How might these affect model coefficients?
- Would you consider dropping or combining any variables?

---
## 4. Dual-Axis Charts

Dual-axis time series charts overlay the **KPI** (e.g., Sales) on one y-axis and a **spend variable**
on the other. This visual approach is invaluable in MMM because:

- You can see whether peaks in spend align with peaks in sales.
- You can visually detect lag effects (spend peak precedes sales peak by 1-2 weeks).
- You can identify periods of heavy spend with no apparent lift -- possible saturation.
- You can spot seasonality patterns that affect both variables.

```python
# Single dual-axis chart: Sales vs TV_Spends
dual_axis_chart(data, kpi_col='Sales_Volume_Total', spend_col='TV_Spends')
```

```python
# Full grid of dual-axis charts for key spend variables
spend_vars = [
    'TV_Spends',
    'Digital_Spends',
    'Sponsorship_Spends',
    'Content_Spends',
    'Online_Spends',
    'Print_Spends'
]

# Filter to columns that actually exist
spend_vars = [c for c in spend_vars if c in data.columns]

dual_axis_chart_all(data, kpi_col='Sales_Volume_Total', spend_cols=spend_vars)
```

**Reflection questions:**

- Do you see any visual patterns between spend and sales?
- For which channels does spend appear to move in sync with sales?
- Do any channels show a clear lag (spend peak *before* sales peak)?
- Are there periods of high spend with no visible sales response?

---
## 5. Your Observations

Write **3-5 bullet points** summarizing your key EDA findings. Consider:

- **Seasonality patterns**: Do sales follow a clear seasonal cycle? Which months or weeks are peaks/troughs?
- **Lag effects**: Which channels show the strongest lagged relationship with sales? What are the optimal lags?
- **Correlation concerns**: Are there multicollinearity issues that need to be addressed before modeling?
- **Data quality issues**: Missing values, zero-inflation, outliers, or unexpected patterns?
- **Channel relationships**: Which channels seem most/least related to sales based on the EDA?

**My key EDA findings:**

1. *[Your observation here]*
2. *[Your observation here]*
3. *[Your observation here]*
4. *[Your observation here]*
5. *[Your observation here]*

---
## 6. EDA on Your Own Data (BYOD)

If you have your own marketing data, try running the same EDA pipeline on it. 
Replace the data loading step below with your own file, then re-run the analysis functions.

**Instructions:**

1. Replace the file path in the cell below with your own data file.
2. Update column names to match your dataset.
3. Run the EDA functions and compare findings with the workshop data.
4. Note any differences in lag structures, correlation patterns, or data quality.

```python
# TODO: Load your own data
# my_data = pd.read_csv('path/to/your/data.csv')
# my_data = pd.read_excel('path/to/your/data.xlsx', sheet_name='Sheet1')
# print(f'Shape: {my_data.shape}')
# my_data.head()
```

```python
# TODO: Run summary statistics on your data
# summary_statistics(my_data)
```

```python
# TODO: Run CCF plots on your data
# Update the KPI and spend column names to match your dataset
# ccf_plot(my_data, 'your_kpi_column', 'your_spend_column')
```

```python
# TODO: Run correlation heatmap on your data
# my_key_columns = ['your_kpi', 'spend_1', 'spend_2', ...]
# correlation_heatmap(my_data[my_key_columns])
```

```python
# TODO: Run dual-axis charts on your data
# dual_axis_chart(my_data, kpi_col='your_kpi', spend_col='your_spend')
```

**BYOD observations:**

1. *[Your observation here]*
2. *[Your observation here]*
3. *[Your observation here]*

