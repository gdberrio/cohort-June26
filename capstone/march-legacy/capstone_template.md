# Capstone Template

# Capstone Project: Marketing Mix Modeling

**Student Name:** [Your Name]  
**Date:** [Date]  
**Dataset:** [Workshop Data / BYOD - describe briefly]

---

## Project Overview

This notebook is your capstone deliverable for the Marketing Science Bootcamp. It should demonstrate your ability to:

1. Prepare and explore marketing data
2. Build and validate a Marketing Mix Model
3. Extract actionable business insights
4. Recommend a budget reallocation strategy
5. Propose a measurement plan with experiments

### Evaluation Criteria

| Criteria | Weight | Description |
|---|---|---|
| Technical Correctness | 30% | Model properly specified, code runs, diagnostics appropriate |
| Diagnostics & Validation | 20% | VIF, DW, MAPE, coefficient signs, out-of-sample checks |
| Business Relevance | 20% | Insights are actionable, recommendations make business sense |
| Communication Quality | 15% | Notebook is readable, charts are clear, narrative flows logically |
| Measurement Plan | 15% | Experiments are well-designed, calibration approach is sound |

---

## 1. Business Context

*Describe the business problem. What decisions will this model inform? What KPI are you modeling?*

[Write your business context here]

---

## 2. Data Overview & EDA

```python
# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

import sys
sys.path.insert(0, '../..')
from utils.mmm_utils import *
from utils.eda_utils import *

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
```

```python
# TODO: Load your data
data = load_workshop_data('../../data/MMM_Workshop_Data.xlsx')
print(f"Data shape: {data.shape}")
print(f"Time range: {data['Month'].min()} to {data['Month'].max()}")
data.head()
```

```python
# TODO: Summary statistics
summary_statistics(data)
```

```python
# TODO: KPI over time
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data['Month'], data['Sales_Volume_Total'], color='navy', linewidth=2)
ax.set_title('KPI Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Sales Volume')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

```python
# TODO: CCF plots for key channels
```

```python
# TODO: Correlation heatmap
```

### EDA Key Findings

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

---

## 3. Data Preparation

```python
# TODO: Handle missing values, create control variables (trend, seasonality dummies),
# apply transformations, etc.
```

---

## 4. Model Building

```python
# TODO: Apply adstock and saturation transformations
# Use grid search or Bayesian approach
```

```python
# TODO: Build your model (OLS, Bayesian, or Meridian)
```

```python
# TODO: Print model summary
```

---

## 5. Model Diagnostics

```python
# TODO: Compute and display diagnostics
# VIF, Durbin-Watson, MAPE, NRMSE, AIC, coefficient signs, DECOMP.RSSD
```

```python
# TODO: Actual vs Predicted plot
```

### Diagnostics Assessment

| Metric | Value | Assessment |
|---|---|---|
| R-squared | | |
| Adj R-squared | | |
| MAPE | | |
| NRMSE | | |
| Durbin-Watson | | |
| Max VIF | | |
| DECOMP.RSSD | | |
| All coefficients positive? | | |

---

## 6. Results & Contribution Analysis

```python
# TODO: Compute and plot contributions
```

```python
# TODO: ROI by channel (if using Bayesian/Meridian, include credible intervals)
```

### Key Results

1. [Top contributing channel and its share]
2. [Channel with highest ROI]
3. [Any surprising findings]

---

## 7. Budget Optimization & Recommendations

```python
# TODO: Response curves or marginal ROI analysis
```

```python
# TODO: Budget reallocation scenario
# What if we shift X% from lowest ROI channel to highest ROI channel?
```

### Budget Recommendations

| Channel | Current Spend | Recommended Spend | Change | Expected Impact |
|---|---|---|---|---|
| | | | | |

**Expected overall impact:** [X% increase in KPI]

---

## 8. Validation & Limitations

### Validation Steps Taken

1. [In-sample fit metrics]
2. [Coefficient sign checks]
3. [Business sense validation]
4. [Out-of-sample validation if applicable]

### Limitations

1. [Limitation 1 - e.g., data granularity, missing channels]
2. [Limitation 2 - e.g., no experimental calibration]
3. [Limitation 3]

---

## 9. Measurement Plan (Next 6 Months)

### Experiment 1
- **Channel:** [e.g., TV]
- **Hypothesis:** [e.g., TV drives X% incremental sales]
- **Design:** [Geo experiment / A-B test]
- **Markets:** [Treatment: ..., Control: ...]
- **Duration:** [X weeks]
- **MDE:** [X%]
- **How results feed back into MMM:** [Use as informative prior for TV coefficient]

### Experiment 2
- **Channel:** [e.g., Social Media]
- **Hypothesis:** [...]
- **Design:** [...]
- **Duration:** [...]
- **How results feed back into MMM:** [...]

### Ongoing Measurement Cadence
- **MMM refresh:** [Quarterly]
- **Attribution monitoring:** [Ongoing, weekly reports]
- **Experiments:** [2-3 per year, staggered]

---

## 10. Next Steps

1. [Next step 1]
2. [Next step 2]
3. [Next step 3]

