# Notebook 00 Explore Dataset

# Notebook 0: Explore the Workshop Dataset

**Marketing Science Bootcamp -- Pre-Work**

---

## Purpose

This notebook is designed as **pre-work** to help you familiarize yourself with the dataset we will use throughout the bootcamp. Before Session 1, please work through every cell and read the instructions carefully.

By the end of this notebook you should:

1. Understand the shape, structure, and contents of the workshop dataset.
2. Be able to identify the key performance indicator (KPI) and the media spend columns.
3. Have written down at least **three observations** about the data to share with the instructor.

---

## 1. Setup & Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting defaults
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["axes.grid"] = True
sns.set_style("whitegrid")

print("Imports complete.")
```

## 2. Load the Data

The dataset is stored in an Excel file. We load the **Data** sheet which contains the main marketing mix data.

```python
data = pd.read_excel("../data/MMM_Workshop_Data.xlsx", sheet_name="Data")
print(f"Data loaded successfully: {data.shape[0]} rows x {data.shape[1]} columns")
```

## 3. First Look at the Data

```python
# First 5 rows
data.head()
```

```python
# Shape and data types
print(f"Shape: {data.shape}")
print()
print("Data types:")
print(data.dtypes)
```

## 4. Summary Statistics

```python
# Transposed describe for easier reading
data.describe().T
```

## 5. Missing Values

```python
missing = data.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
print(f"\nTotal missing cells: {missing.sum()} out of {data.size}")
```

## 6. Zero-Inflated Columns

Some media channels may only be active in certain periods, leading to many zero values. Let's check which columns have a high proportion of zeros.

```python
numeric_cols = data.select_dtypes(include=[np.number]).columns
zero_pct = (data[numeric_cols] == 0).sum() / len(data) * 100

zero_report = zero_pct[zero_pct > 0].sort_values(ascending=False)
if len(zero_report) > 0:
    print("Columns with zero values (% of rows):")
    print(zero_report.round(1).to_string())
else:
    print("No zero-inflated columns detected.")
```

## 7. KPI Over Time

Let's plot the main KPI over time. We will look for `Sales_Volume_Total` or `Sales_Revenue_Total` as the KPI column, and `Month` (or the first date-like column) as the time axis.

```python
# Identify the KPI column
kpi_candidates = [c for c in data.columns if "sales" in c.lower() or "revenue" in c.lower() or "volume" in c.lower()]
print("KPI candidate columns:", kpi_candidates)

# Identify the time column
time_candidates = [c for c in data.columns if "month" in c.lower() or "date" in c.lower() or "week" in c.lower()]
print("Time candidate columns:", time_candidates)

# Select the first match (adjust if needed)
kpi_col = kpi_candidates[0] if kpi_candidates else data.select_dtypes(include=[np.number]).columns[0]
time_col = time_candidates[0] if time_candidates else data.columns[0]

print(f"\nUsing KPI  : {kpi_col}")
print(f"Using Time : {time_col}")
```

```python
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(data[time_col], data[kpi_col], marker="o", linewidth=2, color="#2c3e50")
ax.set_title(f"{kpi_col} Over Time", fontsize=16, fontweight="bold")
ax.set_xlabel(time_col, fontsize=12)
ax.set_ylabel(kpi_col, fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 8. Top 5 Media Spends Over Time

Let's identify the media spend columns and plot the top 5 by total spend.

```python
# Identify media spend columns (heuristic: columns containing 'spend' or 'media' or 'cost')
spend_candidates = [c for c in data.columns if any(kw in c.lower() for kw in ["spend", "media", "cost", "investment"])]

# If no obvious spend columns found, fall back to numeric columns excluding KPI and time
if not spend_candidates:
    spend_candidates = [c for c in numeric_cols if c != kpi_col and c != time_col]

print(f"Found {len(spend_candidates)} potential media spend columns:")
print(spend_candidates)

# Top 5 by total value
top5 = data[spend_candidates].sum().nlargest(5).index.tolist()
print(f"\nTop 5 by total spend: {top5}")
```

```python
fig, ax = plt.subplots(figsize=(14, 6))
for col in top5:
    ax.plot(data[time_col], data[col], marker=".", linewidth=1.5, label=col)

ax.set_title("Top 5 Media Spends Over Time", fontsize=16, fontweight="bold")
ax.set_xlabel(time_col, fontsize=12)
ax.set_ylabel("Spend", fontsize=12)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 9. Correlation with the KPI

Let's compute the Pearson correlation of every numeric column with the KPI and display the results sorted by absolute correlation.

```python
correlations = data[numeric_cols].corr()[kpi_col].drop(kpi_col).sort_values(key=abs, ascending=False)

print(f"Pearson correlation of numeric columns with {kpi_col}:\n")
print(correlations.round(3).to_string())
```

```python
fig, ax = plt.subplots(figsize=(10, max(4, len(correlations) * 0.35)))
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in correlations.values]
correlations.plot.barh(ax=ax, color=colors)
ax.set_title(f"Correlation with {kpi_col}", fontsize=14, fontweight="bold")
ax.set_xlabel("Pearson r")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
```

---

## Your Observations

Based on your exploration above, please write down **at least 3 observations** about the dataset. Think about questions such as:

- What trends or seasonality do you see in the KPI?
- Which media channels appear most correlated with the KPI?
- Are there any columns with lots of zeros or missing values? What might that mean?
- Do any patterns surprise you?

### Observation 1

*Write your first observation here...*

### Observation 2

*Write your second observation here...*

### Observation 3

*Write your third observation here...*

---

## Submit Your Observations

**Please submit your observations to the instructor before Session 1.**

You can share them via email or the course platform. Include:
1. Your name
2. Your 3 (or more) observations
3. Any questions that came up while exploring the data

We will discuss everyone's findings at the start of Session 1. See you there!

