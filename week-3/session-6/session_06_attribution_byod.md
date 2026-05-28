# Session 06 Attribution Byod

# Session 6: Attribution Models & BYOD Workshop

```python
# --- Setup ---
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("All imports loaded successfully.")
```

---
## Part 1: Shapley Value Attribution - Live Coding

We will build Shapley attribution step by step.

```python
# --- Step 1: Generate (or load) journey data ---
channels = ['Paid Search', 'Display', 'Social', 'Email']

journeys = []
conversions = []

for _ in range(1000):
    n_touches = np.random.randint(1, 5)
    journey = list(np.random.choice(channels, n_touches))
    converted = np.random.random() < (0.1 + 0.05 * n_touches)
    journeys.append(journey)
    conversions.append(int(converted))

total_conversions = sum(conversions)
print(f"Journeys: {len(journeys)}, Conversions: {total_conversions} ({total_conversions/len(journeys)*100:.1f}%)")
print(f"\nSample journeys:")
for i in range(3):
    print(f"  {' -> '.join(journeys[i])} [{'CONV' if conversions[i] else 'no conv'}]")
```

```python
# --- Step 2: Define coalition value function ---
# v(S) = conversion rate for journeys where all channels in S appear

def coalition_value(subset, journeys, conversions):
    """Conversion rate of journeys containing all channels in subset."""
    subset_set = set(subset)
    if len(subset_set) == 0:
        return 0.0
    
    matching = [(j, c) for j, c in zip(journeys, conversions) 
                if subset_set.issubset(set(j))]
    
    if len(matching) == 0:
        return 0.0
    return np.mean([c for _, c in matching])

# Compute all coalition values
coalition_values = {}
for r in range(len(channels) + 1):
    for subset in itertools.combinations(channels, r):
        coalition_values[subset] = coalition_value(subset, journeys, conversions)

print(f"Computed {len(coalition_values)} coalition values.")
print(f"\nExamples:")
print(f"  v(empty) = {coalition_values[()]:.4f}")
for ch in channels:
    print(f"  v({{{ch}}}) = {coalition_values[(ch,)]:.4f}")
print(f"  v(all 4) = {coalition_values[tuple(sorted(channels))]:.4f}")
```

```python
# --- Step 3: Compute exact Shapley values ---
def shapley_values(channels, coalition_values):
    n = len(channels)
    shapley = {ch: 0.0 for ch in channels}
    
    for channel_i in channels:
        others = [ch for ch in channels if ch != channel_i]
        
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                s = len(subset)
                weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                
                with_i = tuple(sorted(list(subset) + [channel_i]))
                without_i = tuple(sorted(subset))
                
                marginal = coalition_values.get(with_i, 0) - coalition_values.get(without_i, 0)
                shapley[channel_i] += weight * marginal
    
    return shapley

shapley = shapley_values(channels, coalition_values)

print("Shapley Values:")
for ch in channels:
    print(f"  {ch}: {shapley[ch]:.4f}")
print(f"\nTotal: {sum(shapley.values()):.4f} (should equal overall conversion rate: {np.mean(conversions):.4f})")
```

```python
# --- Step 4: Visualize Shapley attribution ---
shapley_total = sum(shapley.values())
shapley_pct = {ch: shapley[ch] / shapley_total * 100 for ch in channels}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax = axes[0]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax.bar(channels, [shapley[ch] for ch in channels], color=colors)
ax.set_title('Shapley Value by Channel')
ax.set_ylabel('Shapley Value')
for bar, ch in zip(bars, channels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{shapley[ch]:.3f}', ha='center', va='bottom', fontsize=10)

# Pie chart
ax = axes[1]
ax.pie([shapley_pct[ch] for ch in channels], labels=channels, colors=colors,
       autopct='%1.1f%%', startangle=90)
ax.set_title('Shapley Attribution Share')

plt.suptitle('Shapley Value Attribution', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

---
## Part 2: Markov Chain Attribution - Live Coding

```python
# --- Step 1: Build transition matrix ---
states = ['Start'] + channels + ['Conversion', 'Null']
n_states = len(states)
state_idx = {s: i for i, s in enumerate(states)}

transition_counts = np.zeros((n_states, n_states))

for journey, converted in zip(journeys, conversions):
    # Start -> first touch
    transition_counts[state_idx['Start'], state_idx[journey[0]]] += 1
    # Touch -> touch
    for i in range(len(journey) - 1):
        transition_counts[state_idx[journey[i]], state_idx[journey[i+1]]] += 1
    # Last touch -> outcome
    if converted:
        transition_counts[state_idx[journey[-1]], state_idx['Conversion']] += 1
    else:
        transition_counts[state_idx[journey[-1]], state_idx['Null']] += 1

# Normalize
row_sums = transition_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
transition_matrix = transition_counts / row_sums

print("Transition Matrix:")
print(pd.DataFrame(transition_matrix, index=states, columns=states).round(3))
```

```python
# --- Step 2: Visualize as directed graph ---
G = nx.DiGraph()

for i, from_state in enumerate(states):
    for j, to_state in enumerate(states):
        if transition_matrix[i, j] > 0.01:
            G.add_edge(from_state, to_state, weight=transition_matrix[i, j])

fig, ax = plt.subplots(figsize=(14, 8))

pos = nx.spring_layout(G, seed=42, k=2)

color_map = []
for node in G.nodes():
    if node == 'Start':
        color_map.append('lightgreen')
    elif node == 'Conversion':
        color_map.append('gold')
    elif node == 'Null':
        color_map.append('lightcoral')
    else:
        color_map.append('lightblue')

nx.draw(G, pos, ax=ax, with_labels=True, node_color=color_map,
        node_size=2000, font_size=10, font_weight='bold',
        edge_color='gray', arrows=True, arrowsize=20,
        connectionstyle='arc3,rad=0.1')

edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

ax.set_title('Customer Journey Markov Chain', fontsize=14)
plt.tight_layout()
plt.show()
```

```python
# --- Step 3: Removal effect computation ---
def conversion_probability(tm, states, removed_channel=None):
    """Compute P(Conversion | Start) using absorbing Markov chain theory."""
    tm = tm.copy()
    si = {s: i for i, s in enumerate(states)}
    
    if removed_channel is not None:
        ch_idx = si[removed_channel]
        null_idx = si['Null']
        # Redirect transitions to removed channel -> Null
        for i in range(len(states)):
            tm[i, null_idx] += tm[i, ch_idx]
            tm[i, ch_idx] = 0
        tm[ch_idx, :] = 0
        tm[ch_idx, null_idx] = 1.0
    
    absorbing = [si['Conversion'], si['Null']]
    transient = [i for i in range(len(states)) if i not in absorbing]
    
    Q = tm[np.ix_(transient, transient)]
    R = tm[np.ix_(transient, absorbing)]
    N = np.linalg.inv(np.eye(len(transient)) - Q)
    B = N @ R
    
    start_idx = transient.index(si['Start'])
    return B[start_idx, 0]  # Conversion is first absorbing state

# Base probability
base_prob = conversion_probability(transition_matrix, states)
print(f"Base conversion probability: {base_prob:.4f}")

# Removal effects
removal_effects = {}
for channel in channels:
    prob_without = conversion_probability(transition_matrix, states, removed_channel=channel)
    removal_effects[channel] = base_prob - prob_without
    print(f"  Remove {channel}: P(conv) = {prob_without:.4f}, effect = {removal_effects[channel]:.4f}")

# Normalize
total_effect = sum(removal_effects.values())
markov_attribution = {ch: removal_effects[ch] / total_effect for ch in channels}

print(f"\nMarkov Attribution:")
for ch in channels:
    print(f"  {ch}: {markov_attribution[ch]*100:.1f}%")
```

```python
# --- Step 4: Compare Shapley vs Markov ---
shapley_pct_norm = {ch: shapley[ch] / sum(shapley.values()) for ch in channels}

compare_df = pd.DataFrame({
    'Shapley': [shapley_pct_norm[ch] * 100 for ch in channels],
    'Markov': [markov_attribution[ch] * 100 for ch in channels],
}, index=channels)

fig, ax = plt.subplots(figsize=(10, 5))
compare_df.plot(kind='bar', ax=ax, width=0.6, color=['#3498db', '#2ecc71'])
ax.set_title('Shapley vs. Markov Chain Attribution', fontsize=14)
ax.set_ylabel('Attribution Share (%)')
ax.set_xticklabels(channels, rotation=0)
ax.legend(title='Method', fontsize=11)
plt.tight_layout()
plt.show()

print(compare_df.round(1))
print(f"\nCorrelation: {compare_df['Shapley'].corr(compare_df['Markov']):.3f}")
```

---
## Part 3: Meridian Budget Optimization

### Marginal ROI vs. Average ROI

A key insight from MMM is the difference between:

- **Average ROI** = total incremental return / total spend (what you got overall)
- **Marginal ROI (mROI)** = incremental return from the **next dollar** (what you would get from spending more)

Due to diminishing returns (Hill saturation), **mROI decreases as spend increases**. The optimal budget allocation equalizes mROI across all channels - if one channel has higher mROI, shift budget there until mROI equalizes.

Meridian provides built-in budget optimization that exploits this principle.

```python
# --- Budget Optimization with Meridian (Conceptual) ---
# NOTE: This requires a fitted Meridian model from Session 5.
# Below is the conceptual workflow.

# Budget optimization with Meridian
# optimizer = meridian.optimize_budget(model, ...)
# Plot response curves with current spend marked
# Scenario analysis: what if budget +20%?

# --- Simulated Response Curves for Illustration ---
# Using Hill function to demonstrate the concept

def hill_function(x, ec, slope):
    """Hill saturation curve."""
    return x ** slope / (ec ** slope + x ** slope)

# Simulated channel parameters (from a hypothetical Meridian fit)
channel_params = {
    'Paid Search':  {'ec': 50000, 'slope': 1.5, 'beta': 3.0, 'current_spend': 40000},
    'Display':      {'ec': 30000, 'slope': 1.2, 'beta': 2.0, 'current_spend': 25000},
    'Social':       {'ec': 20000, 'slope': 1.8, 'beta': 2.5, 'current_spend': 15000},
    'Email':        {'ec': 10000, 'slope': 2.0, 'beta': 4.0, 'current_spend': 5000},
}

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for ax, (ch, params), color in zip(axes, channel_params.items(), colors):
    spend = np.linspace(0, params['current_spend'] * 2, 200)
    response = params['beta'] * hill_function(spend, params['ec'], params['slope'])
    
    ax.plot(spend, response, color=color, linewidth=2)
    
    # Mark current spend
    current_response = params['beta'] * hill_function(params['current_spend'], params['ec'], params['slope'])
    ax.axvline(params['current_spend'], color='black', linestyle='--', alpha=0.5)
    ax.plot(params['current_spend'], current_response, 'ko', markersize=8)
    ax.annotate('Current\nspend', xy=(params['current_spend'], current_response),
                xytext=(params['current_spend']*1.2, current_response*0.7),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)
    
    ax.set_title(ch, fontsize=11)
    ax.set_xlabel('Spend ($)')
    ax.set_ylabel('Incremental KPI')

plt.suptitle('Response Curves with Current Spend Position', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

print("Channels on the steep part of the curve have high mROI (room to grow).")
print("Channels on the flat part have low mROI (saturated).")
```

```python
# --- Scenario: Reallocate 20% from Lowest ROI to Highest ROI Channel ---

# Compute current ROI and mROI
results = []
for ch, params in channel_params.items():
    spend = params['current_spend']
    response = params['beta'] * hill_function(spend, params['ec'], params['slope'])
    roi = response / spend * 1000  # per $1000
    
    # mROI: derivative of response at current spend
    delta = 100
    response_plus = params['beta'] * hill_function(spend + delta, params['ec'], params['slope'])
    mroi = (response_plus - response) / delta * 1000  # per $1000
    
    results.append({'Channel': ch, 'Spend': spend, 'Response': response,
                    'ROI (per $1K)': roi, 'mROI (per $1K)': mroi})

results_df = pd.DataFrame(results).set_index('Channel')
print("Current Performance:")
print(results_df.round(3))

# Identify lowest and highest mROI channels
lowest_mroi = results_df['mROI (per $1K)'].idxmin()
highest_mroi = results_df['mROI (per $1K)'].idxmax()
print(f"\nLowest mROI: {lowest_mroi}")
print(f"Highest mROI: {highest_mroi}")

# Scenario: move 20% of lowest mROI budget to highest mROI
realloc_amount = results_df.loc[lowest_mroi, 'Spend'] * 0.20
print(f"\nScenario: Reallocate ${realloc_amount:,.0f} from {lowest_mroi} to {highest_mroi}")

# Compute new responses
new_spend = results_df['Spend'].copy()
new_spend[lowest_mroi] -= realloc_amount
new_spend[highest_mroi] += realloc_amount

old_total = 0
new_total = 0
for ch in channel_params:
    p = channel_params[ch]
    old_total += p['beta'] * hill_function(results_df.loc[ch, 'Spend'], p['ec'], p['slope'])
    new_total += p['beta'] * hill_function(new_spend[ch], p['ec'], p['slope'])

print(f"\nTotal response (current): {old_total:.3f}")
print(f"Total response (optimized): {new_total:.3f}")
print(f"Improvement: {(new_total/old_total - 1)*100:.1f}%")
print(f"Same total budget, better allocation.")
```

---
## Part 4: BYOD (Bring Your Own Data) Workshop

### Instructions

It is time to apply what you have learned to **your own data**. This workshop session gives you guided time to:

1. **Format your data** for MMM (time series with media spend and KPI)
2. **Run EDA** (distributions, correlations, time trends)
3. **Fit an initial model** (OLS or Bayesian from Week 2, or Meridian from Session 5)
4. **Review results** with the instructor

### Checklist

Before starting, verify:

- [ ] **Data formatted**: Weekly or daily time series with date column
- [ ] **KPI identified**: Revenue, conversions, or other target variable
- [ ] **Media columns identified**: Spend (and optionally impressions) per channel
- [ ] **Controls identified**: Seasonality, promotions, macro variables
- [ ] **EDA completed**: No obvious data quality issues
- [ ] **Config created**: Column mappings for your model

### Tips

- Start simple: fit an OLS model first to establish a baseline
- Check for multicollinearity between media channels
- Consider log-transforming the KPI if it is skewed
- Include at least basic seasonality controls (month dummies or Fourier terms)

```python
# TODO: Load your own data
# byod_df = pd.read_csv('path/to/your/data.csv')  # or .xlsx
# print(byod_df.shape)
# byod_df.head()
```

```python
# TODO: Define your column mappings
# byod_date_col = 'date'
# byod_kpi_col = 'revenue'
# byod_media_cols = ['channel_1_spend', 'channel_2_spend', ...]
# byod_control_cols = ['seasonality', 'promo', ...]
```

```python
# TODO: Run EDA on your data
# - Time series plots for KPI and media spend
# - Correlation matrix
# - Distribution checks
# - Missing value audit
```

```python
# TODO: Fit an initial model
# Option A: OLS from scratch (Week 2 approach)
# Option B: Bayesian with PyMC (Week 2 approach)
# Option C: Google Meridian (Session 5 approach)
```

```python
# TODO: Review results
# - Channel ROI table
# - Response curves
# - Model fit diagnostics (R-squared, residuals, convergence)
```

---
## Deliverable

By the end of this session, you should have:

1. **Attribution notebook** with Shapley and Markov implementations on synthetic data
2. **BYOD initial model fit** - at minimum, an OLS baseline on your own data
3. **ROI comparison** between workshop data results and your BYOD results

Submit your completed notebook and a brief summary of your BYOD findings.

