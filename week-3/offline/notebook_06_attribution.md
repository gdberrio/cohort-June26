# Notebook 05 Attribution

# Notebook 6: Attribution Models from Scratch

## Overview

**Attribution** assigns credit to touchpoints in customer journeys. While MMM measures aggregate channel effects using time-series data, attribution works at the **individual journey level**.

In this notebook, we implement three families of attribution:

1. **Rule-based** (last-click, first-click, linear, time-decay, position-based)
2. **Shapley value** (game-theoretic, fair allocation)
3. **Markov chain** (data-driven, probabilistic)

We will build each from scratch to deeply understand the mechanics, then compare their results.

```python
# --- Setup ---
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
```

```python
# --- Create Synthetic Journey Data ---
np.random.seed(42)
channels = ['Paid Search', 'Display', 'Social', 'Email']

journeys = []
conversions = []

for _ in range(1000):
    n_touches = np.random.randint(1, 5)
    journey = list(np.random.choice(channels, n_touches))
    converted = np.random.random() < (0.1 + 0.05 * n_touches)
    journeys.append(journey)
    conversions.append(int(converted))

print(f"Total journeys: {len(journeys)}")
print(f"Conversions: {sum(conversions)} ({sum(conversions)/len(conversions)*100:.1f}%)")
print(f"\nExample journeys:")
for i in range(5):
    status = 'CONVERTED' if conversions[i] else 'no conversion'
    print(f"  {' -> '.join(journeys[i])} [{status}]")
```

## Part A: Rule-Based Attribution

Rule-based models apply fixed rules to allocate conversion credit. They are simple and fast but rely on assumptions rather than data.

| Method | Rule |
|--------|------|
| **Last-click** | 100% credit to the final touchpoint |
| **First-click** | 100% credit to the first touchpoint |
| **Linear** | Equal credit to all touchpoints |
| **Time-decay** | More credit to recent touchpoints |
| **Position-based (U-shaped)** | 40% to first, 40% to last, 20% split among middle |

```python
# --- Last-Click Attribution ---
last_click = defaultdict(float)

for journey, converted in zip(journeys, conversions):
    if converted:
        last_click[journey[-1]] += 1.0

print("Last-Click Attribution:")
for ch in channels:
    print(f"  {ch}: {last_click[ch]:.0f} conversions")
```

```python
# --- First-Click Attribution ---
first_click = defaultdict(float)

for journey, converted in zip(journeys, conversions):
    if converted:
        first_click[journey[0]] += 1.0

print("First-Click Attribution:")
for ch in channels:
    print(f"  {ch}: {first_click[ch]:.0f} conversions")
```

```python
# --- Linear Attribution ---
linear = defaultdict(float)

for journey, converted in zip(journeys, conversions):
    if converted:
        credit = 1.0 / len(journey)
        for ch in journey:
            linear[ch] += credit

print("Linear Attribution:")
for ch in channels:
    print(f"  {ch}: {linear[ch]:.1f} conversions")
```

```python
# --- Time-Decay Attribution ---
time_decay = defaultdict(float)
decay_rate = 0.5  # each step back in time halves the credit

for journey, converted in zip(journeys, conversions):
    if converted:
        n = len(journey)
        # Weights: most recent = highest
        weights = [decay_rate ** (n - 1 - i) for i in range(n)]
        total_weight = sum(weights)
        for i, ch in enumerate(journey):
            time_decay[ch] += weights[i] / total_weight

print("Time-Decay Attribution:")
for ch in channels:
    print(f"  {ch}: {time_decay[ch]:.1f} conversions")
```

```python
# --- Position-Based (U-Shaped) Attribution ---
position_based = defaultdict(float)

for journey, converted in zip(journeys, conversions):
    if converted:
        n = len(journey)
        if n == 1:
            position_based[journey[0]] += 1.0
        elif n == 2:
            position_based[journey[0]] += 0.5
            position_based[journey[-1]] += 0.5
        else:
            # 40% first, 40% last, 20% split among middle
            position_based[journey[0]] += 0.4
            position_based[journey[-1]] += 0.4
            middle_credit = 0.2 / (n - 2)
            for ch in journey[1:-1]:
                position_based[ch] += middle_credit

print("Position-Based (U-Shaped) Attribution:")
for ch in channels:
    print(f"  {ch}: {position_based[ch]:.1f} conversions")
```

```python
# --- Compare All 5 Rule-Based Methods ---
methods = {
    'Last-Click': last_click,
    'First-Click': first_click,
    'Linear': linear,
    'Time-Decay': time_decay,
    'Position-Based': position_based,
}

comparison_df = pd.DataFrame({name: {ch: vals[ch] for ch in channels} 
                               for name, vals in methods.items()})
print(comparison_df.round(1))

# Bar chart
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df.plot(kind='bar', ax=ax, width=0.75)
ax.set_title('Rule-Based Attribution: Method Comparison')
ax.set_xlabel('Channel')
ax.set_ylabel('Attributed Conversions')
ax.legend(title='Method')
ax.set_xticklabels(channels, rotation=0)
plt.tight_layout()
plt.show()
```

## Part B: Shapley Value Attribution

### The Game Theory Foundation

Shapley values come from **cooperative game theory** (Lloyd Shapley, 1953, Nobel Prize 2012). The idea:

- Think of channels as **players** in a cooperative game
- The **coalition value** v(S) is the conversion rate when subset S of channels is present in journeys
- Shapley value = each channel's **fair marginal contribution** averaged over all possible orderings

### Formula

For channel i with n total channels:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} \left[ v(S \cup \{i\}) - v(S) \right]$$

This says: for every possible subset S that does NOT include channel i, compute the marginal value of adding i, weighted by the probability of that subset forming.

### Properties (uniquely fair)

Shapley values are the **only** allocation method satisfying:
1. **Efficiency**: total attribution = total conversions
2. **Symmetry**: identical channels get equal credit
3. **Null player**: channels with zero marginal contribution get zero credit
4. **Additivity**: can decompose across sub-games

```python
# --- Compute Coalition Values ---
# v(S) = conversion rate of journeys where ALL channels in S appear

def coalition_value(subset, journeys, conversions):
    """Compute conversion rate for journeys containing all channels in subset."""
    subset_set = set(subset)
    if len(subset_set) == 0:
        return 0.0
    
    matching_journeys = []
    matching_conversions = []
    
    for journey, converted in zip(journeys, conversions):
        journey_channels = set(journey)
        if subset_set.issubset(journey_channels):
            matching_journeys.append(journey)
            matching_conversions.append(converted)
    
    if len(matching_conversions) == 0:
        return 0.0
    
    return np.mean(matching_conversions)

# Compute all 2^n coalition values
n_channels = len(channels)
print(f"Computing {2**n_channels} coalition values for {n_channels} channels...\n")

coalition_values = {}
for r in range(n_channels + 1):
    for subset in itertools.combinations(channels, r):
        v = coalition_value(subset, journeys, conversions)
        coalition_values[subset] = v
        if len(subset) <= 2:  # print small coalitions
            print(f"  v({set(subset) if subset else '{}'}) = {v:.4f}")

print(f"\n  ... ({2**n_channels} total coalitions computed)")
```

```python
# --- Exact Shapley Value Computation ---
import math

def shapley_values(channels, coalition_values):
    """Compute exact Shapley values for each channel."""
    n = len(channels)
    shapley = {ch: 0.0 for ch in channels}
    
    for i, channel_i in enumerate(channels):
        other_channels = [ch for ch in channels if ch != channel_i]
        
        # Iterate over all subsets S of channels \ {i}
        for r in range(len(other_channels) + 1):
            for subset in itertools.combinations(other_channels, r):
                s = len(subset)
                # Weight: |S|! * (n - |S| - 1)! / n!
                weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                
                # Marginal contribution: v(S u {i}) - v(S)
                with_i = tuple(sorted(list(subset) + [channel_i]))
                without_i = tuple(sorted(subset))
                
                marginal = coalition_values.get(with_i, 0) - coalition_values.get(without_i, 0)
                shapley[channel_i] += weight * marginal
    
    return shapley

shapley_result = shapley_values(channels, coalition_values)

print("Shapley Value Attribution:")
for ch in channels:
    print(f"  {ch}: {shapley_result[ch]:.4f}")

print(f"\nSum of Shapley values: {sum(shapley_result.values()):.4f}")
print(f"Overall conversion rate: {np.mean(conversions):.4f}")
```

```python
# --- Shapley vs Last-Click Comparison ---
# Normalize to total conversions for comparison
total_conversions = sum(conversions)
shapley_normalized = {ch: shapley_result[ch] / sum(shapley_result.values()) * total_conversions 
                       for ch in channels}

compare_df = pd.DataFrame({
    'Last-Click': [last_click[ch] for ch in channels],
    'Shapley': [shapley_normalized[ch] for ch in channels],
}, index=channels)

fig, ax = plt.subplots(figsize=(10, 5))
compare_df.plot(kind='bar', ax=ax, width=0.6)
ax.set_title('Last-Click vs. Shapley Value Attribution')
ax.set_ylabel('Attributed Conversions')
ax.set_xticklabels(channels, rotation=0)
plt.tight_layout()
plt.show()
```

### Scaling Challenge: Monte Carlo Approximation

For 10+ channels, we need $2^{10} = 1024$ coalitions. For 20 channels, that is over 1 million. The exact computation becomes infeasible.

**Monte Carlo Shapley** approximates by sampling random permutations of channels and computing marginal contributions along each permutation.

```python
# --- Monte Carlo Shapley Approximation ---
def monte_carlo_shapley(channels, coalition_values, n_permutations=5000):
    """Approximate Shapley values by sampling random permutations."""
    n = len(channels)
    shapley_mc = {ch: 0.0 for ch in channels}
    
    for _ in range(n_permutations):
        # Random permutation of channels
        perm = list(np.random.permutation(channels))
        
        # Walk through permutation, computing marginal contributions
        current_coalition = []
        prev_value = 0.0
        
        for ch in perm:
            current_coalition.append(ch)
            key = tuple(sorted(current_coalition))
            current_value = coalition_values.get(key, 0)
            
            # Marginal contribution of ch
            shapley_mc[ch] += (current_value - prev_value) / n_permutations
            prev_value = current_value
    
    return shapley_mc

shapley_mc = monte_carlo_shapley(channels, coalition_values, n_permutations=10000)

print("Monte Carlo Shapley (10,000 permutations):")
for ch in channels:
    exact = shapley_result[ch]
    approx = shapley_mc[ch]
    print(f"  {ch}: Exact={exact:.4f}, MC={approx:.4f}, Error={abs(exact-approx):.4f}")

print(f"\nMonte Carlo approximation is very close to exact values.")
print(f"This method scales to 50+ channels easily.")
```

## Part C: Markov Chain Attribution

Markov chain attribution models customer journeys as a **stochastic process**:

1. **Build a transition matrix** from observed journey data
2. Each channel is a **state**; add "Start", "Conversion", and "Null" (drop-off) states
3. **Removal effect**: remove one channel at a time, re-compute conversion probability
4. Attribution = normalized removal effects

This is **data-driven** (no arbitrary rules) and captures the **sequential structure** of journeys.

```python
# --- Build Transition Matrix ---
# States: Start, channels, Conversion, Null
states = ['Start'] + channels + ['Conversion', 'Null']
n_states = len(states)
state_idx = {s: i for i, s in enumerate(states)}

# Count transitions
transition_counts = np.zeros((n_states, n_states))

for journey, converted in zip(journeys, conversions):
    # Start -> first touch
    transition_counts[state_idx['Start'], state_idx[journey[0]]] += 1
    
    # Touch -> Touch transitions
    for i in range(len(journey) - 1):
        transition_counts[state_idx[journey[i]], state_idx[journey[i+1]]] += 1
    
    # Last touch -> Conversion or Null
    if converted:
        transition_counts[state_idx[journey[-1]], state_idx['Conversion']] += 1
    else:
        transition_counts[state_idx[journey[-1]], state_idx['Null']] += 1

# Normalize to probabilities
row_sums = transition_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # avoid division by zero
transition_matrix = transition_counts / row_sums

# Display
tm_df = pd.DataFrame(transition_matrix, index=states, columns=states).round(3)
print("Transition Matrix:")
print(tm_df)
```

```python
# --- Visualize as Directed Graph ---
G = nx.DiGraph()

for i, from_state in enumerate(states):
    for j, to_state in enumerate(states):
        if transition_matrix[i, j] > 0.01:  # threshold for visibility
            G.add_edge(from_state, to_state, weight=transition_matrix[i, j])

fig, ax = plt.subplots(figsize=(14, 8))

# Layout
pos = nx.spring_layout(G, seed=42, k=2)

# Draw nodes with colors
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

# Edge labels
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

ax.set_title('Customer Journey Markov Chain', fontsize=14)
plt.tight_layout()
plt.show()
```

```python
# --- Compute Conversion Probability (absorbing Markov chain) ---
def conversion_probability(transition_matrix, states, removed_channel=None):
    """
    Compute probability of reaching 'Conversion' from 'Start'
    using absorbing Markov chain theory.
    
    If removed_channel is specified, redirect all transitions
    to/from that channel to 'Null'.
    """
    tm = transition_matrix.copy()
    state_idx = {s: i for i, s in enumerate(states)}
    
    if removed_channel is not None:
        ch_idx = state_idx[removed_channel]
        null_idx = state_idx['Null']
        
        # Redirect all transitions TO removed channel -> Null
        for i in range(len(states)):
            tm[i, null_idx] += tm[i, ch_idx]
            tm[i, ch_idx] = 0
        
        # Clear removed channel's outgoing transitions
        tm[ch_idx, :] = 0
        tm[ch_idx, null_idx] = 1.0
    
    # Identify transient states (not Conversion or Null)
    absorbing = [state_idx['Conversion'], state_idx['Null']]
    transient = [i for i in range(len(states)) if i not in absorbing]
    
    # Q = transition matrix among transient states
    Q = tm[np.ix_(transient, transient)]
    # R = transition matrix from transient to absorbing states
    R = tm[np.ix_(transient, absorbing)]
    
    # Fundamental matrix: N = (I - Q)^(-1)
    N = np.linalg.inv(np.eye(len(transient)) - Q)
    
    # Absorption probabilities: B = N * R
    B = N @ R
    
    # Find Start state in transient indices
    start_transient_idx = transient.index(state_idx['Start'])
    
    # Conversion is first absorbing state (index 0 in B columns)
    conv_prob = B[start_transient_idx, 0]
    
    return conv_prob

# Base conversion probability
base_prob = conversion_probability(transition_matrix, states)
print(f"Base conversion probability: {base_prob:.4f}")
print(f"Empirical conversion rate: {np.mean(conversions):.4f}")
```

```python
# --- Removal Effect Attribution ---
removal_effects = {}

print("Removal Effect Analysis:")
print(f"  Base conversion probability: {base_prob:.4f}\n")

for channel in channels:
    prob_without = conversion_probability(transition_matrix, states, removed_channel=channel)
    removal_effect = base_prob - prob_without
    removal_effects[channel] = removal_effect
    print(f"  Remove {channel}: prob = {prob_without:.4f}, effect = {removal_effect:.4f}")

# Normalize removal effects to get attribution
total_effect = sum(removal_effects.values())
markov_attribution = {ch: removal_effects[ch] / total_effect for ch in channels}

print(f"\nMarkov Chain Attribution (normalized):")
for ch in channels:
    print(f"  {ch}: {markov_attribution[ch]:.3f} ({markov_attribution[ch]*100:.1f}%)")
```

```python
# --- Compare Markov with Shapley ---
# Normalize Shapley to proportions
shapley_total = sum(shapley_result.values())
shapley_pct = {ch: shapley_result[ch] / shapley_total for ch in channels}

compare_advanced = pd.DataFrame({
    'Shapley': [shapley_pct[ch] for ch in channels],
    'Markov': [markov_attribution[ch] for ch in channels],
}, index=channels)

print("Shapley vs. Markov Attribution (proportions):")
print(compare_advanced.round(3))
print(f"\nCorrelation: {compare_advanced['Shapley'].corr(compare_advanced['Markov']):.3f}")
```

## Comparison: All Methods Side by Side

```python
# --- Side-by-Side Comparison: Last-Click, Shapley, Markov ---
total_conv = sum(conversions)

# Normalize all to number of conversions
last_click_pct = {ch: last_click[ch] / total_conv for ch in channels}

all_methods = pd.DataFrame({
    'Last-Click': [last_click_pct[ch] for ch in channels],
    'Shapley': [shapley_pct[ch] for ch in channels],
    'Markov': [markov_attribution[ch] for ch in channels],
}, index=channels)

fig, ax = plt.subplots(figsize=(10, 6))
all_methods.plot(kind='bar', ax=ax, width=0.7, color=['#e74c3c', '#3498db', '#2ecc71'])
ax.set_title('Attribution Model Comparison', fontsize=14)
ax.set_ylabel('Share of Attributed Conversions')
ax.set_xlabel('Channel')
ax.set_xticklabels(channels, rotation=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(title='Method', fontsize=11)
plt.tight_layout()
plt.show()

print(all_methods.round(3))
```

### Key Insight: Attribution vs. Incrementality

All attribution models (rule-based, Shapley, Markov) describe **observed journeys** - they tell you which channels appeared in converting paths.

They do **NOT** answer the incrementality question: "What would have happened without this channel?" That requires:
- **Experiments** (randomized holdout tests)
- **MMM** (aggregate causal modeling)
- **Causal inference** (quasi-experimental methods)

**Attribution and MMM are complementary:**
- Attribution: journey-level, high granularity, but observational
- MMM: aggregate, causal (with assumptions), captures offline media
- Best practice: use both and triangulate

## Exercise

Modify the journey generation to make some channels clearly more important:
1. Make `Paid Search` appear in 80% of converting journeys
2. Make `Email` always be the last touch before conversion
3. Re-run all attribution methods - which ones detect the changes?

```python
# TODO: Exercise - Modified journey generation
# Create biased journeys where Paid Search is dominant

# np.random.seed(123)
# biased_journeys = []
# biased_conversions = []
# 
# for _ in range(1000):
#     n_touches = np.random.randint(1, 5)
#     journey = list(np.random.choice(channels, n_touches))
#     
#     # Bias: Paid Search increases conversion probability
#     has_paid_search = 'Paid Search' in journey
#     base_rate = 0.3 if has_paid_search else 0.05
#     converted = np.random.random() < base_rate
#     
#     biased_journeys.append(journey)
#     biased_conversions.append(int(converted))
```

```python
# TODO: Re-run last-click, Shapley, and Markov on biased data
# Compare results - which methods detect the bias?
```

