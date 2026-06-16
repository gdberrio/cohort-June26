# Markov Chain Attribution Deep Dive

This notebook extends `notebook_06_attribution.ipynb` with a fuller Markov
chain implementation on the course journey dataset.

Learning goals:

1. Build a transition matrix from observed paths.
2. Compute conversion probability with absorbing Markov chains.
3. Compare hard-removal and bypass-removal effects.
4. Interpret Markov attribution as model-based, not causal, credit.

```python
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11


def find_data_path(filename="synthetic_journeys.csv"):
    """Find the journey dataset from either repo root or notebook folder."""
    candidates = [
        Path("data") / filename,
        Path("../../data") / filename,
        Path("../../../data") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}")


DATA_PATH = find_data_path()
df = pd.read_csv(DATA_PATH)
df["journey"] = df["touchpoint_sequence"].str.split(";")

channels = sorted({channel for path in df["journey"] for channel in path})
paths = df["journey"].tolist()
converted = df["converted"].astype(int).to_numpy()

print(f"Loaded {len(df):,} journeys from {DATA_PATH}")
print(f"Conversion rate: {converted.mean():.2%}")
print(f"Channels: {channels}")
df.head()
```

## Transition Counts

A first-order Markov model uses only the current state to predict the next
state. The states are `Start`, each channel, `Conversion`, and `Null`.

```python
START = "Start"
CONVERSION = "Conversion"
NULL = "Null"
ABSORBING = {CONVERSION, NULL}


def build_transition_matrix(
    paths,
    converted,
    channels,
    removed_channel=None,
    removal_policy=None,
):
    """
    Build a first-order transition matrix.

    removal_policy:
    - None: use observed paths
    - "hard": if a journey enters removed_channel, route to Null
    - "bypass": delete removed_channel and reconnect surrounding touches
    """
    states = [START] + list(channels) + [CONVERSION, NULL]
    state_to_idx = {state: i for i, state in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=float)

    for path, outcome in zip(paths, converted):
        path = list(path)

        if removed_channel is not None and removal_policy == "bypass":
            path = [channel for channel in path if channel != removed_channel]

        previous = START
        routed_to_null = False

        if not path:
            final_state = CONVERSION if outcome else NULL
            counts[state_to_idx[START], state_to_idx[final_state]] += 1
            continue

        for channel in path:
            if removed_channel is not None and removal_policy == "hard":
                if channel == removed_channel:
                    counts[state_to_idx[previous], state_to_idx[NULL]] += 1
                    routed_to_null = True
                    break

            counts[state_to_idx[previous], state_to_idx[channel]] += 1
            previous = channel

        if not routed_to_null:
            final_state = CONVERSION if outcome else NULL
            counts[state_to_idx[previous], state_to_idx[final_state]] += 1

    probabilities = np.zeros_like(counts)
    for i, state in enumerate(states):
        row_sum = counts[i].sum()
        if state in ABSORBING:
            probabilities[i, i] = 1.0
        elif row_sum > 0:
            probabilities[i] = counts[i] / row_sum
        else:
            probabilities[i, state_to_idx[NULL]] = 1.0

    return states, counts, probabilities


states, transition_counts, transition_matrix = build_transition_matrix(
    paths, converted, channels
)

transition_df = pd.DataFrame(
    transition_matrix,
    index=states,
    columns=states,
)

transition_df.round(3)
```

```python
count_rows = []
for i, from_state in enumerate(states):
    for j, to_state in enumerate(states):
        count = transition_counts[i, j]
        if count > 0:
            count_rows.append(
                {
                    "from": from_state,
                    "to": to_state,
                    "count": int(count),
                    "probability": transition_matrix[i, j],
                }
            )

top_transitions = (
    pd.DataFrame(count_rows)
    .sort_values(["count", "probability"], ascending=False)
    .head(15)
)
top_transitions
```

```python
plot_df = top_transitions.copy()
plot_df["transition"] = plot_df["from"] + " -> " + plot_df["to"]

fig, ax = plt.subplots()
ax.barh(plot_df["transition"][::-1], plot_df["count"][::-1], color="#4C78A8")
ax.set_title("Most Common Observed Transitions")
ax.set_xlabel("Transition count")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
```

## Conversion Probability

The absorbing-chain calculation separates transient states from absorbing
states. Starting from `Start`, we can calculate the probability of eventually
ending in `Conversion`.

```python
def conversion_probability(probabilities, states):
    """Compute P(absorb in Conversion | start at Start)."""
    state_to_idx = {state: i for i, state in enumerate(states)}
    absorbing_idx = [state_to_idx[CONVERSION], state_to_idx[NULL]]
    transient_idx = [
        i for i, state in enumerate(states)
        if state not in ABSORBING
    ]

    q_matrix = probabilities[np.ix_(transient_idx, transient_idx)]
    r_matrix = probabilities[np.ix_(transient_idx, absorbing_idx)]

    identity = np.eye(len(transient_idx))
    fundamental = np.linalg.solve(identity - q_matrix, identity)
    absorption_probabilities = fundamental @ r_matrix

    start_row = transient_idx.index(state_to_idx[START])
    conversion_col = absorbing_idx.index(state_to_idx[CONVERSION])
    return absorption_probabilities[start_row, conversion_col]


baseline_probability = conversion_probability(transition_matrix, states)
print(f"Empirical conversion rate: {converted.mean():.4f}")
print(f"Markov conversion probability: {baseline_probability:.4f}")
```

## Removal Effects

Hard removal routes a journey to `Null` as soon as it would enter the removed
channel. Bypass removal deletes the channel from the path and reconnects the
surrounding states.

```python
records = []

for channel in channels:
    hard_states, _, hard_matrix = build_transition_matrix(
        paths,
        converted,
        channels,
        removed_channel=channel,
        removal_policy="hard",
    )
    bypass_states, _, bypass_matrix = build_transition_matrix(
        paths,
        converted,
        channels,
        removed_channel=channel,
        removal_policy="bypass",
    )

    hard_probability = conversion_probability(hard_matrix, hard_states)
    bypass_probability = conversion_probability(bypass_matrix, bypass_states)

    records.append(
        {
            "channel": channel,
            "baseline_probability": baseline_probability,
            "hard_removed_probability": hard_probability,
            "hard_effect": baseline_probability - hard_probability,
            "bypass_removed_probability": bypass_probability,
            "bypass_effect": baseline_probability - bypass_probability,
        }
    )

removal_df = pd.DataFrame(records).sort_values(
    "hard_effect",
    ascending=False,
)
removal_df
```

```python
for effect_col in ["hard_effect", "bypass_effect"]:
    share_col = effect_col.replace("effect", "share")
    positive_effects = removal_df[effect_col].clip(lower=0)
    removal_df[share_col] = positive_effects / positive_effects.sum()

display_cols = [
    "channel",
    "hard_effect",
    "hard_share",
    "bypass_effect",
    "bypass_share",
]

removal_df[display_cols].style.format(
    {
        "hard_effect": "{:.4f}",
        "hard_share": "{:.1%}",
        "bypass_effect": "{:.4f}",
        "bypass_share": "{:.1%}",
    }
)
```

```python
plot_df = removal_df.set_index("channel")[["hard_share", "bypass_share"]]

fig, ax = plt.subplots()
plot_df.sort_values("hard_share").plot(kind="barh", ax=ax)
ax.set_title("Markov Attribution Shares By Removal Policy")
ax.set_xlabel("Attribution share")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.legend(["Hard removal", "Bypass removal"])
plt.tight_layout()
plt.show()
```

## Compare With Last Touch

This contrast is useful in class: last-touch credit rewards the final observed
channel, while Markov removal rewards states that keep the journey graph moving
toward conversion.

```python
last_touch_credit = Counter()

for path, outcome in zip(paths, converted):
    if outcome:
        last_touch_credit[path[-1]] += 1

last_touch_share = pd.Series(last_touch_credit, dtype=float)
last_touch_share = last_touch_share / last_touch_share.sum()

comparison = (
    removal_df.set_index("channel")[["hard_share", "bypass_share"]]
    .join(last_touch_share.rename("last_touch_share"))
    .fillna(0)
    .sort_values("hard_share", ascending=False)
)

comparison.style.format("{:.1%}")
```

## Interpretation Prompt

Use the table above to answer:

1. Which channels get more credit under Markov than last touch?
2. Which channels are sensitive to hard-removal vs bypass-removal?
3. Which result would you trust for tactical optimization?
4. What evidence would you need before making a budget recommendation?

```python
comparison.plot(kind="bar", figsize=(11, 5))
plt.title("Last Touch vs Markov Removal Attribution")
plt.ylabel("Attribution share")
plt.xticks(rotation=30, ha="right")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.show()
```

