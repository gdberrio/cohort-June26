# Probabilistic Attribution Deep Dive

This notebook builds a simple Bayesian/probabilistic attribution model from
journey data. It is intentionally transparent: we estimate smoothed conversion
rates for journeys where each channel is present vs absent, quantify
uncertainty, then allocate credit in converted journeys.

Learning goals:

1. Estimate channel-level conversion lift with Beta-Binomial smoothing.
2. Show posterior uncertainty around attribution inputs.
3. Allocate journey-level credit using positive probabilistic lift.
4. Inspect calibration caveats.

```python
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11


def find_data_path(filename="synthetic_journeys.csv"):
    candidates = [
        Path("data") / filename,
        Path("../../data") / filename,
        Path("../../../data") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}")


rng = np.random.default_rng(42)
df = pd.read_csv(find_data_path())
df["journey"] = df["touchpoint_sequence"].str.split(";")

channels = sorted({channel for path in df["journey"] for channel in path})
df["channel_set"] = df["journey"].apply(lambda path: set(path))

print(f"Journeys: {len(df):,}")
print(f"Conversions: {df['converted'].sum():,}")
print(f"Conversion rate: {df['converted'].mean():.2%}")
print(channels)
df.head()
```

## Beta-Binomial Channel Lift

For each channel, compare journeys where it was present to journeys where it
was absent. We use a `Beta(1, 1)` prior, which adds light smoothing.

```python
def beta_samples(successes, trials, alpha=1.0, beta=1.0, draws=20_000):
    failures = trials - successes
    return rng.beta(successes + alpha, failures + beta, size=draws)


records = []
lift_samples_by_channel = {}

for channel in channels:
    present_mask = df["channel_set"].apply(lambda channel_set: channel in channel_set)
    absent_mask = ~present_mask

    present_trials = int(present_mask.sum())
    absent_trials = int(absent_mask.sum())
    present_successes = int(df.loc[present_mask, "converted"].sum())
    absent_successes = int(df.loc[absent_mask, "converted"].sum())

    present_samples = beta_samples(present_successes, present_trials)
    absent_samples = beta_samples(absent_successes, absent_trials)
    lift_samples = present_samples - absent_samples
    lift_samples_by_channel[channel] = lift_samples

    records.append(
        {
            "channel": channel,
            "present_journeys": present_trials,
            "present_conversions": present_successes,
            "absent_journeys": absent_trials,
            "posterior_present_mean": present_samples.mean(),
            "posterior_absent_mean": absent_samples.mean(),
            "posterior_lift_mean": lift_samples.mean(),
            "lift_p05": np.quantile(lift_samples, 0.05),
            "lift_p95": np.quantile(lift_samples, 0.95),
            "prob_lift_positive": np.mean(lift_samples > 0),
        }
    )

lift_df = pd.DataFrame(records).sort_values(
    "posterior_lift_mean",
    ascending=False,
)

lift_df.style.format(
    {
        "posterior_present_mean": "{:.2%}",
        "posterior_absent_mean": "{:.2%}",
        "posterior_lift_mean": "{:.2%}",
        "lift_p05": "{:.2%}",
        "lift_p95": "{:.2%}",
        "prob_lift_positive": "{:.1%}",
    }
)
```

```python
plot_df = lift_df.set_index("channel")

fig, ax = plt.subplots()
ax.barh(
    plot_df.index[::-1],
    plot_df["posterior_lift_mean"][::-1],
    color="#59A14F",
)
ax.errorbar(
    plot_df["posterior_lift_mean"][::-1],
    plot_df.index[::-1],
    xerr=[
        plot_df["posterior_lift_mean"][::-1] - plot_df["lift_p05"][::-1],
        plot_df["lift_p95"][::-1] - plot_df["posterior_lift_mean"][::-1],
    ],
    fmt="none",
    ecolor="#333333",
    capsize=3,
)
ax.axvline(0, color="#444444", linewidth=1)
ax.set_title("Posterior Channel Lift With 90% Intervals")
ax.set_xlabel("P(conversion | present) - P(conversion | absent)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.show()
```

## Convert Lift Into Journey Credit

For each converted journey, give credit to channels in proportion to their
positive posterior lift. If every channel in a path has non-positive lift, fall
back to equal credit so every conversion is allocated.

```python
positive_lift = (
    lift_df.set_index("channel")["posterior_lift_mean"]
    .clip(lower=0)
    .to_dict()
)


def allocate_probabilistic_credit(path, positive_lift):
    unique_channels = list(dict.fromkeys(path))
    weights = np.array([positive_lift.get(channel, 0.0) for channel in unique_channels])

    if weights.sum() == 0:
        weights = np.ones(len(unique_channels), dtype=float)

    weights = weights / weights.sum()
    return dict(zip(unique_channels, weights))


conversion_credit = defaultdict(float)
revenue_credit = defaultdict(float)

for _, row in df[df["converted"] == 1].iterrows():
    credit = allocate_probabilistic_credit(row["journey"], positive_lift)
    for channel, weight in credit.items():
        conversion_credit[channel] += weight
        revenue_credit[channel] += weight * row["revenue"]

attribution_df = pd.DataFrame(
    {
        "attributed_conversions": pd.Series(conversion_credit),
        "attributed_revenue": pd.Series(revenue_credit),
    }
).fillna(0)

attribution_df["conversion_share"] = (
    attribution_df["attributed_conversions"]
    / attribution_df["attributed_conversions"].sum()
)
attribution_df["revenue_share"] = (
    attribution_df["attributed_revenue"]
    / attribution_df["attributed_revenue"].sum()
)

attribution_df.sort_values("conversion_share", ascending=False).style.format(
    {
        "attributed_conversions": "{:.1f}",
        "attributed_revenue": "${:,.0f}",
        "conversion_share": "{:.1%}",
        "revenue_share": "{:.1%}",
    }
)
```

```python
attribution_df[["conversion_share", "revenue_share"]].sort_values(
    "conversion_share"
).plot(kind="barh")

plt.title("Probabilistic Attribution Shares")
plt.xlabel("Share")
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.show()
```

## Calibration Sanity Check

This simple model is not a full propensity model. Still, we can construct a
rough score and inspect whether higher scores correspond to higher observed
conversion rates.

```python
base_rate = (df["converted"].sum() + 1) / (len(df) + 2)


def rough_score(path):
    unique_channels = set(path)
    score = base_rate + sum(positive_lift.get(channel, 0.0) for channel in unique_channels)
    return float(np.clip(score, 0.001, 0.999))


df["probabilistic_score"] = df["journey"].apply(rough_score)
df["score_decile"] = pd.qcut(
    df["probabilistic_score"],
    q=10,
    duplicates="drop",
)

calibration = (
    df.groupby("score_decile", observed=True)
    .agg(
        journeys=("journey_id", "count"),
        average_score=("probabilistic_score", "mean"),
        observed_conversion_rate=("converted", "mean"),
    )
    .reset_index()
)

calibration.style.format(
    {
        "average_score": "{:.2%}",
        "observed_conversion_rate": "{:.2%}",
    }
)
```

```python
fig, ax = plt.subplots()
ax.plot(
    calibration["average_score"],
    calibration["observed_conversion_rate"],
    marker="o",
)
ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
ax.set_xlim(0, max(calibration["average_score"].max(), calibration["observed_conversion_rate"].max()) * 1.15)
ax.set_ylim(0, max(calibration["average_score"].max(), calibration["observed_conversion_rate"].max()) * 1.15)
ax.set_title("Calibration Check For Rough Probabilistic Score")
ax.set_xlabel("Average predicted score")
ax.set_ylabel("Observed conversion rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.show()
```

## Interpretation Prompt

1. Which channels have positive lift with high posterior confidence?
2. Which channels have wide intervals or ambiguous lift?
3. How different are conversion shares and revenue shares?
4. Why should this not be read as causal incrementality?

