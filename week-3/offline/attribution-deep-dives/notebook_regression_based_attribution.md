# Regression-Based Attribution Deep Dive

This notebook uses regularized logistic regression to predict conversion from
journey features, then computes channel credit from prediction deltas. The key
lesson: attribution should come from model-based marginal effects, not raw
coefficients.

Learning goals:

1. Engineer journey-level features.
2. Fit a validated conversion model.
3. Compare coefficients with prediction-removal effects.
4. Allocate credit for converted journeys.

```python
from pathlib import Path
from collections import Counter, defaultdict
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


df = pd.read_csv(find_data_path())
df["journey"] = df["touchpoint_sequence"].str.split(";")

channels = sorted({channel for path in df["journey"] for channel in path})
channel_slugs = {channel: slug(channel) for channel in channels}

print(f"Journeys: {len(df):,}")
print(f"Conversion rate: {df['converted'].mean():.2%}")
print(channel_slugs)
```

## Feature Engineering

We create interpretable features: channel presence, channel counts, first touch,
last touch, journey length, and pairwise co-occurrence.

```python
def build_features(paths, channels):
    rows = []

    for path in paths:
        counts = Counter(path)
        unique_channels = set(path)
        row = {
            "num_touchpoints": len(path),
            "unique_channels": len(unique_channels),
        }

        for channel in channels:
            channel_slug = channel_slugs[channel]
            row[f"has__{channel_slug}"] = int(channel in unique_channels)
            row[f"count__{channel_slug}"] = counts[channel]
            row[f"first__{channel_slug}"] = int(path[0] == channel)
            row[f"last__{channel_slug}"] = int(path[-1] == channel)

        for i, left in enumerate(channels):
            for right in channels[i + 1:]:
                left_slug = channel_slugs[left]
                right_slug = channel_slugs[right]
                row[f"pair__{left_slug}__{right_slug}"] = int(
                    left in unique_channels and right in unique_channels
                )

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)


X = build_features(df["journey"], channels)
y = df["converted"].astype(int)

print(X.shape)
X.head()
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        solver="lbfgs",
    ),
)

model.fit(X_train, y_train)
test_probabilities = model.predict_proba(X_test)[:, 1]

metrics = {
    "auc": roc_auc_score(y_test, test_probabilities),
    "log_loss": log_loss(y_test, test_probabilities),
    "brier_score": brier_score_loss(y_test, test_probabilities),
    "average_predicted_probability": test_probabilities.mean(),
    "observed_conversion_rate": y_test.mean(),
}

pd.Series(metrics).to_frame("value")
```

## Coefficients

Coefficients are useful diagnostics, but they are not attribution shares.

```python
classifier = model.named_steps["logisticregression"]
coefficient_df = pd.DataFrame(
    {
        "feature": X.columns,
        "coefficient": classifier.coef_[0],
    }
)

top_coefficients = pd.concat(
    [
        coefficient_df.nlargest(10, "coefficient"),
        coefficient_df.nsmallest(10, "coefficient"),
    ]
).sort_values("coefficient", ascending=False)

top_coefficients
```

```python
fig, ax = plt.subplots(figsize=(10, 7))
plot_df = top_coefficients.sort_values("coefficient")
ax.barh(plot_df["feature"], plot_df["coefficient"], color="#F28E2B")
ax.axvline(0, color="#444444", linewidth=1)
ax.set_title("Largest Positive And Negative Logistic Coefficients")
ax.set_xlabel("Coefficient on standardized feature")
plt.tight_layout()
plt.show()
```

## Prediction-Removal Effects

For each channel, set all channel-related features to zero and measure the
change in predicted conversion probability.

```python
def channel_columns(channel):
    channel_slug = channel_slugs[channel]
    return [
        column for column in X.columns
        if column.endswith(f"__{channel_slug}")
        or f"__{channel_slug}__" in column
        or column == f"has__{channel_slug}"
        or column == f"count__{channel_slug}"
        or column == f"first__{channel_slug}"
        or column == f"last__{channel_slug}"
    ]


baseline_test_probability = test_probabilities.mean()
removal_records = []

for channel in channels:
    X_removed = X_test.copy()
    X_removed[channel_columns(channel)] = 0
    removed_probability = model.predict_proba(X_removed)[:, 1].mean()

    removal_records.append(
        {
            "channel": channel,
            "baseline_probability": baseline_test_probability,
            "removed_probability": removed_probability,
            "average_probability_drop": baseline_test_probability - removed_probability,
        }
    )

removal_df = pd.DataFrame(removal_records).sort_values(
    "average_probability_drop",
    ascending=False,
)
positive_drops = removal_df["average_probability_drop"].clip(lower=0)
removal_df["attribution_share"] = positive_drops / positive_drops.sum()

removal_df.style.format(
    {
        "baseline_probability": "{:.2%}",
        "removed_probability": "{:.2%}",
        "average_probability_drop": "{:.2%}",
        "attribution_share": "{:.1%}",
    }
)
```

```python
fig, ax = plt.subplots()
plot_df = removal_df.sort_values("attribution_share")
ax.barh(plot_df["channel"], plot_df["attribution_share"], color="#4C78A8")
ax.set_title("Regression Attribution From Prediction Removal")
ax.set_xlabel("Attribution share")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.show()
```

## Journey-Level Credit

For converted journeys, compute a prediction drop for each channel present in
the journey and allocate credit in proportion to positive drops.

```python
X_all = X.copy()
full_probabilities = model.predict_proba(X_all)[:, 1]

channel_removed_probabilities = {}
for channel in channels:
    X_removed = X_all.copy()
    X_removed[channel_columns(channel)] = 0
    channel_removed_probabilities[channel] = model.predict_proba(X_removed)[:, 1]

conversion_credit = defaultdict(float)
revenue_credit = defaultdict(float)

for row_idx, row in df[df["converted"] == 1].iterrows():
    present_channels = list(dict.fromkeys(row["journey"]))
    drops = np.array(
        [
            max(
                full_probabilities[row_idx]
                - channel_removed_probabilities[channel][row_idx],
                0,
            )
            for channel in present_channels
        ]
    )

    if drops.sum() == 0:
        drops = np.ones(len(present_channels))

    weights = drops / drops.sum()

    for channel, weight in zip(present_channels, weights):
        conversion_credit[channel] += weight
        revenue_credit[channel] += weight * row["revenue"]

journey_credit_df = pd.DataFrame(
    {
        "attributed_conversions": pd.Series(conversion_credit),
        "attributed_revenue": pd.Series(revenue_credit),
    }
).fillna(0)
journey_credit_df["conversion_share"] = (
    journey_credit_df["attributed_conversions"]
    / journey_credit_df["attributed_conversions"].sum()
)
journey_credit_df["revenue_share"] = (
    journey_credit_df["attributed_revenue"]
    / journey_credit_df["attributed_revenue"].sum()
)

journey_credit_df.sort_values("conversion_share", ascending=False).style.format(
    {
        "attributed_conversions": "{:.1f}",
        "attributed_revenue": "${:,.0f}",
        "conversion_share": "{:.1%}",
        "revenue_share": "{:.1%}",
    }
)
```

## Calibration Table

Good attribution starts with a model that at least ranks and calibrates
reasonably.

```python
calibration_df = pd.DataFrame(
    {
        "predicted": test_probabilities,
        "actual": y_test.to_numpy(),
    }
)
calibration_df["decile"] = pd.qcut(
    calibration_df["predicted"],
    q=10,
    duplicates="drop",
)

calibration_table = (
    calibration_df.groupby("decile", observed=True)
    .agg(
        journeys=("actual", "size"),
        average_prediction=("predicted", "mean"),
        observed_conversion_rate=("actual", "mean"),
    )
    .reset_index()
)

calibration_table.style.format(
    {
        "average_prediction": "{:.2%}",
        "observed_conversion_rate": "{:.2%}",
    }
)
```

```python
fig, ax = plt.subplots()
ax.plot(
    calibration_table["average_prediction"],
    calibration_table["observed_conversion_rate"],
    marker="o",
)
ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
limit = max(
    calibration_table["average_prediction"].max(),
    calibration_table["observed_conversion_rate"].max(),
) * 1.1
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_title("Regression Model Calibration")
ax.set_xlabel("Average predicted probability")
ax.set_ylabel("Observed conversion rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.show()
```

## Interpretation Prompt

1. Which channel has the strongest positive prediction-removal effect?
2. Do the largest coefficients tell the same story as the marginal effects?
3. Is the model calibrated enough to trust for tactical optimization?
4. What omitted variables could make these associations misleading?

