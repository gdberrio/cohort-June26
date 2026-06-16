# LSTM And BERT-Like Attribution Deep Dive

This notebook demonstrates sequence-model attribution without requiring heavy
deep learning libraries. It uses an order-aware n-gram model as a lightweight
surrogate, then computes LSTM-like occlusion and BERT-like mask attribution.

The final section includes optional PyTorch model skeletons for students who
want to extend the exercise into a true LSTM or Transformer.

```python
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

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


df = pd.read_csv(find_data_path())
df["journey"] = df["touchpoint_sequence"].str.split(";")

channels = sorted({channel for path in df["journey"] for channel in path})
token_by_channel = {channel: channel.replace(" ", "_") for channel in channels}
channel_by_token = {token: channel for channel, token in token_by_channel.items()}


def path_to_text(path):
    return " ".join(token_by_channel[channel] for channel in path)


df["sequence_text"] = df["journey"].apply(path_to_text)

print(f"Journeys: {len(df):,}")
print(f"Conversion rate: {df['converted'].mean():.2%}")
print(token_by_channel)
df[["touchpoint_sequence", "sequence_text", "converted"]].head()
```

## Train An Order-Aware Sequence Surrogate

Unigrams, bigrams, and trigrams let a simple classifier learn ordered channel
patterns. This is not an LSTM or BERT, but it gives us the same attribution
workflow: score the full journey, perturb the sequence, and measure the
prediction change.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["sequence_text"],
    df["converted"].astype(int),
    test_size=0.30,
    random_state=42,
    stratify=df["converted"],
)

sequence_model = make_pipeline(
    CountVectorizer(
        ngram_range=(1, 3),
        min_df=5,
        token_pattern=r"(?u)\b[\w_]+\b",
    ),
    LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
    ),
)

sequence_model.fit(X_train, y_train)
test_probabilities = sequence_model.predict_proba(X_test)[:, 1]

metrics = {
    "auc": roc_auc_score(y_test, test_probabilities),
    "log_loss": log_loss(y_test, test_probabilities),
    "average_predicted_probability": test_probabilities.mean(),
    "observed_conversion_rate": y_test.mean(),
}

pd.Series(metrics).to_frame("value")
```

```python
vectorizer = sequence_model.named_steps["countvectorizer"]
classifier = sequence_model.named_steps["logisticregression"]

coef_df = pd.DataFrame(
    {
        "ngram": vectorizer.get_feature_names_out(),
        "coefficient": classifier.coef_[0],
    }
)

top_positive = coef_df.nlargest(12, "coefficient")
top_negative = coef_df.nsmallest(12, "coefficient")

pd.concat([top_positive, top_negative]).sort_values(
    "coefficient",
    ascending=False,
)
```

## LSTM-Like Occlusion Attribution

For each converted journey, remove one touchpoint at a time and measure the
drop in predicted conversion probability. This mirrors a common neural model
explanation workflow.

```python
def score_sequences(texts):
    return sequence_model.predict_proba(pd.Series(texts))[:, 1]


def remove_position(tokens, position):
    remaining = tokens[:position] + tokens[position + 1:]
    return " ".join(remaining) if remaining else "EMPTY"


occlusion_credit = defaultdict(float)
occlusion_examples = []

converted_rows = df[df["converted"] == 1].copy()

for row_idx, row in converted_rows.iterrows():
    tokens = row["sequence_text"].split()
    full_probability = score_sequences([row["sequence_text"]])[0]

    position_deltas = []
    for position, token in enumerate(tokens):
        perturbed_text = remove_position(tokens, position)
        perturbed_probability = score_sequences([perturbed_text])[0]
        delta = max(full_probability - perturbed_probability, 0)
        channel = channel_by_token[token]
        occlusion_credit[channel] += delta
        position_deltas.append((channel, delta))

    if len(occlusion_examples) < 5:
        occlusion_examples.append(
            {
                "journey": row["touchpoint_sequence"],
                "full_probability": full_probability,
                "position_deltas": position_deltas,
            }
        )

occlusion_df = pd.Series(occlusion_credit, name="raw_credit").to_frame()
occlusion_df["attribution_share"] = (
    occlusion_df["raw_credit"] / occlusion_df["raw_credit"].sum()
)
occlusion_df.sort_values("attribution_share", ascending=False).style.format(
    {
        "raw_credit": "{:.2f}",
        "attribution_share": "{:.1%}",
    }
)
```

```python
for example in occlusion_examples:
    print(example["journey"])
    print(f"  full probability: {example['full_probability']:.3f}")
    for channel, delta in example["position_deltas"]:
        print(f"  remove {channel:<15} drop = {delta:.3f}")
    print()
```

## BERT-Like Mask Attribution

Instead of deleting the touchpoint, replace it with a `MASK` token. We train a
second model on lightly augmented masked sequences so the mask token is part of
the model vocabulary.

```python
rng = np.random.default_rng(42)


def mask_random_position(text):
    tokens = text.split()
    if not tokens:
        return "EMPTY"
    position = rng.integers(0, len(tokens))
    tokens[position] = "MASK"
    return " ".join(tokens)


augmented_train_text = list(X_train)
augmented_train_labels = list(y_train)

for text, label in zip(X_train, y_train):
    augmented_train_text.append(mask_random_position(text))
    augmented_train_labels.append(label)

mask_model = make_pipeline(
    CountVectorizer(
        ngram_range=(1, 3),
        min_df=5,
        token_pattern=r"(?u)\b[\w_]+\b",
    ),
    LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
    ),
)

mask_model.fit(pd.Series(augmented_train_text), pd.Series(augmented_train_labels))
mask_test_probabilities = mask_model.predict_proba(X_test)[:, 1]

pd.Series(
    {
        "auc": roc_auc_score(y_test, mask_test_probabilities),
        "log_loss": log_loss(y_test, mask_test_probabilities),
        "average_predicted_probability": mask_test_probabilities.mean(),
        "observed_conversion_rate": y_test.mean(),
    }
).to_frame("value")
```

```python
def score_mask_sequences(texts):
    return mask_model.predict_proba(pd.Series(texts))[:, 1]


def mask_position(tokens, position):
    masked = list(tokens)
    masked[position] = "MASK"
    return " ".join(masked)


mask_credit = defaultdict(float)

for _, row in converted_rows.iterrows():
    tokens = row["sequence_text"].split()
    full_probability = score_mask_sequences([row["sequence_text"]])[0]

    for position, token in enumerate(tokens):
        masked_text = mask_position(tokens, position)
        masked_probability = score_mask_sequences([masked_text])[0]
        delta = max(full_probability - masked_probability, 0)
        mask_credit[channel_by_token[token]] += delta

mask_df = pd.Series(mask_credit, name="raw_credit").to_frame()
mask_df["attribution_share"] = mask_df["raw_credit"] / mask_df["raw_credit"].sum()

comparison = (
    occlusion_df[["attribution_share"]]
    .rename(columns={"attribution_share": "delete_occlusion_share"})
    .join(
        mask_df[["attribution_share"]]
        .rename(columns={"attribution_share": "mask_occlusion_share"}),
        how="outer",
    )
    .fillna(0)
    .sort_values("delete_occlusion_share", ascending=False)
)

comparison.style.format("{:.1%}")
```

```python
comparison.sort_values("delete_occlusion_share").plot(kind="barh")
plt.title("Sequence Attribution: Delete vs Mask Occlusion")
plt.xlabel("Attribution share")
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.show()
```

## Optional PyTorch Skeletons

The course environment does not require PyTorch. If students install it
separately, these classes show the shape of a true LSTM or Transformer
conversion model.

```python
try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if not HAS_TORCH:
    print("PyTorch is not installed. Skipping optional neural model skeletons.")
else:

    class LSTMAttributionModel(nn.Module):
        """Minimal LSTM journey classifier."""

        def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                batch_first=True,
            )
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, token_ids):
            embedded = self.embedding(token_ids)
            _, (hidden, _) = self.lstm(embedded)
            logits = self.output(hidden[-1]).squeeze(-1)
            return logits


    class TinyTransformerAttributionModel(nn.Module):
        """Tiny Transformer-style journey classifier."""

        def __init__(
            self,
            vocab_size,
            max_length,
            embedding_dim=32,
            num_heads=4,
            num_layers=2,
        ):
            super().__init__()
            self.token_embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0,
            )
            self.position_embedding = nn.Embedding(max_length, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.output = nn.Linear(embedding_dim, 1)

        def forward(self, token_ids):
            positions = torch.arange(
                token_ids.shape[1],
                device=token_ids.device,
            ).unsqueeze(0)
            embedded = self.token_embedding(token_ids) + self.position_embedding(positions)
            encoded = self.encoder(embedded)
            cls_like = encoded[:, 0, :]
            logits = self.output(cls_like).squeeze(-1)
            return logits


    print("Defined LSTMAttributionModel and TinyTransformerAttributionModel.")
```

## Interpretation Prompt

1. Which channels gain credit from order-aware occlusion?
2. How similar are delete-occlusion and mask-occlusion results?
3. What would a true LSTM or Transformer add beyond n-grams?
4. Why is this still not incrementality?

