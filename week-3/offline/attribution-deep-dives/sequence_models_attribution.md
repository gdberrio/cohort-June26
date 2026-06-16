# LSTM And BERT-Like Attribution Models

**Estimated time: 45-60 minutes**

Modern attribution can treat a customer journey like a language sequence:

```text
Display -> Social Media -> Paid Search -> Email -> Conversion
```

Channels are tokens. Order matters. Time gaps can be features. Conversion is
the target. This opens the door to recurrent neural networks, attention models,
and Transformer-style architectures.

The promise is richer sequence modeling. The risk is that a more powerful model
can hide the same old attribution problem behind a more impressive interface.

## LSTM-Style Attribution

An LSTM reads the journey left to right and updates a hidden state after each
touchpoint:

```text
h_t = LSTM(channel_t, h_{t-1})
P(conversion) = sigmoid(W h_T + b)
```

It can learn patterns such as:

- paid search after display behaves differently from paid search alone
- repeated email touches may saturate
- a channel can matter more early or late in the path

Attribution is usually computed after the model is trained:

- remove one touchpoint and observe the prediction drop
- replace one touchpoint with a mask token
- compute integrated gradients against channel embeddings
- compute SHAP values on engineered sequence features

## BERT-Like Or Transformer Attribution

A BERT-like model reads the whole journey with position embeddings and
self-attention:

```text
input = [CLS], Display, Social, Paid_Search, Email
output = P(conversion | full sequence)
```

Self-attention lets each touchpoint condition on every other touchpoint. This
is useful when the importance of a channel depends on the rest of the journey.

However, attention weights are not automatically explanations. A high attention
weight can be useful for debugging, but channel credit should be checked with
perturbation methods such as occlusion, masking, or integrated gradients.

## BERT-Like Masking For Attribution

A practical attribution workflow:

1. Train a model to predict conversion from ordered journeys.
2. For each converted journey, score the full path.
3. Replace one touchpoint with a `[MASK]` token.
4. Score the masked path.
5. Attribute the probability drop to the masked channel.

```text
credit(channel at position t) =
    P(conversion | full path)
    - P(conversion | path with position t masked)
```

Aggregate the positive drops across journeys and normalize to attribution
shares.

## Why This Is Not Magic

Deep sequence models can improve predictive accuracy, but attribution still
needs measurement judgment:

- The target is observed conversion, not incremental conversion.
- Users with high purchase intent may receive different touch sequences.
- Unobserved touchpoints and offline channels are still missing.
- Models can learn platform logging quirks.
- Strong prediction does not imply causal credit.
- Explanations can be unstable if the model is not calibrated.

Use these models when the journey sequence is rich enough to justify them and
when the team can validate the output.

## Practical Course-Friendly Version

The paired notebook does not require PyTorch or TensorFlow. It uses a
lightweight order-aware model:

- tokenize journeys as channel sequences
- build unigram, bigram, and trigram features
- train a calibrated logistic classifier
- compute occlusion and mask-based attribution
- include optional PyTorch model skeletons for LSTM and Transformer versions

This lets students learn the attribution workflow without making the course
environment depend on heavy neural packages.

## References

- Ren et al., "Learning Multi-touch Conversion Attribution with Dual-attention
  Mechanisms for Online Advertising," CIKM, 2018.
  https://doi.org/10.1145/3269206.3271677
- Li et al., "Deep Neural Net with Attention for Multi-channel Multi-touch
  Attribution," arXiv, 2018.
  https://arxiv.org/abs/1809.02230
- Kang and McAuley, "Self-Attentive Sequential Recommendation," ICDM, 2018.
  https://arxiv.org/abs/1808.09781
- Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder
  Representations from Transformer," CIKM, 2019.
  https://arxiv.org/abs/1904.06690
- Jain and Wallace, "Attention is not Explanation," NAACL, 2019.
  https://aclanthology.org/N19-1357/
- Sundararajan, Taly, and Yan, "Axiomatic Attribution for Deep Networks,"
  ICML, 2017.
  https://proceedings.mlr.press/v70/sundararajan17a.html
