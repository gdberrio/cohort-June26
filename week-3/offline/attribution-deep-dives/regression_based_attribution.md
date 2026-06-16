# Regression-Based Attribution

**Estimated time: 40-50 minutes**

Regression-based attribution predicts conversion or revenue from journey
features, then turns the fitted model into channel credit. It is one of the
most practical bridges between simple attribution and full machine learning
systems because it supports controls, interactions, and validation metrics.

## Core Idea

Build a row per journey:

| Feature family | Example |
|---|---|
| Channel presence | `has_paid_search = 1` |
| Channel counts | `count_email = 3` |
| Position features | `first_display = 1`, `last_search = 1` |
| Journey shape | `num_touchpoints = 5` |
| Interactions | `display_x_search = 1` |
| Timing | `days_since_first_touch = 12` |

Then fit a model:

```text
P(conversion) = f(journey features)
```

Common choices:

- logistic regression for conversion
- regularized logistic regression for many channels
- Poisson or Gamma regression for counts or revenue
- gradient boosted trees for nonlinear interactions

## Coefficients Are Not Credit

A common mistake is to treat the coefficient for a channel as its attribution
share. That is usually wrong.

Reasons:

- Coefficients are on the model's scale, such as log-odds for logistic
  regression.
- Correlated channels can share or swap credit.
- A large coefficient on a rare channel may produce little total contribution.
- Presence, count, first-touch, and last-touch features for the same channel can
  point in different directions.

Instead, use marginal effects or prediction deltas.

## Marginal Effect Attribution

For each channel:

1. Predict conversion probability for each journey with the full model.
2. Set that channel's features to zero.
3. Predict again.
4. Attribute the average probability drop to the channel.

```text
effect(channel) =
    average_prediction(full journey)
    - average_prediction(journey with channel features removed)
```

This is model-based removal attribution. It is analogous to Markov removal
effects, but the journey model is a regression rather than a transition graph.

## Why Regression Helps

Regression can handle questions that rule-based attribution cannot:

- Is email still predictive after controlling for path length?
- Does paid search look strong only because it is often last touch?
- Are display and paid search complementary?
- Is direct traffic acting as an intent signal rather than a marketing channel?

It also gives validation metrics:

- AUC or precision-recall AUC
- log loss
- Brier score
- calibration by score decile
- out-of-time validation performance

## Model Design Choices

| Decision | Conservative choice |
|---|---|
| Target | conversion first, revenue later |
| Split | out-of-time split when timestamps exist |
| Regularization | use L1 or elastic net when features are many and correlated |
| Interactions | add only interpretable interactions at first |
| Credit method | prediction removal or SHAP, not raw coefficients |
| Caveat | call results model-based associations unless calibrated to experiments |

## Where It Fails

Regression is not immune to attribution bias:

- Selection bias: high-intent users receive different media.
- Omitted variables: offline media, promotion, brand demand, and seasonality may
  be missing.
- Bad controls: controlling for a mediator can erase real channel effects.
- Positivity violations: some users never had a chance to receive a channel.
- Multicollinearity: channels that always run together are hard to separate.

Regression-based attribution is stronger than simple rules, but it still needs
causal grounding.

## References

- Shao and Li, "Data-Driven Multi-Touch Attribution Models," KDD Workshop on
  Data Mining for Online Advertising, 2011.
  https://doi.org/10.1145/2020408.2020453
- Dalessandro, Stitelman, Perlich, and Provost, "Causally Motivated Attribution
  for Online Advertising," ADKDD, 2012.
  https://doi.org/10.1145/2351356.2351363
- Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions,"
  NeurIPS, 2017.
  https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- scikit-learn documentation, "Probability calibration."
  https://scikit-learn.org/stable/modules/calibration.html
